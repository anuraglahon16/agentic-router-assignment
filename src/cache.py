"""
Role-aware semantic cache — cannot serve an answer across a permission boundary.

faiss-cpu is optional: without it the same exact search runs in NumPy.
"""
from __future__ import annotations

import re
import time
from typing import Any, Callable, Optional

import numpy as np

from . import auth

try:
    import faiss                      # pip install faiss-cpu
except ImportError:                   # same behaviour with a NumPy brute-force search
    faiss = None


# ── Time-sensitivity: never cache answers that go stale ───────────────────────
# Deliberately over-broad. A false positive costs one cache miss; a false negative
# serves a stale fact quickly, which is the worse failure.
_TIME_SENSITIVE = re.compile(
    r"\b(today|tonight|tomorrow|yesterday|now|currently|current|latest|newest|new|"
    r"recent|recently|this (?:week|month|quarter|year)|breaking|live|trending|"
    r"stocks?|share price|price|weather|forecast|news|scores?)\b",
    re.IGNORECASE,
)


def is_time_sensitive(query: str) -> bool:
    return bool(_TIME_SENSITIVE.search(query or ""))


class RoleAwareSemanticCache:
    """
    A semantic cache that cannot serve an answer across a permission boundary.

    Design choice: PARTITIONED — one FAISS index per knowledge *source*
    (OPENAI_QUERY, 10K_DOCUMENT_QUERY, INTERNET_QUERY), not one per role.

    - Why partitioned, not tagged: with one shared index, a nearest-neighbour search
      happily returns a finance row to an engineer and only a post-filter stands
      between that row and a leak; if the filter is forgotten or buggy, data escapes.
      With partitions, a lookup physically searches only the indexes of sources the
      caller may read, so a forbidden row is never a candidate at all. It also keeps
      the top-k honest: a tagged index can have its top result be someone else's
      row and the caller's valid match sit at rank 2.

    - Why per source, not per role: an answer is built from exactly one source, so
      the source is what decides who may read it. Both roles can read OPENAI_QUERY,
      so an OpenAI-docs answer cached for alice is reusable by bob. A per-role cache
      would compute and store it twice. The trade-off: this is only safe while each
      answer comes from one source. A composed multi-source answer (Part 1) would
      have to be keyed by the full set of sources and served only to users allowed
      all of them — to keep that honest, this cache refuses to store anything whose
      source the caller can't read, and never stores composites.

    - Role changes: permissions are checked against ROLE_PERMISSIONS at lookup time
      and nothing is stored per user, so a role change takes effect on the very next
      request. If bob moves from finance to engineering, the 10-K partition is simply
      no longer searched for him — no cache purge needed, no stale grant to revoke.

    Threshold: squared L2 distance between unit-normalised embeddings, where
    d = 2 − 2·cos, so the default 0.2 means cosine similarity ≥ 0.90.
    """

    SOURCE_TTL: dict[str, int] = {"INTERNET_QUERY": 3600}  # seconds; document sources don't expire

    def __init__(self, threshold: float = 0.2,
                 embed_fn: Optional[Callable[[str], Any]] = None,
                 allowed_sources_fn: Optional[Callable[[str], set]] = None):
        """
        Args:
            threshold: Max squared L2 distance for a hit (0.2 ≈ cosine ≥ 0.90).
            embed_fn: text → vector. Required before the first lookup.
            allowed_sources_fn: user_id → set of readable sources
                                (defaults to auth.allowed_sources, read at call time).
        """
        self.threshold = threshold
        self.embed_fn = embed_fn
        self.allowed_sources_fn = allowed_sources_fn or auth.allowed_sources
        self._partitions: dict[str, dict] = {}  # source -> {"index", "matrix", "entries"}

    # ── internals ──
    def _embed(self, text: str) -> np.ndarray:
        if self.embed_fn is None:
            raise ValueError("RoleAwareSemanticCache needs an embed_fn")
        v = np.asarray(self.embed_fn(text), dtype="float32").reshape(-1)
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def _partition(self, source: str, dim: int) -> dict:
        if source not in self._partitions:
            self._partitions[source] = {
                "index": faiss.IndexFlatL2(dim) if faiss else None,
                "matrix": np.empty((0, dim), dtype="float32"),
                "entries": [],
            }
        return self._partitions[source]

    def _nearest(self, part: dict, emb: np.ndarray, k: int = 10) -> list[tuple[float, dict]]:
        """Up to k (distance, entry) pairs, closest first."""
        n = len(part["entries"])
        if n == 0:
            return []
        k = min(k, n)
        if faiss:
            D, I = part["index"].search(emb[None, :], k)
            return [(float(d), part["entries"][i]) for d, i in zip(D[0], I[0]) if i >= 0]
        d = ((part["matrix"] - emb) ** 2).sum(axis=1)
        order = np.argsort(d)[:k]
        return [(float(d[i]), part["entries"][i]) for i in order]

    def _readable(self, user_id: str, source: Optional[str]) -> list[str]:
        """The partitions this caller may search — the whole security boundary."""
        allowed = self.allowed_sources_fn(user_id)          # empty set for unknown users
        if source is not None:
            return [source] if source in allowed else []
        return [s for s in self._partitions if s in allowed]

    # ── public API ──
    def check(self, user_id: str, question: str, source: Optional[str] = None, embedding=None):
        """
        Return (hit: bool, answer: str | None, embedding, similarity | None).

        `source` is the route the router chose; only that partition is searched, and
        only if the caller's role may read it. Without `source`, every partition the
        caller may read is searched.
        """
        emb = embedding if embedding is not None else self._embed(question)
        now, best = time.time(), None
        for s in self._readable(user_id, source):
            part = self._partitions.get(s)
            if not part:
                continue
            for dist, entry in self._nearest(part, emb):
                if dist > self.threshold:
                    break                                   # sorted: nothing closer left
                if entry["expires_at"] and now > entry["expires_at"]:
                    continue                                # stale — try the next one
                if best is None or dist < best[0]:
                    best = (dist, entry)
                break
        if best is None:
            return False, None, emb, None
        dist, entry = best
        return True, entry["answer"], emb, 1 - dist / 2

    def add(self, user_id: str, question: str, answer: str, embedding=None,
            source: Optional[str] = None) -> None:
        """
        Store an answer scoped to this user's permissions.

        `source` is required: an entry that doesn't know which source produced it
        can't be scoped, so it isn't stored. A caller can only write into a partition
        it may read — no poisoning another role's partition.
        """
        if source is None:
            raise ValueError("source is required — an unscoped answer can't be cached safely")
        if source not in self.allowed_sources_fn(user_id):
            raise PermissionError(f"{user_id!r} may not write to the {source} partition")
        emb = embedding if embedding is not None else self._embed(question)
        part = self._partition(source, emb.shape[0])
        if faiss:
            part["index"].add(emb[None, :])
        else:
            part["matrix"] = np.vstack([part["matrix"], emb[None, :]])
        ttl = self.SOURCE_TTL.get(source)
        part["entries"].append({
            "question": question,
            "answer": answer,
            "source": source,
            "created_by": user_id,            # audit only — never used for access
            "expires_at": time.time() + ttl if ttl else None,
        })

    def __len__(self) -> int:
        return sum(len(p["entries"]) for p in self._partitions.values())
