"""
Citation namespacing and verification.

Each sub-answer numbers its own sources from [1], so two sub-answers would both contain
a "[1]" meaning different documents. Sub-answer k's [n] is rewritten to [k.n] before
composition, and the composed answer is checked to still cite every sub-answer.
"""
from __future__ import annotations

import re

CITATION = re.compile(r"\[(\d+)\]")                      # [1], [2] … not [Direct Answer] or [1.2]
CITATION_GROUP = re.compile(r"\[(\d+(?:\s*,\s*\d+)+)\]")  # [1, 2] or [1,2,3]
TAG = re.compile(r"\[(\d+)\.(\d+)\]")                    # namespaced tags: [k.n]


def namespace_citations(text: str, k: int) -> str:
    """Rewrite sub-answer k's [n] as [k.n] (and [1, 2] as [k.1][k.2])."""
    text = CITATION_GROUP.sub(
        lambda m: "".join(f"[{k}.{n.strip()}]" for n in m.group(1).split(",")), text or ""
    )
    return CITATION.sub(lambda m: f"[{k}.{m.group(1)}]", text)


def cited_sub_questions(text: str) -> set[int]:
    """Which sub-question numbers (the k in [k.n]) are cited in `text`."""
    return {int(k) for k, _ in TAG.findall(text or "")}


def missing_citations(tagged_answers: list[str], composed: str) -> list[int]:
    """
    Sub-question numbers whose sub-answer cited something but whose citations
    no longer appear anywhere in the composed answer.
    """
    needs = {k for k, text in enumerate(tagged_answers, 1) if TAG.search(text or "")}
    return sorted(needs - cited_sub_questions(composed))
