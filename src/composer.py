"""Compose one answer from per-sub-question answers, keeping citations traceable."""
from __future__ import annotations

from typing import Any

from .auth import SOURCE_LABELS
from .citations import missing_citations, namespace_citations
from .router import DEFAULT_MODEL

COMPOSE_PROMPT = """
    The user asked a compound question. It was split into sub-questions and each was
    answered independently from its own source. Write ONE coherent answer to the
    original question using only the sub-answers below.

    Rules:
    - Address every sub-question.
    - Keep citation tags exactly as written, e.g. [1.2] or [2.1], next to the claims
      they support. Do not renumber them, merge them, or invent new ones.
    - If a sub-answer is an error or says no information was found, say so plainly
      rather than guessing.

    Original question: {user_query}

    {blocks}
    """


def deterministic_compose(parts: list[dict]) -> str:
    """No-LLM fallback: one section per sub-question, citations untouched."""
    return "\n\n".join(f"**{k}. {p['question']}**\n{p['tagged']}" for k, p in enumerate(parts, 1))


def source_map(parts: list[dict], labels: dict[str, str] = SOURCE_LABELS) -> str:
    """Citation key: which sub-question and source each [k.x] prefix refers to."""
    lines = [f"[{k}.x] → sub-question {k} ({labels.get(p['action'], p['action'])}): {p['question']}"
             for k, p in enumerate(parts, 1)]
    return "Citation key:\n" + "\n".join(lines)


def compose_answer(user_query: str, parts: list[dict], llm: Any, model: str = DEFAULT_MODEL,
                   labels: dict[str, str] = SOURCE_LABELS, with_method: bool = False):
    """
    Synthesise one answer from the per-sub-query answers, keeping citations.

    Adds a "tagged" field (namespaced answer) to each part. Falls back to
    deterministic_compose() if the LLM call fails or drops any sub-answer's citations.

    Returns:
        str, or (str, method) with method "llm" | "deterministic: <reason>".
    """
    for k, p in enumerate(parts, 1):
        p["tagged"] = namespace_citations(p["answer"], k)

    def done(text: str, method: str):
        return (text, method) if with_method else text

    blocks = "\n\n".join(
        f"### Sub-question {k}: {p['question']}\n"
        f"Source: {labels.get(p['action'], p['action'])}\n"
        f"Answer:\n{p['tagged']}"
        for k, p in enumerate(parts, 1)
    )
    try:
        response = llm.chat.completions.create(
            model=model,
            messages=[{"role": "system",
                       "content": COMPOSE_PROMPT.format(user_query=user_query, blocks=blocks)}],
        )
        composed = response.choices[0].message.content or ""
    except Exception as err:
        print(f"⚠️ Composition failed ({err}); falling back to per-question sections.")
        return done(deterministic_compose(parts), "deterministic: LLM error")

    # Every sub-answer that cited something must still be cited at least once in the
    # composed text; otherwise fall back rather than return an ungrounded merge.
    dropped = missing_citations([p["tagged"] for p in parts], composed)
    if dropped:
        print(f"⚠️ Composition dropped citations for sub-question(s) {dropped}; "
              f"using per-question sections.")
        return done(deterministic_compose(parts), f"deterministic: dropped citations {dropped}")
    return done(composed, "llm")
