"""
Compound-query splitter — turns one user message into independent sub-questions.

Layered so a malformed split can never crash the agent:

    1. structured output  (QuerySplit validated by the SDK)
    2. defensive parser   (fences, prose, bare lists, alternative keys)
    3. fallback           (the original query as the only sub-question)
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional

from pydantic import BaseModel

from .router import DEFAULT_MODEL

MAX_SUB_QUERIES = 5  # guard against a runaway split blowing up cost

# Prompt text unchanged from the notebook's reference sub_queries() cell.
SPLIT_PROMPT = """
  You are a query router. If the input contains multiple distinct questions, break it into sub-questions. Otherwise, keep it as one. Return a JSON object like:

  {{
      "subQuestions": ["..."]
  }}


  Query: "{user_query}"
  Output:
  """

_SPLIT_KEYS = {"subquestions", "subqueries", "questions"}


class QuerySplit(BaseModel):
    subQuestions: list[str]


def _dedupe_key(question: str) -> str:
    """'What was Uber revenue?' and 'what was  uber revenue' are the same question."""
    return re.sub(r"\s+", " ", question).strip().rstrip("?.! ").lower()


def clean_questions(items: Iterable[Any], limit: int = MAX_SUB_QUERIES) -> list[str]:
    """Keep non-blank strings, drop duplicates (order preserved), cap at `limit`."""
    questions, seen = [], set()
    for item in items:
        if not isinstance(item, str) or not item.strip():
            continue
        key = _dedupe_key(item)
        if key and key not in seen:
            seen.add(key)
            questions.append(item.strip())
    return questions[:limit]


def extract_questions(raw: Any) -> Optional[list[str]]:
    """
    Pull a question list out of free-form model text, or None if there isn't one.

    Handles: clean JSON, JSON inside ```json fences, JSON wrapped in prose, a bare
    JSON list, alternative key names (subQuestions / sub_queries / questions).
    """
    if not isinstance(raw, str) or not raw.strip():
        return None

    text = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE)
    candidates = [m.group() for m in (re.search(r"\{.*\}", text, re.DOTALL),
                                      re.search(r"\[.*\]", text, re.DOTALL)) if m]
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = next((v for k, v in data.items()
                         if k.lower().replace("_", "") in _SPLIT_KEYS), None)
        if isinstance(data, list):
            questions = clean_questions(data)
            if questions:
                return questions
    return None


def parse_sub_queries(raw: Any, fallback: str) -> list[str]:
    """Defensive parse of the splitter's raw text; anything unusable → [fallback]."""
    return extract_questions(raw) or [fallback]


def split_query(user_query: str, llm: Any, model: str = DEFAULT_MODEL,
                with_source: bool = False):
    """
    Split a possibly-compound question into sub-questions.

    Args:
        user_query: The user's message.
        llm: An OpenAI client (anything exposing `chat.completions.parse/create`).
        model: Chat model name.
        with_source: Also return how the split was obtained.

    Returns:
        list[str] of 1..MAX_SUB_QUERIES questions, or (list, source) where source is
        "structured" | "parser" | "fallback".
    """
    def done(questions: list[str], source: str):
        return (questions, source) if with_source else questions

    messages = [{"role": "system", "content": SPLIT_PROMPT.format(user_query=user_query)}]

    # 1. Structured output
    raw = None
    try:
        message = llm.chat.completions.parse(
            model=model, messages=messages, response_format=QuerySplit
        ).choices[0].message
        if message.parsed is not None:
            questions = clean_questions(message.parsed.subQuestions)
            if questions:
                return done(questions, "structured")
        raw = message.content
    except Exception:
        pass

    # 2. Plain completion + defensive parser
    try:
        if raw is None:
            raw = llm.chat.completions.create(model=model, messages=messages).choices[0].message.content
    except Exception as err:
        print(f"⚠️ Split failed ({err}); treating the input as one query.")
        return done([user_query], "fallback")

    questions = extract_questions(raw)
    if questions:
        return done(questions, "parser")

    # 3. Fallback — the input is treated as a single question
    return done([user_query], "fallback")
