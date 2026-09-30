"""
Query router — decides which knowledge source should answer ONE question.

Parsing is layered so a bad model response never crashes the agent:

    1. structured output  (OpenAI `chat.completions.parse` → validated RouteDecision)
    2. defensive parser   (plain completion → strip fences/prose → JSON → validate)
    3. fallback route     (INTERNET_QUERY, same as the original router's error path)
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ValidationError

DEFAULT_MODEL = "gpt-5.6-luna"

Route = Literal["OPENAI_QUERY", "10K_DOCUMENT_QUERY", "INTERNET_QUERY"]
ROUTES: tuple[str, ...] = ("OPENAI_QUERY", "10K_DOCUMENT_QUERY", "INTERNET_QUERY")
FALLBACK_ROUTE = "INTERNET_QUERY"


class RouteDecision(BaseModel):
    """The router's answer. `answer` is the original router's ≤5-word quick answer."""
    action: Route
    reason: str
    answer: str


# Prompt text unchanged from the original notebook router (Section 2).
ROUTER_PROMPT = """
    As a professional query router, your objective is to correctly classify user input into one of three categories based on the source most relevant for answering the query:
    1. "OPENAI_QUERY": If the user's query appears to be answerable using information from OpenAI's official documentation about Agents, tools, models, APIs, or services (e.g., guardrails, agents, what is an agent, embeddings, moderation API, usage guidelines).
    2. "10K_DOCUMENT_QUERY": If the user's query pertains to a collection of documents from the 10k annual reports, datasets, or other structured documents, typically for research, analysis, or financial content.
    3. "INTERNET_QUERY": If the query is neither related to OpenAI nor the 10k documents specifically, or if the information might require a broader search (e.g., news, trends, tools outside these platforms), route it here.

    Your decision should be made by assessing the domain of the query.

    Always respond in this valid JSON format:
    {{
        "action": "OPENAI_QUERY" or "10K_DOCUMENT_QUERY" or "INTERNET_QUERY",
        "reason": "brief justification",
        "answer": "AT MAX 5 words answer. Leave empty if INTERNET_QUERY"
    }}

    EXAMPLES:

    - User: "How to fine-tune GPT-3?"
    Response:
    {{
        "action": "OPENAI_QUERY",
        "reason": "Fine-tuning is OpenAI-specific",
        "answer": "Use fine-tuning API"
    }}

    - User: "Where can I find the latest financial reports for the last 10 years?"
    Response:
    {{
        "action": "10K_DOCUMENT_QUERY",
        "reason": "Query related to annual reports",
        "answer": "Access through document database"
    }}

    - User: "Top leadership styles in 2024"
    Response:
    {{
        "action": "INTERNET_QUERY",
        "reason": "Needs current leadership trends",
        "answer": ""
    }}

    - User: "What's the difference between ChatGPT and Claude?"
    Response:
    {{
        "action": "INTERNET_QUERY",
        "reason": "Cross-comparison of different providers",
        "answer": ""
    }}

    Strictly follow this format for every query, and never deviate.
    User: {user_query}
    """


def parse_route_decision(raw: Any) -> Optional[RouteDecision]:
    """
    Defensively extract a RouteDecision from free-form model text.

    Handles code fences and prose around the JSON; a missing `answer` defaults to "".
    Returns None when there is no valid decision (unknown route, bad JSON, no JSON).
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        if not isinstance(data, dict):
            return None
        return RouteDecision.model_validate({"answer": "", **data})
    except (json.JSONDecodeError, ValidationError):
        return None


def route_query(user_query: str, llm: Any, model: str = DEFAULT_MODEL) -> dict:
    """
    Classify a question into OPENAI_QUERY / 10K_DOCUMENT_QUERY / INTERNET_QUERY.

    Args:
        user_query: The question to route.
        llm: An OpenAI client (anything exposing `chat.completions.parse/create`).
        model: Chat model name.

    Returns:
        dict with "action", "reason", "answer" (same shape as the original router)
        plus "parsed_by": "structured" | "parser" | "fallback".
    """
    messages = [{"role": "system", "content": ROUTER_PROMPT.format(user_query=user_query)}]

    # 1. Structured output — the SDK validates the JSON against RouteDecision.
    raw = None
    try:
        message = llm.chat.completions.parse(
            model=model, messages=messages, response_format=RouteDecision
        ).choices[0].message
        if message.parsed is not None:
            return {**message.parsed.model_dump(), "parsed_by": "structured"}
        raw = message.content          # refusal or unparsed text — try the parser on it
    except Exception:
        pass                           # unsupported model/SDK, schema mismatch, API error

    # 2. Plain completion + defensive parser (the original router path).
    try:
        if raw is None:
            raw = llm.chat.completions.create(model=model, messages=messages).choices[0].message.content
        decision = parse_route_decision(raw)
        if decision is not None:
            return {**decision.model_dump(), "parsed_by": "parser"}
        reason = "Router output was not a valid route decision"
    except Exception as err:
        reason = f"Router error: {err}"

    # 3. Fallback — identical to the original router's behaviour on errors.
    return {"action": FALLBACK_ROUTE, "reason": reason, "answer": "", "parsed_by": "fallback"}
