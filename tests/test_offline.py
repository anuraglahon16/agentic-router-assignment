"""
Offline tests — no API keys, no Qdrant, no model downloads.

A scripted LLM stands in for OpenAI and simple functions stand in for the retrieval
handlers, so these tests exercise the real splitter / router / composer / cache /
pipeline code deterministically.  Run:  python -m pytest -q
"""
import asyncio
import hashlib
import json
import re

import numpy as np
import pytest

from src import auth
from src.cache import RoleAwareSemanticCache, is_time_sensitive
from src.evaluation import (ScriptedLLM, citations_traceable, run_cache_matrix,
                            run_citation_tests, run_parser_robustness_tests)
from src.pipeline import AgentContext, agentic_rag_multi, secure_agentic_rag_cached
from src.retrieval import execute_route


# ── Fakes ─────────────────────────────────────────────────────────────────────
def scripted_respond(prompt, fmt):
    """Split on 'and', route by keyword, compose by echoing every [k.n] tag."""
    if "break it into sub-questions" in prompt:
        query = re.search(r'Query: "(.*)"', prompt).group(1)
        parts = [s.strip(" ?") + "?" for s in re.split(r"\band\b", query) if s.strip(" ?")]
        return json.dumps({"subQuestions": parts})
    if "professional query router" in prompt:
        q = prompt.rsplit("User:", 1)[1].lower()
        action = ("10K_DOCUMENT_QUERY" if re.search(r"uber|lyft|revenue", q)
                  else "OPENAI_QUERY" if re.search(r"openai|agent", q) else "INTERNET_QUERY")
        return json.dumps({"action": action, "reason": "keyword", "answer": ""})
    if "compound question" in prompt:
        tags = re.findall(r"\[\d+\.\d+\]", prompt.split("Original question:")[1])
        return "Combined: " + " ".join(dict.fromkeys(tags))
    raise AssertionError("unexpected prompt")


def make_routes(calls):
    async def docs(query, action):                  # async, like retrieve_and_response
        calls.append(action)
        return f"{action} answer [1][2]."

    def web(query, action):                         # sync, like get_internet_content
        calls.append(action)
        return "[Direct Answer] foo\n\n[1] Title\n    snippet"

    return {"OPENAI_QUERY": docs, "10K_DOCUMENT_QUERY": docs, "INTERNET_QUERY": web}


def bow_embed(text):
    v = np.zeros(64, dtype="float32")
    for w in re.findall(r"\w+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
    return v


@pytest.fixture
def ctx():
    handler_calls = []
    llm = ScriptedLLM(scripted_respond)
    c = AgentContext(llm=llm, routes=make_routes(handler_calls))
    c.handler_calls = handler_calls
    return c


# ── Suites shared with the notebook ──────────────────────────────────────────
def test_parser_robustness_suite():
    df = run_parser_robustness_tests()
    assert df["Pass"].all(), df[~df["Pass"]].to_string()


def test_citation_suite():
    df = run_citation_tests()
    assert df["Pass"].all(), df[~df["Pass"]].to_string()


# ── Part 1: sub-query division ───────────────────────────────────────────────
def test_single_query_no_extra_calls(ctx):
    d = agentic_rag_multi(ctx, "what was uber revenue in 2021?", return_details=True, verbose=False)
    assert d["sub_queries"] == ["what was uber revenue in 2021?"]
    assert [p["action"] for p in d["parts"]] == ["10K_DOCUMENT_QUERY"]
    assert ctx.llm.calls == ["split", "route"]              # no composition call
    assert d["answer"] == "10K_DOCUMENT_QUERY answer [1][2]."  # returned as-is
    assert d["split_source"] == "structured"


def test_same_route_compound(ctx):
    d = agentic_rag_multi(ctx, "what was lyft revenue in 2021 and what was uber revenue in 2021",
                          return_details=True, verbose=False)
    assert len(d["sub_queries"]) == 2
    assert [p["action"] for p in d["parts"]] == ["10K_DOCUMENT_QUERY"] * 2
    assert d["compose_method"] == "llm"
    assert all(t in d["answer"] for t in ("[1.1]", "[1.2]", "[2.1]", "[2.2]"))
    assert citations_traceable(d["parts"], d["answer"])


def test_mixed_route_compound(ctx):
    d = agentic_rag_multi(ctx, "what was uber's 2021 revenue and what are the newest LLMs?",
                          return_details=True, verbose=False)
    assert [p["action"] for p in d["parts"]] == ["10K_DOCUMENT_QUERY", "INTERNET_QUERY"]
    assert "[Direct Answer]" in d["parts"][1]["tagged"] and "[2.1]" in d["parts"][1]["tagged"]
    assert citations_traceable(d["parts"], d["answer"])


def test_concurrent_matches_sequential(ctx):
    q = "what was uber's 2021 revenue and what are the newest LLMs?"
    assert agentic_rag_multi(ctx, q, verbose=False) == agentic_rag_multi(ctx, q, concurrent=True, verbose=False)


def test_malformed_split_falls_back_to_single_query():
    llm = ScriptedLLM(lambda p, f: "I cannot do that" if "sub-questions" in p else scripted_respond(p, f))
    c = AgentContext(llm=llm, routes=make_routes([]))
    q = "what was lyft revenue in 2021 and what was uber revenue in 2021"
    d = agentic_rag_multi(c, q, return_details=True, verbose=False)
    assert d["sub_queries"] == [q] and d["split_source"] == "fallback"


def test_execute_route_handles_sync_async_and_unknown():
    routes = make_routes([])
    assert execute_route("q", "OPENAI_QUERY", routes).startswith("OPENAI_QUERY answer")
    assert execute_route("q", "INTERNET_QUERY", routes).startswith("[Direct Answer]")
    assert execute_route("q", "NOPE", routes) == "Unsupported action: NOPE"


def test_execute_route_inside_running_loop():
    """Jupyter case: a loop is already running in the main thread."""
    import nest_asyncio
    nest_asyncio.apply()

    async def main():
        return execute_route("q", "10K_DOCUMENT_QUERY", make_routes([]))
    assert asyncio.run(main()).startswith("10K_DOCUMENT_QUERY answer")


# ── Bonus: RBAC + cache ──────────────────────────────────────────────────────
def test_required_self_check_offline(ctx):
    """Same sequence as the notebook's run_self_check(), plus handler-call accounting."""
    cache = RoleAwareSemanticCache(embed_fn=bow_embed)
    q_fin, q_doc = "what was uber revenue in 2021?", "how do I build an agent with the OpenAI Agents SDK?"
    ask = lambda u, q: secure_agentic_rag_cached(ctx, u, q, cache, verbose=False, audit_log=[])["status"]

    assert ask("bob", q_fin) == "MISS"
    assert ask("bob", q_fin) == "HIT"
    assert ask("alice", q_fin) == "DENIED"
    assert ask("alice", "how much revenue did Uber make in 2021?") == "DENIED"
    assert ask("carol", q_doc) == "DENIED"
    assert ask("alice", q_doc) == "MISS"
    assert ask("alice", q_doc) == "HIT"
    # handlers ran only on the two MISSes — never on a HIT or a DENIED
    assert ctx.handler_calls == ["10K_DOCUMENT_QUERY", "OPENAI_QUERY"]


def test_unknown_user_makes_no_llm_call(ctx):
    cache = RoleAwareSemanticCache(embed_fn=bow_embed)
    secure_agentic_rag_cached(ctx, "mallory", "anything", cache, verbose=False, audit_log=[])
    assert ctx.llm.calls == [] and len(cache) == 0


def test_cache_matrix_offline(ctx):
    df = run_cache_matrix(ctx, lambda: RoleAwareSemanticCache(embed_fn=bow_embed))
    assert df["Pass"].all(), df[~df["Pass"]].to_string()
    assert auth.USERS["bob"] == "finance_analyst"            # role restored


def test_time_sensitivity():
    assert is_time_sensitive("what are the latest LLMs released this week?")
    assert not is_time_sensitive("what was uber revenue in 2021?")
