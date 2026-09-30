"""
Evaluation suites for the Agentic Router.

Live suites (need API keys + data; run the real implementation):
    run_routing_eval()          split + routing accuracy over ROUTING_EVAL_SET
    run_compound_eval()         compound queries end to end, incl. citation traceability
    run_concurrency_benchmark() sequential vs concurrent wall-clock
    run_cache_matrix()          RBAC + semantic cache scenarios

Offline suites (deterministic, scripted LLM — also run by pytest):
    run_parser_robustness_tests()  malformed / fenced / prose / empty model output
    run_citation_tests()           namespacing, collisions, composition fallback
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Callable, Optional, Union

import pandas as pd

from . import auth
from .citations import CITATION, TAG, namespace_citations
from .composer import compose_answer
from .pipeline import AgentContext, agentic_rag_multi, secure_agentic_rag_cached
from .router import route_query
from .splitter import MAX_SUB_QUERIES, split_query

O, K, I = "OPENAI_QUERY", "10K_DOCUMENT_QUERY", "INTERNET_QUERY"

# An expected route is either one label or a tuple of acceptable labels (ambiguous queries).
ExpectedRoute = Union[str, tuple]


# ── Offline stand-in for the OpenAI client ────────────────────────────────────
def _completion(content: Optional[str], parsed: Any = None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, parsed=parsed, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def prompt_kind(prompt: str) -> str:
    """Which pipeline step a prompt belongs to (used to count LLM calls)."""
    if "break it into sub-questions" in prompt:
        return "split"
    if "professional query router" in prompt:
        return "route"
    if "compound question" in prompt:
        return "compose"
    if "Based on the given context" in prompt:
        return "rag"
    return "other"


class ScriptedLLM:
    """
    Minimal fake of the OpenAI client for deterministic tests.

    `respond(prompt, response_format)` returns the model's text (or an Exception to
    raise). `parse()` validates that text against `response_format` like the SDK does;
    invalid text yields `parsed=None` so the caller's defensive parser takes over.
    Set `structured=False` to simulate a model/SDK without structured outputs.
    """

    def __init__(self, respond: Callable[[str, Any], Any], structured: bool = True):
        self.respond = respond
        self.structured = structured
        self.calls: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create, parse=self._parse))

    def _text(self, messages: list[dict], fmt: Any) -> Optional[str]:
        prompt = messages[0]["content"]
        self.calls.append(prompt_kind(prompt))
        out = self.respond(prompt, fmt)
        if isinstance(out, BaseException):
            raise out
        return out

    def _create(self, model: str, messages: list[dict], **_: Any) -> SimpleNamespace:
        return _completion(self._text(messages, None))

    def _parse(self, model: str, messages: list[dict], response_format: Any, **_: Any) -> SimpleNamespace:
        if not self.structured:
            raise RuntimeError("structured outputs not supported by this model")
        text = self._text(messages, response_format)
        try:
            parsed = response_format.model_validate_json(text)
        except Exception:
            parsed = None
        return _completion(text, parsed)


# ── Datasets ──────────────────────────────────────────────────────────────────
ROUTING_EVAL_SET: list[dict] = [
    # OpenAI documentation
    {"category": "openai", "query": "What is an AI agent?", "expected_subquery_count": 1, "expected_routes": [O]},
    {"category": "openai", "query": "How do I add guardrails to an agent with the OpenAI Agents SDK?", "expected_subquery_count": 1, "expected_routes": [O]},
    {"category": "openai", "query": "How do handoffs between agents work in the OpenAI Agents SDK?", "expected_subquery_count": 1, "expected_routes": [O]},
    {"category": "openai", "query": "How to work with chat completions?", "expected_subquery_count": 1, "expected_routes": [O]},
    {"category": "openai", "query": "How do I use the OpenAI moderation API?", "expected_subquery_count": 1, "expected_routes": [O]},
    {"category": "openai", "query": "How do I create embeddings with the OpenAI API?", "expected_subquery_count": 1, "expected_routes": [O]},
    # 10-K financial filings
    {"category": "10k", "query": "What was Uber revenue in 2021?", "expected_subquery_count": 1, "expected_routes": [K]},
    {"category": "10k", "query": "What was Lyft's net loss in 2021?", "expected_subquery_count": 1, "expected_routes": [K]},
    {"category": "10k", "query": "What risk factors did Uber list in its 2021 annual report?", "expected_subquery_count": 1, "expected_routes": [K]},
    {"category": "10k", "query": "How many active riders did Lyft report in its 10-K?", "expected_subquery_count": 1, "expected_routes": [K]},
    {"category": "10k", "query": "What were Uber's total costs and expenses in 2021?", "expected_subquery_count": 1, "expected_routes": [K]},
    {"category": "10k", "query": "How much did Lyft spend on research and development in 2022?", "expected_subquery_count": 1, "expected_routes": [K]},
    # Internet / current
    {"category": "internet", "query": "What are the newest LLMs?", "expected_subquery_count": 1, "expected_routes": [I]},
    {"category": "internet", "query": "What is the weather in Denver today?", "expected_subquery_count": 1, "expected_routes": [I]},
    {"category": "internet", "query": "Who won the most recent FIFA World Cup?", "expected_subquery_count": 1, "expected_routes": [I]},
    {"category": "internet", "query": "What is the current price of Bitcoin?", "expected_subquery_count": 1, "expected_routes": [I]},
    {"category": "internet", "query": "What's the difference between ChatGPT and Claude?", "expected_subquery_count": 1, "expected_routes": [I]},
    {"category": "internet", "query": "What are the best travel destinations in 2026?", "expected_subquery_count": 1, "expected_routes": [I]},
    # Ambiguous — more than one source is defensible
    {"category": "ambiguous", "query": "What is GPT-4o?", "expected_subquery_count": 1, "expected_routes": [(O, I)]},
    {"category": "ambiguous", "query": "How is Uber doing financially right now?", "expected_subquery_count": 1, "expected_routes": [(K, I)]},
    {"category": "ambiguous", "query": "How much does OpenAI charge for its API?", "expected_subquery_count": 1, "expected_routes": [(O, I)]},
    {"category": "ambiguous", "query": "How does Lyft make money?", "expected_subquery_count": 1, "expected_routes": [(K, I)]},
    # Compound, mixed domain
    {"category": "compound", "query": "What was Lyft revenue in 2021 and what was Uber revenue in 2021?", "expected_subquery_count": 2, "expected_routes": [K, K]},
    {"category": "compound", "query": "What was Uber's 2021 revenue and what are the newest LLMs?", "expected_subquery_count": 2, "expected_routes": [K, I]},
    {"category": "compound", "query": "What is an AI agent and how do I add guardrails to one with the OpenAI Agents SDK?", "expected_subquery_count": 2, "expected_routes": [O, O]},
    {"category": "compound", "query": "How do I build an agent with the OpenAI Agents SDK, what was Uber's revenue in 2021, and what is the weather in Paris today?", "expected_subquery_count": 3, "expected_routes": [O, K, I]},
    {"category": "compound", "query": "What was Lyft's revenue in 2021 and what is Lyft's stock price today?", "expected_subquery_count": 2, "expected_routes": [K, I]},
    {"category": "compound", "query": "What are OpenAI embeddings and who won the latest Super Bowl?", "expected_subquery_count": 2, "expected_routes": [O, I]},
]

COMPOUND_EVAL_SET: list[dict] = [
    {"case": "same-route", "query": "What was Lyft revenue in 2021 and what was Uber revenue in 2021?",
     "expected_subquery_count": 2, "expected_routes": [K, K], "full_pipeline": True},
    {"case": "mixed-route", "query": "What was Uber's 2021 revenue and what are the newest LLMs?",
     "expected_subquery_count": 2, "expected_routes": [K, I], "full_pipeline": True},
    {"case": "3-part", "query": "How do I add guardrails with the OpenAI Agents SDK, what was Lyft's revenue in 2021, and what are the newest LLMs?",
     "expected_subquery_count": 3, "expected_routes": [O, K, I], "full_pipeline": True},
    {"case": "duplicate", "query": "What was Uber revenue in 2021? What was Uber revenue in 2021?",
     "expected_subquery_count": 1, "expected_routes": [K], "full_pipeline": False},
    {"case": f"over cap (7 questions → {MAX_SUB_QUERIES})",
     "query": "What is an AI agent? What are guardrails in the OpenAI Agents SDK? What are handoffs in the OpenAI Agents SDK? "
              "What was Uber revenue in 2021? What was Lyft revenue in 2021? What are the newest LLMs? What is the weather in Tokyo today?",
     "expected_subquery_count": MAX_SUB_QUERIES, "expected_routes": [O, O, O, K, K], "full_pipeline": False},
]

BENCHMARK_QUERIES: list[str] = [
    "What was Lyft revenue in 2021 and what was Uber revenue in 2021?",
    "What was Uber's 2021 revenue and what are the newest LLMs?",
    "How do I add guardrails with the OpenAI Agents SDK, what was Lyft's revenue in 2021, and what are the newest LLMs?",
    "What are OpenAI embeddings and who won the latest Super Bowl?",
]


# ── Scoring helpers ───────────────────────────────────────────────────────────
def _options(expected: ExpectedRoute) -> set:
    return set(expected) if isinstance(expected, (tuple, list, set)) else {expected}


def _fmt_routes(routes: list) -> str:
    return " | ".join("/".join(sorted(_options(r))) for r in routes)


def match_routes(expected: list[ExpectedRoute], actual: list[str]) -> int:
    """How many expected route slots are filled by a distinct actual route (order-free)."""
    unused, matched = list(actual), 0
    for slot in expected:
        for i, route in enumerate(unused):
            if route in _options(slot):
                matched += 1
                del unused[i]
                break
    return matched


def citations_traceable(parts: list[dict], final: str) -> bool:
    """
    Every [k.n] in the final answer points at a real citation [n] in sub-answer k,
    and every sub-answer that cited something is still cited.
    """
    tags = TAG.findall(final)
    for k, n in tags:
        k, n = int(k), int(n)
        if not 1 <= k <= len(parts) or n not in {int(x) for x in CITATION.findall(parts[k - 1]["answer"] or "")}:
            return False
    cited = {int(k) for k, _ in tags}
    needs = {k for k, p in enumerate(parts, 1) if CITATION.search(p["answer"] or "")}
    return needs <= cited


# ── Live suites ───────────────────────────────────────────────────────────────
def run_routing_eval(ctx: AgentContext, dataset: list[dict] = ROUTING_EVAL_SET,
                     max_workers: int = 4) -> tuple[pd.DataFrame, dict]:
    """
    Split + route every query with the real implementation (no retrieval).

    Routing Accuracy = correct route decisions / total route decisions, where a query's
                       decisions count max(expected, actual) so over-splitting is penalised.
    Split Accuracy   = queries with the expected number of sub-queries / total queries.
    Latency          = split + routing wall-clock for that query.
    """
    def one(rec: dict) -> dict:
        t0 = time.perf_counter()
        subs, source = ctx.split(rec["query"], with_source=True)
        actual = [ctx.route(s)["action"] for s in subs]
        latency = time.perf_counter() - t0
        correct = match_routes(rec["expected_routes"], actual)
        decisions = max(len(rec["expected_routes"]), len(actual))
        split_ok = len(subs) == rec["expected_subquery_count"]
        return {
            "Category": rec.get("category", ""),
            "Query": rec["query"],
            "Expected Subqueries": rec["expected_subquery_count"],
            "Actual Subqueries": len(subs),
            "Expected Routes": _fmt_routes(rec["expected_routes"]),
            "Actual Routes": " | ".join(actual),
            "Split Correct": split_ok,
            "Route Correct": split_ok and correct == decisions,
            "Latency": round(latency, 2),
            "_correct": correct, "_decisions": decisions, "Split Source": source,
            "Sub-queries": subs,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        rows = list(pool.map(one, dataset))
    df = pd.DataFrame(rows)
    summary = {
        "queries": len(df),
        "route_decisions": int(df["_decisions"].sum()),
        "correct_route_decisions": int(df["_correct"].sum()),
        "routing_accuracy": df["_correct"].sum() / df["_decisions"].sum(),
        "split_accuracy": df["Split Correct"].mean(),
        "mean_latency_s": df["Latency"].mean(),
    }
    return df.drop(columns=["_correct", "_decisions"]), summary


def run_compound_eval(ctx: AgentContext, dataset: list[dict] = COMPOUND_EVAL_SET) -> pd.DataFrame:
    """Compound queries through the real pipeline; full answers + citation checks where flagged."""
    rows = []
    for rec in dataset:
        row = {"Case": rec["case"], "Query": rec["query"],
               "Expected Subqueries": rec["expected_subquery_count"],
               "Expected Routes": _fmt_routes(rec["expected_routes"])}
        try:
            if rec["full_pipeline"]:
                d = agentic_rag_multi(ctx, rec["query"], return_details=True, verbose=False)
                subs, actual = d["sub_queries"], [p["action"] for p in d["parts"]]
                row["Citations Traceable"] = citations_traceable(d["parts"], d["answer"])
                row["Compose Method"] = d["compose_method"]
            else:
                subs = ctx.split(rec["query"])
                actual = [ctx.route(s)["action"] for s in subs]
                row["Citations Traceable"] = "n/a"
                row["Compose Method"] = "n/a (split + route only)"
            row.update({
                "Actual Subqueries": len(subs),
                "Actual Routes": " | ".join(actual),
                "Within Cap": len(subs) <= MAX_SUB_QUERIES,
                "No Duplicates": len({s.strip().lower() for s in subs}) == len(subs),
                "Split Correct": len(subs) == rec["expected_subquery_count"],
                "Route Correct": match_routes(rec["expected_routes"], actual) == len(rec["expected_routes"]) == len(actual),
                "No Crash": True,
            })
        except Exception as err:  # the suite reports a crash instead of dying with it
            row.update({"No Crash": False, "Error": repr(err)})
        checks = [row.get(c) for c in ("No Crash", "Within Cap", "No Duplicates", "Split Correct", "Route Correct")]
        row["Pass"] = all(checks) and row.get("Citations Traceable") in (True, "n/a")
        rows.append(row)
    return pd.DataFrame(rows)


def run_concurrency_benchmark(ctx: AgentContext, queries: list[str] = BENCHMARK_QUERIES
                              ) -> tuple[pd.DataFrame, dict]:
    """
    Wall-clock of agentic_rag_multi() sequential vs concurrent for each query.
    The order alternates per query so neither mode always runs second (warm caches).
    """
    rows = []
    for i, q in enumerate(queries):
        timings, counts = {}, {}
        for mode in ((False, True) if i % 2 == 0 else (True, False)):
            t0 = time.perf_counter()
            d = agentic_rag_multi(ctx, q, concurrent=mode, return_details=True, verbose=False)
            timings[mode] = time.perf_counter() - t0
            counts[mode] = len(d["sub_queries"])
        rows.append({
            "Query": q,
            "Number of Subqueries": counts[False] if counts[False] == counts[True]
                                    else f"{counts[False]} seq / {counts[True]} conc",
            "Sequential Time": round(timings[False], 2),
            "Concurrent Time": round(timings[True], 2),
            "Speedup": round(timings[False] / timings[True], 2),
        })
    df = pd.DataFrame(rows)
    faster = int((df["Concurrent Time"] < df["Sequential Time"]).sum())
    summary = {
        "avg_sequential_s": df["Sequential Time"].mean(),
        "avg_concurrent_s": df["Concurrent Time"].mean(),
        "avg_speedup": df["Speedup"].mean(),
        "concurrent_faster_in": f"{faster}/{len(df)}",
    }
    if faster == len(df) and summary["avg_speedup"] > 1.05:
        summary["conclusion"] = "Concurrent execution was faster on every measured query."
    elif faster > len(df) / 2:
        summary["conclusion"] = "Concurrent execution was faster on most, but not all, measured queries."
    else:
        summary["conclusion"] = "No consistent speed-up was measured; LLM/API latency variance dominates."
    return df, summary


# Ground truth read from the 10-K text in the Qdrant collection (revenue tables).
# "right" accepts any rounding of the asked year's figure; "wrong" flags a neighbouring year's.
ANSWER_SPOT_CHECKS: list[dict] = [
    {"query": "What was Uber's revenue in 2021?", "truth": "$17,455M",
     "right": r"17[.,]\d+\s*(billion|B)|17,455", "wrong": r"11[.,]1\d*\s*(billion|B)|11,139"},
    {"query": "What was Lyft's revenue in 2021?", "truth": "$3,208,323K",
     "right": r"3[.,]2\d*\s*(billion|B)|3,208", "wrong": r"4[.,][01]\d*\s*(billion|B)|4,095|2[.,]36\d*\s*(billion|B)|2,364"},
    {"query": "What was Lyft's revenue in 2022?", "truth": "$4,095,135K",
     "right": r"4[.,][01]\d*\s*(billion|B)|4,095", "wrong": r"3[.,]2\d*\s*(billion|B)|3,208"},
    {"query": "What was Lyft's revenue in 2020?", "truth": "$2,364,681K",
     "right": r"2[.,]36\d*\s*(billion|B)|2[.,]4\s*(billion|B)|2,364", "wrong": r"3[.,]2\d*\s*(billion|B)|3,208|4,095"},
]


def run_answer_spot_check(ctx: AgentContext, checks: list[dict] = ANSWER_SPOT_CHECKS,
                          runs: int = 3, max_workers: int = 4) -> pd.DataFrame:
    """
    Factual spot-check of 10-K answers (retrieval + RAG, routing bypassed): each question is
    answered `runs` times and graded against the filing's figures.
    """
    from .retrieval import execute_route

    def one(check: dict) -> str:
        return execute_route(check["query"], K, ctx.routes)

    rows = []
    for check in checks:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            answers = list(pool.map(lambda _: one(check), range(runs)))
        right = sum(bool(re.search(check["right"], a)) and not re.search(check["wrong"], a) for a in answers)
        wrong = sum(bool(re.search(check["wrong"], a)) for a in answers)
        rows.append({"Query": check["query"], "10-K Figure": check["truth"],
                     "Correct": f"{right}/{runs}", "Wrong Year": f"{wrong}/{runs}",
                     "Sample Answer": answers[0][:120].replace("\n", " "), "Pass": right == runs})
    return pd.DataFrame(rows)


def run_cache_matrix(ctx: AgentContext, make_cache: Callable[[], Any]) -> pd.DataFrame:
    """RBAC + semantic-cache scenarios against the real router and pipeline."""
    cache, log, rows = make_cache(), [], []
    q_fin = "what was uber revenue in 2021?"
    q_doc = "how do I build an agent with the OpenAI Agents SDK?"
    q_live = "what are the latest LLMs released this week?"

    def ask(user: str, q: str) -> dict:
        return secure_agentic_rag_cached(ctx, user, q, cache, verbose=False, audit_log=log)

    def record(scenario: str, user: str, q: str, source: str, expected: str, actual: str) -> None:
        rows.append({"Scenario": scenario, "User": user, "Role": auth.USERS.get(user, "UNKNOWN"),
                     "Query": q, "Query Source": source, "Expected": expected,
                     "Actual": actual, "Pass": expected == actual})

    def scenario(name: str, user: str, q: str, expected: str) -> dict:
        r = ask(user, q)
        record(name, user, q, r["route"] or "-", expected, r["status"])
        return r

    scenario("authorized MISS", "bob", q_fin, "MISS")
    scenario("authorized HIT", "bob", q_fin, "HIT")
    scenario("unauthorized DENIED", "alice", q_fin, "DENIED")
    scenario("paraphrased unauthorized DENIED", "alice", "how much revenue did Uber make in 2021?", "DENIED")
    scenario("unknown user DENIED", "carol", q_doc, "DENIED")
    scenario("shared source MISS", "alice", q_doc, "MISS")
    scenario("shared source reused across roles", "bob", q_doc, "HIT")

    size = len(cache)
    ask("alice", q_live)
    r = ask("alice", q_live)
    record("time-sensitive query bypass (2nd ask)", "alice", q_live, r["route"] or "-",
           f"MISS, cache size {size}", f"{r['status']}, cache size {len(cache)}")

    size = len(cache)
    r = ask("alice", q_fin)
    record("denied request not cached", "alice", q_fin, r["route"] or "-",
           f"DENIED, cache size {size}", f"{r['status']}, cache size {len(cache)}")

    original = auth.USERS["bob"]
    try:
        auth.USERS["bob"] = "engineer"
        scenario("role change (bob → engineer)", "bob", q_fin, "DENIED")
        hit, *_ = cache.check("bob", q_fin, source=K)
        record("role change: direct cache read", "bob", q_fin, K, "no hit", "hit" if hit else "no hit")
    finally:
        auth.USERS["bob"] = original
    scenario("role restored", "bob", q_fin, "HIT")

    try:
        cache.add("alice", "fake question", "fake answer", source=K)
        actual = "write accepted"
    except PermissionError:
        actual = "PermissionError"
    record("unauthorized cache write prevented", "alice", "fake question", K, "PermissionError", actual)
    return pd.DataFrame(rows)


# ── Offline suites ────────────────────────────────────────────────────────────
def run_parser_robustness_tests() -> pd.DataFrame:
    """Splitter + router against scripted bad model output. No API calls."""
    q = "what was uber revenue in 2021 and what are the newest LLMs?"
    A, B = "What was Uber revenue in 2021?", "What are the newest LLMs?"
    many = [f"Question {i}?" for i in range(8)]
    split_cases = [
        # (case, model output, structured supported, expected questions, expected source)
        ("clean structured JSON", '{"subQuestions": ["%s", "%s"]}' % (A, B), True, [A, B], "structured"),
        ("Markdown code fence", '```json\n{"subQuestions": ["%s", "%s"]}\n```' % (A, B), True, [A, B], "parser"),
        ("prose around JSON", 'Sure! Here you go: {"subQuestions": ["%s", "%s"]} Hope that helps.' % (A, B), True, [A, B], "parser"),
        ("bare JSON list", '["%s", "%s"]' % (A, B), True, [A, B], "parser"),
        ("alternative key", '{"sub_queries": ["%s", "%s"]}' % (A, B), True, [A, B], "parser"),
        ("duplicates + blanks", '{"subQuestions": ["%s", "%s", "  what was uber revenue in 2021 ", ""]}' % (A, A), True, [A], "structured"),
        (f"over cap (8 → {MAX_SUB_QUERIES})", '{"subQuestions": %s}' % str(many).replace("'", '"'), True, many[:MAX_SUB_QUERIES], "structured"),
        ("truncated JSON", '{"subQuestions": ["%s", ' % A, True, [q], "fallback"),
        ("empty response", "", True, [q], "fallback"),
        ("None content", None, True, [q], "fallback"),
        ("empty list", '{"subQuestions": []}', True, [q], "fallback"),
        ("no structured outputs → parser", '```json\n{"subQuestions": ["%s", "%s"]}\n```' % (A, B), False, [A, B], "parser"),
        ("API error", RuntimeError("503 Service Unavailable"), True, [q], "fallback"),
    ]
    rows = []
    for case, output, structured, expected, exp_source in split_cases:
        llm = ScriptedLLM(lambda p, f, out=output: out, structured=structured)
        try:
            got, source = split_query(q, llm, with_source=True)
            ok = got == expected and source == exp_source
        except Exception as err:
            got, source, ok = repr(err), "CRASH", False
        rows.append({"Component": "splitter", "Case": case, "Model Output": repr(output)[:60],
                     "Result": got, "Parsed By": source, "Expected": expected, "Pass": ok})

    route_cases = [
        ("clean structured JSON", '{"action": "10K_DOCUMENT_QUERY", "reason": "r", "answer": ""}', True, K, "structured"),
        ("Markdown code fence", '```json\n{"action": "OPENAI_QUERY", "reason": "r"}\n```', True, O, "parser"),
        ("prose around JSON", 'I think: {"action": "10K_DOCUMENT_QUERY", "reason": "r", "answer": ""} done', True, K, "parser"),
        ("unknown route label", '{"action": "SQL_QUERY", "reason": "r", "answer": ""}', True, I, "fallback"),
        ("not JSON", "OPENAI_QUERY", True, I, "fallback"),
        ("no structured outputs → parser", '{"action": "OPENAI_QUERY", "reason": "r", "answer": ""}', False, O, "parser"),
        ("API error", RuntimeError("401 Unauthorized"), True, I, "fallback"),
    ]
    for case, output, structured, expected, exp_source in route_cases:
        llm = ScriptedLLM(lambda p, f, out=output: out, structured=structured)
        try:
            d = route_query("some question", llm)
            got, source = d["action"], d["parsed_by"]
            ok = got == expected and source == exp_source
        except Exception as err:
            got, source, ok = repr(err), "CRASH", False
        rows.append({"Component": "router", "Case": case, "Model Output": repr(output)[:60],
                     "Result": got, "Parsed By": source, "Expected": expected, "Pass": ok})
    return pd.DataFrame(rows)


def run_citation_tests() -> pd.DataFrame:
    """Citation namespacing and composition fallback. No API calls."""
    rows = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        rows.append({"Test": name, "Pass": bool(passed), "Detail": detail})

    out = namespace_citations("Revenue was $17.5B [1].", 1)
    check("[1] from sub-question 1 → [1.1]", out == "Revenue was $17.5B [1.1].", out)
    out = namespace_citations("Newest model is X [1].", 2)
    check("[1] from sub-question 2 → [2.1]", out == "Newest model is X [2.1].", out)

    a, b = namespace_citations("A [1][2]", 1), namespace_citations("B [1]", 2)
    tags_a, tags_b = set(TAG.findall(a)), set(TAG.findall(b))
    check("identifiers do not collide", not (tags_a & tags_b) and len(tags_a | tags_b) == 3,
          f"{sorted(tags_a)} vs {sorted(tags_b)}")

    out = namespace_citations("[Direct Answer] foo\n[1] Title", 2)
    check("[Direct Answer] label untouched", out == "[Direct Answer] foo\n[2.1] Title", out)
    out = namespace_citations("Both agree [1, 2].", 3)
    check("grouped [1, 2] → [3.1][3.2]", out == "Both agree [3.1][3.2].", out)
    out = namespace_citations("See [1.5] and [12].", 1)
    check("decimal tags untouched, multi-digit namespaced", out == "See [1.5] and [1.12].", out)

    def parts() -> list[dict]:
        return [{"question": "Lyft revenue 2021?", "action": K, "answer": "Lyft: $3.2B [1][2]."},
                {"question": "Newest LLMs?", "action": I, "answer": "Model X [1]."}]

    def compose_llm(reply: Any) -> ScriptedLLM:
        return ScriptedLLM(lambda p, f: reply)

    text, method = compose_answer("q", parts(), compose_llm("Lyft made $3.2B [1.1][1.2]; newest is X [2.1]."),
                                  with_method=True)
    check("composition preserves namespaces", method == "llm" and all(t in text for t in ("[1.1]", "[1.2]", "[2.1]")),
          f"method={method}")

    text, method = compose_answer("q", parts(), compose_llm("Lyft made $3.2B [1.1]; newest is X."), with_method=True)
    check("dropped citations → deterministic composition",
          method.startswith("deterministic") and "[2.1]" in text and "**2. Newest LLMs?**" in text, f"method={method}")

    text, method = compose_answer("q", parts(), compose_llm(RuntimeError("timeout")), with_method=True)
    check("composition LLM error → deterministic composition",
          method.startswith("deterministic") and all(t in text for t in ("[1.1]", "[1.2]", "[2.1]")), f"method={method}")

    no_cite = [{"question": "q1", "action": K, "answer": "No relevant content found in the database."},
               {"question": "q2", "action": I, "answer": "Model X [1]."}]
    text, method = compose_answer("q", no_cite, compose_llm("Nothing found for q1; X [2.1]."), with_method=True)
    check("sub-answer without citations is not required to be cited", method == "llm", f"method={method}")
    return pd.DataFrame(rows)
