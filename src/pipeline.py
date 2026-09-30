"""
Orchestrators:

    agentic_rag_multi()         split → route each sub-question → answer → compose
    secure_agentic_rag_cached() identity → route → permission → cache → pipeline
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from . import auth
from .cache import is_time_sensitive
from .composer import compose_answer, source_map
from .retrieval import RouteHandler, execute_route, is_error_answer, run_coro
from .router import DEFAULT_MODEL, route_query
from .splitter import split_query

CYAN, GREY, RED, GREEN, YELLOW, BOLD, RESET = (
    "\033[96m", "\033[90m", "\033[91m", "\033[92m", "\033[93m", "\033[1m", "\033[0m"
)


@dataclass
class AgentContext:
    """What the pipeline needs from the outside world — built once in the notebook."""
    llm: Any                                   # OpenAI client
    routes: dict[str, RouteHandler]            # action → handler(user_query, action)
    model: str = DEFAULT_MODEL
    route_fn: Optional[Callable[[str], dict]] = None   # defaults to router.route_query
    labels: dict[str, str] = field(default_factory=lambda: auth.SOURCE_LABELS)

    def route(self, question: str) -> dict:
        if self.route_fn is not None:
            return self.route_fn(question)
        return route_query(question, self.llm, self.model)

    def split(self, question: str, with_source: bool = False):
        return split_query(question, self.llm, self.model, with_source=with_source)


# ── Part 1: sub-query division ────────────────────────────────────────────────
def answer_sub_query(ctx: AgentContext, sub_query: str) -> dict:
    """Route ONE sub-query on its own and answer it from the chosen source."""
    t0 = time.perf_counter()
    decision = ctx.route(sub_query)
    action = decision.get("action")
    return {
        "question": sub_query,
        "action": action,
        "reason": decision.get("reason"),
        "parsed_by": decision.get("parsed_by"),
        "answer": execute_route(sub_query, action, ctx.routes),
        "latency_s": round(time.perf_counter() - t0, 2),
    }


async def _answer_all_concurrently(ctx: AgentContext, subs: list[str]) -> list[dict]:
    """Stretch: every sub-query's route + retrieval runs in its own thread."""
    return await asyncio.gather(*(asyncio.to_thread(answer_sub_query, ctx, q) for q in subs))


def agentic_rag_multi(ctx: AgentContext, user_query: str, concurrent: bool = False,
                      return_details: bool = False, verbose: bool = True):
    """
    Split a compound query, run each sub-query through the agentic pipeline,
    and synthesise one final answer.

    LLM calls: 1 split + (1 route + the route's own calls) per sub-query
               + 1 composition only when there is more than one sub-query.

    Args:
        ctx: Clients and route handlers.
        user_query: Possibly compound question.
        concurrent: Run sub-queries in parallel threads (stretch goal).
        return_details: Return a dict with the per-sub-query parts as well.
        verbose: Print the trace (sub-queries, routes, answer).

    Returns:
        str: A single composed answer covering every sub-question, or a dict with
        'answer', 'sub_queries', 'split_source', 'parts', 'compose_method' if return_details.
    """
    say = print if verbose else (lambda *a, **k: None)
    say(f"{BOLD}{CYAN}👤 User Query:{RESET} {user_query}\n")

    # 1. Split
    subs, split_source = ctx.split(user_query, with_source=True)
    say(f"{GREY}🔀 {len(subs)} sub-quer{'y' if len(subs) == 1 else 'ies'} ({split_source}):")
    for k, q in enumerate(subs, 1):
        say(f"   {k}. {q}")
    say(RESET)

    # 2. Route + answer each sub-query independently
    if len(subs) > 1 and concurrent:
        parts = run_coro(_answer_all_concurrently(ctx, subs))
    else:
        parts = [answer_sub_query(ctx, q) for q in subs]

    for k, p in enumerate(parts, 1):
        say(f"{GREY}📍 [{k}] Route: {p['action']}  —  {p['reason']}{RESET}")
    say()

    # 3. Compose — a single question is returned as-is, exactly like agentic_rag()
    if len(parts) == 1:
        final, method = parts[0]["answer"], "single (no composition)"
    else:
        composed, method = compose_answer(user_query, parts, ctx.llm, ctx.model,
                                          ctx.labels, with_method=True)
        final = composed + "\n\n" + source_map(parts, ctx.labels)

    say(f"{BOLD}{CYAN}🤖 BOT RESPONSE:{RESET}\n")
    say(f"{final}\n")

    if return_details:
        return {"answer": final, "sub_queries": subs, "split_source": split_source,
                "parts": parts, "compose_method": method}
    return final


# ── Bonus: RBAC + semantic cache ──────────────────────────────────────────────
AUDIT_LOG: list[dict] = []


def secure_agentic_rag_cached(ctx: AgentContext, user_id: str, user_query: str, cache,
                              verbose: bool = True, audit_log: Optional[list] = None) -> dict:
    """
    RBAC-gated agentic RAG with a role-aware semantic cache.

    Order: identity → route → permission → cache → pipeline.

    Returns:
        dict: {"answer": str, "status": "HIT" | "MISS" | "DENIED", "role": str | None,
               "route": str | None}
    """
    log = AUDIT_LOG if audit_log is None else audit_log
    t0 = time.perf_counter()
    role = auth.USERS.get(user_id)

    def finish(answer: str, status: str, route: Optional[str] = None, note: str = "") -> dict:
        log.append({
            "user": user_id, "role": role or "UNKNOWN", "query": user_query,
            "route": route or "-", "decision": "DENY" if status == "DENIED" else "ALLOW",
            "cache": status, "note": note,
            "latency_ms": round((time.perf_counter() - t0) * 1000),
        })
        if verbose:
            colour = {"DENIED": RED, "HIT": GREEN, "MISS": YELLOW}[status]
            print(f"{BOLD}{CYAN}👤 {user_id}{RESET} ({role or 'UNKNOWN'}) · {user_query}")
            print(f"   {colour}{status}{RESET} {GREY}route={route or '-'}  {note}{RESET}")
            if status != "DENIED":
                print(f"   {answer[:200]}{'…' if len(answer) > 200 else ''}")
            print()
        return {"answer": answer, "status": status, "role": role, "route": route}

    # 1. Unknown identity → denied before any embedding, cache lookup or LLM call
    if role is None:
        return finish(f"🚫 Access denied: unknown user '{user_id}'.", "DENIED", note="unknown user")

    # 2. Route — the router decides which source the question needs
    try:
        decision = ctx.route(user_query)
    except Exception as route_err:
        return finish(f"Routing error: {route_err}", "DENIED", note="routing failed (fail closed)")
    action = decision.get("action")

    # 3. Permission gate — BEFORE the cache, so a paraphrase can't reach another
    #    role's entries and a denial is never looked up or stored.
    if not auth.has_access(user_id, action):
        source = ctx.labels.get(action, action)
        return finish(
            f"🚫 Access denied: your role ('{role}') does not have permission to query {source}.",
            "DENIED", action, note=f"role may not query {source}",
        )

    # 4. Cache lookup — only in the partition of the source that was just authorised
    volatile = is_time_sensitive(user_query)
    embedding = None
    if not volatile:
        hit, answer, embedding, sim = cache.check(user_id, user_query, source=action)
        if hit:
            return finish(answer, "HIT", action, note=f"similarity={sim:.3f}")

    # 5. Miss → run the pipeline; store only stable, successful answers
    answer = execute_route(user_query, action, ctx.routes)
    if volatile:
        note = "time-sensitive → not cached"
    elif is_error_answer(answer):
        note = "error → not cached"
    else:
        cache.add(user_id, user_query, answer, embedding, source=action)
        note = "stored"
    return finish(answer, "MISS", action, note=note)


def print_audit_log(log: Optional[list] = None) -> None:
    """The table you'd hand an auditor."""
    log = AUDIT_LOG if log is None else log
    cols = [("user", 7), ("role", 16), ("route", 19), ("decision", 8),
            ("cache", 6), ("latency_ms", 10), ("query", 48)]
    header = " | ".join(f"{c:<{w}}" for c, w in cols)
    print(header)
    print("-" * len(header))
    for rec in log:
        print(" | ".join(f"{str(rec[c])[:w]:<{w}}" for c, w in cols))
