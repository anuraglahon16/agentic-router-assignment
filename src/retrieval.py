"""
Route execution — dispatch a routed question to its handler.

The handlers themselves (Qdrant retrieval + RAG, SerpApi web search) are the course's
code and stay in the notebook; they are passed in as a `routes` dict of
`handler(user_query, action) -> str` (sync) or `-> Awaitable[str]` (async).
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine

RouteHandler = Callable[[str, str], Any]

# Handler outputs that signal failure — never cache these.
ERROR_PREFIXES: tuple[str, ...] = (
    "Embedding error", "Vector DB query error", "RAG response error", "Unexpected error",
    "Invalid action", "Unsupported action", "Execution error", "HTTP error",
    "Request error", "An unexpected error", "No results found", "No relevant content",
)


def run_coro(coro: Coroutine) -> Any:
    """
    Run a coroutine from sync code, in any thread.

    Main Jupyter thread: a loop is already running, so use asyncio.run (nest_asyncio
    makes that legal). Worker thread (concurrent mode) or plain Python: there is no
    running loop, so drive a fresh one — nest_asyncio's patched asyncio.run would
    raise 'no current event loop' in a worker thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    return asyncio.run(coro)


def execute_route(user_query: str, action: str, routes: dict[str, RouteHandler]) -> str:
    """Step 4 of agentic_rag(): call the chosen handler, awaiting it if it is async."""
    route_function = routes.get(action)
    if not route_function:
        return f"Unsupported action: {action}"
    try:
        result = route_function(user_query, action)
        if asyncio.iscoroutine(result):
            result = run_coro(result)
        return result
    except Exception as exec_err:
        return f"Execution error: {exec_err}"


def is_error_answer(answer: Any) -> bool:
    """True for empty, non-string, or handler-error answers."""
    return not isinstance(answer, str) or not answer.strip() or answer.startswith(ERROR_PREFIXES)
