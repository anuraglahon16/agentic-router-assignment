"""Role-based access control: who may reach which knowledge source (Section 6)."""
from __future__ import annotations

# ── Users → role ─────────────────────────────────────────────────────────────
# Stand-in for a real identity provider. In production this comes from SSO/JWT
# claims or an internal users table — never a dict in the notebook.
USERS: dict[str, str] = {
    "alice": "engineer",
    "bob":   "finance_analyst",
}

# ── Roles → the route labels each role may reach ─────────────────────────────
# This is an allow-list: anything not listed here is denied by default.
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "engineer":        {"OPENAI_QUERY", "INTERNET_QUERY"},
    "finance_analyst": {"OPENAI_QUERY", "10K_DOCUMENT_QUERY"},
}

# Human-readable names, used for denial messages and citation keys
SOURCE_LABELS: dict[str, str] = {
    "OPENAI_QUERY":       "OpenAI documentation",
    "10K_DOCUMENT_QUERY": "10-K financial filings",
    "INTERNET_QUERY":     "live internet search",
}


def has_access(user_id: str, action: str) -> bool:
    """True only if this user's role is explicitly allowed to use this route."""
    role = USERS.get(user_id)
    return role is not None and action in ROLE_PERMISSIONS.get(role, set())


def allowed_sources(user_id: str) -> set[str]:
    """Every route label this user may reach (empty for unknown users)."""
    return ROLE_PERMISSIONS.get(USERS.get(user_id), set())
