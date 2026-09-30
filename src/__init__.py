"""
Agentic Router — reusable implementation behind `001. Agentic Router.ipynb`.

The notebook stays the executable walkthrough; this package holds the logic so it can
be tested without API keys (`python -m pytest`) and reviewed module by module:

    auth.py        roles, permissions, has_access()
    router.py      route one question to one knowledge source (structured output + fallback)
    splitter.py    split a compound question (structured output + defensive parser + fallback)
    retrieval.py   dispatch a routed question to its retrieval / web-search handler
    citations.py   namespace [n] → [k.n] and verify citations survive composition
    composer.py    merge sub-answers into one answer (LLM, with deterministic fallback)
    cache.py       role-aware semantic cache (FAISS or NumPy)
    pipeline.py    agentic_rag_multi() and secure_agentic_rag_cached()
    evaluation.py  routing / compound / citation / benchmark / RBAC evaluation suites
"""
