# Agentic Router — Sub-query Division & RBAC-aware Semantic Cache

Assignment for Module 3 of [multi-agent-course](https://github.com/hamzafarooq/multi-agent-course):
`001. Agentic Router.ipynb`, run end to end.

- **Part 1 (required) — sub-query division:** `agentic_rag_multi()` splits a compound
  question, routes each sub-question independently, and composes one answer with
  namespaced citations (`[1.1]`, `[2.1]`). Falls back safely on malformed splits;
  optional concurrent mode.
- **Bonus — RBAC + semantic cache:** `RoleAwareSemanticCache` (FAISS, partitioned per
  knowledge source) and `secure_agentic_rag_cached()`; the no-leak self-check passes.

## Running

Colab: add `OPENAI_API_KEY` and `SERP_API_KEY` in the Secrets panel.
Locally: `cp .env.example .env`, fill in the keys, and run the notebook from the
course's `Module_3_Production_Agentic_RAG_AI_Systems` folder (it needs `Agentic_RAG/qdrant_data`).
On macOS, set `KMP_DUPLICATE_LIB_OK=TRUE` if torch and faiss crash together.
