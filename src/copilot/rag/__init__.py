"""RAG subgraph (PLAN.md §3.2) — rewrite -> retrieve -> rerank -> grade -> compress.

Compiled and unit-tested in isolation: it knows nothing about the
orchestrator's `AgentState` and is exposed to it only through `RagQuery` and
`RagResult` (`state.py`).
"""
