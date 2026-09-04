"""Pure-Python compliance tools — no LLM in this package (PLAN.md §11, Phase
2). Each tool is a plain function over Pydantic models, unit-tested to 100%
on its decision table; LangChain `@tool` wrapping and free-text extraction
are Phase 3 concerns layered on top, not part of these functions themselves.
"""
