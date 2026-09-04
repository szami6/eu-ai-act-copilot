"""Cheap token-count approximation shared by the ingest chunker's size
thresholds (PLAN.md §5.2) and the RAG subgraph's `compress` node (§3.2).

Not tied to any specific embedding/LLM tokenizer on purpose: the ~600-token
figures in PLAN.md are soft chunk-sizing heuristics, not exact context-window
accounting, and pinning a real tokenizer here would need to change every time
the embedding model does (§4.5 documents `multilingual-e5-base` as a swap
candidate). The classic ~4-chars-per-token estimate for English prose is
within a few percent of real BPE/WordPiece counts — well inside the margin
these thresholds need.
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)
