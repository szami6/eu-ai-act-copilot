"""Query-time embedding for the RAG subgraph (PLAN.md §4.5).

Deliberately calls `TextEmbedding.query_embed`, not the `.embed` that
`ingest/index.py` uses for chunks: bge-base-en-v1.5 is an asymmetric model —
`query_embed` applies a retrieval-query instruction prefix internally that
`.embed` does not. Passing a query through `.embed` doesn't raise anything;
it just quietly returns a worse vector, so the distinction is easy to lose
without checking the library source (confirmed live via `inspect`/a
throwaway script, not assumed).
"""

from __future__ import annotations

from copilot.embeddings import get_embedding_model


def embed_query(text: str) -> list[float]:
    model = get_embedding_model()
    (vector,) = model.query_embed(text)
    return vector.tolist()
