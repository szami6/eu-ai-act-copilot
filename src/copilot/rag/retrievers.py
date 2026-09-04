"""Hybrid retrieval: dense (Qdrant HNSW) + sparse (BM25), fused by Reciprocal
Rank Fusion (PLAN.md §3.2 `retrieve`, §4.5).

Deliberately *not* Qdrant's native sparse-vector/RRF support: PLAN.md §4.5
keeps BM25 in-process "because it keeps the hybrid logic readable in
retrievers.py — with a README note that at scale it moves into Qdrant's
native sparse vectors." Qdrant remains the single source of truth for chunk
text + payload (`ingest/index.py`); `load_bm25_index` just mirrors it into an
in-memory `rank_bm25` index at startup rather than persisting a second copy.

Pure fusion/scoring logic (`build_bm25_index`, `Bm25Index.search`,
`reciprocal_rank_fusion`) takes already-fetched chunks or already-embedded
query vectors, so it is unit-testable without a live Qdrant or embedding
model — mirroring `ingest.chunk`'s pure/network-free design. Only
`dense_search` and `load_bm25_index` touch a live `QdrantClient`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

_RRF_K = 60
_SCROLL_PAGE_SIZE = 256


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, numbers and roman numerals kept as-is — this
    corpus's exact identifiers ("Annex III", "point 4") are what BM25 is
    here to catch (PLAN.md §3.2's `retrieve` rationale).

    Deliberately not `BM25Okapi(corpus, tokenizer=_tokenize)`: passing a
    tokenizer makes `rank_bm25` tokenize the corpus through a multiprocessing
    `Pool`, which is Windows-hostile (spawn/pickling overhead) and pure
    overhead at ~1k chunks — pre-tokenizing once, up front, with this same
    function sidesteps it entirely.
    """
    return re.findall(r"\w+", text.lower())


@dataclass
class IndexedChunk:
    """One chunk's Qdrant payload, mirrored into the in-process BM25 index."""

    chunk_id: str
    text: str
    payload: dict[str, object]


@dataclass
class Bm25Index:
    chunks: list[IndexedChunk]
    bm25: BM25Okapi | None
    by_id: dict[str, IndexedChunk] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.by_id = {c.chunk_id: c for c in self.chunks}

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Up to `k` `(chunk_id, score)` pairs, highest score first."""
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [(self.chunks[i].chunk_id, float(scores[i])) for i in ranked[:k]]


def build_bm25_index(chunks: list[IndexedChunk]) -> Bm25Index:
    tokenized_corpus = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus) if chunks else None
    return Bm25Index(chunks=chunks, bm25=bm25)


def load_bm25_index(client: QdrantClient, collection_name: str) -> Bm25Index:
    """Scroll every point out of `collection_name` and build a fresh BM25
    index from its `text` payload field. Call once at process startup —
    each `search()` call afterwards is in-memory and network-free."""
    chunks: list[IndexedChunk] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=_SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            payload = record.payload or {}
            chunks.append(
                IndexedChunk(
                    chunk_id=str(payload["chunk_id"]),
                    text=str(payload["text"]),
                    payload=payload,
                )
            )
        if offset is None:
            break
    return build_bm25_index(chunks)


def _build_filter(filters: dict[str, str | int | bool] | None) -> models.Filter | None:
    if not filters:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in filters.items()
        ]
    )


def dense_search(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    k: int,
    filters: dict[str, str | int | bool] | None = None,
) -> list[tuple[str, float]]:
    """Up to `k` `(chunk_id, score)` pairs, highest score first."""
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=k,
        query_filter=_build_filter(filters),
        with_payload=["chunk_id"],
    )
    return [
        (str(point.payload["chunk_id"]), point.score) for point in response.points if point.payload
    ]


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]], rrf_k: int = _RRF_K
) -> list[tuple[str, float]]:
    """Fuse any number of ranked `(chunk_id, score)` lists (dense and/or
    sparse, across every query variant) into one ranking, highest first.

    Ignores each list's own raw scores by design — RRF's whole point is
    combining rankings on incomparable scales (cosine similarity vs. BM25)
    purely by rank position: `score(doc) = sum(1 / (rrf_k + rank))` over
    every ranking it appears in, rank 1-indexed.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (chunk_id, _score) in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
