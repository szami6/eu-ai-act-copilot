"""Cross-encoder reranking + MMR fallback (PLAN.md §3.2 `rerank`, §4.5).

The cross-encoder is the primary, config-gated path. When it's disabled,
`mmr_rerank` diversifies the RRF-fused candidates instead, using dense
vectors fetched back from Qdrant by the stable `chunk_id` payload field —
this module doesn't need to know `ingest.index`'s point-id derivation
(a UUID5 of `chunk_id`), only the field every other module already keys on.

Which path runs is the `rerank` node's decision (it reads config); this
module just provides both mechanics; independently testable.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import replace
from functools import lru_cache
from typing import Protocol, cast

from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

from copilot.rag.state import Evidence

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


class _CrossEncoder(Protocol):
    def rerank(self, query: str, documents: list[str]) -> Iterable[float]: ...


@lru_cache
def get_reranker_model() -> TextCrossEncoder:
    """Cached: loading the ONNX model is a one-off cost worth paying once.

    fastembed does not package a separately int8-quantized build of this
    model (checked `TextCrossEncoder.list_supported_models()` — only the
    standard ONNX export is listed), so this runs at whatever precision that
    export ships at; PLAN.md §4.5's "ONNX int8" is the aspiration, not what
    actually loads here. Worth re-checking if fastembed adds one later.
    """
    return TextCrossEncoder(model_name=RERANKER_MODEL_NAME)


def cross_encoder_rerank(
    query: str,
    candidates: list[Evidence],
    top_n: int = 5,
    model: _CrossEncoder | None = None,
) -> list[Evidence]:
    """Score every candidate against `query`, return the top `top_n` with
    `.score` replaced by the cross-encoder's relevance score.

    `model` is injectable (defaults to the real cached one) so tests can
    pass a tiny fake instead of downloading the ~1 GB real model.
    """
    if not candidates:
        return []
    active_model = model or get_reranker_model()
    scores = list(active_model.rerank(query, [c.text for c in candidates]))
    reranked = [replace(c, score=s) for c, s in zip(candidates, scores, strict=True)]
    reranked.sort(key=lambda e: e.score, reverse=True)
    return reranked[:top_n]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def mmr_rerank(
    query_vector: list[float],
    candidates: list[Evidence],
    candidate_vectors: dict[str, list[float]],
    top_n: int = 5,
    lambda_mult: float = 0.5,
) -> list[Evidence]:
    """Maximal Marginal Relevance — the fallback when the cross-encoder is
    config-disabled. Greedily picks the candidate maximising
    `lambda_mult * relevance - (1 - lambda_mult) * redundancy`, where
    redundancy is the highest cosine similarity to anything already picked.

    Candidates with no entry in `candidate_vectors` (a lookup miss) are
    dropped — they simply can't be scored for diversity.
    """
    pool = [c for c in candidates if c.chunk_id in candidate_vectors]
    selected: list[Evidence] = []

    while pool and len(selected) < top_n:
        best: Evidence | None = None
        best_score = float("-inf")
        for candidate in pool:
            vec = candidate_vectors[candidate.chunk_id]
            relevance = _cosine(query_vector, vec)
            redundancy = max(
                (_cosine(vec, candidate_vectors[s.chunk_id]) for s in selected),
                default=0.0,
            )
            score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
            if score > best_score:
                best, best_score = candidate, score
        assert best is not None  # pool is non-empty inside this loop
        selected.append(best)
        pool.remove(best)

    return selected


def fetch_vectors(
    client: QdrantClient, collection_name: str, chunk_ids: list[str]
) -> dict[str, list[float]]:
    """Look up dense vectors for a small set of chunk ids, keyed by the
    stable `chunk_id` payload field rather than Qdrant's own point id."""
    if not chunk_ids:
        return {}
    records, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="chunk_id", match=models.MatchAny(any=chunk_ids))]
        ),
        limit=len(chunk_ids),
        with_payload=["chunk_id"],
        with_vectors=True,
    )
    return {
        str(record.payload["chunk_id"]): cast(list[float], record.vector)
        for record in records
        if record.payload and record.vector is not None
    }
