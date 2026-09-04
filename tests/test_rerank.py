"""Unit tests for cross-encoder reranking + MMR fallback (PLAN.md §3.2, §4.5).

`cross_encoder_rerank` takes an injectable fake model so these never
download the real ~1 GB `bge-reranker-base` ONNX model; `mmr_rerank` is pure
vector math and needs no model at all — both mirror `ingest.chunk`'s
fixture-based, network-free testing.
"""

from __future__ import annotations

from copilot.rag.rerank import cross_encoder_rerank, mmr_rerank
from copilot.rag.state import Evidence


class _FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return self._scores


def _evidence(chunk_id: str, text: str = "text", score: float = 0.0) -> Evidence:
    return Evidence(
        chunk_id=chunk_id,
        doc_id="ai_act:art:6",
        chunk_type="article",
        text=text,
        title=None,
        chapter=None,
        paragraph_no=None,
        article_no=6,
        cross_refs=[],
        source_url="https://example.invalid",
        score=score,
    )


def test_cross_encoder_rerank_sorts_by_score_descending() -> None:
    candidates = [_evidence("a"), _evidence("b"), _evidence("c")]
    model = _FakeCrossEncoder(scores=[0.1, 0.9, 0.5])

    reranked = cross_encoder_rerank("query", candidates, top_n=5, model=model)

    assert [e.chunk_id for e in reranked] == ["b", "c", "a"]
    assert [e.score for e in reranked] == [0.9, 0.5, 0.1]


def test_cross_encoder_rerank_truncates_to_top_n() -> None:
    candidates = [_evidence(str(i)) for i in range(10)]
    model = _FakeCrossEncoder(scores=list(range(10)))

    reranked = cross_encoder_rerank("query", candidates, top_n=3, model=model)

    assert len(reranked) == 3


def test_cross_encoder_rerank_on_empty_candidates_returns_empty() -> None:
    assert cross_encoder_rerank("query", [], model=_FakeCrossEncoder(scores=[])) == []


def test_cross_encoder_rerank_preserves_non_score_fields() -> None:
    candidates = [_evidence("a", text="high-risk AI systems")]
    model = _FakeCrossEncoder(scores=[0.42])

    reranked = cross_encoder_rerank("query", candidates, model=model)

    assert reranked[0].chunk_id == "a"
    assert reranked[0].text == "high-risk AI systems"
    assert reranked[0].score == 0.42


def test_mmr_rerank_picks_most_relevant_candidate_first() -> None:
    query_vector = [1.0, 0.0]
    candidates = [_evidence("a"), _evidence("b"), _evidence("c")]
    vectors = {"a": [0.8, 0.6], "b": [0.78, 0.63], "c": [0.6, -0.8]}

    selected = mmr_rerank(query_vector, candidates, vectors, top_n=1)

    assert [e.chunk_id for e in selected] == ["a"]


def test_mmr_rerank_prefers_diverse_pick_over_near_duplicate() -> None:
    # "b" is a near-duplicate of "a" (the first pick) with slightly lower
    # raw relevance than "a" but still higher than "c"; "c" is orthogonal to
    # "a" (fully diverse) despite lower raw relevance. A relevance-only
    # rerank would take "b" second; MMR should prefer "c" instead.
    query_vector = [1.0, 0.0]
    candidates = [_evidence("a"), _evidence("b"), _evidence("c")]
    vectors = {"a": [0.8, 0.6], "b": [0.78, 0.63], "c": [0.6, -0.8]}

    selected = mmr_rerank(query_vector, candidates, vectors, top_n=2, lambda_mult=0.5)

    assert [e.chunk_id for e in selected] == ["a", "c"]


def test_mmr_rerank_skips_candidates_missing_vectors() -> None:
    query_vector = [1.0, 0.0]
    candidates = [_evidence("a"), _evidence("no_vector")]
    vectors = {"a": [1.0, 0.0]}

    selected = mmr_rerank(query_vector, candidates, vectors, top_n=5)

    assert [e.chunk_id for e in selected] == ["a"]


def test_mmr_rerank_respects_top_n() -> None:
    query_vector = [1.0, 0.0]
    candidates = [_evidence(str(i)) for i in range(5)]
    vectors = {str(i): [1.0 - i * 0.1, i * 0.1] for i in range(5)}

    selected = mmr_rerank(query_vector, candidates, vectors, top_n=2)

    assert len(selected) == 2
