"""Unit tests for hybrid retrieval mechanics (PLAN.md §3.2, §4.5).

Only the pure, network-free surface is covered here — `build_bm25_index` and
`reciprocal_rank_fusion` take already-fetched chunks / already-ranked lists,
mirroring `ingest.chunk`'s fixture-based testing. `dense_search` and
`load_bm25_index` need a live Qdrant and are exercised by the ingest smoke
test instead.
"""

from __future__ import annotations

from copilot.rag.retrievers import IndexedChunk, build_bm25_index, reciprocal_rank_fusion


def _indexed(chunk_id: str, text: str) -> IndexedChunk:
    return IndexedChunk(chunk_id=chunk_id, text=text, payload={"chunk_id": chunk_id, "text": text})


def test_bm25_ranks_exact_term_match_above_unrelated_chunk() -> None:
    index = build_bm25_index(
        [
            _indexed("a", "high-risk AI systems must undergo conformity assessment"),
            _indexed("b", "providers shall establish a quality management system"),
            _indexed("c", "member states shall designate market surveillance authorities"),
        ]
    )

    results = index.search("conformity assessment", k=3)

    assert results[0][0] == "a"


def test_bm25_search_respects_k_limit() -> None:
    index = build_bm25_index([_indexed(str(i), "shared filler text repeated") for i in range(10)])

    results = index.search("filler", k=3)

    assert len(results) == 3


def test_bm25_search_on_empty_index_returns_empty_list() -> None:
    index = build_bm25_index([])

    assert index.search("anything", k=5) == []


def test_bm25_index_by_id_resolves_original_payload() -> None:
    chunk = _indexed("ai_act:art:6:3", "classification rules for high-risk AI systems")
    index = build_bm25_index([chunk])

    assert index.by_id["ai_act:art:6:3"].payload == chunk.payload


def test_bm25_search_is_case_insensitive() -> None:
    index = build_bm25_index([_indexed("a", "Annex III high-risk AI systems")])

    results = index.search("ANNEX iii", k=1)

    assert results[0][0] == "a"


def test_reciprocal_rank_fusion_boosts_item_ranked_highly_in_both_lists() -> None:
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("b", 12.0), ("a", 9.0), ("d", 3.0)]

    fused = reciprocal_rank_fusion([dense, sparse])
    fused_ids = [chunk_id for chunk_id, _ in fused]

    # "a" and "b" both appear near the top of both rankings; "c" and "d"
    # each appear in only one list, so the two doubly-ranked items should
    # out-score either single-list-only item.
    assert set(fused_ids[:2]) == {"a", "b"}
    assert fused_ids[-1] in {"c", "d"}


def test_reciprocal_rank_fusion_uses_rank_position_not_raw_score() -> None:
    # "a" is first by position despite a tiny raw score; "b" is second
    # despite a huge one — RRF must follow rank, not the score field.
    ranking = [("a", 0.001), ("b", 999.0)]

    fused = reciprocal_rank_fusion([ranking])

    assert [chunk_id for chunk_id, _ in fused] == ["a", "b"]


def test_reciprocal_rank_fusion_with_no_overlap_keeps_first_ranking_order() -> None:
    fused = reciprocal_rank_fusion([[("a", 1.0), ("b", 1.0)], [("c", 1.0), ("d", 1.0)]])

    assert {chunk_id for chunk_id, _ in fused} == {"a", "b", "c", "d"}
