"""Pure, hand-computed unit tests for the retrieval-eval metrics (PLAN.md
§7.2). No live infra: `ranked_chunk_ids`/`build_context`/`main` (the parts
that touch Qdrant, the cross-encoder, and the filesystem) are exercised only
by the real `make eval` run, not here.
"""

from __future__ import annotations

import math
from pathlib import Path

from evals.run_eval import (
    GoldQuery,
    load_gold,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_query,
)


def test_recall_at_k_counts_gold_hits_within_k() -> None:
    ranked = ["a", "b", "c", "d", "e"]
    assert recall_at_k(ranked, ["c", "x"], k=5) == 0.5


def test_recall_at_k_truncates_to_k() -> None:
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, ["c"], k=2) == 0.0


def test_recall_at_k_empty_gold_is_zero() -> None:
    assert recall_at_k(["a", "b"], [], k=5) == 0.0


def test_precision_at_k_divides_by_k_not_gold_size() -> None:
    ranked = ["a", "b", "c", "d", "e"]
    assert precision_at_k(ranked, ["a", "c"], k=5) == 0.4


def test_precision_at_k_zero_k_is_zero() -> None:
    assert precision_at_k(["a"], ["a"], k=0) == 0.0


def test_reciprocal_rank_uses_first_hit() -> None:
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == 1.0 / 3
    assert reciprocal_rank(["a", "b", "c"], ["a", "c"]) == 1.0


def test_reciprocal_rank_no_hit_is_zero() -> None:
    assert reciprocal_rank(["a", "b", "c"], ["z"]) == 0.0


def test_ndcg_at_k_perfect_ranking_is_one() -> None:
    assert ndcg_at_k(["a", "b"], ["a", "b"], k=2) == 1.0


def test_ndcg_at_k_no_hits_is_zero() -> None:
    assert ndcg_at_k(["a", "b", "c"], ["z"], k=3) == 0.0


def test_ndcg_at_k_empty_gold_is_zero() -> None:
    assert ndcg_at_k(["a", "b"], [], k=2) == 0.0


def test_ndcg_at_k_truncates_to_k() -> None:
    # Gold hit sits at rank 3, but k=2 only looks at the first two positions,
    # so the ideal (rank-1 hit) still scores relative to a miss actually seen.
    ranked = ["a", "b", "c"]
    idcg = 1.0 / math.log2(2)  # one gold id, best case: hit at rank 1
    assert ndcg_at_k(ranked, ["c"], k=2) == 0.0 / idcg


def test_ndcg_at_k_rewards_higher_rank_more() -> None:
    ranked = ["a", "b", "c", "d"]
    gold = ["b", "d"]
    dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)  # hits at ranks 2 and 4
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)  # ideal: hits at ranks 1 and 2
    assert ndcg_at_k(ranked, gold, k=4) == dcg / idcg


def test_score_query_combines_all_metrics() -> None:
    gold = GoldQuery(query="q", category="cat", gold_chunk_ids=("b",))
    score = score_query(gold, ["a", "b", "c"])

    assert score.gold is gold
    assert score.ranked_ids == ("a", "b", "c")
    assert score.recall_at_5 == 1.0
    assert score.recall_at_10 == 1.0
    assert score.precision_at_5 == 1.0 / 5
    assert score.reciprocal_rank == 1.0 / 2
    assert score.ndcg_at_10 == 1.0 / math.log2(3) / (1.0 / math.log2(2))


def test_load_gold_parses_query_category_and_chunk_ids(tmp_path: Path) -> None:
    gold_path = tmp_path / "retrieval_gold.yaml"
    gold_path.write_text(
        "- query: What is a high-risk AI system?\n"
        "  category: classification\n"
        "  gold_chunk_ids: [ai_act:art:6:1]\n"
        "- query: second query\n"
        "  category: definition\n"
        "  gold_chunk_ids: [ai_act:def:1, ai_act:def:2]\n",
        encoding="utf-8",
    )

    gold = load_gold(gold_path)

    assert len(gold) == 2
    assert gold[0] == GoldQuery(
        query="What is a high-risk AI system?",
        category="classification",
        gold_chunk_ids=("ai_act:art:6:1",),
    )
    assert gold[1].gold_chunk_ids == ("ai_act:def:1", "ai_act:def:2")
