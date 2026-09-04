"""Node-level RAG-subgraph eval (PLAN.md §7.2): Recall@5, Recall@10, MRR,
nDCG@10 against `data/eval/retrieval_gold.yaml`, plus the `rerank` ablation
(Δ precision@5 vs. fusion-only). "Isolated harness, no LLM in the loop":
each gold query is fed to `retrieve`/`rerank` directly as its sole
`query_variants` entry, so `rewrite`'s LLM call never runs — otherwise a
paraphrase, good or bad, gets scored as retrieval quality. That's a real,
observed confound, not a hypothetical one: `rewrite`'s dummy-LLM fixture is
a fixed string, so a manual full-graph smoke test showed it injecting the
same two off-topic variants into every query regardless of content.

Phase 2/3 add sibling `eval_*` functions here for triage / classify_risk_tier
/ end-to-end judged eval (this file's own PLAN.md file-tree entry: "node-level
+ end-to-end + ablations") — only retrieval exists yet.

Run via `make eval` (== `uv run python -m evals.run_eval`) from the host,
with `docker compose up qdrant` (or the full stack) already running — Qdrant
must be reachable at `settings.qdrant_url` (default: the compose-published
host port, see `config.py`).
"""

from __future__ import annotations

import math
import statistics
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from io import TextIOWrapper
from pathlib import Path
from typing import cast

import yaml
from langgraph.runtime import Runtime
from qdrant_client import QdrantClient

from copilot.config import Settings, get_settings
from copilot.ingest.index import COLLECTION_NAME
from copilot.llm import DummyChatModel
from copilot.rag.graph import RagContext, rerank, retrieve
from copilot.rag.retrievers import load_bm25_index
from copilot.rag.state import RagState

REPORTS_DIR = Path("evals/reports")
# One ranked list per query serves both Recall@5 (sliced) and Recall@10/MRR/nDCG@10.
_TOP_K = 10


@dataclass(frozen=True)
class GoldQuery:
    query: str
    category: str
    gold_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class QueryScore:
    gold: GoldQuery
    ranked_ids: tuple[str, ...]
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    reciprocal_rank: float
    ndcg_at_10: float


def load_gold(path: Path) -> list[GoldQuery]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        GoldQuery(
            query=e["query"],
            category=e["category"],
            gold_chunk_ids=tuple(e["gold_chunk_ids"]),
        )
        for e in raw
    ]


def recall_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    hits = len(set(ranked_ids[:k]) & set(gold_ids))
    return hits / len(gold_ids)


def precision_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = len(set(ranked_ids[:k]) & set(gold_ids))
    return hits / k


def reciprocal_rank(ranked_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    gold_set = set(gold_ids)
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    gold_set = set(gold_ids)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranked_ids[:k], start=1)
        if chunk_id in gold_set
    )
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def score_query(gold: GoldQuery, ranked_ids: list[str]) -> QueryScore:
    return QueryScore(
        gold=gold,
        ranked_ids=tuple(ranked_ids),
        recall_at_5=recall_at_k(ranked_ids, gold.gold_chunk_ids, 5),
        recall_at_10=recall_at_k(ranked_ids, gold.gold_chunk_ids, 10),
        precision_at_5=precision_at_k(ranked_ids, gold.gold_chunk_ids, 5),
        reciprocal_rank=reciprocal_rank(ranked_ids, gold.gold_chunk_ids),
        ndcg_at_10=ndcg_at_k(ranked_ids, gold.gold_chunk_ids, 10),
    )


def ranked_chunk_ids(context: RagContext, query: str, k: int = _TOP_K) -> list[str]:
    """Run `retrieve` -> `rerank` directly, seeding `query_variants` with only
    the raw gold query — `rewrite` (the LLM node) never runs."""
    runtime: Runtime[RagContext] = Runtime(context=context)
    state: RagState = RagState(
        query=query,
        k=k,
        filters=None,
        query_variants=[query],
        candidates=[],
        sufficient=False,
        retry_count=0,
        evidence=[],
    )
    state.update(cast("RagState", retrieve(state, runtime)))
    state.update(cast("RagState", rerank(state, runtime)))
    return [c.chunk_id for c in state["candidates"]]


def eval_retrieval(context: RagContext, gold: list[GoldQuery]) -> list[QueryScore]:
    return [score_query(g, ranked_chunk_ids(context, g.query)) for g in gold]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _aggregate(scores: list[QueryScore]) -> dict[str, float]:
    return {
        "recall@5": _mean([s.recall_at_5 for s in scores]),
        "recall@10": _mean([s.recall_at_10 for s in scores]),
        "precision@5": _mean([s.precision_at_5 for s in scores]),
        "mrr": _mean([s.reciprocal_rank for s in scores]),
        "ndcg@10": _mean([s.ndcg_at_10 for s in scores]),
    }


def _by_category(scores: list[QueryScore]) -> dict[str, dict[str, float]]:
    categories = sorted({s.gold.category for s in scores})
    return {cat: _aggregate([s for s in scores if s.gold.category == cat]) for cat in categories}


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [f"| {' | '.join(headers)} |", f"|{'|'.join(['---'] * len(headers))}|"]
    lines += [f"| {' | '.join(row)} |" for row in rows]
    return "\n".join(lines)


def render_report(
    git_sha: str, with_rerank: list[QueryScore], without_rerank: list[QueryScore]
) -> str:
    overall = _aggregate(with_rerank)
    without = _aggregate(without_rerank)
    by_cat = _by_category(with_rerank)
    misses = [s for s in with_rerank if s.recall_at_10 < 1.0]

    lines = [
        f"# Retrieval eval — {git_sha}",
        "",
        f"Generated {datetime.now(UTC).isoformat()} · {len(with_rerank)} gold queries · "
        "`data/eval/retrieval_gold.yaml` · no LLM in the loop (PLAN.md §7.2).",
        "",
        "## Headline (retrieve -> rerank, production default: `rerank_enabled=True`)",
        "",
        _format_table(
            ["Recall@5", "Recall@10", "MRR", "nDCG@10"],
            [
                [
                    f"{overall['recall@5']:.3f}",
                    f"{overall['recall@10']:.3f}",
                    f"{overall['mrr']:.3f}",
                    f"{overall['ndcg@10']:.3f}",
                ]
            ],
        ),
        "",
        "## By category",
        "",
        _format_table(
            ["Category", "n", "Recall@5", "Recall@10", "MRR", "nDCG@10"],
            [
                [
                    cat,
                    str(len([s for s in with_rerank if s.gold.category == cat])),
                    f"{m['recall@5']:.3f}",
                    f"{m['recall@10']:.3f}",
                    f"{m['mrr']:.3f}",
                    f"{m['ndcg@10']:.3f}",
                ]
                for cat, m in by_cat.items()
            ],
        ),
        "",
        "## Rerank ablation (Δ precision@5, cross-encoder vs. MMR fusion-only)",
        "",
        _format_table(
            ["Config", "Precision@5", "Recall@5", "MRR"],
            [
                [
                    "rerank on (cross-encoder)",
                    f"{overall['precision@5']:.3f}",
                    f"{overall['recall@5']:.3f}",
                    f"{overall['mrr']:.3f}",
                ],
                [
                    "rerank off (MMR fusion-only)",
                    f"{without['precision@5']:.3f}",
                    f"{without['recall@5']:.3f}",
                    f"{without['mrr']:.3f}",
                ],
            ],
        ),
        "",
        "## Misses (gold id absent from top-10, rerank on)",
        "",
    ]
    if misses:
        lines.append(
            _format_table(
                ["Category", "Query", "Gold", "Top-5 returned"],
                [
                    [
                        s.gold.category,
                        s.gold.query,
                        ", ".join(s.gold.gold_chunk_ids),
                        ", ".join(s.ranked_ids[:5]) or "(none)",
                    ]
                    for s in misses
                ],
            )
        )
    else:
        lines.append("None — every gold id appeared in the top 10.")
    lines.append("")
    return "\n".join(lines)


def build_context(settings: Settings, client: QdrantClient, *, rerank_enabled: bool) -> RagContext:
    return RagContext(
        qdrant_client=client,
        bm25_index=load_bm25_index(client, COLLECTION_NAME),
        chat_model=DummyChatModel(),  # unused: retrieve/rerank never touch chat_model
        settings=settings.model_copy(update={"rerank_enabled": rerank_enabled}),
    )


def main() -> int:
    # Windows' console/redirect encoding defaults to the active code page
    # (e.g. cp1250), which can't encode the report's "Δ" — crashed a real run
    # here after all 24 queries had already scored correctly. `errors="replace"`
    # is defense in depth for any future non-ASCII gold-query text.
    cast(TextIOWrapper, sys.stdout).reconfigure(encoding="utf-8", errors="replace")

    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url)
    gold = load_gold(settings.data_dir / "eval" / "retrieval_gold.yaml")
    print(f"Loaded {len(gold)} gold queries")

    bm25_index = load_bm25_index(client, COLLECTION_NAME)
    print(f"BM25 index mirrors {len(bm25_index.chunks)} chunks from '{COLLECTION_NAME}'")

    context_on = RagContext(
        qdrant_client=client,
        bm25_index=bm25_index,
        chat_model=DummyChatModel(),
        settings=settings.model_copy(update={"rerank_enabled": True}),
    )
    context_off = RagContext(
        qdrant_client=client,
        bm25_index=bm25_index,
        chat_model=context_on.chat_model,
        settings=settings.model_copy(update={"rerank_enabled": False}),
    )

    print("Scoring with rerank ON (cross-encoder — the slow arm)...")
    with_rerank = []
    for i, g in enumerate(gold, start=1):
        score = score_query(g, ranked_chunk_ids(context_on, g.query))
        with_rerank.append(score)
        print(f"  [{i}/{len(gold)}] recall@5={score.recall_at_5:.2f}  {g.query[:70]!r}")

    print("Scoring with rerank OFF (MMR fusion-only)...")
    without_rerank = []
    for i, g in enumerate(gold, start=1):
        score = score_query(g, ranked_chunk_ids(context_off, g.query))
        without_rerank.append(score)
        print(f"  [{i}/{len(gold)}] recall@5={score.recall_at_5:.2f}  {g.query[:70]!r}")

    git_sha = _git_sha()
    report = render_report(git_sha, with_rerank, without_rerank)

    # Persist before printing: report content is valuable and expensive to
    # regenerate (the rerank arm alone is ~15-25 min), so a console-encoding
    # failure on the print below must never cost the computed results.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"retrieval_eval_{git_sha}.md"
    report_path.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
