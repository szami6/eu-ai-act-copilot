"""RAG subgraph types (PLAN.md §3.2, §3.4): `RagQuery -> RagResult`, `RagState`.

`Evidence` is shared with the orchestrator's `AgentState.evidence` (§3.4) — it
is defined here, in the subgraph that produces it, and the orchestrator
(Phase 3) will import it from this module rather than the reverse, since only
`RagResult` is meant to cross the subgraph boundary.

`RagQuery`/`RagResult`/`Evidence` are plain dataclasses (no external
validation boundary to justify Pydantic here, matching `ingest.chunk.Chunk`).
`RagState` is a `TypedDict` because it is threaded through a LangGraph
`StateGraph`, which expects a mapping-shaped schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict


@dataclass
class Evidence:
    """One cited chunk, carrying enough provenance to render a citation and
    resolve it back to source (PLAN.md §3.4's "node -> tool -> chunk ->
    article" chain) without re-fetching from Qdrant.

    Field names mirror `ingest.chunk.Chunk` deliberately, since building
    `Evidence` from a retrieved `Chunk` (or its Qdrant payload) is meant to be
    a straightforward field-by-field copy.
    """

    chunk_id: str
    doc_id: str
    chunk_type: str  # "article" | "recital" | "annex" | "definition"
    text: str  # `compress`'s trimmed, query-bearing text
    title: str | None
    chapter: str | None
    paragraph_no: str | None
    article_no: int | None
    cross_refs: list[str]
    source_url: str
    score: float
    # Distinguishes a directly-retrieved chunk from one `compress` pulled in
    # off it (a referenced Art. 3 definition, or a one-hop cross-ref
    # neighbour, §5.2.6) — both end up in the same evidence list.
    origin: Literal["retrieved", "definition", "cross_ref"] = "retrieved"
    terms_used: list[str] = field(default_factory=list)


@dataclass
class RagQuery:
    """The subgraph's external input — Tool 1's `rag_search(query, k)`."""

    query: str
    k: int = 5
    # Optional Qdrant payload filter (by article/chapter/doc_id, §4.5) —
    # left for `retrievers.py` to interpret; not every caller needs one.
    # str | int | bool matches what Qdrant's `MatchValue` accepts.
    filters: dict[str, str | int | bool] | None = None


@dataclass
class RagResult:
    """The subgraph's external output — the only thing that crosses back to
    the orchestrator's `AgentState` (§3.4's "subgraph isolation")."""

    evidence: list[Evidence]
    sufficient: bool
    query_variants: list[str]
    retries: int


class RagState(TypedDict):
    """Internal state threaded through rewrite -> retrieve -> rerank ->
    grade -> compress. Never leaves the subgraph — `graph.py` seeds it from
    a `RagQuery` and reads a `RagResult` back out of the final state.

    `candidates` is replaced wholesale by each node (rerank fully reorders
    retrieve's output; a retry re-retrieves rather than accumulating stale
    hits alongside new ones) — unlike `AgentState.evidence`, nothing here
    needs an append-only reducer.
    """

    query: str
    k: int
    filters: dict[str, str | int | bool] | None

    query_variants: list[str]
    candidates: list[Evidence]
    sufficient: bool
    retry_count: int
    retry_pending: bool

    evidence: list[Evidence]
