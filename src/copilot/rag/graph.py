"""RAG subgraph assembly (PLAN.md §3.2): rewrite -> retrieve -> rerank -> grade -> compress.

`RagContext` carries the run-scoped dependencies (Qdrant client, in-memory
BM25 mirror, chat model, settings) through LangGraph's `Runtime[T]`
dependency injection — verified live against the installed langgraph
1.2.11 (`StateGraph(..., context_schema=RagContext)`, node signature
`(state, runtime)`, `compiled.invoke(state, context=...)`), which is how
this version does it, not the `configurable` dict older tutorials use. This
keeps `RagState` a plain, checkpointer-safe `TypedDict` with no live objects
in it (PLAN §3.4).

`build_rag_graph()` compiles the graph once at process startup; `run_rag` is
the `RagQuery -> RagResult` boundary Tool 1 calls — the only thing that
crosses into the orchestrator (PLAN §3.2's "subgraph isolation").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from copilot.config import Settings
from copilot.ingest.chunk import definition_chunk_id
from copilot.ingest.index import COLLECTION_NAME
from copilot.rag.embeddings import embed_query
from copilot.rag.rerank import cross_encoder_rerank, fetch_vectors, mmr_rerank
from copilot.rag.retrievers import Bm25Index, dense_search, reciprocal_rank_fusion
from copilot.rag.state import Evidence, RagQuery, RagResult, RagState
from copilot.tokens import estimate_tokens

_MAX_RETRIES = 1
# PLAN.md §3.2 names "top-30" as the reranker's input pool; reused as both
# the per-variant dense/sparse fetch size and the fused pool size taken
# forward, rather than inventing a second unrelated constant for one idea.
_CANDIDATE_POOL_SIZE = 30
# No figure is given in PLAN.md for compress's "token budget" — chosen so a
# trimmed chunk keeps a small handful of sentences, well under the ~600-token
# whole-chunk ceiling `ingest.chunk` allows.
_COMPRESS_TOKEN_BUDGET = 150

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\w+")


@dataclass
class RagContext:
    """Run-scoped dependencies injected via `Runtime[RagContext]` — kept out
    of `RagState` itself, which must stay checkpointer-serializable."""

    qdrant_client: QdrantClient
    bm25_index: Bm25Index
    chat_model: BaseChatModel
    settings: Settings


class RewriteOutput(BaseModel):
    """`rewrite`'s structured output (PLAN §3.2 `rewrite`).

    The node — not the LLM — always keeps the original query as one variant;
    code doesn't depend on the model faithfully echoing it back unparaphrased.
    """

    keyword_variant: str = Field(description="A keyword-dense phrasing of the question")
    legal_variant: str = Field(
        description="The question re-phrased in EU AI Act legal terminology"
    )
    hyde_passage: str | None = Field(
        default=None,
        description=(
            "Only for a definitional question: a short hypothetical passage "
            "that would define the term, for HyDE-style retrieval. Omit otherwise."
        ),
    )


class ChunkGrade(BaseModel):
    chunk_id: str
    relevant: bool


class GradeOutput(BaseModel):
    """`grade`'s structured output (PLAN §3.2 `grade`).

    `sufficient` is the LLM's own holistic verdict, reported as-is. Whether a
    retry actually happens is `route_after_grade`'s deterministic call,
    bounded to `_MAX_RETRIES` regardless of this field — grade proposes,
    routing decides (PLAN's "recoverable, observable event").
    """

    grades: list[ChunkGrade]
    sufficient: bool
    broadened_query: str | None = Field(
        default=None, description="A wider/rephrased query. Only meaningful when sufficient=False."
    )


def _evidence_from_payload(
    chunk_id: str,
    payload: dict[str, object],
    score: float,
    origin: Literal["retrieved", "definition", "cross_ref"] = "retrieved",
) -> Evidence:
    """`Evidence`'s fields mirror `ingest.index`'s payload dict 1:1 (`state.py`
    docstring) — this is the one place that copy happens."""
    return Evidence(
        chunk_id=chunk_id,
        doc_id=str(payload["doc_id"]),
        chunk_type=str(payload["chunk_type"]),
        text=str(payload["text"]),
        title=cast("str | None", payload.get("title")),
        chapter=cast("str | None", payload.get("chapter")),
        paragraph_no=cast("str | None", payload.get("paragraph_no")),
        article_no=cast("int | None", payload.get("article_no")),
        cross_refs=list(cast("list[str]", payload.get("cross_refs") or [])),
        source_url=str(payload["source_url"]),
        score=score,
        origin=origin,
        terms_used=list(cast("list[str]", payload.get("terms_used") or [])),
    )


def rewrite(state: RagState, runtime: Runtime[RagContext]) -> dict[str, object]:
    prompt = (
        "Rewrite this AI Act compliance question for retrieval.\n\n"
        f'Question: "{state["query"]}"\n\n'
        "Give a keyword-dense phrasing and a phrasing in EU AI Act legal "
        "terminology. Only if the question asks what a term means, also give "
        "a short hypothetical passage that would define it."
    )
    structured = runtime.context.chat_model.with_structured_output(RewriteOutput)
    output = structured.invoke([HumanMessage(content=prompt)])
    assert isinstance(output, RewriteOutput)  # include_raw=False guarantees this

    variants = [state["query"], output.keyword_variant, output.legal_variant]
    if output.hyde_passage:
        variants.append(output.hyde_passage)
    deduped = list(dict.fromkeys(v.strip() for v in variants if v.strip()))
    return {"query_variants": deduped}


def _matches_filters(
    payload: dict[str, object], filters: dict[str, str | int | bool] | None
) -> bool:
    if not filters:
        return True
    return all(payload.get(key) == value for key, value in filters.items())


def retrieve(state: RagState, runtime: Runtime[RagContext]) -> dict[str, object]:
    ctx = runtime.context
    variants = state["query_variants"] or [state["query"]]
    fetch_k = max(state["k"], _CANDIDATE_POOL_SIZE)

    rankings: list[list[tuple[str, float]]] = []
    for variant in variants:
        vector = embed_query(variant)
        rankings.append(
            dense_search(ctx.qdrant_client, COLLECTION_NAME, vector, fetch_k, state["filters"])
        )

        sparse_hits = ctx.bm25_index.search(variant, fetch_k)
        rankings.append(
            [
                (chunk_id, score)
                for chunk_id, score in sparse_hits
                if (indexed := ctx.bm25_index.by_id.get(chunk_id)) is not None
                and _matches_filters(indexed.payload, state["filters"])
            ]
        )

    fused = reciprocal_rank_fusion(rankings)[:_CANDIDATE_POOL_SIZE]

    candidates: list[Evidence] = []
    seen: set[str] = set()
    for chunk_id, score in fused:
        indexed = ctx.bm25_index.by_id.get(chunk_id)
        if indexed is None:
            continue
        candidates.append(_evidence_from_payload(chunk_id, indexed.payload, score))
        seen.add(chunk_id)

    # One-hop cross-reference expansion (PLAN §5.2 item 6): each primary
    # candidate's cited recitals/articles join the pool for `rerank` to
    # score alongside everything else — available, not force-included.
    for candidate in list(candidates):
        for ref_id in candidate.cross_refs:
            if ref_id in seen:
                continue
            indexed = ctx.bm25_index.by_id.get(ref_id)
            if indexed is None:
                continue
            candidates.append(
                _evidence_from_payload(ref_id, indexed.payload, 0.0, origin="cross_ref")
            )
            seen.add(ref_id)

    return {"candidates": candidates}


def rerank(state: RagState, runtime: Runtime[RagContext]) -> dict[str, object]:
    ctx = runtime.context
    candidates = state["candidates"]
    if ctx.settings.rerank_enabled:
        reranked = cross_encoder_rerank(state["query"], candidates, top_n=state["k"])
    else:
        query_vector = embed_query(state["query"])
        vectors = fetch_vectors(
            ctx.qdrant_client, COLLECTION_NAME, [c.chunk_id for c in candidates]
        )
        reranked = mmr_rerank(query_vector, candidates, vectors, top_n=state["k"])
    return {"candidates": reranked}


def _format_candidates_for_grading(candidates: list[Evidence]) -> str:
    return "\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in candidates)


def grade(state: RagState, runtime: Runtime[RagContext]) -> dict[str, object]:
    prompt = (
        f'Question: "{state["query"]}"\n\n'
        "Candidate passages:\n\n"
        f"{_format_candidates_for_grading(state['candidates'])}\n\n"
        "Grade each candidate's relevance to the question by its id in "
        "brackets, and judge whether together they are sufficient to answer "
        "it. If not, propose a broadened query."
    )
    structured = runtime.context.chat_model.with_structured_output(GradeOutput)
    output = structured.invoke([HumanMessage(content=prompt)])
    assert isinstance(output, GradeOutput)

    # An empty verdict (the dummy fixture's fixed response can't reference
    # the real candidate ids it was shown; a real model's malformed reply
    # would look the same) passes everything through rather than silently
    # discarding every candidate.
    relevant_ids = {g.chunk_id for g in output.grades if g.relevant}
    graded = (
        [c for c in state["candidates"] if c.chunk_id in relevant_ids]
        if output.grades
        else state["candidates"]
    )

    update: dict[str, object] = {
        "candidates": graded,
        "sufficient": output.sufficient,
        "retry_pending": False,
    }
    if not output.sufficient and state["retry_count"] < _MAX_RETRIES:
        update["retry_count"] = state["retry_count"] + 1
        update["query"] = output.broadened_query or state["query"]
        update["query_variants"] = [str(update["query"])]
        update["k"] = state["k"] * 2
        update["retry_pending"] = True
    return update


def route_after_grade(state: RagState) -> Literal["retrieve", "compress"]:
    return "retrieve" if state["retry_pending"] else "compress"


def _tokenize_words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _trim_to_query_sentences(text: str, query_terms: set[str]) -> str:
    """Keep the breadcrumb header (PLAN's "parent article header") and trim
    the body to query-bearing sentences under `_COMPRESS_TOKEN_BUDGET`.

    The first sentence is always kept regardless of overlap (a lead-in
    sentence often carries no query terms itself but is needed context, and
    this guarantees the body is never trimmed to nothing).
    """
    header, sep, body = text.partition("\n")
    if not sep:
        return text

    sentences = [s for s in _SENTENCE_SPLIT_RE.split(body) if s.strip()]
    if not sentences:
        return text

    selected: list[str] = []
    budget = _COMPRESS_TOKEN_BUDGET
    for sentence in sentences:
        has_overlap = bool(_tokenize_words(sentence) & query_terms)
        if not has_overlap and selected:
            continue
        cost = estimate_tokens(sentence)
        if selected and budget - cost < 0:
            break
        selected.append(sentence)
        budget -= cost

    return f"{header}\n{' '.join(selected)}"


def _inject_definitions(evidence: list[Evidence], bm25_index: Bm25Index) -> list[Evidence]:
    """Pull in each evidence chunk's referenced Art. 3 definitions by direct
    id lookup (PLAN §5.2 item 5) — no retrieval needed, `definition_chunk_id`
    is deterministic from the term alone."""
    present = {e.chunk_id for e in evidence}
    seen_terms: set[str] = set()
    new_defs: list[Evidence] = []
    for e in evidence:
        for term in e.terms_used:
            key = term.strip().lower()
            if key in seen_terms:
                continue
            seen_terms.add(key)
            def_id = definition_chunk_id(term)
            if def_id in present:
                continue
            indexed = bm25_index.by_id.get(def_id)
            if indexed is None:
                continue
            new_defs.append(
                _evidence_from_payload(def_id, indexed.payload, 0.0, origin="definition")
            )
            present.add(def_id)
    new_defs.sort(key=lambda e: e.chunk_id)
    return evidence + new_defs


def compress(state: RagState, runtime: Runtime[RagContext]) -> dict[str, object]:
    query_terms = _tokenize_words(state["query"])
    trimmed = [
        replace(c, text=_trim_to_query_sentences(c.text, query_terms)) for c in state["candidates"]
    ]
    evidence = _inject_definitions(trimmed, runtime.context.bm25_index)
    return {"evidence": evidence}


def _seed_state(query: RagQuery) -> RagState:
    return RagState(
        query=query.query,
        k=query.k,
        filters=query.filters,
        query_variants=[],
        candidates=[],
        sufficient=False,
        retry_count=0,
        retry_pending=False,
        evidence=[],
    )


def _to_result(state: RagState) -> RagResult:
    return RagResult(
        evidence=state["evidence"],
        sufficient=state["sufficient"],
        query_variants=state["query_variants"],
        retries=state["retry_count"],
    )


def build_rag_graph() -> CompiledStateGraph[RagState, RagContext, RagState, RagState]:
    graph = StateGraph(RagState, context_schema=RagContext)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("grade", grade)
    graph.add_node("compress", compress)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "grade")
    graph.add_conditional_edges(
        "grade", route_after_grade, {"retrieve": "retrieve", "compress": "compress"}
    )
    graph.add_edge("compress", END)

    return graph.compile()


def run_rag(
    compiled: CompiledStateGraph[RagState, RagContext, RagState, RagState],
    query: RagQuery,
    context: RagContext,
) -> RagResult:
    """Tool 1's `rag_search(query, k)` boundary (PLAN §3.3) — the only point
    where a `RagQuery`/`RagResult` crosses into the orchestrator."""
    final_state = cast("RagState", compiled.invoke(_seed_state(query), context=context))
    return _to_result(final_state)
