"""Unit tests for the RAG subgraph's LLM-only and pure-logic surface
(PLAN.md §3.2): `rewrite`, `grade`, `route_after_grade`, `compress`.

`retrieve` and `rerank`'s cross-encoder path need a live Qdrant/embedding
model and are deliberately excluded here — CI runs with no service
container (`.github/workflows/ci.yml` sets only `LLM_PROVIDER=dummy`), so
that pair is exercised by a manual smoke test instead, matching how
`test_rerank.py` already leaves `get_reranker_model` untested.

Every node takes `(state, runtime)` — `Runtime` is a real dataclass
(confirmed live via `inspect.signature`), so it's constructed directly
rather than faked.
"""

from __future__ import annotations

import re
from typing import cast

import pytest
from langgraph.runtime import Runtime
from qdrant_client import QdrantClient

from copilot import llm as llm_module
from copilot.config import Settings
from copilot.ingest.chunk import definition_chunk_id
from copilot.llm import DummyChatModel
from copilot.rag.graph import RagContext, compress, grade, rewrite, route_after_grade
from copilot.rag.retrievers import Bm25Index, IndexedChunk, build_bm25_index
from copilot.rag.state import Evidence, RagState


def _context(
    *,
    chat_model: DummyChatModel | None = None,
    bm25_index: Bm25Index | None = None,
) -> RagContext:
    return RagContext(
        qdrant_client=cast(QdrantClient, None),  # unused by every node under test
        bm25_index=bm25_index if bm25_index is not None else build_bm25_index([]),
        chat_model=chat_model if chat_model is not None else DummyChatModel(),
        settings=Settings(_env_file=None),
    )


def _state(
    *,
    query: str = "what is a high-risk AI system?",
    k: int = 5,
    query_variants: list[str] | None = None,
    candidates: list[Evidence] | None = None,
    sufficient: bool = False,
    retry_count: int = 0,
    retry_pending: bool = False,
) -> RagState:
    return RagState(
        query=query,
        k=k,
        filters=None,
        query_variants=query_variants or [],
        candidates=candidates or [],
        sufficient=sufficient,
        retry_count=retry_count,
        retry_pending=retry_pending,
        evidence=[],
    )


def _evidence(
    chunk_id: str,
    text: str = "Breadcrumb\nBody text.",
    terms_used: list[str] | None = None,
) -> Evidence:
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
        score=1.0,
        terms_used=terms_used or [],
    )


def _payload(e: Evidence) -> dict[str, object]:
    return {
        "chunk_id": e.chunk_id,
        "doc_id": e.doc_id,
        "chunk_type": e.chunk_type,
        "text": e.text,
        "title": e.title,
        "chapter": e.chapter,
        "paragraph_no": e.paragraph_no,
        "article_no": e.article_no,
        "cross_refs": e.cross_refs,
        "source_url": e.source_url,
        "terms_used": e.terms_used,
    }


def _index_of(*evidence: Evidence) -> Bm25Index:
    return build_bm25_index(
        [IndexedChunk(chunk_id=e.chunk_id, text=e.text, payload=_payload(e)) for e in evidence]
    )


# Every node returns a `dict[str, object]` partial-state update (LangGraph's
# own convention — which keys are present varies per node, so a single
# precise return type isn't expressible). These narrow it back for
# assertions, the same way `rerank.py`/`retrievers.py` `cast` at a
# similarly loosely-typed boundary.
def _candidates(update: dict[str, object]) -> list[Evidence]:
    return cast(list[Evidence], update["candidates"])


def _evidence_list(update: dict[str, object]) -> list[Evidence]:
    return cast(list[Evidence], update["evidence"])


def _variants(update: dict[str, object]) -> list[str]:
    return cast(list[str], update["query_variants"])


# --- rewrite -----------------------------------------------------------------


def test_rewrite_includes_original_query_and_llm_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"Rewrite this"),
                '{"keyword_variant": "kw phrasing", "legal_variant": "legal phrasing", '
                '"hyde_passage": null}',
            )
        ],
    )
    state = _state(query="what is a CV screening tool?")

    update = rewrite(state, Runtime(context=_context()))

    assert update["query_variants"] == [
        "what is a CV screening tool?",
        "kw phrasing",
        "legal phrasing",
    ]


def test_rewrite_appends_hyde_passage_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"Rewrite this"),
                '{"keyword_variant": "kw", "legal_variant": "legal", '
                '"hyde_passage": "A high-risk AI system is a system that..."}',
            )
        ],
    )

    update = rewrite(_state(query="what does high-risk mean?"), Runtime(context=_context()))

    assert _variants(update)[-1] == "A high-risk AI system is a system that..."


def test_rewrite_dedupes_variant_identical_to_original(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"Rewrite this"),
                '{"keyword_variant": "same query", "legal_variant": "legal phrasing", '
                '"hyde_passage": null}',
            )
        ],
    )

    update = rewrite(_state(query="same query"), Runtime(context=_context()))

    assert update["query_variants"] == ["same query", "legal phrasing"]


# --- grade ---------------------------------------------------------------


def test_grade_filters_to_relevant_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"Grade each"),
                '{"grades": [{"chunk_id": "a", "relevant": true}, '
                '{"chunk_id": "b", "relevant": false}], "sufficient": true, '
                '"broadened_query": null}',
            )
        ],
    )
    state = _state(candidates=[_evidence("a"), _evidence("b")])

    update = grade(state, Runtime(context=_context()))

    assert [c.chunk_id for c in _candidates(update)] == ["a"]
    assert update["sufficient"] is True
    assert "retry_count" not in update


def test_grade_empty_verdict_passes_all_candidates_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"Grade each"),
                '{"grades": [], "sufficient": true, "broadened_query": null}',
            )
        ],
    )
    state = _state(candidates=[_evidence("a"), _evidence("b")])

    update = grade(state, Runtime(context=_context()))

    assert [c.chunk_id for c in _candidates(update)] == ["a", "b"]


def test_grade_proposes_retry_when_insufficient_and_budget_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"Grade each"),
                '{"grades": [], "sufficient": false, "broadened_query": "broader query"}',
            )
        ],
    )
    state = _state(query="narrow query", k=5, retry_count=0, candidates=[_evidence("a")])

    update = grade(state, Runtime(context=_context()))

    assert update["sufficient"] is False
    assert update["retry_count"] == 1
    assert update["query"] == "broader query"
    assert update["query_variants"] == ["broader query"]
    assert update["k"] == 10
    assert update["retry_pending"] is True


def test_grade_does_not_propose_retry_once_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"Grade each"),
                '{"grades": [], "sufficient": false, "broadened_query": "broader query"}',
            )
        ],
    )
    state = _state(query="narrow query", k=5, retry_count=1, candidates=[_evidence("a")])

    update = grade(state, Runtime(context=_context()))

    assert update["sufficient"] is False
    assert "retry_count" not in update
    assert "query" not in update
    assert "k" not in update
    assert update["retry_pending"] is False


def test_grade_falls_back_to_original_query_when_no_broadened_query_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"Grade each"),
                '{"grades": [], "sufficient": false, "broadened_query": null}',
            )
        ],
    )
    state = _state(query="original", k=5, retry_count=0, candidates=[_evidence("a")])

    update = grade(state, Runtime(context=_context()))

    assert update["query"] == "original"


# --- route_after_grade -----------------------------------------------------


def test_route_after_grade_to_compress_when_sufficient() -> None:
    state = _state(sufficient=True, retry_count=0)
    assert route_after_grade(state) == "compress"


def test_route_after_grade_to_retrieve_when_insufficient_and_budget_available() -> None:
    state = _state(sufficient=False, retry_count=1, retry_pending=True)
    assert route_after_grade(state) == "retrieve"


def test_route_after_grade_to_compress_when_retry_budget_exhausted() -> None:
    state = _state(sufficient=False, retry_count=1, retry_pending=False)
    assert route_after_grade(state) == "compress"


def test_grade_update_routes_to_one_broadened_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"Grade each"),
                '{"grades": [], "sufficient": false, "broadened_query": "broader query"}',
            )
        ],
    )
    state = _state(query="narrow query", query_variants=["narrow query"])

    state.update(cast("RagState", grade(state, Runtime(context=_context()))))

    assert route_after_grade(state) == "retrieve"
    assert state["query_variants"] == ["broader query"]


# --- compress ----------------------------------------------------------------


def test_compress_keeps_header_and_trims_to_query_overlapping_sentences() -> None:
    text = (
        "AI Act › Article 6\n"
        "This is the lead-in sentence. High-risk AI systems must meet strict "
        "requirements. This unrelated sentence talks about cats and dogs."
    )
    state = _state(query="high-risk requirements", candidates=[_evidence("a", text=text)])

    update = compress(state, Runtime(context=_context()))

    trimmed_text = _evidence_list(update)[0].text
    assert trimmed_text.startswith("AI Act › Article 6\n")
    assert "High-risk AI systems must meet strict requirements." in trimmed_text
    assert "cats and dogs" not in trimmed_text


def test_compress_on_empty_candidates_returns_empty_evidence() -> None:
    update = compress(_state(candidates=[]), Runtime(context=_context()))
    assert update["evidence"] == []


def test_compress_injects_referenced_definition_by_id() -> None:
    term = "high-risk AI system"
    primary = _evidence("ai_act:art:6:1", terms_used=[term])
    definition = _evidence(
        definition_chunk_id(term),
        text=f"AI Act › Definitions › {term}\nMeans a system that...",
        terms_used=[term],
    )
    state = _state(query="what is a high-risk AI system", candidates=[primary])

    update = compress(state, Runtime(context=_context(bm25_index=_index_of(primary, definition))))

    evidence = _evidence_list(update)
    assert [e.chunk_id for e in evidence] == [primary.chunk_id, definition.chunk_id]
    assert evidence[1].origin == "definition"


def test_compress_skips_definition_already_present_in_candidates() -> None:
    term = "high-risk AI system"
    definition = _evidence(definition_chunk_id(term), terms_used=[term])
    primary = _evidence("ai_act:art:6:1", terms_used=[term])
    state = _state(candidates=[primary, definition])

    update = compress(state, Runtime(context=_context(bm25_index=_index_of(primary, definition))))

    assert len(_evidence_list(update)) == 2


def test_compress_skips_definition_with_no_matching_chunk() -> None:
    primary = _evidence("ai_act:art:6:1", terms_used=["a term nobody indexed"])
    state = _state(candidates=[primary])

    update = compress(state, Runtime(context=_context(bm25_index=_index_of(primary))))

    assert [e.chunk_id for e in _evidence_list(update)] == [primary.chunk_id]
