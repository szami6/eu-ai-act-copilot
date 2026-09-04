"""Unit tests for structured units -> retrievable chunks (PLAN.md §5.2).

Builds small in-memory `ParsedDoc`/`RawUnit` fixtures directly (chunk.py's
whole point is to be pure and network/parse-free) to exercise the group/
split/merge algorithm and breadcrumb formatting without touching parse.py
or the network.
"""

from __future__ import annotations

from copilot.ingest.chunk import build_corpus_chunks, build_definitions_index, chunk_document
from copilot.ingest.parse import DefinitionHit, ParsedDoc, RawUnit
from copilot.ingest.sources import DocKind


def _unit(
    unit_id: str,
    label: str,
    depth: int = 0,
    text: str = "text",
    continuation: bool = False,
) -> RawUnit:
    return RawUnit(
        unit_id=unit_id, label=label, depth=depth, text=text, is_continuation=continuation
    )


def _doc(
    units: list[RawUnit],
    kind: DocKind = DocKind.AI_ACT_ARTICLE,
    doc_id: str = "ai_act:art:9",
    number: int = 9,
    definitions: list[DefinitionHit] | None = None,
    chapter: str | None = "Chapter I",
    section: str | None = "Section 1",
) -> ParsedDoc:
    return ParsedDoc(
        doc_id=doc_id,
        kind=kind,
        number=number,
        title="Test Title",
        chapter=chapter,
        section=section,
        source_url="https://example.test/celex",
        units=units,
        definitions=definitions or [],
    )


def _filler(n_tokens: int) -> str:
    """~n_tokens worth of text under the `len // 4` estimator (`tokens.py`)."""
    return ("a " * (n_tokens * 2)).strip()


def test_whole_paragraph_under_budget_becomes_one_chunk() -> None:
    doc = _doc(
        [
            _unit("ai_act:art:9:1", "1.", text=_filler(200)),
            _unit("ai_act:art:9:1.a", "(a)", depth=1, text=_filler(100)),
            _unit("ai_act:art:9:1.b", "(b)", depth=1, text=_filler(100)),
        ]
    )
    chunks = chunk_document(doc)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_id == "ai_act:art:9:1"
    assert c.paragraph_no == "1"
    assert c.article_no == 9
    assert c.chunk_type == "article"
    assert c.text.startswith("AI Act › Chapter I › Section 1 › Article 9(1)\n")


def test_oversized_paragraph_splits_at_point_boundaries_without_bleed() -> None:
    # Regression (Bug A): split-off points must key their id/citation off the
    # specific point, not the shared parent paragraph every point repeats.
    doc = _doc(
        [
            _unit("ai_act:art:9:1", "1.", text="Shared lead-in sentence."),
            _unit("ai_act:art:9:1.a", "(a)", depth=1, text=_filler(500) + " MARKER_A"),
            _unit("ai_act:art:9:1.b", "(b)", depth=1, text=_filler(500) + " MARKER_B"),
        ]
    )
    chunks = chunk_document(doc)

    by_id = {c.chunk_id: c for c in chunks}
    assert set(by_id) == {"ai_act:art:9:1.a", "ai_act:art:9:1.b"}  # not both "...:1"

    a, b = by_id["ai_act:art:9:1.a"], by_id["ai_act:art:9:1.b"]
    assert a.paragraph_no == "1(a)"
    assert b.paragraph_no == "1(b)"
    assert a.text.split("\n")[0].endswith("Article 9(1)(a)")  # not the double-parened "9(1(a))"
    assert "Shared lead-in sentence." in a.text
    assert "Shared lead-in sentence." in b.text
    assert "MARKER_A" in a.text and "MARKER_A" not in b.text
    assert "MARKER_B" in b.text and "MARKER_B" not in a.text


def test_short_sibling_paragraphs_merge_into_one_chunk() -> None:
    doc = _doc(
        [
            _unit("ai_act:art:9:1", "1.", text=_filler(25)),
            _unit("ai_act:art:9:2", "2.", text=_filler(25)),
            _unit("ai_act:art:9:3", "3.", text=_filler(25)),
        ]
    )
    chunks = chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].paragraph_no == "1–3"
    assert chunks[0].chunk_id == "ai_act:art:9:1"  # anchored on the first merged paragraph


def test_recital_is_always_exactly_one_chunk_regardless_of_size() -> None:
    doc = _doc(
        [_unit("ai_act:recital:1", "", text=_filler(900))],
        kind=DocKind.AI_ACT_RECITAL,
        doc_id="ai_act:recital:1",
        number=1,
        chapter=None,
        section=None,
    )
    chunks = chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "ai_act:recital:1"
    assert chunks[0].chunk_type == "recital"
    assert chunks[0].text.startswith("AI Act › Recital 1\n")


def test_annex_breadcrumb_uses_roman_numeral_and_point_phrasing() -> None:
    doc = _doc(
        [_unit("ai_act:annex:3:1", "1.", text=_filler(100))],
        kind=DocKind.AI_ACT_ANNEX,
        doc_id="ai_act:annex:3",
        number=3,
        chapter=None,
        section=None,
    )
    chunks = chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "ai_act:annex:III:1"  # roman numeral, not "annex:3:1"
    assert chunks[0].text.startswith("AI Act › Annex III › point 1\n")


def test_gdpr_breadcrumb_uses_gdpr_root() -> None:
    doc = _doc(
        [_unit("gdpr:art:22:1", "1.", text=_filler(100))],
        kind=DocKind.GDPR_ARTICLE,
        doc_id="gdpr:art:22",
        number=22,
        chapter="Chapter III: Rights of the data subject",
        section=None,
    )
    chunks = chunk_document(doc)

    assert chunks[0].text.startswith(
        "GDPR › Chapter III: Rights of the data subject › Article 22(1)\n"
    )
    assert chunks[0].article_no == 22


def test_definitions_index_dedupes_case_insensitive_term_variants() -> None:
    # Regression (Bug C): the same term recurs in different casing depending
    # on where a sentence uses it.
    hits = [
        DefinitionHit(term="fundamental rights", body="body text", source="Recital 1"),
        DefinitionHit(term="Fundamental Rights", body="body text", source="Article 3"),
    ]
    index = build_definitions_index(hits)

    assert len(index) == 1
    assert index["fundamental rights"].term == "fundamental rights"  # first-seen casing wins


def test_definition_chunks_use_lowercased_slug_from_index_key() -> None:
    doc1 = _doc(
        [_unit("ai_act:art:9:1", "1.", text="text")],
        definitions=[DefinitionHit(term="fundamental rights", body="body one", source="Art. 3")],
    )
    doc2 = _doc(
        [_unit("ai_act:art:3:2", "2.", text="text")],
        doc_id="ai_act:art:3",
        number=3,
        definitions=[DefinitionHit(term="Fundamental Rights", body="body two", source="Recital 1")],
    )
    chunks, definitions = build_corpus_chunks([doc1, doc2])

    def_chunks = [c for c in chunks if c.chunk_type == "definition"]
    assert len(def_chunks) == 1
    assert def_chunks[0].chunk_id == "ai_act:def:fundamental-rights"
    assert def_chunks[0].title == "fundamental rights"  # first-seen casing
    assert len(definitions) == 1
