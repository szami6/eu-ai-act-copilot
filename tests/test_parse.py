"""Unit tests for HTML -> structured tree parsing (PLAN.md §5.2 step 1).

Small inline HTML fixtures mirror the real DOM shapes documented in
`parse.py`'s module docstring, so these stay fast and network-free while
still exercising the anchor/id logic against realistic markup — including
the id-collision bugs found and fixed against the live corpus.
"""

from __future__ import annotations

from copilot.ingest.parse import (
    DefinitionHit,
    parse_ai_act_article_or_annex,
    parse_ai_act_recital,
    parse_gdpr_article,
)
from copilot.ingest.sources import DocKind, SourceDoc


def _doc(
    doc_id: str = "ai_act:art:9",
    number: int = 9,
    kind: DocKind = DocKind.AI_ACT_ARTICLE,
) -> SourceDoc:
    return SourceDoc(doc_id=doc_id, kind=kind, number=number, fetch_url="https://example.test/x")


def _para(label: str, text: str, depth: int = 0, continuation: bool = False) -> str:
    classes = f"para depth-{depth}" + (" continuation" if continuation else "")
    return (
        f'<div class="{classes}">'
        f'<span class="para-label">{label}</span>'
        f'<span class="para-text">{text}</span>'
        f"</div>"
    )


def _article_html(*paras: str, heading: str = "Article 9: Test Title") -> str:
    body = "".join(paras)
    return f"""
    <div class="content-constrain">
      <div class="chapter-block">Chapter I: Test Chapter</div>
      <div class="section-block">Section 1: Test Section</div>
      <section>
        <h2 class="item-heading">{heading}</h2>
        {body}
      </section>
    </div>
    """


def test_basic_paragraph_and_points_get_hierarchical_ids() -> None:
    html = _article_html(
        _para("1.", "Lead-in text."),
        _para("(a)", "Point a text.", depth=1),
        _para("(b)", "Point b text.", depth=1),
    )
    doc = parse_ai_act_article_or_annex(html, _doc())

    assert [u.unit_id for u in doc.units] == [
        "ai_act:art:9:1",
        "ai_act:art:9:1.a",
        "ai_act:art:9:1.b",
    ]
    assert doc.title == "Test Title"
    assert doc.chapter == "Chapter I: Test Chapter"
    assert doc.section == "Section 1: Test Section"
    assert doc.units[1].depth == 1
    assert doc.units[1].text == "Point a text."


def test_depth2_nests_under_its_depth1_parent_not_the_paragraph() -> None:
    # Regression: Art. 5(1)(c)(i)/(ii) and (1)(h)(i) must anchor under "1.c"
    # and "1.h" respectively, not both collapse onto the paragraph "1".
    html = _article_html(
        _para("1.", "Lead-in."),
        _para("(c)", "Point c text.", depth=1),
        _para("(i)", "Sub-point i.", depth=2),
        _para("(ii)", "Sub-point ii.", depth=2),
        _para("(h)", "Point h text.", depth=1),
        _para("(i)", "Sub-point i under h.", depth=2),
    )
    doc = parse_ai_act_article_or_annex(html, _doc())

    ids = [u.unit_id for u in doc.units]
    assert ids == [
        "ai_act:art:9:1",
        "ai_act:art:9:1.c",
        "ai_act:art:9:1.c.i",
        "ai_act:art:9:1.c.ii",
        "ai_act:art:9:1.h",
        "ai_act:art:9:1.h.i",
    ]
    assert len(set(ids)) == len(ids)  # no collision despite both (c) and (h) having an (i)


def test_restart_after_continuation_reanchors_second_point_list() -> None:
    # Regression: Art. 43(1) has a second (a)/(b) list introduced by an
    # intervening unlabelled continuation — it must not collide with the
    # first list's own (a)/(b) ids.
    html = _article_html(
        _para("1.", "First lead-in."),
        _para("(a)", "First list point a.", depth=1),
        _para("(b)", "First list point b.", depth=1),
        _para("", "An unlabelled continuation introducing a second list.", continuation=True),
        _para("(a)", "Second list point a.", depth=1),
        _para("(b)", "Second list point b.", depth=1),
    )
    doc = parse_ai_act_article_or_annex(html, _doc())

    ids = [u.unit_id for u in doc.units]
    assert ids[:3] == ["ai_act:art:9:1", "ai_act:art:9:1.a", "ai_act:art:9:1.b"]
    assert ids[3] == "ai_act:art:9:1:cont"
    assert ids[4:] == ["ai_act:art:9:1:cont.a", "ai_act:art:9:1:cont.b"]
    assert len(set(ids)) == len(ids)


def test_unlabelled_header_does_not_prefix_a_single_lead_in_sentence() -> None:
    # Annex III shape: one lead-in sentence, then plain numbered points — they
    # must not inherit an unneeded "intro."-style disambiguation prefix.
    html = _article_html(
        _para("", "Unlabelled lead-in sentence."),
        _para("1.", "First point."),
        _para("2.", "Second point."),
        heading="Annex III: Test Title",
    )
    doc = parse_ai_act_article_or_annex(
        html, _doc(doc_id="ai_act:annex:3", number=3, kind=DocKind.AI_ACT_ANNEX)
    )

    assert [u.unit_id for u in doc.units] == [
        "ai_act:annex:3:intro",
        "ai_act:annex:3:1",
        "ai_act:annex:3:2",
    ]


def test_second_unlabelled_header_prefixes_its_own_restarted_numbering() -> None:
    # Annex VIII shape: several sibling "Section A/B" headers, each restarting
    # "1, 2, 3..." — only the 2nd+ needs disambiguation from the 1st, whose
    # numbers should stay clean.
    html = _article_html(
        _para("", "Section A lead-in."),
        _para("1.", "Section A point one."),
        _para("", "Section B lead-in."),
        _para("1.", "Section B point one."),
        heading="Annex VIII: Test Title",
    )
    doc = parse_ai_act_article_or_annex(
        html, _doc(doc_id="ai_act:annex:8", number=8, kind=DocKind.AI_ACT_ANNEX)
    )

    assert [u.unit_id for u in doc.units] == [
        "ai_act:annex:8:intro",
        "ai_act:annex:8:1",
        "ai_act:annex:8:intro2",
        "ai_act:annex:8:intro2.1",
    ]


def test_term_tip_is_stripped_from_text_and_recorded_as_definition() -> None:
    html = _article_html(
        _para(
            "1.",
            'The provider shall ensure that the <span class="term-ref">AI system'
            '<span class="term-tip">'
            '<span class="term-tip-key">AI system</span>'
            '<span class="term-tip-body">a machine-based system that infers from input.</span>'
            '<span class="term-tip-src">Article 3(1)</span>'
            "</span></span> is safe.",
        )
    )
    doc = parse_ai_act_article_or_annex(html, _doc())

    assert doc.units[0].text == "The provider shall ensure that the AI system is safe."
    assert doc.units[0].terms_used == ["AI system"]
    assert doc.definitions == [
        DefinitionHit(
            term="AI system",
            body="a machine-based system that infers from input.",
            source="Article 3(1)",
        )
    ]


def test_related_refs_attach_cross_ref_recitals_forward_and_retroactively() -> None:
    html = _article_html(
        '<div class="related-refs">Related recitals: 1, 2</div>',
        _para("1.", "Forward-attached paragraph."),
        _para("2.", "Retroactively-attached paragraph."),
        '<div class="related-refs related-refs-para">Related recitals: 5</div>',
    )
    doc = parse_ai_act_article_or_annex(html, _doc())

    assert doc.units[0].cross_ref_recitals == ["ai_act:recital:1", "ai_act:recital:2"]
    assert doc.units[1].cross_ref_recitals == ["ai_act:recital:5"]


def test_recital_strips_leading_number_and_term_tips() -> None:
    # Regression: recitals must go through the same tooltip-stripping as
    # articles/annexes — otherwise the hidden tip body gets concatenated into
    # the visible text (e.g. "...a high-risk AI system a high-risk AI system
    # an AI system referred to...").
    html = """
    <div class="recital-block">
      <div class="related-refs">See Article 6 and Article 9.</div>
      <div class="recital">(3) The placing on the market of a
        <span class="term-ref">high-risk AI system
          <span class="term-tip">
            <span class="term-tip-key">high-risk AI system</span>
            <span class="term-tip-body">an AI system referred to in Article 6.</span>
            <span class="term-tip-src">Article 3(2)</span>
          </span>
        </span> requires conformity assessment.
      </div>
    </div>
    """
    doc = parse_ai_act_recital(
        html, _doc(doc_id="ai_act:recital:3", number=3, kind=DocKind.AI_ACT_RECITAL)
    )

    assert len(doc.units) == 1
    unit = doc.units[0]
    assert unit.text == (
        "The placing on the market of a high-risk AI system requires conformity assessment."
    )
    assert unit.cross_ref_recitals == ["ai_act:art:6", "ai_act:art:9"]
    assert doc.definitions == [
        DefinitionHit(
            term="high-risk AI system",
            body="an AI system referred to in Article 6.",
            source="Article 3(2)",
        )
    ]


def test_gdpr_article_builds_nested_ids_and_filters_self_reference() -> None:
    html = """
    <div class="entry-content">
      <ol>
        <li>Top-level text for point one.
          <ol>
            <li>Nested point a text.</li>
            <li>Nested point b text.</li>
          </ol>
        </li>
        <li>Top-level point two, see Article 5 and this Article 22.</li>
      </ol>
    </div>
    """
    doc = parse_gdpr_article(
        html, _doc(doc_id="gdpr:art:22", number=22, kind=DocKind.GDPR_ARTICLE)
    )

    assert [u.unit_id for u in doc.units] == [
        "gdpr:art:22:1",
        "gdpr:art:22:1.a",
        "gdpr:art:22:1.b",
        "gdpr:art:22:2",
    ]
    assert [u.label for u in doc.units] == ["1.", "(a)", "(b)", "2."]
    assert doc.units[0].text == "Top-level text for point one."
    assert doc.units[-1].cross_ref_recitals == ["gdpr:art:5"]  # self-ref to Art. 22 excluded
    assert doc.chapter == "Chapter III: Rights of the data subject"
