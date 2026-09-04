"""HTML -> structured tree (PLAN.md §5.2 step 1).

Parses the two mirror sources' per-page DOM into a small, source-agnostic
model (`ParsedDoc` / `RawUnit`) that `chunk.py` turns into retrievable
chunks. Two DOM shapes are handled:

- AI Act article/annex pages: a `<section>` of repeated
  `div.para.depth-{0,1}` blocks (label + text), with definition tooltips
  (`span.term-ref > span.term-tip`) inlined by the site's JS-glossary and
  stripped back out here.
- AI Act recital pages: a single `div.recital` text block, no sub-points.
- GDPR pages: a plain nested `<ol><li>` — no labels in the markup, so this
  assigns "1.", "2.", ... / "(a)", "(b)", ... by list position.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup
from bs4.element import Tag

from copilot.ingest.sources import DocKind, SourceDoc, eurlex_citation_url, to_roman_annex

_GDPR_CHAPTERS: dict[int, str] = {
    5: "Chapter II: Principles",
    6: "Chapter II: Principles",
    9: "Chapter II: Principles",
    22: "Chapter III: Rights of the data subject",
    35: "Chapter IV: Controller and processor",
}


@dataclass
class DefinitionHit:
    term: str
    body: str
    source: str


@dataclass
class RawUnit:
    """The smallest chunkable unit: one numbered paragraph or lettered point."""

    unit_id: str
    label: str
    depth: int
    text: str
    is_continuation: bool
    cross_ref_recitals: list[str] = field(default_factory=list)
    terms_used: list[str] = field(default_factory=list)


@dataclass
class ParsedDoc:
    doc_id: str
    kind: DocKind
    number: int
    title: str
    chapter: str | None
    section: str | None
    source_url: str
    units: list[RawUnit]
    definitions: list[DefinitionHit]


def _strip_term_tips(text_span: Tag) -> tuple[str, list[str], list[DefinitionHit]]:
    """Remove nested `.term-tip` glossary bodies from a `.para-text` span.

    Without this, `get_text()` on a span containing a `term-ref` yields the
    visible term *and* its hidden tooltip definition concatenated inline
    (e.g. "the AI system AI system means a machine-based system...").
    """
    span = copy.copy(text_span)
    terms: list[str] = []
    definitions: list[DefinitionHit] = []
    # Nested term-refs occur only inside another term's own tip-body (a
    # definition mentioning a second defined term) — walking those too would
    # both double-count and, since decomposing an outer tip while a later
    # loop iteration still holds a reference to a now-detached nested node,
    # corrupt that nested node's text. Top-level only.
    top_level_refs = [
        r for r in span.select("span.term-ref") if r.find_parent("span", class_="term-tip") is None
    ]
    for term_ref in top_level_refs:
        tip = term_ref.select_one("span.term-tip")
        key_el = tip.select_one("span.term-tip-key") if tip else None
        body_el = tip.select_one("span.term-tip-body") if tip else None
        src_el = tip.select_one("span.term-tip-src") if tip else None
        term = key_el.get_text(strip=True) if key_el else term_ref.get_text(strip=True)
        terms.append(term)
        if tip is not None:
            if body_el is not None and src_el is not None:
                definitions.append(
                    DefinitionHit(
                        term=term,
                        body=body_el.get_text(" ", strip=True),
                        source=src_el.get_text(strip=True),
                    )
                )
            tip.decompose()
    text = span.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:)])", r"\1", text).strip()  # e.g. "AI system ," -> "AI system,"
    return text, _dedupe(terms), definitions


def _dedupe(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        if item:
            seen[item] = None
    return list(seen)


def _related_recitals(el: Tag) -> list[str]:
    text = el.get_text(" ", strip=True)
    nums = re.findall(r"\d+", text.split(":", 1)[-1])
    return [f"ai_act:recital:{n}" for n in nums]


def parse_ai_act_article_or_annex(html: str, doc: SourceDoc) -> ParsedDoc:
    soup = BeautifulSoup(html, "lxml")
    content = soup.select_one("div.content-constrain")
    if content is None:
        raise ValueError(f"no content-constrain block found for {doc.doc_id}")
    section = content.select_one("section")
    if section is None:
        raise ValueError(f"no <section> found for {doc.doc_id}")

    heading = section.select_one("h2.item-heading")
    heading_text = heading.get_text(" ", strip=True) if heading else ""
    title = heading_text.split(":", 1)[1].strip() if ":" in heading_text else heading_text

    chapter_el = content.select_one("div.chapter-block")
    section_el = content.select_one("div.section-block")
    chapter = chapter_el.get_text(" ", strip=True) if chapter_el else None
    section_label = section_el.get_text(" ", strip=True) if section_el else None

    units: list[RawUnit] = []
    definitions: list[DefinitionHit] = []
    pending_cross_refs: list[str] = []
    # Hierarchical ids ("6:1", "6:1.a", "6:1:cont") instead of a flat "nth
    # depth-N span in the article" counter, which would number every
    # article's (a)/(b) points 1, 2, 3, ... regardless of which paragraph
    # they belong to and collide with unrelated paragraph numbers. Anchors
    # track "the id a continuation should attach beneath" per depth, since
    # a depth-1 continuation (e.g. Annex III(1)(a)'s follow-on sentence)
    # belongs to the point, not the enclosing numbered paragraph.
    anchor: dict[int, str] = {0: "intro", 1: "intro", 2: "intro"}
    seq_counts: dict[str, int] = {}
    # Some annexes (VIII, and similarly VII/X/XI) restart "1, 2, 3..." under
    # each of several sibling unlabelled section headers ("Section A —",
    # "Section B —", ...). Every unlabelled depth-0 block is therefore
    # treated as opening a new section scope that subsequent *numbered*
    # depth-0 paragraphs are namespaced under, until the next one resets it.
    # Plain articles never hit an unlabelled depth-0 block, so this prefix
    # stays "" for them and their ids are unaffected.
    section_prefix = ""
    # A handful of paragraphs (e.g. Art. 43(1)) restart "(a), (b), ..." a
    # second time after an intervening unlabelled continuation — the second
    # group is a fresh list introduced by that continuation, not a
    # continuation of the first group's own points. depth1_seen detects the
    # relabel and, from that point on, nests points under the continuation's
    # own id instead of the paragraph's, until the next numbered paragraph.
    depth0_container = "intro"
    depth1_seen: set[str] = set()
    last_continuation_id: str | None = None

    def next_in_sequence(scope: str) -> str:
        """1st call for a scope returns it unsuffixed, later calls get 2, 3, ...

        Some annexes (VII, VIII, X, XI) have several distinct unlabelled
        lead-in blocks and unlabelled points, not just one — so "intro" and
        the depth-1 "x" fallback both need their own per-scope counters,
        the same way repeated unlabelled continuations already do.
        """
        seq_counts[scope] = seq_counts.get(scope, 0) + 1
        n = seq_counts[scope]
        return scope if n == 1 else f"{scope}{n}"

    for child in section.find_all("div", recursive=False):
        classes = child.get_attribute_list("class")
        if "related-refs" in classes:
            pending_cross_refs = _related_recitals(child)
            if units and "related-refs-para" in classes:
                units[-1].cross_ref_recitals = _dedupe(
                    units[-1].cross_ref_recitals + pending_cross_refs
                )
                pending_cross_refs = []
            continue
        if "para" not in classes:
            continue  # para-cif ("comes into force" badge) and similar UI chrome

        label_el = child.select_one("span.para-label")
        text_spans = [
            s
            for s in child.select("span.para-text")
            if not s.has_attr("hidden") and "om-old" not in (s.get("class") or [])
        ]
        if not text_spans:
            continue
        text, terms, defs = _strip_term_tips(text_spans[0])
        if not text:
            continue
        definitions.extend(defs)

        depth = 2 if "depth-2" in classes else 1 if "depth-1" in classes else 0
        is_continuation = "continuation" in classes
        label = label_el.get_text(strip=True) if label_el else ""

        if is_continuation:
            cont_scope = next_in_sequence(f"{anchor[depth]}:cont")
            unit_id = f"{doc.doc_id}:{cont_scope}"
            if depth == 0:
                last_continuation_id = cont_scope
        elif depth == 0 and not label:
            header_id = next_in_sequence("intro")
            # Only the 2nd+ unlabelled header needs a disambiguating prefix on
            # its numbering (true multi-section annexes, e.g. VII/VIII/X/XI,
            # restart "1, 2, 3..." under each of several sibling headers). A
            # single lead-in sentence (e.g. Annex III) is the common case and
            # its numbers are already unique without one.
            if header_id != "intro":
                section_prefix = f"{header_id}."
            anchor[0] = anchor[1] = header_id
            depth0_container = header_id
            depth1_seen = set()
            last_continuation_id = None
            unit_id = f"{doc.doc_id}:{header_id}"
        elif depth == 0:
            anchor[0] = f"{section_prefix}{label.rstrip('.')}"
            anchor[1] = anchor[0]  # a fresh depth-0 paragraph resets its points' anchor too
            depth0_container = anchor[0]
            depth1_seen = set()
            last_continuation_id = None
            unit_id = f"{doc.doc_id}:{anchor[0]}"
        elif depth == 1:
            letter = label.strip("()") or next_in_sequence(f"{anchor[0]}.x")
            if letter in depth1_seen:
                # A second, unrelated (a)/(b)/... list started after a
                # continuation (e.g. Art. 43(1)) — nest it under the
                # continuation's id instead of colliding with the first list.
                depth0_container = last_continuation_id or next_in_sequence(f"{anchor[0]}:grp")
                depth1_seen = set()
            depth1_seen.add(letter)
            anchor[1] = f"{depth0_container}.{letter}"
            unit_id = f"{doc.doc_id}:{anchor[1]}"
        else:  # depth == 2, nests under the enclosing depth-1 point, not the paragraph
            letter = label.strip("()") or next_in_sequence(f"{anchor[1]}.x")
            anchor[2] = f"{anchor[1]}.{letter}"
            unit_id = f"{doc.doc_id}:{anchor[2]}"

        units.append(
            RawUnit(
                unit_id=unit_id,
                label=label,
                depth=depth,
                text=text,
                is_continuation=is_continuation,
                cross_ref_recitals=pending_cross_refs,
                terms_used=terms,
            )
        )
        pending_cross_refs = []

    return ParsedDoc(
        doc_id=doc.doc_id,
        kind=doc.kind,
        number=doc.number,
        title=title,
        chapter=chapter,
        section=section_label,
        source_url=eurlex_citation_url(doc.kind),
        units=units,
        definitions=definitions,
    )


def parse_ai_act_recital(html: str, doc: SourceDoc) -> ParsedDoc:
    soup = BeautifulSoup(html, "lxml")
    block = soup.select_one("div.recital-block")
    if block is None:
        raise ValueError(f"no recital-block found for {doc.doc_id}")

    related = block.select_one("div.related-refs")
    body = block.select_one("div.recital")
    text, terms, definitions = _strip_term_tips(body) if body is not None else ("", [], [])
    text = re.sub(rf"^\(?{doc.number}\)?\s*", "", text)

    cross_refs: list[str] = []
    if related is not None:
        nums = re.findall(r"Article\s+(\d+)", related.get_text(" ", strip=True))
        cross_refs = _dedupe([f"ai_act:art:{n}" for n in nums])

    unit = RawUnit(
        unit_id=f"{doc.doc_id}",
        label="",
        depth=0,
        text=text,
        is_continuation=False,
        cross_ref_recitals=cross_refs,
        terms_used=terms,
    )
    return ParsedDoc(
        doc_id=doc.doc_id,
        kind=doc.kind,
        number=doc.number,
        title=f"Recital {doc.number}",
        chapter=None,
        section=None,
        source_url=eurlex_citation_url(doc.kind),
        units=[unit],
        definitions=definitions,
    )


_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def parse_gdpr_article(html: str, doc: SourceDoc) -> ParsedDoc:
    soup = BeautifulSoup(html, "lxml")
    content = soup.select_one("div.entry-content")
    if content is None:
        raise ValueError(f"no entry-content found for {doc.doc_id}")
    top_ol = content.select_one("ol")

    units: list[RawUnit] = []

    def emit(li: Tag, depth: int, seq: int, parent_seq: int | None) -> None:
        nested = li.find("ol")
        own_text_el = copy.copy(li)
        if nested is not None:
            match = own_text_el.find("ol")
            if match is not None:
                match.decompose()
        text = re.sub(r"\s+", " ", own_text_el.get_text(" ", strip=True)).strip()
        label = f"{seq}." if depth == 0 else f"({_LETTERS[seq - 1]})"
        unit_id = (
            f"{doc.doc_id}:{seq}"
            if depth == 0
            else f"{doc.doc_id}:{parent_seq}.{_LETTERS[seq - 1]}"
        )
        ref_nums = [n for n in re.findall(r"Article\s+(\d+)", text) if int(n) != doc.number]
        cross_refs = _dedupe([f"gdpr:art:{n}" for n in ref_nums])
        units.append(
            RawUnit(
                unit_id=unit_id,
                label=label,
                depth=depth,
                text=text,
                is_continuation=False,
                cross_ref_recitals=cross_refs,
            )
        )
        if nested is not None:
            for i, sub_li in enumerate(nested.find_all("li", recursive=False), start=1):
                emit(sub_li, depth + 1, i, seq)

    if top_ol is not None:
        for i, li in enumerate(top_ol.find_all("li", recursive=False), start=1):
            emit(li, 0, i, None)

    return ParsedDoc(
        doc_id=doc.doc_id,
        kind=doc.kind,
        number=doc.number,
        title=doc.title or f"Article {doc.number}",
        chapter=_GDPR_CHAPTERS.get(doc.number),
        section=None,
        source_url=eurlex_citation_url(doc.kind),
        units=units,
        definitions=[],
    )


def parse_source_doc(html: str, doc: SourceDoc) -> ParsedDoc:
    if doc.kind in (DocKind.AI_ACT_ARTICLE, DocKind.AI_ACT_ANNEX):
        return parse_ai_act_article_or_annex(html, doc)
    if doc.kind == DocKind.AI_ACT_RECITAL:
        return parse_ai_act_recital(html, doc)
    if doc.kind == DocKind.GDPR_ARTICLE:
        return parse_gdpr_article(html, doc)
    raise ValueError(f"unknown doc kind: {doc.kind}")


def chunk_id_prefix(doc: ParsedDoc) -> str:
    """The legal-citation-style id prefix used in chunk ids (PLAN.md examples)."""
    if doc.kind == DocKind.AI_ACT_ANNEX:
        return f"ai_act:annex:{to_roman_annex(doc.number)}"
    return doc.doc_id
