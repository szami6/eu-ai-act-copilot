"""Structured units -> retrievable chunks (PLAN.md §5.2 steps 2-6).

Regroups each ParsedDoc's flat RawUnit list back under its owning depth-0
paragraph ("one article-paragraph per chunk"). A group over ~600 tokens is
split back apart at its point boundaries, with the paragraph's own shared
text (its lead-in and any trailing qualifiers) repeated in every split-off
point's chunk so each still reads standalone. A run of very short paragraphs
is folded into one sibling-sized chunk instead of standing alone as a
near-empty one. Every chunk is prefixed with a breadcrumb header before
embedding, and Art. 3's defined terms are collected into a side-index
`compress` can inject by term lookup rather than by retrieval.

Recitals are the one exception to grouping/splitting: each is already a
single short, individually-cited unit — Art. 3 cross-references point at
specific recital ids — so every recital becomes exactly one chunk.

Pure and network-free by design: takes already-`parse_source_doc`-parsed
docs, so it is unit-testable against small in-memory fixtures without
fetching anything or loading an embedding model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from copilot.ingest.parse import DefinitionHit, ParsedDoc, RawUnit, chunk_id_prefix
from copilot.ingest.sources import DocKind, eurlex_citation_url
from copilot.tokens import estimate_tokens

_MAX_CHUNK_TOKENS = 600
# No figure for this is given in PLAN.md §5.2 ("merge very short paragraphs
# with siblings") — chosen so a handful of one-sentence paragraphs (e.g. Art.
# 5(8), most GDPR articles) merge into a sibling-sized chunk instead of each
# standing alone as a near-empty one.
_MIN_CHUNK_TOKENS = 40

_CHUNK_TYPE_BY_KIND: dict[DocKind, str] = {
    DocKind.AI_ACT_ARTICLE: "article",
    DocKind.AI_ACT_RECITAL: "recital",
    DocKind.AI_ACT_ANNEX: "annex",
    DocKind.GDPR_ARTICLE: "article",
}


def definition_chunk_id(term: str) -> str:
    """The deterministic id a defined term's chunk is stored under — shared
    with `rag.compress`, which looks definitions up by this same id rather
    than through retrieval (PLAN.md §5.2.5)."""
    slug = re.sub(r"[^a-z0-9]+", "-", term.strip().lower()).strip("-")
    return f"ai_act:def:{slug}"


@dataclass
class Chunk:
    """One retrievable unit — the payload fields PLAN.md §5.2.4 lists."""

    chunk_id: str
    doc_id: str
    chunk_type: str  # "article" | "recital" | "annex" | "definition"
    text: str  # breadcrumb-prefixed, this is what gets embedded
    title: str | None
    chapter: str | None
    paragraph_no: str | None
    article_no: int | None
    cross_refs: list[str]
    source_url: str
    terms_used: list[str] = field(default_factory=list)


def _render(units: list[RawUnit]) -> str:
    parts = [f"{u.label} {u.text}".strip() if u.label else u.text for u in units]
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _dedupe(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        seen[item] = None
    return list(seen)


def _group_by_paragraph(units: list[RawUnit]) -> list[list[RawUnit]]:
    """One group per depth-0 paragraph/header, with all its points and
    continuations (at any depth) folded in.
    """
    groups: list[list[RawUnit]] = []
    for u in units:
        if u.depth == 0 and not u.is_continuation:
            groups.append([u])
        else:
            if not groups:  # defensive: a doc that opens on a continuation/point
                groups.append([])
            groups[-1].append(u)
    return groups


def _split_at_points(
    group: list[RawUnit],
) -> tuple[list[RawUnit], list[list[RawUnit]], list[RawUnit]]:
    """A paragraph's own shared text (before/after its points) plus each
    point's own subtree, so every split-off chunk can be built as
    `pre + point_units + post` and still read standalone.
    """
    pre: list[RawUnit] = []
    post: list[RawUnit] = []
    points: list[list[RawUnit]] = []
    current: list[RawUnit] | None = None
    for u in group:
        if u.depth == 0:
            (post if current is not None else pre).append(u)
        elif u.depth == 1 and not u.is_continuation:
            current = [u]
            points.append(current)
        elif current is not None:
            current.append(u)
        else:  # a sub-point/continuation before any point opened it — shouldn't
            pre.append(u)  # happen on real data, but keep the content rather than drop it
    return pre, points, post


def _regroup(groups: list[list[RawUnit]]) -> list[tuple[list[RawUnit], RawUnit | None]]:
    """Split groups over `_MAX_CHUNK_TOKENS` at point boundaries; merge runs
    of groups under `_MIN_CHUNK_TOKENS` into one sibling-sized chunk.

    Returns (units, split_point) pairs. `split_point` is the specific point a
    group was split off for — chunk id/citation must key off *it*, not the
    parent paragraph shared by every other point split from the same
    paragraph — or None for a whole or merged-siblings group.
    """
    final: list[tuple[list[RawUnit], RawUnit | None]] = []
    pending: list[RawUnit] = []

    def flush_pending() -> None:
        if not pending:
            return
        if final and estimate_tokens(_render(pending)) < _MIN_CHUNK_TOKENS:
            prev_units, prev_point = final[-1]
            final[-1] = (prev_units + pending, prev_point)  # still too short even combined
        else:
            final.append((list(pending), None))
        pending.clear()

    for group in groups:
        tokens = estimate_tokens(_render(group))
        if tokens > _MAX_CHUNK_TOKENS:
            flush_pending()
            pre, points, post = _split_at_points(group)
            if points:
                final.extend((pre + point_units + post, point_units[0]) for point_units in points)
            else:
                final.append((group, None))  # nothing to split at; keep whole over dropping content
        else:
            pending.extend(group)
            if estimate_tokens(_render(pending)) >= _MIN_CHUNK_TOKENS:
                flush_pending()
    flush_pending()
    return final


def _breadcrumb(doc: ParsedDoc, paragraph_no: str) -> str:
    root = "GDPR" if doc.kind == DocKind.GDPR_ARTICLE else "AI Act"
    parts = [root]
    if doc.chapter:
        parts.append(doc.chapter)
    if doc.section:
        parts.append(doc.section)

    if doc.kind == DocKind.AI_ACT_RECITAL:
        parts.append(f"Recital {doc.number}")
    elif doc.kind == DocKind.AI_ACT_ANNEX:
        roman = chunk_id_prefix(doc).rsplit(":", 1)[-1]
        parts.append(f"Annex {roman}")
        if paragraph_no:
            parts.append(f"point {paragraph_no}")
    else:
        article = f"Article {doc.number}"
        if not paragraph_no:
            parts.append(article)
        else:
            # A split-off point's paragraph_no is "1(a)" (parent+point) — render
            # as the standard citation "Article 5(1)(a)", not "Article 5(1(a))".
            m = re.match(r"^(.+)\((.+)\)$", paragraph_no)
            if m:
                parts.append(f"{article}({m.group(1)})({m.group(2)})")
            else:
                parts.append(f"{article}({paragraph_no})")
    return " › ".join(parts)


def _make_chunk_from_group(
    doc: ParsedDoc, units: list[RawUnit], split_point: RawUnit | None
) -> Chunk:
    depth0_anchors = [u for u in units if u.depth == 0 and not u.is_continuation]

    if split_point is not None and depth0_anchors:
        anchor = split_point
        parent_label = depth0_anchors[0].label.rstrip(".")
        paragraph_no = f"{parent_label}({split_point.label.strip('()')})"
    else:
        anchor = depth0_anchors[0] if depth0_anchors else units[0]
        parent_labels = [u.label.rstrip(".") for u in depth0_anchors if u.label]
        if not parent_labels:
            paragraph_no = ""
        elif len(parent_labels) == 1:
            paragraph_no = parent_labels[0]
        else:
            paragraph_no = f"{parent_labels[0]}–{parent_labels[-1]}"

    prefix = f"{doc.doc_id}:"
    bare_suffix = anchor.unit_id.removeprefix(prefix)
    chunk_id = (
        chunk_id_prefix(doc)
        if bare_suffix == anchor.unit_id
        else f"{chunk_id_prefix(doc)}:{bare_suffix}"
    )

    body = _render(units)
    cross_refs = _dedupe([r for u in units for r in u.cross_ref_recitals])
    terms_used = _dedupe([t for u in units for t in u.terms_used])
    is_numbered_article = doc.kind in (DocKind.AI_ACT_ARTICLE, DocKind.GDPR_ARTICLE)

    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc.doc_id,
        chunk_type=_CHUNK_TYPE_BY_KIND[doc.kind],
        text=f"{_breadcrumb(doc, paragraph_no)}\n{body}",
        title=doc.title,
        chapter=doc.chapter,
        paragraph_no=paragraph_no or None,
        article_no=doc.number if is_numbered_article else None,
        cross_refs=cross_refs,
        source_url=doc.source_url,
        terms_used=terms_used,
    )


def chunk_document(doc: ParsedDoc) -> list[Chunk]:
    if doc.kind == DocKind.AI_ACT_RECITAL:
        return [_make_chunk_from_group(doc, doc.units, None)]
    groups = _regroup(_group_by_paragraph(doc.units))
    return [_make_chunk_from_group(doc, units, split_point) for units, split_point in groups]


def build_definitions_index(definitions: list[DefinitionHit]) -> dict[str, DefinitionHit]:
    """One entry per unique term (PLAN.md §5.2.5), keyed case-insensitively —
    the same term appears in different casing depending on where a sentence
    uses it (e.g. "fundamental rights" vs "Fundamental Rights" leading a
    sentence). Every page that uses a defined term inlines an identical copy
    of its tooltip, so first-seen casing/body wins rather than needing Art. 3
    specifically.
    """
    index: dict[str, DefinitionHit] = {}
    for d in definitions:
        index.setdefault(d.term.strip().lower(), d)
    return index


def _definition_chunks(index: dict[str, DefinitionHit]) -> list[Chunk]:
    chunks = []
    for key, hit in index.items():
        term = hit.term
        chunks.append(
            Chunk(
                chunk_id=definition_chunk_id(key),
                doc_id="ai_act:art:3",
                chunk_type="definition",
                text=f"AI Act › Definitions › {term}\n{hit.body}",
                title=term,
                chapter=None,
                paragraph_no=None,
                article_no=None,
                cross_refs=[],
                source_url=eurlex_citation_url(DocKind.AI_ACT_ARTICLE),
                terms_used=[term],
            )
        )
    return chunks


def build_corpus_chunks(docs: list[ParsedDoc]) -> tuple[list[Chunk], dict[str, DefinitionHit]]:
    chunks: list[Chunk] = []
    all_definitions: list[DefinitionHit] = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
        all_definitions.extend(doc.definitions)

    definitions_index = build_definitions_index(all_definitions)
    chunks.extend(_definition_chunks(definitions_index))
    return chunks, definitions_index
