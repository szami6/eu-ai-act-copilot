"""Corpus source registry (PLAN.md §5.1).

EUR-Lex itself sits behind an AWS WAF JavaScript challenge — verified during
Phase 1: every `eur-lex.europa.eu/legal-content/...` path (HTML and PDF
alike, with or without browser-like headers) returns HTTP 202 with an
`x-amzn-waf-action: challenge` header, which a plain HTTP client cannot
solve. Headless-browser automation could defeat it, but that is a heavy,
fragile dependency to add purely to re-host text that is already public
elsewhere.

We fetch chunk *content* from two structurally-clean mirrors instead —
artificialintelligenceact.eu (Future of Life Institute; per-article/
recital/annex pages with the article tree already marked up in the DOM,
discoverable via its WordPress sitemaps) and gdpr-info.eu (per-article
pages, simple nested <ol> structure) — and record EUR-Lex as the citation
of record (`source_url`) on every chunk, since that remains the
authoritative legal text. See `eurlex_citation_url`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

AI_ACT_SITEMAP = "https://artificialintelligenceact.eu/wp-sitemap-posts-{kind}-1.xml"
AI_ACT_CELEX = "32024R1689"
GDPR_CELEX = "32016R0679"

_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII"]


def to_roman_annex(number: int) -> str:
    """AI Act annexes are cited by Roman numeral (Annex III, not Annex 3)."""
    return _ROMAN[number - 1]


class DocKind(StrEnum):
    AI_ACT_ARTICLE = "ai_act_article"
    AI_ACT_RECITAL = "ai_act_recital"
    AI_ACT_ANNEX = "ai_act_annex"
    GDPR_ARTICLE = "gdpr_article"


@dataclass(frozen=True)
class SourceDoc:
    """One fetchable page: an AI Act article/recital/annex, or a GDPR article."""

    doc_id: str
    kind: DocKind
    number: int
    fetch_url: str
    title: str | None = None


def eurlex_citation_url(kind: DocKind) -> str:
    """The authoritative EUR-Lex document a chunk's `source_url` points to.

    Per-article anchors on the current EUR-Lex site could not be verified
    (the site is unreachable programmatically, see module docstring), so
    this deliberately resolves to the top-level consolidated text rather
    than guessing at an unverified `#art_N` fragment.
    """
    celex = AI_ACT_CELEX if kind != DocKind.GDPR_ARTICLE else GDPR_CELEX
    return f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"


# GDPR: PLAN.md §5.1 scopes exactly these five articles — enough for a
# genuine cross-regulation multi-hop question (Art. 22 automated
# decision-making <-> AI Act high-risk) without ingesting all 99 articles.
GDPR_ARTICLES: tuple[tuple[int, str], ...] = (
    (5, "Principles relating to processing of personal data"),
    (6, "Lawfulness of processing"),
    (9, "Processing of special categories of personal data"),
    (22, "Automated individual decision-making, including profiling"),
    (35, "Data protection impact assessment"),
)


def gdpr_source_docs() -> list[SourceDoc]:
    return [
        SourceDoc(
            doc_id=f"gdpr:art:{no}",
            kind=DocKind.GDPR_ARTICLE,
            number=no,
            fetch_url=f"https://gdpr-info.eu/art-{no}-gdpr/",
            title=title,
        )
        for no, title in GDPR_ARTICLES
    ]
