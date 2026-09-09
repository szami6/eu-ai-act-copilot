"""Fetch raw corpus HTML (PLAN.md §5.1).

Content-hash snapshotted: every fetch is recorded in `data/raw/manifest.json`
with a sha256 of its body. Re-running ingest re-fetches everything (the
corpus is small — a few hundred short pages) and reports upstream drift;
evaluation artifacts bind themselves to the resulting corpus manifest.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import asdict
from pathlib import Path

import httpx

from copilot.ingest.sources import AI_ACT_SITEMAP, DocKind, SourceDoc, gdpr_source_docs

logger = logging.getLogger(__name__)

USER_AGENT = (
    "eu-ai-act-copilot/0.1 (+https://github.com/szami6/eu-ai-act-copilot; "
    "educational RAG demo corpus builder; low request rate)"
)
_MAX_CONCURRENCY = 6
_REQUEST_DELAY_S = 0.15


async def discover_ai_act_docs(client: httpx.AsyncClient) -> list[SourceDoc]:
    """Enumerate every AI Act article/recital/annex via the site's sitemaps.

    The site has no working in-page table of contents reachable without
    JavaScript; its WordPress sitemap index does list one sitemap per post
    type, which is what this walks.
    """
    kind_map = {
        "article": DocKind.AI_ACT_ARTICLE,
        "recital": DocKind.AI_ACT_RECITAL,
        "annex": DocKind.AI_ACT_ANNEX,
    }
    docs: list[SourceDoc] = []
    for slug, kind in kind_map.items():
        resp = await client.get(AI_ACT_SITEMAP.format(kind=slug))
        resp.raise_for_status()
        urls = re.findall(r"<loc>(.*?)</loc>", resp.text)
        for url in urls:
            m = re.search(rf"/{slug}/(\d+)/?$", url)
            if not m:
                logger.warning("Skipping unparseable %s sitemap URL: %s", slug, url)
                continue
            number = int(m.group(1))
            doc_id = f"ai_act:{'art' if kind == DocKind.AI_ACT_ARTICLE else slug}:{number}"
            docs.append(SourceDoc(doc_id=doc_id, kind=kind, number=number, fetch_url=url))
    return docs


async def _fetch_one(
    client: httpx.AsyncClient, doc: SourceDoc, sem: asyncio.Semaphore
) -> tuple[SourceDoc, str]:
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.get(doc.fetch_url)
                resp.raise_for_status()
                await asyncio.sleep(_REQUEST_DELAY_S)
                return doc, resp.text
            except httpx.HTTPError as exc:
                if attempt == 2:
                    raise
                logger.warning("Retrying %s after %s (attempt %d)", doc.fetch_url, exc, attempt + 1)
                await asyncio.sleep(1.0 * (attempt + 1))
        raise AssertionError("unreachable")


def _safe_filename(doc_id: str) -> str:
    return doc_id.replace(":", "_") + ".html"


async def fetch_all(raw_dir: Path) -> tuple[dict[str, dict[str, str]], list[tuple[SourceDoc, str]]]:
    """Fetch every source document, write raw HTML under `raw_dir`.

    Returns the manifest dict (also written to `raw_dir/manifest.json`):
    `{doc_id: {url, sha256, kind, number}}`, alongside the `(doc, html)`
    pairs themselves so a caller can parse them without a redundant
    round-trip back through disk.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.json"
    previous: dict[str, dict[str, str]] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))

    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, timeout=20.0, follow_redirects=True
    ) as client:
        ai_act_docs = await discover_ai_act_docs(client)
        docs = ai_act_docs + gdpr_source_docs()
        logger.info("Discovered %d source documents", len(docs))

        sem = asyncio.Semaphore(_MAX_CONCURRENCY)
        results = await asyncio.gather(*(_fetch_one(client, d, sem) for d in docs))

    manifest: dict[str, dict[str, str]] = {}
    for doc, html in results:
        digest = hashlib.sha256(html.encode("utf-8")).hexdigest()
        prev = previous.get(doc.doc_id)
        if prev and prev["sha256"] != digest:
            logger.warning(
                "Content drift detected for %s: %s -> %s (source changed since last ingest)",
                doc.doc_id,
                prev["sha256"][:12],
                digest[:12],
            )
        (raw_dir / _safe_filename(doc.doc_id)).write_text(html, encoding="utf-8")
        manifest[doc.doc_id] = {"sha256": digest, **asdict(doc)}

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote %d raw documents to %s", len(manifest), raw_dir)
    return manifest, results
