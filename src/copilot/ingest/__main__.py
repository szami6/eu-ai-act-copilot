"""Entrypoint for `python -m copilot.ingest` (the `ingest` compose service).

One-shot pipeline (PLAN.md §5): fetch -> parse -> chunk -> embed -> upsert
into Qdrant, then write the corpus manifest `/health` surfaces. Idempotent —
safe to re-run after a source edit or an embedding-model change.
"""

from __future__ import annotations

import asyncio
import logging

from qdrant_client import QdrantClient

from copilot.config import get_settings
from copilot.ingest.chunk import build_corpus_chunks
from copilot.ingest.fetch import fetch_all
from copilot.ingest.index import index_chunks, write_manifest
from copilot.ingest.parse import parse_source_doc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def run() -> None:
    settings = get_settings()

    source_manifest, fetched = await fetch_all(settings.data_dir / "raw")
    logger.info("Fetched %d source documents", len(fetched))

    parsed = [parse_source_doc(html, doc) for doc, html in fetched]
    chunks, definitions = build_corpus_chunks(parsed)
    logger.info("Built %d chunks (%d defined terms)", len(chunks), len(definitions))

    client = QdrantClient(url=settings.qdrant_url)
    index_chunks(client, chunks)

    manifest = write_manifest(
        settings.data_dir / "manifest.json", source_manifest, len(chunks), definitions
    )
    logger.info("Ingestion complete: %s", manifest)


def main() -> int:
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
