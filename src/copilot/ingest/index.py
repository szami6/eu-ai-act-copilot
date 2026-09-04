"""Embed chunks and upsert into Qdrant; write the corpus manifest (PLAN.md §5.2).

Qdrant is the single persisted source of truth for chunk text + payload —
the RAG subgraph's in-process BM25 index (§4.5) is rebuilt at startup by
scrolling every point back out of this collection rather than persisting a
second copy of the corpus.

Idempotent by design: a point's id is a UUID5 derived from its chunk id, so
re-running ingest after a source edit *overwrites* the same points instead
of accumulating duplicates alongside them.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient, models

from copilot.embeddings import EMBEDDING_DIM, EMBEDDING_MODEL_NAME, get_embedding_model
from copilot.ingest.chunk import Chunk
from copilot.ingest.parse import DefinitionHit

logger = logging.getLogger(__name__)

COLLECTION_NAME = "ai_act_chunks"
_UPSERT_BATCH_SIZE = 200
# fastembed's own default (256) OOM'd ONNX Runtime's attention kernel on a
# memory-constrained host; the corpus is small enough that a smaller batch
# costs a negligible amount of extra wall-clock for a much lower peak.
_EMBED_BATCH_SIZE = 16
# Arbitrary but fixed — only its stability across runs matters, not its value.
_POINT_ID_NAMESPACE = uuid.UUID("6f6b6e35-2c1a-4b8e-9c1d-5a1e2f3b4c5d")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))


def _payload(chunk: Chunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "chunk_type": chunk.chunk_type,
        "text": chunk.text,
        "title": chunk.title,
        "chapter": chunk.chapter,
        "paragraph_no": chunk.paragraph_no,
        "article_no": chunk.article_no,
        "cross_refs": chunk.cross_refs,
        "source_url": chunk.source_url,
        "terms_used": chunk.terms_used,
    }


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
    )
    logger.info("Created Qdrant collection %r", COLLECTION_NAME)


def index_chunks(client: QdrantClient, chunks: list[Chunk]) -> None:
    """Embed every chunk's (breadcrumb-prefixed) text and upsert into Qdrant."""
    ensure_collection(client)
    model = get_embedding_model()
    vectors = list(model.embed([c.text for c in chunks], batch_size=_EMBED_BATCH_SIZE))

    for start in range(0, len(chunks), _UPSERT_BATCH_SIZE):
        end = start + _UPSERT_BATCH_SIZE
        points = [
            models.PointStruct(id=_point_id(c.chunk_id), vector=v.tolist(), payload=_payload(c))
            for c, v in zip(chunks[start:end], vectors[start:end], strict=True)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info("Upserted %d/%d chunks", end, len(chunks))


def write_manifest(
    manifest_path: Path,
    source_manifest: dict[str, dict[str, str]],
    chunk_count: int,
    definitions: dict[str, DefinitionHit],
) -> dict[str, object]:
    """The corpus-version summary `/health` surfaces (PLAN.md §5.2 closing paragraph)."""
    manifest: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "collection_name": COLLECTION_NAME,
        "chunk_count": chunk_count,
        "definitions_count": len(definitions),
        "source_hashes": {doc_id: entry["sha256"] for doc_id, entry in source_manifest.items()},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote ingest manifest to %s", manifest_path)
    return manifest
