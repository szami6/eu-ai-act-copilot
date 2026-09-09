from __future__ import annotations

from typing import Any

import pytest

from copilot.ingest import index as index_module
from copilot.ingest.chunk import Chunk
from copilot.ingest.index import COLLECTION_NAME, index_chunks


class _Vector:
    def tolist(self) -> list[float]:
        return [0.0] * 768


class _EmbeddingModel:
    def embed(self, texts: list[str], batch_size: int) -> list[_Vector]:
        assert texts == ["AI Act › Article 1\nBody"]
        assert batch_size == 16
        return [_Vector()]


class _Client:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def collection_exists(self, name: str) -> bool:
        self.events.append(("exists", name))
        return True

    def delete_collection(self, name: str) -> None:
        self.events.append(("delete", name))

    def create_collection(self, **kwargs: Any) -> None:
        self.events.append(("create", kwargs["collection_name"]))

    def upsert(self, **kwargs: Any) -> None:
        self.events.append(("upsert", kwargs["collection_name"]))


def test_index_replaces_collection_before_upserting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(index_module, "get_embedding_model", lambda: _EmbeddingModel())
    client = _Client()
    chunk = Chunk(
        chunk_id="ai_act:art:1",
        doc_id="ai_act:art:1",
        chunk_type="article",
        text="AI Act › Article 1\nBody",
        title="Subject matter",
        chapter=None,
        paragraph_no=None,
        article_no=1,
        cross_refs=[],
        source_url="https://example.invalid",
    )

    index_chunks(client, [chunk])  # type: ignore[arg-type]

    assert client.events == [
        ("exists", COLLECTION_NAME),
        ("delete", COLLECTION_NAME),
        ("create", COLLECTION_NAME),
        ("upsert", COLLECTION_NAME),
    ]
