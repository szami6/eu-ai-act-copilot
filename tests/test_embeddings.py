"""Unit tests for query-time embedding (PLAN.md §4.5).

`embed_query` is a thin wrapper, so the fake stands in for whatever
`TextEmbedding.query_embed` returns (a one-item list of numpy-array-likes) —
enough to prove the unwrap-and-`tolist()` plumbing without loading the real
~200 MB ONNX model.
"""

from __future__ import annotations

import pytest

from copilot.rag import embeddings as embeddings_module
from copilot.rag.embeddings import embed_query


class _FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class _FakeEmbeddingModel:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def query_embed(self, text: str) -> list[_FakeVector]:
        return [_FakeVector(self._values)]


def test_embed_query_returns_plain_float_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        embeddings_module, "get_embedding_model", lambda: _FakeEmbeddingModel([1.0, 2.0, 3.0])
    )

    vector = embed_query("what is a high-risk AI system?")

    assert vector == [1.0, 2.0, 3.0]
