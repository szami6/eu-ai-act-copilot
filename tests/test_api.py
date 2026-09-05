"""Unit tests for the FastAPI `/health` (§5.2/§9) and `/chat` (§2.1) endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from fastapi.testclient import TestClient

from copilot.agent.graph import build_agent_graph
from copilot.api.main import app, get_app_runtime
from copilot.api.runtime import AppRuntime
from copilot.config import LLMProvider, Settings, get_settings
from copilot.llm import DummyChatModel
from copilot.rag.graph import build_rag_graph
from copilot.rag.retrievers import build_bm25_index


def _get(settings: Settings) -> dict[str, object]:
    # `data_dir` is always overridden explicitly (a `tmp_path`, empty or
    # seeded) so this test's outcome can't depend on whether ingestion
    # happens to have been run in whatever checkout it executes in.
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        with TestClient(app) as client:
            resp = client.get("/health")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    return dict(resp.json())


def test_health_ok_with_dummy_provider(tmp_path: Path) -> None:
    body = _get(Settings(_env_file=None, llm_provider=LLMProvider.DUMMY, data_dir=tmp_path))

    assert body["status"] == "ok"
    assert body["llm_provider"] == "dummy"
    assert body["llm_reachable"] is None


def test_health_corpus_manifest_none_before_ingest_has_run(tmp_path: Path) -> None:
    body = _get(Settings(_env_file=None, llm_provider=LLMProvider.DUMMY, data_dir=tmp_path))
    assert body["corpus_manifest"] is None


def test_health_corpus_manifest_reflects_written_manifest(tmp_path: Path) -> None:
    manifest = {"chunk_count": 1021, "collection_name": "ai_act_chunks"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    body = _get(Settings(_env_file=None, llm_provider=LLMProvider.DUMMY, data_dir=tmp_path))

    assert body["corpus_manifest"] == manifest


def test_health_corpus_manifest_none_on_torn_write(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text('{"chunk_count": 10', encoding="utf-8")

    body = _get(Settings(_env_file=None, llm_provider=LLMProvider.DUMMY, data_dir=tmp_path))

    assert body["corpus_manifest"] is None


def _fake_runtime() -> AppRuntime:
    """A `/chat`-capable runtime with no live Qdrant (`test_events.py`'s
    same rationale: the default dummy fixtures never call `rag_search`) —
    overriding `get_app_runtime` means these tests never pay for (or
    depend on) a real connection attempt, unlike the lazily-built default."""
    return AppRuntime(
        qdrant_client=cast(Any, None),
        bm25_index=build_bm25_index([]),
        chat_model=DummyChatModel(),
        base_settings=Settings(_env_file=None),
        corpus_ready=False,
        agent_compiled=build_agent_graph(),
        rag_compiled=build_rag_graph(),
    )


def _post_chat(payload: dict[str, object]) -> str:
    app.dependency_overrides[get_app_runtime] = _fake_runtime
    try:
        with TestClient(app) as client:
            resp = client.post("/chat", json=payload)
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    return resp.text


def _sse_events(raw: str) -> list[str]:
    return [line.removeprefix("event: ") for line in raw.splitlines() if line.startswith("event: ")]


def test_chat_streams_session_node_and_done_events() -> None:
    body = _post_chat({"message": "Is our CV screener high-risk and what are our obligations?"})

    events = _sse_events(body)
    assert events[0] == "session"
    assert "node" in events
    assert "token" in events
    assert events[-1] == "done"


def test_chat_reuses_the_supplied_session_id() -> None:
    body = _post_chat({"message": "hi", "session_id": "my-session"})

    assert '"session_id": "my-session"' in body


def test_chat_mints_a_session_id_when_none_given() -> None:
    body = _post_chat({"message": "hi"})

    first_line = next(line for line in body.splitlines() if line.startswith("data: "))
    session_id = json.loads(first_line.removeprefix("data: "))["session_id"]
    assert session_id
