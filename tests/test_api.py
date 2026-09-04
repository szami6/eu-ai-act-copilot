"""Unit tests for the FastAPI health endpoint — see PLAN.md §5.2 / §9."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from copilot.api.main import app
from copilot.config import LLMProvider, Settings, get_settings


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
