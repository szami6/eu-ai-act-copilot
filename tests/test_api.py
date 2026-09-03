"""Unit tests for the FastAPI health endpoint — see PLAN.md §5.2 / §9."""

from __future__ import annotations

from fastapi.testclient import TestClient

from copilot.api.main import app
from copilot.config import LLMProvider, Settings, get_settings


def test_health_ok_with_dummy_provider() -> None:
    # Explicit override isolates this test from whatever a developer's local
    # `.env` or shell environment happens to contain.
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, llm_provider=LLMProvider.DUMMY
    )
    try:
        with TestClient(app) as client:
            resp = client.get("/health")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "dummy"
    assert body["llm_reachable"] is None
    assert body["corpus_manifest"] is None
