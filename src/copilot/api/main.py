"""FastAPI application — Phase 0 exposes only `/health`.

The chat endpoint (`/chat`, SSE-streamed node events per PLAN.md §2.1) is
built in Phase 4, once the orchestrator graph (Phase 3) exists to serve it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI
from pydantic import BaseModel

from copilot.config import LLMProvider, Settings, get_settings


class HealthResponse(BaseModel):
    status: str
    app_env: str
    git_sha: str
    llm_provider: LLMProvider
    llm_reachable: bool | None  # None: not applicable — the dummy provider has nothing to reach
    qdrant_reachable: bool
    corpus_manifest: dict[str, object] | None  # populated once Phase 1 ingest writes manifest.json


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.http_client = httpx.AsyncClient(timeout=5.0)
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="EU AI Act Copilot API", lifespan=lifespan)


async def _check_llm(settings: Settings, client: httpx.AsyncClient) -> bool | None:
    if settings.llm_provider is LLMProvider.DUMMY:
        return None
    try:
        resp = await client.get(f"{settings.llm_base_url.rstrip('/')}/models")
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


async def _check_qdrant(settings: Settings, client: httpx.AsyncClient) -> bool:
    try:
        resp = await client.get(f"{settings.qdrant_url.rstrip('/')}/healthz")
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def _read_corpus_manifest(settings: Settings) -> dict[str, object] | None:
    """`None` before Phase 1 ingest has ever run, and also — like
    `_check_llm`/`_check_qdrant` above — if `ingest` is mid-rewrite of this
    same bind-mounted file when a request lands (compose lets a corpus
    refresh run without taking `api` down), rather than 500ing `/health`
    over a transient torn read.
    """
    manifest_path = settings.data_dir / "manifest.json"
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@app.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Report service status without hard-failing on a down dependency.

    The LLM may be local (compose `gpu`/`cpu` profile), remote over the LAN
    (PLAN.md §2.3 topology B), or absent (`dummy`) — so reachability is
    reported, not enforced. An unreachable LLM or Qdrant degrades what a
    caller sees on `/chat`, not the liveness of this process.
    """
    client: httpx.AsyncClient = app.state.http_client
    return HealthResponse(
        status="ok",
        app_env=settings.app_env,
        git_sha=settings.git_sha,
        llm_provider=settings.llm_provider,
        llm_reachable=await _check_llm(settings, client),
        qdrant_reachable=await _check_qdrant(settings, client),
        corpus_manifest=_read_corpus_manifest(settings),
    )
