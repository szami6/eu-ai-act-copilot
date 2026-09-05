"""FastAPI application: `/health` (Phase 0) and `/chat` (Phase 4, PLAN.md
§2.1 — SSE-streamed node events over the Phase 3 orchestrator graph).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from copilot.api.events import chat_events
from copilot.api.runtime import AppRuntime, build_agent_context, build_app_runtime
from copilot.config import LLMProvider, Settings, get_settings


class HealthResponse(BaseModel):
    status: str
    app_env: str
    git_sha: str
    llm_provider: LLMProvider
    llm_model: str
    llm_reachable: bool | None  # None: not applicable — the dummy provider has nothing to reach
    qdrant_reachable: bool
    corpus_manifest: dict[str, object] | None  # populated once Phase 1 ingest writes manifest.json


class ChatRequest(BaseModel):
    message: str
    # None starts a new conversation — the server mints an id and the first
    # SSE frame reports it back (PLAN §2.1's UI is the only intended caller,
    # and it doesn't have one to send until then).
    session_id: str | None = None
    # PLAN §6 sidebar overrides: "the toggles let a reviewer reproduce the
    # §7.4 ablation live" — per-turn, not persisted past this one request.
    top_k: int | None = Field(default=None, ge=1, le=20)
    rerank_enabled: bool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.http_client = httpx.AsyncClient(timeout=5.0)
    # Built lazily (`get_app_runtime` below), not here: it makes a live
    # Qdrant connection attempt, and `/health` must stay cheap/instant even
    # when nothing has ever called `/chat` yet (or ever will, in a test that
    # only exercises `/health`).
    app.state.runtime = None
    app.state.sessions = {}
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="EU AI Act Copilot API", lifespan=lifespan)


async def get_app_runtime(settings: Settings = Depends(get_settings)) -> AppRuntime:
    """Built once per app lifetime, on the first request that needs it —
    `qdrant_client.scroll` is blocking, so it runs off the event loop
    thread. Cached on `app.state` rather than `functools.lru_cache`: it must
    reset with `lifespan` (a fresh `TestClient` per test must not reuse a
    prior test's runtime, built from that test's own overridden settings).
    """
    if app.state.runtime is None:
        app.state.runtime = await asyncio.to_thread(build_app_runtime, settings)
    return app.state.runtime


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
        llm_model=settings.llm_model,
        llm_reachable=await _check_llm(settings, client),
        qdrant_reachable=await _check_qdrant(settings, client),
        corpus_manifest=_read_corpus_manifest(settings),
    )


@app.post("/chat")
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    runtime: AppRuntime = Depends(get_app_runtime),
) -> StreamingResponse:
    """Streams one turn as SSE frames (`api.events.chat_events`): a
    `session` frame, one `node` frame per finished orchestrator node, a
    `token` frame per `synthesize` chunk, and a final `done` frame. Session
    history lives in `app.state.sessions`, in-process — a single-container
    demo deployment (PLAN §2.3) has nowhere else that needs it, and
    `agent.graph.run_agent`/`stream_agent` already expect the full history
    passed in explicitly each turn rather than owning it themselves.
    """
    session_id = request.session_id or uuid4().hex
    history = app.state.sessions.get(session_id, [])

    overrides: dict[str, object] = {}
    if request.top_k is not None:
        overrides["rag_default_k"] = request.top_k
    if request.rerank_enabled is not None:
        overrides["rerank_enabled"] = request.rerank_enabled
    turn_settings = settings.model_copy(update=overrides) if overrides else settings
    agent_context = build_agent_context(runtime, turn_settings)

    return StreamingResponse(
        chat_events(
            runtime,
            app.state.sessions,
            session_id=session_id,
            history=history,
            user_input=request.message,
            agent_context=agent_context,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
