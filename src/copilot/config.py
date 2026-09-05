"""Application settings — the single source of truth for configuration.

Every other module reads configuration through `Settings`, never
`os.environ` directly, so the full set of knobs is discoverable in one
place and is identically overridable via environment variables, a `.env`
file, or compose `environment:` blocks across every deployment topology
(PLAN.md §2.3).
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    """Which chat-model backend `copilot.llm.get_chat_model` returns."""

    DUMMY = "dummy"
    OPENAI_COMPATIBLE = "openai_compatible"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="dev", description="dev | ci | prod, surfaced at /health")
    git_sha: str = Field(default="unknown", description="Set by the Docker build ARG")

    # --- LLM: always reached through an OpenAI-compatible base URL (PLAN.md §2.1) ---
    llm_provider: LLMProvider = LLMProvider.DUMMY
    llm_base_url: str = "http://llm-gpu:8000/v1"
    llm_api_key: str = "devkey"
    llm_model: str = "Qwen/Qwen3-8B-FP8"
    llm_request_timeout_s: float = 30.0

    # --- Retrieval backend (wired up in Phase 1) ---
    # `api`/`ingest` never actually read this default: compose's own
    # `environment:` block sets the real `QDRANT_URL` (the compose-internal
    # `http://qdrant:6333`) for both containers regardless of `.env`. This
    # default is only ever consulted by host-run tools with no container
    # around them (pytest, `make eval`/`run_eval.py`) — for those, Qdrant is
    # reachable at its compose-published host port, not the container DNS
    # name — so `.env.example` matches this value, not compose's.
    qdrant_url: str = "http://localhost:6333"
    # Where `ingest` writes raw docs + manifest.json and `/health` reads the
    # manifest back from — one field so both agree even though they're
    # different containers/processes (PLAN.md §2.3). The `api` service's
    # compose entry mounts the same host `./data` read-only for this reason.
    data_dir: Path = Path("data")
    # Cross-encoder rerank is config-gated (PLAN.md §3.2 `rerank`): the
    # ablation in §7.4 needs to turn it off, and §8.5 may move it to GPU —
    # both start from this one flag rather than an env-specific branch.
    rerank_enabled: bool = True
    # `execute`'s implicit rag_search steps (and the UI's top-k slider,
    # PLAN.md §6) read this rather than a literal — the one place both
    # agree on the default without either hardcoding the other's number.
    rag_default_k: int = 5

    # --- Orchestrator bounding (PLAN.md §3.1 "Bounding") ---
    max_plan_steps: int = 4
    max_verify_retries: int = 1
    recursion_limit: int = 25
    groundedness_threshold: float = 0.7
    # Not given a figure in PLAN.md, unlike the four above. Chosen relative
    # to `llm_request_timeout_s`: a turn can chain several sequential LLM
    # calls (triage, planner, per-dependency-batch extraction, synthesize,
    # verify, and a full repeat of the last three on one retry) whose
    # *expected* latency is a few seconds each — 60s comfortably covers that
    # common case while still being a real, user-facing bound. It is a wall
    # clock the graph checks itself (`deadline_ts` in state), independent of
    # any single call's own `llm_request_timeout_s`.
    turn_deadline_s: float = 60.0

    # --- API service ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- UI service: how the Streamlit process reaches the API ---
    api_base_url: str = "http://api:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
