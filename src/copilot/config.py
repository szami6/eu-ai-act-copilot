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
    qdrant_url: str = "http://qdrant:6333"

    # --- API service ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- UI service: how the Streamlit process reaches the API ---
    api_base_url: str = "http://api:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
