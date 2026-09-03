"""Unit tests for Settings — see PLAN.md §2.2."""

from __future__ import annotations

import pytest

from copilot.config import LLMProvider, Settings


def test_defaults_to_dummy_provider() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_provider is LLMProvider.DUMMY


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    settings = Settings(_env_file=None)
    assert settings.llm_provider is LLMProvider.OPENAI_COMPATIBLE
    assert settings.llm_model == "test-model"
