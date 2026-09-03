"""Unit tests for the LLM client factory — see PLAN.md §4.6."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from copilot.config import LLMProvider, Settings
from copilot.llm import DummyChatModel, get_chat_model


def test_dummy_provider_returns_dummy_chat_model() -> None:
    settings = Settings(_env_file=None, llm_provider=LLMProvider.DUMMY)
    model = get_chat_model(settings)
    assert isinstance(model, DummyChatModel)


def test_dummy_chat_model_is_deterministic() -> None:
    model = DummyChatModel()
    first = model.invoke([HumanMessage(content="hello there")])
    second = model.invoke([HumanMessage(content="hello there")])
    assert first.content == second.content
    assert "Hello" in str(first.content)


def test_dummy_chat_model_default_fixture_for_unmatched_input() -> None:
    model = DummyChatModel()
    result = model.invoke([HumanMessage(content="what is the capital of France?")])
    assert "dummy-llm" in str(result.content)
