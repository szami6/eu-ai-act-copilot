"""Unit tests for the LLM client factory — see PLAN.md §4.6."""

from __future__ import annotations

import re

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from copilot import llm as llm_module
from copilot.config import LLMProvider, Settings
from copilot.llm import DummyChatModel, get_chat_model


class _Colour(BaseModel):
    name: str
    hex: str


def test_dummy_provider_returns_dummy_chat_model() -> None:
    settings = Settings(_env_file=None, llm_provider=LLMProvider.DUMMY)
    model = get_chat_model(settings)
    assert isinstance(model, DummyChatModel)


def test_openai_compatible_provider_uses_deterministic_temperature() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider=LLMProvider.OPENAI_COMPATIBLE,
        llm_base_url="http://example.invalid/v1",
    )

    model = get_chat_model(settings)

    assert model.temperature == 0


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


def test_with_structured_output_parses_matching_json_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [(re.compile(r"favourite colour"), '{"name": "blue", "hex": "#0000ff"}')],
    )
    structured = DummyChatModel().with_structured_output(_Colour)

    result = structured.invoke([HumanMessage(content="what is your favourite colour?")])

    assert result == _Colour(name="blue", hex="#0000ff")


def test_with_structured_output_include_raw_returns_raw_and_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [(re.compile(r"favourite colour"), '{"name": "red", "hex": "#ff0000"}')],
    )
    structured = DummyChatModel().with_structured_output(_Colour, include_raw=True)

    result = structured.invoke([HumanMessage(content="favourite colour?")])

    assert isinstance(result, dict)
    assert result["parsed"] == _Colour(name="red", hex="#ff0000")
    assert "red" in str(result["raw"].content)


def test_with_structured_output_rejects_non_pydantic_schema() -> None:
    model = DummyChatModel()
    with pytest.raises(NotImplementedError):
        model.with_structured_output({"type": "object"})
