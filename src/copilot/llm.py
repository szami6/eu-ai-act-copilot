"""LLM client factory.

The application only ever talks to an OpenAI-compatible chat-completions
endpoint (PLAN.md §2.1: "the LLM is only ever reached through an
OpenAI-compatible base URL"). `get_chat_model` returns a LangChain
`BaseChatModel`; which concrete class it returns is the only thing that
changes across the `dummy` / real-provider branches, so graph nodes written
against this interface never need to know which one they got.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from copilot.config import LLMProvider, Settings

_FIXTURES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bhello\b|\bhi\b", re.IGNORECASE),
        "Hello! I'm running against the dummy LLM fixture.",
    ),
]
_DEFAULT_RESPONSE = (
    "[dummy-llm] No real model is configured (LLM_PROVIDER=dummy). "
    "This is a deterministic placeholder response used for tests and CI."
)


class DummyChatModel(BaseChatModel):
    """Deterministic, fixture-backed stand-in for a real chat model.

    Selected via `LLM_PROVIDER=dummy`. Lets graph-topology tests and a
    from-clone `make demo` run with no model download and no GPU (PLAN.md
    §4.6). Response selection is a keyword match against `_FIXTURES`;
    Phase 3 extends this table with structured-output and tool-call
    fixtures once the orchestrator's node contracts are defined.
    """

    model_name: str = "dummy-llm"

    @property
    def _llm_type(self) -> str:
        return "dummy-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_human = next((str(m.content) for m in reversed(messages) if m.type == "human"), "")
        response = _DEFAULT_RESPONSE
        for pattern, fixture in _FIXTURES:
            if pattern.search(last_human):
                response = fixture
                break
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> DummyChatModel:
        # Phase 3 gives this real fixture-based tool-call emission once the
        # orchestrator's tool contracts exist. Tools are accepted (so call
        # sites don't need a provider check) and ignored for now.
        return self


def get_chat_model(settings: Settings) -> BaseChatModel:
    """Return the chat model selected by `settings.llm_provider`."""
    if settings.llm_provider is LLMProvider.DUMMY:
        return DummyChatModel()

    # OPENAI_COMPATIBLE: vLLM (GPU profile), llama.cpp (CPU profile), or a
    # remote vLLM over the LAN (topology B) all speak this same protocol.
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=SecretStr(settings.llm_api_key),
        model=settings.llm_model,
        timeout=settings.llm_request_timeout_s,
    )
