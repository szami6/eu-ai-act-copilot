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
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda, RunnableMap, RunnablePassthrough
from pydantic import BaseModel

from copilot.config import LLMProvider, Settings

_FIXTURES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bhello\b|\bhi\b", re.IGNORECASE),
        "Hello! I'm running against the dummy LLM fixture.",
    ),
    (
        # rag.graph.rewrite (Phase 1) — the exact variants don't matter under
        # LLM_PROVIDER=dummy, only that this parses as a valid RewriteOutput.
        re.compile(r"Rewrite this AI Act compliance question"),
        '{"keyword_variant": "high-risk AI system requirements providers", '
        '"legal_variant": "obligations applicable to providers of high-risk AI systems", '
        '"hyde_passage": null}',
    ),
    (
        # rag.graph.grade (Phase 1) — a fixed response can't name the real
        # candidate ids it was shown, so it leaves `grades` empty; the node
        # treats that as "pass everything through" rather than "grade
        # everything irrelevant" (see graph.py's `grade`).
        re.compile(r"Grade each candidate's relevance"),
        '{"grades": [], "sufficient": true, "broadened_query": null}',
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
    §4.6). Response selection is a keyword match against `_FIXTURES`; the
    RAG subgraph's `rewrite`/`grade` nodes (Phase 1) and the orchestrator's
    routing/planning nodes (Phase 3) extend this table as their prompts are
    defined.
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

    def with_structured_output(
        self, schema: dict[str, Any] | type, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        """Parse the matched fixture's text as JSON for `schema` directly,
        bypassing `BaseChatModel`'s default tool-calling-based machinery.

        That default (see `langchain_core`) binds `schema` as a forced tool
        call and parses the result out of the response's `.tool_calls` —
        `bind_tools` above is a no-op stand-in, so `.tool_calls` is always
        empty and that path always raises. A fixture matching a structured
        prompt should return a JSON string valid for `schema`; real-provider
        parity in shape (something invocable that yields a `schema`
        instance) is what nodes depend on, not the parsing mechanism.
        """
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            msg = "DummyChatModel.with_structured_output needs a Pydantic schema"
            raise NotImplementedError(msg)

        def _parse(message: BaseMessage) -> BaseModel:
            return schema.model_validate_json(str(message.content))

        if include_raw:
            return RunnableMap(raw=self) | RunnablePassthrough.assign(
                parsed=lambda d: _parse(d["raw"]), parsing_error=lambda _d: None
            )
        return self | RunnableLambda(_parse)


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
