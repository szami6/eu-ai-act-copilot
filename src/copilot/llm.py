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
    (
        # agent.nodes.triage (Phase 3) — as with `rewrite` above, a fixed
        # `normalized_query` can't reflect the real latest message; routing
        # to "plan" exercises the deeper (planner -> execute) path by
        # default rather than the shallower "direct" one.
        re.compile(r"You triage one turn"),
        '{"route": "plan", "normalized_query": "Is our CV screener high-risk and what are '
        'our obligations?", "refusal_reason": null}',
    ),
    (
        # agent.nodes.planner (Phase 3) — mirrors PLAN §1.4's own flagship
        # composite example (classify_risk_tier -> compliance_timeline),
        # chained with the extract/compliance-timeline fixtures below so a
        # full dummy run exercises `execute`'s dependency-batching.
        re.compile(r"Decompose this question into an ordered list"),
        '{"subtasks": ['
        '{"step_id": 0, "tool": "classify_risk_tier", "query": "An automated CV screening '
        'system used by an employer to filter job applicants.", "depends_on": []}, '
        '{"step_id": 1, "tool": "compliance_timeline", "query": "Compute the obligation '
        'timeline for this system.", "depends_on": [0]}]}',
    ),
    (
        # agent.tools.extract (Phase 3) — flags the employment domain so the
        # planner fixture's step 0 lands on HIGH_RISK via Annex III, kept
        # consistent with the compliance-timeline fixture below rather than
        # the all-empty/MINIMAL_RISK default.
        re.compile(r"Extract a structured feature vector from this description"),
        '{"prohibited_practices": [], "claimed_exceptions": [], '
        '"is_safety_component_of_regulated_product": false, '
        '"requires_third_party_conformity_assessment": false, '
        '"is_solely_non_safety_convenience_feature": false, "safety_relevant": false, '
        '"annex_iii_domains": ["employment_worker_management"], '
        '"performs_profiling_of_natural_persons": false, "derogation_grounds": [], '
        '"transparency_triggers": []}',
    ),
    (
        # agent.nodes.execute's compliance_timeline argument extraction
        # (Phase 3) — tier/high_risk_basis match the extract fixture above
        # (HIGH_RISK via Annex III) rather than re-deriving independently.
        re.compile(r"Extract the arguments needed to compute a compliance timeline"),
        '{"tier": "high_risk", "role": "provider", "placing_on_market_date": "2026-01-01", '
        '"high_risk_basis": "annex_iii", "prohibited_practice": null, "is_gpai_model": false, '
        '"is_public_authority": false, "is_large_scale_it_system_component": false, '
        '"generates_synthetic_content": false}',
    ),
    (
        # agent.nodes.synthesize (Phase 3) — free text, no schema; one
        # citation marker so a dummy run has something for `verify`'s
        # citation-resolvability check to actually resolve.
        re.compile(r"Answer the question using only the passages"),
        "Providers of high-risk AI systems must implement a risk management system "
        "and maintain technical documentation before placing the system on the "
        "market. [Art. 9(1)]",
    ),
    (
        # agent.nodes.verify (Phase 3) — comfortably above the default
        # groundedness_threshold (0.7) so a dummy run reaches `finalize`
        # without needing the retry loop.
        re.compile(r"Judge whether this draft answer is grounded"),
        '{"groundedness": 0.85, "ungrounded_claims": [], "retry_query": null}',
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
