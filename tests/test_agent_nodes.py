"""Unit tests for the orchestrator's 6 nodes and both conditional routes
(PLAN.md §3.1): `triage`, `planner`, `execute`, `synthesize`, `verify`,
`finalize`, `route_after_triage`, `route_after_execute`, `route_after_verify`.

All driven by `DummyChatModel` fixtures (`monkeypatch.setattr(llm_module,
"_FIXTURES", [...])`, per-test as in `tests/test_graph.py`) so nothing here
needs a live Qdrant/GPU. `execute`'s tests deliberately only ever plan
`classify_risk_tier`/`compliance_timeline` steps, never `rag_search` — that
tool wraps the real RAG subgraph, which `test_graph.py` already excludes
from unit tests for the same reason (needs live Qdrant).
"""

from __future__ import annotations

import re
import time
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.runtime import Runtime

from copilot import llm as llm_module
from copilot.agent.context import AgentContext
from copilot.agent.nodes.execute import execute, route_after_execute
from copilot.agent.nodes.finalize import finalize
from copilot.agent.nodes.planner import planner
from copilot.agent.nodes.synthesize import synthesize
from copilot.agent.nodes.triage import route_after_triage, triage
from copilot.agent.nodes.verify import route_after_verify, verify
from copilot.agent.prompts.synthesize import build_synthesize_prompt
from copilot.agent.state import AgentState, NodeEvent, SubTask, ToolCallRecord
from copilot.agent.tools.bindings import make_tools
from copilot.config import Settings
from copilot.llm import DummyChatModel
from copilot.rag.state import Evidence

_ALL_DEFAULT_FEATURES = (
    '{"prohibited_practices": [], "claimed_exceptions": [], '
    '"is_safety_component_of_regulated_product": false, '
    '"requires_third_party_conformity_assessment": false, '
    '"is_solely_non_safety_convenience_feature": false, "safety_relevant": false, '
    '"annex_iii_domains": [], "performs_profiling_of_natural_persons": false, '
    '"derogation_grounds": [], "transparency_triggers": []}'
)


def _context(
    *, chat_model: DummyChatModel | None = None, settings: Settings | None = None
) -> AgentContext:
    model = chat_model if chat_model is not None else DummyChatModel()
    return AgentContext(
        chat_model=model,
        tools=make_tools(model, cast(Any, None), cast(Any, None)),
        settings=settings if settings is not None else Settings(_env_file=None),
    )


def _state(
    *,
    messages: list[AnyMessage] | None = None,
    route: str = "direct",
    normalized_query: str = "what are the obligations for high-risk AI systems?",
    refusal_reason: str | None = None,
    plan: list[SubTask] | None = None,
    cursor: int = 0,
    evidence: list[Evidence] | None = None,
    tool_calls: list[ToolCallRecord] | None = None,
    draft: str | None = None,
    groundedness: float | None = None,
    verify_retries: int = 0,
    deadline_ts: float | None = None,
    partial: bool = False,
) -> AgentState:
    return AgentState(
        messages=messages if messages is not None else [HumanMessage(content=normalized_query)],
        session_id="test-session",
        route=cast(Any, route),
        normalized_query=normalized_query,
        refusal_reason=refusal_reason,
        plan=plan or [],
        cursor=cursor,
        evidence=evidence or [],
        tool_calls=tool_calls or [],
        draft=draft,
        groundedness=groundedness,
        verify_retries=verify_retries,
        deadline_ts=deadline_ts if deadline_ts is not None else time.time() + 60.0,
        partial=partial,
        trace=[],
    )


def _evidence(chunk_id: str, text: str = "Breadcrumb\nBody text.") -> Evidence:
    return Evidence(
        chunk_id=chunk_id,
        doc_id="ai_act:art:6",
        chunk_type="article",
        text=text,
        title=None,
        chapter=None,
        paragraph_no=None,
        article_no=6,
        cross_refs=[],
        source_url="https://example.invalid",
        score=1.0,
        origin="retrieved",
        terms_used=[],
    )


# --- triage ------------------------------------------------------------------


async def test_triage_sets_route_and_normalized_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"triage"),
                '{"route": "direct", "normalized_query": "resolved question", '
                '"refusal_reason": null}',
            )
        ],
    )

    update = await triage(_state(), Runtime(context=_context()))

    assert update["route"] == "direct"
    assert update["normalized_query"] == "resolved question"
    assert update["refusal_reason"] is None


async def test_triage_refuse_carries_refusal_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"triage"),
                '{"route": "refuse", "normalized_query": "n/a", '
                '"refusal_reason": "This assistant only covers the EU AI Act and GDPR."}',
            )
        ],
    )

    update = await triage(_state(), Runtime(context=_context()))

    assert update["route"] == "refuse"
    assert update["refusal_reason"] == "This assistant only covers the EU AI Act and GDPR."


def test_route_after_triage_refuse_goes_to_finalize() -> None:
    assert route_after_triage(_state(route="refuse")) == "finalize"


def test_route_after_triage_direct_goes_to_execute() -> None:
    assert route_after_triage(_state(route="direct")) == "execute"


def test_route_after_triage_plan_goes_to_planner() -> None:
    assert route_after_triage(_state(route="plan")) == "planner"


# --- planner -------------------------------------------------------------


async def test_planner_builds_plan_from_llm_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"Decompose"),
                '{"subtasks": [{"step_id": 0, "tool": "rag_search", "query": "q1", '
                '"depends_on": []}]}',
            )
        ],
    )

    update = await planner(_state(), Runtime(context=_context()))

    plan = cast(list[SubTask], update["plan"])
    assert len(plan) == 1
    assert plan[0].tool == "rag_search"
    assert update["cursor"] == 0


async def test_planner_truncates_to_max_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    subtasks = ", ".join(
        f'{{"step_id": {i}, "tool": "rag_search", "query": "q{i}", "depends_on": []}}'
        for i in range(6)
    )
    monkeypatch.setattr(
        llm_module, "_FIXTURES", [(re.compile(r"Decompose"), f'{{"subtasks": [{subtasks}]}}')]
    )
    settings = Settings(_env_file=None, max_plan_steps=2)

    update = await planner(_state(), Runtime(context=_context(settings=settings)))

    assert len(cast(list[SubTask], update["plan"])) == 2


async def test_planner_drops_forward_and_self_referencing_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"Decompose"),
                '{"subtasks": ['
                '{"step_id": 0, "tool": "rag_search", "query": "q0", "depends_on": [1]}, '
                '{"step_id": 1, "tool": "rag_search", "query": "q1", "depends_on": [1]}]}',
            )
        ],
    )

    update = await planner(_state(), Runtime(context=_context()))

    plan = cast(list[SubTask], update["plan"])
    assert plan[0].depends_on == []  # 1 is not < 0 (forward reference)
    assert plan[1].depends_on == []  # 1 is not < 1 (self-reference)


# --- execute ---------------------------------------------------------------


async def test_execute_runs_single_step_and_advances_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_module, "_FIXTURES", [(re.compile(r"feature vector"), _ALL_DEFAULT_FEATURES)]
    )
    plan = [
        SubTask(step_id=0, tool="classify_risk_tier", query="a CV screening tool", depends_on=[])
    ]

    update = await execute(_state(plan=plan, cursor=0), Runtime(context=_context()))

    tool_calls = cast(list[ToolCallRecord], update["tool_calls"])
    assert len(tool_calls) == 1
    assert tool_calls[0].tool == "classify_risk_tier"
    assert tool_calls[0].ok is True
    assert update["cursor"] == 1


async def test_execute_runs_independent_steps_concurrently_in_one_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_module, "_FIXTURES", [(re.compile(r"feature vector"), _ALL_DEFAULT_FEATURES)]
    )
    plan = [
        SubTask(step_id=0, tool="classify_risk_tier", query="system A", depends_on=[]),
        SubTask(step_id=1, tool="classify_risk_tier", query="system B", depends_on=[]),
    ]

    update = await execute(_state(plan=plan, cursor=0), Runtime(context=_context()))

    assert update["cursor"] == 2
    assert len(cast(list[ToolCallRecord], update["tool_calls"])) == 2


async def test_execute_waits_for_dependency_before_running_dependent_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"feature vector"),
                '{"prohibited_practices": [], "claimed_exceptions": [], '
                '"is_safety_component_of_regulated_product": false, '
                '"requires_third_party_conformity_assessment": false, '
                '"is_solely_non_safety_convenience_feature": false, "safety_relevant": false, '
                '"annex_iii_domains": ["employment_worker_management"], '
                '"performs_profiling_of_natural_persons": false, "derogation_grounds": [], '
                '"transparency_triggers": []}',
            ),
            (
                re.compile(r"compute a compliance timeline"),
                '{"tier": "high_risk", "role": "provider", '
                '"placing_on_market_date": "2026-01-01", "high_risk_basis": null, '
                '"prohibited_practice": null, "is_gpai_model": false, '
                '"is_public_authority": false, "is_large_scale_it_system_component": false, '
                '"generates_synthetic_content": false}',
            ),
        ],
    )
    plan = [
        SubTask(step_id=0, tool="classify_risk_tier", query="a CV screener", depends_on=[]),
        SubTask(step_id=1, tool="compliance_timeline", query="its timeline", depends_on=[0]),
    ]
    ctx = _context()

    first = await execute(_state(plan=plan, cursor=0), Runtime(context=ctx))
    assert first["cursor"] == 1
    first_calls = cast(list[ToolCallRecord], first["tool_calls"])
    assert [tc.tool for tc in first_calls] == ["classify_risk_tier"]
    assert "Annex III, point 4" in first_calls[0].summary
    prompt = build_synthesize_prompt("classify it", [], first_calls)
    assert '"triggering_criteria"' in prompt
    assert "Annex III, point 4" in prompt

    second_state = _state(plan=plan, cursor=1, tool_calls=first_calls)
    second = await execute(second_state, Runtime(context=ctx))

    assert second["cursor"] == 2
    second_calls = cast(list[ToolCallRecord], second["tool_calls"])
    assert [tc.tool for tc in second_calls] == ["compliance_timeline"]
    assert second_calls[0].ok is True


async def test_execute_records_failed_step_when_tool_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"compute a compliance timeline"),
                # tier=high_risk with no high_risk_basis: compliance_timeline
                # itself raises ValueError.
                '{"tier": "high_risk", "role": "provider", '
                '"placing_on_market_date": "2026-01-01", "high_risk_basis": null, '
                '"prohibited_practice": null, "is_gpai_model": false, '
                '"is_public_authority": false, "is_large_scale_it_system_component": false, '
                '"generates_synthetic_content": false}',
            )
        ],
    )
    plan = [SubTask(step_id=0, tool="compliance_timeline", query="x", depends_on=[])]

    update = await execute(_state(plan=plan, cursor=0), Runtime(context=_context()))

    tool_calls = cast(list[ToolCallRecord], update["tool_calls"])
    assert tool_calls[0].ok is False
    assert update["cursor"] == 1  # cursor still advances past a failed step


async def test_execute_deadline_exceeded_short_circuits_without_running_steps() -> None:
    plan = [SubTask(step_id=0, tool="classify_risk_tier", query="x", depends_on=[])]
    state = _state(plan=plan, cursor=0, deadline_ts=time.time() - 1.0)

    update = await execute(state, Runtime(context=_context()))

    assert update["partial"] is True
    assert "tool_calls" not in update


def test_route_after_execute_loops_when_more_ready_steps_remain() -> None:
    plan = [
        SubTask(step_id=0, tool="rag_search", query="q0", depends_on=[]),
        SubTask(step_id=1, tool="rag_search", query="q1", depends_on=[]),
    ]
    assert route_after_execute(_state(plan=plan, cursor=0)) == "execute"


def test_route_after_execute_synthesizes_when_plan_drained() -> None:
    plan = [SubTask(step_id=0, tool="rag_search", query="q0", depends_on=[])]
    assert route_after_execute(_state(plan=plan, cursor=1)) == "synthesize"


def test_route_after_execute_synthesizes_when_stalled_on_unsatisfiable_dependency() -> None:
    plan = [
        SubTask(step_id=0, tool="rag_search", query="q0", depends_on=[]),
        SubTask(step_id=1, tool="rag_search", query="q1", depends_on=[5]),
    ]
    assert route_after_execute(_state(plan=plan, cursor=1)) == "synthesize"


def test_route_after_execute_finalizes_when_partial() -> None:
    assert route_after_execute(_state(plan=[], cursor=0, partial=True)) == "finalize"


# --- synthesize --------------------------------------------------------------


async def test_synthesize_produces_draft_from_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_module, "_FIXTURES", [(re.compile(r"passages"), "Some answer. [Art. 6(1)]")]
    )

    update = await synthesize(_state(), Runtime(context=_context()))

    assert update["draft"] == "Some answer. [Art. 6(1)]"


# --- verify ------------------------------------------------------------------


async def test_verify_passes_when_citation_resolves_and_llm_score_is_high(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"grounded"),
                '{"groundedness": 0.9, "ungrounded_claims": [], "retry_query": null}',
            )
        ],
    )
    evidence = [_evidence("a", text="AI Act › Article 6(2)\nHigh-risk systems must register.")]
    state = _state(draft="Systems must register. [Art. 6(2)]", evidence=evidence, verify_retries=0)

    update = await verify(state, Runtime(context=_context()))

    assert cast(float, update["groundedness"]) >= 0.7
    assert "plan" not in update


async def test_verify_unresolvable_citation_caps_groundedness_despite_high_llm_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"grounded"),
                '{"groundedness": 0.95, "ungrounded_claims": [], "retry_query": null}',
            )
        ],
    )
    evidence = [_evidence("a", text="AI Act › Article 6(2)\nBody text.")]
    state = _state(draft="Something unrelated. [Art. 40(1)]", evidence=evidence)

    update = await verify(state, Runtime(context=_context()))

    assert update["groundedness"] == 0.0


async def test_verify_resolves_citations_from_deterministic_tool_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"grounded"),
                '{"groundedness": 0.9, "ungrounded_claims": [], "retry_query": null}',
            )
        ],
    )
    tool_call = ToolCallRecord(
        step_id=0,
        tool="classify_risk_tier",
        input_summary="CV screener",
        result=None,
        summary="tier=high_risk, criteria=Annex III, point 4: employment",
        ok=True,
    )
    state = _state(
        draft="The system is high-risk. [Annex III, point 4]", tool_calls=[tool_call]
    )

    update = await verify(state, Runtime(context=_context()))

    assert update["groundedness"] == 0.9


async def test_verify_appends_retry_step_when_below_threshold_and_budget_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"grounded"),
                '{"groundedness": 0.3, "ungrounded_claims": ["x"], "retry_query": "broader query"}',
            )
        ],
    )
    state = _state(draft="Unsupported claim. [Art. 99(9)]", evidence=[], plan=[], verify_retries=0)

    update = await verify(state, Runtime(context=_context()))

    plan = cast(list[SubTask], update["plan"])
    assert len(plan) == 1
    assert plan[0].tool == "rag_search"
    assert plan[0].query == "broader query"
    assert update["verify_retries"] == 1


async def test_verify_does_not_retry_once_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"grounded"),
                '{"groundedness": 0.3, "ungrounded_claims": [], "retry_query": "broader query"}',
            )
        ],
    )
    state = _state(draft="text [Art. 1]", evidence=[], plan=[], verify_retries=1)

    update = await verify(state, Runtime(context=_context()))

    assert "plan" not in update
    assert "verify_retries" not in update


def test_route_after_verify_execute_when_unexecuted_step_appended() -> None:
    plan = [SubTask(step_id=0, tool="rag_search", query="q", depends_on=[])]
    assert route_after_verify(_state(plan=plan, cursor=0)) == "execute"


def test_route_after_verify_finalize_when_plan_fully_executed() -> None:
    plan = [SubTask(step_id=0, tool="rag_search", query="q", depends_on=[])]
    assert route_after_verify(_state(plan=plan, cursor=1)) == "finalize"


# --- finalize ------------------------------------------------------------


async def test_finalize_uses_refusal_reason_on_refuse_route() -> None:
    state = _state(route="refuse", refusal_reason="Out of scope for this assistant.", draft=None)

    update = await finalize(state)

    content = str(cast(list[AIMessage], update["messages"])[0].content)
    assert "Out of scope for this assistant." in content
    assert "decision support" in content


async def test_finalize_uses_draft_when_present() -> None:
    state = _state(route="direct", draft="The answer. [Art. 1]")

    update = await finalize(state)

    content = str(cast(list[AIMessage], update["messages"])[0].content)
    assert "The answer. [Art. 1]" in content


async def test_finalize_falls_back_to_partial_summary_when_no_draft() -> None:
    tool_calls = [
        ToolCallRecord(
            step_id=0,
            tool="classify_risk_tier",
            input_summary="x",
            result=None,
            summary="tier=high_risk, confidence=1.00",
            ok=True,
        )
    ]
    state = _state(route="plan", draft=None, tool_calls=tool_calls, partial=True)

    update = await finalize(state)

    content = str(cast(list[AIMessage], update["messages"])[0].content)
    assert "time budget" in content
    assert "tier=high_risk, confidence=1.00" in content
    trace = cast(list[NodeEvent], update["trace"])
    assert trace[0].detail == "partial"
