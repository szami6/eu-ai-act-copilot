"""`execute` (PLAN.md §3.1, node 3): runs the plan's tool calls, one
dependency-respecting batch per visit, looping back to itself via
`route_after_execute` until the plan is drained (or stalls / time runs out).

Tool-argument extraction is split three ways (PLAN §3.3): `rag_search`'s
argument *is* the planner's `query`, `classify_risk_tier` extracts
internally (`tools/extract.py`), and `compliance_timeline`'s fully-typed
arguments are extracted here, the one case this node itself makes an LLM
call for.
"""

from __future__ import annotations

import asyncio
import time
from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from pydantic import ValidationError

from copilot.agent.context import AgentContext
from copilot.agent.prompts.execute import (
    ComplianceTimelineArgs,
    build_compliance_timeline_extraction_prompt,
)
from copilot.agent.state import AgentState, NodeEvent, SubTask, ToolCallRecord
from copilot.agent.tools.compliance_timeline import HighRiskBasis, ObligationDeadline
from copilot.agent.tools.risk_tier import ProhibitedPractice, RiskTier, RiskTierResult
from copilot.rag.state import Evidence, RagResult

_StepResult = tuple[list[Evidence], ToolCallRecord]


def _implicit_plan(state: AgentState) -> list[SubTask]:
    """The 'direct' route skips `planner` entirely: a single `rag_search`
    step over the triaged `normalized_query` is the whole plan."""
    return [SubTask(step_id=0, tool="rag_search", query=state["normalized_query"], depends_on=[])]


def _ready_batch(plan: list[SubTask], cursor: int) -> list[SubTask]:
    """The maximal contiguous prefix of `plan[cursor:]` whose every
    `depends_on` index is already completed (strictly before `cursor`) —
    plans are capped at `max_plan_steps` (default 4), so a contiguous-prefix
    scan is enough and no general dependency-graph scheduler is needed. A
    step depending on another step in the *same* batch is deliberately not
    included: batches run concurrently via `asyncio.gather`, so same-batch
    dependencies would not actually be satisfied yet.
    """
    batch = []
    for task in plan[cursor:]:
        if any(d >= cursor for d in task.depends_on):
            break
        batch.append(task)
    return batch


def _deadline_exceeded(state: AgentState) -> bool:
    return time.time() > state["deadline_ts"]


def _summarize_timeline(deadlines: list[ObligationDeadline]) -> str:
    if not deadlines:
        return "No obligations computed."
    parts = []
    for o in deadlines:
        when = f"due {o.deadline}" + (" [OVERDUE]" if o.overdue else "") if o.deadline else (
            "no fixed deadline yet"
        )
        parts.append(f"{o.obligation} ({o.article}): {when}")
    return "; ".join(parts)


def _summarize_risk_tier(result: RiskTierResult) -> str:
    parts = [f"tier={result.tier.value}", f"confidence={result.confidence:.2f}"]
    if result.triggering_criteria:
        criteria = "; ".join(
            f"{criterion.article}: {criterion.description}"
            for criterion in result.triggering_criteria
        )
        parts.append(f"criteria={criteria}")
    if result.unresolved_features:
        parts.append(f"unresolved={' | '.join(result.unresolved_features)}")
    return ", ".join(parts)


def _risk_dependency(subtask: SubTask, state: AgentState) -> RiskTierResult | None:
    for call in state["tool_calls"]:
        if call.step_id in subtask.depends_on and isinstance(call.result, RiskTierResult):
            return call.result
    return None


def _high_risk_basis(result: RiskTierResult) -> HighRiskBasis | None:
    articles = [criterion.article for criterion in result.triggering_criteria]
    if any(article.startswith("Art. 6(1)") for article in articles):
        return HighRiskBasis.ANNEX_I
    if any(
        article.startswith("Annex III") or article.startswith("Art. 6(2)")
        for article in articles
    ):
        return HighRiskBasis.ANNEX_III
    return None


def _prohibited_practice(result: RiskTierResult) -> ProhibitedPractice | None:
    articles = {criterion.article for criterion in result.triggering_criteria}
    if "Art. 5(1)(ba)" in articles:
        return ProhibitedPractice.NON_CONSENSUAL_INTIMATE_IMAGERY
    if "Art. 5(1)(bb)" in articles:
        return ProhibitedPractice.CSAM_ADJACENT_CONTENT
    return None


async def _run_rag_search(subtask: SubTask, ctx: AgentContext) -> _StepResult:
    k = ctx.settings.rag_default_k
    result = await ctx.tools["rag_search"].ainvoke({"query": subtask.query, "k": k})
    assert isinstance(result, RagResult)
    record = ToolCallRecord(
        step_id=subtask.step_id,
        tool="rag_search",
        input_summary=subtask.query,
        result=result,
        summary=f"Retrieved {len(result.evidence)} passage(s).",
        ok=True,
    )
    return list(result.evidence), record


async def _run_classify_risk_tier(subtask: SubTask, ctx: AgentContext) -> _StepResult:
    result = await ctx.tools["classify_risk_tier"].ainvoke({"system_description": subtask.query})
    assert isinstance(result, RiskTierResult)
    record = ToolCallRecord(
        step_id=subtask.step_id,
        tool="classify_risk_tier",
        input_summary=subtask.query,
        result=result,
        summary=_summarize_risk_tier(result),
        ok=True,
    )
    return [], record


async def _run_compliance_timeline(
    subtask: SubTask, state: AgentState, ctx: AgentContext
) -> _StepResult:
    dependency_summary = "\n".join(
        f"{tc.tool}: {tc.summary}" for tc in state["tool_calls"] if tc.step_id in subtask.depends_on
    ) or None
    prompt = build_compliance_timeline_extraction_prompt(
        context=state["normalized_query"],
        instruction=subtask.query,
        dependency_summary=dependency_summary,
    )
    try:
        args = await ctx.chat_model.with_structured_output(ComplianceTimelineArgs).ainvoke(
            [HumanMessage(content=prompt)]
        )
    except ValidationError:
        return [], ToolCallRecord(
            step_id=subtask.step_id,
            tool="compliance_timeline",
            input_summary=subtask.query,
            result=None,
            summary="Could not extract the arguments needed to compute a compliance timeline.",
            ok=False,
        )
    assert isinstance(args, ComplianceTimelineArgs)

    dependency = _risk_dependency(subtask, state)
    if dependency is not None:
        overrides: dict[str, object] = {"tier": dependency.tier}
        if dependency.tier == RiskTier.HIGH_RISK:
            overrides["high_risk_basis"] = _high_risk_basis(dependency)
        elif dependency.tier == RiskTier.PROHIBITED:
            overrides["prohibited_practice"] = _prohibited_practice(dependency)
        args = args.model_copy(update=overrides)

    try:
        result = await ctx.tools["compliance_timeline"].ainvoke(args.model_dump())
    except ValueError as exc:
        return [], ToolCallRecord(
            step_id=subtask.step_id,
            tool="compliance_timeline",
            input_summary=repr(args),
            result=None,
            summary=str(exc),
            ok=False,
        )
    assert isinstance(result, list)

    record = ToolCallRecord(
        step_id=subtask.step_id,
        tool="compliance_timeline",
        input_summary=repr(args),
        result=result,
        summary=_summarize_timeline(result),
        ok=True,
    )
    return [], record


async def _run_step(subtask: SubTask, state: AgentState, ctx: AgentContext) -> _StepResult:
    try:
        if subtask.tool == "rag_search":
            return await _run_rag_search(subtask, ctx)
        if subtask.tool == "classify_risk_tier":
            return await _run_classify_risk_tier(subtask, ctx)
        return await _run_compliance_timeline(subtask, state, ctx)
    except Exception as exc:
        # A tool/LLM call is a system boundary (network, external process);
        # any failure here degrades to one failed step, not a crashed run.
        return [], ToolCallRecord(
            step_id=subtask.step_id,
            tool=subtask.tool,
            input_summary=subtask.query,
            result=None,
            summary=f"Step failed: {exc}",
            ok=False,
        )


async def execute(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, object]:
    ctx = runtime.context
    plan = state["plan"] or _implicit_plan(state)

    if _deadline_exceeded(state):
        return {
            "plan": plan,
            "partial": True,
            "trace": [NodeEvent(node="execute", detail="deadline exceeded — stopping")],
        }

    batch = _ready_batch(plan, state["cursor"])
    results = await asyncio.gather(*(_run_step(task, state, ctx) for task in batch))

    new_evidence: list[Evidence] = []
    new_tool_calls: list[ToolCallRecord] = []
    trace: list[NodeEvent] = []
    for evidence, record in results:
        new_evidence.extend(evidence)
        new_tool_calls.append(record)
        status = "ok" if record.ok else "failed"
        detail = f"step {record.step_id} ({record.tool}): {status}"
        trace.append(NodeEvent(node="execute", detail=detail))

    return {
        "plan": plan,
        "cursor": state["cursor"] + len(batch),
        "evidence": new_evidence,
        "tool_calls": new_tool_calls,
        "trace": trace,
    }


def route_after_execute(state: AgentState) -> Literal["execute", "synthesize", "finalize"]:
    if state["partial"]:
        return "finalize"
    if state["cursor"] >= len(state["plan"]):
        return "synthesize"
    if not _ready_batch(state["plan"], state["cursor"]):
        # Remaining steps have an unsatisfiable dependency (a planner bug) -
        # stop and synthesize over whatever evidence exists rather than
        # looping forever; `recursion_limit` is the last-resort backstop.
        return "synthesize"
    return "execute"
