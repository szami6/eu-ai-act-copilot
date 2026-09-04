"""Orchestrator state and cross-node schemas (PLAN.md §3.4).

`AgentState` is the literal shape from §3.4: reducers (`add_messages`,
`operator.add`) rather than mutation keep every node an independently
testable pure `State -> PartialState` function, and `evidence`/`tool_calls`
being append-only means no node can silently overwrite another's findings —
the provenance chain (node -> tool -> chunk -> article) survives to
`finalize`.

`Evidence` is imported from `rag.state`, not redefined here — §3.4's own
docstring already flags it as shared and says the orchestrator imports it
from the subgraph that produces it, never the reverse.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from copilot.rag.state import Evidence

ToolName = Literal["rag_search", "classify_risk_tier", "compliance_timeline"]


class SubTask(BaseModel):
    """One decomposed step from `planner` (PLAN §3.1): a target tool, a
    focused query, and `depends_on` so `execute` can run independent steps
    concurrently.

    `query` is free text in every case — a search query for `rag_search`, a
    system description for `classify_risk_tier` — even for
    `compliance_timeline`, whose actual typed arguments `execute` extracts
    separately (§3.3: that tool takes no free-text argument at all); here it
    is just the planner's restatement of what that step needs to establish,
    used as extraction context.
    """

    step_id: int
    tool: ToolName
    query: str
    depends_on: list[int] = Field(default_factory=list)


@dataclass
class ToolCallRecord:
    """One completed tool invocation — the audit trail `finalize` renders as
    a "tool card" (PLAN §6) and `execute` reads back to feed a dependent
    step's argument extraction (e.g. a `compliance_timeline` step depending
    on an earlier `classify_risk_tier` step)."""

    step_id: int
    tool: ToolName
    input_summary: str
    result: object
    summary: str
    ok: bool


@dataclass
class NodeEvent:
    """One line of the trace the UI's node timeline renders (PLAN §6),
    e.g. `node="triage", detail="-> plan"` or
    `node="verify", detail="-> execute (groundedness 0.41 < 0.70)"`."""

    node: str
    detail: str


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    session_id: str

    route: Literal["refuse", "direct", "plan"]
    normalized_query: str
    refusal_reason: str | None

    plan: list[SubTask]
    cursor: int
    evidence: Annotated[list[Evidence], operator.add]
    tool_calls: Annotated[list[ToolCallRecord], operator.add]

    draft: str | None
    groundedness: float | None
    verify_retries: int

    deadline_ts: float
    partial: bool
    trace: Annotated[list[NodeEvent], operator.add]
