"""`triage` (PLAN.md §3.1, node 1): the graph's sole entry point. Classifies
the latest turn into refuse/direct/plan and resolves it into a
self-contained `normalized_query` every downstream node reads instead of
raw `messages`, so `planner`/`execute`/`synthesize` never need their own
history-resolution logic.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from copilot.agent.context import AgentContext
from copilot.agent.prompts.triage import Triage, build_triage_prompt
from copilot.agent.state import AgentState, NodeEvent


async def triage(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, object]:
    ctx = runtime.context
    history = state["messages"][:-1]
    latest = str(state["messages"][-1].content) if state["messages"] else ""

    structured = ctx.chat_model.with_structured_output(Triage)
    output = await structured.ainvoke(
        [HumanMessage(content=build_triage_prompt(history, latest))]
    )
    assert isinstance(output, Triage)

    return {
        "route": output.route,
        "normalized_query": output.normalized_query,
        "refusal_reason": output.refusal_reason,
        "trace": [NodeEvent(node="triage", detail=f"-> {output.route}")],
    }


def route_after_triage(state: AgentState) -> Literal["finalize", "execute", "planner"]:
    route = state["route"]
    if route == "refuse":
        return "finalize"
    if route == "direct":
        return "execute"
    return "planner"
