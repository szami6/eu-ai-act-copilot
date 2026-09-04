"""`planner` (PLAN.md §3.1, node 2): decomposes a 'plan'-routed query into
`SubTask`s. Runs only on the 'plan' route — 'direct' skips straight to
`execute`, which seeds an implicit single-step plan itself (see
`nodes/execute.py`).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from copilot.agent.context import AgentContext
from copilot.agent.prompts.planner import PlannerOutput, build_planner_prompt
from copilot.agent.state import AgentState, NodeEvent, SubTask


def _normalize(subtasks: list[SubTask], max_steps: int) -> list[SubTask]:
    """Re-derives `step_id` from list position and drops any `depends_on`
    reference that isn't a strictly-earlier step. `execute`'s dependency
    check assumes `plan[i].step_id == i`; trusting the model's own numbering
    verbatim would let a mislabelled or forward-referencing plan stall
    `execute` forever instead of just running with the bad edge dropped.
    """
    truncated = subtasks[:max_steps]
    normalized = []
    for i, task in enumerate(truncated):
        valid_deps = [d for d in task.depends_on if 0 <= d < i]
        normalized.append(task.model_copy(update={"step_id": i, "depends_on": valid_deps}))
    return normalized


async def planner(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, object]:
    ctx = runtime.context
    max_steps = ctx.settings.max_plan_steps

    structured = ctx.chat_model.with_structured_output(PlannerOutput)
    output = await structured.ainvoke(
        [HumanMessage(content=build_planner_prompt(state["normalized_query"], max_steps))]
    )
    assert isinstance(output, PlannerOutput)

    plan = _normalize(output.subtasks, max_steps)
    return {
        "plan": plan,
        "cursor": 0,
        "trace": [NodeEvent(node="planner", detail=f"{len(plan)} step(s)")],
    }
