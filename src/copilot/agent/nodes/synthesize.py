"""`synthesize` (PLAN.md §3.1, node 4): drafts a cited answer from whatever
evidence/tool results `execute` gathered. Free text, no structured-output
schema — `prompts/synthesize.py`'s docstring explains why.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from copilot.agent.context import AgentContext
from copilot.agent.prompts.synthesize import build_synthesize_prompt
from copilot.agent.state import AgentState, NodeEvent


async def synthesize(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, object]:
    ctx = runtime.context
    prompt = build_synthesize_prompt(
        state["normalized_query"], state["evidence"], state["tool_calls"]
    )
    response = await ctx.chat_model.ainvoke([HumanMessage(content=prompt)])
    draft = str(response.content)
    return {
        "draft": draft,
        "trace": [NodeEvent(node="synthesize", detail=f"{len(draft)} chars")],
    }
