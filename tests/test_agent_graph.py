"""End-to-end integration test for the Phase-3 orchestrator graph (PLAN.md
§3.1): the full triage -> planner -> execute (x2, dependency-ordered) ->
synthesize -> verify -> finalize path, driven entirely by `DummyChatModel`'s
module-level default fixtures (`copilot.llm._FIXTURES`, unmodified — no
per-test monkeypatching here) so it needs no live Qdrant/GPU. Those
defaults deliberately route the default plan through
classify_risk_tier -> compliance_timeline and never rag_search, for exactly
this reason (see the comments next to each fixture in `llm.py`).
"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import AIMessage

from copilot.agent.context import AgentContext
from copilot.agent.graph import build_agent_graph, run_agent, stream_agent
from copilot.agent.state import SubTask, ToolCallRecord
from copilot.agent.tools.bindings import make_tools
from copilot.config import Settings
from copilot.llm import DummyChatModel


def _context() -> AgentContext:
    chat_model = DummyChatModel()
    return AgentContext(
        chat_model=chat_model,
        tools=make_tools(chat_model, cast(Any, None), cast(Any, None)),
        settings=Settings(_env_file=None),
    )


async def test_full_graph_runs_composite_plan_end_to_end() -> None:
    compiled = build_agent_graph()

    final_state = await run_agent(
        compiled,
        session_id="test-session",
        history=[],
        user_input="Is our CV screener high-risk and what are our obligations?",
        context=_context(),
    )

    assert final_state["route"] == "plan"

    plan = cast(list[SubTask], final_state["plan"])
    tool_calls = cast(list[ToolCallRecord], final_state["tool_calls"])
    assert [tc.tool for tc in tool_calls] == ["classify_risk_tier", "compliance_timeline"]
    assert all(tc.ok for tc in tool_calls)
    assert final_state["cursor"] == len(plan) == 2
    assert final_state["partial"] is False

    last_message = final_state["messages"][-1]
    assert isinstance(last_message, AIMessage)
    assert "decision support" in str(last_message.content)


async def test_stream_agent_yields_update_events_for_every_node() -> None:
    """Smoke test for PLAN §11's "streaming events": wiring mistakes (a bad
    `stream_mode` argument, a broken config) would surface here even though
    no test asserts on token-level `"messages"` output, since `DummyChatModel`
    doesn't stream tokens incrementally.
    """
    compiled = build_agent_graph()

    seen_nodes: list[str] = []
    modes: set[str] = set()
    async for mode, data in stream_agent(
        compiled,
        session_id="test-session-stream",
        history=[],
        user_input="Is our CV screener high-risk and what are our obligations?",
        context=_context(),
    ):
        modes.add(mode)
        if mode == "updates":
            seen_nodes.extend(cast(dict[str, Any], data).keys())

    assert "updates" in modes
    assert "finalize" in seen_nodes
    assert seen_nodes.index("triage") < seen_nodes.index("finalize")
