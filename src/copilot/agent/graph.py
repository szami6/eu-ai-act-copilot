"""The Phase-3 orchestrator graph (PLAN.md §3.1): 6 async nodes, both
conditional routes, bounded loops, concurrent plan steps, per-turn
checkpointing.

    triage --refuse--> finalize --> END
           --direct---> execute
           --plan-----> planner --> execute
    execute <--loop (route_after_execute)--> execute
           --> synthesize --> verify --retry--> execute
                                     --------> finalize --> END

`build_agent_graph` wires the topology; `run_agent`/`stream_agent` are the
two places a caller (a test, the eval harness, Phase 4's API) crosses into
it — mirroring `rag.graph.run_rag`'s role for the RAG subgraph, plus a
streaming counterpart for PLAN §11's "streaming events" (Phase 4's SSE
`/chat` endpoint is the eventual consumer, not built yet).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any, cast
from uuid import uuid4

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from copilot.agent.context import AgentContext
from copilot.agent.nodes.execute import execute, route_after_execute
from copilot.agent.nodes.finalize import finalize
from copilot.agent.nodes.planner import planner
from copilot.agent.nodes.synthesize import synthesize
from copilot.agent.nodes.triage import route_after_triage, triage
from copilot.agent.nodes.verify import route_after_verify, verify
from copilot.agent.state import AgentState

_AgentGraph = CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]


def build_agent_graph() -> _AgentGraph:
    """`checkpointer=MemorySaver()` is the dev/test default (ships in core
    langgraph, no extra dependency) — swap to `SqliteSaver` (a mounted
    volume) at compose time for durability across API restarts, which the
    RAG subgraph doesn't need since it has no loops to resume mid-way
    through (PLAN §3.4: "MemorySaver in dev, SqliteSaver (volume) in
    compose").
    """
    graph = StateGraph(AgentState, context_schema=AgentContext)
    graph.add_node("triage", triage)
    graph.add_node("planner", planner)
    graph.add_node("execute", execute)
    graph.add_node("synthesize", synthesize)
    graph.add_node("verify", verify)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("triage")
    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {"finalize": "finalize", "execute": "execute", "planner": "planner"},
    )
    graph.add_edge("planner", "execute")
    graph.add_conditional_edges(
        "execute",
        route_after_execute,
        {"execute": "execute", "synthesize": "synthesize", "finalize": "finalize"},
    )
    graph.add_edge("synthesize", "verify")
    graph.add_conditional_edges(
        "verify", route_after_verify, {"execute": "execute", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=MemorySaver())


def _seed_state(
    *, session_id: str, history: list[AnyMessage], user_input: str, context: AgentContext
) -> AgentState:
    return {
        "messages": [*history, HumanMessage(content=user_input)],
        "session_id": session_id,
        "route": "direct",
        "normalized_query": "",
        "refusal_reason": None,
        "plan": [],
        "cursor": 0,
        "evidence": [],
        "tool_calls": [],
        "draft": None,
        "groundedness": None,
        "verify_retries": 0,
        "deadline_ts": time.time() + context.settings.turn_deadline_s,
        "partial": False,
        "trace": [],
    }


def _run_config(session_id: str, context: AgentContext) -> RunnableConfig:
    # A fresh thread per turn — see `run_agent`'s docstring below for why.
    return RunnableConfig(
        configurable={"thread_id": f"{session_id}:{uuid4()}"},
        recursion_limit=context.settings.recursion_limit,
    )


async def run_agent(
    compiled: _AgentGraph,
    *,
    session_id: str,
    history: list[AnyMessage],
    user_input: str,
    context: AgentContext,
) -> AgentState:
    """Runs exactly one turn to completion. `history` is the full prior
    conversation — Phase 4's API layer owns storing/retrieving it, not the
    graph — and each turn gets a fresh checkpointer thread (`session_id`
    plus a turn-local suffix) so the `operator.add`-reduced per-turn fields
    (`evidence`, `tool_calls`, `trace`) start empty instead of accumulating
    across turns indefinitely: a thread only needs to survive resuming
    *this* turn after an interruption, not the whole conversation.
    """
    initial = _seed_state(
        session_id=session_id, history=history, user_input=user_input, context=context
    )
    result = await compiled.ainvoke(
        initial, context=context, config=_run_config(session_id, context)
    )
    return cast("AgentState", result)


async def stream_agent(
    compiled: _AgentGraph,
    *,
    session_id: str,
    history: list[AnyMessage],
    user_input: str,
    context: AgentContext,
) -> AsyncIterator[tuple[str, Any]]:
    """Streaming counterpart to `run_agent` (PLAN §11's "streaming events"):
    yields `(mode, data)` pairs exactly as LangGraph's own `astream` does —
    `"updates"` (one `{node_name: partial_state_update}` per finished node,
    the UI's node timeline, PLAN §6) interleaved with `"messages"`
    (`(token, metadata)` pairs, so `synthesize`'s draft can be shown
    word-by-word). A thin pass-through rather than a bespoke event schema:
    turning LangGraph's already-well-defined event shape into SSE frames is
    Phase 4's job, not this one's.
    """
    initial = _seed_state(
        session_id=session_id, history=history, user_input=user_input, context=context
    )
    async for mode, data in compiled.astream(
        initial,
        context=context,
        config=_run_config(session_id, context),
        stream_mode=["updates", "messages"],
    ):
        yield mode, data
