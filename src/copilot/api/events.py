"""Turns `agent.graph.stream_agent`'s raw `(mode, data)` pairs into the SSE
frames `/chat` sends the UI (PLAN.md §2.1, §6): a `session` frame first (so
a fresh conversation learns its server-assigned id), one `node` frame per
finished orchestrator node (timing + the full jsonable state delta, for the
live node timeline / plan / evidence / tool-card panels), a `token` frame
per `synthesize` chunk (word-by-word answer streaming), and a final `done`
frame with the accumulated turn result.

`stream_agent` is deliberately a thin, unopinionated pass-through over
LangGraph's own stream (Phase 3) — this module is where the product's
opinion about what the UI should see actually lives, e.g. filtering
`"messages"`-mode chunks down to just `synthesize`'s (every node's own LLM
call, not only the user-facing draft, shows up in that stream — triage's
route JSON, planner's plan JSON, etc. — and only the draft belongs on
screen as "the answer typing itself out").
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage

from copilot.agent.context import AgentContext
from copilot.agent.graph import stream_agent
from copilot.api.runtime import AppRuntime
from copilot.api.serialize import to_jsonable

logger = logging.getLogger(__name__)

# Keys `AgentState` reduces with `operator.add` (state.py) — the only ones
# an "updates" event *appends* to rather than overwrites. `messages` also
# accumulates, but through `add_messages`'s id-aware semantics; plain
# extend is equivalent here because only `finalize` ever returns a
# `messages` key in this graph, exactly once per turn.
_APPEND_KEYS = frozenset({"evidence", "tool_calls", "trace", "messages"})


def _empty_accumulator() -> dict[str, Any]:
    return {
        "route": None,
        "normalized_query": "",
        "refusal_reason": None,
        "plan": [],
        "cursor": 0,
        "evidence": [],
        "tool_calls": [],
        "draft": None,
        "groundedness": None,
        "verify_retries": 0,
        "partial": False,
        "trace": [],
        "messages": [],
    }


def _merge_update(accum: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if key in _APPEND_KEYS:
            accum.setdefault(key, []).extend(value)
        else:
            accum[key] = value


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def chat_events(
    runtime: AppRuntime,
    sessions: dict[str, list[AnyMessage]],
    *,
    session_id: str,
    history: list[AnyMessage],
    user_input: str,
    agent_context: AgentContext,
) -> AsyncIterator[str]:
    yield _sse("session", {"session_id": session_id})

    accum = _empty_accumulator()
    last_ts = time.monotonic()
    try:
        async for mode, data in stream_agent(
            runtime.agent_compiled,
            session_id=session_id,
            history=history,
            user_input=user_input,
            context=agent_context,
        ):
            if mode == "updates":
                for node_name, node_update in data.items():
                    now = time.monotonic()
                    duration_s = now - last_ts
                    last_ts = now
                    _merge_update(accum, node_update)
                    yield _sse(
                        "node",
                        {
                            "node": node_name,
                            "duration_s": round(duration_s, 3),
                            "update": to_jsonable(node_update),
                        },
                    )
            elif mode == "messages":
                chunk, metadata = data
                if metadata.get("langgraph_node") != "synthesize":
                    continue
                token = str(chunk.content)
                if token:
                    yield _sse("token", {"node": "synthesize", "token": token})
    except Exception as exc:
        logger.exception("Error while streaming a /chat turn for session %s", session_id)
        yield _sse("error", {"detail": str(exc)})
        return

    final_messages = [*history, HumanMessage(content=user_input), *accum["messages"]]
    sessions[session_id] = final_messages

    answer = str(final_messages[-1].content) if final_messages else ""
    yield _sse("done", {**to_jsonable(accum), "answer": answer})
