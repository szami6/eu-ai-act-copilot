"""Unit tests for `api.events.chat_events` (PLAN.md §2.1's SSE frames),
driven directly as an async generator — no HTTP, no live Qdrant. Mirrors
`test_agent_graph.py`'s reliance on `copilot.llm._FIXTURES`' default chain
(classify_risk_tier -> compliance_timeline, never rag_search) so a
hand-built `AppRuntime` can pass `cast(Any, None)` for the Qdrant client,
exactly like `test_bindings.py`/`test_agent_graph.py` already do for the
same reason.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from copilot.agent.graph import build_agent_graph
from copilot.api.events import chat_events
from copilot.api.runtime import AppRuntime, build_agent_context
from copilot.config import Settings
from copilot.llm import DummyChatModel
from copilot.rag.graph import build_rag_graph
from copilot.rag.retrievers import build_bm25_index


def _runtime(settings: Settings | None = None) -> AppRuntime:
    chat_model = DummyChatModel()
    return AppRuntime(
        qdrant_client=cast(Any, None),
        bm25_index=build_bm25_index([]),
        chat_model=chat_model,
        base_settings=settings or Settings(_env_file=None),
        corpus_ready=False,
        agent_compiled=build_agent_graph(),
        rag_compiled=build_rag_graph(),
    )


def _parse_sse(raw: str) -> list[tuple[str, dict[str, Any]]]:
    frames = []
    for block in raw.strip("\n").split("\n\n"):
        if not block:
            continue
        lines = block.splitlines()
        event = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
        data = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
        frames.append((event, json.loads(data)))
    return frames


async def _collect(runtime: AppRuntime, sessions: dict[str, Any], **kwargs: Any) -> list[Any]:
    chunks = [chunk async for chunk in chat_events(runtime, sessions, **kwargs)]
    return _parse_sse("".join(chunks))


async def test_chat_events_emits_session_first_and_done_last() -> None:
    runtime = _runtime()
    agent_context = build_agent_context(runtime, runtime.base_settings)

    frames = await _collect(
        runtime,
        {},
        session_id="s1",
        history=[],
        user_input="Is our CV screener high-risk and what are our obligations?",
        agent_context=agent_context,
    )

    assert frames[0][0] == "session"
    assert frames[0][1] == {"session_id": "s1"}
    assert frames[-1][0] == "done"


async def test_chat_events_emits_one_node_frame_per_orchestrator_node() -> None:
    runtime = _runtime()
    agent_context = build_agent_context(runtime, runtime.base_settings)

    frames = await _collect(
        runtime,
        {},
        session_id="s1",
        history=[],
        user_input="Is our CV screener high-risk and what are our obligations?",
        agent_context=agent_context,
    )

    node_names = [data["node"] for event, data in frames if event == "node"]
    assert node_names == [
        "triage",
        "planner",
        "execute",
        "execute",
        "synthesize",
        "verify",
        "finalize",
    ]
    assert all(isinstance(data["duration_s"], float) for event, data in frames if event == "node")


async def test_chat_events_node_frame_carries_jsonable_tool_call() -> None:
    runtime = _runtime()
    agent_context = build_agent_context(runtime, runtime.base_settings)

    frames = await _collect(
        runtime,
        {},
        session_id="s1",
        history=[],
        user_input="Is our CV screener high-risk and what are our obligations?",
        agent_context=agent_context,
    )

    execute_frames = [
        data for event, data in frames if event == "node" and data["node"] == "execute"
    ]
    first_call = execute_frames[0]["update"]["tool_calls"][0]
    assert first_call["tool"] == "classify_risk_tier"
    assert first_call["result"]["tier"] == "high_risk"


async def test_chat_events_done_frame_has_final_answer_and_full_trace() -> None:
    runtime = _runtime()
    agent_context = build_agent_context(runtime, runtime.base_settings)

    frames = await _collect(
        runtime,
        {},
        session_id="s1",
        history=[],
        user_input="Is our CV screener high-risk and what are our obligations?",
        agent_context=agent_context,
    )

    done = next(data for event, data in frames if event == "done")
    assert "decision support" in done["answer"]
    assert done["partial"] is False
    assert len(done["tool_calls"]) == 2
    assert any(t["node"] == "finalize" for t in done["trace"])


async def test_chat_events_persists_session_history_for_next_turn() -> None:
    runtime = _runtime()
    agent_context = build_agent_context(runtime, runtime.base_settings)
    sessions: dict[str, Any] = {}

    await _collect(
        runtime,
        sessions,
        session_id="s1",
        history=[],
        user_input="Is our CV screener high-risk?",
        agent_context=agent_context,
    )

    assert "s1" in sessions
    assert sessions["s1"][0].content == "Is our CV screener high-risk?"
    assert sessions["s1"][-1].type == "ai"


async def test_chat_events_only_streams_synthesize_tokens() -> None:
    """Every node's own structured-output call shows up on LangGraph's raw
    `"messages"` stream (triage's route JSON, planner's plan JSON, ...) —
    only `synthesize`'s free-text draft should reach the UI as a token."""
    runtime = _runtime()
    agent_context = build_agent_context(runtime, runtime.base_settings)

    frames = await _collect(
        runtime,
        {},
        session_id="s1",
        history=[],
        user_input="Is our CV screener high-risk and what are our obligations?",
        agent_context=agent_context,
    )

    token_frames = [data for event, data in frames if event == "token"]
    assert token_frames
    assert all(t["node"] == "synthesize" for t in token_frames)
    assert "[Art. 9(1)]" in "".join(t["token"] for t in token_frames)


async def test_chat_events_yields_error_frame_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime()
    agent_context = build_agent_context(runtime, runtime.base_settings)

    async def _broken_stream_agent(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr("copilot.api.events.stream_agent", _broken_stream_agent)

    frames = await _collect(
        runtime, {}, session_id="s1", history=[], user_input="hi", agent_context=agent_context
    )

    assert frames[-1] == ("error", {"detail": "boom"})
