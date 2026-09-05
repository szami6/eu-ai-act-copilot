"""Unit tests for `api.serialize.to_jsonable` (PLAN.md §2.1's SSE frames):
every shape `AgentState` and `ToolCallRecord.result` can actually hold —
plain dataclasses, Pydantic models (including one nested inside a
dataclass, mirroring a tool call's `result`), enums, dates, and LangChain
messages — must round-trip through `json.dumps` with no `TypeError`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from enum import Enum

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from copilot.agent.state import NodeEvent, SubTask, ToolCallRecord
from copilot.agent.tools.risk_tier import RiskTier, RiskTierResult, TriggeringCriterion
from copilot.api.serialize import to_jsonable


def test_plain_values_pass_through() -> None:
    assert to_jsonable("x") == "x"
    assert to_jsonable(3) == 3
    assert to_jsonable(None) is None
    assert to_jsonable(True) is True


def test_dataclass_converts_to_dict() -> None:
    event = NodeEvent(node="triage", detail="-> plan")
    assert to_jsonable(event) == {"node": "triage", "detail": "-> plan"}


def test_pydantic_model_converts_to_dict() -> None:
    subtask = SubTask(step_id=0, tool="rag_search", query="q", depends_on=[])
    assert to_jsonable(subtask) == {
        "step_id": 0,
        "tool": "rag_search",
        "query": "q",
        "depends_on": [],
    }


def test_enum_converts_to_its_value() -> None:
    class Color(Enum):
        RED = "red"

    assert to_jsonable(Color.RED) == "red"


def test_date_converts_to_isoformat() -> None:
    assert to_jsonable(date(2027, 1, 1)) == "2027-01-01"


def test_langchain_message_converts_to_type_and_content() -> None:
    message = AIMessage(content="hello")
    assert to_jsonable(message) == {"type": "ai", "content": "hello"}


def test_dataclass_with_nested_pydantic_result_is_fully_jsonable() -> None:
    """Mirrors `ToolCallRecord.result` holding a `RiskTierResult` — the
    shape `execute.py` actually produces for a `classify_risk_tier` step."""
    result = RiskTierResult(
        tier=RiskTier.HIGH_RISK,
        label="High-risk",
        triggering_criteria=[TriggeringCriterion(article="Art. 6(2)", description="d")],
        confidence=0.9,
    )
    record = ToolCallRecord(
        step_id=0,
        tool="classify_risk_tier",
        input_summary="a CV screener",
        result=result,
        summary="tier=high_risk",
        ok=True,
    )

    jsonable = to_jsonable(record)

    assert jsonable["result"]["tier"] == "high_risk"
    assert jsonable["result"]["triggering_criteria"][0]["article"] == "Art. 6(2)"
    json.dumps(jsonable)  # must not raise


def test_nested_dict_and_list_recurse() -> None:
    @dataclass
    class Wrapper:
        items: list[NodeEvent]

    wrapper = Wrapper(items=[NodeEvent(node="a", detail="1"), NodeEvent(node="b", detail="2")])
    assert to_jsonable(wrapper) == {
        "items": [{"node": "a", "detail": "1"}, {"node": "b", "detail": "2"}]
    }


def test_result_none_passes_through() -> None:
    record = ToolCallRecord(
        step_id=0, tool="compliance_timeline", input_summary="x", result=None, summary="s", ok=False
    )
    assert to_jsonable(record)["result"] is None


def test_pydantic_and_dataclass_both_reachable_through_plain_object() -> None:
    class Inner(BaseModel):
        value: int

    @dataclass
    class Outer:
        inner: Inner

    assert to_jsonable(Outer(inner=Inner(value=1))) == {"inner": {"value": 1}}
