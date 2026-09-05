"""Recursively converts orchestrator state (dataclasses, Pydantic models,
enums, dates, LangChain messages) into plain JSON-safe values for `/chat`'s
SSE frames (PLAN.md §2.1, §6).

One function rather than a per-type serializer in each module: `AgentState`
mixes `SubTask`/`RiskTierResult`/`ObligationDeadline` (Pydantic),
`ToolCallRecord`/`NodeEvent`/`Evidence` (plain dataclasses), and
`AnyMessage` (LangChain's own Pydantic models) — a `ToolCallRecord.result`
in particular can be any of several tool return types nested inside a
dataclass, so the converter has to recurse generically rather than assume a
shape.
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime
from enum import Enum
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel


def to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        return {"type": value.type, "content": value.content}
    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [to_jsonable(v) for v in value]
    return value
