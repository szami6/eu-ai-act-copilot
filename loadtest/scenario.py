"""Dependency-light helpers shared by the Locust load test and its tests."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.cases import load_cases


@dataclass(frozen=True)
class LoadCase:
    case_id: str
    category: str
    question: str


def load_load_cases(path: Path = Path("data/eval/qa_set.yaml")) -> list[LoadCase]:
    return [LoadCase(case.id, case.category, case.question) for case in load_cases(path)]


def parse_sse(lines: Iterable[str | bytes]) -> list[tuple[str, dict[str, Any]]]:
    """Parse an SSE stream, tolerating blank lines and byte responses."""
    events: list[tuple[str, dict[str, Any]]] = []
    event_name: str | None = None
    data_lines: list[str] = []
    for raw_line in lines:
        text = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        for line in text.splitlines():
            if not line:
                if event_name is not None and data_lines:
                    events.append((event_name, json.loads("\n".join(data_lines))))
                event_name = None
                data_lines = []
                continue
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())
    if event_name is not None and data_lines:
        events.append((event_name, json.loads("\n".join(data_lines))))
    return events


def consume_chat_stream(lines: Iterable[str | bytes]) -> dict[str, Any]:
    """Summarise one `/chat` response for Locust's success/failure decision."""
    events = parse_sse(lines)
    names = [name for name, _ in events]
    errors = [
        data.get("detail", "unknown stream error")
        for name, data in events
        if name == "error"
    ]
    done = next((data for name, data in reversed(events) if name == "done"), None)
    session = next((data for name, data in events if name == "session"), None)
    node_durations: dict[str, float] = {}
    for name, data in events:
        if name == "node":
            node = str(data.get("node", "unknown"))
            node_durations[node] = node_durations.get(node, 0.0) + float(
                data.get("duration_s", 0.0)
            )
    return {
        "events": len(events),
        "event_names": names,
        "session_id": session.get("session_id") if session else None,
        "node_count": names.count("node"),
        "token_count": names.count("token"),
        "done": done is not None,
        "error": errors[0] if errors else None,
        "done_payload": done,
        "node_durations_s": node_durations,
    }


def percentile(values: Iterable[float], quantile: float) -> float:
    """Linear-interpolated percentile with an explicit empty-input result."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_latencies(latencies_ms: Iterable[float]) -> dict[str, float]:
    values = list(latencies_ms)
    return {
        "count": float(len(values)),
        "mean_ms": sum(values) / len(values) if values else 0.0,
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
    }
