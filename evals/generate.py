"""Run the end-to-end evaluation set and persist every turn.

This is deliberately a separate phase from judging. A generation run is
expensive and may use the small serving model; the saved JSON can therefore
be re-judged repeatedly with a larger model or a changed rubric.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from copilot.agent.context import AgentContext
from copilot.agent.graph import _AgentGraph, run_agent
from copilot.api.runtime import build_agent_context, build_app_runtime
from copilot.api.serialize import to_jsonable
from copilot.config import get_settings
from evals.cases import EvalCase, load_cases


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"


def record_from_state(
    case: EvalCase, state: Mapping[str, Any], latency_ms: float, error: str | None = None
) -> dict[str, Any]:
    return {
        "id": case.id,
        "question": case.question,
        "expected": case.model_dump(mode="json"),
        "latency_ms": round(latency_ms, 1),
        "error": error,
        "output": to_jsonable(state),
    }


async def generate_cases(
    cases: list[EvalCase], compiled: _AgentGraph, context: AgentContext
) -> list[dict[str, Any]]:
    """Run cases sequentially so the persisted latency is easy to interpret."""
    records: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        try:
            state = await run_agent(
                compiled,
                session_id=f"eval-{case.id}",
                history=[],
                user_input=case.question,
                context=context,
            )
            # The branch above is replaced below; keeping the invocation
            # explicit makes the graph/context boundary visible to callers.
            records.append(record_from_state(case, state, (time.perf_counter() - started) * 1000))
        except Exception as exc:  # one bad case must not discard 19 useful outputs
            records.append(
                record_from_state(
                    case,
                    {"route": None, "messages": [], "tool_calls": [], "evidence": []},
                    (time.perf_counter() - started) * 1000,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/eval/qa_set.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases")
    args = parser.parse_args()

    settings = get_settings()
    cases = load_cases(args.input)
    if args.limit:
        cases = cases[: args.limit]
    runtime = build_app_runtime(settings)
    context = build_agent_context(runtime, settings)
    records = asyncio.run(generate_cases(cases, runtime.agent_compiled, context))
    output = args.output or Path("evals/reports") / f"generation_{git_sha()}.json"
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "settings": {
            "llm_provider": settings.llm_provider.value,
            "llm_model": settings.llm_model,
            "rerank_enabled": settings.rerank_enabled,
        },
        "cases": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {len(records)} cases -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
