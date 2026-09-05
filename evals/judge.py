"""Score persisted evaluation generations without re-running the system.

The default judge is deterministic and suitable for CI. It reports the
checks that can be defended mechanically; a later model-based judge can be
added without changing the generation file or these aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from evals.cases import EvalCase, load_cases

_CITATION_RE = re.compile(r"\[(?:Art\.|Article)\s+([^\]]+)\]", re.IGNORECASE)
_LOCATOR_RE = re.compile(r"[\dIVXLCDM][\w().,\-–]*", re.IGNORECASE)


def _answer(record: dict[str, Any]) -> str:
    if record.get("answer"):
        return str(record["answer"])
    messages = record.get("output", {}).get("messages", [])
    return str(messages[-1].get("content", "")) if messages else ""


def citation_resolvability(record: dict[str, Any]) -> float:
    """Return the fraction of cited article locators present in evidence."""
    citations = _CITATION_RE.findall(_answer(record))
    if not citations:
        return 1.0
    evidence = record.get("output", {}).get("evidence", [])
    haystack = "\n".join(
        f"{item.get('chunk_id', '')} {item.get('text', '')}" for item in evidence
    )
    resolved = 0
    for citation in citations:
        match = _LOCATOR_RE.search(citation)
        if match and match.group(0) in haystack:
            resolved += 1
    return resolved / len(citations)


def _contains_all(answer: str, phrases: list[str]) -> bool:
    lowered = answer.casefold()
    return all(phrase.casefold() in lowered for phrase in phrases)


def _tool_output_matches(record: dict[str, Any], case: EvalCase) -> bool:
    if not case.expected_tool:
        return True
    for call in record.get("output", {}).get("tool_calls", []):
        if call.get("tool") != case.expected_tool:
            continue
        result = call.get("result", {})
        possible_results = result if isinstance(result, list) else [result]
        if any(
            isinstance(item, dict)
            and all(item.get(key) == value for key, value in case.expected_tool_output.items())
            for item in possible_results
        ):
            return True
    return False


def judge_case(record: dict[str, Any], case: EvalCase) -> dict[str, Any]:
    """Apply all deterministic checks to one persisted generation record."""
    output = record.get("output", {})
    answer = _answer(record)
    route = output.get("route")
    return {
        "id": case.id,
        "error_free": not bool(record.get("error")),
        "route_correct": route == case.expected_route,
        "refusal_correct": (not case.expected_refusal) or route == "refuse",
        "abstention_expected": case.expected_abstention,
        "abstention_signal": (
            (not case.expected_abstention)
            or any(
                marker in answer.casefold()
                for marker in ("not enough", "cannot determine", "insufficient evidence")
            )
        ),
        "must_include": _contains_all(answer, case.must_include),
        "must_not_include": not any(
            phrase.casefold() in answer.casefold() for phrase in case.must_not_include
        ),
        "citation_resolvability": round(citation_resolvability(record), 4),
        "tool_expected": bool(case.expected_tool),
        "tool_output_exact": _tool_output_matches(record, case),
        "latency_ms": record.get("latency_ms"),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(scores: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate scores while keeping abstention and refusal visible."""
    if not scores:
        return {}
    abstention_scores = [s for s in scores if s["abstention_expected"]]
    tool_scores = [s for s in scores if s["tool_expected"]]
    return {
        "cases": float(len(scores)),
        "error_free_rate": _mean([float(s["error_free"]) for s in scores]),
        "route_accuracy": _mean([float(s["route_correct"]) for s in scores]),
        "refusal_accuracy": _mean([float(s["refusal_correct"]) for s in scores]),
        "abstention_signal_rate": _mean(
            [float(s["abstention_signal"]) for s in abstention_scores]
        ),
        "must_include_rate": _mean([float(s["must_include"]) for s in scores]),
        "must_not_include_rate": _mean([float(s["must_not_include"]) for s in scores]),
        "citation_resolvability": _mean([s["citation_resolvability"] for s in scores]),
        "tool_exact_match_rate": _mean([float(s["tool_output_exact"]) for s in tool_scores]),
        "mean_latency_ms": _mean(
            [float(s["latency_ms"]) for s in scores if s["latency_ms"] is not None]
        ),
    }


def find_latest_generation(directory: Path) -> Path:
    candidates = sorted(directory.glob("generation_*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No generation_*.json found in {directory}")
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--cases", type=Path, default=Path("data/eval/qa_set.yaml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    input_path = args.input or find_latest_generation(Path("evals/reports"))
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    # Loading the source set also catches a deleted/relabelled case before a
    # report silently scores an incomplete experiment.
    cases = {case.id: case for case in load_cases(args.cases)}
    scores = [
        judge_case(record, cases[record["id"]])
        for record in payload["cases"]
        if record["id"] in cases
    ]
    report = {"input": str(input_path), "summary": aggregate(scores), "cases": scores}
    default_name = input_path.stem.replace("generation_", "eval_") + ".json"
    output = args.output or input_path.with_name(default_name)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
