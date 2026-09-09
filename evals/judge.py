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

_CITATION_RE = re.compile(r"\[((?:Art\.|Article|Annex)\s+[^\]]+)\]", re.IGNORECASE)
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
        return 0.0
    evidence = record.get("output", {}).get("evidence", [])
    evidence_text = "\n".join(
        f"{item.get('chunk_id', '')} {item.get('text', '')}" for item in evidence
    )
    tool_text = json.dumps(
        record.get("output", {}).get("tool_calls", []), ensure_ascii=False
    )
    haystack = f"{evidence_text}\n{tool_text}"
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


def _gold_evidence_recall(record: dict[str, Any], case: EvalCase) -> float:
    if not case.gold_chunk_ids:
        return 1.0
    returned = {
        str(item.get("chunk_id")) for item in record.get("output", {}).get("evidence", [])
    }
    return len(returned & set(case.gold_chunk_ids)) / len(set(case.gold_chunk_ids))


def judge_case(record: dict[str, Any], case: EvalCase) -> dict[str, Any]:
    """Apply all deterministic checks to one persisted generation record."""
    output = record.get("output", {})
    answer = _answer(record)
    route = output.get("route")
    error_free = not bool(record.get("error"))
    citation_expected = bool(case.gold_articles)
    evidence_expected = case.category in {"single_hop_factual", "multi_hop_composite"}
    refusal_predicted = route == "refuse"
    return {
        "id": case.id,
        "error_free": error_free,
        "route_correct": error_free and route == case.expected_route,
        "refusal_expected": case.expected_refusal,
        "refusal_correct": error_free and refusal_predicted == case.expected_refusal,
        "abstention_expected": case.expected_abstention,
        "abstention_signal": (
            error_free
            and (
                (not case.expected_abstention)
                or any(
                marker in answer.casefold()
                for marker in ("not enough", "cannot determine", "insufficient evidence")
                )
            )
        ),
        "must_include": error_free and _contains_all(answer, case.must_include),
        "must_not_include": error_free
        and not any(
            phrase.casefold() in answer.casefold() for phrase in case.must_not_include
        ),
        "citation_expected": citation_expected,
        "citation_present": error_free
        and ((not citation_expected) or bool(_CITATION_RE.search(answer))),
        "citation_resolvability": (
            round(citation_resolvability(record), 4) if error_free and citation_expected else 0.0
        ),
        "evidence_expected": evidence_expected,
        "gold_evidence_recall": (
            round(_gold_evidence_recall(record, case), 4)
            if error_free and evidence_expected
            else 0.0
        ),
        "tool_expected": bool(case.expected_tool),
        "tool_output_exact": error_free and _tool_output_matches(record, case),
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
    citation_scores = [s for s in scores if s["citation_expected"]]
    evidence_scores = [s for s in scores if s["evidence_expected"]]
    refusal_scores = [s for s in scores if s["refusal_expected"]]
    non_refusal_scores = [s for s in scores if not s["refusal_expected"]]
    return {
        "cases": float(len(scores)),
        "error_free_rate": _mean([float(s["error_free"]) for s in scores]),
        "route_accuracy": _mean([float(s["route_correct"]) for s in scores]),
        "refusal_accuracy": _mean([float(s["refusal_correct"]) for s in scores]),
        "refusal_recall": _mean([float(s["refusal_correct"]) for s in refusal_scores]),
        "false_refusal_rate": _mean(
            [float(not s["refusal_correct"]) for s in non_refusal_scores]
        ),
        "abstention_signal_rate": _mean(
            [float(s["abstention_signal"]) for s in abstention_scores]
        ),
        "must_include_rate": _mean([float(s["must_include"]) for s in scores]),
        "must_not_include_rate": _mean([float(s["must_not_include"]) for s in scores]),
        "citation_presence_rate": _mean(
            [float(s["citation_present"]) for s in citation_scores]
        ),
        "citation_resolvability": _mean(
            [s["citation_resolvability"] for s in citation_scores]
        ),
        "gold_evidence_recall": _mean(
            [s["gold_evidence_recall"] for s in evidence_scores]
        ),
        "tool_exact_match_rate": _mean([float(s["tool_output_exact"]) for s in tool_scores]),
        "mean_latency_ms": _mean(
            [float(s["latency_ms"]) for s in scores if s["latency_ms"] is not None]
        ),
    }


def validate_generation(
    payload: dict[str, Any], current_cases: list[EvalCase], *, allow_partial: bool = False
) -> list[tuple[dict[str, Any], EvalCase]]:
    records = payload.get("cases")
    if not isinstance(records, list):
        raise ValueError("Generation payload must contain a cases list")

    current = {case.id: case for case in current_cases}
    record_ids = [str(record.get("id")) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Generation contains duplicate case ids")
    unknown = set(record_ids) - set(current)
    if unknown:
        raise ValueError(f"Generation contains unknown case ids: {sorted(unknown)}")
    missing = set(current) - set(record_ids)
    if missing and not allow_partial:
        raise ValueError(f"Generation is incomplete; missing case ids: {sorted(missing)}")

    validated = []
    for record in records:
        case_id = str(record["id"])
        embedded = EvalCase.model_validate(record.get("expected"))
        if embedded != current[case_id]:
            raise ValueError(f"Evaluation contract changed since generation for case {case_id}")
        validated.append((record, embedded))
    return validated


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
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    input_path = args.input or find_latest_generation(Path("evals/reports"))
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    # Loading the source set also catches a deleted/relabelled case before a
    # report silently scores an incomplete experiment.
    validated = validate_generation(
        payload, load_cases(args.cases), allow_partial=args.allow_partial
    )
    scores = [judge_case(record, case) for record, case in validated]
    report = {
        "input": str(input_path),
        "run_id": payload.get("run_id"),
        "summary": aggregate(scores),
        "cases": scores,
    }
    default_name = input_path.stem.replace("generation_", "eval_") + ".json"
    output = args.output or input_path.with_name(default_name)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {output}")
    return 0 if all(score["error_free"] for score in scores) else 1


if __name__ == "__main__":
    raise SystemExit(main())
