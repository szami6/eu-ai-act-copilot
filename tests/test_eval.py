"""Contract tests for the Phase-5 end-to-end evaluation harness."""

from __future__ import annotations

from pathlib import Path

from evals.cases import load_cases
from evals.judge import aggregate, citation_resolvability, judge_case


def test_qa_set_has_twenty_unique_cases() -> None:
    cases = load_cases(Path("data/eval/qa_set.yaml"))

    assert len(cases) == 20
    assert len({case.id for case in cases}) == 20
    assert sum(case.expected_refusal for case in cases) == 3
    assert sum(case.expected_abstention for case in cases) == 3


def test_citation_resolvability_checks_evidence_locators() -> None:
    record = {
        "output": {
            "messages": [{"content": "Risk management applies. [Art. 9(1)]"}],
            "evidence": [{"chunk_id": "ai_act:art:9", "text": "Article 9(1) risk management"}],
        }
    }

    assert citation_resolvability(record) == 1.0


def test_judge_case_reports_content_and_tool_failures() -> None:
    case = load_cases(Path("data/eval/qa_set.yaml"))[10]
    record = {
        "output": {
            "route": "plan",
            "messages": [{"content": "The system is high-risk."}],
            "evidence": [],
            "tool_calls": [
                {"tool": "classify_risk_tier", "result": {"tier": "minimal_risk"}}
            ],
        },
        "latency_ms": 12.5,
    }

    score = judge_case(record, case)

    assert score["route_correct"] is True
    assert score["must_include"] is True
    assert score["tool_output_exact"] is False
    assert aggregate([score])["cases"] == 1.0
