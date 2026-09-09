"""Contract tests for the Phase-5 end-to-end evaluation harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from copilot.config import LLMProvider, Settings
from evals.cases import load_cases
from evals.generate import require_model_endpoint
from evals.judge import aggregate, citation_resolvability, judge_case, validate_generation


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


def test_citation_resolvability_is_zero_without_citations() -> None:
    assert citation_resolvability({"output": {"messages": [{"content": "No citation."}]}}) == 0.0


def test_citation_resolvability_supports_annex_citations() -> None:
    record = {
        "output": {
            "messages": [{"content": "Recruitment systems are covered. [Annex III, point 4]"}],
            "evidence": [{"chunk_id": "ai_act:annex:III:4", "text": "Annex III point 4"}],
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


def test_failed_case_cannot_receive_positive_quality_scores() -> None:
    case = load_cases(Path("data/eval/qa_set.yaml"))[0]
    record = {
        "error": "OpenAIConnectionError: Connection error.",
        "output": {"route": None, "messages": [], "evidence": [], "tool_calls": []},
        "latency_ms": 10.0,
    }

    score = judge_case(record, case)
    summary = aggregate([score])

    assert score["refusal_correct"] is False
    assert score["must_not_include"] is False
    assert score["citation_resolvability"] == 0.0
    assert summary["error_free_rate"] == 0.0
    assert summary["refusal_accuracy"] == 0.0
    assert summary["citation_resolvability"] == 0.0


def test_validate_generation_rejects_contract_drift_and_missing_cases() -> None:
    cases = load_cases(Path("data/eval/qa_set.yaml"))
    payload = {
        "cases": [
            {"id": case.id, "expected": case.model_dump(mode="json")} for case in cases
        ]
    }
    payload["cases"][0]["expected"]["must_include"] = ["changed"]

    with pytest.raises(ValueError, match="contract changed"):
        validate_generation(payload, cases)

    partial = {"cases": payload["cases"][1:]}
    with pytest.raises(ValueError, match="incomplete"):
        validate_generation(partial, cases)


def test_model_preflight_skips_dummy_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("evals.generate.httpx.get", lambda *args, **kwargs: pytest.fail())

    require_model_endpoint(Settings(_env_file=None, llm_provider=LLMProvider.DUMMY))
