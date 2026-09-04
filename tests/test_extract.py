"""Unit tests for `agent.tools.extract.extract_system_features` (PLAN.md
§3.3 Tool 2's free-text -> `SystemFeatures` extraction, the Phase-3 half of
the split `test_risk_tier.py` covers the other half of).
"""

from __future__ import annotations

import re

import pytest

from copilot import llm as llm_module
from copilot.agent.tools.extract import extract_system_features
from copilot.agent.tools.risk_tier import AnnexIIIDomain, SystemFeatures
from copilot.llm import DummyChatModel


def test_extract_system_features_parses_llm_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"feature vector"),
                '{"prohibited_practices": [], "claimed_exceptions": [], '
                '"is_safety_component_of_regulated_product": false, '
                '"requires_third_party_conformity_assessment": false, '
                '"is_solely_non_safety_convenience_feature": false, "safety_relevant": false, '
                '"annex_iii_domains": ["employment_worker_management"], '
                '"performs_profiling_of_natural_persons": false, "derogation_grounds": [], '
                '"transparency_triggers": []}',
            )
        ],
    )

    features = extract_system_features("A CV screening tool.", DummyChatModel())

    assert features.annex_iii_domains == {AnnexIIIDomain.EMPLOYMENT_WORKER_MANAGEMENT}


def test_extract_system_features_falls_back_to_empty_on_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"feature vector"),
                # derogation_grounds without annex_iii_domains: SystemFeatures'
                # own model_validator raises, wrapped by pydantic into the
                # ValidationError extract_system_features must catch.
                '{"prohibited_practices": [], "claimed_exceptions": [], '
                '"is_safety_component_of_regulated_product": false, '
                '"requires_third_party_conformity_assessment": false, '
                '"is_solely_non_safety_convenience_feature": false, "safety_relevant": false, '
                '"annex_iii_domains": [], "performs_profiling_of_natural_persons": false, '
                '"derogation_grounds": ["narrow_procedural_task"], "transparency_triggers": []}',
            )
        ],
    )

    features = extract_system_features("An ambiguous description.", DummyChatModel())

    assert features == SystemFeatures()
