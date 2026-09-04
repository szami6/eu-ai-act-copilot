"""Unit tests for the Art. 5/6/50 risk-tier decision engine (PLAN.md §7.2:
"the rule engine is separately unit-tested to 100% on the decision table").

Every branch of `classify_risk_tier` is exercised directly through
`SystemFeatures` — no LLM, no free text, matching the module's own pure-data
contract.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from copilot.agent.tools.risk_tier import (
    AnnexIIIDomain,
    DerogationGround,
    ProhibitedPractice,
    RiskTier,
    SystemFeatures,
    TransparencyTrigger,
    _load_table,
    classify_risk_tier,
)


def test_empty_features_is_minimal_risk_with_low_confidence() -> None:
    result = classify_risk_tier(SystemFeatures())

    assert result.tier == RiskTier.MINIMAL_RISK
    assert result.confidence == pytest.approx(0.2)
    assert result.triggering_criteria == []
    assert len(result.unresolved_features) == 1
    assert "No risk-relevant features" in result.unresolved_features[0]


def test_near_miss_on_annex_i_is_minimal_risk_with_full_confidence() -> None:
    """Distinguishes a genuinely-assessed minimal-risk system (some features
    considered, none matched) from the degenerate empty-input case above —
    only one of Art. 6(1)'s two conjunctive conditions is met."""
    features = SystemFeatures(is_safety_component_of_regulated_product=True)

    result = classify_risk_tier(features)

    assert result.tier == RiskTier.MINIMAL_RISK
    assert result.confidence == pytest.approx(1.0)
    assert result.unresolved_features == []


def test_single_prohibited_practice() -> None:
    features = SystemFeatures(prohibited_practices={ProhibitedPractice.SOCIAL_SCORING})

    result = classify_risk_tier(features)

    assert result.tier == RiskTier.PROHIBITED
    assert result.label == "Prohibited (unacceptable risk)"
    assert result.confidence == pytest.approx(1.0)
    assert result.unresolved_features == []
    assert [c.article for c in result.triggering_criteria] == ["Art. 5(1)(c)"]


def test_multiple_prohibited_practices_sorted_and_all_cited() -> None:
    features = SystemFeatures(
        prohibited_practices={
            ProhibitedPractice.REALTIME_RBI_LAW_ENFORCEMENT_PUBLIC_SPACE,
            ProhibitedPractice.SUBLIMINAL_MANIPULATION,
        }
    )

    result = classify_risk_tier(features)

    assert result.tier == RiskTier.PROHIBITED
    articles = [c.article for c in result.triggering_criteria]
    assert articles == ["Art. 5(1)(a)", "Art. 5(1)(h)"]


def test_claimed_exception_suppresses_prohibition_but_flags_for_review() -> None:
    features = SystemFeatures(
        prohibited_practices={ProhibitedPractice.INDIVIDUAL_CRIME_RISK_PROFILING},
        claimed_exceptions={ProhibitedPractice.INDIVIDUAL_CRIME_RISK_PROFILING},
    )

    result = classify_risk_tier(features)

    assert result.tier == RiskTier.MINIMAL_RISK
    assert result.confidence == pytest.approx(0.7)
    assert len(result.unresolved_features) == 1
    assert "Art. 5(1)(d)" in result.unresolved_features[0]


def test_one_practice_excepted_another_not_still_prohibits_on_the_other() -> None:
    features = SystemFeatures(
        prohibited_practices={
            ProhibitedPractice.SOCIAL_SCORING,
            ProhibitedPractice.EXPLOITS_VULNERABLE_GROUPS,
        },
        claimed_exceptions={ProhibitedPractice.EXPLOITS_VULNERABLE_GROUPS},
    )

    result = classify_risk_tier(features)

    assert result.tier == RiskTier.PROHIBITED
    assert [c.article for c in result.triggering_criteria] == ["Art. 5(1)(c)"]
    assert result.confidence == pytest.approx(0.7)
    assert len(result.unresolved_features) == 1
    assert "Art. 5(1)(b)" in result.unresolved_features[0]


def test_annex_i_safety_component_is_high_risk() -> None:
    features = SystemFeatures(
        is_safety_component_of_regulated_product=True,
        requires_third_party_conformity_assessment=True,
    )

    result = classify_risk_tier(features)

    assert result.tier == RiskTier.HIGH_RISK
    assert result.triggering_criteria[0].article == "Art. 6(1)"
    assert result.confidence == pytest.approx(1.0)


def test_annex_i_convenience_carveout_blocks_high_risk() -> None:
    features = SystemFeatures(
        is_safety_component_of_regulated_product=True,
        requires_third_party_conformity_assessment=True,
        is_solely_non_safety_convenience_feature=True,
    )

    result = classify_risk_tier(features)

    assert result.tier != RiskTier.HIGH_RISK


def test_annex_i_safety_relevant_overrides_convenience_carveout() -> None:
    features = SystemFeatures(
        is_safety_component_of_regulated_product=True,
        requires_third_party_conformity_assessment=True,
        is_solely_non_safety_convenience_feature=True,
        safety_relevant=True,
    )

    result = classify_risk_tier(features)

    assert result.tier == RiskTier.HIGH_RISK


def test_annex_iii_domain_without_derogation_is_high_risk() -> None:
    features = SystemFeatures(annex_iii_domains={AnnexIIIDomain.EMPLOYMENT_WORKER_MANAGEMENT})

    result = classify_risk_tier(features)

    assert result.tier == RiskTier.HIGH_RISK
    assert result.triggering_criteria[0].article == "Annex III, point 4"
    assert result.confidence == pytest.approx(1.0)


def test_annex_iii_multiple_domains_sorted() -> None:
    features = SystemFeatures(
        annex_iii_domains={AnnexIIIDomain.LAW_ENFORCEMENT, AnnexIIIDomain.BIOMETRICS}
    )

    result = classify_risk_tier(features)

    articles = [c.article for c in result.triggering_criteria]
    assert articles == ["Annex III, point 1", "Annex III, point 6"]


def test_annex_iii_derogation_avoids_high_risk() -> None:
    features = SystemFeatures(
        annex_iii_domains={AnnexIIIDomain.EDUCATION_VOCATIONAL_TRAINING},
        derogation_grounds={DerogationGround.PREPARATORY_TASK_ONLY},
    )

    result = classify_risk_tier(features)

    assert result.tier == RiskTier.MINIMAL_RISK
    assert result.confidence == pytest.approx(0.8)
    assert len(result.unresolved_features) == 1
    assert "Art. 6(3)" in result.unresolved_features[0]


def test_annex_iii_derogation_with_transparency_trigger_is_limited_risk() -> None:
    features = SystemFeatures(
        annex_iii_domains={AnnexIIIDomain.EDUCATION_VOCATIONAL_TRAINING},
        derogation_grounds={DerogationGround.PREPARATORY_TASK_ONLY},
        transparency_triggers={TransparencyTrigger.DIRECT_INTERACTION},
    )

    result = classify_risk_tier(features)

    assert result.tier == RiskTier.LIMITED_RISK
    assert result.confidence == pytest.approx(0.8)
    assert result.triggering_criteria[0].article == "Art. 50(1)"
    assert len(result.unresolved_features) == 1


def test_profiling_overrides_derogation_regardless_of_claimed_grounds() -> None:
    features = SystemFeatures(
        annex_iii_domains={AnnexIIIDomain.EDUCATION_VOCATIONAL_TRAINING},
        derogation_grounds={DerogationGround.PREPARATORY_TASK_ONLY},
        performs_profiling_of_natural_persons=True,
    )

    result = classify_risk_tier(features)

    assert result.tier == RiskTier.HIGH_RISK
    assert result.confidence == pytest.approx(1.0)
    assert result.unresolved_features == []


def test_transparency_trigger_alone_is_limited_risk() -> None:
    trigger = TransparencyTrigger.SYNTHETIC_CONTENT_GENERATION
    features = SystemFeatures(transparency_triggers={trigger})

    result = classify_risk_tier(features)

    assert result.tier == RiskTier.LIMITED_RISK
    assert result.triggering_criteria[0].article == "Art. 50(2)"
    assert result.confidence == pytest.approx(1.0)


def test_multiple_transparency_triggers_sorted() -> None:
    features = SystemFeatures(
        transparency_triggers={
            TransparencyTrigger.DEEPFAKE_OR_PUBLIC_INTEREST_TEXT,
            TransparencyTrigger.DIRECT_INTERACTION,
        }
    )

    result = classify_risk_tier(features)

    articles = [c.article for c in result.triggering_criteria]
    assert articles == ["Art. 50(1)", "Art. 50(4)"]


def test_prohibited_takes_precedence_over_annex_iii_and_transparency() -> None:
    features = SystemFeatures(
        prohibited_practices={ProhibitedPractice.SOCIAL_SCORING},
        annex_iii_domains={AnnexIIIDomain.LAW_ENFORCEMENT},
        transparency_triggers={TransparencyTrigger.DIRECT_INTERACTION},
    )

    result = classify_risk_tier(features)

    assert result.tier == RiskTier.PROHIBITED


def test_claimed_exceptions_must_be_subset_of_prohibited_practices() -> None:
    with pytest.raises(ValidationError):
        SystemFeatures(claimed_exceptions={ProhibitedPractice.SOCIAL_SCORING})


def test_derogation_grounds_require_an_annex_iii_domain() -> None:
    with pytest.raises(ValidationError):
        SystemFeatures(derogation_grounds={DerogationGround.PREPARATORY_TASK_ONLY})


@pytest.mark.parametrize("practice", list(ProhibitedPractice))
def test_every_prohibited_practice_has_a_table_entry(practice: ProhibitedPractice) -> None:
    table = _load_table()
    assert practice.value in table["prohibited_practices"]


@pytest.mark.parametrize("domain", list(AnnexIIIDomain))
def test_every_annex_iii_domain_enum_member_has_a_table_entry(domain: AnnexIIIDomain) -> None:
    table = _load_table()
    assert domain.value in table["annex_iii_domains"]


@pytest.mark.parametrize("ground", list(DerogationGround))
def test_every_derogation_ground_enum_member_has_a_table_entry(ground: DerogationGround) -> None:
    table = _load_table()
    assert ground.value in table["derogation_grounds"]


@pytest.mark.parametrize("trigger", list(TransparencyTrigger))
def test_every_transparency_trigger_has_a_table_entry(trigger: TransparencyTrigger) -> None:
    table = _load_table()
    assert trigger.value in table["transparency_triggers"]


@pytest.mark.parametrize("tier", list(RiskTier))
def test_every_risk_tier_enum_member_has_a_table_entry(tier: RiskTier) -> None:
    table = _load_table()
    assert tier.value in table["tiers"]
    assert table["tiers"][tier.value]["label"]
