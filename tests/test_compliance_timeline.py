"""Unit tests for the Art. 111/113 compliance-deadline calculator (PLAN.md
§7.2: "Exact match | `freezegun`-pinned dates").

Most cases pin `as_of` explicitly for directness; a couple use `freeze_time`
to also exercise the default (`as_of=None` -> `date.today()`) code path.
"""

from __future__ import annotations

from datetime import date

import pytest
from freezegun import freeze_time

from copilot.agent.tools.compliance_timeline import HighRiskBasis, Role, compliance_timeline
from copilot.agent.tools.risk_tier import ProhibitedPractice, RiskTier


def test_high_risk_requires_a_basis() -> None:
    with pytest.raises(ValueError, match="high_risk_basis"):
        compliance_timeline(RiskTier.HIGH_RISK, Role.PROVIDER, date(2026, 1, 1))


def test_prohibited_default_practice_uses_general_date() -> None:
    result = compliance_timeline(
        RiskTier.PROHIBITED, Role.PROVIDER, date(2026, 1, 1), as_of=date(2026, 1, 1)
    )

    assert len(result) == 1
    assert result[0].article == "Art. 113(a)"
    assert result[0].deadline == date(2025, 2, 2)
    assert result[0].overdue is True
    assert result[0].days_remaining == (date(2025, 2, 2) - date(2026, 1, 1)).days


@pytest.mark.parametrize(
    "practice",
    [ProhibitedPractice.NON_CONSENSUAL_INTIMATE_IMAGERY, ProhibitedPractice.CSAM_ADJACENT_CONTENT],
)
def test_prohibited_ba_bb_practices_use_later_date(practice: ProhibitedPractice) -> None:
    result = compliance_timeline(
        RiskTier.PROHIBITED,
        Role.PROVIDER,
        date(2026, 1, 1),
        prohibited_practice=practice,
        as_of=date(2026, 1, 1),
    )

    assert result[0].deadline == date(2026, 12, 2)
    assert result[0].overdue is False


def test_high_risk_annex_iii_new_system_deadline_is_placing_date() -> None:
    placing = date(2028, 1, 1)  # after the 2027-12-02 Annex III phase-in
    result = compliance_timeline(
        RiskTier.HIGH_RISK,
        Role.PROVIDER,
        placing,
        high_risk_basis=HighRiskBasis.ANNEX_III,
        as_of=date(2026, 1, 1),
    )

    assert len(result) == 1
    assert result[0].article == "Art. 113(c)"
    assert result[0].deadline == placing


def test_high_risk_annex_i_uses_a_later_base_deadline_than_annex_iii() -> None:
    """Same placing date: 'new system' under Annex III's earlier base
    deadline (2027-12-02), but still 'legacy' under Annex I's later one
    (2028-08-02)."""
    placing = date(2028, 1, 1)

    annex_iii = compliance_timeline(
        RiskTier.HIGH_RISK,
        Role.PROVIDER,
        placing,
        high_risk_basis=HighRiskBasis.ANNEX_III,
        as_of=date(2026, 1, 1),
    )
    annex_i = compliance_timeline(
        RiskTier.HIGH_RISK,
        Role.PROVIDER,
        placing,
        high_risk_basis=HighRiskBasis.ANNEX_I,
        as_of=date(2026, 1, 1),
    )

    assert annex_iii[0].article == "Art. 113(c)"
    assert annex_iii[0].deadline == placing

    assert annex_i[0].article == "Art. 111(2)"
    assert annex_i[0].deadline is None


def test_high_risk_legacy_system_has_no_fixed_deadline() -> None:
    result = compliance_timeline(
        RiskTier.HIGH_RISK,
        Role.PROVIDER,
        date(2020, 1, 1),  # long before the 2027-12-02 Annex III phase-in
        high_risk_basis=HighRiskBasis.ANNEX_III,
        as_of=date(2026, 1, 1),
    )

    assert result[0].article == "Art. 111(2)"
    assert result[0].deadline is None
    assert result[0].days_remaining is None
    assert result[0].overdue is False


def test_high_risk_legacy_public_authority_has_fixed_deadline() -> None:
    result = compliance_timeline(
        RiskTier.HIGH_RISK,
        Role.DEPLOYER,
        date(2020, 1, 1),
        high_risk_basis=HighRiskBasis.ANNEX_III,
        is_public_authority=True,
        as_of=date(2026, 1, 1),
    )

    assert result[0].article == "Art. 111(2)"
    assert result[0].deadline == date(2030, 8, 2)


def test_high_risk_legacy_public_authority_carveout_only_applies_to_provider_or_deployer() -> None:
    result = compliance_timeline(
        RiskTier.HIGH_RISK,
        Role.IMPORTER,
        date(2020, 1, 1),
        high_risk_basis=HighRiskBasis.ANNEX_III,
        is_public_authority=True,
        as_of=date(2026, 1, 1),
    )

    assert result[0].deadline is None


def test_large_scale_it_system_component_gets_special_deadline() -> None:
    result = compliance_timeline(
        RiskTier.HIGH_RISK,
        Role.PROVIDER,
        date(2026, 1, 1),  # before the 2027-08-02 Art. 111(1) cutoff
        high_risk_basis=HighRiskBasis.ANNEX_III,
        is_large_scale_it_system_component=True,
        as_of=date(2026, 1, 1),
    )

    assert len(result) == 1
    assert result[0].article == "Art. 111(1)"
    assert result[0].deadline == date(2030, 12, 31)


def test_large_scale_it_system_component_after_cutoff_falls_back_to_normal_rules() -> None:
    # After both the 2027-08-02 Art. 111(1) cutoff and the 2027-12-02 Annex
    # III base deadline, so this clears the legacy branch too.
    placing = date(2028, 1, 1)
    result = compliance_timeline(
        RiskTier.HIGH_RISK,
        Role.PROVIDER,
        placing,
        high_risk_basis=HighRiskBasis.ANNEX_III,
        is_large_scale_it_system_component=True,
        as_of=date(2026, 1, 1),
    )

    assert result[0].article == "Art. 113(c)"
    assert result[0].deadline == placing


def test_limited_risk_synthetic_content_grandfathered() -> None:
    result = compliance_timeline(
        RiskTier.LIMITED_RISK,
        Role.PROVIDER,
        date(2026, 1, 1),  # before the 2026-08-02 cutoff
        generates_synthetic_content=True,
        as_of=date(2026, 1, 1),
    )

    assert len(result) == 1
    assert result[0].article == "Art. 111(4)"
    assert result[0].deadline == date(2026, 12, 2)


def test_limited_risk_synthetic_content_after_cutoff_uses_general_date() -> None:
    placing = date(2027, 1, 1)
    result = compliance_timeline(
        RiskTier.LIMITED_RISK,
        Role.PROVIDER,
        placing,
        generates_synthetic_content=True,
        as_of=date(2026, 1, 1),
    )

    assert result[0].article == "Art. 113"
    assert result[0].deadline == placing


def test_limited_risk_without_synthetic_content_uses_general_date() -> None:
    result = compliance_timeline(
        RiskTier.LIMITED_RISK, Role.PROVIDER, date(2020, 1, 1), as_of=date(2026, 1, 1)
    )

    assert result[0].deadline == date(2026, 8, 2)


def test_minimal_risk_has_no_obligations() -> None:
    result = compliance_timeline(
        RiskTier.MINIMAL_RISK, Role.PROVIDER, date(2026, 1, 1), as_of=date(2026, 1, 1)
    )

    assert result == []


def test_gpai_model_is_additive_to_minimal_risk_tier() -> None:
    result = compliance_timeline(
        RiskTier.MINIMAL_RISK,
        Role.PROVIDER,
        date(2024, 1, 1),  # before the 2025-08-02 GPAI cutoff
        is_gpai_model=True,
        as_of=date(2026, 1, 1),
    )

    assert len(result) == 1
    assert result[0].article == "Art. 111(3)"
    assert result[0].deadline == date(2027, 8, 2)


def test_gpai_model_is_additive_alongside_a_tier_obligation() -> None:
    result = compliance_timeline(
        RiskTier.LIMITED_RISK,
        Role.PROVIDER,
        date(2024, 1, 1),
        is_gpai_model=True,
        as_of=date(2026, 1, 1),
    )

    assert len(result) == 2
    articles = {o.article for o in result}
    assert articles == {"Art. 113", "Art. 111(3)"}


def test_gpai_model_after_cutoff_uses_general_chapter_v_date() -> None:
    placing = date(2025, 9, 1)
    result = compliance_timeline(
        RiskTier.MINIMAL_RISK, Role.PROVIDER, placing, is_gpai_model=True, as_of=date(2026, 1, 1)
    )

    assert result[0].deadline == placing


def test_results_sorted_earliest_deadline_first() -> None:
    result = compliance_timeline(
        RiskTier.LIMITED_RISK,
        Role.PROVIDER,
        date(2024, 1, 1),
        generates_synthetic_content=True,  # -> 2026-12-02
        is_gpai_model=True,  # -> 2027-08-02
        as_of=date(2020, 1, 1),
    )

    deadlines = [o.deadline for o in result]
    assert deadlines == sorted(deadlines)  # type: ignore[type-var]


def test_open_ended_entries_sort_last() -> None:
    result = compliance_timeline(
        RiskTier.HIGH_RISK,
        Role.PROVIDER,
        date(2020, 1, 1),  # legacy -> Art. 111(2), no fixed date
        high_risk_basis=HighRiskBasis.ANNEX_III,
        is_gpai_model=True,  # -> a fixed, dated entry
        as_of=date(2026, 1, 1),
    )

    assert len(result) == 2
    assert result[-1].deadline is None


def test_overdue_and_days_remaining_computed_against_as_of() -> None:
    past = compliance_timeline(
        RiskTier.MINIMAL_RISK,
        Role.PROVIDER,
        date(2020, 1, 1),
        is_gpai_model=True,
        as_of=date(2028, 1, 1),
    )
    future = compliance_timeline(
        RiskTier.MINIMAL_RISK,
        Role.PROVIDER,
        date(2020, 1, 1),
        is_gpai_model=True,
        as_of=date(2020, 1, 1),
    )

    assert past[0].overdue is True
    assert past[0].days_remaining is not None
    assert past[0].days_remaining < 0

    assert future[0].overdue is False
    assert future[0].days_remaining is not None
    assert future[0].days_remaining > 0


def test_deadline_exactly_on_as_of_is_not_overdue() -> None:
    result = compliance_timeline(
        RiskTier.HIGH_RISK,
        Role.PROVIDER,
        date(2027, 12, 2),
        high_risk_basis=HighRiskBasis.ANNEX_III,
        as_of=date(2027, 12, 2),
    )

    assert result[0].deadline == date(2027, 12, 2)
    assert result[0].days_remaining == 0
    assert result[0].overdue is False


@freeze_time("2026-06-15")
def test_as_of_defaults_to_today() -> None:
    result = compliance_timeline(RiskTier.PROHIBITED, Role.PROVIDER, date(2026, 1, 1))

    assert result[0].days_remaining == (date(2025, 2, 2) - date(2026, 6, 15)).days
    assert result[0].overdue is True
