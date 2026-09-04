"""Art. 111/113 compliance-deadline calculator (PLAN.md §3.3, §11 Phase 2).

`compliance_timeline` answers "what do we have to do, and by when" for a
system that already has a `RiskTier` (from `risk_tier.classify_risk_tier`) —
it does date arithmetic over the Act's staggered phase-in dates (Art. 113)
and legacy/transitional carve-outs (Art. 111), nothing else. No free text,
no LLM: like `classify_risk_tier`, this is complete and final as plain
Python — PLAN.md's signature for this tool (`tier, role,
placing_on_market_date`) has no free-text argument at all, so unlike
`classify_risk_tier` there is no Phase 3 wrapper needed here.

Every date below is quoted directly from the ingested Art. 111/113 chunks
(`ai_act:art:111`, `ai_act:art:113`), not invented — see the constants'
comments for the specific clause each one comes from.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel

from copilot.agent.tools.risk_tier import ProhibitedPractice, RiskTier

# --- Art. 113: staggered application dates ---------------------------------
# Art. 113(a): Chapters I-II, incl. Art. 5 generally.
_CHAPTERS_I_II = date(2025, 2, 2)
# Art. 113(a) exception: Art. 5(1)(ba)/(bb), 5(1a)/(1b).
_PROHIBITED_BA_BB = date(2026, 12, 2)
# Art. 113(b): notifying authorities, GPAI (Chapter V), Chapter VII/XII, Art. 78 (except Art. 101).
_CH_III_S4_V_VII_XII_ART78 = date(2025, 8, 2)
# Art. 113(d) (Articles 102-110, 27 July 2026) isn't modeled: those articles
# govern institutional machinery (market surveillance, the AI Office, the
# database of high-risk systems), not a specific provider/deployer's own
# obligations, so they fall outside what a per-system timeline reports.
#
# Art. 113(c)(i): Chapter III §§1-3 for Art. 6(2)/Annex III systems.
_HIGH_RISK_ANNEX_III = date(2027, 12, 2)
# Art. 113(c)(ii): Chapter III §§1-3 for Art. 6(1)/Annex I systems.
_HIGH_RISK_ANNEX_I = date(2028, 8, 2)
# Art. 113 chapeau: the residual default date (incl. Chapter IV/Art. 50).
_GENERAL = date(2026, 8, 2)

# --- Art. 111: legacy / transitional rules ----------------------------------
# Art. 111(1): "placed on market... before 2 August 2027".
_LARGE_SCALE_IT_CUTOFF = date(2027, 8, 2)
_LARGE_SCALE_IT_DEADLINE = date(2030, 12, 31)  # Art. 111(1)
# Art. 111(2): providers/deployers for public authorities.
_LEGACY_PUBLIC_AUTHORITY_DEADLINE = date(2030, 8, 2)
# Art. 111(3): "placed on the market before 2 August 2025".
_GPAI_GRANDFATHER_CUTOFF = date(2025, 8, 2)
_GPAI_GRANDFATHER_DEADLINE = date(2027, 8, 2)  # Art. 111(3)
# Art. 111(4): placed on market before 2 August 2026.
_SYNTHETIC_CONTENT_GRANDFATHER_CUTOFF = date(2026, 8, 2)
_SYNTHETIC_CONTENT_GRANDFATHER_DEADLINE = date(2026, 12, 2)  # Art. 111(4), Art. 50(2) specifically


class Role(StrEnum):
    PROVIDER = "provider"
    DEPLOYER = "deployer"
    IMPORTER = "importer"
    DISTRIBUTOR = "distributor"


class HighRiskBasis(StrEnum):
    """Which Art. 6 route classified the system as high-risk — Art. 113(c)
    sets a different compliance date for each, so `compliance_timeline`
    cannot pick a date for a HIGH_RISK tier without knowing this."""

    ANNEX_I = "annex_i"
    ANNEX_III = "annex_iii"


class ObligationDeadline(BaseModel):
    obligation: str
    article: str
    # `None` means a legacy system that is only in scope on a future,
    # not-yet-occurred significant design change (Art. 111(2)) — a real
    # "no fixed calendar deadline yet" state, not a missing value.
    deadline: date | None
    days_remaining: int | None
    overdue: bool


def _obligation(
    obligation: str, article: str, deadline: date | None, today: date
) -> ObligationDeadline:
    if deadline is None:
        return ObligationDeadline(
            obligation=obligation,
            article=article,
            deadline=None,
            days_remaining=None,
            overdue=False,
        )
    return ObligationDeadline(
        obligation=obligation,
        article=article,
        deadline=deadline,
        days_remaining=(deadline - today).days,
        overdue=deadline < today,
    )


def compliance_timeline(
    tier: RiskTier,
    role: Role,
    placing_on_market_date: date,
    *,
    high_risk_basis: HighRiskBasis | None = None,
    prohibited_practice: ProhibitedPractice | None = None,
    is_gpai_model: bool = False,
    is_public_authority: bool = False,
    is_large_scale_it_system_component: bool = False,
    generates_synthetic_content: bool = False,
    as_of: date | None = None,
) -> list[ObligationDeadline]:
    """Returns a sorted obligation timeline (earliest deadline first;
    open-ended Art. 111(2) legacy entries last) with `days_remaining` and
    `overdue` computed against `as_of` (default: today).

    `high_risk_basis` is required when `tier` is HIGH_RISK — Art. 6(1)/Annex I
    systems get until 2028-08-02, Art. 6(2)/Annex III systems until
    2027-12-02 (Art. 113(c)); a caller with a `RiskTierResult` in hand always
    knows which route fired, so silently guessing would only ever mask a
    caller bug. `prohibited_practice` is optional even for the PROHIBITED
    tier: only 2 of the 10 Art. 5(1) practices ((ba)/(bb)) get a later date,
    so defaulting to the general (earlier) date when unspecified is the
    conservative direction — it never tells anyone they have more time than
    they actually do.
    """
    if tier == RiskTier.HIGH_RISK and high_risk_basis is None:
        raise ValueError(
            "high_risk_basis is required when tier is HIGH_RISK — Art. 113(c) sets a different "
            "compliance date for the Annex I route (2028-08-02) than the Annex III route "
            "(2027-12-02)"
        )

    today = as_of if as_of is not None else date.today()
    entries: list[tuple[str, str, date | None]] = []

    if tier == RiskTier.PROHIBITED:
        ba_bb = (
            ProhibitedPractice.NON_CONSENSUAL_INTIMATE_IMAGERY,
            ProhibitedPractice.CSAM_ADJACENT_CONTENT,
        )
        if prohibited_practice in ba_bb:
            entries.append((
                "Cease the prohibited practice (Art. 5(1)(ba)/(bb)) — do not place on market, "
                "put into service, or use",
                "Art. 113(a)",
                _PROHIBITED_BA_BB,
            ))
        else:
            entries.append((
                "Cease the prohibited practice (Art. 5(1)) — do not place on market, "
                "put into service, or use",
                "Art. 113(a)",
                _CHAPTERS_I_II,
            ))

    elif tier == RiskTier.HIGH_RISK:
        assert high_risk_basis is not None  # guarded above; narrows the type for mypy
        if high_risk_basis == HighRiskBasis.ANNEX_III:
            base_deadline, basis_article = _HIGH_RISK_ANNEX_III, "Art. 6(2)"
        else:
            base_deadline, basis_article = _HIGH_RISK_ANNEX_I, "Art. 6(1)"

        if is_large_scale_it_system_component and placing_on_market_date < _LARGE_SCALE_IT_CUTOFF:
            entries.append((
                "Bring the large-scale IT system component into compliance",
                "Art. 111(1)",
                _LARGE_SCALE_IT_DEADLINE,
            ))
        elif placing_on_market_date < base_deadline:
            # Art. 111(2): a system already placed on market/put into
            # service before Chapter III applied to its category is only
            # brought into scope by a later significant design change — no
            # fixed date — except a public-authority provider/deployer,
            # which must comply by a fixed date regardless.
            if is_public_authority and role in (Role.PROVIDER, Role.DEPLOYER):
                entries.append((
                    f"Bring the legacy public-authority high-risk system ({basis_article}) "
                    "into compliance",
                    "Art. 111(2)",
                    _LEGACY_PUBLIC_AUTHORITY_DEADLINE,
                ))
            else:
                entries.append((
                    f"Full high-risk requirements ({basis_article}) — in scope only upon "
                    "a significant design change; no fixed deadline until then",
                    "Art. 111(2)",
                    None,
                ))
        else:
            # No max() needed: the elif above already established
            # placing_on_market_date >= base_deadline, so placing_on_market_date
            # is always the later (or equal) of the two here.
            entries.append((
                f"Full high-risk requirements ({basis_article})",
                "Art. 113(c)",
                placing_on_market_date,
            ))

    elif tier == RiskTier.LIMITED_RISK:
        grandfathered = placing_on_market_date < _SYNTHETIC_CONTENT_GRANDFATHER_CUTOFF
        if generates_synthetic_content and grandfathered:
            entries.append((
                "Comply with the synthetic-content marking duty (Art. 50(2)) — "
                "grandfathered system",
                "Art. 111(4)",
                _SYNTHETIC_CONTENT_GRANDFATHER_DEADLINE,
            ))
        else:
            # Unlike the two branches above, max() is genuinely needed here:
            # this branch also covers generates_synthetic_content=False, where
            # placing_on_market_date has no established relationship to
            # _GENERAL and could be arbitrarily far in the past.
            entries.append((
                "Comply with applicable Art. 50 transparency obligations",
                "Art. 113",
                max(_GENERAL, placing_on_market_date),
            ))

    # GPAI-model obligations (Chapter V) are a separate, additive track: a
    # provider's model can owe Chapter V duties independent of whatever tier
    # the system built on it lands in under Art. 5/6/50.
    if is_gpai_model:
        if placing_on_market_date < _GPAI_GRANDFATHER_CUTOFF:
            entries.append((
                "Comply with general-purpose AI model obligations (Chapter V) — "
                "grandfathered model",
                "Art. 111(3)",
                _GPAI_GRANDFATHER_DEADLINE,
            ))
        else:
            # No max() needed: the if above already established
            # placing_on_market_date >= _GPAI_GRANDFATHER_CUTOFF, which falls
            # on the same calendar date as _CH_III_S4_V_VII_XII_ART78
            # (2025-08-02) — so placing_on_market_date is always the later
            # (or equal) of the two here.
            entries.append((
                "Comply with general-purpose AI model obligations (Chapter V)",
                "Art. 113(b)",
                placing_on_market_date,
            ))

    results = [
        _obligation(obligation, article, deadline, today)
        for obligation, article, deadline in entries
    ]
    results.sort(key=lambda o: (o.deadline is None, o.deadline or date.max))
    return results
