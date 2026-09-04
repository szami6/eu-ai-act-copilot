"""Art. 5 / Art. 6 risk-tier decision engine (PLAN.md §3.3, §11 Phase 2).

`classify_risk_tier` is the *pure rule engine*: it takes an already-extracted
`SystemFeatures` vector and returns a `RiskTierResult`, with zero LLM calls
and no free-text handling. PLAN.md §11 scopes Phase 2 as "pure Python, no
LLM"; §7.2 independently confirms the split this implies — its own eval row
for `classify_risk_tier` measures tier accuracy against fixture systems while
noting "the rule engine is separately unit-tested to 100% on the decision
table, so this isolates extraction quality." That only makes sense if the
rule engine and the free-text -> `SystemFeatures` extraction step are two
different things, tested separately. This module is the former. The latter —
`classify_risk_tier(system_description: str)`, matching §3.3's tool signature
literally — is a thin LLM-backed wrapper deferred to Phase 3: it extracts
`SystemFeatures` via structured decoding, then calls the function below.

Decision order mirrors the Act's own structure, checked most-severe-first
since an earlier match makes later ones moot:
  1. Art. 5(1) prohibited practices -> tier=PROHIBITED.
  2. Art. 6(1): safety component of / itself a product under Annex I Union
     harmonisation legislation requiring third-party conformity assessment
     -> tier=HIGH_RISK.
  3. Art. 6(2)/Annex III: one of the 8 high-risk domains applies, subject to
     the Art. 6(3) derogation (never available if the system profiles
     natural persons — 6(3) final subparagraph overrides it) -> HIGH_RISK
     unless derogated.
  4. Art. 50 transparency-obligation triggers -> tier=LIMITED_RISK.
  5. Otherwise -> tier=MINIMAL_RISK.
"""

from __future__ import annotations

import functools
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

_TABLE_PATH = Path(__file__).parent / "risk_tier_table.yaml"
_Table = dict[str, dict[str, dict[str, Any]]]


class RiskTier(StrEnum):
    PROHIBITED = "prohibited"
    HIGH_RISK = "high_risk"
    LIMITED_RISK = "limited_risk"
    MINIMAL_RISK = "minimal_risk"


class ProhibitedPractice(StrEnum):
    """Art. 5(1) unacceptable-risk practices."""

    SUBLIMINAL_MANIPULATION = "subliminal_manipulation"
    EXPLOITS_VULNERABLE_GROUPS = "exploits_vulnerable_groups"
    NON_CONSENSUAL_INTIMATE_IMAGERY = "non_consensual_intimate_imagery"
    CSAM_ADJACENT_CONTENT = "csam_adjacent_content"
    SOCIAL_SCORING = "social_scoring"
    INDIVIDUAL_CRIME_RISK_PROFILING = "individual_crime_risk_profiling"
    UNTARGETED_FACIAL_SCRAPING = "untargeted_facial_scraping"
    WORKPLACE_EDUCATION_EMOTION_INFERENCE = "workplace_education_emotion_inference"
    BIOMETRIC_CATEGORIZATION_PROTECTED_ATTRIBUTES = "biometric_categorization_protected_attributes"
    REALTIME_RBI_LAW_ENFORCEMENT_PUBLIC_SPACE = "realtime_rbi_law_enforcement_public_space"


class AnnexIIIDomain(StrEnum):
    """The 8 high-risk domains of Annex III (Art. 6(2))."""

    BIOMETRICS = "biometrics"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"
    EDUCATION_VOCATIONAL_TRAINING = "education_vocational_training"
    EMPLOYMENT_WORKER_MANAGEMENT = "employment_worker_management"
    ESSENTIAL_SERVICES = "essential_services"
    LAW_ENFORCEMENT = "law_enforcement"
    MIGRATION_ASYLUM_BORDER_CONTROL = "migration_asylum_border_control"
    JUSTICE_DEMOCRATIC_PROCESSES = "justice_democratic_processes"


class DerogationGround(StrEnum):
    """Art. 6(3)(a)-(d): narrow-task grounds that take an Annex III system
    out of high-risk — unless it profiles natural persons (Art. 6(3), final
    subparagraph), which always overrides these."""

    NARROW_PROCEDURAL_TASK = "narrow_procedural_task"
    IMPROVES_COMPLETED_HUMAN_ACTIVITY = "improves_completed_human_activity"
    DETECTS_DEVIATION_WITH_HUMAN_REVIEW = "detects_deviation_with_human_review"
    PREPARATORY_TASK_ONLY = "preparatory_task_only"


class TransparencyTrigger(StrEnum):
    """Art. 50(1)-(4) transparency obligations — orthogonal to the
    prohibited/high-risk tiers (Chapter IV, not Chapter II/III); a system
    that isn't otherwise high-risk but matches one of these is LIMITED_RISK.
    """

    DIRECT_INTERACTION = "direct_interaction"
    SYNTHETIC_CONTENT_GENERATION = "synthetic_content_generation"
    EMOTION_OR_BIOMETRIC_CATEGORIZATION_DISCLOSURE = (
        "emotion_or_biometric_categorization_disclosure"
    )
    DEEPFAKE_OR_PUBLIC_INTEREST_TEXT = "deepfake_or_public_interest_text"


class SystemFeatures(BaseModel):
    """The extracted feature vector `classify_risk_tier` decides over.
    Phase 3's free-text wrapper is responsible for populating this from a
    `system_description`; here it is just data.
    """

    prohibited_practices: set[ProhibitedPractice] = Field(default_factory=set)
    # Practices in `prohibited_practices` for which a statutory exception
    # (e.g. Art. 5(1)(d)'s human-assessment carve-out, Art. 5(1)(f)'s
    # medical/safety carve-out) is *claimed*. A claimed exception suppresses
    # the automatic PROHIBITED tier but never silently — it always surfaces
    # as an `unresolved_features` entry, since whether the exception's
    # narrow factual conditions actually hold is exactly the kind of
    # judgment call this tool defers to a human rather than assuming.
    claimed_exceptions: set[ProhibitedPractice] = Field(default_factory=set)

    is_safety_component_of_regulated_product: bool = False
    requires_third_party_conformity_assessment: bool = False
    # Art. 6(1a): a system used solely for non-safety-related convenience
    # aspects does not count as a safety component...
    is_solely_non_safety_convenience_feature: bool = False
    # ...(6(1b)) unless its failure or malfunction would itself endanger
    # health or safety, which overrides the 6(1a) carve-out.
    safety_relevant: bool = False

    annex_iii_domains: set[AnnexIIIDomain] = Field(default_factory=set)
    performs_profiling_of_natural_persons: bool = False
    derogation_grounds: set[DerogationGround] = Field(default_factory=set)

    transparency_triggers: set[TransparencyTrigger] = Field(default_factory=set)

    @model_validator(mode="after")
    def _check_consistency(self) -> SystemFeatures:
        if not self.claimed_exceptions <= self.prohibited_practices:
            raise ValueError(
                "claimed_exceptions must be a subset of prohibited_practices "
                "— an exception can't be claimed for a practice that wasn't "
                "flagged as present"
            )
        if self.derogation_grounds and not self.annex_iii_domains:
            raise ValueError(
                "derogation_grounds is only meaningful when annex_iii_domains "
                "is non-empty — the Art. 6(3) derogation is a carve-out from "
                "Annex III high-risk status specifically"
            )
        return self


class TriggeringCriterion(BaseModel):
    article: str
    description: str


class RiskTierResult(BaseModel):
    tier: RiskTier
    label: str
    triggering_criteria: list[TriggeringCriterion]
    confidence: float
    unresolved_features: list[str] = Field(default_factory=list)


@functools.lru_cache(maxsize=1)
def _load_table() -> _Table:
    return yaml.safe_load(_TABLE_PATH.read_text(encoding="utf-8"))


def _criterion(table: _Table, section: str, key: str) -> TriggeringCriterion:
    entry = table[section][key]
    return TriggeringCriterion(article=entry["article"], description=entry["description"])


def _clamp(confidence: float) -> float:
    return max(0.0, min(1.0, confidence))


def _tier_result(
    table: _Table,
    tier: RiskTier,
    criteria: list[TriggeringCriterion],
    *,
    confidence: float,
    unresolved: list[str],
) -> RiskTierResult:
    return RiskTierResult(
        tier=tier,
        label=table["tiers"][tier.value]["label"],
        triggering_criteria=criteria,
        confidence=_clamp(confidence),
        unresolved_features=unresolved,
    )


def _transparency_or_minimal(
    features: SystemFeatures,
    table: _Table,
    *,
    confidence: float,
    unresolved: list[str],
) -> RiskTierResult:
    if features.transparency_triggers:
        # Declaration order mirrors Art. 50(1)-(4)'s own numbering; sorting by
        # `.value` instead would sort alphabetically by slug (e.g. "deepfake"
        # before "direct"), scrambling the citation order for no reason.
        criteria = [
            _criterion(table, "transparency_triggers", t.value)
            for t in sorted(features.transparency_triggers, key=list(TransparencyTrigger).index)
        ]
        return _tier_result(
            table, RiskTier.LIMITED_RISK, criteria, confidence=confidence, unresolved=unresolved
        )
    return _tier_result(
        table, RiskTier.MINIMAL_RISK, [], confidence=confidence, unresolved=unresolved
    )


def classify_risk_tier(features: SystemFeatures) -> RiskTierResult:
    """The pure Art. 5/6/50 decision engine — see the module docstring for
    the decision order and PLAN.md §7.2 for why extraction isn't this
    function's concern."""
    table = _load_table()

    if features == SystemFeatures():
        return _tier_result(
            table,
            RiskTier.MINIMAL_RISK,
            [],
            confidence=0.2,
            unresolved=[
                "No risk-relevant features were extracted from the system "
                "description — verify the description was specific enough "
                "to classify (domain, safety-component status, profiling, "
                "and transparency triggers were all absent)."
            ],
        )

    confidence = 1.0
    unresolved: list[str] = []

    # 1. Art. 5(1) — prohibited practices, checked first: nothing below
    # matters if the practice is banned outright.
    prohibited_criteria: list[TriggeringCriterion] = []
    # Declaration order mirrors Art. 5(1)(a)-(h)'s own lettering (including
    # (ba)/(bb) slotted between (b) and (c)); `.value` would sort
    # alphabetically by slug instead, e.g. "realtime..." before "subliminal...".
    for practice in sorted(features.prohibited_practices, key=list(ProhibitedPractice).index):
        entry = table["prohibited_practices"][practice.value]
        if practice in features.claimed_exceptions:
            unresolved.append(
                f"{entry['article']}: a statutory exception was claimed for "
                f'"{entry["description"]}" — verify the exception\'s narrow '
                "factual conditions actually apply before relying on this "
                "classification."
            )
            confidence -= 0.3
            continue
        prohibited_criteria.append(
            TriggeringCriterion(article=entry["article"], description=entry["description"])
        )
    if prohibited_criteria:
        return _tier_result(
            table,
            RiskTier.PROHIBITED,
            prohibited_criteria,
            confidence=confidence,
            unresolved=unresolved,
        )

    # 2. Art. 6(1) — Annex I product-safety route. 6(1a)/(1b): a solely-
    # convenience feature doesn't qualify as a safety component unless its
    # failure would endanger health/safety.
    if (
        features.is_safety_component_of_regulated_product
        and features.requires_third_party_conformity_assessment
        and (not features.is_solely_non_safety_convenience_feature or features.safety_relevant)
    ):
        return _tier_result(
            table,
            RiskTier.HIGH_RISK,
            [
                TriggeringCriterion(
                    article="Art. 6(1)",
                    description=(
                        "Safety component of, or itself, a product requiring third-party "
                        "conformity assessment under Annex I Union harmonisation legislation"
                    ),
                )
            ],
            confidence=confidence,
            unresolved=unresolved,
        )

    # 3. Art. 6(2)/Annex III route, subject to the Art. 6(3) derogation.
    if features.annex_iii_domains:
        # Declaration order mirrors Annex III's own point 1-8 numbering.
        domain_criteria = [
            _criterion(table, "annex_iii_domains", d.value)
            for d in sorted(features.annex_iii_domains, key=list(AnnexIIIDomain).index)
        ]
        if features.performs_profiling_of_natural_persons:
            # 6(3), final subparagraph: profiling always defeats the
            # derogation, regardless of which ground(s) below are claimed.
            return _tier_result(
                table,
                RiskTier.HIGH_RISK,
                domain_criteria,
                confidence=confidence,
                unresolved=unresolved,
            )
        if features.derogation_grounds:
            grounds = ", ".join(sorted(g.value for g in features.derogation_grounds))
            return _transparency_or_minimal(
                features,
                table,
                confidence=confidence - 0.2,
                unresolved=[
                    *unresolved,
                    f"Art. 6(3) derogation claimed on ground(s): {grounds} — verify the system "
                    "genuinely does not pose a significant risk of harm to health, safety, or "
                    "fundamental rights before relying on the non-high-risk classification.",
                ],
            )
        return _tier_result(
            table, RiskTier.HIGH_RISK, domain_criteria, confidence=confidence, unresolved=unresolved
        )

    # 4. Not high-risk via either route: Art. 50 transparency, else minimal.
    return _transparency_or_minimal(features, table, confidence=confidence, unresolved=unresolved)
