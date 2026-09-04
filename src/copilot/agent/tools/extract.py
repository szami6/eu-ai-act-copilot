"""Free-text -> `SystemFeatures` extraction (PLAN.md §3.3 Tool 2): "The LLM's
only job is structured extraction... The decision itself is pure Python over
the table." This is the Phase-3 half of the split `risk_tier.py`'s own
module docstring describes; `classify_risk_tier` (the pure function) is the
other half, called from here with whatever this extracts.

Category glosses are pulled from `risk_tier_table.yaml` via `_load_table`
rather than restated here, so the extraction prompt and the citations
`classify_risk_tier` later attaches never drift apart.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from copilot.agent.tools.risk_tier import SystemFeatures, _load_table

_PROMPT_TEMPLATE = """Extract a structured feature vector from this description of an AI system, \
for classification under the EU AI Act. Only set a field when the description actually supports \
it — leave sets empty and booleans false rather than guessing.

Prohibited practices (Art. 5(1)) — set any that clearly apply:
{prohibited_practices}
If the description explicitly claims a statutory exception applies to one of the above, also add \
it to claimed_exceptions.

High-risk product-safety route (Art. 6(1)): set is_safety_component_of_regulated_product and \
requires_third_party_conformity_assessment if the system is, or is a safety component of, a \
product regulated under Union product-safety law (machinery, medical devices, toys, lifts, \
etc.). Set is_solely_non_safety_convenience_feature if it only adds a non-safety convenience to \
such a product; set safety_relevant instead if its failure could endanger health or safety.

Annex III high-risk domains (Art. 6(2)) — set any that clearly apply:
{annex_iii_domains}
Set performs_profiling_of_natural_persons if the system profiles individuals.

Art. 6(3) derogation grounds — only meaningful alongside an Annex III domain above, and only if \
the description supports one of:
{derogation_grounds}

Art. 50 transparency triggers — set any that clearly apply:
{transparency_triggers}

System description: "{description}\""""


def _render_category(table: dict[str, dict[str, dict[str, object]]], section: str) -> str:
    return "\n".join(f"- {key}: {entry['description']}" for key, entry in table[section].items())


def build_extraction_prompt(system_description: str) -> str:
    table = _load_table()
    return _PROMPT_TEMPLATE.format(
        prohibited_practices=_render_category(table, "prohibited_practices"),
        annex_iii_domains=_render_category(table, "annex_iii_domains"),
        derogation_grounds=_render_category(table, "derogation_grounds"),
        transparency_triggers=_render_category(table, "transparency_triggers"),
        description=system_description,
    )


def extract_system_features(system_description: str, chat_model: BaseChatModel) -> SystemFeatures:
    """Falls back to `SystemFeatures()` (all-empty) on a validation failure —
    malformed or self-inconsistent model output — rather than raising:
    `classify_risk_tier` already treats the empty vector as its own
    degenerate case (MINIMAL_RISK, confidence=0.2, an explicit
    "no risk-relevant features were extracted" note), which is exactly the
    honest, low-confidence signal a failed extraction should produce.
    """
    structured = chat_model.with_structured_output(SystemFeatures)
    prompt = build_extraction_prompt(system_description)
    try:
        result = structured.invoke([HumanMessage(content=prompt)])
    except ValidationError:
        return SystemFeatures()
    assert isinstance(result, SystemFeatures)
    return result
