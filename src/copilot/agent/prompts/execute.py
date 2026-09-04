"""Argument extraction for the one tool `execute` cannot call as-is:
`compliance_timeline` takes fully typed arguments and no free text at all
(PLAN.md §3.3), so unlike `rag_search` (the planner's own `query` already
*is* the argument) and `classify_risk_tier` (which extracts internally,
`tools/extract.py`), `execute` must extract this tool's arguments itself
before invoking it — this is the "LLM call only for tool-argument
extraction" the node responsibility table (§3.1) means for this tool.

A dependency step's `ToolCallRecord.summary` (typically a prior
`classify_risk_tier` call) is threaded into the prompt so the model reads
the tier/basis off that result rather than re-deriving it from scratch.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from copilot.agent.tools.compliance_timeline import HighRiskBasis, ProhibitedPractice, Role
from copilot.agent.tools.risk_tier import RiskTier


class ComplianceTimelineArgs(BaseModel):
    tier: RiskTier = Field(description="Use the dependency step's result if one is given.")
    role: Role
    placing_on_market_date: date = Field(
        description="ISO date the system was/will be placed on the market or put into service."
    )
    high_risk_basis: HighRiskBasis | None = Field(
        default=None, description="Required when tier is high_risk: which Art. 6 route applied."
    )
    prohibited_practice: ProhibitedPractice | None = None
    is_gpai_model: bool = False
    is_public_authority: bool = False
    is_large_scale_it_system_component: bool = False
    generates_synthetic_content: bool = False


_PROMPT_TEMPLATE = """Extract the arguments needed to compute a compliance timeline.

Conversation context: "{context}"

{dependency_section}

Step instruction: "{instruction}"

If a prior tool result is shown above, use its tier (and high-risk basis, if any) exactly rather \
than re-deriving them. Extract role and placing_on_market_date from the context; set the \
optional flags only when the context actually supports them, otherwise leave them at their \
defaults."""


def build_compliance_timeline_extraction_prompt(
    context: str, instruction: str, dependency_summary: str | None
) -> str:
    dependency_section = (
        f"Prior result: {dependency_summary}"
        if dependency_summary
        else "(no prior tool result available for this step)"
    )
    return _PROMPT_TEMPLATE.format(
        context=context, dependency_section=dependency_section, instruction=instruction
    )
