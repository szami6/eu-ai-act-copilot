"""`planner`'s structured output and prompt (PLAN.md §3.1, node 2)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from copilot.agent.state import SubTask


class PlannerOutput(BaseModel):
    subtasks: list[SubTask] = Field(
        description="1 to max_steps ordered steps. step_id starts at 0 and increments by 1."
    )


_PROMPT_TEMPLATE = """Decompose this question into an ordered list of at most {max_steps} steps, \
each using exactly one tool:

- rag_search: retrieve AI Act/GDPR text for a focused query. Use for any sub-question answerable \
from the regulation's text.
- classify_risk_tier: determine an AI system's risk tier from a free-text description of what it \
does. `query` should be the fullest description of the system available.
- compliance_timeline: compute obligation deadlines for a system that already has a known risk \
tier. Only use this after a step that establishes the tier — set `depends_on` to that step's \
step_id, and write `query` as a short instruction (its actual arguments come from the \
dependency plus conversation context, not from this text).

Example: "Is our CV screener high-risk and what are our obligations?" decomposes into step 0 \
(classify_risk_tier, describing the CV screener) and step 1 (compliance_timeline, \
depends_on=[0]) — not a rag_search, since the tier is a computed fact, not something to look up.

A question needing only one lookup still gets a single-step plan; do not invent extra steps.

Question: "{query}\""""


def build_planner_prompt(normalized_query: str, max_steps: int) -> str:
    return _PROMPT_TEMPLATE.format(max_steps=max_steps, query=normalized_query)
