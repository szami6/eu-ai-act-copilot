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
- compliance_timeline: compute obligation deadlines when the user explicitly asks when, for a \
deadline, or for a timeline and provides the role and placing-on-market date. If another step \
establishes the tier, set `depends_on` to that step's step_id. Do not use this tool as a \
substitute for retrieving the substantive duties that apply.

Example: "Is our CV screener high-risk and what provider obligations apply?" decomposes into \
step 0 (classify_risk_tier, describing the CV screener) and step 1 (rag_search, focused on the \
substantive provider duties for the resulting high-risk category). A separate timeline step is \
only needed if the user also asks when those duties apply.

A question needing only one lookup still gets a single-step plan; do not invent extra steps.

Question: "{query}\""""


def build_planner_prompt(normalized_query: str, max_steps: int) -> str:
    return _PROMPT_TEMPLATE.format(max_steps=max_steps, query=normalized_query)
