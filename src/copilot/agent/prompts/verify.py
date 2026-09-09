"""`verify`'s structured output and prompt (PLAN.md §3.1, node 5).

The LLM's holistic groundedness score is one of two checks `nodes/verify.py`
combines — the other is a deterministic citation-resolvability check over
the draft's `[Art. X(y)]` markers, which needs no prompt and lives there.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from copilot.agent.prompts.synthesize import render_tool_results
from copilot.agent.state import ToolCallRecord
from copilot.rag.state import Evidence


class VerifyOutput(BaseModel):
    groundedness: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "0.0-1.0: the fraction of factual claims in the draft that are actually supported "
            "by the passages, judged independently of whether a citation marker is present."
        ),
    )
    ungrounded_claims: list[str] = Field(
        default_factory=list, description="Sentences from the draft not supported by any passage."
    )
    retry_query: str | None = Field(
        default=None,
        description="A broadened or rephrased retrieval query. Only meaningful when groundedness "
        "is low because evidence was missing rather than the draft misreading it.",
    )


_PROMPT_TEMPLATE = """Judge whether this draft answer is grounded in the passages it was given.

Passages:
{evidence}

Deterministic tool results:
{tool_results}

Draft answer:
{draft}

Score groundedness from 0.0 (unsupported) to 1.0 (every claim traceable to a passage), list any \
ungrounded sentences, and if the passages were simply insufficient, propose a broadened \
retrieval query."""


def _render_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(none)"
    return "\n\n".join(f"[{e.chunk_id}]\n{e.text}" for e in evidence)


def build_verify_prompt(
    draft: str, evidence: list[Evidence], tool_calls: list[ToolCallRecord]
) -> str:
    return _PROMPT_TEMPLATE.format(
        evidence=_render_evidence(evidence),
        tool_results=render_tool_results(tool_calls),
        draft=draft,
    )
