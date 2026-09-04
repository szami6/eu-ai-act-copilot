"""`synthesize`'s prompt (PLAN.md §3.1, node 4).

No structured-output schema here — unlike `triage`/`planner`/`verify`, the
draft answer is free text, streamed to the UI (Phase 4) as it is generated.
"""

from __future__ import annotations

from copilot.agent.state import ToolCallRecord
from copilot.rag.state import Evidence

_PROMPT_TEMPLATE = """Answer the question using only the passages and tool results below. Every \
sentence that states a fact must end with the article citation shown in that passage's header \
or tool result, in the form [Art. X(y)] (or [Annex III, point N] / [Art. X, recital N] as \
shown). If the passages and tool results do not contain enough to answer, say so plainly rather \
than guessing — do not use any outside knowledge of the AI Act or GDPR.

Question: "{query}"

Retrieved passages:
{evidence}

Tool results:
{tool_results}

Answer:"""


def _render_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(none retrieved)"
    return "\n\n".join(f"[{e.chunk_id}]\n{e.text}" for e in evidence)


def _render_tool_results(tool_calls: list[ToolCallRecord]) -> str:
    if not tool_calls:
        return "(none)"
    return "\n\n".join(f"{tc.tool} (step {tc.step_id}): {tc.summary}" for tc in tool_calls)


def build_synthesize_prompt(
    query: str, evidence: list[Evidence], tool_calls: list[ToolCallRecord]
) -> str:
    return _PROMPT_TEMPLATE.format(
        query=query,
        evidence=_render_evidence(evidence),
        tool_results=_render_tool_results(tool_calls),
    )
