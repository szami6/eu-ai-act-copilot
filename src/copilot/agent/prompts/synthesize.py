"""`synthesize`'s prompt (PLAN.md §3.1, node 4).

No structured-output schema here — unlike `triage`/`planner`/`verify`, the
draft answer is free text, streamed to the UI (Phase 4) as it is generated.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from copilot.agent.state import ToolCallRecord
from copilot.rag.state import Evidence

_PROMPT_TEMPLATE = """Answer the question concisely in 1-3 sentences using only the passages and tool results below. Every \
sentence that states a fact must end with the article citation shown in that passage's header \
or tool result, formatted strictly in the form [Art. X(y)] or [Article X] or [Annex III, point N] (e.g. [Art. 5(1)(a)]). \
Do not add other words like 'AI Act:' inside the citation bracket. \
If the passages and tool results do not contain enough to answer, say so plainly rather \
than guessing — do not use any outside knowledge of the AI Act or GDPR. Do not repeat whole passages verbatim.

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


def render_tool_results(tool_calls: list[ToolCallRecord]) -> str:
    if not tool_calls:
        return "(none)"
    rendered = []
    for call in tool_calls:
        detail = call.summary
        if call.tool != "rag_search":
            if isinstance(call.result, BaseModel):
                detail = json.dumps(call.result.model_dump(mode="json"), ensure_ascii=False)
            elif isinstance(call.result, list) and all(
                isinstance(item, BaseModel) for item in call.result
            ):
                detail = json.dumps(
                    [item.model_dump(mode="json") for item in call.result], ensure_ascii=False
                )
        rendered.append(f"{call.tool} (step {call.step_id}, ok={call.ok}): {detail}")
    return "\n\n".join(rendered)


def build_synthesize_prompt(
    query: str, evidence: list[Evidence], tool_calls: list[ToolCallRecord]
) -> str:
    return _PROMPT_TEMPLATE.format(
        query=query,
        evidence=_render_evidence(evidence),
        tool_results=render_tool_results(tool_calls),
    )
