"""`finalize` (PLAN.md §3.1, node 6): deterministic, no LLM. Picks the final
answer text — the draft, a refusal, or a partial-evidence fallback — and
appends it to `messages` so the next turn's `triage` sees it in history.
Resolved citations, tool cards, and the trace itself are read directly off
`state.evidence` / `state.tool_calls` / `state.trace` by Phase 4's API layer
rather than duplicated into a field here.

Also the direct target of `triage`'s 'refuse' edge (PLAN §3.1's diagram
labels this entry point "respond(finalize)") — one node, two ways in, which
is why the graph has 6 nodes rather than 7.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from copilot.agent.state import AgentState, NodeEvent

_DISCLAIMER = "This is decision support, not legal advice — verify before relying on it."


def _fallback_summary(state: AgentState) -> str:
    if not state["tool_calls"] and not state["evidence"]:
        return "No answer could be produced before this turn's time budget was reached."
    lines = ["The time budget was reached before a full answer could be composed. Gathered so far:"]
    lines.extend(f"- {tc.tool}: {tc.summary}" for tc in state["tool_calls"])
    if state["evidence"]:
        lines.append(f"- {len(state['evidence'])} passage(s) retrieved but not yet synthesized.")
    return "\n".join(lines)


async def finalize(state: AgentState) -> dict[str, object]:
    if state["route"] == "refuse":
        answer = state["refusal_reason"] or "This request is out of scope for this assistant."
    elif state["draft"]:
        answer = state["draft"]
    else:
        answer = _fallback_summary(state)

    return {
        "messages": [AIMessage(content=f"{answer}\n\n{_DISCLAIMER}")],
        "trace": [NodeEvent(node="finalize", detail="partial" if state["partial"] else "done")],
    }
