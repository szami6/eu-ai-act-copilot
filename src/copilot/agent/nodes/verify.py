"""`verify` (PLAN.md §3.1, node 5): checks the draft's groundedness before
it reaches the user, combining a deterministic citation-resolvability check
with the LLM's own holistic judgment. The deterministic half dominates
(`min(...)`): an unresolvable `[Art. X(y)]` marker is a fact the LLM's own
self-assessment cannot talk itself out of.

On failure, appends one retry `rag_search` step to the plan rather than
looping the whole turn — `route_after_execute`'s dependency-based batching
picks it straight up, and `route_after_verify` (below) notices the plan
has unexecuted work again and sends it back through `execute`.
"""

from __future__ import annotations

import re
from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from copilot.agent.context import AgentContext
from copilot.agent.prompts.verify import VerifyOutput, build_verify_prompt
from copilot.agent.state import AgentState, NodeEvent, SubTask
from copilot.rag.state import Evidence

_CITATION_RE = re.compile(r"\[([^\]]+)\]")
_LOCATOR_RE = re.compile(r"[\dIVXLCDM][\w().,\-–]*")


def _locator(citation_text: str) -> str | None:
    """The digits/roman-numerals after a citation's "Art."/"Annex"/etc.
    prefix, e.g. "6(3)" from "Art. 6(3)" — matched against evidence text by
    substring rather than exact header equality, since a passage's header
    may spell out "Article" where a citation abbreviates to "Art.", or vice
    versa.
    """
    match = _LOCATOR_RE.search(citation_text)
    return match.group(0) if match else None


def _citation_resolvability(draft: str, evidence: list[Evidence]) -> float:
    """The fraction of `[...]` markers in `draft` whose locator appears in
    some evidence chunk's text. No citations at all is treated as fully
    resolvable — there is nothing to contradict — so this only ever
    penalises a draft that cites something absent from what was retrieved.
    """
    citations = _CITATION_RE.findall(draft)
    if not citations:
        return 1.0
    haystack = "\n".join(e.text for e in evidence)
    locators = [_locator(c) for c in citations]
    resolved = sum(1 for loc in locators if loc and loc in haystack)
    return resolved / len(citations)


async def verify(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, object]:
    ctx = runtime.context
    draft = state["draft"] or ""
    citation_score = _citation_resolvability(draft, state["evidence"])

    structured = ctx.chat_model.with_structured_output(VerifyOutput)
    output = await structured.ainvoke(
        [HumanMessage(content=build_verify_prompt(draft, state["evidence"]))]
    )
    assert isinstance(output, VerifyOutput)

    groundedness = min(citation_score, output.groundedness)
    threshold = ctx.settings.groundedness_threshold
    passed = groundedness >= threshold
    can_retry = state["verify_retries"] < ctx.settings.max_verify_retries

    comparator = ">=" if passed else "<"
    trace = [
        NodeEvent(
            node="verify", detail=f"groundedness {groundedness:.2f} {comparator} {threshold:.2f}"
        )
    ]
    update: dict[str, object] = {"groundedness": groundedness, "trace": trace}

    if not passed and can_retry and output.retry_query:
        retry_step = SubTask(
            step_id=len(state["plan"]), tool="rag_search", query=output.retry_query, depends_on=[]
        )
        update["plan"] = [*state["plan"], retry_step]
        update["verify_retries"] = state["verify_retries"] + 1
        update["trace"] = [
            *trace,
            NodeEvent(node="verify", detail=f"-> execute (retry: {output.retry_query!r})"),
        ]
    return update


def route_after_verify(state: AgentState) -> Literal["execute", "finalize"]:
    # A retry step appended above is unexecuted work at index `len(plan)-1`
    # with `cursor` still behind it — the same signal `route_after_execute`
    # uses, read here without needing settings/threshold again.
    if state["cursor"] < len(state["plan"]):
        return "execute"
    return "finalize"
