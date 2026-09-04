"""The 3 LangChain `@tool`s (PLAN.md §3.3): "All tools are LangChain `@tool`s
with Pydantic schemas, bindable through vLLM's native tool-calling and
callable directly from the eval harness."

`make_tools` is a factory rather than 3 module-level `@tool`s because
`rag_search` needs the compiled RAG subgraph and `classify_risk_tier` needs
the chat model for its own internal extraction call (`tools/extract.py`) —
dependencies that must not appear in the tool's own schema (the model calling
`rag_search` supplies `query`/`k`, never a Qdrant client). Closing over them
keeps each function's signature exactly PLAN §3.3's, with the schema
LangChain infers from it being only the arguments a caller actually supplies.

`compliance_timeline` takes no chat model at all: PLAN §3.3 gives it fully
typed arguments and no free text, so it is exposed as-is over the pure
function in `tools/compliance_timeline.py` — nothing to extract internally.
`execute` (Phase 3's orchestrator, not this module) is what extracts those
typed arguments from conversation context before calling it.
"""

from __future__ import annotations

from datetime import date

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph

from copilot.agent.tools.compliance_timeline import HighRiskBasis, ObligationDeadline, Role
from copilot.agent.tools.compliance_timeline import compliance_timeline as _compliance_timeline
from copilot.agent.tools.extract import extract_system_features
from copilot.agent.tools.risk_tier import ProhibitedPractice, RiskTier, RiskTierResult
from copilot.agent.tools.risk_tier import classify_risk_tier as _classify_risk_tier
from copilot.rag.graph import RagContext, run_rag
from copilot.rag.state import RagQuery, RagResult, RagState

_RagGraph = CompiledStateGraph[RagState, RagContext, RagState, RagState]


def make_tools(
    chat_model: BaseChatModel,
    rag_compiled: _RagGraph,
    rag_context: RagContext,
) -> dict[str, BaseTool]:
    @tool
    def rag_search(query: str, k: int = 5) -> RagResult:
        """Retrieve EU AI Act / GDPR passages relevant to a query, with citations."""
        return run_rag(rag_compiled, RagQuery(query=query, k=k), rag_context)

    @tool
    def classify_risk_tier(system_description: str) -> RiskTierResult:
        """Classify an AI system's EU AI Act risk tier from a free-text description of it."""
        features = extract_system_features(system_description, chat_model)
        return _classify_risk_tier(features)

    @tool
    def compliance_timeline(
        tier: RiskTier,
        role: Role,
        placing_on_market_date: date,
        high_risk_basis: HighRiskBasis | None = None,
        prohibited_practice: ProhibitedPractice | None = None,
        is_gpai_model: bool = False,
        is_public_authority: bool = False,
        is_large_scale_it_system_component: bool = False,
        generates_synthetic_content: bool = False,
    ) -> list[ObligationDeadline]:
        """Compute the Art. 111/113 obligation timeline for a system with a known risk tier."""
        return _compliance_timeline(
            tier,
            role,
            placing_on_market_date,
            high_risk_basis=high_risk_basis,
            prohibited_practice=prohibited_practice,
            is_gpai_model=is_gpai_model,
            is_public_authority=is_public_authority,
            is_large_scale_it_system_component=is_large_scale_it_system_component,
            generates_synthetic_content=generates_synthetic_content,
        )

    tools: list[BaseTool] = [rag_search, classify_risk_tier, compliance_timeline]
    return {t.name: t for t in tools}
