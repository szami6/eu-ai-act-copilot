"""Unit tests for `agent.tools.bindings.make_tools` (PLAN.md §3.3): tool
names and the 2 tools directly callable with no live service —
`rag_search` wraps the real RAG subgraph and is excluded for the same
reason `tests/test_graph.py` excludes `retrieve`/`rerank` (needs a live
Qdrant).
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, cast

import pytest
from langchain_core.tools import BaseTool

from copilot import llm as llm_module
from copilot.agent.tools.bindings import make_tools
from copilot.agent.tools.compliance_timeline import ObligationDeadline
from copilot.agent.tools.risk_tier import RiskTier, RiskTierResult
from copilot.llm import DummyChatModel


def _tools() -> dict[str, BaseTool]:
    return make_tools(DummyChatModel(), cast(Any, None), cast(Any, None))


def test_make_tools_returns_all_three_by_name() -> None:
    assert set(_tools()) == {"rag_search", "classify_risk_tier", "compliance_timeline"}


async def test_classify_risk_tier_tool_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_module,
        "_FIXTURES",
        [
            (
                re.compile(r"feature vector"),
                '{"prohibited_practices": [], "claimed_exceptions": [], '
                '"is_safety_component_of_regulated_product": false, '
                '"requires_third_party_conformity_assessment": false, '
                '"is_solely_non_safety_convenience_feature": false, "safety_relevant": false, '
                '"annex_iii_domains": [], "performs_profiling_of_natural_persons": false, '
                '"derogation_grounds": [], "transparency_triggers": []}',
            )
        ],
    )

    result = await _tools()["classify_risk_tier"].ainvoke({"system_description": "A chatbot."})

    assert isinstance(result, RiskTierResult)
    assert result.tier == RiskTier.MINIMAL_RISK


async def test_compliance_timeline_tool_end_to_end() -> None:
    result = await _tools()["compliance_timeline"].ainvoke(
        {
            "tier": "limited_risk",
            "role": "provider",
            "placing_on_market_date": date(2027, 1, 1),
            "generates_synthetic_content": False,
        }
    )

    assert isinstance(result, list)
    assert result
    assert all(isinstance(o, ObligationDeadline) for o in result)
