"""Assembles the process-lifetime objects `/chat` needs (PLAN.md §2.1): the
compiled orchestrator graph, the compiled RAG subgraph, and the shared
clients/indices both close over. Built once in `main.py`'s `lifespan`, not
per-request — mirroring `retrievers.load_bm25_index`'s own "call once at
process startup" contract.

Building this must never crash the process: `/health`'s own docstring
already establishes that an unreachable LLM or Qdrant "degrades what a
caller sees on `/chat`, not the liveness of this process", and Qdrant may
genuinely not have anything indexed yet the moment `api` starts (`ingest`
is a separate one-shot compose profile, §2.3). If the BM25 mirror can't be
loaded, `/chat` still works for any plan that never calls `rag_search`
(e.g. the classify_risk_tier -> compliance_timeline path) — a `rag_search`
step then fails as one ordinary failed tool call (`execute.py`'s own
per-step boundary), not a startup crash.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import QdrantClient

from copilot.agent.context import AgentContext
from copilot.agent.graph import build_agent_graph
from copilot.agent.state import AgentState
from copilot.agent.tools.bindings import make_tools
from copilot.config import Settings
from copilot.ingest.index import COLLECTION_NAME
from copilot.llm import get_chat_model
from copilot.rag.graph import RagContext, build_rag_graph
from copilot.rag.retrievers import Bm25Index, build_bm25_index, load_bm25_index
from copilot.rag.state import RagState

logger = logging.getLogger(__name__)

# Mirrors `agent.graph`'s own private alias (itself mirroring
# `agent.tools.bindings._RagGraph`) — each module that names "the compiled
# X graph" type defines its own local alias rather than importing another
# module's underscore-private one.
_AgentGraph = CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]


@dataclass
class AppRuntime:
    """The expensive, request-invariant half of what `/chat` needs. A
    per-request `AgentContext` is still built fresh off `base_settings` (see
    `main.chat`) so a turn's sidebar overrides (top-k, rerank on/off, PLAN
    §6) never mutate state another concurrent request is reading."""

    qdrant_client: QdrantClient
    bm25_index: Bm25Index
    chat_model: BaseChatModel
    base_settings: Settings
    corpus_ready: bool
    agent_compiled: _AgentGraph
    rag_compiled: CompiledStateGraph[RagState, RagContext, RagState, RagState]


def _load_bm25_index_or_empty(client: QdrantClient, settings: Settings) -> tuple[Bm25Index, bool]:
    try:
        index = load_bm25_index(client, COLLECTION_NAME)
    except Exception:
        logger.warning(
            "Could not load the BM25 mirror from Qdrant at %s — rag_search will fail until "
            "ingest has run and the API is restarted (or /chat is retried after a reload).",
            settings.qdrant_url,
            exc_info=True,
        )
        return build_bm25_index([]), False
    else:
        return index, bool(index.chunks)


def build_app_runtime(settings: Settings) -> AppRuntime:
    qdrant_client = QdrantClient(url=settings.qdrant_url)
    bm25_index, corpus_ready = _load_bm25_index_or_empty(qdrant_client, settings)
    return AppRuntime(
        qdrant_client=qdrant_client,
        bm25_index=bm25_index,
        chat_model=get_chat_model(settings),
        base_settings=settings,
        corpus_ready=corpus_ready,
        agent_compiled=build_agent_graph(),
        rag_compiled=build_rag_graph(),
    )


def build_agent_context(runtime: AppRuntime, settings: Settings) -> AgentContext:
    """A fresh `AgentContext` (and the `RagContext`/tools it closes over)
    for one turn, built from `settings` — which may be `runtime.base_settings`
    verbatim or a per-request override copy (`main.chat`'s sidebar knobs).
    Cheap: every argument here is already-connected/constructed, and the
    compiled graphs are reused from `runtime` rather than rebuilt — this
    only allocates closures and dataclasses, no I/O."""
    rag_context = RagContext(
        qdrant_client=runtime.qdrant_client,
        bm25_index=runtime.bm25_index,
        chat_model=runtime.chat_model,
        settings=settings,
    )
    tools = make_tools(runtime.chat_model, runtime.rag_compiled, rag_context)
    return AgentContext(chat_model=runtime.chat_model, tools=tools, settings=settings)
