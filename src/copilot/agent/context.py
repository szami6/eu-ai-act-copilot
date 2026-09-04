"""Run-scoped orchestrator dependencies, injected via `Runtime[AgentContext]`
(PLAN.md §3.1), mirroring `rag.graph.RagContext`.

Kept in its own module rather than in `graph.py`: `graph.py` imports every
node from `nodes/`, and each node needs `AgentContext` to type its
`Runtime[AgentContext]` parameter, so defining it in `graph.py` would make
`nodes/` import back from the module that imports `nodes/` — a cycle. Both
sides importing this instead avoids it.

`tools` is built once by `agent.tools.bindings.make_tools` (closing over the
chat model and compiled RAG subgraph) and handed in already-assembled —
`execute` just looks up `tools[subtask.tool]`.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from copilot.config import Settings


@dataclass
class AgentContext:
    chat_model: BaseChatModel
    tools: dict[str, BaseTool]
    settings: Settings
