"""Streamlit UI — Phase 0 skeleton.

Proves the UI -> API wiring across the compose topology (PLAN.md §2.1,
§2.3) before any chat/agent functionality exists. The chat view and trace
pane (PLAN.md §6) are built in Phase 4.
"""

from __future__ import annotations

import httpx
import streamlit as st

from copilot.config import get_settings

st.set_page_config(page_title="EU AI Act Copilot", page_icon="⚖️")
st.title("⚖️ EU AI Act Copilot")
st.caption("Phase 0 skeleton — chat and the agent trace pane arrive in Phase 4. See PLAN.md.")

settings = get_settings()

st.subheader("API connectivity")
try:
    resp = httpx.get(f"{settings.api_base_url.rstrip('/')}/health", timeout=5.0)
    resp.raise_for_status()
    st.success(f"Connected to API at {settings.api_base_url}")
    st.json(resp.json())
except httpx.HTTPError as exc:
    st.error(f"Could not reach API at {settings.api_base_url}: {exc}")

st.divider()
st.info(
    "This placeholder confirms the Streamlit -> FastAPI -> Qdrant/LLM wiring "
    "works end to end. The chat interface, agent trace timeline, evidence "
    "panel and tool cards described in PLAN.md §6 land in Phase 4, once "
    "the orchestrator graph (Phase 3) exists."
)
