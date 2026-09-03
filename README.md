# EU AI Act Copilot

An agentic RAG chatbot that answers EU AI Act compliance questions with
article-level citations, classifies AI systems into regulatory risk tiers,
and computes their compliance deadlines — built with LangGraph, served by
vLLM, and evaluated end to end.

> 🚧 **Status: Phase 0 (skeleton).** Repo scaffolding, configuration, the
> Docker/Compose topology, and CI are in place; the RAG subgraph, agent
> graph, tools, and UI land in the phases that follow. See
> [PLAN.md](PLAN.md) for the full design, the phase-by-phase build plan,
> and the reasoning behind every decision below. The original task brief
> (Hungarian) is in [docs/](docs/).

## Quickstart

```bash
cp .env.example .env
make up          # api + ui + qdrant, LLM_PROVIDER=dummy — no GPU needed
```

Then open http://localhost:8501 (UI) or http://localhost:8000/health (API).

To use a real model instead of the dummy fixture, edit `.env` (see the
comments in `.env.example`) and:

```bash
make up-gpu       # requires an NVIDIA GPU + Container Toolkit — see PLAN.md §9
# or
make up-cpu       # CPU-only fallback, slow, no GPU required
```

## Why this project

See [PLAN.md](PLAN.md) §1 for the problem, the target user, and — just as
importantly — the argument for *and against* an agentic architecture over
plain RAG.

## Development

```bash
uv sync           # install dependencies (Python 3.11+)
make check        # lint + type-check + test — exactly what CI runs
```

## Repository layout

See [PLAN.md](PLAN.md) §2.2.
