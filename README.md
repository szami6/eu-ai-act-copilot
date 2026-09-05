# EU AI Act Copilot

An agentic RAG chatbot that answers EU AI Act compliance questions with
article-level citations, classifies AI systems into regulatory risk tiers,
and computes their compliance deadlines — built with LangGraph, served by
vLLM, and evaluated end to end.

> 🚧 **Status: Phase 5 foundation.** The ingestion pipeline, modular RAG
> subgraph, deterministic tools, six-node agent graph, streaming API/UI, and
> CI checks are implemented. The retrieval evaluator and the end-to-end
> evaluation generation/judging foundation are now included; measured
> real-model evaluation and load testing remain. See [PLAN.md](PLAN.md) for
> the full design and phase-by-phase build plan. The original task brief
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

make eval-generate  # run the 20-case end-to-end set and persist every output
make eval-judge     # score the latest generation without rerunning the model
make load            # Locust UI; set LOADTEST_HOST for a remote API
make metrics         # sample API/vLLM/GPU telemetry during the run
make report          # render Locust CSV + per-node timings as Markdown
```

Run load tests from a client machine, not the GPU box running the stack. The
Locust scenario samples the 20 evaluation questions, keeps a session per
virtual user, and writes `loadtest/reports/run_*.csv` plus per-node timing
records to `loadtest/reports/node_timings.jsonl`.

## Repository layout

See [PLAN.md](PLAN.md) §2.2.
