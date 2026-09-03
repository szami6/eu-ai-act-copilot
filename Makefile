.PHONY: up up-gpu up-cpu down build ingest lint typecheck test check bench eval load demo logs

GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo local)
export GIT_SHA

## api + ui + qdrant only. LLM_PROVIDER=dummy in .env -> no model/GPU needed.
up:
	docker compose up --build

## Same, plus the GPU-served LLM (requires the NVIDIA Container Toolkit).
## Set LLM_PROVIDER=openai_compatible and LLM_BASE_URL=http://llm-gpu:8000/v1 in .env first.
up-gpu:
	docker compose --profile gpu up --build

## Same, plus the CPU-served LLM fallback (no GPU required, slow).
## Set LLM_PROVIDER=openai_compatible and LLM_BASE_URL=http://llm-cpu:8000/v1 in .env first.
up-cpu:
	docker compose --profile cpu up --build

down:
	docker compose --profile gpu --profile cpu --profile ingest down -v

build:
	docker compose build

## One-shot corpus ingestion (Phase 1 — not yet implemented). Requires qdrant to be up.
ingest:
	docker compose --profile ingest run --rm ingest

lint:
	uv run ruff check .

typecheck:
	uv run mypy src

test:
	uv run pytest -v

## Everything CI runs, locally.
check: lint typecheck test

## Inference-tier baseline (PLAN.md §8.1). Run ON the GPU box, after `make up-gpu`.
bench:
	bash loadtest/bench_inference.sh

## Functional eval harness (Phase 5 — not yet implemented).
eval:
	uv run python -m evals.run_eval

## End-to-end load test (Phase 6 — not yet implemented).
## Run from a machine that is NOT the system under test (PLAN.md §8.1).
load:
	uv run locust -f loadtest/locustfile.py

## Bring the stack up and confirm the api is answering.
demo: up
	curl -sf http://localhost:8000/health | python -m json.tool

logs:
	docker compose logs -f
