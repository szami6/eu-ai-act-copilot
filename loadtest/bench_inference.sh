#!/usr/bin/env bash
# Inference-tier baseline (PLAN.md §8.1, §8.3) — run this ON THE GPU BOX,
# after `make up-gpu` reports the llm-gpu service healthy.
#
# This is the ceiling every later end-to-end number (PLAN.md §7, §8) is
# measured against: it isolates vLLM's own TTFT / TPOT / throughput from
# everything the agent's Python side adds on top.
#
# Runs `vllm bench serve` *inside* the running llm-gpu container (that image
# already has the vllm CLI installed — this project's own CUDA-free image
# deliberately does not, see PLAN.md §4.1), across a concurrency sweep.
set -euo pipefail

MODEL="${LLM_MODEL:-Qwen/Qwen3-8B-FP8}"
OUT_DIR="loadtest/reports"
mkdir -p "$OUT_DIR"

echo "==> Smoke-testing the llm-gpu service"
docker compose --profile gpu exec llm-gpu curl -sf http://localhost:8000/v1/models > /dev/null || {
  echo "llm-gpu is not reachable. Is 'make up-gpu' healthy? (docker compose ps)" >&2
  exit 1
}

echo "==> Running vllm bench serve (concurrency sweep 1/4/8/16/32)"
echo "    Results land in ${OUT_DIR}/bench_c*.json (via the container's /tmp/bench mount)."
for CONC in 1 4 8 16 32; do
  echo "--- concurrency=${CONC} ---"
  docker compose --profile gpu exec llm-gpu vllm bench serve \
    --base-url "http://localhost:8000/v1" \
    --model "${MODEL}" \
    --dataset-name random \
    --random-input-len 3000 \
    --random-output-len 300 \
    --max-concurrency "${CONC}" \
    --num-prompts "$((CONC * 10))" \
    --save-result \
    --result-dir /tmp/bench \
    --result-filename "bench_c${CONC}.json"
done

echo "==> Done. Summarise ${OUT_DIR}/bench_c*.json into"
echo "    ${OUT_DIR}/inference_baseline.md per PLAN.md §8.3 before Phase 6."
echo ""
echo "NOTE: if the installed vllm version does not have the 'vllm bench serve'"
echo "subcommand (check: docker compose --profile gpu exec llm-gpu vllm bench --help),"
echo "fall back to that version's benchmark_serving.py script with equivalent flags."
