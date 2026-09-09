"""Run the end-to-end evaluation set and persist every turn.

This is deliberately a separate phase from judging. A generation run is
expensive and may use the small serving model; the saved JSON can therefore
be re-judged repeatedly with a larger model or a changed rubric.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from copilot.agent.context import AgentContext
from copilot.agent.graph import _AgentGraph, run_agent
from copilot.api.runtime import build_agent_context, build_app_runtime
from copilot.api.serialize import to_jsonable
from copilot.config import LLMProvider, Settings, get_settings
from copilot.embeddings import EMBEDDING_MODEL_NAME
from copilot.rag.rerank import RERANKER_MODEL_NAME
from evals.cases import EvalCase, load_cases


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_corpus_manifest(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"A valid corpus manifest is required: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Corpus manifest must be a JSON object: {path}")
    return value


def require_model_endpoint(settings: Settings) -> None:
    if settings.llm_provider is LLMProvider.DUMMY:
        return
    url = f"{settings.llm_base_url.rstrip('/')}/models"
    try:
        response = httpx.get(
            url,
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=min(settings.llm_request_timeout_s, 10.0),
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        message = f"LLM endpoint is not reachable from the evaluation host: {url}"
        raise RuntimeError(message) from exc


def record_from_state(
    case: EvalCase, state: Mapping[str, Any], latency_ms: float, error: str | None = None
) -> dict[str, Any]:
    return {
        "id": case.id,
        "question": case.question,
        "expected": case.model_dump(mode="json"),
        "latency_ms": round(latency_ms, 1),
        "error": error,
        "output": to_jsonable(state),
    }


async def generate_cases(
    cases: list[EvalCase], compiled: _AgentGraph, context: AgentContext
) -> list[dict[str, Any]]:
    """Run cases sequentially so the persisted latency is easy to interpret."""
    records: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        try:
            state = await run_agent(
                compiled,
                session_id=f"eval-{case.id}",
                history=[],
                user_input=case.question,
                context=context,
            )
            # The branch above is replaced below; keeping the invocation
            # explicit makes the graph/context boundary visible to callers.
            records.append(record_from_state(case, state, (time.perf_counter() - started) * 1000))
        except Exception as exc:  # one bad case must not discard 19 useful outputs
            records.append(
                record_from_state(
                    case,
                    {"route": None, "messages": [], "tool_calls": [], "evidence": []},
                    (time.perf_counter() - started) * 1000,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/eval/qa_set.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases")
    parser.add_argument(
        "--llm-base-url",
        help="Host-reachable OpenAI-compatible endpoint; overrides LLM_BASE_URL from .env",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.llm_base_url:
        settings = settings.model_copy(update={"llm_base_url": args.llm_base_url})
    cases = load_cases(args.input)
    if args.limit:
        cases = cases[: args.limit]
    require_model_endpoint(settings)
    runtime = build_app_runtime(settings)
    corpus_manifest = load_corpus_manifest(settings.data_dir)
    expected_chunks = int(corpus_manifest.get("chunk_count", -1))
    if not runtime.corpus_ready or len(runtime.bm25_index.chunks) != expected_chunks:
        raise RuntimeError(
            "Qdrant corpus does not match data/manifest.json: "
            f"expected {expected_chunks} chunks, found {len(runtime.bm25_index.chunks)}"
        )
    context = build_agent_context(runtime, settings)
    records = asyncio.run(generate_cases(cases, runtime.agent_compiled, context))
    generated_at = datetime.now(UTC)
    sha = git_sha()
    run_id = f"{sha}-{generated_at.strftime('%Y%m%dT%H%M%SZ')}"
    output = args.output or Path("evals/reports") / f"generation_{run_id}.json"
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "generated_at": generated_at.isoformat(),
        "git_sha": sha,
        "dataset": {"path": str(args.input), "sha256": file_sha256(args.input)},
        "corpus_manifest": corpus_manifest,
        "settings": {
            "llm_provider": settings.llm_provider.value,
            "llm_model": settings.llm_model,
            "llm_base_url": settings.llm_base_url,
            "temperature": 0,
            "llm_request_timeout_s": settings.llm_request_timeout_s,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "reranker_model": RERANKER_MODEL_NAME,
            "rerank_enabled": settings.rerank_enabled,
            "rag_default_k": settings.rag_default_k,
            "groundedness_threshold": settings.groundedness_threshold,
            "max_plan_steps": settings.max_plan_steps,
            "max_verify_retries": settings.max_verify_retries,
        },
        "cases": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {len(records)} cases -> {output}")
    errors = sum(bool(record["error"]) for record in records)
    if errors:
        print(f"Generation completed with {errors} errored case(s)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
