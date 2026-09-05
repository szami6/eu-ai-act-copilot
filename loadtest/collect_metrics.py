"""Collect API, vLLM Prometheus, and GPU snapshots during a load run."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


def parse_prometheus(text: str) -> dict[str, list[dict[str, Any]]]:
    """Parse Prometheus exposition lines while preserving labels."""
    result: dict[str, list[dict[str, Any]]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric, _, raw_value = line.rpartition(" ")
        if not metric or not raw_value:
            continue
        name, labels = metric, {}
        if "{" in metric and metric.endswith("}"):
            name, raw_labels = metric.split("{", 1)
            raw_labels = raw_labels[:-1]
            for item in raw_labels.split(","):
                key, _, label_value = item.partition("=")
                labels[key] = label_value.strip('"')
        try:
            value: float | str = float(raw_value)
        except ValueError:
            value = raw_value
        result.setdefault(name, []).append({"labels": labels, "value": value})
    return result


def fetch_text(client: httpx.Client, url: str) -> dict[str, Any]:
    try:
        response = client.get(url)
        response.raise_for_status()
        return {"ok": True, "status_code": response.status_code, "body": response.text}
    except httpx.HTTPError as exc:
        return {"ok": False, "error": str(exc)}


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {"ok": True, "rows": rows}


def collect_snapshot(client: httpx.Client, api_url: str, vllm_url: str) -> dict[str, Any]:
    api = fetch_text(client, f"{api_url.rstrip('/')}/health")
    vllm = fetch_text(client, f"{vllm_url.rstrip('/')}/metrics")
    if vllm.get("ok"):
        vllm["metrics"] = parse_prometheus(vllm.pop("body"))
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "api": api,
        "vllm": vllm,
        "gpu": gpu_snapshot(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--vllm-url", default="http://localhost:8001")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=Path("loadtest/reports/metrics.json"))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")

    with httpx.Client(timeout=5.0) as client:
        snapshots = []
        for index in range(args.samples):
            snapshots.append(collect_snapshot(client, args.api_url, args.vllm_url))
            if index + 1 < args.samples:
                time.sleep(args.interval)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"snapshots": snapshots}, indent=2), encoding="utf-8")
    print(f"Wrote {len(snapshots)} metrics snapshots to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
