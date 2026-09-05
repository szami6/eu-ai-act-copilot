"""Render a compact Markdown report from Locust and node-timing artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from loadtest.scenario import percentile


def load_locust_stats(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def load_node_timings(path: Path) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for node, seconds in record.get("nodes", {}).items():
            values[node].append(float(seconds) * 1000)
    return values


def render_report(
    stats: list[dict[str, Any]],
    node_timings: dict[str, list[float]],
    metrics_path: Path | None = None,
) -> str:
    lines = ["# Load-test report", "", "## Locust request summary", ""]
    if stats:
        lines.extend([
            "| Name | Requests | Failures | Median (ms) | 95th percentile (ms) | RPS |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in stats:
            lines.append(
                "| {Name} | {Request Count} | {Failure Count} | {Median Response Time} | "
                "{95%} | {Requests/s} |".format(**row)
            )
    else:
        lines.append("No Locust stats CSV was found.")

    lines.extend(["", "## Per-node timing", ""])
    if node_timings:
        lines.extend([
            "| Node | Samples | Mean (ms) | p50 (ms) | p95 (ms) |",
            "|---|---:|---:|---:|---:|",
        ])
        for node in sorted(node_timings):
            values = node_timings[node]
            lines.append(
                f"| {node} | {len(values)} | {sum(values) / len(values):.1f} | "
                f"{percentile(values, 0.5):.1f} | {percentile(values, 0.95):.1f} |"
            )
    else:
        lines.append("No node-timing JSONL was found.")

    lines.extend(["", "## Telemetry", ""])
    lines.append(
        f"Metrics snapshots: `{metrics_path}`."
        if metrics_path
        else "Metrics snapshots were not supplied."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=Path("loadtest/reports/run_stats.csv"))
    parser.add_argument(
        "--node-timings", type=Path, default=Path("loadtest/reports/node_timings.jsonl")
    )
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output", type=Path, default=Path("loadtest/reports/loadtest.md"))
    args = parser.parse_args()
    stats = load_locust_stats(args.stats) if args.stats.exists() else []
    report = render_report(stats, load_node_timings(args.node_timings), args.metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
