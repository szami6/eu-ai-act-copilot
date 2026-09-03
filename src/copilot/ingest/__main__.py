"""Entrypoint for `python -m copilot.ingest` (the `ingest` compose service).

This stub exists so that service is a valid, runnable target from Phase 0
onward rather than a dangling reference — real ingestion (fetch, parse,
chunk, index) lands in Phase 1 (PLAN.md §5).
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "Ingestion is not implemented yet (Phase 1, see PLAN.md §5). "
        "Nothing to do; exiting cleanly.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
