"""Locust client for the streaming `/chat` endpoint.

Run from a machine other than the system under test, for example::

    locust -f loadtest/locustfile.py --host http://gpu-box:8000

The default user mixes all 20 evaluation questions. Set ``LOADTEST_CASES``
to a comma-separated subset of ids, and use ``LOADTEST_TOP_K`` or
``LOADTEST_RERANK_ENABLED`` to reproduce retrieval ablations.
"""

from __future__ import annotations

import json
import os
import random
import threading
from pathlib import Path

from locust import HttpUser, between, task

from loadtest.scenario import consume_chat_stream, load_load_cases

_TIMING_LOCK = threading.Lock()


def _write_node_timings(case_id: str, timings: dict[str, float]) -> None:
    path = Path(os.getenv("LOADTEST_NODE_TIMINGS", "loadtest/reports/node_timings.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with _TIMING_LOCK, path.open("a", encoding="utf-8") as output:
        output.write(json.dumps({"case_id": case_id, "nodes": timings}) + "\n")


class ChatUser(HttpUser):
    host = os.getenv("LOADTEST_HOST", "http://localhost:8000")
    wait_time = between(1.0, 3.0)

    def on_start(self) -> None:
        selected = {
            item.strip()
            for item in os.getenv("LOADTEST_CASES", "").split(",")
            if item.strip()
        }
        cases = load_load_cases(Path(os.getenv("LOADTEST_QA_SET", "data/eval/qa_set.yaml")))
        self.cases = [case for case in cases if not selected or case.case_id in selected]
        if not self.cases:
            raise ValueError("LOADTEST_CASES did not select any known evaluation cases")
        self.session_id: str | None = None
        self.random = random.Random()

    @task
    def chat(self) -> None:
        case = self.random.choice(self.cases)
        payload: dict[str, object] = {"message": case.question}
        if self.session_id:
            payload["session_id"] = self.session_id
        if top_k := os.getenv("LOADTEST_TOP_K"):
            payload["top_k"] = int(top_k)
        if rerank := os.getenv("LOADTEST_RERANK_ENABLED"):
            payload["rerank_enabled"] = rerank.casefold() in {"1", "true", "yes", "on"}

        with self.client.post(
            "/chat",
            json=payload,
            name=f"/chat [{case.category}]",
            stream=True,
            catch_response=True,
            timeout=float(os.getenv("LOADTEST_TIMEOUT_S", "120")),
        ) as response:
            summary = consume_chat_stream(response.iter_lines(decode_unicode=True))
            _write_node_timings(case.case_id, summary["node_durations_s"])
            if summary["session_id"]:
                self.session_id = summary["session_id"]
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
            elif summary["error"]:
                response.failure(str(summary["error"]))
            elif not summary["done"]:
                response.failure("stream ended without a done event")
