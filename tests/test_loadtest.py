"""Offline tests for load-test parsing and metric helpers."""

from __future__ import annotations

from loadtest.collect_metrics import parse_prometheus
from loadtest.report import render_report
from loadtest.scenario import consume_chat_stream, parse_sse, percentile, summarize_latencies


def test_parse_sse_handles_bytes_and_multiple_data_frames() -> None:
    stream = [
        b"event: session\n",
        b'data: {"session_id":"abc"}\n',
        b"\n",
        "event: done\n",
        'data: {"answer":"ok"}\n',
        "\n",
    ]

    assert parse_sse(stream) == [("session", {"session_id": "abc"}), ("done", {"answer": "ok"})]


def test_consume_chat_stream_requires_done_and_surfaces_errors() -> None:
    summary = consume_chat_stream(
        [
            'event: session\ndata: {"session_id":"abc"}\n\n',
            'event: error\ndata: {"detail":"backend down"}\n\n',
        ]
    )

    assert summary["session_id"] == "abc"
    assert summary["done"] is False
    assert summary["error"] == "backend down"


def test_consume_chat_stream_extracts_node_timings() -> None:
    summary = consume_chat_stream(
        [
            'event: node\ndata: {"node":"triage","duration_s":0.25}\n\n',
            'event: node\ndata: {"node":"execute","duration_s":0.75}\n\n',
            'event: done\ndata: {}\n\n',
        ]
    )

    assert summary["node_durations_s"] == {"triage": 0.25, "execute": 0.75}


def test_percentiles_and_latency_summary_are_deterministic() -> None:
    assert percentile([10, 20, 30, 40], 0.5) == 25
    assert summarize_latencies([10, 20, 30])["p95_ms"] == 29


def test_parse_prometheus_preserves_metric_labels() -> None:
    parsed = parse_prometheus(
        '# HELP demo A test\n'
        'demo{model="qwen",phase="decode"} 4.5\n'
        'demo_total 7\n'
    )

    assert parsed["demo"][0] == {
        "labels": {"model": "qwen", "phase": "decode"},
        "value": 4.5,
    }
    assert parsed["demo_total"][0]["value"] == 7.0


def test_render_report_includes_request_and_node_sections() -> None:
    report = render_report(
        [{
            "Name": "/chat",
            "Request Count": "10",
            "Failure Count": "1",
            "Median Response Time": "100",
            "95%": "200",
            "Requests/s": "2",
        }],
        {"triage": [10.0, 20.0]},
    )

    assert "Locust request summary" in report
    assert "/chat" in report
    assert "triage" in report
