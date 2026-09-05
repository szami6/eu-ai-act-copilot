"""Shared schema and loading helpers for the end-to-end evaluation set."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    id: str
    question: str
    category: str
    expected_route: Literal["refuse", "direct", "plan"]
    expected_refusal: bool = False
    expected_abstention: bool = False
    expected_tool: str | None = None
    expected_tool_output: dict[str, Any] = Field(default_factory=dict)
    gold_articles: list[str] = Field(default_factory=list)
    gold_chunk_ids: list[str] = Field(default_factory=list)
    reference_answer: str
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)


def load_cases(path: Path) -> list[EvalCase]:
    """Load and validate the complete QA contract before any model calls."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Evaluation set must be a YAML list: {path}")

    cases = [EvalCase.model_validate(item) for item in raw]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation case ids must be unique")
    if len(cases) != 20:
        raise ValueError(f"Expected 20 evaluation cases, found {len(cases)}")
    expected_categories = Counter(
        {
            "single_hop_factual": 6,
            "multi_hop_composite": 4,
            "tool_required": 4,
            "negative_out_of_scope": 3,
            "unanswerable_from_corpus": 3,
        }
    )
    if Counter(case.category for case in cases) != expected_categories:
        raise ValueError(
            "Evaluation categories must contain 6 factual, 4 composite, 4 tool, "
            "3 negative, and 3 unanswerable cases"
        )
    return cases
