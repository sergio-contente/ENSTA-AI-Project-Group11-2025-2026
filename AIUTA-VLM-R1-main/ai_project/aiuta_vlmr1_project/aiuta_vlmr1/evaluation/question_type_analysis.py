"""
Per-question-type tables for IDKVQA (types are **heuristic**; see ``idkvqa_kg.coarse_question_taxonomy``).
"""
from __future__ import annotations

from typing import Any

from .idkvqa_types import QAExampleResult, aggregate_idkvqa_metrics


def metrics_by_mode_and_question_type(
    mode_to_results: dict[str, list[QAExampleResult]],
) -> dict[str, Any]:
    """
    Nested structure: ``modes[mode][question_type] = aggregate_idkvqa_metrics chunk``.

    The full per-type dict is under ``per_question_type`` in each mode's aggregate metrics.
    """
    out: dict[str, Any] = {
        "question_type_source": "heuristic_coarse_taxonomy",
        "modes": {},
    }
    for mode, results in mode_to_results.items():
        agg = aggregate_idkvqa_metrics(results)
        out["modes"][mode] = {
            "overall": {k: v for k, v in agg.items() if k != "per_question_type"},
            "per_question_type": agg.get("per_question_type", {}),
        }
    return out
