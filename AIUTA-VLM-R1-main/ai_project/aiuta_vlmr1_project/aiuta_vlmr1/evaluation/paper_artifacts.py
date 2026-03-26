"""
Paper-oriented JSON/CSV exports built from ``QAExampleResult`` lists and ``aggregate_idkvqa_metrics``.

All paths assume **IDKVQA** as the primary offline benchmark.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .idkvqa_types import QAExampleResult, aggregate_idkvqa_metrics


def main_benchmark_row(mode: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """One row: headline accuracy, per-class, abstention, coverage, latency, model calls."""
    ca = metrics.get("cost_accounting", {})
    cal = metrics.get("calibration_abstention", {})
    pg = metrics.get("per_gt_class", {})
    return {
        "mode": mode,
        "accuracy_pct": metrics.get("accuracy"),
        "yes_accuracy_pct": pg.get("Yes", {}).get("accuracy_pct"),
        "no_accuracy_pct": pg.get("No", {}).get("accuracy_pct"),
        "idk_accuracy_pct": pg.get("I don't know", {}).get("accuracy_pct"),
        "abstention_rate_pct": cal.get("abstention_rate"),
        "coverage_pct": metrics.get("coverage"),
        "avg_latency_sec": metrics.get("avg_latency_sec"),
        "avg_total_latency_sec": ca.get("avg_total_latency_sec"),
        "avg_num_model_calls": ca.get("avg_num_model_calls"),
    }


def reliability_row(mode: str, metrics: dict[str, Any]) -> dict[str, Any]:
    cal = metrics.get("calibration_abstention", {})
    eff = metrics.get("effective_reliability", {})
    return {
        "mode": mode,
        "overclaim_rate_pct": cal.get("overclaim_rate"),
        "underclaim_rate_pct": cal.get("underclaim_rate"),
        "uncertain_when_uncertain_pct": cal.get("uncertain_when_uncertain_rate"),
        "certain_when_certain_pct": cal.get("certain_when_certain_rate"),
        "phi_c1_pct": eff.get("phi_c1_pct"),
        "phi_c05_pct": eff.get("phi_c05_pct"),
    }


def cost_row(mode: str, metrics: dict[str, Any]) -> dict[str, Any]:
    ca = metrics.get("cost_accounting", {})
    return {
        "mode": mode,
        "avg_num_model_calls": ca.get("avg_num_model_calls"),
        "avg_num_detector_calls": ca.get("avg_num_detector_calls"),
        "avg_num_questioner_calls": ca.get("avg_num_questioner_calls"),
        "avg_num_trigger_calls": ca.get("avg_num_trigger_calls"),
        "avg_num_questions_asked": ca.get("avg_num_questions_asked"),
        "avg_num_kg_nodes": ca.get("avg_num_kg_nodes"),
        "avg_total_latency_sec": ca.get("avg_total_latency_sec"),
        "avg_detector_latency_sec": ca.get("avg_detector_latency_sec"),
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        write_json(path.with_suffix(".json"), [])
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def export_mode_tables(
    mode_to_results: dict[str, list[QAExampleResult]],
    out_dir: str | Path,
    *,
    threshold_sweep_rows: list[dict[str, Any]] | None = None,
    transition_rows: list[dict[str, Any]] | None = None,
    question_type_matrix: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """
    Write ``main_benchmark``, ``reliability``, ``cost`` CSV/JSON; optional sweep / transitions / q-type.
    Returns map artifact name -> path.
    """
    out = Path(out_dir)
    paths: dict[str, Path] = {}

    main_rows: list[dict[str, Any]] = []
    rel_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for mode, results in mode_to_results.items():
        m = aggregate_idkvqa_metrics(results)
        main_rows.append(main_benchmark_row(mode, m))
        rel_rows.append(reliability_row(mode, m))
        cost_rows.append(cost_row(mode, m))

    p_main = out / "paper_main_benchmark.csv"
    p_rel = out / "paper_reliability.csv"
    p_cost = out / "paper_cost.csv"
    write_csv(p_main, main_rows)
    write_csv(p_rel, rel_rows)
    write_csv(p_cost, cost_rows)
    write_json(out / "paper_main_benchmark.json", main_rows)
    write_json(out / "paper_reliability.json", rel_rows)
    write_json(out / "paper_cost.json", cost_rows)
    paths["main_benchmark"] = p_main
    paths["reliability"] = p_rel
    paths["cost"] = p_cost

    if threshold_sweep_rows is not None:
        sweep_main = []
        for row in threshold_sweep_rows:
            sweep_main.append(main_benchmark_row(row.get("mode_key", "sweep"), row["metrics"]))
        write_json(out / "paper_threshold_sweep.json", threshold_sweep_rows)
        write_csv(out / "paper_threshold_sweep.csv", sweep_main)
        paths["threshold_sweep"] = out / "paper_threshold_sweep.json"

    if transition_rows is not None:
        write_json(out / "paper_transitions.json", transition_rows)
        write_csv(out / "paper_transitions.csv", transition_rows)
        paths["transitions"] = out / "paper_transitions.json"

    if question_type_matrix is not None:
        write_json(out / "paper_question_type_matrix.json", question_type_matrix)
        paths["question_type"] = out / "paper_question_type_matrix.json"

    meta = {
        "primary_benchmark": "IDKVQA_offline",
        "artifacts": {k: str(v) for k, v in paths.items()},
    }
    write_json(out / "paper_artifacts_index.json", meta)
    paths["index"] = out / "paper_artifacts_index.json"
    return paths
