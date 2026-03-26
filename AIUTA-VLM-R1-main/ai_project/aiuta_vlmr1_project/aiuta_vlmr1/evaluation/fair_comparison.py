"""
Fair comparison table across IDKVQA modes.

Reports binary AND CoIN reliability metrics side by side, with explicit
notes about which thresholds were tuned on the val set vs. fixed heuristics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .answer_normalization import LABEL_IDK, normalize_yes_no_idk
from .idkvqa_types import (
    QAExampleResult,
    aggregate_idkvqa_metrics,
    compute_effective_reliability_binary,
    compute_effective_reliability_coin,
)
from .uncertainty_abstention import apply_uncertainty_threshold

FAIR_COLUMNS = (
    "mode",
    "calls",
    "accuracy_pct",
    "phi_binary_c1_pct",
    "phi_coin_c1_pct",
    "abstention_rate_pct",
    "overclaim_rate_pct",
    "note",
)

_MODE_NOTES: dict[str, str] = {
    "raw": "no gating",
    "threshold": "tau swept on val",
    "kg": "fixed heuristics",
    "kg_threshold": "tau swept on val + fixed KG heuristics",
    "raw_two_pass": "compute control (no KG)",
    "two_pass_kg": "fixed heuristics",
    "two_pass_kg_relaxed": "fixed heuristics",
    "two_pass_kg_entropy": "fixed heuristics",
}

TAU_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)


def _rescore_threshold(
    results: list[QAExampleResult],
    tau: float,
    rule: str = "entropy_above_tau_to_idk",
) -> list[QAExampleResult]:
    """Re-apply a different tau to per-sample results (offline sweep)."""
    rescored: list[QAExampleResult] = []
    for r in results:
        dec = apply_uncertainty_threshold(
            r.raw_prediction, r.uncertainty_score, tau, rule,
        )
        from dataclasses import replace
        rescored.append(replace(
            r,
            final_prediction=dec.final_prediction,
            abstained=dec.abstained,
        ))
    return rescored


def build_fair_row(
    mode: str,
    results: list[QAExampleResult],
    note: str | None = None,
) -> dict[str, Any]:
    """Build one row of the fair comparison table."""
    metrics = aggregate_idkvqa_metrics(results)
    ca = metrics.get("cost_accounting", {})
    cal = metrics.get("calibration_abstention", {})
    eff = metrics.get("effective_reliability", {})
    return {
        "mode": mode,
        "calls": ca.get("avg_num_model_calls", 0),
        "accuracy_pct": metrics.get("accuracy", 0),
        "phi_binary_c1_pct": eff.get("phi_c1_pct", 0),
        "phi_coin_c1_pct": eff.get("phi_coin_c1_pct"),
        "abstention_rate_pct": cal.get("abstention_rate", 0),
        "overclaim_rate_pct": cal.get("overclaim_rate", 0),
        "note": note or _MODE_NOTES.get(mode, ""),
    }


def run_fair_comparison(
    results_dir: str,
    output_path: str,
) -> list[dict[str, Any]]:
    """
    Load all IDKVQA result JSONs from ``results_dir``, build a fair comparison table.

    For ``threshold`` mode, re-scores with multiple tau values.
    """
    results_path = Path(results_dir)
    runs: dict[str, list[QAExampleResult]] = {}

    for json_file in sorted(results_path.glob("*.json")):
        # Skip aggregate outputs (list of rows) and other non-benchmark JSON.
        if json_file.name in ("fair_table.json",):
            continue
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        mode = data.get("mode")
        per_sample = data.get("per_sample", [])
        if not mode or not per_sample:
            continue
        results_list: list[QAExampleResult] = []
        for row in per_sample:
            results_list.append(QAExampleResult(
                sample_id=str(row.get("sample_id", "")),
                question=row.get("question", ""),
                ground_truth=row.get("ground_truth", ""),
                raw_prediction=row.get("raw_prediction", ""),
                final_prediction=row.get("final_prediction", ""),
                confidence_score=row.get("confidence_score"),
                entropy_score=row.get("entropy_score"),
                used_kg=row.get("used_kg", False),
                used_threshold=row.get("used_threshold", False),
                used_abstention=row.get("used_abstention", False),
                latency_sec=float(row.get("latency_sec", 0)),
                metadata=row.get("metadata", {}),
                mode=mode,
                raw_prediction_label=row.get("raw_prediction_label", ""),
                correct=row.get("correct", False),
                num_model_calls=int(row.get("num_model_calls", 0)),
                num_detector_calls=int(row.get("num_detector_calls", 0)),
                num_questioner_calls=int(row.get("num_questioner_calls", 0)),
                num_trigger_calls=int(row.get("num_trigger_calls", 0)),
                num_questions_asked=int(row.get("num_questions_asked", 0)),
                num_kg_nodes=row.get("num_kg_nodes"),
                total_latency_sec=float(row.get("total_latency_sec", 0)),
                detector_latency_sec=row.get("detector_latency_sec"),
                decision_latency_sec=row.get("decision_latency_sec"),
                uncertainty_score=row.get("uncertainty_score"),
                threshold=row.get("threshold"),
                abstained=row.get("abstained", False),
                annotator_answers=row.get("annotator_answers"),
            ))
        runs[mode] = results_list

    rows: list[dict[str, Any]] = []
    for mode, results_list in sorted(runs.items()):
        if mode == "threshold":
            for tau in TAU_GRID:
                rescored = _rescore_threshold(results_list, tau)
                rows.append(build_fair_row(
                    f"threshold_tau={tau:.2f}",
                    rescored,
                    note=f"tau={tau:.2f} swept on val",
                ))
        else:
            rows.append(build_fair_row(mode, results_list))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False, default=str)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair comparison table across IDKVQA modes")
    parser.add_argument("--results-dir", type=str, required=True, help="Dir with IDKVQA result JSONs")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    args = parser.parse_args()
    rows = run_fair_comparison(args.results_dir, args.output)
    print(json.dumps(rows, indent=2, ensure_ascii=False, default=str))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
