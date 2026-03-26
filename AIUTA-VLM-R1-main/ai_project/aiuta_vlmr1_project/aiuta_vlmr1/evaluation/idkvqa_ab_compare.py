"""
Compare two IDKVQA runs (A/B), report underclaim deltas by question type,
and optionally export a standardized threshold sweep for both runs.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

from .idkvqa_types import QAExampleResult, aggregate_idkvqa_metrics
from .threshold_sweep import default_tau_grid, sweep_entropy_threshold


def _load_examples(path: Path) -> list[QAExampleResult]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("per_sample") or data.get("results") or data.get("examples") or []
    valid_keys = set(inspect.signature(QAExampleResult).parameters.keys())
    clean_rows = [{k: v for k, v in row.items() if k in valid_keys} for row in rows]
    return [QAExampleResult(**row) for row in clean_rows]


def _underclaim_delta_rows(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
) -> list[dict[str, Any]]:
    q_a = metrics_a.get("per_question_type", {})
    q_b = metrics_b.get("per_question_type", {})
    keys = sorted(set(q_a.keys()) | set(q_b.keys()))
    rows: list[dict[str, Any]] = []
    for key in keys:
        a_row = q_a.get(key, {})
        b_row = q_b.get(key, {})
        ua = float(a_row.get("underclaim_rate_pct", 0.0) or 0.0)
        ub = float(b_row.get("underclaim_rate_pct", 0.0) or 0.0)
        count = int(b_row.get("count", a_row.get("count", 0)) or 0)
        rows.append({
            "question_type": key,
            "count": count,
            "underclaim_A_pct": round(ua, 2),
            "underclaim_B_pct": round(ub, 2),
            "delta_B_minus_A_pct": round(ub - ua, 2),
        })
    return sorted(rows, key=lambda r: (r["delta_B_minus_A_pct"], -r["count"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B compare for IDKVQA JSON runs.")
    parser.add_argument("--input-a", type=str, required=True, help="Baseline run JSON path")
    parser.add_argument("--input-b", type=str, required=True, help="Variant run JSON path")
    parser.add_argument("--label-a", type=str, default="A")
    parser.add_argument("--label-b", type=str, default="B")
    parser.add_argument("--output", type=str, required=True, help="Output JSON summary path")
    parser.add_argument("--export-threshold-sweep-dir", type=str, default=None)
    parser.add_argument("--tau-step", type=float, default=0.05)
    parser.add_argument("--abstention-rule", type=str, default="entropy_above_tau_to_idk")
    args = parser.parse_args()

    in_a = Path(args.input_a)
    in_b = Path(args.input_b)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    res_a = _load_examples(in_a)
    res_b = _load_examples(in_b)
    met_a = aggregate_idkvqa_metrics(res_a)
    met_b = aggregate_idkvqa_metrics(res_b)

    delta_rows = _underclaim_delta_rows(met_a, met_b)
    summary: dict[str, Any] = {
        "labels": {"a": args.label_a, "b": args.label_b},
        "global": {
            "accuracy_a_pct": met_a.get("accuracy"),
            "accuracy_b_pct": met_b.get("accuracy"),
            "coverage_a_pct": met_a.get("coverage"),
            "coverage_b_pct": met_b.get("coverage"),
            "underclaim_a_pct": met_a.get("calibration_abstention", {}).get("underclaim_rate"),
            "underclaim_b_pct": met_b.get("calibration_abstention", {}).get("underclaim_rate"),
        },
        "underclaim_by_question_type": delta_rows,
    }

    if args.export_threshold_sweep_dir:
        sweep_dir = Path(args.export_threshold_sweep_dir)
        sweep_dir.mkdir(parents=True, exist_ok=True)
        taus = default_tau_grid(args.tau_step)
        sweep_a = sweep_entropy_threshold(res_a, taus, rule=args.abstention_rule)
        sweep_b = sweep_entropy_threshold(res_b, taus, rule=args.abstention_rule)
        with open(sweep_dir / f"{args.label_a}_threshold_sweep.json", "w", encoding="utf-8") as f:
            json.dump(sweep_a, f, indent=2, ensure_ascii=False)
        with open(sweep_dir / f"{args.label_b}_threshold_sweep.json", "w", encoding="utf-8") as f:
            json.dump(sweep_b, f, indent=2, ensure_ascii=False)
        summary["threshold_sweep"] = {
            "tau_step": args.tau_step,
            "rule": args.abstention_rule,
            "paths": {
                "a": str(sweep_dir / f"{args.label_a}_threshold_sweep.json"),
                "b": str(sweep_dir / f"{args.label_b}_threshold_sweep.json"),
            },
        }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote A/B summary {out}")


if __name__ == "__main__":
    main()
