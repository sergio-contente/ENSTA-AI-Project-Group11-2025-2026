"""
KG vs raw comparison on IDKVQA -- thin wrapper around the primary benchmark.

All logic lives in ``idkvqa_eval.run_idkvqa_benchmark`` and shared metrics in
``idkvqa_types``. This script runs multiple modes and writes one JSON for side-by-side tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..config import Config
from .idkvqa_eval import IDKVQA_MODES, run_idkvqa_benchmark
from .idkvqa_types import QAExampleResult, aggregate_idkvqa_metrics
from .mode_transition_analysis import compare_mode_transitions, transitions_to_paper_row
from .paper_artifacts import export_mode_tables
from .question_type_analysis import metrics_by_mode_and_question_type
from .threshold_sweep import default_tau_grid, sweep_entropy_threshold


def run_kg_vs_raw_comparison(
    config: Config,
    limit: int | None = None,
    seed: int = 42,
    split: str = "val",
    modes: tuple[str, ...] = ("raw", "raw_two_pass", "threshold", "kg", "kg_threshold"),
    output_path: str | Path = "results/kg_comparison/benchmark.json",
    paper_artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run each requested mode and return a combined JSON-serializable dict."""
    bundle: dict[str, Any] = {
        "benchmark": "IDKVQA",
        "role": "kg_vs_raw_ablation",
        "modes": {},
        "config": config.to_serializable_dict(),
        "seed": seed,
        "not_official_coin_online": True,
    }
    mode_results: dict[str, list[QAExampleResult]] = {}
    for m in modes:
        if m not in IDKVQA_MODES:
            raise ValueError(f"Unknown mode {m!r}")
        results = run_idkvqa_benchmark(
            config, mode=m, limit=limit, seed=seed, split=split,
        )
        mode_results[m] = results
        metrics = aggregate_idkvqa_metrics(results)
        bundle["modes"][m] = {
            "metrics": metrics,
            "per_sample": [r.to_serializable() for r in results],
        }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False, default=str)

    if paper_artifacts_dir:
        transition_rows: list[dict[str, Any]] = []
        pairs = [
            ("raw", "threshold"),
            ("raw", "raw_two_pass"),
            ("raw", "kg"),
            ("raw", "kg_threshold"),
            ("raw_two_pass", "kg"),
        ]
        for a, b in pairs:
            if a in mode_results and b in mode_results:
                tr = compare_mode_transitions(mode_results[a], mode_results[b])
                transition_rows.append(transitions_to_paper_row(a, b, tr))

        sweep_rows: list[dict[str, Any]] | None = None
        if "raw" in mode_results:
            sweep_rows = sweep_entropy_threshold(mode_results["raw"], default_tau_grid(0.05))

        qtm = metrics_by_mode_and_question_type(mode_results)

        export_mode_tables(
            mode_results,
            paper_artifacts_dir,
            threshold_sweep_rows=sweep_rows,
            transition_rows=transition_rows,
            question_type_matrix=qtm,
        )

    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="KG vs raw IDKVQA ablation (uses primary benchmark)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/kg_comparison/benchmark.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--paper-artifacts-dir",
        type=str,
        default=None,
        help="If set, write paper_main_benchmark.csv/json, reliability, cost, transitions, sweep, q-type matrix.",
    )
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config) if args.config else Config()
    run_kg_vs_raw_comparison(
        cfg,
        limit=args.limit,
        seed=args.seed,
        output_path=args.output,
        paper_artifacts_dir=args.paper_artifacts_dir,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
