"""
Entropy-threshold sweep on frozen VQA outputs (``entropy_score`` + raw label).

Uses the same uncertainty rule as ``run_idkvqa_benchmark`` threshold mode; output rows match
``paper_artifacts`` table schema for integration with paper tables.
"""
from __future__ import annotations

from statistics import mean
from typing import Any

from .answer_normalization import LABEL_IDK, normalize_yes_no_idk
from .idkvqa_types import QAExampleResult, aggregate_idkvqa_metrics
from .uncertainty_abstention import apply_uncertainty_threshold


def sweep_entropy_threshold(
    results: list[QAExampleResult],
    taus: list[float],
    rule: str = "entropy_above_tau_to_idk",
) -> list[dict[str, Any]]:
    """
    For each ``tau``, apply thresholding to the **same** per-sample ``entropy_score`` and the
    model's pre-threshold normalized label (``raw_prediction_label``).

    Returns one row per tau, schema-compatible with ``main_benchmark_row``.
    """
    rows: list[dict[str, Any]] = []
    for tau in taus:
        synth: list[QAExampleResult] = []
        for r in results:
            base_label = normalize_yes_no_idk(r.raw_prediction_label or r.metadata.get("raw_prediction_normalized", r.raw_prediction))
            dec = apply_uncertainty_threshold(base_label, r.entropy_score, tau, rule)
            synth.append(
                QAExampleResult(
                    sample_id=r.sample_id,
                    question=r.question,
                    ground_truth=r.ground_truth,
                    raw_prediction=r.raw_prediction,
                    final_prediction=dec.final_prediction,
                    confidence_score=r.confidence_score,
                    entropy_score=r.entropy_score,
                    used_kg=False,
                    used_threshold=True,
                    used_abstention=dec.abstained,
                    latency_sec=r.latency_sec,
                    metadata={**r.metadata, "sweep_tau": tau, "sweep_rule": rule},
                    question_type=r.question_type,
                    mode=f"sweep_threshold@{tau}",
                    raw_prediction_label=r.raw_prediction_label,
                    correct=normalize_yes_no_idk(dec.final_prediction) == r.ground_truth,
                    num_model_calls=r.num_model_calls,
                    num_detector_calls=r.num_detector_calls,
                    num_questioner_calls=r.num_questioner_calls,
                    num_trigger_calls=r.num_trigger_calls,
                    num_questions_asked=r.num_questions_asked,
                    num_kg_nodes=r.num_kg_nodes,
                    total_latency_sec=r.total_latency_sec,
                    detector_latency_sec=r.detector_latency_sec,
                    decision_latency_sec=r.decision_latency_sec,
                    uncertainty_score=r.uncertainty_score,
                    threshold=tau,
                    abstained=normalize_yes_no_idk(dec.final_prediction) == LABEL_IDK,
                )
            )
        agg = aggregate_idkvqa_metrics(synth)
        rows.append({
            "mode_key": f"entropy_sweep_tau_{tau}",
            "tau": tau,
            "rule": rule,
            "metrics": agg,
        })
    return rows


def default_tau_grid(step: float = 0.05) -> list[float]:
    return [round(i * step, 4) for i in range(int(1.0 / step) + 1)]


def sweep_mean_latency_sec(rows: list[dict[str, Any]]) -> float:
    """Helper: average ``avg_total_latency_sec`` from embedded metrics (if present)."""
    vals = [r["metrics"]["cost_accounting"]["avg_total_latency_sec"] for r in rows if "metrics" in r]
    return float(mean(vals)) if vals else 0.0

if __name__ == "__main__":
    import argparse
    import json
    from aiuta_vlmr1.evaluation.idkvqa_types import QAExampleResult

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--step", type=float, default=0.02)
    args = parser.parse_args()

    # [*] Load JSON
    with open(args.input) as f:
        data = json.load(f)

    import inspect

    # [*] Detect correct key
    examples = (
        data.get("examples")
        or data.get("results")
        or data.get("per_sample")
    )

    if examples is None:
        raise ValueError("No per-sample data found in input JSON")

    # [*] Filter fields dynamically
    valid_keys = set(inspect.signature(QAExampleResult).parameters.keys())

    def filter_dict(d):
        return {k: v for k, v in d.items() if k in valid_keys}

    results = [QAExampleResult(**filter_dict(r)) for r in examples]

    # [*] Sweep
    taus = default_tau_grid(step=args.step)
    rows = sweep_entropy_threshold(results, taus)

    # [*] Save
    with open(args.output, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"[OK] Sweep saved to {args.output}")
