"""
Export stratified samples for manual inspection of ``normalize_yes_no_idk`` behavior.

Intended for appendix / audit trails; not used to compute primary benchmark numbers.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

from .answer_normalization import normalize_yes_no_idk
from .idkvqa_types import QAExampleResult

AuditStrategy = Literal["random", "first_mismatches", "ambiguous"]


def select_audit_indices(
    results: list[QAExampleResult],
    n: int,
    strategy: AuditStrategy,
    seed: int = 42,
) -> list[int]:
    """Return up to ``n`` indices into ``results`` for export."""
    if not results or n <= 0:
        return []
    rng = random.Random(seed)
    if strategy == "random":
        idxs = list(range(len(results)))
        rng.shuffle(idxs)
        return idxs[: min(n, len(results))]

    if strategy == "first_mismatches":
        out: list[int] = []
        for i, r in enumerate(results):
            if normalize_yes_no_idk(r.final_prediction) != r.ground_truth:
                out.append(i)
            if len(out) >= n:
                break
        return out

    # ambiguous: normalized label differs from raw substring hints (heuristic)
    scored: list[tuple[int, float]] = []
    for i, r in enumerate(results):
        raw_n = normalize_yes_no_idk(r.raw_prediction)
        # crude ambiguity: long free text that normalizes to Yes/No
        amb = 0.0
        if len(r.raw_prediction) > 80 and raw_n in ("Yes", "No"):
            amb = 1.0
        elif "maybe" in r.raw_prediction.lower() or "perhaps" in r.raw_prediction.lower():
            amb = 0.5
        scored.append((i, amb))
    scored.sort(key=lambda x: -x[1])
    return [i for i, _ in scored[: min(n, len(scored))]]


def export_normalization_audit(
    results: list[QAExampleResult],
    path: str | Path,
    n: int,
    strategy: AuditStrategy = "random",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Write JSON array of audit rows: ``sample_id``, question, GT, raw text, normalized, mode, etc.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    idxs = select_audit_indices(results, n, strategy, seed=seed)
    rows: list[dict[str, Any]] = []
    for i in idxs:
        r = results[i]
        raw_norm = normalize_yes_no_idk(r.raw_prediction)
        rows.append({
            "sample_id": r.sample_id,
            "question": r.question,
            "ground_truth": r.ground_truth,
            "raw_prediction_text": r.raw_prediction,
            "normalized_from_raw_text": raw_norm,
            "final_prediction": r.final_prediction,
            "normalized_final": normalize_yes_no_idk(r.final_prediction),
            "mode": r.mode,
            "uncertainty_score": r.uncertainty_score,
            "question_type": r.question_type,
            "question_type_source": "heuristic_coarse_taxonomy",
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "audit_strategy": strategy,
                "audit_size_requested": n,
                "num_rows": len(rows),
                "primary_benchmark": "IDKVQA_offline",
                "samples": rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    return rows
