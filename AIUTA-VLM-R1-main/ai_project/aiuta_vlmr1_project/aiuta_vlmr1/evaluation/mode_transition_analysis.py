"""
Pairwise comparison of IDKVQA benchmark modes (sample-aligned by ``sample_id``).

Used for paper-style transition / error analysis (what KG + threshold changes vs baselines).
"""
from __future__ import annotations

from typing import Any

from .answer_normalization import LABEL_IDK, LABEL_NO, LABEL_YES, normalize_yes_no_idk
from .idkvqa_types import QAExampleResult


def _pred(r: QAExampleResult) -> str:
    return normalize_yes_no_idk(r.final_prediction)


def _is_correct(r: QAExampleResult) -> bool:
    return _pred(r) == r.ground_truth


def _overclaim(r: QAExampleResult) -> bool:
    return r.ground_truth == LABEL_IDK and _pred(r) in (LABEL_YES, LABEL_NO)


def _underclaim(r: QAExampleResult) -> bool:
    return r.ground_truth in (LABEL_YES, LABEL_NO) and _pred(r) == LABEL_IDK


def _confident_wrong(r: QAExampleResult) -> bool:
    p = _pred(r)
    return p in (LABEL_YES, LABEL_NO) and p != r.ground_truth


def compare_mode_transitions(
    baseline_results: list[QAExampleResult],
    improved_results: list[QAExampleResult],
) -> dict[str, Any]:
    """
    Align rows by ``sample_id`` and count correctness / calibration transitions.

    ``baseline_results`` / ``improved_results`` must refer to the same benchmark split
    (same IDs, same ordering not required).
    """
    by_id_base = {r.sample_id: r for r in baseline_results}
    by_id_imp = {r.sample_id: r for r in improved_results}
    common = sorted(set(by_id_base.keys()) & set(by_id_imp.keys()))
    n = len(common)

    def count_pairs(pred_b, pred_i) -> int:
        return sum(
            1 for sid in common
            if pred_b(by_id_base[sid]) and pred_i(by_id_imp[sid])
        )

    cc = sum(
        1 for sid in common
        if _is_correct(by_id_base[sid]) and _is_correct(by_id_imp[sid])
    )
    cw = sum(
        1 for sid in common
        if _is_correct(by_id_base[sid]) and not _is_correct(by_id_imp[sid])
    )
    wc = sum(
        1 for sid in common
        if not _is_correct(by_id_base[sid]) and _is_correct(by_id_imp[sid])
    )
    ww = sum(
        1 for sid in common
        if not _is_correct(by_id_base[sid]) and not _is_correct(by_id_imp[sid])
    )

    out: dict[str, Any] = {
        "num_aligned_samples": n,
        "baseline_correct_to_improved_correct": cc,
        "baseline_correct_to_improved_wrong": cw,
        "baseline_wrong_to_improved_correct": wc,
        "baseline_wrong_to_improved_wrong": ww,
        "baseline_overclaim_to_improved_abstain": count_pairs(_overclaim, lambda r: _pred(r) == LABEL_IDK),
        "baseline_abstain_to_improved_correct": count_pairs(
            lambda b: _pred(b) == LABEL_IDK,
            _is_correct,
        ),
        "baseline_confident_wrong_to_improved_abstain": count_pairs(
            _confident_wrong,
            lambda r: _pred(r) == LABEL_IDK,
        ),
        # When GT is IDK, improved run is correct (typically both abstain / IDK).
        "baseline_gt_idk_improved_correct": sum(
            1 for sid in common
            if by_id_base[sid].ground_truth == LABEL_IDK and _is_correct(by_id_imp[sid])
        ),
        "baseline_correct_to_improved_abstain": count_pairs(
            _is_correct,
            lambda r: _pred(r) == LABEL_IDK,
        ),
    }
    return out


def transitions_to_paper_row(
    baseline_mode: str,
    improved_mode: str,
    transitions: dict[str, Any],
) -> dict[str, Any]:
    """Single CSV/JSON row wrapper for ``compare_mode_transitions`` output."""
    return {
        "baseline_mode": baseline_mode,
        "improved_mode": improved_mode,
        **transitions,
    }
