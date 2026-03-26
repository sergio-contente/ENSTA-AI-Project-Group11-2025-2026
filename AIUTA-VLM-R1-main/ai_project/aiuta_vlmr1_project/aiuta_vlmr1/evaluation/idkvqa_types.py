"""Types and aggregate metrics for IDKVQA offline benchmarking."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from statistics import mean
from typing import Any

from .answer_normalization import LABEL_IDK, LABEL_NO, LABEL_YES, normalize_yes_no_idk


@dataclass
class QAExampleResult:
    """
    Single-sample result for the unified IDKVQA benchmark.

    ``question_type`` is **heuristic** (see ``idkvqa_kg.coarse_question_taxonomy``); the dataset
    does not ship authoritative type labels.
    """
    sample_id: str
    question: str
    ground_truth: str
    raw_prediction: str
    final_prediction: str
    confidence_score: float | None
    entropy_score: float | None
    used_kg: bool
    used_threshold: bool
    used_abstention: bool
    latency_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)
    question_type: str = "unknown_other"
    mode: str = ""
    raw_prediction_label: str = ""
    correct: bool = False
    num_model_calls: int = 0
    num_detector_calls: int = 0
    num_questioner_calls: int = 0
    num_trigger_calls: int = 0
    num_questions_asked: int = 0
    num_kg_nodes: int | None = None
    total_latency_sec: float = 0.0
    detector_latency_sec: float | None = None
    decision_latency_sec: float | None = None
    uncertainty_score: float | None = None
    threshold: float | None = None
    abstained: bool = False
    annotator_answers: dict[str, int] | None = None

    def to_serializable(self) -> dict[str, Any]:
        d = asdict(self)
        d["raw_prediction_text"] = self.raw_prediction
        return d


def compute_effective_reliability_binary(
    predictions: list[str],
    ground_truths: list[str],
    cost: float = 1.0,
) -> float:
    """Phi_c style (binary matching): penalize wrong confident answers; IDK scores 0."""
    n = len(predictions)
    if n == 0:
        return 0.0
    n_correct = 0
    n_wrong_confident = 0
    for pred, gt in zip(predictions, ground_truths, strict=True):
        p = normalize_yes_no_idk(pred)
        if p == gt:
            n_correct += 1
        elif p != LABEL_IDK:
            n_wrong_confident += 1
    return (n_correct - cost * n_wrong_confident) / n


# Backward compatibility alias
compute_effective_reliability = compute_effective_reliability_binary


def compute_effective_reliability_coin(
    predictions: list[str],
    ground_truths: list[str],
    answers_list: list[dict[str, int] | None],
    cost: float = 1.0,
) -> float:
    """
    CoIN-paper VQAEvaluator formula: score = min(k/3, 1) if k>0, else -cost.

    For each sample:
    - If model predicts IDK: score = 0 (abstention is neutral).
    - If model predicts Yes/No: k = number of annotators who agree with the prediction.
      score = min(k/3, 1) if k > 0, else -cost.
    - ER = mean(scores) * 100.

    ``answers_list[i]`` is the annotator votes dict, e.g. {"Yes": 3, "No": 1, "I don't know": 1}.
    If an entry is None, that sample is scored with binary matching as fallback.
    """
    n = len(predictions)
    if n == 0:
        return 0.0
    total = 0.0
    for pred, gt, answers in zip(predictions, ground_truths, answers_list, strict=True):
        p = normalize_yes_no_idk(pred)
        if p == LABEL_IDK:
            total += 0.0
            continue
        if answers is None:
            # Fallback to binary matching when annotator votes unavailable
            total += 1.0 if p == gt else -cost
            continue
        k = answers.get(p, 0)
        if k > 0:
            total += min(k / 3.0, 1.0)
        else:
            total += -cost
    return total / n


def _metrics_core(
    results: list[QAExampleResult],
) -> tuple[list[str], list[str], list[bool], Any]:
    preds = [normalize_yes_no_idk(r.final_prediction) for r in results]
    gts = [r.ground_truth for r in results]
    correct = [p == g for p, g in zip(preds, gts, strict=True)]
    return preds, gts, correct, None


def aggregate_idkvqa_metrics(
    results: list[QAExampleResult],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Global, per-class, calibration, cost, and per-question-type metrics."""
    n = len(results)
    if n == 0:
        return {"num_samples": 0}

    preds, gts, correct, _ = _metrics_core(results)
    accuracy = sum(correct) / n

    def acc_for(gt_val: str) -> tuple[int, float]:
        sub = [i for i, g in enumerate(gts) if g == gt_val]
        if not sub:
            return 0, 0.0
        return len(sub), sum(correct[i] for i in sub) / len(sub)

    n_yes, acc_yes = acc_for(LABEL_YES)
    n_no, acc_no = acc_for(LABEL_NO)
    n_idk, acc_idk = acc_for(LABEL_IDK)

    from collections import Counter

    pred_dist = Counter(preds)
    frac_yes = pred_dist.get(LABEL_YES, 0) / n
    frac_no = pred_dist.get(LABEL_NO, 0) / n
    frac_idk = pred_dist.get(LABEL_IDK, 0) / n

    gt_idk_indices = [i for i, g in enumerate(gts) if g == LABEL_IDK]
    gt_certain_indices = [i for i, g in enumerate(gts) if g in (LABEL_YES, LABEL_NO)]

    overclaim = sum(1 for i in gt_idk_indices if preds[i] in (LABEL_YES, LABEL_NO))
    underclaim = sum(1 for i in gt_certain_indices if preds[i] == LABEL_IDK)

    uncertain_when_uncertain = sum(1 for i in gt_idk_indices if preds[i] == LABEL_IDK)
    certain_when_certain = sum(1 for i in gt_certain_indices if preds[i] in (LABEL_YES, LABEL_NO))

    overclaim_rate = overclaim / len(gt_idk_indices) if gt_idk_indices else 0.0
    underclaim_rate = underclaim / len(gt_certain_indices) if gt_certain_indices else 0.0
    uwu_rate = uncertain_when_uncertain / len(gt_idk_indices) if gt_idk_indices else 0.0
    cwc_rate = certain_when_certain / len(gt_certain_indices) if gt_certain_indices else 0.0

    abstention_rate = frac_idk
    coverage = 1.0 - abstention_rate
    thresholded = [r for r in results if r.used_threshold]
    threshold_coverage = len(thresholded) / n if n else 0.0

    entropies = [r.entropy_score for r in results if r.entropy_score is not None and r.entropy_score >= 0]
    entropy_summary: dict[str, Any] = {}
    if entropies:
        entropy_summary = {
            "mean": round(mean(entropies), 4),
            "min": round(min(entropies), 4),
            "max": round(max(entropies), 4),
        }

    phi_1 = compute_effective_reliability_binary(preds, gts, cost=1.0)
    phi_05 = compute_effective_reliability_binary(preds, gts, cost=0.5)

    # CoIN-paper formula using annotator agreement (when available)
    answers_list = [r.annotator_answers for r in results]
    has_annotator_answers = any(a is not None for a in answers_list)
    if has_annotator_answers:
        phi_coin_1 = compute_effective_reliability_coin(preds, gts, answers_list, cost=1.0)
        phi_coin_05 = compute_effective_reliability_coin(preds, gts, answers_list, cost=0.5)
    else:
        phi_coin_1 = None
        phi_coin_05 = None

    cost_accounting = {
        "avg_num_model_calls": round(mean(r.num_model_calls for r in results), 4),
        "avg_num_detector_calls": round(mean(r.num_detector_calls for r in results), 4),
        "avg_num_questioner_calls": round(mean(r.num_questioner_calls for r in results), 4),
        "avg_num_trigger_calls": round(mean(r.num_trigger_calls for r in results), 4),
        "avg_num_questions_asked": round(mean(r.num_questions_asked for r in results), 4),
        "avg_total_latency_sec": round(mean(r.total_latency_sec or r.latency_sec for r in results), 4),
        "avg_detector_latency_sec": round(
            mean([r.detector_latency_sec for r in results if r.detector_latency_sec is not None]), 4,
        )
        if any(r.detector_latency_sec is not None for r in results)
        else 0.0,
    }
    decs = [r.decision_latency_sec for r in results if r.decision_latency_sec is not None]
    if decs:
        cost_accounting["avg_decision_latency_sec"] = round(mean(decs), 6)
    kg_vals = [r.num_kg_nodes for r in results if r.num_kg_nodes is not None]
    if kg_vals:
        cost_accounting["avg_num_kg_nodes"] = round(mean(kg_vals), 4)

    per_qt: dict[str, Any] = {}
    by_type: dict[str, list[QAExampleResult]] = {}
    for r in results:
        qt = r.question_type or "unknown_other"
        by_type.setdefault(qt, []).append(r)

    for qt, sub in sorted(by_type.items()):
        sp, sg, sc, _ = _metrics_core(sub)
        if not sub:
            continue
        sub_n = len(sub)
        sub_preds = [normalize_yes_no_idk(x.final_prediction) for x in sub]
        sub_gts = [x.ground_truth for x in sub]
        sub_correct = [a == b for a, b in zip(sub_preds, sub_gts, strict=True)]
        gt_idk_i = [i for i, g in enumerate(sub_gts) if g == LABEL_IDK]
        gt_cert_i = [i for i, g in enumerate(sub_gts) if g in (LABEL_YES, LABEL_NO)]
        oc = sum(1 for i in gt_idk_i if sub_preds[i] in (LABEL_YES, LABEL_NO))
        uc = sum(1 for i in gt_cert_i if sub_preds[i] == LABEL_IDK)
        per_qt[qt] = {
            "count": sub_n,
            "accuracy_pct": round(sum(sub_correct) / sub_n * 100, 2),
            "yes_accuracy_pct": _acc_pct(sub, LABEL_YES),
            "no_accuracy_pct": _acc_pct(sub, LABEL_NO),
            "idk_accuracy_pct": _acc_pct(sub, LABEL_IDK),
            "overclaim_rate_pct": round(oc / len(gt_idk_i) * 100, 2) if gt_idk_i else 0.0,
            "underclaim_rate_pct": round(uc / len(gt_cert_i) * 100, 2) if gt_cert_i else 0.0,
            "abstention_rate_pct": round(
                sum(1 for p in sub_preds if p == LABEL_IDK) / sub_n * 100, 2,
            ),
        }

    out: dict[str, Any] = {
        "num_samples": n,
        "accuracy": round(accuracy * 100, 2),
        "exact_match_accuracy": round(accuracy * 100, 2),
        "avg_latency_sec": round(mean(r.latency_sec for r in results), 4),
        "prediction_distribution": {
            "fraction_yes": round(frac_yes * 100, 2),
            "fraction_no": round(frac_no * 100, 2),
            "fraction_i_dont_know": round(frac_idk * 100, 2),
        },
        "coverage": round(coverage * 100, 2),
        "per_gt_class": {
            LABEL_YES: {"count": n_yes, "accuracy_pct": round(acc_yes * 100, 2)},
            LABEL_NO: {"count": n_no, "accuracy_pct": round(acc_no * 100, 2)},
            LABEL_IDK: {"count": n_idk, "accuracy_pct": round(acc_idk * 100, 2)},
        },
        "calibration_abstention": {
            "overclaim_rate": round(overclaim_rate * 100, 2),
            "underclaim_rate": round(underclaim_rate * 100, 2),
            "uncertain_when_uncertain_rate": round(uwu_rate * 100, 2),
            "certain_when_certain_rate": round(cwc_rate * 100, 2),
            "abstention_rate": round(abstention_rate * 100, 2),
            "threshold_coverage": round(threshold_coverage * 100, 2),
        },
        "effective_reliability": {
            "phi_c1_pct": round(phi_1 * 100, 2),
            "phi_c05_pct": round(phi_05 * 100, 2),
            **({"phi_coin_c1_pct": round(phi_coin_1 * 100, 2),
                "phi_coin_c05_pct": round(phi_coin_05 * 100, 2)}
               if phi_coin_1 is not None else {}),
        },
        "entropy_summary": entropy_summary,
        "cost_accounting": cost_accounting,
        "per_question_type": per_qt,
        "question_type_source": "heuristic_coarse_taxonomy",
        "primary_benchmark": "IDKVQA_offline",
        "note_official_coin": (
            "These metrics are for offline IDKVQA; they are not comparable to online Habitat CoIN SR/SPL."
        ),
    }
    if extra:
        out["extra"] = extra
    return out


def _acc_pct(samples: list[QAExampleResult], gt_val: str) -> float:
    sub = [r for r in samples if r.ground_truth == gt_val]
    if not sub:
        return 0.0
    preds = [normalize_yes_no_idk(r.final_prediction) for r in sub]
    return round(sum(1 for p, r in zip(preds, sub) if p == r.ground_truth) / len(sub) * 100, 2)


def benchmark_run_metadata(
    config_dict: dict[str, Any],
    mode: str,
    seed: int,
    model_id: str,
    processor_id: str,
) -> dict[str, Any]:
    return {
        "config": config_dict,
        "seed": seed,
        "mode": mode,
        "model": {"model_id": model_id, "processor_id": processor_id},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_role": "primary_offline_calibration",
    }
