"""Reusable uncertainty thresholding and abstention for IDKVQA and related benchmarks."""
from __future__ import annotations

from dataclasses import dataclass

from .answer_normalization import LABEL_IDK, normalize_yes_no_idk


@dataclass
class AbstentionDecision:
    raw_prediction: str
    final_prediction: str
    score: float | None
    threshold: float | None
    abstained: bool


def apply_uncertainty_threshold(
    raw_prediction: str,
    score: float | None,
    threshold: float,
    rule: str = "entropy_above_tau_to_idk",
) -> AbstentionDecision:
    """
    Apply abstention rule to a (possibly unnormalized) raw prediction.

    Rules:
    - entropy_above_tau_to_idk: if normalized entropy score > threshold -> I don't know
    - maxprob_below_tau_to_idk: if max prob score < threshold -> I don't know (score is max prob in [0,1])
    """
    normalized = normalize_yes_no_idk(raw_prediction)

    if score is None:
        return AbstentionDecision(
            raw_prediction=raw_prediction,
            final_prediction=normalized,
            score=None,
            threshold=threshold,
            abstained=False,
        )

    abstain = False
    if rule == "entropy_above_tau_to_idk":
        abstain = score > threshold
    elif rule == "maxprob_below_tau_to_idk":
        abstain = score < threshold
    else:
        raise ValueError(f"Unknown abstention rule: {rule}")

    if abstain:
        return AbstentionDecision(
            raw_prediction=raw_prediction,
            final_prediction=LABEL_IDK,
            score=score,
            threshold=threshold,
            abstained=True,
        )

    return AbstentionDecision(
        raw_prediction=raw_prediction,
        final_prediction=normalized,
        score=score,
        threshold=threshold,
        abstained=False,
    )
