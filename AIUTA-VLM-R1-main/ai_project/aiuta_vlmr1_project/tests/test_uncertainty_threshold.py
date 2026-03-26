"""Uncertainty / abstention helper tests."""
from __future__ import annotations

from aiuta_vlmr1.evaluation.answer_normalization import LABEL_IDK, LABEL_YES
from aiuta_vlmr1.evaluation.uncertainty_abstention import apply_uncertainty_threshold


def test_abstain_high_entropy():
    d = apply_uncertainty_threshold(LABEL_YES, 0.9, 0.5, "entropy_above_tau_to_idk")
    assert d.abstained and d.final_prediction == LABEL_IDK


def test_preserve_when_not_abstaining():
    d = apply_uncertainty_threshold(LABEL_YES, 0.1, 0.5, "entropy_above_tau_to_idk")
    assert not d.abstained and d.final_prediction == LABEL_YES


def test_no_score_no_abstain():
    d = apply_uncertainty_threshold(LABEL_YES, None, 0.5)
    assert not d.abstained
