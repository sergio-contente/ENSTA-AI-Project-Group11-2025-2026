"""Unit tests for CoIN metrics."""
from aiuta_vlmr1.evaluation.coin_metrics import (
    EpisodeResult, compute_sr, compute_spl, compute_nq, compute_all_metrics,
)

def _ep(success, path_len, shortest, nq):
    return EpisodeResult(
        episode_id="test", split="val_seen", target_category="cabinet",
        success=success, path_length=path_len, shortest_path_length=shortest,
        num_questions=nq,
    )

class TestCoINMetrics:
    def test_sr(self):
        results = [_ep(True, 5, 3, 1), _ep(False, 10, 3, 2), _ep(True, 4, 3, 0)]
        assert abs(compute_sr(results) - 2/3) < 0.01

    def test_spl_perfect(self):
        results = [_ep(True, 3, 3, 0)]  # perfect path
        assert abs(compute_spl(results) - 1.0) < 0.01

    def test_spl_longer_path(self):
        results = [_ep(True, 6, 3, 0)]  # path twice as long
        assert abs(compute_spl(results) - 0.5) < 0.01

    def test_spl_failure(self):
        results = [_ep(False, 10, 3, 0)]
        assert compute_spl(results) == 0.0

    def test_nq(self):
        results = [_ep(True, 3, 3, 2), _ep(True, 3, 3, 0), _ep(False, 3, 3, 3)]
        assert abs(compute_nq(results) - 5/3) < 0.01

    def test_all_metrics(self):
        results = [_ep(True, 3, 3, 1), _ep(False, 10, 3, 2)]
        m = compute_all_metrics(results)
        assert m["num_episodes"] == 2
        assert m["SR"] == 50.0
        assert m["NQ"] == 1.5
