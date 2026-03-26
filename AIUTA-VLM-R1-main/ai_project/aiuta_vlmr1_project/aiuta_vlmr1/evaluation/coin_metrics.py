"""
coin_metrics.py -- Online / embodied CoIN-style metrics: SR, SPL, NQ.

These require **Habitat-style navigation** (path length, success distance, etc.).
They are **not** produced by offline IDKVQA static QA runs or by the auxiliary
``episode_runner`` offline static integration path.

Metrics (from CoIN paper, Section 5):
  - SR (Success Rate): fraction of episodes where agent stops within
    threshold distance of the target
  - SPL (Success weighted by Path Length): SR penalized by path efficiency
  - NQ (Number of Questions): avg number of user interactions per episode

Also includes IoU-based detection metrics reused from evaluate_coco.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


# CoIN success threshold (meters) -- from GOAT-Bench / Habitat standard
SUCCESS_DISTANCE_THRESHOLD = 1.0  # meters


@dataclass
class EpisodeResult:
    """Result from a single CoIN episode."""
    episode_id: str
    split: str
    target_category: str
    success: bool                       # did agent stop within threshold?
    path_length: float                  # actual path length (meters)
    shortest_path_length: float         # geodesic distance (meters)
    num_questions: int                  # NQ for this episode
    num_detections: int = 0             # how many times on_detection was called
    num_kg_nodes: int = 0               # KG objects at end of episode
    final_distance_to_target: float = float("inf")
    total_timesteps: int = 0
    total_inference_time: float = 0.0   # sum of VLM-R1 latencies
    episode_log: list[dict] = field(default_factory=list)


def compute_spl(results: Sequence[EpisodeResult]) -> float:
    """
    Success weighted by Path Length.

    SPL = (1/N) * sum_i [ S_i * (l_i / max(p_i, l_i)) ]

    Where:
      S_i = 1 if success, 0 otherwise
      l_i = shortest path length (geodesic)
      p_i = actual path length
    """
    if not results:
        return 0.0

    total = 0.0
    for r in results:
        if r.success:
            if r.path_length > 0:
                total += r.shortest_path_length / max(r.path_length, r.shortest_path_length)
            else:
                total += 1.0  # perfect path (shouldn't happen but just in case)

    return total / len(results)


def compute_sr(results: Sequence[EpisodeResult]) -> float:
    """Success Rate: fraction of successful episodes."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.success) / len(results)


def compute_nq(results: Sequence[EpisodeResult]) -> float:
    """Average Number of Questions across all episodes."""
    if not results:
        return 0.0
    return sum(r.num_questions for r in results) / len(results)


def compute_all_metrics(results: Sequence[EpisodeResult]) -> dict:
    """Compute all CoIN metrics at once."""
    if not results:
        return {"SR": 0.0, "SPL": 0.0, "NQ": 0.0, "num_episodes": 0}

    sr = compute_sr(results)
    spl = compute_spl(results)
    nq = compute_nq(results)

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    return {
        "SR": round(sr * 100, 2),
        "SPL": round(spl * 100, 2),
        "NQ": round(nq, 2),
        "num_episodes": len(results),
        "num_successful": len(successful),
        "num_failed": len(failed),
        "avg_path_length": round(
            sum(r.path_length for r in results) / len(results), 2
        ),
        "avg_detections_per_episode": round(
            sum(r.num_detections for r in results) / len(results), 2
        ),
        "avg_kg_nodes_per_episode": round(
            sum(r.num_kg_nodes for r in results) / len(results), 2
        ),
        "avg_inference_time_per_episode": round(
            sum(r.total_inference_time for r in results) / len(results), 2
        ),
    }


def compute_metrics_by_split(
    results: Sequence[EpisodeResult],
) -> dict[str, dict]:
    """Compute metrics grouped by split."""
    by_split: dict[str, list[EpisodeResult]] = {}
    for r in results:
        by_split.setdefault(r.split, []).append(r)

    output = {}
    for split, split_results in sorted(by_split.items()):
        output[split] = compute_all_metrics(split_results)

    output["overall"] = compute_all_metrics(list(results))
    return output


def compute_metrics_by_category(
    results: Sequence[EpisodeResult],
) -> dict[str, dict]:
    """Compute metrics grouped by target category."""
    by_cat: dict[str, list[EpisodeResult]] = {}
    for r in results:
        by_cat.setdefault(r.target_category, []).append(r)

    output = {}
    for cat, cat_results in sorted(by_cat.items()):
        output[cat] = compute_all_metrics(cat_results)

    return output


# =========================================================================
# Detection-level metrics (reused from evaluate_coco.py)
# =========================================================================

def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def geodesic_distance(pos_a: list[float], pos_b: list[float]) -> float:
    """
    Euclidean distance as fallback when geodesic is unavailable.
    In Habitat, use habitat_sim.geo.geodesic_distance() instead.
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_a, pos_b)))
