"""
compare_configs.py -- Compare results across different experiment configurations.

Loads result JSONs from multiple configs (A, B, C, D) and produces
a comparison table and plots.
"""

from __future__ import annotations
import json
from pathlib import Path


def load_results(result_paths: dict[str, str]) -> dict[str, dict]:
    """Load result JSONs keyed by config name."""
    results = {}
    for name, path in result_paths.items():
        with open(path, "r") as f:
            results[name] = json.load(f)
    return results


def compare_coin_metrics(results: dict[str, dict]) -> str:
    """Generate a comparison table string."""
    header = f"{'Config':<20} {'SR':>8} {'SPL':>8} {'NQ':>8} {'Calls/det':>10}"
    lines = [header, "-" * len(header)]
    for name, data in results.items():
        m = data.get("metrics", data)
        sr = m.get("SR", "N/A")
        spl = m.get("SPL", "N/A")
        nq = m.get("NQ", "N/A")
        calls = m.get("avg_calls_per_detection", "N/A")
        lines.append(f"{name:<20} {sr:>8} {spl:>8} {nq:>8} {calls:>10}")
    return "\n".join(lines)
