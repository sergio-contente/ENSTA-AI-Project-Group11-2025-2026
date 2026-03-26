"""
Generate publication-quality comparison tables from IDKVQA fair_table.json.

Usage:
    conda activate aiuta
    cd ~/ai_project/aiuta_vlmr1_project
    python scripts/plot_fair_table.py

Outputs:
    results/figures/fair_comparison_table.png
    results/figures/threshold_sweep_table.png
    results/figures/full_ablation_table.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


RESULTS_DIR = Path("results/idkvqa")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# CoIN paper reference (Table 5)
AIUTA_PAPER = {
    "Normalized Entropy (AIUTA, LLaVA 7B)": {"phi_binary": 21.12, "note": "CoIN paper"},
    "Energy Score (LLaVA 7B)": {"phi_binary": 20.45, "note": "CoIN paper"},
    "MaxProb (LLaVA 7B)": {"phi_binary": 15.94, "note": "CoIN paper"},
}


def load_fair_table() -> list[dict]:
    p = RESULTS_DIR / "fair_table.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run: python -m aiuta_vlmr1.evaluation.fair_comparison --results-dir results/idkvqa/ --output results/idkvqa/fair_table.json")
    with open(p) as f:
        return json.load(f)


def color_cell(val: float, vmin: float, vmax: float, cmap_name: str = "RdYlGn") -> str:
    """Return hex color for a cell value."""
    cmap = plt.get_cmap(cmap_name)
    norm = (val - vmin) / (vmax - vmin + 1e-9)
    norm = max(0.0, min(1.0, norm))
    rgba = cmap(norm)
    return mcolors.to_hex(rgba)


def render_table(
    cell_text: list[list[str]],
    col_labels: list[str],
    row_colors: list[list[str]] | None = None,
    title: str = "",
    output_path: str | Path = "table.png",
    highlight_rows: list[int] | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Render a matplotlib table to PNG."""
    n_rows = len(cell_text)
    n_cols = len(col_labels)

    if figsize is None:
        figsize = (max(n_cols * 1.6, 10), max(n_rows * 0.45 + 1.2, 3))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        cell.set_edgecolor("#2c3e50")

    # Style data rows
    for i in range(n_rows):
        for j in range(n_cols):
            cell = table[i + 1, j]
            cell.set_edgecolor("#ddd")

            # Alternating row colors
            if i % 2 == 0:
                cell.set_facecolor("#f8f9fa")
            else:
                cell.set_facecolor("white")

            # Apply heatmap colors if provided
            if row_colors and row_colors[i][j]:
                cell.set_facecolor(row_colors[i][j])

            # Bold highlight rows
            if highlight_rows and i in highlight_rows:
                cell.set_text_props(fontweight="bold")
                if not (row_colors and row_colors[i][j]):
                    cell.set_facecolor("#fff3cd")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=20, loc="left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {output_path}")


def plot_main_comparison():
    """Table 1: Main comparison (our modes + AIUTA paper baselines)."""
    rows_data = load_fair_table()

    # Select key modes (no tau sweep variants)
    key_modes = [
        "threshold_tau=0.10",
        "threshold_tau=0.15",
        "kg_threshold",
        "kg",
        "two_pass_kg",
        "two_pass_kg_entropy",
        "raw",
        "two_pass_kg_relaxed",
        "raw_two_pass",
    ]

    mode_map = {r["mode"]: r for r in rows_data}
    selected = [mode_map[m] for m in key_modes if m in mode_map]

    # Build table
    col_labels = ["Mode", "Calls", "Acc %", "phi_1 binary", "phi_1 CoIN", "Abst %", "Overclaim %", "Note"]

    cell_text = []
    phi_bin_vals = []
    phi_coin_vals = []

    for r in selected:
        coin = f"{r['phi_coin_c1_pct']:.1f}" if r['phi_coin_c1_pct'] is not None else "--"
        cell_text.append([
            r["mode"].replace("threshold_tau=", "threshold tau=").replace("_", " "),
            str(int(r["calls"])),
            f"{r['accuracy_pct']:.1f}",
            f"{r['phi_binary_c1_pct']:.2f}",
            coin,
            f"{r['abstention_rate_pct']:.1f}",
            f"{r['overclaim_rate_pct']:.1f}",
            r["note"],
        ])
        phi_bin_vals.append(r["phi_binary_c1_pct"])
        phi_coin_vals.append(r["phi_coin_c1_pct"] if r["phi_coin_c1_pct"] is not None else 0)

    # Add AIUTA paper baselines
    for name, d in AIUTA_PAPER.items():
        cell_text.append([
            name,
            "5-8",
            "--",
            f"{d['phi_binary']:.2f}",
            "--",
            "--",
            "--",
            d["note"],
        ])
        phi_bin_vals.append(d["phi_binary"])
        phi_coin_vals.append(0)

    # Heatmap colors for phi columns
    bin_min, bin_max = min(phi_bin_vals), max(phi_bin_vals)
    coin_min, coin_max = min(phi_coin_vals), max(phi_coin_vals)

    row_colors = []
    for i, r in enumerate(cell_text):
        colors = [""] * len(col_labels)
        colors[3] = color_cell(phi_bin_vals[i], bin_min, bin_max)
        if phi_coin_vals[i] > 0:
            colors[4] = color_cell(phi_coin_vals[i], coin_min, coin_max)
        row_colors.append(colors)

    # Highlight best binary and best coin
    best_bin_idx = int(np.argmax(phi_bin_vals))
    best_coin_idx = int(np.argmax(phi_coin_vals))
    highlight = list(set([best_bin_idx, best_coin_idx]))

    render_table(
        cell_text,
        col_labels,
        row_colors=row_colors,
        title="IDKVQA Fair Comparison -- VLM-R1 3B (ours) vs AIUTA (LLaVA 7B + GPT-4o)",
        output_path=FIGURES_DIR / "fair_comparison_table.png",
        highlight_rows=highlight,
    )


def plot_threshold_sweep():
    """Table 2: Threshold sweep (tau = 0.05 to 0.35)."""
    rows_data = load_fair_table()
    sweep = [r for r in rows_data if r["mode"].startswith("threshold_tau=")]
    sweep.sort(key=lambda r: float(r["mode"].split("=")[1]))

    col_labels = ["tau", "Acc %", "phi_1 binary", "phi_1 CoIN", "Abst %", "Coverage %", "Overclaim %"]

    cell_text = []
    phi_bin_vals = []

    for r in sweep:
        tau = r["mode"].split("=")[1]
        cov = 100.0 - r["abstention_rate_pct"]
        coin = f"{r['phi_coin_c1_pct']:.1f}" if r['phi_coin_c1_pct'] is not None else "--"
        cell_text.append([
            tau,
            f"{r['accuracy_pct']:.1f}",
            f"{r['phi_binary_c1_pct']:.2f}",
            coin,
            f"{r['abstention_rate_pct']:.1f}",
            f"{cov:.1f}",
            f"{r['overclaim_rate_pct']:.1f}",
        ])
        phi_bin_vals.append(r["phi_binary_c1_pct"])

    bin_min, bin_max = min(phi_bin_vals), max(phi_bin_vals)
    row_colors = []
    for i in range(len(cell_text)):
        colors = [""] * len(col_labels)
        colors[2] = color_cell(phi_bin_vals[i], bin_min, bin_max)
        row_colors.append(colors)

    best_idx = int(np.argmax(phi_bin_vals))

    render_table(
        cell_text,
        col_labels,
        row_colors=row_colors,
        title="Entropy Threshold Sweep -- VLM-R1 3B on IDKVQA (502 samples)",
        output_path=FIGURES_DIR / "threshold_sweep_table.png",
        highlight_rows=[best_idx],
        figsize=(9, 5),
    )


def plot_cost_table():
    """Table 3: Cost/latency comparison."""
    rows_data = load_fair_table()

    key_modes = [
        "raw",
        "threshold_tau=0.10",
        "kg",
        "two_pass_kg",
        "two_pass_kg_entropy",
        "raw_two_pass",
    ]
    mode_map = {r["mode"]: r for r in rows_data}

    # Load per-mode latency from individual result JSONs
    latency = {}
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name == "fair_table.json":
            continue
        try:
            data = json.loads(f.read_text())
        except:
            continue
        if not isinstance(data, dict):
            continue
        mode = data.get("mode", "")
        metrics = data.get("metrics", {})
        ca = metrics.get("cost_accounting", {})
        if mode and ca:
            latency[mode] = {
                "avg_latency": ca.get("avg_total_latency_sec", 0),
                "avg_detector": ca.get("avg_detector_latency_sec", 0),
            }

    col_labels = ["Mode", "VLM Calls", "phi_1 binary", "Avg Latency (s)", "Det. Latency (s)", "API Cost"]

    cell_text = []
    for m in key_modes:
        if m not in mode_map:
            continue
        r = mode_map[m]
        lat = latency.get(m.replace("threshold_tau=0.10", "threshold"), {})
        display_mode = m.replace("threshold_tau=", "threshold tau=").replace("_", " ")
        cell_text.append([
            display_mode,
            str(int(r["calls"])),
            f"{r['phi_binary_c1_pct']:.2f}",
            f"{lat.get('avg_latency', 0):.2f}" if lat else "--",
            f"{lat.get('avg_detector', 0):.2f}" if lat else "--",
            "$0",
        ])

    # Add AIUTA paper reference
    cell_text.append([
        "AIUTA (LLaVA + GPT-4o)",
        "5-8",
        "21.12",
        "--",
        "--",
        "$$$ (API)",
    ])

    render_table(
        cell_text,
        col_labels,
        title="Cost & Latency -- VLM-R1 3B (local) vs AIUTA (API-dependent)",
        output_path=FIGURES_DIR / "cost_table.png",
        highlight_rows=[len(cell_text) - 1],
        figsize=(11, 4.5),
    )


if __name__ == "__main__":
    print("Generating tables from results/idkvqa/fair_table.json ...\n")
    plot_main_comparison()
    plot_threshold_sweep()
    plot_cost_table()
    print(f"\nAll figures in {FIGURES_DIR}/")
