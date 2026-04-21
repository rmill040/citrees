"""Build the paper-facing classification high-p boundary summary.

The figure replaces the pooled high-p table with a classification-focused,
downstream-stratified summary:
  - left panel: where CIF first reaches its best observed score
  - right panel: mean score change from k=100 to k=p

Regression remains in prose as the smaller directional mirror.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ARXIV_FIGURES_DIR = Path(__file__).resolve().parents[2] / "arxiv" / "figures"

DOWNSTREAMS = ["lr", "svm", "knn"]
DISPLAY_NAMES = {"lr": "LR", "svm": "SVM", "knn": "KNN"}
CATEGORY_ORDER = ["under_100_share", "k100_share", "intermediate_share", "endpoint_share"]
CATEGORY_LABELS = {
    "under_100_share": r"$k < 100$",
    "k100_share": r"$k = 100$",
    "intermediate_share": r"$100 < k < p$",
    "endpoint_share": r"$k = p$",
}
CATEGORY_COLORS = {
    "under_100_share": "#E5E7EB",
    "k100_share": "#94A3B8",
    "intermediate_share": "#2563EB",
    "endpoint_share": "#B91C1C",
}


def _setup_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARXIV_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _setup_style()

    best = pd.read_csv(TABLES_DIR / "paper_high_p_cif_best_observed_k_summary.csv")
    endpoint = pd.read_csv(TABLES_DIR / "paper_high_p_cif_endpoint_summary.csv")

    best = best[(best["task"] == "classification") & (best["downstream_model"].isin(DOWNSTREAMS))].copy()
    endpoint = endpoint[(endpoint["task"] == "classification") & (endpoint["downstream_model"].isin(DOWNSTREAMS))].copy()
    best = best.set_index("downstream_model").loc[DOWNSTREAMS].reset_index()
    endpoint = endpoint.set_index("downstream_model").loc[DOWNSTREAMS].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.7))
    x = np.arange(len(DOWNSTREAMS))

    bottoms = np.zeros(len(DOWNSTREAMS))
    for category in CATEGORY_ORDER:
        values = best[category].to_numpy(dtype=float)
        axes[0].bar(
            x,
            values,
            bottom=bottoms,
            color=CATEGORY_COLORS[category],
            edgecolor="white",
            linewidth=0.8,
            label=CATEGORY_LABELS[category],
        )
        bottoms += values

    axes[0].set_title("A. First best CIF budget", pad=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DISPLAY_NAMES[d] for d in DOWNSTREAMS])
    axes[0].set_ylabel("Share of high-$p$ cells")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_yticks(np.linspace(0.0, 1.0, 6))
    score_values = endpoint["mean_score_endpoint_minus_k100"].to_numpy(dtype=float)
    bar_colors = ["#15803D" if val > 0 else "#B45309" for val in score_values]
    axes[1].bar(x, score_values, width=0.62, color=bar_colors, edgecolor="white", linewidth=0.8)
    axes[1].axhline(0.0, color="#9CA3AF", linestyle="--", linewidth=1.0)
    axes[1].set_title(r"B. Mean score change: $k=p$ minus $k=100$", pad=2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DISPLAY_NAMES[d] for d in DOWNSTREAMS])
    axes[1].set_ylabel("Balanced-accuracy change")
    axes[1].grid(True, axis="y")
    axes[1].set_ylim(float(score_values.min()) - 0.006, float(score_values.max()) + 0.012)

    for xpos, val in zip(x, score_values, strict=True):
        offset = 0.004 if val >= 0 else -0.004
        va = "bottom" if val >= 0 else "top"
        axes[1].text(xpos, val + offset, f"{val:+.3f}", ha="center", va=va, fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.12, 1.02),
        columnspacing=1.2,
    )

    fig.subplots_adjust(top=0.76, wspace=0.28)

    for out_dir in (FIGURES_DIR, ARXIV_FIGURES_DIR):
        out_path = out_dir / "high_p_boundary_summary.png"
        fig.savefig(out_path)
        print(f"saved {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
