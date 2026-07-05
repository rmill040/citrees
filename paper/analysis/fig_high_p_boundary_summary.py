"""Build the paper-facing classification high-p boundary summary.

The figure replaces the pooled high-p table with a classification-focused,
downstream-stratified summary:
  - left panel: where CIF first reaches its best observed score
  - right panel: mean score change from k=100 to k=p

Regression remains in prose as the smaller directional mirror.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from paper.analysis.artifact_outputs import add_write_arxiv_argument, figure_output_dirs

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ARXIV_FIGURES_DIR = Path(__file__).resolve().parents[1] / "arxiv" / "figures"

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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_write_arxiv_argument(parser)
    args = parser.parse_args(argv)
    output_dirs = figure_output_dirs(
        FIGURES_DIR,
        ARXIV_FIGURES_DIR,
        write_arxiv=args.write_arxiv,
    )
    _setup_style()

    best = pd.read_csv(TABLES_DIR / "paper_high_p_cif_best_observed_k_summary.csv")
    endpoint = pd.read_csv(TABLES_DIR / "paper_high_p_cif_endpoint_summary.csv")

    best = best[
        (best["task"] == "classification") & (best["downstream_model"].isin(DOWNSTREAMS))
    ].copy()
    endpoint = endpoint[
        (endpoint["task"] == "classification") & (endpoint["downstream_model"].isin(DOWNSTREAMS))
    ].copy()
    best = best.set_index("downstream_model").loc[DOWNSTREAMS].reset_index()
    endpoint = endpoint.set_index("downstream_model").loc[DOWNSTREAMS].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(8.9, 3.8))
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

    axes[0].set_title(r"First $k$ with best CIF score", pad=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DISPLAY_NAMES[d] for d in DOWNSTREAMS])
    axes[0].set_ylabel(r"\% of high-$p$ datasets")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_yticks(np.linspace(0.0, 1.0, 6))
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    score_values = endpoint["mean_score_endpoint_minus_k100"].to_numpy(dtype=float)
    bar_colors = ["#2F855A" if val > 0 else "#B91C1C" for val in score_values]
    axes[1].bar(x, score_values, width=0.62, color=bar_colors, edgecolor="white", linewidth=0.8)
    axes[1].axhline(0.0, color="#9CA3AF", linestyle="--", linewidth=1.0)
    axes[1].set_title(r"Mean score change: $k=p$ minus $k=100$", pad=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DISPLAY_NAMES[d] for d in DOWNSTREAMS])
    axes[1].set_ylabel("Balanced-accuracy change")
    axes[1].grid(True, axis="y")
    axes[1].set_ylim(float(score_values.min()) - 0.018, float(score_values.max()) + 0.014)

    for xpos, val in zip(x, score_values, strict=True):
        if val >= 0:
            axes[1].text(xpos, val + 0.004, f"{val:+.3f}", ha="center", va="bottom", fontsize=10)
        else:
            axes[1].text(
                xpos,
                val - 0.004,
                f"{val:+.3f}",
                ha="center",
                va="top",
                fontsize=10,
                color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.9,
                },
            )

    axes[0].legend(
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.26),
        columnspacing=1.0,
        handlelength=1.4,
    )

    axes[0].text(
        0.5,
        -0.18,
        r"(A)",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )
    axes[1].text(
        0.5,
        -0.18,
        r"(B)",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )

    fig.subplots_adjust(top=0.73, bottom=0.24, wspace=0.28)

    for out_dir in output_dirs:
        out_path = out_dir / "high_p_boundary_summary.png"
        fig.savefig(out_path)
        print(f"saved {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
