"""Build the paper-facing forest feature-count figure for the sparse mechanism design.

This figure preserves the original feature-index count display used in the
paper, but cleans up the styling for main-text readability.

Outputs:
  - paper/results/figures/paper_mechanism_grid_forest_classification_feature_counts_p1000_i2_1000trees.png
  - paper/arxiv/figures/paper_mechanism_grid_forest_classification_feature_counts_p1000_i2_1000trees.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ARXIV_FIGURES_DIR = Path(__file__).resolve().parents[2] / "arxiv" / "figures"

DATASET = "make_classification_n250_p1000_i2"
METHOD_ORDER = ("cif", "cif_all", "rf", "et")
DISPLAY_NAMES = {
    "cif": "CIF",
    "cif_all": "CIF (all features)",
    "rf": "RF",
    "et": "ExtraTrees",
}
OUTPUT_NAME = "paper_mechanism_grid_forest_classification_feature_counts_p1000_i2_1000trees.png"


def _load_counts() -> pd.DataFrame:
    counts = pd.read_csv(TABLES_DIR / "paper_mechanism_grid_forest_classification_feature_counts.csv")
    counts = counts[
        (counts["study"] == "ensemble_split_counts")
        & (counts["dataset"] == DATASET)
        & (counts["n_estimators_per_fit"] == 1000)
        & (counts["method"].isin(METHOD_ORDER))
    ].copy()
    if counts.empty:
        raise RuntimeError("No mechanism feature-count rows matched the paper figure filter.")
    return counts.sort_values(["method", "feature_idx"]).reset_index(drop=True)


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
            "legend.fontsize": 10,
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
    _setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARXIV_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    counts = _load_counts()
    noise_color = "#64748B"
    signal_color = "#1F2937"

    fig, axes_grid = plt.subplots(2, 2, figsize=(8.8, 6.3), sharex=True)
    axes = axes_grid.ravel()

    for ax, method in zip(axes, METHOD_ORDER, strict=True):
        sub = counts[counts["method"] == method].sort_values("feature_idx")
        noise = sub[sub["is_informative"] == 0]
        signal = sub[sub["is_informative"] == 1]

        ax.vlines(
            noise["feature_idx"],
            0,
            noise["tree_use_count"],
            color=noise_color,
            linewidth=0.65,
            alpha=0.78,
            zorder=1,
        )
        ax.vlines(
            signal["feature_idx"],
            0,
            signal["tree_use_count"],
            color=signal_color,
            linewidth=1.0,
            alpha=1.0,
            zorder=3,
        )
        ax.scatter(
            signal["feature_idx"],
            signal["tree_use_count"],
            color=signal_color,
            edgecolor="white",
            linewidth=0.5,
            s=18,
            zorder=4,
        )

        ax.set_title(DISPLAY_NAMES[method], pad=6)
        ax.set_xlim(-5, 1005)
        ax.set_xticks([0, 250, 500, 750, 1000])
        ax.grid(axis="y")
        ax.grid(axis="x", alpha=0.15)

    for ax in axes_grid[-1, :]:
        ax.set_xlabel("Feature index")
    for ax in axes_grid[:, 0]:
        ax.set_ylabel("Split-use count")

    fig.legend(
        handles=[
            Patch(facecolor=signal_color, edgecolor="none", label="informative feature"),
            Patch(facecolor=noise_color, edgecolor="none", label="other feature"),
        ],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.97),
        frameon=False,
        columnspacing=1.6,
        handlelength=1.5,
    )
    fig.subplots_adjust(top=0.88, wspace=0.24, hspace=0.38, bottom=0.10)

    for out_dir in (FIGURES_DIR, ARXIV_FIGURES_DIR):
        out_path = out_dir / OUTPUT_NAME
        fig.savefig(out_path)
        print(f"Saved {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
