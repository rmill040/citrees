"""Build a paper-facing appendix figure for forest mechanism scaling.

This figure broadens the main-text sparse-case snapshot by showing how the
classification forest mechanism changes with feature dimension at 1000 trees.
For each method and feature dimension, the point marks the mean over
`n_informative in {1,2,5,10}` and the vertical segment shows the min-max range.
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

METHODS = ["cif", "cif_all", "rf", "et"]
DISPLAY_NAMES = {
    "cif": "CIF",
    "cif_all": "CIF (all features)",
    "rf": "RF",
    "et": "ExtraTrees",
}
COLORS = {
    "cif": "#2563EB",
    "cif_all": "#0F766E",
    "rf": "#B91C1C",
    "et": "#A16207",
}
OFFSETS = {"cif": -0.18, "cif_all": -0.06, "rf": 0.06, "et": 0.18}
FEATURE_LEVELS = [100, 500, 1000]


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


def _load_summary() -> pd.DataFrame:
    df = pd.read_csv(TABLES_DIR / "paper_mechanism_grid_forest_classification_summary.csv")
    df = df[
        (df["study"] == "ensemble_split_counts")
        & (df["task"] == "classification")
        & (df["n_estimators_per_fit"] == 1000)
        & (df["method"].isin(METHODS))
        & (df["n_features"].isin(FEATURE_LEVELS))
    ].copy()
    if df.empty:
        raise RuntimeError("No rows found for classification mechanism dimension curves.")
    return df


def _aggregate(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        df.groupby(["method", "n_features"], as_index=False)
        .agg(
            mean=(value_col, "mean"),
            min=(value_col, "min"),
            max=(value_col, "max"),
        )
    )


def _plot_panel(ax: plt.Axes, agg: pd.DataFrame, ylabel: str, title: str) -> None:
    x_base = np.arange(len(FEATURE_LEVELS))

    for method in METHODS:
        method_rows = agg[agg["method"] == method].set_index("n_features").loc[FEATURE_LEVELS].reset_index()
        x = x_base + OFFSETS[method]
        color = COLORS[method]
        ax.vlines(x, method_rows["min"], method_rows["max"], color=color, linewidth=2.2, alpha=0.95)
        ax.plot(
            x,
            method_rows["mean"],
            color=color,
            linewidth=2.2 if method == "cif" else 2.0,
            marker="o",
            markersize=6.5 if method == "cif" else 5.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=DISPLAY_NAMES[method],
        )

    ax.set_title(title, pad=6)
    ax.set_xticks(x_base)
    ax.set_xticklabels([str(p) for p in FEATURE_LEVELS])
    ax.set_xlabel(r"Number of features ($p$)")
    ax.set_ylabel(ylabel)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARXIV_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _setup_style()

    df = _load_summary()
    split_agg = _aggregate(df, "informative_split_share")
    false_agg = _aggregate(df, "distinct_false_features_used")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.1))
    _plot_panel(
        axes[0],
        split_agg,
        ylabel="Informative split share",
        title="A. Informative split share across sparse designs",
    )
    axes[0].set_ylim(0.0, 1.05)

    _plot_panel(
        axes[1],
        false_agg,
        ylabel="Distinct false features used",
        title="B. Noise spread across sparse designs",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.subplots_adjust(top=0.78, wspace=0.26)

    for out_dir in (FIGURES_DIR, ARXIV_FIGURES_DIR):
        out_path = out_dir / "paper_mechanism_grid_forest_classification_dimension_curves_1000trees.png"
        fig.savefig(out_path)
        print(f"saved {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
