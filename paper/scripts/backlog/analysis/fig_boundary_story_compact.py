"""Build a compact paper-facing boundary figure.

Archived story figure; not part of the current arXiv figure bundle.

The figure combines the two boundary claims that now matter in the main paper:
  1. On the high-p real-data classification cohort, CIF is strongest at
     intermediate budgets rather than at the full endpoint.
  2. Sparse-forest diagnostics make candidate-set exposure the clearest
     mechanism behind that pattern.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ARXIV_FIGURES_DIR = Path(__file__).resolve().parents[2] / "arxiv" / "figures"

CATEGORY_ORDER = ["under_100", "k100", "intermediate", "endpoint"]
CATEGORY_DISPLAY = {
    "under_100": "$k<100$",
    "k100": "$k=100$",
    "intermediate": "$100<k<p$",
    "endpoint": "$k=p$",
}
CATEGORY_COLORS = {
    "under_100": "#E2E8F0",
    "k100": "#94A3B8",
    "intermediate": "#2563EB",
    "endpoint": "#B91C1C",
}

RIGHT_METHODS = ["cif", "cif_all", "rf", "et"]
RIGHT_DISPLAY = {
    "cif": "CIF",
    "cif_all": "CIF (all features)",
    "rf": "RF",
    "et": "ExtraTrees",
}
RIGHT_COLORS = {
    "cif": "#2563EB",
    "cif_all": "#0F766E",
    "rf": "#B91C1C",
    "et": "#A16207",
}


def _load_left_panel() -> pd.DataFrame:
    df = pd.read_csv(TABLES_DIR / "paper_high_p_cif_best_observed_k_examples.csv")
    df = df[df["task"] == "classification"].copy()
    df["category"] = df["best_k_bucket"].map(
        {
            "under_100": "under_100",
            "k100": "k100",
            "between_100_and_endpoint": "intermediate",
            "endpoint": "endpoint",
        }
    )

    model_order = ["lr", "svm", "knn"]
    dataset_order = (
        df.groupby("dataset")["endpoint_k"].max().sort_values(ascending=False).index.tolist()
    )

    heatmap = (
        df.assign(
            downstream_model=pd.Categorical(df["downstream_model"], categories=model_order, ordered=True),
            dataset=pd.Categorical(df["dataset"], categories=dataset_order, ordered=True),
        )
        .pivot(index="dataset", columns="downstream_model", values="category")
        .loc[dataset_order, model_order]
    )
    return heatmap


def _load_right_panel() -> pd.DataFrame:
    mech = pd.read_csv(TABLES_DIR / "paper_mechanism_grid_forest_classification_summary.csv")
    mech = mech[
        (mech["study"] == "ensemble_split_counts")
        & (mech["task"] == "classification")
        & (mech["n_estimators_per_fit"] == 1000)
        & (mech["n_informative"] == 2)
        & (mech["method"].isin(RIGHT_METHODS))
    ].copy()
    return mech.sort_values(["method", "n_features"]).reset_index(drop=True)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARXIV_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    left = _load_left_panel()
    right = _load_right_panel()

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )

    fig = plt.figure(figsize=(11.4, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1, 1], wspace=0.18, hspace=0.16)
    ax_left = fig.add_subplot(gs[:, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_top)

    category_to_code = {category: idx for idx, category in enumerate(CATEGORY_ORDER)}
    display_labels = {"lr": "LR", "svm": "SVM", "knn": "KNN"}
    heatmap_codes = left.apply(lambda col: col.map(category_to_code)).astype(int).to_numpy()
    cmap = ListedColormap([CATEGORY_COLORS[category] for category in CATEGORY_ORDER])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    ax_left.imshow(heatmap_codes, cmap=cmap, norm=norm, aspect="auto", interpolation="none")
    ax_left.set_xticks(range(len(left.columns)))
    ax_left.set_xticklabels([display_labels[str(col)] for col in left.columns])
    ax_left.set_yticks(range(len(left.index)))
    ax_left.set_yticklabels(left.index)
    ax_left.set_xlabel("Downstream model")
    ax_left.set_title("A. First best CIF budget by high-$p$ classification cell", fontsize=12, loc="left", pad=8)
    ax_left.tick_params(axis="both", length=0)
    ax_left.set_xticks([x - 0.5 for x in range(1, len(left.columns))], minor=True)
    ax_left.set_yticks([y - 0.5 for y in range(1, len(left.index))], minor=True)
    ax_left.grid(False)
    ax_left.grid(which="minor", color="white", linewidth=1.2)

    legend_handles = [
        Patch(facecolor=CATEGORY_COLORS[category], edgecolor="none", label=CATEGORY_DISPLAY[category])
        for category in CATEGORY_ORDER
    ]
    ax_left.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.12),
        frameon=False,
        fontsize=8.4,
        ncol=2,
    )
    for method in RIGHT_METHODS:
        method_rows = right[right["method"] == method].sort_values("n_features")
        color = RIGHT_COLORS[method]
        label = RIGHT_DISPLAY[method]
        ax_top.plot(
            method_rows["n_features"],
            method_rows["informative_split_share"],
            color=color,
            linewidth=2.4 if method == "cif" else 2.0,
            marker="o",
            markersize=5.5,
            label=label,
        )
        ax_bottom.plot(
            method_rows["n_features"],
            method_rows["distinct_false_features_used"],
            color=color,
            linewidth=2.4 if method == "cif" else 2.0,
            marker="o",
            markersize=5.5,
        )

    ax_top.set_title("B. Candidate exposure", fontsize=12, loc="left")
    ax_top.set_ylabel("Informative split share")
    ax_top.set_ylim(0.0, 1.05)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8.3)

    ax_bottom.set_title("C. Noise spread", fontsize=12, loc="left")
    ax_bottom.set_ylabel("Distinct false features")
    ax_bottom.set_xlabel("Number of features ($p$)")
    ax_bottom.set_xticks([100, 500, 1000])

    for out_dir in (FIGURES_DIR, ARXIV_FIGURES_DIR):
        out_path = out_dir / "boundary_story.png"
        fig.savefig(out_path)
        print(f"saved {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
