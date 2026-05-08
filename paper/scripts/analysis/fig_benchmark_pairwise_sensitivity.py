"""Build paper-facing downstream x k pairwise sensitivity heatmaps.

The figures show pairwise comparisons before the headline table averages over
downstream models and feature budgets. They are intentionally narrow:
  - focus method: CIF
  - compared methods: the historical conditional-inference references and CIT
  - cells: downstream model x standard feature budget

Each cell shows the mean metric difference between CIF and the compared method
before within-dataset pooling over downstream models and budgets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ARXIV_FIGURES_DIR = Path(__file__).resolve().parents[2] / "arxiv" / "figures"

BASELINES = ["r_ctree", "r_cforest", "cit"]
BASELINE_TITLES = {
    "r_ctree": "CIF vs ctree",
    "r_cforest": "CIF vs cforest",
    "cit": "CIF vs CIT",
}
DOWNSTREAMS = {
    "classification": ["lr", "svm", "knn"],
    "regression": ["ridge", "svr", "knn"],
}
DOWNSTREAM_TITLES = {
    "lr": "LR",
    "svm": "SVM",
    "knn": "KNN",
    "ridge": "Ridge",
    "svr": "SVR",
}
K_VALUES = [5, 10, 25, 50, 100]
SIGNED_GAIN_CMAP = LinearSegmentedColormap.from_list(
    "signed_gain_red_neutral_green",
    ["#B91C1C", "#F3F4F6", "#2F855A"],
)
TASK_CONFIG = {
    "classification": {
        "output_name": "benchmark_pairwise_sensitivity.png",
        "metric_label": "Mean balanced-accuracy delta",
        "color_abs_limit": None,
    },
    "regression": {
        "output_name": "regression_benchmark_pairwise_sensitivity.png",
        "metric_label": r"Mean $R^2$ delta",
        "color_abs_limit": 1.0,
    },
}


def _setup_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def _load_table(task: str) -> pd.DataFrame:
    df = pd.read_csv(TABLES_DIR / "paper_benchmark_pairwise_stratified.csv")
    df = df[
        (df["task"] == task)
        & (df["focus_method"] == "cif")
        & (df["baseline"].isin(BASELINES))
        & (df["downstream_model"].isin(DOWNSTREAMS[task]))
        & (df["k"].isin(K_VALUES))
    ].copy()
    if df.empty:
        raise RuntimeError(f"No rows available for the {task} pairwise sensitivity heatmap.")
    return df


def _render_task(task: str) -> None:
    df = _load_table(task)
    max_abs = float(df["mean_delta"].abs().max())
    color_abs_limit = TASK_CONFIG[task]["color_abs_limit"]
    color_abs = max_abs if color_abs_limit is None else min(max_abs, float(color_abs_limit))
    norm = TwoSlopeNorm(vmin=-color_abs, vcenter=0.0, vmax=color_abs)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.2), sharey=True, layout="constrained")

    for ax, baseline in zip(axes, BASELINES, strict=True):
        grid = (
            df[df["baseline"] == baseline]
            .pivot(index="downstream_model", columns="k", values="mean_delta")
            .reindex(index=DOWNSTREAMS[task], columns=K_VALUES)
        )
        display_values = np.clip(grid.to_numpy(dtype=float), -color_abs, color_abs)
        image = ax.imshow(
            display_values,
            cmap=SIGNED_GAIN_CMAP,
            norm=norm,
            aspect="auto",
        )
        ax.set_title(BASELINE_TITLES[baseline], pad=8)
        ax.set_xticks(np.arange(len(K_VALUES)))
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_xlabel(r"Feature budget $k$")
        ax.set_yticks(np.arange(len(DOWNSTREAMS[task])))
        ax.set_yticklabels([DOWNSTREAM_TITLES[d] for d in DOWNSTREAMS[task]])
        ax.set_xticks(np.arange(-0.5, len(K_VALUES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(DOWNSTREAMS[task]), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i, downstream in enumerate(DOWNSTREAMS[task]):
            for j, k_value in enumerate(K_VALUES):
                val = float(grid.loc[downstream, k_value])
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8.0,
                )

    axes[0].set_ylabel("Downstream model")
    extend = "both" if max_abs > color_abs else "neither"
    cbar = fig.colorbar(image, ax=axes, fraction=0.028, pad=0.025, shrink=0.88, extend=extend)
    cbar.set_label(TASK_CONFIG[task]["metric_label"])

    for out_dir in (FIGURES_DIR, ARXIV_FIGURES_DIR):
        out_path = out_dir / TASK_CONFIG[task]["output_name"]
        fig.savefig(out_path)
        print(f"saved {out_path}")

    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARXIV_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _setup_style()
    for task in ("classification", "regression"):
        _render_task(task)


if __name__ == "__main__":
    main()
