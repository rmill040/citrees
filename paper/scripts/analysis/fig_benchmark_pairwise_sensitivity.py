"""Build the paper-facing classification downstream x k sensitivity heatmap.

The figure exposes the classification benchmark surface that the headline table
partly pools away. It is intentionally narrow:
  - focus method: CIF
  - baselines: the historical conditional-inference references and CIT
  - cells: downstream model x standard feature budget

Each cell shows the mean CIF-minus-baseline balanced-accuracy delta on the
classification benchmark before within-dataset pooling over downstream models
and budgets.
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

BASELINES = ["r_ctree", "r_cforest", "cit"]
BASELINE_TITLES = {
    "r_ctree": r"CIF vs \texttt{r\_ctree}",
    "r_cforest": r"CIF vs \texttt{r\_cforest}",
    "cit": "CIF vs CIT",
}
DOWNSTREAMS = ["lr", "svm", "knn"]
DOWNSTREAM_TITLES = {"lr": "LR", "svm": "SVM", "knn": "KNN"}
K_VALUES = [5, 10, 25, 50, 100]


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


def _load_table() -> pd.DataFrame:
    df = pd.read_csv(TABLES_DIR / "paper_benchmark_pairwise_stratified.csv")
    df = df[
        (df["task"] == "classification")
        & (df["focus_method"] == "cif")
        & (df["baseline"].isin(BASELINES))
        & (df["downstream_model"].isin(DOWNSTREAMS))
        & (df["k"].isin(K_VALUES))
    ].copy()
    if df.empty:
        raise RuntimeError("No rows available for the classification pairwise sensitivity heatmap.")
    return df


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARXIV_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _setup_style()

    df = _load_table()
    vmax = float(df["mean_delta"].max())

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), sharey=True)

    for ax, baseline in zip(axes, BASELINES, strict=True):
        grid = (
            df[df["baseline"] == baseline]
            .pivot(index="downstream_model", columns="k", values="mean_delta")
            .reindex(index=DOWNSTREAMS, columns=K_VALUES)
        )
        image = ax.imshow(grid.to_numpy(), cmap="Blues", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_title(BASELINE_TITLES[baseline], pad=8)
        ax.set_xticks(np.arange(len(K_VALUES)))
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_xlabel(r"Feature budget $k$")
        ax.set_yticks(np.arange(len(DOWNSTREAMS)))
        ax.set_yticklabels([DOWNSTREAM_TITLES[d] for d in DOWNSTREAMS])

        for i, downstream in enumerate(DOWNSTREAMS):
            for j, k_value in enumerate(K_VALUES):
                val = float(grid.loc[downstream, k_value])
                x = j - 0.08 if j == len(K_VALUES) - 1 else j
                ax.text(
                    x,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if val > 0.06 else "#111827",
                    fontsize=8.0,
                )

    axes[0].set_ylabel("Downstream model")
    cbar = fig.colorbar(image, ax=axes, fraction=0.024, pad=0.045)
    cbar.set_label("Mean balanced-accuracy delta")
    fig.subplots_adjust(wspace=0.18)

    for out_dir in (FIGURES_DIR, ARXIV_FIGURES_DIR):
        out_path = out_dir / "benchmark_pairwise_sensitivity.png"
        fig.savefig(out_path)
        print(f"saved {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
