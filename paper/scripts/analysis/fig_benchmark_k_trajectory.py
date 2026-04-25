"""Build the paper-facing classification rank-by-k figure.

The figure uses the canonical stratified benchmark table and includes every
classification method in that table. The heatmap avoids the visual clutter of
seventeen overlaid lines while still showing how mean rank changes with the
number of selected features.
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

STANDARD_K = [5, 10, 25, 50, 100]
EXPECTED_N_METHODS = 17

DISPLAY_NAMES = {
    "boruta": "Boruta",
    "cat": "CatBoost",
    "cif": "CIF",
    "cit": "CIT",
    "cpi": "CPI",
    "dt": "DT",
    "et": "ExtraTrees",
    "lgbm": "LightGBM",
    "pi": "PI",
    "ptest_mc": "MC filter",
    "ptest_rdc": "RDC filter",
    "r_cforest": "cforest",
    "r_ctree": "ctree",
    "rf": "RF",
    "rfe": "RFE",
    "rt": "RT",
    "xgb": "XGBoost",
}


def _load_classification_ranks() -> tuple[pd.DataFrame, dict[int, int]]:
    path = TABLES_DIR / "paper_benchmark_stratified.csv"
    df = pd.read_csv(path)
    df = df[
        (df["task"] == "classification")
        & (df["metric"] == "balanced_accuracy")
        & (df["support_type"] == "all_method_complete_case_standard_k")
        & (df["k"].isin(STANDARD_K))
    ].copy()

    methods = sorted(df["method_base"].unique())
    if len(methods) != EXPECTED_N_METHODS:
        raise ValueError(f"Expected {EXPECTED_N_METHODS} classification methods, found {len(methods)}: {methods}")

    missing_names = sorted(set(methods) - set(DISPLAY_NAMES))
    if missing_names:
        raise ValueError(f"Missing display names for methods: {missing_names}")

    support = df.groupby("k")["n_complete_datasets"].first().to_dict()
    if set(support) != set(STANDARD_K):
        raise ValueError(f"Missing support counts for k values: {sorted(set(STANDARD_K) - set(support))}")

    ranks = (
        df.groupby(["method_base", "k"], as_index=False)["mean_rank"]
        .mean()
        .pivot(index="method_base", columns="k", values="mean_rank")
        .reindex(columns=STANDARD_K)
    )
    ranks["mean_over_k"] = ranks.mean(axis=1)
    ranks = ranks.sort_values("mean_over_k")
    ranks = ranks.drop(columns=["mean_over_k"])
    return ranks, {int(k): int(v) for k, v in support.items()}


def _write_rank_table(ranks: pd.DataFrame) -> None:
    out = ranks.reset_index().rename(columns={"index": "method_base"})
    out.insert(1, "display_name", out["method_base"].map(DISPLAY_NAMES))
    out.columns = [str(column) for column in out.columns]
    out.to_csv(TABLES_DIR / "k_trajectory_ranks.csv", index=False)


def _annotate_heatmap(ax: plt.Axes, values: np.ndarray) -> None:
    midpoint = (np.nanmin(values) + np.nanmax(values)) / 2
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            color = "white" if value > midpoint else "#111827"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7.5,
            )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARXIV_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ranks, support = _load_classification_ranks()
    _write_rank_table(ranks)

    plt.style.use("default")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 8.5,
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )

    values = ranks.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    image = ax.imshow(values, cmap="viridis_r", aspect="auto", vmin=1, vmax=EXPECTED_N_METHODS)
    _annotate_heatmap(ax, values)

    x_labels = [f"{k}\n(n={support[k]})" for k in STANDARD_K]
    y_labels = [DISPLAY_NAMES[method] for method in ranks.index]
    ax.set_xticks(np.arange(len(STANDARD_K)), labels=x_labels)
    ax.set_yticks(np.arange(len(ranks.index)), labels=y_labels)
    ax.set_xlabel("Number of selected features $k$")
    ax.set_ylabel("")
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-0.5, len(STANDARD_K), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ranks.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    for label in ax.get_yticklabels():
        if label.get_text() == "CIF":
            label.set_fontweight("bold")

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean rank")
    cbar.ax.invert_yaxis()

    fig.tight_layout()

    for out_dir in (FIGURES_DIR, ARXIV_FIGURES_DIR):
        out_path = out_dir / "k_trajectory.png"
        fig.savefig(out_path)
        print(f"saved {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
