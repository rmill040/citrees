"""Build the paper-facing rank-by-k figures.

The figures use the canonical stratified benchmark table and include every
method in each task-specific table. The heatmaps avoid the visual clutter of
overlaid lines while still showing how mean rank changes with the number of
selected features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ARXIV_FIGURES_DIR = Path(__file__).resolve().parents[2] / "arxiv" / "figures"

STANDARD_K: Final[tuple[int, ...]] = (5, 10, 25, 50, 100)
RANK_CMAP: Final[LinearSegmentedColormap] = LinearSegmentedColormap.from_list(
    "rank_green_gray_red",
    ["#2F855A", "#A7D7A4", "#F3F4F6", "#F2B8A2", "#B91C1C"],
)


class TaskPlotConfig(NamedTuple):
    """Configuration for one task-specific k-trajectory heatmap."""

    task: str
    metric: str
    expected_n_methods: int
    output_name: str
    table_name: str


TASKS: Final[tuple[TaskPlotConfig, ...]] = (
    TaskPlotConfig(
        task="classification",
        metric="balanced_accuracy",
        expected_n_methods=17,
        output_name="k_trajectory.png",
        table_name="k_trajectory_ranks.csv",
    ),
    TaskPlotConfig(
        task="regression",
        metric="r2",
        expected_n_methods=18,
        output_name="regression_k_trajectory.png",
        table_name="regression_k_trajectory_ranks.csv",
    ),
)

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
    "ptest_dc": "DC filter",
    "ptest_mc": "MC filter",
    "ptest_pc": "PC filter",
    "ptest_rdc": "RDC filter",
    "r_cforest": "cforest",
    "r_ctree": "ctree",
    "rf": "RF",
    "rfe": "RF-RFE",
    "rt": "RT",
    "xgb": "XGBoost",
}


def _load_ranks(config: TaskPlotConfig) -> tuple[pd.DataFrame, dict[int, int]]:
    path = TABLES_DIR / "paper_benchmark_stratified.csv"
    df = pd.read_csv(path)
    df = df[
        (df["task"] == config.task)
        & (df["metric"] == config.metric)
        & (df["support_type"] == "all_method_complete_case_standard_k")
        & (df["k"].isin(STANDARD_K))
    ].copy()

    methods = sorted(df["method_base"].unique())
    if len(methods) != config.expected_n_methods:
        raise ValueError(
            f"Expected {config.expected_n_methods} {config.task} methods, found {len(methods)}: {methods}"
        )

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


def _write_rank_table(config: TaskPlotConfig, ranks: pd.DataFrame) -> None:
    out = ranks.reset_index().rename(columns={"index": "method_base"})
    out.insert(1, "display_name", out["method_base"].map(DISPLAY_NAMES))
    out.columns = [str(column) for column in out.columns]
    out.to_csv(TABLES_DIR / config.table_name, index=False)


def _annotate_heatmap(ax: plt.Axes, values: np.ndarray) -> None:
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=7.5,
            )


def _setup_style() -> None:
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


def _render_heatmap(config: TaskPlotConfig, ranks: pd.DataFrame, support: dict[int, int]) -> None:
    values = ranks.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    image = ax.imshow(values, cmap=RANK_CMAP, aspect="auto", vmin=1, vmax=config.expected_n_methods)
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
        out_path = out_dir / config.output_name
        fig.savefig(out_path)
        print(f"saved {out_path}")

    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARXIV_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    _setup_style()
    for config in TASKS:
        ranks, support = _load_ranks(config)
        _write_rank_table(config, ranks)
        _render_heatmap(config, ranks, support)


if __name__ == "__main__":
    main()
