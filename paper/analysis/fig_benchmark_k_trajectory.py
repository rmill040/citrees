"""Build the paper-facing rank-by-k figure.

The figure uses the canonical stratified benchmark table and includes every
method of each task, with classification and regression as two heatmap panels
that share one colour scale. The heatmaps avoid the visual clutter of overlaid
lines while still showing how mean rank changes with the number of selected
features. The figure is drawn at its printed size (0.95 of a 6.5-inch text
width), so its font sizes are the printed sizes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import offset_copy

from paper.analysis.artifact_outputs import add_write_arxiv_argument, figure_output_dirs

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ARXIV_FIGURES_DIR = Path(__file__).resolve().parents[1] / "arxiv" / "figures"

STANDARD_K: Final[tuple[int, ...]] = (5, 10, 25, 50, 100)
OUTPUT_NAME: Final[str] = "benchmark_k_trajectory.png"
PRINT_WIDTH_IN: Final[float] = 0.95 * 6.5
PRINT_HEIGHT_IN: Final[float] = 3.6
CELL_PT: Final[float] = 7.5
TICK_PT: Final[float] = 8.0
LABEL_PT: Final[float] = 8.5
TITLE_PT: Final[float] = 9.0
TICK_PAD_PT: Final[float] = 3.0
RANK_CMAP: Final[LinearSegmentedColormap] = LinearSegmentedColormap.from_list(
    "rank_green_gray_red",
    ["#2F855A", "#A7D7A4", "#F3F4F6", "#F2B8A2", "#B91C1C"],
)


class TaskPlotConfig(NamedTuple):
    """Configuration for one task-specific k-trajectory heatmap."""

    task: str
    metric: str
    expected_n_methods: int
    table_name: str


TASKS: Final[tuple[TaskPlotConfig, ...]] = (
    TaskPlotConfig(
        task="classification",
        metric="balanced_accuracy",
        expected_n_methods=17,
        table_name="classification_k_trajectory_ranks.csv",
    ),
    TaskPlotConfig(
        task="regression",
        metric="r2",
        expected_n_methods=18,
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
        raise ValueError(
            f"Missing support counts for k values: {sorted(set(STANDARD_K) - set(support))}"
        )

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
            "savefig.bbox": "standard",
        }
    )


VMAX: Final[int] = max(config.expected_n_methods for config in TASKS)


def _draw_panel(
    ax: plt.Axes,
    config: TaskPlotConfig,
    ranks: pd.DataFrame,
    support: dict[int, int],
) -> plt.AxesImage:
    values = ranks.to_numpy(dtype=float)
    image = ax.imshow(values, cmap=RANK_CMAP, aspect="auto", vmin=1, vmax=VMAX)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{values[row_idx, col_idx]:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=CELL_PT,
            )

    x_labels = [f"{k}\n{support[k]}" for k in STANDARD_K]
    # usetex ignores set_fontweight, so CIF is set in bold through LaTeX.
    y_labels = [
        r"\textbf{CIF}" if method == "cif" else DISPLAY_NAMES[method] for method in ranks.index
    ]
    ax.set_xticks(np.arange(len(STANDARD_K)), labels=x_labels, fontsize=TICK_PT)
    ax.set_yticks(np.arange(len(ranks.index)), labels=y_labels, fontsize=TICK_PT)
    ax.tick_params(axis="x", length=0, pad=TICK_PAD_PT)
    ax.tick_params(axis="y", length=0, pad=TICK_PAD_PT)
    ax.set_xticks(np.arange(-0.5, len(STANDARD_K), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ranks.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("Number of selected features $k$", fontsize=LABEL_PT, labelpad=3)
    ax.set_title(config.task.capitalize(), fontsize=TITLE_PT, pad=4)

    # Row header for the two-line tick labels, right-aligned under the method names.
    trans = offset_copy(ax.transAxes, fig=ax.figure, x=-TICK_PAD_PT, y=-TICK_PAD_PT, units="points")
    ax.text(
        0.0,
        0.0,
        "$k$\ndatasets",
        transform=trans,
        ha="right",
        va="top",
        fontsize=TICK_PT,
        multialignment="right",
    )
    return image


def _render_figure(
    panels: list[tuple[TaskPlotConfig, pd.DataFrame, dict[int, int]]],
    output_dirs: tuple[Path, ...],
) -> None:
    fig, axes = plt.subplots(
        1, len(panels), figsize=(PRINT_WIDTH_IN, PRINT_HEIGHT_IN), layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.04, wspace=0.04)
    image = None
    for ax, (config, ranks, support) in zip(axes, panels, strict=True):
        image = _draw_panel(ax, config, ranks, support)

    cbar = fig.colorbar(image, ax=list(axes), fraction=0.03, pad=0.015, aspect=35)
    cbar.set_label("Mean rank", fontsize=LABEL_PT)
    cbar.ax.tick_params(labelsize=TICK_PT)
    cbar.set_ticks([1, 3, 6, 9, 12, 15, 18])
    cbar.ax.invert_yaxis()

    for out_dir in output_dirs:
        out_path = out_dir / OUTPUT_NAME
        fig.savefig(out_path, bbox_inches=None, pad_inches=0)
        print(f"saved {out_path}")
    plt.close(fig)


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
    panels = []
    for config in TASKS:
        ranks, support = _load_ranks(config)
        _write_rank_table(config, ranks)
        panels.append((config, ranks, support))
    _render_figure(panels, output_dirs)


if __name__ == "__main__":
    main()
