"""Build a cleaner paper-facing classification k-trajectory figure.

The figure answers one question: where does CIF sit on the standard-budget
classification benchmark as the feature budget increases?

Design choices:
  - One panel instead of three downstream-specific panels.
  - Only the methods needed to tell the benchmark story are plotted:
    CIF, the strongest generic ensembles, and the conditional-inference
    references.
  - Support counts are moved into the subtitle and caption rather than printed
    above every point.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
ARXIV_FIGURES_DIR = Path(__file__).resolve().parents[2] / "arxiv" / "figures"

STANDARD_K = [5, 10, 25, 50, 100]
ALL_CLF_METHODS = [
    "lgbm",
    "xgb",
    "cat",
    "cif",
    "rf",
    "rfe",
    "et",
    "cit",
    "boruta",
    "pi",
    "cpi",
    "ptest_mc",
    "ptest_rdc",
    "r_ctree",
    "r_cforest",
]
PLOT_METHODS = ["lgbm", "xgb", "cat", "cif", "cit", "r_cforest", "r_ctree"]

DISPLAY_NAMES = {
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "cat": "CatBoost",
    "cif": "CIF",
    "cit": "CIT",
    "r_cforest": "R cforest",
    "r_ctree": "R ctree",
}

METHOD_STYLE = {
    "cif": {"color": "#2563EB", "linewidth": 2.8, "linestyle": "-", "marker": "o", "zorder": 5},
    "lgbm": {"color": "#166534", "linewidth": 2.0, "linestyle": "-", "marker": "s", "zorder": 4},
    "xgb": {"color": "#B91C1C", "linewidth": 2.0, "linestyle": "-", "marker": "^", "zorder": 4},
    "cat": {"color": "#D97706", "linewidth": 2.0, "linestyle": "-", "marker": "D", "zorder": 4},
    "cit": {"color": "#7FB3FF", "linewidth": 1.8, "linestyle": "--", "marker": "o", "zorder": 3},
    "r_cforest": {"color": "#64748B", "linewidth": 1.8, "linestyle": "--", "marker": "o", "zorder": 2},
    "r_ctree": {"color": "#94A3B8", "linewidth": 1.8, "linestyle": ":", "marker": "o", "zorder": 1},
}


def _select_task_best_configs(df: pd.DataFrame) -> pd.DataFrame:
    perf = df.groupby(["method_base", "method_id"], as_index=False)["balanced_accuracy"].mean()
    best_idx = perf.groupby("method_base")["balanced_accuracy"].idxmax()
    return perf.loc[best_idx, ["method_base", "method_id"]]


def _compute_mean_ranks(df: pd.DataFrame, best_ids: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int]]:
    best = df.merge(best_ids, on=["method_base", "method_id"], how="inner")
    agg = (
        best.groupby(["method_base", "dataset", "downstream_model", "k"], as_index=False)["balanced_accuracy"]
        .mean()
    )

    parts: list[pd.DataFrame] = []
    support: dict[int, int] = {}

    for k_value in STANDARD_K:
        kdata = agg[agg["k"] == k_value].copy()
        cell_counts = (
            kdata.groupby(["dataset", "downstream_model"], as_index=False)["method_base"]
            .nunique()
            .rename(columns={"method_base": "n_methods"})
        )
        complete_cells = cell_counts[cell_counts["n_methods"] == len(ALL_CLF_METHODS)][["dataset", "downstream_model"]]
        kdata = kdata.merge(complete_cells, on=["dataset", "downstream_model"], how="inner")
        support[k_value] = int(kdata["dataset"].nunique())

        if kdata.empty:
            continue

        kdata["rank"] = kdata.groupby(["dataset", "downstream_model"])["balanced_accuracy"].rank(
            ascending=False,
            method="average",
        )
        parts.append(kdata)

    combined = pd.concat(parts, ignore_index=True)
    mean_ranks = combined.groupby(["method_base", "k"], as_index=False)["rank"].mean()
    return mean_ranks, support


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARXIV_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(RESULTS_DIR / "clf_evaluation.parquet")
    df = df[df["dataset_source"] == "real"].copy()
    df = df[df["k"].isin(STANDARD_K)]
    df = df[df["method_base"].isin(ALL_CLF_METHODS)]

    best_ids = _select_task_best_configs(df)
    mean_ranks, support = _compute_mean_ranks(df, best_ids)

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

    fig, ax = plt.subplots(figsize=(8.8, 4.9))

    for method in PLOT_METHODS:
        method_rows = mean_ranks[mean_ranks["method_base"] == method].sort_values("k")
        style = METHOD_STYLE[method]
        ax.plot(
            method_rows["k"],
            method_rows["rank"],
            label=DISPLAY_NAMES[method],
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=6.5 if method == "cif" else 5.5,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=style["zorder"],
        )

    ax.set_xlabel("Feature budget $k$")
    ax.set_ylabel("Mean rank (lower is better)")
    ax.set_xticks(STANDARD_K)
    ax.set_ylim(13.5, 3.2)
    ax.invert_yaxis()

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.6,
    )

    fig.tight_layout()

    for out_dir in (FIGURES_DIR, ARXIV_FIGURES_DIR):
        out_path = out_dir / "k_trajectory.png"
        fig.savefig(out_path)
        print(f"saved {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
