"""Build the paper-facing synthetic top-k focus curves from the saved focus table.

This figure matches the main-text 2x2 top-k comparison used in the paper while
avoiding a dependency on the full synthetic ranking parquet files.

Outputs:
  - paper/results/figures/synthetic_topk_focus_curves.png
  - paper/arxiv/figures/synthetic_topk_focus_curves.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ARXIV_FIGURES_DIR = Path(__file__).resolve().parents[2] / "arxiv" / "figures"

FOCUS_K = [5, 10, 25, 50, 100]
FOCUS_K_POSITIONS = {k: i for i, k in enumerate(FOCUS_K)}
FOCUS_METHODS = ["cit", "dt", "rt", "cif", "rf", "et", "xgb", "lgbm", "cat"]
FAMILY_PANELS = (
    ("Single trees", ("cit", "dt", "rt")),
    ("Forests", ("cif", "rf", "et")),
    ("Boosted trees", ("cif", "xgb", "lgbm", "cat")),
)
DISPLAY_NAMES = {
    "cif": "CIF",
    "cit": "CIT",
    "dt": "DT",
    "rt": "RT",
    "rf": "RF",
    "et": "ExtraTrees",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "cat": "CatBoost",
}
PANEL_COLORS = ("#60A5FA", "#94A3B8", "#FBBF24", "#A78BFA")
TASK_TITLES = {
    "classification": "Classification",
    "regression": "Regression",
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


def _plot_family(
    ax: plt.Axes,
    df: pd.DataFrame,
    task: str,
    methods: tuple[str, ...],
    title: str,
) -> None:
    task_df = df[(df["task"] == task) & (df["k"].isin(FOCUS_K))].copy()

    for position, method in enumerate(methods):
        method_df = task_df[task_df["method_base"] == method].sort_values("k")
        if method_df.empty:
            continue

        is_focus = method in {"cit", "cif"}
        x = method_df["k"].map(FOCUS_K_POSITIONS)
        ax.plot(
            x,
            method_df["informative_share"],
            color=PANEL_COLORS[position],
            linewidth=2.6 if is_focus else 1.8,
            alpha=1.0 if is_focus else 0.9,
            marker="o" if method in {"cif", "cit"} else "s",
            markersize=6.5 if is_focus else 5.0,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=DISPLAY_NAMES[method],
            zorder=10 if is_focus else 5,
        )

    ax.set_title(title, pad=6)
    ax.set_xticks(range(len(FOCUS_K)))
    ax.set_xticklabels([str(k) for k in FOCUS_K])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, axis="both")
    ax.legend(loc="upper right", frameon=False, fontsize=8, handlelength=1.6)


def main() -> None:
    _setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARXIV_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    focus = pd.read_csv(TABLES_DIR / "synthetic_topk_composition_focus.csv")
    focus = focus[focus["method_base"].isin(FOCUS_METHODS) & focus["k"].isin(FOCUS_K)].copy()
    if focus.empty:
        raise RuntimeError("No focus rows available for the synthetic top-k figure.")

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.2), sharex=True, sharey=True)
    for row_idx, task in enumerate(("classification", "regression")):
        for col_idx, (family_title, methods) in enumerate(FAMILY_PANELS):
            title = family_title if row_idx == 0 else ""
            _plot_family(axes[row_idx, col_idx], focus, task=task, methods=methods, title=title)
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(
                    f"{TASK_TITLES[task]}\n\\% informative\namong selected"
                )
            if row_idx == 1:
                axes[row_idx, col_idx].set_xlabel(r"Number of selected features ($k$)")

    fig.subplots_adjust(top=0.92, hspace=0.18, wspace=0.16, bottom=0.12, left=0.12)

    for out_dir in (FIGURES_DIR, ARXIV_FIGURES_DIR):
        out_path = out_dir / "synthetic_topk_focus_curves.png"
        fig.savefig(out_path)
        print(f"Saved {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
