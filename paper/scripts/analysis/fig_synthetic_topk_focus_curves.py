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

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ARXIV_FIGURES_DIR = Path(__file__).resolve().parents[2] / "arxiv" / "figures"

FOCUS_K = [5, 10, 25, 50, 100]
FOCUS_METHODS = ["cif", "rf", "et", "cit"]
DISPLAY_NAMES = {
    "cif": "CIF",
    "rf": "RF",
    "et": "ExtraTrees",
    "cit": "CIT",
}
METHOD_COLORS = {
    "cif": "#2563EB",
    "rf": "#EA580C",
    "et": "#CA8A04",
    "cit": "#60A5FA",
}
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


def _plot_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    task: str,
    metric: str,
    title: str,
    ylabel: str | None = None,
) -> None:
    task_df = df[(df["task"] == task) & (df["k"].isin(FOCUS_K))].copy()

    for method in FOCUS_METHODS:
        method_df = task_df[task_df["method_base"] == method].sort_values("k")
        if method_df.empty:
            continue

        is_cif = method == "cif"
        ax.plot(
            method_df["k"],
            method_df[metric],
            color=METHOD_COLORS[method],
            linewidth=2.6 if is_cif else 1.8,
            alpha=1.0 if is_cif else 0.9,
            marker="o" if method in {"cif", "cit"} else "s",
            markersize=6.5 if is_cif else 5.0,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=DISPLAY_NAMES[method],
            zorder=10 if is_cif else 5,
        )

    ax.set_title(f"{TASK_TITLES[task]}: {title}", pad=6)
    ax.set_xlabel(r"Feature budget $k$")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xticks(FOCUS_K)
    ax.set_xticklabels([str(k) for k in FOCUS_K])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, axis="both")


def main() -> None:
    _setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARXIV_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    focus = pd.read_csv(TABLES_DIR / "synthetic_topk_composition_focus.csv")
    focus = focus[focus["method_base"].isin(FOCUS_METHODS) & focus["k"].isin(FOCUS_K)].copy()
    if focus.empty:
        raise RuntimeError("No focus rows available for the synthetic top-k figure.")

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), sharex=True, sharey="col")

    _plot_metric(
        axes[0, 0],
        focus,
        task="classification",
        metric="informative_share",
        title="Informative Share",
        ylabel="Share of top-k",
    )
    _plot_metric(
        axes[0, 1],
        focus,
        task="classification",
        metric="pure_noise_share",
        title="Pure-Noise Share",
    )
    _plot_metric(
        axes[1, 0],
        focus,
        task="regression",
        metric="informative_share",
        title="Informative Share",
        ylabel="Share of top-k",
    )
    _plot_metric(
        axes[1, 1],
        focus,
        task="regression",
        metric="pure_noise_share",
        title="Pure-Noise Share",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.955),
        frameon=False,
        handlelength=2.0,
        columnspacing=1.5,
    )
    fig.suptitle(
        "Synthetic Top-k Recovery: CIF vs RF / ExtraTrees / CIT",
        fontsize=13,
        y=0.99,
    )
    fig.subplots_adjust(top=0.84, hspace=0.28, wspace=0.16, bottom=0.10)

    for out_dir in (FIGURES_DIR, ARXIV_FIGURES_DIR):
        out_path = out_dir / "synthetic_topk_focus_curves.png"
        fig.savefig(out_path)
        print(f"Saved {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
