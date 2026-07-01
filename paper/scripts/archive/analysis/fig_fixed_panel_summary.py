"""Build a 14-dataset benchmark figure for the main classification claim.

Archived story figure; not part of the current arXiv figure bundle.

The main paper now leans on the 14-dataset benchmark rather than the
changing-support k-trajectory. The figure combines two views:
  1. The full 15-method 14-dataset ordering by mean balanced accuracy.
  2. CIF's paired mean balanced-accuracy delta against selected baselines.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ARXIV_FIGURES_DIR = Path(__file__).resolve().parents[2] / "arxiv" / "figures"

BASELINES = ["lgbm", "xgb", "cat", "rf", "cit", "r_cforest", "r_ctree"]
DISPLAY_NAMES = {
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "cat": "CatBoost",
    "rf": "RF",
    "cif": "CIF",
    "rfe": "RFE",
    "et": "ExtraTrees",
    "boruta": "Boruta",
    "cit": "CIT",
    "ptest_mc": "MC filter",
    "ptest_rdc": "RDC filter",
    "r_cforest": "cforest",
    "r_ctree": "ctree",
    "pi": "PI",
    "cpi": "CPI",
}
COLORS = {
    "lgbm": "#166534",
    "xgb": "#B91C1C",
    "cat": "#D97706",
    "rf": "#475569",
    "cif": "#2563EB",
    "rfe": "#94A3B8",
    "et": "#64748B",
    "boruta": "#A16207",
    "cit": "#7FB3FF",
    "ptest_mc": "#CBD5E1",
    "ptest_rdc": "#CBD5E1",
    "r_cforest": "#64748B",
    "r_ctree": "#94A3B8",
    "pi": "#CBD5E1",
    "cpi": "#E2E8F0",
}


def _load_pairwise() -> pd.DataFrame:
    df = pd.read_csv(TABLES_DIR / "paper_benchmark_fixed_panel_pairwise_ci.csv")
    df = df[(df["task"] == "classification") & (df["focus_method"] == "cif")].copy()
    df = df[df["baseline"].isin(BASELINES)].copy()
    df["display_name"] = df["baseline"].map(DISPLAY_NAMES)
    df["group"] = df["baseline"].map(
        {
            "lgbm": "Generic ensembles",
            "xgb": "Generic ensembles",
            "cat": "Generic ensembles",
            "rf": "Generic ensembles",
            "cit": "Conditional-inference references",
            "r_cforest": "Conditional-inference references",
            "r_ctree": "Conditional-inference references",
        }
    )
    order = {name: i for i, name in enumerate(BASELINES)}
    df["order"] = df["baseline"].map(order)
    return df.sort_values("order").reset_index(drop=True)


def _load_aggregate() -> pd.DataFrame:
    df = pd.read_csv(TABLES_DIR / "paper_benchmark_fixed_panel_aggregate.csv")
    df = df[df["task"] == "classification"].copy()
    df["display_name"] = df["method_base"].map(DISPLAY_NAMES)
    df["color"] = df["method_base"].map(COLORS).fillna("#CBD5E1")
    return df.sort_values("rank_position", ascending=True).reset_index(drop=True)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARXIV_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pairwise = _load_pairwise()
    aggregate = _load_aggregate()

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

    fig = plt.figure(figsize=(10.8, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.16)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    left_y = list(range(len(aggregate)))[::-1]
    ax_left.barh(
        left_y,
        aggregate["mean_score"],
        color=aggregate["color"],
        edgecolor="none",
        height=0.72,
    )
    ax_left.set_yticks(left_y)
    ax_left.set_yticklabels(aggregate["display_name"])
    ax_left.set_xlim(0.58, 0.83)
    ax_left.set_xlabel("Mean balanced accuracy")
    ax_left.set_title("A. 14-dataset ordering", fontsize=12, loc="left")
    ax_left.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax_left.grid(axis="y", visible=False)

    for y_pos, (_, row) in zip(left_y, aggregate.iterrows(), strict=False):
        ax_left.text(
            float(row["mean_score"]) + 0.0025,
            y_pos,
            f"#{int(row['rank_position'])}",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#334155",
        )

    cif_row = aggregate[aggregate["method_base"] == "cif"].iloc[0]
    cif_y = left_y[int(cif_row.name)]
    ax_left.axhline(cif_y, color="#BFDBFE", linewidth=12, alpha=0.18, zorder=0)

    right_y = list(range(len(pairwise)))[::-1]
    ax_right.axvline(0.0, color="#0F172A", linewidth=1.0, linestyle="--", zorder=1)

    split_y = right_y[3] - 0.5
    ax_right.axhline(split_y, color="#CBD5E1", linewidth=1.0)

    for y_pos, (_, row) in zip(right_y, pairwise.iterrows(), strict=False):
        baseline = str(row["baseline"])
        color = COLORS[baseline]
        mean_delta = float(row["mean_delta"])
        ci_lower = float(row["ci_lower"])
        ci_upper = float(row["ci_upper"])
        lower_err = mean_delta - ci_lower
        upper_err = ci_upper - mean_delta

        ax_right.errorbar(
            mean_delta,
            y_pos,
            xerr=[[lower_err], [upper_err]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.0,
            capsize=3.5,
            markersize=7.0,
            zorder=3,
        )
        ax_right.text(
            0.173,
            y_pos,
            f"{int(row['wins'])}/{int(row['n_datasets'])}",
            va="center",
            ha="right",
            fontsize=8.5,
            color="#334155",
        )

    ax_right.set_yticks(right_y)
    ax_right.set_yticklabels(pairwise["display_name"])
    ax_right.set_xlabel("Mean CIF - baseline balanced accuracy")
    ax_right.set_xlim(-0.04, 0.18)
    ax_right.set_ylim(-1, len(pairwise))
    ax_right.set_title("B. CIF vs selected baselines", fontsize=12, loc="left")

    ax_right.text(
        -0.038,
        right_y[0] + 0.8,
        "Generic ensembles",
        fontsize=9,
        color="#334155",
        fontweight="bold",
    )
    ax_right.text(
        -0.038,
        right_y[4] + 0.8,
        "Conditional-inference references",
        fontsize=9,
        color="#334155",
        fontweight="bold",
    )
    ax_right.text(0.173, len(pairwise) - 0.15, "Wins", fontsize=8.5, ha="right", color="#334155")

    for out_dir in (FIGURES_DIR, ARXIV_FIGURES_DIR):
        out_path = out_dir / "fixed_panel_story.png"
        fig.savefig(out_path)
        print(f"saved {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
