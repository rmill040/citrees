"""Precision@k curve flatness: 2-panel figure (line plot + retention bars).

Archived exploratory figure; not part of the current paper-facing rebuild path.

Left panel:  Precision@k curves for best config per method_base (by mean P@10).
Right panel: Precision retention (P@20 / P@5) bar chart, sorted descending.

Usage:
    uv run python paper/scripts/archive/analysis/fig_synthetic_precision_retention.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS = Path("paper/results")
FIGURES = RESULTS / "figures"
DATA_PATH = RESULTS / "synthetic_analysis_clf.parquet"
OUT_PATH = FIGURES / "precision_curve_flatness.png"

# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------
DISPLAY_NAMES: dict[str, str] = {
    "cif": "CIF",
    "cit": "CIT",
    "rf": "RF",
    "et": "ExtraTrees",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "cat": "CatBoost",
    "rfe": "RFE",
    "boruta": "Boruta",
    "pi": "PI",
    "cpi": "CPI",
    "ptest_mc": "MC-ptest",
    "ptest_rdc": "RDC-ptest",
    "r_ctree": "R-ctree",
    "r_cforest": "R-cforest",
}

# ---------------------------------------------------------------------------
# Category -> color palettes
# ---------------------------------------------------------------------------
FOREST_COLORS: dict[str, str] = {
    "cif": "#2563EB",  # bold blue (CIF hero)
    "rf": "#3B82F6",  # blue-500
    "et": "#60A5FA",  # blue-400
    "r_cforest": "#93C5FD",  # blue-300
}

BOOSTING_COLORS: dict[str, str] = {
    "xgb": "#DC2626",  # red-600
    "lgbm": "#F97316",  # orange-500
    "cat": "#FB923C",  # orange-400
}

WRAPPER_COLORS: dict[str, str] = {
    "rfe": "#16A34A",  # green-600
    "boruta": "#4ADE80",  # green-400
}

FILTER_COLORS: dict[str, str] = {
    "ptest_mc": "#9CA3AF",  # gray-400
    "ptest_rdc": "#6B7280",  # gray-500
    "pi": "#D1D5DB",  # gray-300
    "cpi": "#E5E7EB",  # gray-200
}

R_COLORS: dict[str, str] = {
    "r_ctree": "#FDBA74",  # orange-300
    "r_cforest": "#93C5FD",  # already in forest
}

CIT_COLOR = "#60A5FA"

# Build unified color map (category priority: forest > boosting > wrapper > R > filter)
METHOD_COLORS: dict[str, str] = {}
METHOD_COLORS.update(FILTER_COLORS)
METHOD_COLORS.update({"r_ctree": "#FDBA74"})
METHOD_COLORS.update(WRAPPER_COLORS)
METHOD_COLORS.update(BOOSTING_COLORS)
METHOD_COLORS.update(FOREST_COLORS)
METHOD_COLORS["cit"] = CIT_COLOR

# Category membership for bar chart coloring
CATEGORY_MAP: dict[str, str] = {}
for m in FOREST_COLORS:
    CATEGORY_MAP[m] = "forest"

for m in BOOSTING_COLORS:
    CATEGORY_MAP[m] = "boosting"
for m in WRAPPER_COLORS:
    CATEGORY_MAP[m] = "wrapper"
for m in FILTER_COLORS:
    CATEGORY_MAP[m] = "filter"
CATEGORY_MAP["r_ctree"] = "R"
# r_cforest is forest (already set)

BAR_CAT_COLORS: dict[str, str] = {
    "forest": "#3B82F6",
    "boosting": "#DC2626",
    "wrapper": "#16A34A",
    "filter": "#9CA3AF",
    "R": "#F97316",
}


def _setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 300,
        }
    )


def _load_and_prepare() -> pd.DataFrame:
    """Load synthetic CLF data, pick best config per method_base by mean P@10."""
    df = pd.read_parquet(DATA_PATH)
    df["method_base"] = df["method"].str.split("__").str[0]

    # Mean P@10 for each (method_base, method) pair
    p10 = (
        df.groupby(["method_base", "method"])["precision@10"]
        .mean()
        .reset_index()
        .sort_values("precision@10", ascending=False)
        .drop_duplicates("method_base", keep="first")
    )
    best_methods = p10["method"].tolist()

    # Filter to best configs and aggregate across all seeds/folds/datasets
    sub = df[df["method"].isin(best_methods)].copy()
    agg = (
        sub.groupby("method_base")[["precision@5", "precision@10", "precision@20"]]
        .mean()
        .sort_values("precision@10", ascending=False)
    )
    return agg


def _left_panel(ax: plt.Axes, agg: pd.DataFrame) -> None:
    """Precision@k line plot."""
    k_vals = [5, 10, 20]

    for method_base, row in agg.iterrows():
        color = METHOD_COLORS.get(method_base, "#6B7280")
        label = DISPLAY_NAMES.get(method_base, method_base)
        prec = [row["precision@5"], row["precision@10"], row["precision@20"]]
        lw = 2.5 if method_base == "cif" else 1.3
        alpha = 1.0 if method_base == "cif" else 0.8
        zorder = 10 if method_base == "cif" else 2
        marker = "o" if method_base == "cif" else "."
        ms = 7 if method_base == "cif" else 4

        ax.plot(
            k_vals,
            prec,
            color=color,
            linewidth=lw,
            alpha=alpha,
            zorder=zorder,
            marker=marker,
            markersize=ms,
            label=label,
        )

    ax.set_xlabel("k", fontsize=10)
    ax.set_ylabel("Mean precision", fontsize=10)
    ax.set_title("Precision@k curves (synthetic CLF)", fontsize=11, fontweight="bold")
    ax.set_xticks(k_vals)
    ax.set_xticklabels([str(k) for k in k_vals])

    # Legend: small, outside bottom or right
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        fontsize=6.5,
        ncol=3,
        loc="lower left",
        framealpha=0.9,
        edgecolor="#D1D5DB",
    )


def _right_panel(ax: plt.Axes, agg: pd.DataFrame) -> None:
    """Retention (P@20 / P@5) bar chart, sorted descending."""
    # Exclude r_ctree: its retention is >1 (artifact of near-zero P@5) and
    # dominates the scale, making the chart unreadable for all other methods.
    agg_filt = agg.drop("r_ctree", errors="ignore")

    retention = (agg_filt["precision@20"] / agg_filt["precision@5"]).sort_values(ascending=False)

    colors = [BAR_CAT_COLORS.get(CATEGORY_MAP.get(m, "filter"), "#9CA3AF") for m in retention.index]
    display_labels = [DISPLAY_NAMES.get(m, m) for m in retention.index]

    bars = ax.barh(
        range(len(retention)),
        retention.values,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    # Annotate retention %
    for _i, (bar, val) in enumerate(zip(bars, retention.values, strict=False)):
        ax.text(
            val + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#374151",
        )

    # Mean line
    mean_ret = retention.mean()
    ax.axvline(mean_ret, color="#6B7280", linestyle="--", linewidth=1.0, alpha=0.7, zorder=0)
    ax.text(
        mean_ret + 0.003,
        len(retention) - 0.3,
        f"mean={mean_ret:.3f}",
        fontsize=7,
        color="#6B7280",
        va="top",
    )

    ax.set_yticks(range(len(retention)))
    ax.set_yticklabels(display_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Retention (P@20 / P@5)", fontsize=10)
    ax.set_title("Precision retention (P@20 / P@5)", fontsize=11, fontweight="bold")

    # Category legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=BAR_CAT_COLORS["forest"], label="Forest"),
        Patch(facecolor=BAR_CAT_COLORS["boosting"], label="Boosting"),
        Patch(facecolor=BAR_CAT_COLORS["wrapper"], label="Wrapper"),
        Patch(facecolor=BAR_CAT_COLORS["filter"], label="Filter"),
        Patch(facecolor=BAR_CAT_COLORS["R"], label="R"),
    ]
    ax.legend(
        handles=legend_elements,
        fontsize=7,
        loc="lower right",
        framealpha=0.9,
        edgecolor="#D1D5DB",
    )


def main() -> None:
    _setup_style()
    FIGURES.mkdir(parents=True, exist_ok=True)

    agg = _load_and_prepare()

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        gridspec_kw={"width_ratios": [1, 1]},
    )

    _left_panel(ax_left, agg)
    _right_panel(ax_right, agg)

    fig.tight_layout(pad=1.5)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {OUT_PATH}")
    print(f"Size:  {OUT_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
