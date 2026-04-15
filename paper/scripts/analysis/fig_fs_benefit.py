"""Feature-selection benefit heatmap (CLF | REG).

2-panel figure (1x2): for each task, shows how much selecting k features
helps or hurts compared to using all features (k=p).

Delta = mean_score_at_k - mean_score_at_k_p

Usage:
    uv run python paper/scripts/analysis/fig_fs_benefit.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS = Path("paper/results")
FIGURES = RESULTS / "figures"
OUT_PATH = FIGURES / "fs_benefit.png"

K_BUDGETS = [5, 10, 25, 50, 100]
TOP_N = 8

DISPLAY_NAMES = {
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
    "ptest_pc": "PC-ptest",
    "ptest_dc": "DC-ptest",
    "r_ctree": "R-ctree",
    "r_cforest": "R-cforest",
}


def _dn(m: str) -> str:
    return DISPLAY_NAMES.get(m, m)


def _compute_deltas(
    df: pd.DataFrame,
    downstream_model: str,
    metric: str,
) -> pd.DataFrame:
    """Compute delta = mean_score_at_k - mean_score_at_k_p for best config per method_base.

    Returns a DataFrame with columns: method_base, k, delta, mean_score.
    """
    sub = df[(df["dataset_source"] == "real") & (df["downstream_model"] == downstream_model)].copy()

    # Build the k=p baseline: for each dataset, k=p equals n_features (max k).
    baseline = sub[sub["k"] == sub["n_features"]].copy()
    baseline_scores = (
        baseline.groupby(["method_base", "method_id", "dataset"])[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: "score_kp"})
    )

    # Scores at standard k budgets.
    k_scores = sub[sub["k"].isin(K_BUDGETS)].copy()
    k_agg = (
        k_scores.groupby(["method_base", "method_id", "dataset", "k"])[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: "score_k"})
    )

    # Merge to get delta per (method_id, dataset, k).
    merged = k_agg.merge(baseline_scores, on=["method_base", "method_id", "dataset"], how="inner")
    merged["delta"] = merged["score_k"] - merged["score_kp"]

    # For each method_base, pick the best config (highest mean score across all k and datasets).
    config_mean = merged.groupby(["method_base", "method_id"])["score_k"].mean().reset_index()
    best_config = config_mean.loc[config_mean.groupby("method_base")["score_k"].idxmax()]
    best_ids = best_config[["method_base", "method_id"]]

    merged = merged.merge(best_ids, on=["method_base", "method_id"], how="inner")

    # Average delta across datasets for each (method_base, k).
    deltas = merged.groupby(["method_base", "k"]).agg(
        delta=("delta", "mean"),
        mean_score=("score_k", "mean"),
    ).reset_index()

    return deltas


def _build_panel(
    ax: plt.Axes,
    deltas: pd.DataFrame,
    title: str,
) -> None:
    """Draw a single heatmap panel on the given axes."""
    # Select top-N methods by mean score across all k values.
    method_rank = deltas.groupby("method_base")["mean_score"].mean().sort_values(ascending=False)
    top_methods = list(method_rank.head(TOP_N).index)

    deltas_top = deltas[deltas["method_base"].isin(top_methods)].copy()

    # Pivot: rows = method_base (sorted by mean score), columns = k.
    pivot = deltas_top.pivot(index="method_base", columns="k", values="delta")
    # Reorder rows by mean score (best on top).
    pivot = pivot.loc[[m for m in method_rank.index if m in pivot.index]]
    pivot = pivot[K_BUDGETS]  # Ensure column order.

    # Rename for display.
    pivot.index = [_dn(m) for m in pivot.index]
    pivot.columns = [f"k={k}" for k in pivot.columns]

    # Determine symmetric color limits.
    vmax = max(abs(pivot.values[np.isfinite(pivot.values)].min()), abs(pivot.values[np.isfinite(pivot.values)].max()))
    vmax = max(vmax, 0.005)  # Ensure some color range even if deltas are tiny.

    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Delta score vs k=p", "shrink": 0.8},
        annot_kws={"size": 8},
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="y", rotation=0)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })

    # --- CLF ---
    print("Loading CLF evaluation data...")
    clf_df = pd.read_parquet(RESULTS / "clf_evaluation.parquet")
    clf_deltas = _compute_deltas(clf_df, downstream_model="lr", metric="balanced_accuracy")
    print(f"  CLF deltas: {len(clf_deltas)} rows, {clf_deltas['method_base'].nunique()} methods")

    # --- REG ---
    print("Loading REG evaluation data...")
    reg_df = pd.read_parquet(RESULTS / "reg_evaluation.parquet")
    reg_deltas = _compute_deltas(reg_df, downstream_model="ridge", metric="r2")
    print(f"  REG deltas: {len(reg_deltas)} rows, {reg_deltas['method_base'].nunique()} methods")

    # --- Figure ---
    fig, (ax_clf, ax_reg) = plt.subplots(1, 2, figsize=(12, 5))

    _build_panel(ax_clf, clf_deltas, "Classification (LR, balanced accuracy)")
    _build_panel(ax_reg, reg_deltas, "Regression (Ridge, R\u00b2)")

    fig.tight_layout(w_pad=3.0)
    fig.savefig(OUT_PATH)
    plt.close(fig)

    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
