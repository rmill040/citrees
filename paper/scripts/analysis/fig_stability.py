"""Stability--accuracy scatter: ranking instability vs downstream variance.

Produces a single-panel scatter showing that methods with unstable feature
rankings do not necessarily have high downstream accuracy variance (and vice
versa).  One point per method (15 methods), colored by category, sized by
mean balanced accuracy.

Usage:
    uv run python paper/scripts/analysis/fig_stability.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS = Path("paper/results")
FIGURES = RESULTS / "figures"
OUT_PATH = FIGURES / "stability_paradox.png"

# ---------------------------------------------------------------------------
# Display names (consistent with figures_benchmark.py)
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
# Category assignments and colors
# ---------------------------------------------------------------------------
CATEGORIES: dict[str, list[str]] = {
    "Forest": ["cif", "rf", "et", "r_cforest"],
    "Boosting": ["xgb", "lgbm", "cat"],
    "Wrapper": ["boruta", "rfe", "pi", "cpi"],
    "Filter": ["ptest_mc", "ptest_rdc"],
    "Single tree": ["r_ctree", "cit"],
}

CATEGORY_COLORS: dict[str, str] = {
    "Forest": "#2563EB",
    "Boosting": "#DC2626",
    "Wrapper": "#16A34A",
    "Filter": "#7C3AED",
    "Single tree": "#F97316",
}


def _method_to_category(m: str) -> str:
    for cat, members in CATEGORIES.items():
        if m in members:
            return cat
    return "Other"


def _dn(m: str) -> str:
    return DISPLAY_NAMES.get(m, m)


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------
def _nogueira_at_k(rankings: list[np.ndarray], k: int, p: int) -> float:
    """Nogueira stability index for top-k feature sets.

    Parameters
    ----------
    rankings : list of arrays
        Each array is a full feature ranking (ordered indices).
    k : int
        Number of top features to consider as "selected".
    p : int
        Total number of features in the dataset.

    Returns
    -------
    float
        Stability index in [-1, 1]. Higher = more stable.
    """
    M = len(rankings)
    if M < 2 or p == 0:
        return 1.0

    # Selection frequency per feature
    freq = np.zeros(p)
    for r in rankings:
        for f in r[:k]:
            if 0 <= f < p:
                freq[f] += 1
    freq /= M

    k_bar = float(k)
    numerator = (1.0 / p) * np.sum(freq * (1.0 - freq))
    denominator = (k_bar / p) * (1.0 - k_bar / p)
    if denominator == 0:
        return 1.0
    return float(1.0 - numerator / denominator)


def compute_instability(df_rank: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    """Compute ranking instability per method_base using Nogueira index at multiple k.

    For each dataset x method_base, compute the Nogueira stability index
    across all 25 seed x fold ranking vectors at each k value.
    Instability = 1 - Nogueira.  Average across datasets.

    Parameters
    ----------
    df_rank : pd.DataFrame
        Rankings dataframe with columns: method_base, dataset, feature_ranking
    k_values : list[int]
        List of k values to compute stability at (e.g., [5, 10, 25, 50, 100])

    Returns
    -------
    pd.DataFrame
        Long-form dataframe with columns: method_base, k, instability
    """
    records: list[dict] = []

    for mb, grp in df_rank.groupby("method_base"):
        for k in k_values:
            dataset_stabs: list[float] = []
            for _ds, ds_grp in grp.groupby("dataset"):
                rankings = ds_grp["feature_ranking"].values.tolist()
                if len(rankings) < 2:
                    continue
                # Total features = length of first ranking vector
                p = len(rankings[0])
                # Skip if k > p
                if k > p:
                    continue
                stab = _nogueira_at_k(rankings, k=k, p=p)
                dataset_stabs.append(stab)

            if not dataset_stabs:
                continue

            mean_stab = float(np.mean(dataset_stabs))
            records.append({"method_base": mb, "k": k, "instability": 1.0 - mean_stab})

    return pd.DataFrame(records)


def compute_accuracy_stats(df_eval: pd.DataFrame) -> pd.DataFrame:
    """Compute within-dataset prediction variance per method_base at each k.

    For each method x k, compute std of balanced accuracy across seeds/folds
    within each dataset, then average across datasets.  This measures
    how stable the method's predictions are, not how variable the datasets are.

    Parameters
    ----------
    df_eval : pd.DataFrame
        Evaluation dataframe with columns: method_base, k, dataset, balanced_accuracy

    Returns
    -------
    pd.DataFrame
        Long-form dataframe with columns: method_base, k, acc_std, acc_mean
    """
    records: list[dict] = []

    for (mb, k), grp in df_eval.groupby(["method_base", "k"]):
        # Within-dataset std: for each dataset, std across seed x fold runs
        ds_stds = grp.groupby("dataset")["balanced_accuracy"].std()
        records.append({
            "method_base": mb,
            "k": k,
            "acc_std": float(ds_stds.mean()),
            "acc_mean": float(grp["balanced_accuracy"].mean()),
        })

    return pd.DataFrame(records)


def select_best_config(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the best method_id per method_base (highest mean BA)."""
    mean_ba = df.groupby(["method_base", "method_id"])["balanced_accuracy"].mean()
    best_ids = mean_ba.groupby("method_base").idxmax()
    best_method_ids = {mb: mid for mb, (_, mid) in best_ids.items()}

    masks = []
    for mb, mid in best_method_ids.items():
        masks.append((df["method_base"] == mb) & (df["method_id"] == mid))
    keep = masks[0]
    for m in masks[1:]:
        keep = keep | m
    return df[keep].copy(), best_method_ids


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(plot_df: pd.DataFrame, k_display: int) -> None:
    """Render and save the stability-paradox scatter plot for a specific k.

    Parameters
    ----------
    plot_df : pd.DataFrame
        Plotting data for a single k value
    k_display : int
        The k value being displayed
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Quadrant lines at medians
    med_x = plot_df["instability"].median()
    med_y = plot_df["acc_std"].median()
    ax.axvline(med_x, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, zorder=1)
    ax.axhline(med_y, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, zorder=1)

    # Size: proportional to mean accuracy (scale for visibility)
    size_min, size_max = 60, 280
    acc_vals = plot_df["acc_mean"].values
    acc_norm = (acc_vals - acc_vals.min()) / (acc_vals.max() - acc_vals.min() + 1e-10)
    sizes = size_min + acc_norm * (size_max - size_min)

    # Plot by category for legend grouping
    for cat, color in CATEGORY_COLORS.items():
        mask = plot_df["category"] == cat
        if not mask.any():
            continue
        sub = plot_df[mask]
        idx = sub.index
        ax.scatter(
            sub["instability"],
            sub["acc_std"],
            c=color,
            s=sizes[idx],
            alpha=0.85,
            edgecolors="white",
            linewidth=0.7,
            label=cat,
            zorder=3,
        )

    # Labels for each point
    for i, row in plot_df.iterrows():
        label = _dn(row["method_base"])
        # CIF gets an arrow annotation
        if row["method_base"] == "cif":
            ax.annotate(
                label,
                xy=(row["instability"], row["acc_std"]),
                xytext=(25, 20),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color=CATEGORY_COLORS["Forest"],
                arrowprops=dict(
                    arrowstyle="->",
                    color=CATEGORY_COLORS["Forest"],
                    linewidth=1.2,
                    connectionstyle="arc3,rad=0.15",
                ),
                zorder=5,
            )
        else:
            ax.annotate(
                label,
                xy=(row["instability"], row["acc_std"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=6.5,
                alpha=0.8,
                zorder=4,
            )

    # Spearman correlation annotation
    rho, pval = stats.spearmanr(plot_df["instability"], plot_df["acc_std"])
    ax.text(
        0.03,
        0.97,
        f"$\\rho$ = {rho:.2f}, $p$ = {pval:.2f}",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xlabel(f"Ranking instability  (1 $-$ Nogueira@{k_display})", fontsize=10)
    ax.set_ylabel("Within-dataset prediction std", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9, edgecolor="gray")

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved -> {OUT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("STABILITY PARADOX FIGURE (ALL K VALUES)")
    print("=" * 60)

    # K values to analyze
    K_VALUES = [5, 10, 25, 50, 100]
    K_DISPLAY = 10  # Representative k for figure

    # ------------------------------------------------------------------
    # 1. Load evaluation data and pick best config per method_base
    # ------------------------------------------------------------------
    print(f"[1/5] Loading evaluation data (real, LR, k ∈ {K_VALUES})...")
    df_eval = pd.read_parquet(RESULTS / "clf_evaluation.parquet")
    df_eval = df_eval[
        (df_eval["dataset_source"] == "real")
        & (df_eval["downstream_model"] == "lr")
        & (df_eval["k"].isin(K_VALUES))
    ]

    # Select best config based on mean BA across all k values
    df_eval_best, best_ids = select_best_config(df_eval)
    print(f"  {len(best_ids)} methods, {len(df_eval_best)} observations")

    # ------------------------------------------------------------------
    # 2. Load rankings data, filter to best configs and real datasets
    # ------------------------------------------------------------------
    print("[2/5] Loading rankings data (real, best configs)...")
    df_rank = pd.read_parquet(RESULTS / "clf_rankings.parquet")
    df_rank = df_rank[df_rank["dataset_source"] == "real"]

    # Keep only best config per method_base
    masks = []
    for mb, mid in best_ids.items():
        masks.append((df_rank["method_base"] == mb) & (df_rank["method_id"] == mid))
    keep = masks[0]
    for m in masks[1:]:
        keep = keep | m
    df_rank_best = df_rank[keep].copy()
    print(f"  {df_rank_best['method_base'].nunique()} methods, {len(df_rank_best)} ranking vectors")

    # ------------------------------------------------------------------
    # 3. Compute instability and accuracy stats at all k values
    # ------------------------------------------------------------------
    print(f"[3/5] Computing instability and accuracy stats at k ∈ {K_VALUES}...")
    instab_df = compute_instability(df_rank_best, k_values=K_VALUES)
    acc_df = compute_accuracy_stats(df_eval_best)

    # Merge on method_base AND k
    full_df = instab_df.merge(acc_df, on=["method_base", "k"])
    full_df["category"] = full_df["method_base"].apply(_method_to_category)

    print(f"  {len(full_df)} method x k combinations")
    print(f"  Methods: {full_df['method_base'].nunique()}, K values: {sorted(full_df['k'].unique())}")

    # ------------------------------------------------------------------
    # 4. Analyze stability-accuracy paradox at each k
    # ------------------------------------------------------------------
    print("\n[4/5] Stability-accuracy paradox analysis:")
    print("=" * 80)

    correlation_results = []
    for k in sorted(full_df["k"].unique()):
        k_df = full_df[full_df["k"] == k].copy()
        if len(k_df) < 3:  # Need at least 3 points for correlation
            continue

        rho, pval = stats.spearmanr(k_df["instability"], k_df["acc_std"])
        correlation_results.append({
            "k": k,
            "rho": rho,
            "pval": pval,
            "n_methods": len(k_df)
        })

        print(f"\nk = {k:3d} (n={len(k_df):2d} methods):")
        print(f"  Spearman ρ = {rho:+.3f}, p = {pval:.4f}")
        print(f"  Instability range: [{k_df['instability'].min():.3f}, {k_df['instability'].max():.3f}]")
        print(f"  Acc std range:     [{k_df['acc_std'].min():.4f}, {k_df['acc_std'].max():.4f}]")
        print(f"  Acc mean range:    [{k_df['acc_mean'].min():.3f}, {k_df['acc_mean'].max():.3f}]")

        # Show top 5 most/least stable methods
        k_df_sorted = k_df.sort_values("instability")
        print(f"\n  Most stable (lowest instability):")
        for _, row in k_df_sorted.head(5).iterrows():
            print(
                f"    {_dn(row['method_base']):15s}  "
                f"instab={row['instability']:.3f}  "
                f"acc_std={row['acc_std']:.4f}  "
                f"acc_mean={row['acc_mean']:.3f}"
            )

        print(f"\n  Least stable (highest instability):")
        for _, row in k_df_sorted.tail(5).iterrows():
            print(
                f"    {_dn(row['method_base']):15s}  "
                f"instab={row['instability']:.3f}  "
                f"acc_std={row['acc_std']:.4f}  "
                f"acc_mean={row['acc_mean']:.3f}"
            )

    # Print correlation summary table
    print("\n" + "=" * 80)
    print("CORRELATION SUMMARY ACROSS K VALUES:")
    print("=" * 80)
    print(f"{'k':>5s}  {'n':>3s}  {'ρ':>7s}  {'p-value':>9s}  {'Interpretation':s}")
    print("-" * 80)
    for result in correlation_results:
        interp = "weak/no correlation" if abs(result["rho"]) < 0.3 else "moderate correlation"
        print(
            f"{result['k']:5d}  {result['n_methods']:3d}  "
            f"{result['rho']:+7.3f}  {result['pval']:9.4f}  {interp}"
        )

    # Overall interpretation
    mean_rho = np.mean([r["rho"] for r in correlation_results])
    print("-" * 80)
    print(f"Mean ρ across k: {mean_rho:+.3f}")
    if abs(mean_rho) < 0.3:
        print("→ Ranking instability and prediction variance are WEAKLY correlated.")
        print("→ This is the 'stability paradox': unstable rankings ≠ unstable predictions.")
    else:
        print("→ Ranking instability and prediction variance show MODERATE correlation.")

    # ------------------------------------------------------------------
    # 5. Generate figure for representative k
    # ------------------------------------------------------------------
    print(f"\n[5/5] Generating figure for k={K_DISPLAY}...")
    plot_df = full_df[full_df["k"] == K_DISPLAY].copy().reset_index(drop=True)

    if len(plot_df) == 0:
        print(f"ERROR: No data for k={K_DISPLAY}")
        return

    make_figure(plot_df, k_display=K_DISPLAY)
    print("\nDone.")


if __name__ == "__main__":
    main()
