"""Dataset-level winner analysis: who wins at each (dataset, k) combination?

Determines whether there is a "universal best" feature-selection method or
whether the winner is dataset-dependent.  Computes per-downstream-model
frequency tables, dataset-level stability profiles, CIF catchup rates, and
a heatmap of winning methods with CIF rank annotations.

Outputs:
  paper/results/figures/dataset_winners_heatmap.png

Usage:
    UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/fig_dataset_winners.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS = Path("paper/results")
FIGURES = RESULTS / "figures"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K_VALUES = [5, 10, 25, 50, 100]
DOWNSTREAM_MODELS = ["lr", "svm", "knn"]

# All 15 method_base values in the experiment
ALL_METHODS = [
    "boruta", "cat", "cif", "cit", "cpi", "et", "lgbm",
    "pi", "ptest_mc", "ptest_rdc", "r_cforest", "r_ctree", "rf", "rfe", "xgb",
]

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

# Distinct colors for each method (for heatmap cells)
METHOD_COLORS: dict[str, str] = {
    "cif": "#2563EB",
    "cit": "#7DD3FC",
    "rf": "#EA580C",
    "et": "#CA8A04",
    "lgbm": "#16A34A",
    "xgb": "#DC2626",
    "cat": "#9333EA",
    "rfe": "#0891B2",
    "boruta": "#DB2777",
    "pi": "#78716C",
    "cpi": "#A3A3A3",
    "ptest_mc": "#65A30D",
    "ptest_rdc": "#059669",
    "r_ctree": "#F59E0B",
    "r_cforest": "#D97706",
}


def _setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_and_prepare(downstream_model: str) -> pd.DataFrame:
    """Load clf_evaluation, filter to real data, select best config per method_base.

    Returns a DataFrame with columns: method_base, dataset, k, balanced_accuracy
    (averaged across seeds and folds).
    """
    df = pd.read_parquet(RESULTS / "clf_evaluation.parquet")

    mask = (
        (df["dataset_source"] == "real")
        & (df["downstream_model"] == downstream_model)
        & (df["k"].isin(K_VALUES))
    )
    sub = df.loc[mask].copy()

    # Best global config per method_base: highest mean BA across all
    global_perf = (
        sub.groupby(["method_base", "method_id"])["balanced_accuracy"]
        .mean()
        .reset_index()
    )
    best_ids = (
        global_perf.sort_values("balanced_accuracy", ascending=False)
        .groupby("method_base")["method_id"]
        .first()
        .to_dict()
    )

    sub = sub[
        sub.apply(lambda r: r["method_id"] == best_ids.get(r["method_base"]), axis=1)
    ].copy()

    # Average across seeds and folds
    dataset_means = (
        sub.groupby(["method_base", "dataset", "k"])["balanced_accuracy"]
        .mean()
        .reset_index()
    )

    return dataset_means


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------
def _compute_winners(dataset_means: pd.DataFrame) -> pd.DataFrame:
    """For each (dataset, k), find the winner, runner-up, and CIF rank/gap.

    Returns one row per (dataset, k) with columns:
        dataset, k, winner, winner_ba, runner_up, runner_up_ba,
        cif_rank, cif_ba, gap_to_winner, n_methods
    """
    records: list[dict] = []

    for (ds, k), grp in dataset_means.groupby(["dataset", "k"]):
        ranked = grp.sort_values("balanced_accuracy", ascending=False).reset_index(drop=True)
        n_methods = len(ranked)

        winner = ranked.iloc[0]["method_base"]
        winner_ba = ranked.iloc[0]["balanced_accuracy"]

        runner_up = ranked.iloc[1]["method_base"] if n_methods > 1 else None
        runner_up_ba = ranked.iloc[1]["balanced_accuracy"] if n_methods > 1 else None

        cif_row = ranked[ranked["method_base"] == "cif"]
        if cif_row.empty:
            cif_rank = np.nan
            cif_ba = np.nan
            gap = np.nan
        else:
            cif_idx = cif_row.index[0]
            cif_rank = cif_idx + 1  # 1-indexed rank
            cif_ba = cif_row.iloc[0]["balanced_accuracy"]
            gap = winner_ba - cif_ba

        records.append({
            "dataset": ds,
            "k": k,
            "winner": winner,
            "winner_ba": winner_ba,
            "runner_up": runner_up,
            "runner_up_ba": runner_up_ba,
            "cif_rank": cif_rank,
            "cif_ba": cif_ba,
            "gap_to_winner": gap,
            "n_methods": n_methods,
        })

    return pd.DataFrame(records)


def _frequency_table(winners_df: pd.DataFrame) -> pd.DataFrame:
    """How often does each method_base win across all (dataset, k) combinations?"""
    freq = (
        winners_df["winner"]
        .value_counts()
        .reset_index()
        .rename(columns={"winner": "method_base", "count": "wins"})
    )
    freq["pct"] = (freq["wins"] / len(winners_df) * 100).round(1)
    return freq


def _dataset_profiles(winners_df: pd.DataFrame) -> pd.DataFrame:
    """For each dataset: who wins at each k? Is the winner stable or contested?"""
    pivot = winners_df.pivot(index="dataset", columns="k", values="winner")
    pivot = pivot[K_VALUES]

    # Classify: stable if same winner at all k, contested otherwise
    def classify(row: pd.Series) -> str:
        unique = row.dropna().unique()
        return "stable" if len(unique) == 1 else "contested"

    pivot["stability"] = pivot.apply(classify, axis=1)

    # On contested datasets, does CIF win at the highest k present?
    def cif_wins_high_k(row: pd.Series) -> bool | None:
        if row["stability"] == "stable":
            return None
        # Check from highest k downward
        for k in reversed(K_VALUES):
            if pd.notna(row.get(k)):
                return row[k] == "cif"
        return None

    pivot["cif_wins_at_max_k"] = pivot.apply(cif_wins_high_k, axis=1)

    return pivot


def _cif_catchup_rate(winners_df: pd.DataFrame) -> dict:
    """What fraction of datasets see CIF's rank improve from k=5 to k=100?"""
    datasets = winners_df["dataset"].unique()
    improved = 0
    worsened = 0
    unchanged = 0
    valid = 0

    for ds in datasets:
        sub = winners_df[winners_df["dataset"] == ds]
        r5 = sub[sub["k"] == 5]["cif_rank"].values
        r100 = sub[sub["k"] == 100]["cif_rank"].values
        if len(r5) == 0 or len(r100) == 0 or np.isnan(r5[0]) or np.isnan(r100[0]):
            continue
        valid += 1
        if r100[0] < r5[0]:
            improved += 1
        elif r100[0] > r5[0]:
            worsened += 1
        else:
            unchanged += 1

    return {
        "valid_datasets": valid,
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
        "catchup_rate": improved / valid if valid > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------
def _plot_heatmap(
    winners_df: pd.DataFrame,
    ax: plt.Axes,
    title: str,
) -> set[str]:
    """Draw a heatmap: datasets (rows) x k (cols), colored by winner, CIF rank annotated.

    Returns the set of method_base values that appear as winners.
    """
    datasets = sorted(winners_df["dataset"].unique())
    n_ds = len(datasets)
    n_k = len(K_VALUES)

    ds_to_idx = {ds: i for i, ds in enumerate(datasets)}

    # Build color grid and annotation grid
    used_methods: set[str] = set()

    for _, row in winners_df.iterrows():
        ds_idx = ds_to_idx[row["dataset"]]
        k_idx = K_VALUES.index(int(row["k"]))
        winner = row["winner"]
        color = METHOD_COLORS.get(winner, "#D1D5DB")
        used_methods.add(winner)

        # Draw colored rectangle
        rect = plt.Rectangle(
            (k_idx - 0.5, ds_idx - 0.5), 1, 1,
            facecolor=color, edgecolor="white", linewidth=0.5,
        )
        ax.add_patch(rect)

        # Annotate CIF rank
        cif_rank = row["cif_rank"]
        if not np.isnan(cif_rank):
            rank_int = int(cif_rank)
            # White text on dark backgrounds, dark on light
            text_color = "white" if winner in ("cif", "xgb", "cat", "boruta", "ptest_rdc") else "#1F2937"
            ax.text(
                k_idx, ds_idx, str(rank_int),
                ha="center", va="center",
                fontsize=7, fontweight="bold" if rank_int <= 3 else "normal",
                color=text_color,
            )

    ax.set_xlim(-0.5, n_k - 0.5)
    ax.set_ylim(-0.5, n_ds - 0.5)
    ax.invert_yaxis()

    ax.set_xticks(range(n_k))
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("Feature budget $k$", fontsize=9)

    ax.set_yticks(range(n_ds))
    ax.set_yticklabels(datasets, fontsize=7)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    return used_methods


def _build_heatmap_figure(
    winners_lr: pd.DataFrame,
    winners_svm: pd.DataFrame,
    winners_knn: pd.DataFrame,
) -> plt.Figure:
    """Three-panel heatmap figure (one per downstream model)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 10), sharey=True)

    all_used: set[str] = set()
    for ax, wdf, dm_label in zip(
        axes,
        [winners_lr, winners_svm, winners_knn],
        ["LR downstream", "SVM downstream", "KNN downstream"],
    ):
        used = _plot_heatmap(wdf, ax, dm_label)
        all_used |= used

    # Only first panel gets y-axis labels
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])

    # Legend: one patch per winning method, sorted by frequency
    all_winners = pd.concat([winners_lr, winners_svm, winners_knn])
    win_counts = all_winners["winner"].value_counts()
    legend_order = [m for m in win_counts.index if m in all_used]

    patches = [
        mpatches.Patch(
            facecolor=METHOD_COLORS.get(m, "#D1D5DB"),
            edgecolor="white",
            label=f"{DISPLAY_NAMES.get(m, m)} ({win_counts[m]})",
        )
        for m in legend_order
    ]

    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=min(len(patches), 8),
        fontsize=7.5,
        framealpha=0.9,
        edgecolor="#E5E7EB",
        fancybox=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Winning method per (dataset, $k$) -- cell number = CIF rank",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.99])
    return fig


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------
def _print_frequency_table(freq: pd.DataFrame, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Win frequency: {label}")
    print(f"{'='*60}")
    for _, row in freq.iterrows():
        bar = "#" * int(row["pct"] / 2)
        print(f"  {DISPLAY_NAMES.get(row['method_base'], row['method_base']):>12s}  "
              f"{row['wins']:3d} wins ({row['pct']:5.1f}%)  {bar}")
    total = freq["wins"].sum()
    print(f"  {'TOTAL':>12s}  {total:3d}")


def _print_profiles(profiles: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print("  Dataset-level profiles (LR downstream)")
    print(f"{'='*60}")
    n_stable = (profiles["stability"] == "stable").sum()
    n_contested = (profiles["stability"] == "contested").sum()
    print(f"  Stable winner (same at all k):  {n_stable}")
    print(f"  Contested (winner changes):     {n_contested}")

    print(f"\n  {'Dataset':<16s} ", end="")
    for k in K_VALUES:
        print(f"{'k='+str(k):>10s}", end="")
    print(f"  {'Status':>10s}")
    print("  " + "-" * 78)

    for ds in profiles.index:
        row = profiles.loc[ds]
        print(f"  {ds:<16s} ", end="")
        for k in K_VALUES:
            name = DISPLAY_NAMES.get(row[k], row[k]) if pd.notna(row[k]) else "---"
            print(f"{name:>10s}", end="")
        print(f"  {row['stability']:>10s}")

    # Contested datasets: CIF wins at max k?
    contested = profiles[profiles["stability"] == "contested"]
    if not contested.empty:
        cif_max_k = contested["cif_wins_at_max_k"].sum()
        print(f"\n  On {len(contested)} contested datasets, CIF wins at k=100 on {cif_max_k}")


def _print_catchup(catchup: dict) -> None:
    print(f"\n{'='*60}")
    print("  CIF catchup rate (k=5 -> k=100)")
    print(f"{'='*60}")
    print(f"  Valid datasets:  {catchup['valid_datasets']}")
    print(f"  Rank improved:   {catchup['improved']}")
    print(f"  Rank worsened:   {catchup['worsened']}")
    print(f"  Rank unchanged:  {catchup['unchanged']}")
    print(f"  Catchup rate:    {catchup['catchup_rate']:.1%}")


def _print_turnover_summary(
    freq_lr: pd.DataFrame,
    freq_svm: pd.DataFrame,
    freq_knn: pd.DataFrame,
) -> None:
    """Cross-downstream comparison: does the same method dominate everywhere?"""
    print(f"\n{'='*60}")
    print("  Cross-downstream dominance check")
    print(f"{'='*60}")
    for label, freq in [("LR", freq_lr), ("SVM", freq_svm), ("KNN", freq_knn)]:
        top = freq.iloc[0]
        second = freq.iloc[1] if len(freq) > 1 else None
        pct = top["pct"]
        method = DISPLAY_NAMES.get(top["method_base"], top["method_base"])
        print(f"  {label:>4s} top winner: {method} ({pct:.1f}%)", end="")
        if second is not None:
            m2 = DISPLAY_NAMES.get(second["method_base"], second["method_base"])
            print(f"  |  runner-up: {m2} ({second['pct']:.1f}%)")
        else:
            print()

    # Check if same method is top across all three
    tops = {
        freq_lr.iloc[0]["method_base"],
        freq_svm.iloc[0]["method_base"],
        freq_knn.iloc[0]["method_base"],
    }
    if len(tops) == 1:
        m = tops.pop()
        print(f"\n  => SAME method ({DISPLAY_NAMES.get(m, m)}) dominates across all downstreams")
    else:
        print(f"\n  => HIGH TURNOVER: different top winners across downstreams ({', '.join(DISPLAY_NAMES.get(m, m) for m in sorted(tops))})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _setup_style()
    FIGURES.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and compute winners for each downstream model
    # ------------------------------------------------------------------
    print("Loading data and computing winners...")

    results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for dm in DOWNSTREAM_MODELS:
        means = _load_and_prepare(dm)
        winners = _compute_winners(means)
        freq = _frequency_table(winners)
        results[dm] = (winners, freq)

    winners_lr, freq_lr = results["lr"]
    winners_svm, freq_svm = results["svm"]
    winners_knn, freq_knn = results["knn"]

    # ------------------------------------------------------------------
    # 2. Print frequency tables
    # ------------------------------------------------------------------
    _print_frequency_table(freq_lr, "LR downstream")
    _print_frequency_table(freq_svm, "SVM downstream")
    _print_frequency_table(freq_knn, "KNN downstream")

    # ------------------------------------------------------------------
    # 3. Cross-downstream dominance
    # ------------------------------------------------------------------
    _print_turnover_summary(freq_lr, freq_svm, freq_knn)

    # ------------------------------------------------------------------
    # 4. Dataset-level profiles (LR)
    # ------------------------------------------------------------------
    profiles = _dataset_profiles(winners_lr)
    _print_profiles(profiles)

    # ------------------------------------------------------------------
    # 5. CIF catchup rate (LR)
    # ------------------------------------------------------------------
    catchup = _cif_catchup_rate(winners_lr)
    _print_catchup(catchup)

    # ------------------------------------------------------------------
    # 6. Heatmap figure
    # ------------------------------------------------------------------
    print("\nGenerating heatmap...")
    fig = _build_heatmap_figure(winners_lr, winners_svm, winners_knn)
    out = FIGURES / "dataset_winners_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")

    # ------------------------------------------------------------------
    # 7. Detailed CIF rank trajectory per dataset (LR)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  CIF rank by dataset and k (LR downstream)")
    print(f"{'='*60}")
    print(f"  {'Dataset':<16s}", end="")
    for k in K_VALUES:
        print(f"  {'k='+str(k):>6s}", end="")
    print(f"  {'trend':>8s}")
    print("  " + "-" * 64)

    for ds in sorted(winners_lr["dataset"].unique()):
        sub = winners_lr[winners_lr["dataset"] == ds].sort_values("k")
        print(f"  {ds:<16s}", end="")
        ranks = []
        for k in K_VALUES:
            row = sub[sub["k"] == k]
            if row.empty or np.isnan(row.iloc[0]["cif_rank"]):
                print(f"  {'---':>6s}", end="")
            else:
                r = int(row.iloc[0]["cif_rank"])
                ranks.append(r)
                print(f"  {r:>6d}", end="")

        # Trend arrow
        if len(ranks) >= 2:
            if ranks[-1] < ranks[0]:
                print(f"  {'UP':>8s}", end="")
            elif ranks[-1] > ranks[0]:
                print(f"  {'DOWN':>8s}", end="")
            else:
                print(f"  {'FLAT':>8s}", end="")
        print()


if __name__ == "__main__":
    main()
