"""Exploratory effective-rank scatter: spectral complexity vs CIF gap-to-best.

Archived exploratory figure; not part of the current paper-facing rebuild path.

Produces a multi-panel scatter plot showing how the effective rank of the
standardized feature matrix (sum(sigma)/max(sigma) from SVD) relates to the
gap between CIF balanced accuracy and the best competing method across all
k values (5, 10, 25, 50, 100).

This is an exploratory diagnostic rather than a manuscript-facing benchmark
summary. It uses LR only and a per-dataset/per-k oracle competitor, so it does
not follow the task-wide best-config contract in `benchmark_common.py`.

Figure: paper/results/figures/exploratory_effective_rank_scatter.png

Usage:
    UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/backlog/analysis/fig_exploratory_effective_rank.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS = Path("paper/results")
FIGURES = RESULTS / "figures"
DATA_DIR = Path("paper/data/classification/real")

# ---------------------------------------------------------------------------
# Dataset categories
# ---------------------------------------------------------------------------
FACE_IMAGE = {"Yale", "ORL", "warpAR10P", "warpPIE10P", "pixraw10P", "orlraws10P"}
GENOMIC_BIO = {"ALLAML", "CLL_SUB_111", "TOX_171", "arcene"}
TEXT_SPARSE = {"dexter", "gisette", "madelon"}
# Everything else -> "Tabular"

CATEGORY_COLORS = {
    "Face / image": "#2563EB",
    "Genomic / bio": "#16A34A",
    "Text / sparse": "#DC2626",
    "Tabular": "#9CA3AF",
}

OUTLIER_LABELS = {"dexter", "gisette", "wine", "gamma"}


def _category(ds: str) -> str:
    if ds in FACE_IMAGE:
        return "Face / image"
    if ds in GENOMIC_BIO:
        return "Genomic / bio"
    if ds in TEXT_SPARSE:
        return "Text / sparse"
    return "Tabular"


# ---------------------------------------------------------------------------
# 1. Compute effective rank per dataset
# ---------------------------------------------------------------------------
def _compute_effective_ranks() -> pd.DataFrame:
    """Load each real CLF dataset, standardize, SVD, return effective rank."""
    rows: list[dict] = []
    for path in sorted(DATA_DIR.glob("clf_*.parquet")):
        ds_name = path.stem.removeprefix("clf_")
        df = pd.read_parquet(path)
        X = df.drop(columns=["y"]).values.astype(np.float64)
        n, p = X.shape

        # Standardize: zero-mean, unit-variance per column
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma == 0] = 1.0  # constant columns -> avoid div-by-zero
        X_std = (X - mu) / sigma

        # SVD (economy)
        _, s, _ = np.linalg.svd(X_std, full_matrices=False)

        # Effective rank = sum(sigma_i) / max(sigma_i)
        eff_rank = s.sum() / s.max()

        rows.append({
            "dataset": ds_name,
            "n": n,
            "p": p,
            "effective_rank": eff_rank,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Compute CIF gap-to-best per dataset across all k values
# ---------------------------------------------------------------------------
def _compute_cif_gap() -> pd.DataFrame:
    """LR-only oracle gap: best BA minus CIF BA at each k for each dataset."""
    df = pd.read_parquet(RESULTS / "clf_evaluation.parquet")

    # Filter: real datasets, LR downstream, all k values
    k_values = [5, 10, 25, 50, 100]
    sub = df[
        (df["dataset_type"] == "real")
        & (df["downstream_model"] == "lr")
        & (df["k"].isin(k_values))
    ].copy()

    rows: list[dict] = []

    for k in k_values:
        k_sub = sub[sub["k"] == k].copy()

        # Mean BA over seeds x folds per (method_id, dataset)
        mean_ba = (
            k_sub.groupby(["method_id", "method_base", "dataset"])["balanced_accuracy"]
            .mean()
            .reset_index()
        )

        # Best config per method_base per dataset
        best_config = (
            mean_ba.sort_values("balanced_accuracy", ascending=False)
            .groupby(["method_base", "dataset"])
            .first()
            .reset_index()
        )

        # Pivot: rows = dataset, columns = method_base, values = BA
        pivot = best_config.pivot(index="dataset", columns="method_base", values="balanced_accuracy")

        # CIF BA
        if "cif" not in pivot.columns:
            raise ValueError(f"CIF not found in evaluation data at k={k}")

        cif_ba = pivot["cif"]

        # Best-method BA per dataset (across ALL methods)
        best_ba = pivot.max(axis=1)

        # Compute gap for this k
        for dataset in pivot.index:
            rows.append({
                "dataset": dataset,
                "k": k,
                "cif_ba": cif_ba[dataset],
                "best_ba": best_ba[dataset],
                "gap": best_ba[dataset] - cif_ba[dataset],
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Plot
# ---------------------------------------------------------------------------
def main() -> None:
    print("Computing exploratory effective-rank diagnostic from raw datasets...")
    eff_df = _compute_effective_ranks()
    print(f"  {len(eff_df)} datasets with effective rank")

    print("Computing LR-only oracle CIF gap-to-best across all k values...")
    gap_df = _compute_cif_gap()
    print(f"  {len(gap_df)} dataset-k pairs with CIF gap")

    # Merge — keep only datasets present in BOTH
    plot_df = eff_df.merge(gap_df, on="dataset", how="inner")
    n_datasets = plot_df["dataset"].nunique()
    print(f"  {n_datasets} datasets in both sources")

    # Add category
    plot_df["category"] = plot_df["dataset"].apply(_category)

    # Log-transform effective rank for x-axis
    plot_df["log_eff_rank"] = np.log10(plot_df["effective_rank"])

    # Point size proportional to log(p)
    plot_df["size"] = np.log(plot_df["p"]) * 15  # scale factor for visibility

    # -----------------------------------------------------------------------
    # Spearman correlation per k
    # -----------------------------------------------------------------------
    k_values = sorted(plot_df["k"].unique())
    corr_results = []

    print("\nEffective rank vs CIF gap-to-best:")
    for k in k_values:
        k_sub = plot_df[plot_df["k"] == k]
        rho, pval = sp_stats.spearmanr(k_sub["log_eff_rank"], k_sub["gap"])
        n = len(k_sub)
        corr_results.append({"k": k, "rho": rho, "pval": pval, "n": n})
        print(f"  k={k:3d}:  rho={rho:5.2f}, p={pval:.4f}, n={n} datasets")

    corr_df = pd.DataFrame(corr_results)

    # -----------------------------------------------------------------------
    # Figure: Multi-panel scatter (one per k)
    # -----------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "figure.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    # Create 2x3 grid (5 panels + 1 for the correlation trajectory)
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Plot each k in a separate panel
    for i, k in enumerate(k_values):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        k_sub = plot_df[plot_df["k"] == k]

        # Scatter by category
        for cat, color in CATEGORY_COLORS.items():
            mask = k_sub["category"] == cat
            if not mask.any():
                continue
            sub = k_sub[mask]
            label = cat if i == 0 else None  # Only show legend in first panel
            ax.scatter(
                sub["log_eff_rank"],
                sub["gap"],
                s=sub["size"],
                c=color,
                alpha=0.85,
                edgecolors="white",
                linewidth=0.6,
                label=label,
                zorder=3,
            )

        # OLS regression line
        slope, intercept, _, _, _ = sp_stats.linregress(k_sub["log_eff_rank"], k_sub["gap"])
        x_line = np.linspace(k_sub["log_eff_rank"].min() - 0.05, k_sub["log_eff_rank"].max() + 0.05, 100)
        ax.plot(x_line, intercept + slope * x_line, "--", color="#4B5563", linewidth=1.2, alpha=0.7, zorder=2)

        # Zero line
        ax.axhline(0, color="#9CA3AF", linewidth=0.6, linestyle=":", zorder=1)

        # Annotate outlier points (only for k=10 to reduce clutter)
        if k == 10:
            for _, row in k_sub.iterrows():
                if row["dataset"] in OUTLIER_LABELS:
                    ax.annotate(
                        row["dataset"],
                        (row["log_eff_rank"], row["gap"]),
                        fontsize=6,
                        alpha=0.85,
                        ha="left",
                        va="bottom",
                        xytext=(3, 3),
                        textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color="#6B7280", linewidth=0.3, alpha=0.5),
                    )

        # Correlation annotation
        rho = corr_df[corr_df["k"] == k]["rho"].iloc[0]
        pval = corr_df[corr_df["k"] == k]["pval"].iloc[0]
        ax.text(
            0.05,
            0.95,
            f"$\\rho$ = {rho:.2f}, $p$ = {pval:.4f}",
            transform=ax.transAxes,
            fontsize=7.5,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#D1D5DB", alpha=0.9),
        )

        # Labels
        if i >= 3:  # Bottom row
            ax.set_xlabel(r"$\log_{10}$(effective rank)", fontsize=9)
        if i % 3 == 0:  # Left column
            ax.set_ylabel("LR oracle gap to best (BA)", fontsize=9)

        # Title
        ax.set_title(f"k = {k}", fontsize=10, fontweight="bold")

        # Legend (only first panel)
        if i == 0:
            ax.legend(
                loc="upper right",
                fontsize=6.5,
                framealpha=0.9,
                edgecolor="#D1D5DB",
                markerscale=0.7,
            )

    # -----------------------------------------------------------------------
    # 6th panel: Correlation trajectory across k
    # -----------------------------------------------------------------------
    ax_traj = fig.add_subplot(gs[1, 2])

    # Plot rho vs k
    ax_traj.plot(corr_df["k"], corr_df["rho"], "-o", color="#2563EB", linewidth=2, markersize=6, zorder=3)
    ax_traj.axhline(0, color="#9CA3AF", linewidth=0.8, linestyle=":", zorder=1)

    # Significance thresholds
    ax_traj.axhline(0.5, color="#16A34A", linewidth=0.8, linestyle="--", alpha=0.5, label=r"$\rho$ = 0.5")
    ax_traj.axhline(-0.5, color="#DC2626", linewidth=0.8, linestyle="--", alpha=0.5)

    # Labels
    ax_traj.set_xlabel("Number of features selected (k)", fontsize=9)
    ax_traj.set_ylabel(r"Spearman $\rho$", fontsize=9)
    ax_traj.set_title("Exploratory correlation trajectory", fontsize=10, fontweight="bold")
    ax_traj.set_xticks(k_values)
    ax_traj.grid(True, alpha=0.3)

    # Annotate each point with rho value
    for _, row in corr_df.iterrows():
        ax_traj.annotate(
            f"{row['rho']:.2f}",
            (row["k"], row["rho"]),
            fontsize=7,
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )

    # Save
    FIGURES.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES / "exploratory_effective_rank_scatter.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Saved -> {out_path}")
    print(f"  Total datasets: {n_datasets}")
    print(f"  k values: {k_values}")


if __name__ == "__main__":
    main()
