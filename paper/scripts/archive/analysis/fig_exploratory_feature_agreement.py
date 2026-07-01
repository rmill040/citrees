"""Exploratory cross-method feature agreement analysis.

Archived exploratory figure; not part of the current paper-facing rebuild path.

For each pair of 15 feature selection methods, compute Jaccard similarity
between their "consensus top-10" feature sets, averaged across the current
real classification benchmark datasets. The consensus top-10 for a method on a
dataset is the 10 features that appear most frequently in the top-10 across
the 25 seed x fold ranking vectors.

This is an exploratory diagnostic. It uses a classification-only, LR-selected
best-config slice rather than the manuscript's all-downstream benchmark
contract.

Pipeline:
  1. Load clf_evaluation.parquet to select the best config per method_base
     (highest mean balanced_accuracy under LR on real datasets).
  2. Load clf_rankings.parquet, filter to real datasets + best configs.
  3. For each (method, dataset): compute consensus top-10.
  4. For each (method_A, method_B, dataset): Jaccard(top10_A, top10_B).
  5. Average Jaccard across datasets -> 15x15 agreement matrix.
  6. Hierarchically cluster the matrix and render a heatmap.
  7. Print within-family agreement, cross-family disagreement, CIF affinity,
     and agreement-vs-accuracy correlation.

Outputs:
  paper/results/figures/exploratory_feature_agreement_matrix.png

Usage:
    uv run python paper/scripts/archive/analysis/fig_exploratory_feature_agreement.py
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS = Path("paper/results")
FIGURES = RESULTS / "figures"
OUT_PATH = FIGURES / "exploratory_feature_agreement_matrix.png"

# ---------------------------------------------------------------------------
# Display names and categories (consistent with other figure scripts)
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

CONSENSUS_K = 10


def _dn(m: str) -> str:
    return DISPLAY_NAMES.get(m, m)


def _method_to_category(m: str) -> str:
    for cat, members in CATEGORIES.items():
        if m in members:
            return cat
    return "Other"


# ---------------------------------------------------------------------------
# 1. Best config selection (same logic as other scripts)
# ---------------------------------------------------------------------------
def _select_best_configs(df_eval: pd.DataFrame) -> dict[str, str]:
    """Return {method_base: best_method_id} using highest mean BA under LR on real data."""
    sub = df_eval[(df_eval["dataset_source"] == "real") & (df_eval["downstream_model"] == "lr")]
    mean_ba = sub.groupby(["method_base", "method_id"])["balanced_accuracy"].mean().reset_index()
    best = (
        mean_ba.sort_values("balanced_accuracy", ascending=False)
        .groupby("method_base")["method_id"]
        .first()
        .to_dict()
    )
    return best


# ---------------------------------------------------------------------------
# 2. Consensus top-k computation
# ---------------------------------------------------------------------------
def _consensus_top_k(
    rankings: list[np.ndarray],
    k: int,
) -> set[int]:
    """Compute consensus top-k: the k features appearing most often in top-k across runs.

    Parameters
    ----------
    rankings : list of arrays
        Each array is a full feature ranking (ordered indices, best first).
    k : int
        Number of top features to consider.

    Returns
    -------
    set of int
        The k most frequently selected features across runs.
    """
    counter: Counter[int] = Counter()
    for r in rankings:
        for feat in r[:k]:
            counter[feat] += 1
    # Take the k features with highest frequency
    return {feat for feat, _ in counter.most_common(k)}


# ---------------------------------------------------------------------------
# 3. Jaccard similarity
# ---------------------------------------------------------------------------
def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# 4. Build the agreement matrix
# ---------------------------------------------------------------------------
def _build_agreement_matrix(
    df_rank: pd.DataFrame,
    methods: list[str],
    k: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, set[int]]]]:
    """Build the pairwise Jaccard agreement matrix averaged across datasets.

    Returns
    -------
    matrix : pd.DataFrame
        Square matrix (methods x methods) of mean Jaccard similarities.
    consensus : dict
        Nested dict: consensus[method_base][dataset] = set of top-k features.
    """
    # Pre-compute consensus top-k for every (method, dataset) pair
    consensus: dict[str, dict[str, set[int]]] = {}
    for mb in methods:
        consensus[mb] = {}
        mb_data = df_rank[df_rank["method_base"] == mb]
        for ds, ds_grp in mb_data.groupby("dataset"):
            rankings = ds_grp["feature_ranking"].values.tolist()
            if len(rankings) < 2:
                continue
            consensus[mb][str(ds)] = _consensus_top_k(rankings, k)

    # Compute pairwise Jaccard averaged across common datasets
    n = len(methods)
    matrix = np.eye(n)  # diagonal = 1.0

    for i, j in combinations(range(n), 2):
        m_a, m_b = methods[i], methods[j]
        ds_a = set(consensus[m_a].keys())
        ds_b = set(consensus[m_b].keys())
        common_ds = ds_a & ds_b
        if not common_ds:
            continue
        jaccards = [_jaccard(consensus[m_a][ds], consensus[m_b][ds]) for ds in common_ds]
        mean_j = float(np.mean(jaccards))
        matrix[i, j] = mean_j
        matrix[j, i] = mean_j

    return pd.DataFrame(matrix, index=methods, columns=methods), consensus


# ---------------------------------------------------------------------------
# 5. Hierarchical clustering for row/column ordering
# ---------------------------------------------------------------------------
def _cluster_order(matrix: pd.DataFrame) -> list[str]:
    """Return methods ordered by hierarchical clustering (average linkage)."""
    # Convert similarity to distance
    dist = 1.0 - matrix.values
    np.fill_diagonal(dist, 0.0)
    # Make symmetric (in case of floating point)
    dist = (dist + dist.T) / 2.0
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    dn = dendrogram(Z, no_plot=True)
    order = [matrix.index[i] for i in dn["leaves"]]
    return order


# ---------------------------------------------------------------------------
# 6. Analysis: within-family, cross-family, CIF affinity, accuracy corr.
# ---------------------------------------------------------------------------
def _analyze_families(matrix: pd.DataFrame) -> dict[str, float]:
    """Compute mean within-family Jaccard for each family."""
    results: dict[str, float] = {}
    for family, members in CATEGORIES.items():
        present = [m for m in members if m in matrix.index]
        if len(present) < 2:
            results[family] = float("nan")
            continue
        vals = []
        for a, b in combinations(present, 2):
            vals.append(matrix.loc[a, b])
        results[family] = float(np.mean(vals))
    return results


def _cross_family_agreement(matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute mean agreement between each pair of families."""
    families = list(CATEGORIES.keys())
    n_fam = len(families)
    cross = np.full((n_fam, n_fam), np.nan)

    for i, fam_a in enumerate(families):
        members_a = [m for m in CATEGORIES[fam_a] if m in matrix.index]
        for j, fam_b in enumerate(families):
            members_b = [m for m in CATEGORIES[fam_b] if m in matrix.index]
            if not members_a or not members_b:
                continue
            if i == j:
                # Within-family
                if len(members_a) < 2:
                    cross[i, j] = float("nan")
                else:
                    vals = [matrix.loc[a, b] for a, b in combinations(members_a, 2)]
                    cross[i, j] = float(np.mean(vals))
            else:
                vals = [matrix.loc[a, b] for a in members_a for b in members_b]
                cross[i, j] = float(np.mean(vals))

    return pd.DataFrame(cross, index=families, columns=families)


def _cif_affinity(matrix: pd.DataFrame) -> dict[str, float]:
    """How much does CIF agree with each other family (mean Jaccard)?"""
    if "cif" not in matrix.index:
        return {}
    results: dict[str, float] = {}
    for family, members in CATEGORIES.items():
        # Exclude CIF itself from "Forest"
        others = [m for m in members if m in matrix.index and m != "cif"]
        if not others:
            continue
        vals = [matrix.loc["cif", m] for m in others]
        results[family] = float(np.mean(vals))
    return results


def _agreement_accuracy_correlation(
    matrix: pd.DataFrame,
    df_eval: pd.DataFrame,
    best_ids: dict[str, str],
) -> tuple[float, float]:
    """Spearman correlation between pairwise Jaccard and pairwise accuracy delta.

    For each pair of methods, compute:
      - Mean Jaccard (feature agreement)
      - Mean absolute accuracy difference across datasets (aggregated across all k)
    Then correlate across pairs.
    """
    from scipy import stats as sp_stats

    # Build per-method mean accuracy (real, LR, aggregated across all k values)
    sub = df_eval[(df_eval["dataset_source"] == "real") & (df_eval["downstream_model"] == "lr")]
    # Filter to best config
    sub = sub[sub.apply(lambda r: r["method_id"] == best_ids.get(r["method_base"]), axis=1)]

    # Dataset-level means (average across all k values)
    ds_means = sub.groupby(["method_base", "dataset"])["balanced_accuracy"].mean().reset_index()

    methods = list(matrix.index)
    jac_vals = []
    acc_deltas = []

    for m_a, m_b in combinations(methods, 2):
        jac_vals.append(matrix.loc[m_a, m_b])
        # Paired accuracy comparison
        a_acc = ds_means[ds_means["method_base"] == m_a].set_index("dataset")["balanced_accuracy"]
        b_acc = ds_means[ds_means["method_base"] == m_b].set_index("dataset")["balanced_accuracy"]
        common = a_acc.index.intersection(b_acc.index)
        if len(common) == 0:
            acc_deltas.append(float("nan"))
        else:
            acc_deltas.append(float(np.mean(np.abs(a_acc.loc[common] - b_acc.loc[common]))))

    # Drop nan pairs
    pairs = [(j, a) for j, a in zip(jac_vals, acc_deltas, strict=False) if not np.isnan(a)]
    if len(pairs) < 3:
        return float("nan"), float("nan")

    jj, aa = zip(*pairs, strict=False)
    rho, pval = sp_stats.spearmanr(jj, aa)
    return float(rho), float(pval)


# ---------------------------------------------------------------------------
# 7. Heatmap figure
# ---------------------------------------------------------------------------
def _make_figure(
    matrix: pd.DataFrame,
    order: list[str],
    n_datasets: int,
) -> None:
    """Render and save the clustered Jaccard agreement heatmap."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )

    # Reorder matrix and mask diagonal for better color contrast
    mat = matrix.loc[order, order].copy()
    display_labels = [_dn(m) for m in order]
    n = len(order)

    # Off-diagonal range for color scale
    off_diag_vals = mat.values[~np.eye(n, dtype=bool)]
    vmin = max(0.0, float(off_diag_vals.min()) - 0.02)
    vmax = min(1.0, float(off_diag_vals.max()) + 0.02)

    fig, ax = plt.subplots(figsize=(9, 7.5))

    # Color-code tick labels by category
    cat_colors = []
    for m in order:
        cat = _method_to_category(m)
        cat_colors.append(CATEGORY_COLORS.get(cat, "#374151"))

    # Plot off-diagonal with tight color range for contrast
    plot_data = mat.values.copy()
    np.fill_diagonal(plot_data, np.nan)  # mask diagonal

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#E5E7EB")  # light gray for diagonal

    im = ax.imshow(
        plot_data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = mat.values[i, j]
            if i == j:
                # Diagonal: gray text on gray background
                ax.text(
                    j,
                    i,
                    "1.00",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="#6B7280",
                    fontstyle="italic",
                )
            else:
                text_color = "white" if val > (vmin + 0.6 * (vmax - vmin)) else "black"
                ax.text(
                    j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=text_color
                )

    # Ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(display_labels, fontsize=8)

    # Color tick labels by category
    for idx, (xtl, ytl) in enumerate(zip(ax.get_xticklabels(), ax.get_yticklabels(), strict=False)):
        xtl.set_color(cat_colors[idx])
        ytl.set_color(cat_colors[idx])
        if order[idx] == "cif":
            xtl.set_fontweight("bold")
            ytl.set_fontweight("bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Mean Jaccard similarity (consensus top-10)", fontsize=9)

    # Category legend as a single line below the heatmap
    legend_parts = []
    for cat, color in CATEGORY_COLORS.items():
        legend_parts.append((cat, color))

    total_cats = len(legend_parts)
    spacing = 1.0 / total_cats
    for i, (cat, color) in enumerate(legend_parts):
        ax.annotate(
            cat,
            xy=(spacing * i + spacing / 2, -0.14),
            xycoords="axes fraction",
            fontsize=8,
            fontweight="bold",
            color=color,
            ha="center",
            va="top",
        )

    ax.set_title(
        "Exploratory cross-method feature agreement\n"
        f"(Jaccard similarity of consensus top-10, LR-selected best configs, {n_datasets} datasets)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved -> {OUT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("EXPLORATORY CROSS-METHOD FEATURE AGREEMENT ANALYSIS")
    print("=" * 70)
    print("This diagnostic uses classification-only, LR-selected best configs.")

    # ------------------------------------------------------------------
    # 1. Select best config per method_base
    # ------------------------------------------------------------------
    print("\n[1/6] Loading evaluation data to select best configs...")
    df_eval = pd.read_parquet(RESULTS / "clf_evaluation.parquet")
    best_ids = _select_best_configs(df_eval)
    print(f"  {len(best_ids)} methods with best configs selected")

    # ------------------------------------------------------------------
    # 2. Load rankings and filter
    # ------------------------------------------------------------------
    print("[2/6] Loading rankings data (real datasets, best configs)...")
    df_rank = pd.read_parquet(RESULTS / "clf_rankings.parquet")
    df_rank = df_rank[df_rank["dataset_source"] == "real"]

    # Filter to best config per method_base
    mask = df_rank.apply(
        lambda r: r["method_id"] == best_ids.get(r["method_base"]),
        axis=1,
    )
    df_rank = df_rank[mask].copy()

    methods = sorted(df_rank["method_base"].unique())
    n_datasets = df_rank["dataset"].nunique()
    print(f"  {len(methods)} methods, {n_datasets} datasets, {len(df_rank)} ranking vectors")

    # ------------------------------------------------------------------
    # 3. Build agreement matrix
    # ------------------------------------------------------------------
    print(f"[3/6] Computing consensus top-{CONSENSUS_K} and pairwise Jaccard...")
    matrix, consensus = _build_agreement_matrix(df_rank, methods, k=CONSENSUS_K)

    # Print raw matrix summary
    off_diag = matrix.values[np.triu_indices_from(matrix.values, k=1)]
    print(f"  Overall mean Jaccard (off-diagonal): {off_diag.mean():.3f}")
    print(f"  Min: {off_diag.min():.3f}, Max: {off_diag.max():.3f}, Std: {off_diag.std():.3f}")

    # ------------------------------------------------------------------
    # 4. Cluster and order
    # ------------------------------------------------------------------
    print("[4/6] Hierarchical clustering...")
    order = _cluster_order(matrix)
    print(f"  Cluster order: {' -> '.join(_dn(m) for m in order)}")

    # ------------------------------------------------------------------
    # 5. Analysis
    # ------------------------------------------------------------------
    print("[5/6] Computing analyses...\n")

    # 5a. Within-family agreement
    within = _analyze_families(matrix)
    print("  (a) Within-family mean Jaccard:")
    for fam in sorted(
        within, key=lambda f: within[f] if not np.isnan(within[f]) else -1, reverse=True
    ):
        val = within[fam]
        members = [_dn(m) for m in CATEGORIES[fam] if m in matrix.index]
        if np.isnan(val):
            print(f"      {fam:14s}: N/A (only {len(members)} member)")
        else:
            print(f"      {fam:14s}: {val:.3f}  ({', '.join(members)})")

    highest_fam = max(
        (f for f in within if not np.isnan(within[f])),
        key=lambda f: within[f],
    )
    print(f"\n      -> Highest within-family agreement: {highest_fam} ({within[highest_fam]:.3f})")

    # 5b. Cross-family disagreement
    cross = _cross_family_agreement(matrix)
    print("\n  (b) Cross-family agreement matrix:")
    # Pretty print
    fam_names = list(cross.index)
    header = f"{'':14s}" + "".join(f"{f:>12s}" for f in fam_names)
    print(f"      {header}")
    for fam in fam_names:
        row_str = f"      {fam:14s}"
        for fam2 in fam_names:
            val = cross.loc[fam, fam2]
            if np.isnan(val):
                row_str += f"{'N/A':>12s}"
            else:
                row_str += f"{val:>12.3f}"
        print(row_str)

    # Find most and least agreeing family pairs
    cross_pairs: list[tuple[str, str, float]] = []
    for i, f1 in enumerate(fam_names):
        for j, f2 in enumerate(fam_names):
            if i < j and not np.isnan(cross.loc[f1, f2]):
                cross_pairs.append((f1, f2, cross.loc[f1, f2]))
    if cross_pairs:
        cross_pairs.sort(key=lambda x: x[2])
        f1, f2, v = cross_pairs[0]
        print(f"\n      -> Most disagreement: {f1} vs {f2} ({v:.3f})")
        f1, f2, v = cross_pairs[-1]
        print(f"      -> Most agreement:    {f1} vs {f2} ({v:.3f})")

    # 5c. CIF affinity
    cif_aff = _cif_affinity(matrix)
    print("\n  (c) CIF affinity to each family (mean Jaccard):")
    for fam in sorted(cif_aff, key=lambda f: cif_aff[f], reverse=True):
        members = [_dn(m) for m in CATEGORIES[fam] if m in matrix.index and m != "cif"]
        print(f"      {fam:14s}: {cif_aff[fam]:.3f}  (vs {', '.join(members)})")

    if cif_aff:
        closest = max(cif_aff, key=lambda f: cif_aff[f])
        furthest = min(cif_aff, key=lambda f: cif_aff[f])
        print(f"\n      -> CIF agrees most with: {closest} ({cif_aff[closest]:.3f})")
        print(f"      -> CIF agrees least with: {furthest} ({cif_aff[furthest]:.3f})")

    # 5d. Agreement vs accuracy correlation
    rho, pval = _agreement_accuracy_correlation(matrix, df_eval, best_ids)
    print("\n  (d) Agreement vs accuracy correlation:")
    print(f"      Spearman rho = {rho:.3f}, p = {pval:.3f}")
    if pval < 0.05:
        if rho > 0:
            print("      -> Significant: higher feature agreement is associated with")
            print("         larger downstream accuracy gaps.")
        else:
            print("      -> Significant: higher feature agreement is associated with")
            print("         smaller downstream accuracy gaps.")
    else:
        print("      -> Not significant: feature agreement does not predict accuracy similarity.")

    # 5e. Per-method summary: most and least similar partner
    print("\n  (e) Per-method closest and furthest partner:")
    for m in order:
        row = matrix.loc[m].drop(m)
        closest = row.idxmax()
        furthest = row.idxmin()
        print(
            f"      {_dn(m):12s}  closest={_dn(closest):12s} ({row[closest]:.3f})  "
            f"furthest={_dn(furthest):12s} ({row[furthest]:.3f})"
        )

    # ------------------------------------------------------------------
    # 6. Generate figure
    # ------------------------------------------------------------------
    print("\n[6/6] Generating heatmap figure...")
    _make_figure(matrix, order, n_datasets=n_datasets)
    print("\nDone.")


if __name__ == "__main__":
    main()
