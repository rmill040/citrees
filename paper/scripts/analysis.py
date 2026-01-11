"""Comprehensive analysis for citrees paper.

Statistical tests:
- Friedman test for overall method comparison
- Nemenyi post-hoc test for pairwise comparisons

Visualizations:
- Critical difference diagrams
- Performance heatmaps
- Box plots by experimental factors

Tables:
- Rankings with critical differences
- Summary statistics (LaTeX and CSV)

Usage:
    uv run python scripts/analysis.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"


def friedman_test(data: pd.DataFrame, methods: list[str], metric: str) -> tuple[float, float]:
    """Perform Friedman test across methods.

    Parameters
    ----------
    data : pd.DataFrame
        Results with columns for each method's metric.
    methods : List[str]
        List of method names.
    metric : str
        Metric column suffix (e.g., 'precision@10', 'downstream_acc_mean').

    Returns
    -------
    Tuple[float, float]
        Friedman chi-square statistic and p-value.
    """
    cols = [f"{m}_{metric}" for m in methods]
    available_cols = [c for c in cols if c in data.columns]
    if len(available_cols) < 3:
        return np.nan, np.nan

    # Get values for each method
    values = [data[col].dropna().values for col in available_cols]
    min_len = min(len(v) for v in values)
    values = [v[:min_len] for v in values]

    if min_len < 2:
        return np.nan, np.nan

    stat, pvalue = stats.friedmanchisquare(*values)
    return stat, pvalue


def nemenyi_critical_difference(n_methods: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Compute Nemenyi critical difference.

    CD = q_alpha * sqrt(k(k+1) / (6*n))

    where k = number of methods, n = number of datasets.
    """
    # Critical values for Nemenyi test (q_alpha for two-tailed test)
    # From: https://www.itl.nist.gov/div898/handbook/prc/section4/prc453.htm
    q_values = {
        # (n_methods, alpha): q_value
        (3, 0.05): 2.343,
        (3, 0.10): 2.052,
        (4, 0.05): 2.569,
        (4, 0.10): 2.291,
        (5, 0.05): 2.728,
        (5, 0.10): 2.459,
        (6, 0.05): 2.850,
        (6, 0.10): 2.589,
        (7, 0.05): 2.949,
        (7, 0.10): 2.693,
        (8, 0.05): 3.031,
        (8, 0.10): 2.780,
        (9, 0.05): 3.102,
        (9, 0.10): 2.855,
        (10, 0.05): 3.164,
        (10, 0.10): 2.920,
    }

    q = q_values.get((n_methods, alpha), 2.728)  # Default to 5 methods
    cd = q * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
    return cd


def compute_ranks(
    data: pd.DataFrame, methods: list[str], metric: str, higher_is_better: bool = True
) -> pd.DataFrame:
    """Compute average ranks for each method.

    Parameters
    ----------
    data : pd.DataFrame
        Results dataframe.
    methods : List[str]
        List of method names.
    metric : str
        Metric column suffix.
    higher_is_better : bool
        If True, higher values get better (lower) ranks.

    Returns
    -------
    pd.DataFrame
        DataFrame with method, avg_rank, std_rank columns.
    """
    cols = [f"{m}_{metric}" for m in methods]
    available = [(m, f"{m}_{metric}") for m in methods if f"{m}_{metric}" in data.columns]

    if len(available) < 2:
        return pd.DataFrame()

    methods_available = [m for m, _ in available]
    cols_available = [c for _, c in available]

    # Get values matrix
    values = data[cols_available].values

    # Compute ranks per row (1 = best)
    if higher_is_better:
        ranks = stats.rankdata(-values, axis=1)  # Negative for descending
    else:
        ranks = stats.rankdata(values, axis=1)

    avg_ranks = np.nanmean(ranks, axis=0)
    std_ranks = np.nanstd(ranks, axis=0)

    return pd.DataFrame(
        {
            "method": methods_available,
            "avg_rank": avg_ranks,
            "std_rank": std_ranks,
        }
    ).sort_values("avg_rank")


def pairwise_nemenyi(ranks_df: pd.DataFrame, cd: float) -> pd.DataFrame:
    """Compute pairwise Nemenyi significance matrix.

    Two methods are significantly different if |rank_i - rank_j| > CD.
    """
    methods = ranks_df["method"].tolist()
    avg_ranks = ranks_df.set_index("method")["avg_rank"]

    results = []
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i < j:
                diff = abs(avg_ranks[m1] - avg_ranks[m2])
                significant = diff > cd
                results.append(
                    {
                        "method1": m1,
                        "method2": m2,
                        "rank_diff": diff,
                        "cd": cd,
                        "significant": significant,
                    }
                )

    return pd.DataFrame(results)


def generate_friedman_table(
    data: pd.DataFrame, methods: list[str], metrics: list[str], output_path: Path
) -> None:
    """Generate Friedman test results table."""
    results = []

    for metric in metrics:
        stat, pvalue = friedman_test(data, methods, metric)
        results.append(
            {
                "metric": metric,
                "chi_square": stat,
                "p_value": pvalue,
                "significant": pvalue < 0.05 if not np.isnan(pvalue) else False,
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(output_path.with_suffix(".csv"), index=False)

    # LaTeX version
    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{Friedman test results across methods.}")
    latex.append(r"\label{tab:friedman}")
    latex.append(r"\begin{tabular}{lcc}")
    latex.append(r"\toprule")
    latex.append(r"Metric & $\chi^2$ & p-value \\")
    latex.append(r"\midrule")

    for _, row in df.iterrows():
        sig = "*" if row["significant"] else ""
        pval = f"{row['p_value']:.4f}" if not np.isnan(row["p_value"]) else "N/A"
        chi2 = f"{row['chi_square']:.2f}" if not np.isnan(row["chi_square"]) else "N/A"
        latex.append(f"{row['metric']} & {chi2} & {pval}{sig} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    with open(output_path.with_suffix(".tex"), "w") as f:
        f.write("\n".join(latex))

    print(f"Saved: {output_path.with_suffix('.csv')}")
    print(f"Saved: {output_path.with_suffix('.tex')}")


def generate_ranking_table(
    data: pd.DataFrame,
    methods: list[str],
    metric: str,
    output_path: Path,
    higher_is_better: bool = True,
) -> None:
    """Generate method ranking table with CD test."""
    ranks_df = compute_ranks(data, methods, metric, higher_is_better)

    if ranks_df.empty:
        print(f"No data for metric {metric}")
        return

    # Compute critical difference
    n_methods = len(ranks_df)
    n_datasets = len(data)
    cd = nemenyi_critical_difference(n_methods, n_datasets)

    # Get pairwise comparisons
    pairwise = pairwise_nemenyi(ranks_df, cd)

    # Save ranks
    ranks_df["cd"] = cd
    ranks_df.to_csv(output_path.with_suffix(".csv"), index=False)

    # LaTeX version
    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(f"\\caption{{Method rankings for {metric}. CD = {cd:.3f} at $\\alpha=0.05$.}}")
    latex.append(f"\\label{{tab:ranks_{metric}}}")
    latex.append(r"\begin{tabular}{lcc}")
    latex.append(r"\toprule")
    latex.append(r"Method & Avg Rank & Std \\")
    latex.append(r"\midrule")

    for _, row in ranks_df.iterrows():
        latex.append(f"{row['method']} & {row['avg_rank']:.2f} & {row['std_rank']:.2f} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    with open(output_path.with_suffix(".tex"), "w") as f:
        f.write("\n".join(latex))

    print(f"Saved: {output_path.with_suffix('.csv')}")
    print(f"Saved: {output_path.with_suffix('.tex')}")


def generate_summary_table(
    data: pd.DataFrame, methods: list[str], metrics: list[str], output_path: Path
) -> None:
    """Generate summary statistics table."""
    results = []

    for method in methods:
        row = {"method": method}
        for metric in metrics:
            col = f"{method}_{metric}"
            if col in data.columns:
                row[f"{metric}_mean"] = data[col].mean()
                row[f"{metric}_std"] = data[col].std()
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def plot_critical_difference(
    ranks_df: pd.DataFrame, cd: float, metric: str, output_path: Path
) -> None:
    """Plot Critical Difference diagram.

    Methods with rank differences <= CD are connected by a bar (not significantly different).
    """
    methods = ranks_df["method"].tolist()
    avg_ranks = ranks_df["avg_rank"].tolist()
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(10, max(3, n_methods * 0.4)))

    # Draw the axis
    ax.set_xlim(0.5, n_methods + 0.5)
    ax.set_ylim(0, n_methods + 1)
    ax.axhline(y=0.5, color="black", linewidth=1)

    # Draw tick marks and labels
    for i, (method, rank) in enumerate(zip(methods, avg_ranks)):
        # Tick mark
        ax.plot([rank, rank], [0.3, 0.7], color="black", linewidth=1)
        # Label (alternate above/below)
        y_pos = -0.5 if i % 2 == 0 else 1.0
        ax.text(rank, y_pos, f"{method}\n({rank:.2f})", ha="center", va="center", fontsize=9)

    # Draw CD bar
    ax.plot([1, 1 + cd], [n_methods + 0.3, n_methods + 0.3], color="red", linewidth=2)
    ax.text(1 + cd / 2, n_methods + 0.6, f"CD = {cd:.2f}", ha="center", fontsize=10, color="red")

    # Draw connections for non-significant pairs
    # Group methods that are not significantly different
    groups = []
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            if abs(avg_ranks[i] - avg_ranks[j]) <= cd:
                # Find or create group
                found = False
                for g in groups:
                    if i in g or j in g:
                        g.add(i)
                        g.add(j)
                        found = True
                        break
                if not found:
                    groups.append({i, j})

    # Merge overlapping groups
    merged = True
    while merged:
        merged = False
        new_groups = []
        for g in groups:
            added = False
            for ng in new_groups:
                if g & ng:
                    ng.update(g)
                    added = True
                    merged = True
                    break
            if not added:
                new_groups.append(g)
        groups = new_groups

    # Draw group bars
    y_offset = 1.5
    for group in groups:
        group_ranks = [avg_ranks[i] for i in group]
        min_rank = min(group_ranks)
        max_rank = max(group_ranks)
        ax.plot([min_rank, max_rank], [y_offset, y_offset], color="black", linewidth=3)
        y_offset += 0.8

    ax.set_title(f"Critical Difference Diagram: {metric}", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_performance_heatmap(
    data: pd.DataFrame, methods: list[str], metric: str, group_cols: list[str], output_path: Path
) -> None:
    """Plot performance heatmap across experimental factors."""
    # Pivot data
    pivot_data = []
    for method in methods:
        col = f"{method}_{metric}"
        if col not in data.columns:
            continue
        for _, row in data.groupby(group_cols)[col].mean().reset_index().iterrows():
            pivot_data.append(
                {
                    "method": method,
                    **{c: row[c] for c in group_cols},
                    "value": row[col],
                }
            )

    if not pivot_data:
        return

    pivot_df = pd.DataFrame(pivot_data)

    # Create heatmap for each factor
    for group_col in group_cols:
        pivot_table = pivot_df.pivot_table(
            values="value", index="method", columns=group_col, aggfunc="mean"
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(pivot_table.values, cmap="RdYlGn", aspect="auto")

        # Labels
        ax.set_xticks(range(len(pivot_table.columns)))
        ax.set_xticklabels(pivot_table.columns)
        ax.set_yticks(range(len(pivot_table.index)))
        ax.set_yticklabels(pivot_table.index)
        ax.set_xlabel(group_col)
        ax.set_ylabel("Method")
        ax.set_title(f"{metric} by {group_col}")

        # Add values
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                val = pivot_table.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(
            output_path.parent / f"{output_path.stem}_{group_col}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"Saved: {output_path.parent / f'{output_path.stem}_{group_col}.png'}")


def plot_boxplots(data: pd.DataFrame, methods: list[str], metric: str, output_path: Path) -> None:
    """Plot box plots comparing methods."""
    cols = [f"{m}_{metric}" for m in methods if f"{m}_{metric}" in data.columns]
    available_methods = [m for m in methods if f"{m}_{metric}" in data.columns]

    if not cols:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    values = [data[col].dropna().values for col in cols]
    bp = ax.boxplot(values, labels=available_methods, patch_artist=True)

    # Color boxes
    colors = plt.cm.tab10(np.linspace(0, 1, len(available_methods)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(metric)
    ax.set_title(f"Method Comparison: {metric}")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def analyze_synthetic_results(input_path: Path, tables_dir: Path, figures_dir: Path) -> None:
    """Analyze synthetic experiment results."""
    print("\n=== Analyzing Synthetic Experiments ===")

    if not input_path.exists():
        print(f"No results found at {input_path}")
        return

    data = pd.read_parquet(input_path)
    print(f"Loaded {len(data)} experiments")

    # Infer methods from columns
    precision_cols = [c for c in data.columns if c.endswith("_precision@10")]
    methods = [c.replace("_precision@10", "") for c in precision_cols]
    print(f"Methods: {methods}")

    # Metrics to analyze
    metrics = ["precision@5", "precision@10", "precision@20", "recall@10", "downstream_acc_mean"]

    # Create output directories
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # === TABLES ===
    print("\n--- Generating Tables ---")

    # 1. Friedman test
    generate_friedman_table(data, methods, metrics, tables_dir / "friedman_synthetic")

    # 2. Rankings for key metrics
    for metric in ["precision@10", "downstream_acc_mean"]:
        generate_ranking_table(data, methods, metric, tables_dir / f"ranks_{metric}")

    # 3. Summary statistics
    generate_summary_table(data, methods, metrics, tables_dir / "summary_synthetic.csv")

    # === FIGURES ===
    print("\n--- Generating Figures ---")

    # 4. Critical difference diagrams
    for metric in ["precision@10", "downstream_acc_mean"]:
        ranks_df = compute_ranks(data, methods, metric)
        if not ranks_df.empty:
            n_methods = len(ranks_df)
            n_datasets = len(data)
            cd = nemenyi_critical_difference(n_methods, n_datasets)
            plot_critical_difference(ranks_df, cd, metric, figures_dir / f"cd_{metric}.png")

    # 5. Box plots
    for metric in ["precision@10", "downstream_acc_mean"]:
        plot_boxplots(data, methods, metric, figures_dir / f"boxplot_{metric}.png")

    # 6. Heatmaps by experimental factors
    experimental_factors = ["n_features", "n_informative", "n_samples", "class_sep"]
    for metric in ["precision@10"]:
        plot_performance_heatmap(
            data, methods, metric, experimental_factors, figures_dir / f"heatmap_{metric}"
        )

    # === PRINT SUMMARY ===
    print("\n=== Results Summary ===")
    for n_features in sorted(data["n_features"].unique()):
        subset = data[data["n_features"] == n_features]
        print(f"\nn_features = {n_features}:")
        for method in methods:
            col = f"{method}_precision@10"
            if col in subset.columns:
                print(f"  {method}: {subset[col].mean():.3f} +/- {subset[col].std():.3f}")


def main():
    """Run comprehensive analysis."""
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Analyze synthetic experiment results
    synthetic_path = Path(__file__).parent.parent / "results" / "synthetic_experiments.parquet"
    analyze_synthetic_results(synthetic_path, TABLES_DIR, FIGURES_DIR)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nOutput directories:")
    print(f"  Tables:  {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
