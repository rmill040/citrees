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


# ==============================================================================
# Statistical Functions
# ==============================================================================


def bootstrap_ci(
    scores: np.ndarray, n_bootstrap: int = 2000, ci: float = 0.95, random_state: int = 42
) -> tuple[float, float]:
    """Bootstrap confidence interval.

    Important: Bootstrap across independent seeds/datasets, NOT CV folds.

    Parameters
    ----------
    scores : np.ndarray
        Array of scores (e.g., accuracy from each seed).
    n_bootstrap : int
        Number of bootstrap samples.
    ci : float
        Confidence level (e.g., 0.95 for 95% CI).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of CI.
    """
    rng = np.random.default_rng(random_state)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))
    alpha = 1 - ci
    return (
        float(np.percentile(bootstrap_means, alpha / 2 * 100)),
        float(np.percentile(bootstrap_means, (1 - alpha / 2) * 100)),
    )


def generate_summary_with_ci(
    data_wide: pd.DataFrame,
    methods: list[str],
    metrics: list[str],
    output_path: Path,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Generate summary table with bootstrap confidence intervals.

    Follows "Define Once, Apply Everywhere" architecture.
    Works for synthetic, classification, and regression results.

    Parameters
    ----------
    data_wide : pd.DataFrame
        Wide-format data with columns: {method}_{metric}
    methods : list[str]
        Method names (e.g., ['cif', 'rf', 'boruta'])
    metrics : list[str]
        Metric names (e.g., ['precision@10', 'accuracy'])
    output_path : Path
        Where to save CSV output
    n_bootstrap : int
        Number of bootstrap resamples (default: 2000)
    ci_level : float
        Confidence level (default: 0.95 for 95% CI)

    Returns
    -------
    pd.DataFrame
        Summary with columns:
        - method
        - {metric}_mean, {metric}_ci_lo, {metric}_ci_hi, {metric}_formatted
    """
    results = []

    for method in methods:
        row = {"method": method}
        for metric in metrics:
            col = f"{method}_{metric}"
            if col in data_wide.columns:
                values = data_wide[col].dropna().values
                if len(values) >= 5:  # Need minimum samples for meaningful CI
                    mean = float(np.mean(values))
                    lo, hi = bootstrap_ci(values, n_bootstrap=n_bootstrap, ci=ci_level)
                    row[f"{metric}_mean"] = mean
                    row[f"{metric}_ci_lo"] = lo
                    row[f"{metric}_ci_hi"] = hi
                    row[f"{metric}_formatted"] = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
                else:
                    row[f"{metric}_mean"] = float(np.mean(values)) if len(values) > 0 else np.nan
                    row[f"{metric}_ci_lo"] = np.nan
                    row[f"{metric}_ci_hi"] = np.nan
                    row[f"{metric}_formatted"] = (
                        f"{np.mean(values):.3f} [insufficient data]" if len(values) > 0 else "N/A"
                    )
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"  Saved summary with CI: {output_path}")
    return df


def pairwise_wilcoxon_holm(
    data: pd.DataFrame, methods: list[str], metric: str
) -> pd.DataFrame:
    """Wilcoxon signed-rank test with Holm-Bonferroni correction.

    Parameters
    ----------
    data : pd.DataFrame
        Results with columns for each method's metric.
    methods : list[str]
        List of method names.
    metric : str
        Metric column suffix.

    Returns
    -------
    pd.DataFrame
        Pairwise comparison results with corrected p-values.
    """
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    pvalues = []
    pairs = []

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i < j:
                col1 = f"{m1}_{metric}"
                col2 = f"{m2}_{metric}"
                if col1 in data.columns and col2 in data.columns:
                    v1 = data[col1].dropna().values
                    v2 = data[col2].dropna().values
                    min_len = min(len(v1), len(v2))
                    if min_len >= 10:
                        stat, pval = wilcoxon(v1[:min_len], v2[:min_len], alternative="two-sided")
                        pvalues.append(pval)
                        pairs.append((m1, m2, stat))

    if not pvalues:
        return pd.DataFrame()

    _, corrected_pvals, _, _ = multipletests(pvalues, method="holm")

    results = []
    for (m1, m2, stat), pval, corrected in zip(pairs, pvalues, corrected_pvals):
        results.append({
            "method1": m1,
            "method2": m2,
            "statistic": stat,
            "p_value": pval,
            "p_value_corrected": corrected,
            "significant": corrected < 0.05,
        })

    return pd.DataFrame(results)


def kendalls_w(chi2_friedman: float, n_datasets: int, k_methods: int) -> float:
    """Kendall's W effect size for Friedman test.

    Interpretation: 0.1=small, 0.3=medium, 0.5=large.

    Parameters
    ----------
    chi2_friedman : float
        Friedman chi-square statistic.
    n_datasets : int
        Number of datasets/observations.
    k_methods : int
        Number of methods compared.

    Returns
    -------
    float
        Kendall's W coefficient of concordance.
    """
    return chi2_friedman / (n_datasets * (k_methods - 1))


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size (standardized mean difference).

    Interpretation: 0.2=small, 0.5=medium, 0.8=large.

    Parameters
    ----------
    group1, group2 : np.ndarray
        Arrays of values for each group.

    Returns
    -------
    float
        Cohen's d value.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def interpret_cohens_d(d: float) -> str:
    """Return human-readable effect size interpretation.

    Parameters
    ----------
    d : float
        Cohen's d value.

    Returns
    -------
    str
        Effect size category: negligible, small, medium, or large.
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def nogueira_stability_index(feature_sets: list[list[int]], total_features: int) -> float:
    """Nogueira stability index with correction for chance.

    Measures consistency of feature selection across runs (e.g., CV folds, seeds).

    Parameters
    ----------
    feature_sets : list[list[int]]
        List of selected feature indices from each run.
    total_features : int
        Total number of features in the dataset.

    Returns
    -------
    float
        Stability index in range [-1, 1]. Higher is more stable.
    """
    M = len(feature_sets)
    p = total_features

    if M < 2:
        return 1.0

    # Compute selection frequency for each feature
    freq = np.zeros(p)
    for fs in feature_sets:
        for f in fs:
            if 0 <= f < p:
                freq[f] += 1
    freq /= M

    # Average number of selected features
    k_bar = np.mean([len(fs) for fs in feature_sets])

    # Nogueira formula
    numerator = (1 / p) * np.sum(freq * (1 - freq))
    denominator = (k_bar / p) * (1 - k_bar / p)

    if denominator == 0:
        return 1.0
    return float(1 - numerator / denominator)


def compute_noise_selection_rate(
    feature_ranking: list[int], noise_indices: list[int], k: int
) -> float:
    """Compute rate of noise features in top-k (false positive rate).

    This is the KEY metric for selection bias analysis. It measures how often
    a feature selection method incorrectly ranks noise features in the top-k.

    Parameters
    ----------
    feature_ranking : list[int]
        Feature indices sorted by importance (best first).
    noise_indices : list[int]
        Indices of known noise features (from ground truth).
    k : int
        Number of top features to consider.

    Returns
    -------
    float
        Fraction of top-k that are noise features. Range [0, 1].
        Lower is better. 0.0 = no noise selected, 1.0 = all noise.

    Examples
    --------
    >>> # Perfect selection: informative features ranked first
    >>> compute_noise_selection_rate([0, 1, 2, 3, 4], [10, 11, 12], k=3)
    0.0

    >>> # Worst case: all noise in top-k
    >>> compute_noise_selection_rate([10, 11, 12, 0, 1], [10, 11, 12], k=3)
    1.0

    >>> # Mixed: half noise in top-k
    >>> compute_noise_selection_rate([0, 10, 1, 11], [10, 11, 12], k=4)
    0.5
    """
    if k == 0 or len(noise_indices) == 0:
        return 0.0
    top_k = set(feature_ranking[:k])
    noise_set = set(noise_indices)
    return len(top_k & noise_set) / k


def analyze_selection_bias(
    data: pd.DataFrame,
    ground_truth: dict[str, dict],
    methods: list[str],
    tables_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    """Analyze selection bias on datasets with high-cardinality noise.

    This proves citrees' core claim: unbiased feature selection.
    RF/XGBoost tend to favor high-cardinality features, leading to elevated
    noise selection rates. Conditional inference methods should maintain
    noise selection near the nominal alpha level.

    Parameters
    ----------
    data : pd.DataFrame
        Results with columns: dataset, method, feature_ranking.
    ground_truth : dict[str, dict]
        Ground truth metadata keyed by dataset name. Each entry should have:
        - 'informative_indices': list of true informative feature indices
        - 'noise_indices': list of noise feature indices
    methods : list[str]
        List of method names to analyze.
    tables_dir : Path
        Directory to save output tables.
    figures_dir : Path
        Directory to save output figures.

    Returns
    -------
    pd.DataFrame
        Detailed results with noise selection rate for each method/dataset/k.
    """
    print("\n" + "=" * 60)
    print("SELECTION BIAS ANALYSIS")
    print("=" * 60)

    # Filter to bias datasets (those with noise_indices in ground truth)
    bias_datasets = [
        name for name, meta in ground_truth.items()
        if meta.get("noise_indices") and len(meta.get("noise_indices", [])) > 0
    ]

    if not bias_datasets:
        print("No datasets with noise_indices found in ground truth.")
        return pd.DataFrame()

    records = []

    for dataset in bias_datasets:
        meta = ground_truth[dataset]
        informative_indices = meta.get("informative_indices", [])
        noise_indices = meta.get("noise_indices", [])

        # Get results for this dataset
        dataset_results = data[data["dataset"] == dataset]

        for _, row in dataset_results.iterrows():
            method = row["method"]
            if method not in methods:
                continue

            # Parse feature ranking
            feature_ranking = row.get("feature_ranking", [])
            if isinstance(feature_ranking, str):
                import ast
                feature_ranking = ast.literal_eval(feature_ranking)

            if not feature_ranking:
                continue

            # Compute metrics at different k values
            for k in [5, 10, 20]:
                if k > len(feature_ranking):
                    continue

                noise_rate = compute_noise_selection_rate(feature_ranking, noise_indices, k)

                records.append({
                    "dataset": dataset,
                    "method": method,
                    "k": k,
                    "noise_selection_rate": noise_rate,
                    "n_noise_features": len(noise_indices),
                    "n_informative": len(informative_indices),
                })

    if not records:
        print("No results found for bias datasets.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Aggregate by method (mean across datasets and k values)
    summary = (
        df.groupby("method")
        .agg({"noise_selection_rate": ["mean", "std"]})
        .round(3)
    )
    summary.columns = ["noise_rate_mean", "noise_rate_std"]
    summary = summary.sort_values("noise_rate_mean")

    print("\n=== SELECTION BIAS SUMMARY (mean across datasets) ===")
    print(summary.to_string())

    # Save summary
    summary_path = tables_dir / "selection_bias_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nSaved: {summary_path}")

    # Key comparison: RF methods vs citrees methods
    rf_methods = {"rf", "et", "xgb", "lgbm", "cat"}
    citrees_methods = {"cit", "cif"}

    rf_data = df[df["method"].isin(rf_methods)]
    ci_data = df[df["method"].isin(citrees_methods)]

    if not rf_data.empty and not ci_data.empty:
        rf_noise = rf_data["noise_selection_rate"].mean()
        ci_noise = ci_data["noise_selection_rate"].mean()

        print("\n=== KEY FINDING ===")
        print(f"RF/Boosting methods noise selection rate: {rf_noise:.3f}")
        print(f"citrees methods noise selection rate: {ci_noise:.3f}")
        if rf_noise > 0:
            print(f"Reduction: {(rf_noise - ci_noise) / rf_noise * 100:.1f}%")

    # Generate visualization
    if not summary.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        methods_sorted = summary.index.tolist()
        x = range(len(methods_sorted))

        # Color by method type
        colors = []
        for m in methods_sorted:
            if m in rf_methods:
                colors.append("indianred")
            elif m in citrees_methods:
                colors.append("seagreen")
            else:
                colors.append("steelblue")

        ax.bar(x, summary["noise_rate_mean"], color=colors, edgecolor="black", linewidth=0.5)
        ax.errorbar(x, summary["noise_rate_mean"], yerr=summary["noise_rate_std"],
                    fmt="none", color="black", capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(methods_sorted, rotation=45, ha="right")
        ax.set_ylabel("Noise Selection Rate (lower is better)")
        ax.set_title("Selection Bias: Rate of Spurious Noise Feature Selection")
        ax.axhline(y=0.05, color="gray", linestyle="--", linewidth=1, label="α = 0.05")
        ax.legend()
        ax.set_ylim(0, min(1.0, summary["noise_rate_mean"].max() * 1.3))

        plt.tight_layout()
        fig_path = figures_dir / "selection_bias_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path}")

    return df


# ==============================================================================
# Core Statistical Tests
# ==============================================================================


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


def analyze_stratified_results(
    data: pd.DataFrame, methods: list[str], metric: str, tables_dir: Path
) -> None:
    """Analyze results stratified by dataset characteristics.

    Parameters
    ----------
    data : pd.DataFrame
        Results dataframe with columns: dataset, method, n_samples, n_features, class_sep.
    methods : list[str]
        List of method names.
    metric : str
        Metric column name (e.g., 'precision@10').
    tables_dir : Path
        Directory to save output tables.
    """
    if metric not in data.columns:
        print(f"Metric {metric} not found in data")
        return

    # Work with a copy to avoid modifying original
    df = data.copy()

    # 1. By synthetic type
    if "dataset_type" in df.columns:
        stratified = df.groupby(["dataset_type", "method"])[metric].agg(["mean", "std"])
        out_path = tables_dir / f"stratified_by_type_{metric.replace('@', '_at_')}.csv"
        stratified.to_csv(out_path)
        print(f"Saved: {out_path}")
    elif "dataset" in df.columns:
        # Fallback: extract type from dataset name
        df["synthetic_type"] = df["dataset"].str.extract(r"synthetic_(\w+)_")[0]
        if df["synthetic_type"].notna().any():
            stratified = df.groupby(["synthetic_type", "method"])[metric].agg(["mean", "std"])
            out_path = tables_dir / f"stratified_by_type_{metric.replace('@', '_at_')}.csv"
            stratified.to_csv(out_path)
            print(f"Saved: {out_path}")

    # 2. By sample size bins
    if "n_samples" in df.columns:
        df["size_bin"] = pd.cut(
            df["n_samples"], bins=[0, 500, 1000, np.inf], labels=["small", "medium", "large"]
        )
        stratified_size = df.groupby(["size_bin", "method"])[metric].agg(["mean", "std"])
        out_path = tables_dir / f"stratified_by_size_{metric.replace('@', '_at_')}.csv"
        stratified_size.to_csv(out_path)
        print(f"Saved: {out_path}")

    # 3. By dimensionality (p/n ratio)
    if "n_samples" in df.columns:
        feature_col = "n_features_final" if "n_features_final" in df.columns else "n_features"
        if feature_col in df.columns:
            df["pn_ratio"] = df[feature_col] / df["n_samples"]
        else:
            df["pn_ratio"] = np.nan
        df["dim_bin"] = pd.cut(
            df["pn_ratio"], bins=[0, 0.5, 1.0, np.inf], labels=["low", "medium", "high"]
        )
        stratified_dim = df.groupby(["dim_bin", "method"])[metric].agg(["mean", "std"])
        out_path = tables_dir / f"stratified_by_dim_{metric.replace('@', '_at_')}.csv"
        stratified_dim.to_csv(out_path)
        print(f"Saved: {out_path}")

    # 4. By signal strength (class_sep)
    if "class_sep" in df.columns:
        stratified_signal = df.groupby(["class_sep", "method"])[metric].agg(["mean", "std"])
        out_path = tables_dir / f"stratified_by_signal_{metric.replace('@', '_at_')}.csv"
        stratified_signal.to_csv(out_path)
        print(f"Saved: {out_path}")


def compute_stability_analysis(
    rankings_data: pd.DataFrame, methods: list[str], top_k: int, tables_dir: Path
) -> None:
    """Compute Nogueira stability index for each method.

    Parameters
    ----------
    rankings_data : pd.DataFrame
        DataFrame with columns: dataset, method, seed, feature_ranking.
        feature_ranking should be a list of feature indices in rank order.
    methods : list[str]
        List of method names.
    top_k : int
        Number of top features to consider for stability.
    tables_dir : Path
        Directory to save output tables.
    """
    if "feature_ranking" not in rankings_data.columns:
        print("No feature_ranking column found - skipping stability analysis")
        return

    results = []

    for method in methods:
        method_data = rankings_data[rankings_data["method"] == method]

        for dataset in method_data["dataset"].unique():
            dataset_method = method_data[method_data["dataset"] == dataset]

            # Get feature sets (top-k features) across seeds
            feature_sets = []
            total_features = 0
            for _, row in dataset_method.iterrows():
                ranking = row["feature_ranking"]
                if isinstance(ranking, str):
                    import ast

                    ranking = ast.literal_eval(ranking)
                if ranking:
                    feature_sets.append(ranking[:top_k])
                    total_features = len(ranking)

            if len(feature_sets) >= 2 and total_features > 0:
                stability = nogueira_stability_index(feature_sets, total_features)
                results.append(
                    {
                        "method": method,
                        "dataset": dataset,
                        f"stability@{top_k}": stability,
                        "n_seeds": len(feature_sets),
                    }
                )

    if results:
        df = pd.DataFrame(results)
        # Summary by method
        summary = df.groupby("method")[f"stability@{top_k}"].agg(["mean", "std"])
        out_path = tables_dir / f"stability_at_{top_k}.csv"
        summary.to_csv(out_path)
        print(f"Saved: {out_path}")


def analyze_synthetic_results(input_path: Path, tables_dir: Path, figures_dir: Path) -> None:
    """Analyze synthetic experiment results.

    Expects data from synthetic_analysis.py with columns:
    - dataset, method, seed, fold_idx
    - n_features, n_informative, n_samples, class_sep, etc.
    - precision@k, recall@k, f1@k for various k values
    - precision_ir@k, recall_ir@k, f1_ir@k (informative+redundant)
    - confounder_rate@k (correlated noise selection rate)
    """
    print("\n=== Analyzing Synthetic Experiments ===")

    if not input_path.exists():
        print(f"No results found at {input_path}")
        return

    data = pd.read_parquet(input_path)
    print(f"Loaded {len(data)} result rows")

    # Get methods from the 'method' column
    methods = sorted(data["method"].unique())
    print(f"Methods: {methods}")

    # Find available metrics
    metric_cols = [
        c
        for c in data.columns
        if c.startswith(
            (
                "precision@",
                "recall@",
                "f1@",
                "precision_ir@",
                "recall_ir@",
                "f1_ir@",
            )
        )
    ]
    print(f"Metrics: {metric_cols}")

    # Create output directories
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate by dataset and method (mean across folds)
    agg_cols = ["dataset", "method", "n_features", "n_informative", "n_samples"]
    if "class_sep" in data.columns:
        agg_cols.append("class_sep")

    data_agg = data.groupby(agg_cols)[metric_cols].mean().reset_index()

    # Pivot to wide format for analysis (method as columns)
    # This matches the format expected by generate_friedman_table, etc.
    data_wide = data_agg.pivot(
        index=["dataset", "n_features", "n_informative", "n_samples"],
        columns="method",
        values=metric_cols,
    )
    # Flatten column names: (precision@10, method) -> method_precision@10
    data_wide.columns = [f"{col[1]}_{col[0]}" for col in data_wide.columns]
    data_wide = data_wide.reset_index()

    # Update methods list with column format
    methods_prefixed = [f"{m}_precision@10" for m in methods]

    # === TABLES ===
    print("\n--- Generating Summary Tables ---")

    # Summary by method
    summary = data.groupby("method")[metric_cols].agg(["mean", "std"])
    summary_path = tables_dir / "summary_synthetic.csv"
    summary.to_csv(summary_path)
    print(f"Saved: {summary_path}")

    # === FIGURES ===
    print("\n--- Generating Figures ---")

    # Box plots by method for precision@10
    if "precision@10" in data.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        data.boxplot(column="precision@10", by="method", ax=ax)
        ax.set_title("Precision@10 by Method (Synthetic Datasets)")
        ax.set_xlabel("Method")
        ax.set_ylabel("Precision@10")
        plt.suptitle("")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        boxplot_path = figures_dir / "boxplot_precision10_synthetic.png"
        plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {boxplot_path}")

    # === CRITICAL DIFFERENCE DIAGRAMS ===
    print("\n--- Generating Critical Difference Diagrams ---")

    for metric in ["precision@5", "precision@10", "precision@20"]:
        if metric in metric_cols:
            # Compute ranks across datasets for each method
            ranks_df = compute_ranks(data_wide, methods, metric, higher_is_better=True)
            if not ranks_df.empty:
                n_methods = len(ranks_df)
                n_datasets = len(data_wide)
                cd = nemenyi_critical_difference(n_methods, n_datasets)

                # Generate CD diagram
                cd_path = figures_dir / f"cd_{metric.replace('@', '_at_')}.png"
                plot_critical_difference(ranks_df, cd, metric, cd_path)

    # === RANKING TABLES ===
    print("\n--- Generating Ranking Tables ---")

    for metric in ["precision@10", "recall@10", "f1@10"]:
        if metric in metric_cols:
            ranking_path = tables_dir / f"ranking_{metric.replace('@', '_at_')}"
            generate_ranking_table(data_wide, methods, metric, ranking_path, higher_is_better=True)

    # Generate Friedman test table
    friedman_path = tables_dir / "friedman_synthetic"
    generate_friedman_table(data_wide, methods, metric_cols, friedman_path)

    # === STRATIFIED ANALYSIS ===
    print("\n--- Generating Stratified Analysis ---")
    analyze_stratified_results(data, methods, "precision@10", tables_dir)

    # === STABILITY ANALYSIS ===
    print("\n--- Generating Stability Analysis ---")
    # Stability requires raw feature_ranking column from original data
    if "feature_ranking" in data.columns:
        for top_k in [5, 10, 20]:
            compute_stability_analysis(data, methods, top_k, tables_dir)
    else:
        print("No feature_ranking column - skipping stability analysis")

    # === PRINT SUMMARY ===
    print("\n=== Results Summary ===")
    print(summary.round(3).to_string())


# ==============================================================================
# Generic Statistical Analysis (Define Once, Apply Everywhere)
# ==============================================================================


def run_statistical_analysis(
    data_wide: pd.DataFrame,
    methods: list[str],
    metrics: list[str],
    output_prefix: str,
    tables_dir: Path,
    figures_dir: Path,
    higher_is_better: dict[str, bool] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run ALL statistical analyses on ANY results dataset.

    This is the SINGLE entry point for statistical analysis.
    Apply to: synthetic, classification, regression results.

    Generates:
    - Friedman omnibus test
    - Pairwise Wilcoxon + Holm with Cohen's d effect sizes
    - Ranking tables with bootstrap CIs
    - Critical difference diagrams

    Parameters
    ----------
    data_wide : pd.DataFrame
        Results with columns: {method}_{metric} for each method/metric combo.
        One row per dataset.
    methods : list[str]
        Method names (e.g., ['rf', 'cif', 'boruta', ...])
    metrics : list[str]
        Metric names (e.g., ['precision@10', 'accuracy', 'r2'])
    output_prefix : str
        Prefix for output files (e.g., 'synthetic', 'clf', 'reg')
    tables_dir, figures_dir : Path
        Output directories
    higher_is_better : dict[str, bool], optional
        Whether higher values are better for each metric.
        Default: True for all metrics.

    Returns
    -------
    dict[str, pd.DataFrame]
        All generated tables keyed by name.
    """
    if higher_is_better is None:
        higher_is_better = {m: True for m in metrics}

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    print(f"\n{'='*60}")
    print(f"STATISTICAL ANALYSIS: {output_prefix.upper()}")
    print(f"{'='*60}")
    print(f"Methods: {methods}")
    print(f"Metrics: {metrics}")
    print(f"Datasets: {len(data_wide)}")

    for metric in metrics:
        print(f"\n--- {metric} ---")
        hib = higher_is_better.get(metric, True)

        # Check if we have enough data
        cols_present = [f"{m}_{metric}" for m in methods if f"{m}_{metric}" in data_wide.columns]
        if len(cols_present) < 2:
            print(f"  Skipping: insufficient methods with {metric}")
            continue

        # 1. Friedman omnibus test
        try:
            chi2, p_friedman = friedman_test(data_wide, methods, metric)
            n_datasets = len(data_wide)
            k_methods = len(methods)
            w = kendalls_w(chi2, n_datasets, k_methods)
            print(f"  Friedman: chi2={chi2:.2f}, p={p_friedman:.4f}, W={w:.3f}")

            friedman_row = {
                "metric": metric,
                "chi2": chi2,
                "p_value": p_friedman,
                "kendalls_w": w,
                "n_datasets": n_datasets,
                "n_methods": k_methods,
            }
            results[f"{output_prefix}_friedman_{metric}"] = pd.DataFrame([friedman_row])
        except Exception as e:
            print(f"  Friedman failed: {e}")

        # 2. Pairwise Wilcoxon + Holm + Cohen's d
        pairwise_df = pairwise_wilcoxon_holm(data_wide, methods, metric)
        if not pairwise_df.empty:
            # Add Cohen's d and interpretation
            cohens_d_vals = []
            effect_sizes = []
            for _, row in pairwise_df.iterrows():
                col1 = f"{row['method1']}_{metric}"
                col2 = f"{row['method2']}_{metric}"
                if col1 in data_wide.columns and col2 in data_wide.columns:
                    v1 = data_wide[col1].dropna().values
                    v2 = data_wide[col2].dropna().values
                    d = cohens_d(v1, v2)
                    cohens_d_vals.append(d)
                    effect_sizes.append(interpret_cohens_d(d))
                else:
                    cohens_d_vals.append(np.nan)
                    effect_sizes.append("unknown")

            pairwise_df["cohens_d"] = cohens_d_vals
            pairwise_df["effect_size"] = effect_sizes

            # Save
            out_path = tables_dir / f"{output_prefix}_pairwise_{metric.replace('@', '_at_').replace('/', '_')}.csv"
            pairwise_df.to_csv(out_path, index=False)
            print(f"  Saved: {out_path.name}")

            # Print significant pairs
            sig_pairs = pairwise_df[pairwise_df["significant"]]
            if not sig_pairs.empty:
                print(f"  Significant pairs ({len(sig_pairs)}):")
                for _, row in sig_pairs.head(5).iterrows():
                    print(f"    {row['method1']} vs {row['method2']}: p_adj={row['p_value_corrected']:.4f}, d={row['cohens_d']:.2f} ({row['effect_size']})")
                if len(sig_pairs) > 5:
                    print(f"    ... and {len(sig_pairs) - 5} more")

            results[f"{output_prefix}_pairwise_{metric}"] = pairwise_df
        else:
            print(f"  No pairwise comparisons (insufficient data)")

        # 3. Ranking table with bootstrap CIs
        try:
            ranking_path = tables_dir / f"{output_prefix}_ranking_{metric.replace('@', '_at_').replace('/', '_')}"
            generate_ranking_table(data_wide, methods, metric, ranking_path, higher_is_better=hib)
        except Exception as e:
            print(f"  Ranking table failed: {e}")

        # 4. Critical difference diagram
        try:
            ranks_df = compute_ranks(data_wide, methods, metric, higher_is_better=hib)
            if not ranks_df.empty:
                n_methods = len(ranks_df)
                n_datasets = len(data_wide)
                cd = nemenyi_critical_difference(n_methods, n_datasets)
                cd_path = figures_dir / f"{output_prefix}_cd_{metric.replace('@', '_at_').replace('/', '_')}.png"
                plot_critical_difference(ranks_df, cd, f"{output_prefix} - {metric}", cd_path)
        except Exception as e:
            print(f"  CD diagram failed: {e}")

    # 5. Summary table with bootstrap CIs (aggregates all metrics)
    try:
        summary_ci_path = tables_dir / f"{output_prefix}_summary_with_ci.csv"
        summary_df = generate_summary_with_ci(
            data_wide=data_wide,
            methods=methods,
            metrics=metrics,
            output_path=summary_ci_path,
        )
        results[f"{output_prefix}_summary_with_ci"] = summary_df
    except Exception as e:
        print(f"  Summary with CI failed: {e}")

    return results


def load_and_pivot_results(input_path: Path, methods: list[str], metric_cols: list[str]) -> pd.DataFrame:
    """Load results and pivot to wide format for statistical analysis.

    Parameters
    ----------
    input_path : Path
        Path to parquet file with results.
    methods : list[str]
        Method names to include.
    metric_cols : list[str]
        Metric column names to include.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with columns: {method}_{metric}.
        One row per dataset.
    """
    data = pd.read_parquet(input_path)

    # Aggregate by dataset and method (mean across folds/seeds)
    agg_cols = ["dataset", "method"]
    for col in ["n_features", "n_informative", "n_samples", "class_sep"]:
        if col in data.columns:
            agg_cols.append(col)

    available_metrics = [m for m in metric_cols if m in data.columns]
    data_agg = data.groupby(agg_cols)[available_metrics].mean().reset_index()

    # Pivot to wide format (one row per dataset, columns = {method}_{metric})
    index_cols = [c for c in agg_cols if c != "method"]
    data_wide = data_agg.pivot(
        index=index_cols,
        columns="method",
        values=available_metrics,
    )
    # Flatten column names: (precision@10, rf) -> rf_precision@10
    data_wide.columns = [f"{col[1]}_{col[0]}" for col in data_wide.columns]
    data_wide = data_wide.reset_index()

    return data_wide


def main():
    """Run comprehensive analysis on ALL dataset types."""
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CITREES COMPREHENSIVE ANALYSIS")
    print("=" * 60)

    # === SYNTHETIC DATASETS ===
    # First run the existing analysis for detailed synthetic-specific outputs
    synthetic_path = Path(__file__).parent.parent / "results" / "synthetic_analysis.parquet"
    if synthetic_path.exists():
        analyze_synthetic_results(synthetic_path, TABLES_DIR, FIGURES_DIR)

        # Then run the generic statistical analysis for standardized outputs
        data = pd.read_parquet(synthetic_path)
        methods = sorted(data["method"].unique())
        metric_cols = [c for c in data.columns if c.startswith(("precision@", "recall@", "f1@"))]
        data_wide = load_and_pivot_results(synthetic_path, methods, metric_cols)

        run_statistical_analysis(
            data_wide=data_wide,
            methods=methods,
            metrics=["precision@10", "recall@10", "f1@10"],
            output_prefix="synthetic",
            tables_dir=TABLES_DIR,
            figures_dir=FIGURES_DIR,
        )
    else:
        print(f"\nSkipping synthetic analysis: {synthetic_path} not found")

    # === CLASSIFICATION DATASETS ===
    clf_eval_path = Path(__file__).parent.parent / "results" / "clf_evaluation.parquet"
    if clf_eval_path.exists():
        print("\n" + "=" * 60)
        print("CLASSIFICATION EVALUATION")
        print("=" * 60)
        data = pd.read_parquet(clf_eval_path)
        methods = sorted(data["method"].unique())
        metric_cols = [c for c in data.columns if c in ["accuracy", "f1_macro", "auc", "balanced_accuracy"]]
        data_wide = load_and_pivot_results(clf_eval_path, methods, metric_cols)

        run_statistical_analysis(
            data_wide=data_wide,
            methods=methods,
            metrics=["accuracy", "f1_macro", "balanced_accuracy"],
            output_prefix="clf",
            tables_dir=TABLES_DIR,
            figures_dir=FIGURES_DIR,
        )
    else:
        print(f"\nSkipping classification analysis: {clf_eval_path} not found")
        print("  (Run evaluation experiments first to generate this file)")

    # === REGRESSION DATASETS ===
    reg_eval_path = Path(__file__).parent.parent / "results" / "reg_evaluation.parquet"
    if reg_eval_path.exists():
        print("\n" + "=" * 60)
        print("REGRESSION EVALUATION")
        print("=" * 60)
        data = pd.read_parquet(reg_eval_path)
        methods = sorted(data["method"].unique())
        metric_cols = [c for c in data.columns if c in ["r2", "mse", "mae", "rmse"]]
        data_wide = load_and_pivot_results(reg_eval_path, methods, metric_cols)

        run_statistical_analysis(
            data_wide=data_wide,
            methods=methods,
            metrics=["r2", "mse", "mae"],
            output_prefix="reg",
            tables_dir=TABLES_DIR,
            figures_dir=FIGURES_DIR,
            higher_is_better={"r2": True, "mse": False, "mae": False},
        )
    else:
        print(f"\nSkipping regression analysis: {reg_eval_path} not found")
        print("  (Run evaluation experiments first to generate this file)")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nOutput directories:")
    print(f"  Tables:  {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
