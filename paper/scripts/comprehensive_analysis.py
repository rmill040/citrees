"""Comprehensive analysis for citrees paper experiments.

This script performs analysis on both real and synthetic datasets:

1. FEATURE SELECTION QUALITY
   - Precision@k, Recall@k for synthetic datasets (ground truth known)
   - Friedman + Nemenyi tests across all datasets

2. SELECTION BIAS ANALYSIS
   - Compare RF vs citrees on high-cardinality noise datasets
   - Show RF spuriously selects noise features, citrees doesn't

3. SELECTOR COMPARISON
   - MC vs RDC vs MI on linear vs nonlinear datasets
   - When to use which selector

4. TIMING ANALYSIS
   - Wall-clock time by method
   - Scaling with n_samples, n_features

5. DOWNSTREAM ACCURACY
   - How does feature selection quality translate to model performance

Usage:
    # After running experiments
    uv run python paper/scripts/comprehensive_analysis.py
"""
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "comprehensive_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

# Ground truth for synthetic datasets
GROUND_TRUTH_PATH = DATA_DIR / "synthetic_ground_truth.json"


def load_ground_truth() -> Dict[str, Any]:
    """Load ground truth for synthetic datasets."""
    if GROUND_TRUTH_PATH.exists():
        with open(GROUND_TRUTH_PATH) as f:
            return json.load(f)
    return {}


def compute_precision_recall_at_k(
    feature_ranks: List[int],
    informative_indices: List[int],
    k: int
) -> Tuple[float, float]:
    """Compute precision@k and recall@k.

    Parameters
    ----------
    feature_ranks : List[int]
        Ranked feature indices (most important first).
    informative_indices : List[int]
        Ground truth informative feature indices.
    k : int
        Number of top features to consider.

    Returns
    -------
    Tuple[float, float]
        (precision@k, recall@k)
    """
    if k == 0 or len(informative_indices) == 0:
        return 0.0, 0.0

    top_k = set(feature_ranks[:k])
    true_set = set(informative_indices)

    true_positives = len(top_k & true_set)
    precision = true_positives / k
    recall = true_positives / len(true_set)

    return precision, recall


def compute_noise_selection_rate(
    feature_ranks: List[int],
    noise_indices: List[int],
    k: int
) -> float:
    """Compute rate of noise features in top-k (false positive rate).

    This is the key metric for selection bias analysis.
    """
    if k == 0 or len(noise_indices) == 0:
        return 0.0

    top_k = set(feature_ranks[:k])
    noise_set = set(noise_indices)

    return len(top_k & noise_set) / k


def friedman_test(scores: np.ndarray) -> Tuple[float, float]:
    """Perform Friedman test.

    Parameters
    ----------
    scores : np.ndarray
        Shape (n_datasets, n_methods)

    Returns
    -------
    Tuple[float, float]
        (chi_square, p_value)
    """
    if scores.shape[0] < 2 or scores.shape[1] < 3:
        return np.nan, np.nan

    # Remove rows with NaN
    mask = ~np.any(np.isnan(scores), axis=1)
    scores = scores[mask]

    if scores.shape[0] < 2:
        return np.nan, np.nan

    stat, pvalue = stats.friedmanchisquare(*[scores[:, i] for i in range(scores.shape[1])])
    return stat, pvalue


def nemenyi_cd(n_methods: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Compute Nemenyi critical difference."""
    q_values = {
        (3, 0.05): 2.343, (4, 0.05): 2.569, (5, 0.05): 2.728,
        (6, 0.05): 2.850, (7, 0.05): 2.949, (8, 0.05): 3.031,
        (9, 0.05): 3.102, (10, 0.05): 3.164, (11, 0.05): 3.219,
        (12, 0.05): 3.268, (13, 0.05): 3.313, (14, 0.05): 3.354,
        (15, 0.05): 3.391,
    }
    q = q_values.get((n_methods, alpha), 2.728)
    return q * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))


def compute_ranks(scores: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """Compute ranks per row."""
    if higher_is_better:
        return stats.rankdata(-scores, axis=1)
    return stats.rankdata(scores, axis=1)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_selection_bias(
    results_df: pd.DataFrame,
    ground_truth: Dict[str, Any],
    output_dir: Path
) -> pd.DataFrame:
    """Analyze selection bias on high-cardinality noise datasets.

    This is THE key experiment to prove citrees' core claim.
    """
    print("\n" + "=" * 60)
    print("SELECTION BIAS ANALYSIS")
    print("=" * 60)

    # Filter to bias datasets
    bias_datasets = [name for name in ground_truth.keys() if "bias" in name]

    if not bias_datasets:
        print("No selection bias datasets found.")
        return pd.DataFrame()

    records = []

    for dataset in bias_datasets:
        meta = ground_truth[dataset]
        informative_indices = meta["informative_indices"]
        noise_indices = meta.get("noise_indices", [])

        if not noise_indices:
            continue

        # Get results for this dataset
        dataset_results = results_df[results_df["dataset"] == dataset]

        for _, row in dataset_results.iterrows():
            feature_ranks = [int(x) for x in row["feature_ranks"].split(",")]

            for k in [5, 10, 20]:
                precision, recall = compute_precision_recall_at_k(
                    feature_ranks, informative_indices, k
                )
                noise_rate = compute_noise_selection_rate(
                    feature_ranks, noise_indices, k
                )

                records.append({
                    "dataset": dataset,
                    "method": row["method"],
                    "k": k,
                    "precision": precision,
                    "recall": recall,
                    "noise_selection_rate": noise_rate,
                    "n_noise_features": len(noise_indices),
                    "high_cardinality_levels": meta.get("high_cardinality_levels", 0),
                })

    if not records:
        print("No results found for bias datasets.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Aggregate by method
    summary = df.groupby("method").agg({
        "precision": "mean",
        "recall": "mean",
        "noise_selection_rate": "mean",
    }).round(3)

    print("\n=== SELECTION BIAS SUMMARY (mean across datasets) ===")
    print(summary.sort_values("noise_selection_rate"))

    # Save
    summary.to_csv(output_dir / "selection_bias_summary.csv")

    # Key comparison: RF vs citrees
    rf_methods = ["rf", "et", "dt", "xgb", "lightgbm"]
    citrees_methods = ["cit", "cif"]

    rf_noise = df[df["method"].isin(rf_methods)]["noise_selection_rate"].mean()
    ci_noise = df[df["method"].isin(citrees_methods)]["noise_selection_rate"].mean()

    print(f"\n=== KEY FINDING ===")
    print(f"RF/Boosting methods noise selection rate: {rf_noise:.3f}")
    print(f"citrees methods noise selection rate: {ci_noise:.3f}")
    print(f"Reduction: {(rf_noise - ci_noise) / rf_noise * 100:.1f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = summary.index.tolist()
    x = range(len(methods))
    ax.bar(x, summary["noise_selection_rate"], color=["red" if m in rf_methods else "green" for m in methods])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Noise Selection Rate (lower is better)")
    ax.set_title("Selection Bias: Rate of Spurious Noise Feature Selection")
    ax.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "selection_bias_comparison.png", dpi=300)
    plt.close()

    return df


def analyze_selector_comparison(
    results_df: pd.DataFrame,
    ground_truth: Dict[str, Any],
    output_dir: Path
) -> pd.DataFrame:
    """Compare selectors (MC vs RDC vs MI) on different dataset types."""
    print("\n" + "=" * 60)
    print("SELECTOR COMPARISON ANALYSIS")
    print("=" * 60)

    # Separate linear vs nonlinear datasets
    linear_datasets = [name for name in ground_truth.keys()
                       if "syn_p" in name and "nonlinear" not in name]
    nonlinear_datasets = [name for name in ground_truth.keys() if "nonlinear" in name]

    selector_methods = ["mc", "mi", "rdc", "ptest_mc", "ptest_mi", "ptest_rdc"]

    records = []

    for dataset_type, datasets in [("linear", linear_datasets), ("nonlinear", nonlinear_datasets)]:
        for dataset in datasets:
            if dataset not in ground_truth:
                continue

            meta = ground_truth[dataset]
            informative_indices = meta["informative_indices"]

            dataset_results = results_df[
                (results_df["dataset"] == dataset) &
                (results_df["method"].isin(selector_methods))
            ]

            for _, row in dataset_results.iterrows():
                feature_ranks = [int(x) for x in row["feature_ranks"].split(",")]

                for k in [5, 10, len(informative_indices)]:
                    precision, recall = compute_precision_recall_at_k(
                        feature_ranks, informative_indices, k
                    )

                    records.append({
                        "dataset_type": dataset_type,
                        "dataset": dataset,
                        "method": row["method"],
                        "k": k,
                        "precision": precision,
                        "recall": recall,
                    })

    if not records:
        print("No selector comparison data found.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Summary by dataset type and method
    summary = df.groupby(["dataset_type", "method"]).agg({
        "precision": "mean",
        "recall": "mean",
    }).round(3)

    print("\n=== SELECTOR COMPARISON ===")
    print(summary)

    summary.to_csv(output_dir / "selector_comparison.csv")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, dtype in zip(axes, ["linear", "nonlinear"]):
        subset = df[(df["dataset_type"] == dtype) & (df["k"] == 10)]
        method_means = subset.groupby("method")["precision"].mean().sort_values(ascending=False)

        ax.bar(range(len(method_means)), method_means.values)
        ax.set_xticks(range(len(method_means)))
        ax.set_xticklabels(method_means.index, rotation=45, ha="right")
        ax.set_ylabel("Precision@10")
        ax.set_title(f"{dtype.capitalize()} Relationships")

    plt.tight_layout()
    plt.savefig(output_dir / "selector_comparison.png", dpi=300)
    plt.close()

    return df


def analyze_timing(results_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Analyze timing/computational cost."""
    print("\n" + "=" * 60)
    print("TIMING ANALYSIS")
    print("=" * 60)

    if "elapsed_seconds" not in results_df.columns:
        print("No timing data available.")
        return pd.DataFrame()

    # Summary by method
    timing_summary = results_df.groupby("method")["elapsed_seconds"].agg([
        "mean", "std", "min", "max", "count"
    ]).round(3)

    print("\n=== TIMING BY METHOD (seconds) ===")
    print(timing_summary.sort_values("mean"))

    timing_summary.to_csv(output_dir / "timing_by_method.csv")

    # Timing by n_samples and n_features
    if "n_samples" in results_df.columns and "n_features" in results_df.columns:
        timing_by_size = results_df.groupby(["method", "n_samples", "n_features"])["elapsed_seconds"].mean()
        timing_by_size.to_csv(output_dir / "timing_by_size.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    methods = timing_summary.sort_values("mean").index.tolist()
    means = timing_summary.loc[methods, "mean"]
    stds = timing_summary.loc[methods, "std"]

    ax.barh(range(len(methods)), means, xerr=stds, capsize=3)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Computational Cost by Method")
    plt.tight_layout()
    plt.savefig(output_dir / "timing_comparison.png", dpi=300)
    plt.close()

    return timing_summary


def analyze_feature_selection_quality(
    results_df: pd.DataFrame,
    ground_truth: Dict[str, Any],
    output_dir: Path
) -> pd.DataFrame:
    """Analyze overall feature selection quality with Friedman/Nemenyi tests."""
    print("\n" + "=" * 60)
    print("FEATURE SELECTION QUALITY ANALYSIS")
    print("=" * 60)

    # Get synthetic datasets with ground truth
    synthetic_datasets = [name for name in ground_truth.keys()]

    if not synthetic_datasets:
        print("No synthetic datasets with ground truth found.")
        return pd.DataFrame()

    records = []

    for dataset in synthetic_datasets:
        if dataset not in ground_truth:
            continue

        meta = ground_truth[dataset]
        informative_indices = meta["informative_indices"]

        dataset_results = results_df[results_df["dataset"] == dataset]

        for _, row in dataset_results.iterrows():
            feature_ranks = [int(x) for x in row["feature_ranks"].split(",")]

            k = len(informative_indices)
            precision, recall = compute_precision_recall_at_k(
                feature_ranks, informative_indices, k
            )

            records.append({
                "dataset": dataset,
                "method": row["method"],
                "precision": precision,
                "recall": recall,
                "n_informative": k,
            })

    if not records:
        print("No results found for synthetic datasets.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Pivot for Friedman test
    methods = df["method"].unique().tolist()
    datasets = df["dataset"].unique().tolist()

    score_matrix = np.zeros((len(datasets), len(methods)))
    score_matrix[:] = np.nan

    for i, dataset in enumerate(datasets):
        for j, method in enumerate(methods):
            subset = df[(df["dataset"] == dataset) & (df["method"] == method)]
            if len(subset) > 0:
                score_matrix[i, j] = subset["precision"].mean()

    # Friedman test
    chi2, pvalue = friedman_test(score_matrix)
    print(f"\nFriedman test: χ² = {chi2:.2f}, p = {pvalue:.4f}")

    # Compute average ranks
    ranks = compute_ranks(score_matrix, higher_is_better=True)
    avg_ranks = np.nanmean(ranks, axis=0)

    rank_df = pd.DataFrame({
        "method": methods,
        "avg_rank": avg_ranks,
        "mean_precision": np.nanmean(score_matrix, axis=0),
    }).sort_values("avg_rank")

    # Critical difference
    n_valid_datasets = np.sum(~np.any(np.isnan(score_matrix), axis=1))
    cd = nemenyi_cd(len(methods), n_valid_datasets)
    rank_df["cd"] = cd

    print("\n=== METHOD RANKINGS ===")
    print(rank_df.round(3))

    rank_df.to_csv(output_dir / "method_rankings.csv", index=False)

    # Summary by method
    summary = df.groupby("method").agg({
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
    }).round(3)

    print("\n=== PRECISION/RECALL SUMMARY ===")
    print(summary)

    summary.to_csv(output_dir / "precision_recall_summary.csv")

    return df


def main():
    """Run comprehensive analysis."""
    print("=" * 60)
    print("CITREES COMPREHENSIVE ANALYSIS")
    print("=" * 60)

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    ground_truth = load_ground_truth()
    print(f"Loaded ground truth for {len(ground_truth)} synthetic datasets")

    # Load results (from parquet files in results directory)
    results_files = list(RESULTS_DIR.glob("*.parquet"))

    if not results_files:
        print("No results files found. Run experiments first.")
        print(f"Looking in: {RESULTS_DIR}")
        return

    # Combine all results
    dfs = []
    for f in results_files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
            print(f"Loaded {len(df)} records from {f.name}")
        except Exception as e:
            print(f"Error loading {f.name}: {e}")

    if not dfs:
        print("No valid results loaded.")
        return

    results_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal records: {len(results_df)}")

    # Run analyses
    analyze_feature_selection_quality(results_df, ground_truth, OUTPUT_DIR)
    analyze_selection_bias(results_df, ground_truth, OUTPUT_DIR)
    analyze_selector_comparison(results_df, ground_truth, OUTPUT_DIR)
    analyze_timing(results_df, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
