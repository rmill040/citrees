#!/usr/bin/env python3
"""Analyze synthetic experiment results using ground truth.

Computes precision/recall/F1@k for synthetic datasets with known ground truth
informative indices. Also reports informative+redundant variants and
confounder selection rates when available in metadata.

Usage:
    # After running experiments and downloading results from S3:
    uv run python paper/scripts/analysis/synthetic_analysis.py \
        --results-dir paper/results/rankings/classification \
        --data-dir paper/data/classification/synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from paper.scripts.utils.metrics import f1_at_k, precision_at_k, recall_at_k


# =============================================================================
# Metadata Loading
# =============================================================================


def dataset_type_from_config(config: dict) -> str:
    """Map synthetic config to a dataset type label."""
    if config.get("toeplitz_rho", 0.0) > 0:
        return "toeplitz"
    if config.get("weak_signal", False):
        return "weak_signal"
    if config.get("nonlinear", False):
        return "nonlinear"
    if config.get("n_high_cardinality_noise", 0) > 0:
        return "bias"
    if config.get("n_correlated_noise", 0) > 0:
        return "confounder"
    if config.get("n_correlated_blocks", 0) > 0:
        return "correlated"
    if config.get("n_redundant", 0) > 0:
        return "redundant"
    return "standard"


def load_synthetic_metadata(data_dir: Path) -> dict[str, dict]:
    """Load ground truth from all synthetic datasets.

    Parameters
    ----------
    data_dir : Path
        Directory containing synthetic parquet files.

    Returns
    -------
    dict[str, dict]
        Mapping from dataset name to metadata (config, informative_indices, etc.).
    """
    metadata = {}

    for filepath in data_dir.glob("clf_synthetic_*.parquet"):
        table = pq.read_table(filepath, columns=[])  # Just read schema
        schema_meta = table.schema.metadata

        if schema_meta and schema_meta.get(b"synthetic") == b"true":
            name = filepath.stem.replace("clf_", "")
            metadata[name] = {
                "config": json.loads(schema_meta[b"config"]),
                "informative_indices": json.loads(schema_meta[b"informative_indices"]),
                "redundant_indices": json.loads(schema_meta.get(b"redundant_indices", b"[]")),
                "noise_indices": json.loads(schema_meta.get(b"noise_indices", b"[]")),
                "correlated_indices": json.loads(schema_meta.get(b"correlated_indices", b"[]")),
                "correlated_noise_indices": json.loads(schema_meta.get(b"correlated_noise_indices", b"[]")),
                "n_features_final": int(schema_meta.get(b"n_features_final", b"0")),
            }

    return metadata


# =============================================================================
# Results Analysis
# =============================================================================


def analyze_results(
    results_dir: Path,
    data_dir: Path,
    k_values: list[int] | None = None,
) -> pd.DataFrame:
    """Analyze synthetic experiment results.

    Parameters
    ----------
    results_dir : Path
        Directory containing ranking results (parquet files from S3).
    data_dir : Path
        Directory containing synthetic datasets with ground truth.
    k_values : list[int] | None
        Values of k for precision/recall@k. Default: [5, 10, 20, k_informative].

    Returns
    -------
    pd.DataFrame
        Analysis results with precision/recall@k for each method and dataset.
    """
    # Load ground truth
    metadata = load_synthetic_metadata(data_dir)
    logger.info(f"Loaded metadata for {len(metadata)} synthetic datasets")

    if not metadata:
        logger.warning("No synthetic datasets found!")
        return pd.DataFrame()

    results = []

    # Find ranking files
    for ranking_file in results_dir.glob("**/*.parquet"):
        # Parse filename: {method}_seed{seed}.parquet
        parts = ranking_file.stem.rsplit("_seed", 1)
        if len(parts) != 2:
            continue

        method = parts[0]
        seed = int(parts[1])

        # Dataset name from parent directory
        dataset = ranking_file.parent.name

        # Check if this is a synthetic dataset
        if not dataset.startswith("synthetic_"):
            continue

        if dataset not in metadata:
            logger.warning(f"No metadata for {dataset}")
            continue

        meta = metadata[dataset]
        true_indices = meta["informative_indices"]
        redundant_indices = meta.get("redundant_indices", [])
        config = meta["config"]
        informative_plus_redundant = sorted(set(true_indices) | set(redundant_indices))
        confounder_indices = meta.get("correlated_noise_indices") or meta.get("correlated_indices", [])
        n_features_final = meta.get("n_features_final", 0)

        # Determine k values
        n_informative = config["n_informative"]
        ks = k_values or [5, 10, 20, n_informative]

        # Load rankings
        try:
            df = pd.read_parquet(ranking_file)
        except Exception as e:
            logger.error(f"Error reading {ranking_file}: {e}")
            continue

        # Process each fold
        for _, row in df.iterrows():
            ranking = row["feature_ranking"]
            fold_idx = row["fold_idx"]

            row_data = {
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "fold_idx": fold_idx,
                # Config params for grouping
                "n_features": config["n_features"],
                "n_features_final": n_features_final,
                "n_informative": config["n_informative"],
                "n_samples": config["n_samples"],
                "class_sep": config.get("class_sep", 1.0),
                "flip_y": config.get("flip_y", 0.0),
                "nonlinear": config.get("nonlinear", False),
                "n_high_cardinality_noise": config.get("n_high_cardinality_noise", 0),
                "n_correlated_blocks": config.get("n_correlated_blocks", 0),
                "n_correlated_noise": config.get("n_correlated_noise", 0),
                "n_redundant": config.get("n_redundant", 0),
                "toeplitz_rho": config.get("toeplitz_rho", 0.0),
                "weak_signal": config.get("weak_signal", False),
                "dataset_type": dataset_type_from_config(config),
            }

            # Compute metrics for each k
            for k in ks:
                row_data[f"precision@{k}"] = precision_at_k(ranking, true_indices, k)
                row_data[f"recall@{k}"] = recall_at_k(ranking, true_indices, k)
                row_data[f"f1@{k}"] = f1_at_k(ranking, true_indices, k)
                row_data[f"precision_ir@{k}"] = precision_at_k(
                    ranking, informative_plus_redundant, k
                )
                row_data[f"recall_ir@{k}"] = recall_at_k(
                    ranking, informative_plus_redundant, k
                )
                row_data[f"f1_ir@{k}"] = f1_at_k(ranking, informative_plus_redundant, k)
                row_data[f"confounder_rate@{k}"] = precision_at_k(
                    ranking, confounder_indices, k
                )

            results.append(row_data)

    return pd.DataFrame(results)


def summarize_by_method(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize results by method.

    Parameters
    ----------
    df : pd.DataFrame
        Full results from analyze_results().

    Returns
    -------
    pd.DataFrame
        Summary with mean precision/recall@k per method.
    """
    if df.empty:
        return df

    # Find metric columns
    metric_cols = [
        c
        for c in df.columns
        if c.startswith(
            (
                "precision@",
                "recall@",
                "f1@",
                "f1@",
                "precision_ir@",
                "recall_ir@",
                "f1_ir@",
                "confounder_rate@",
            )
        )
    ]

    # Group by method and aggregate
    summary = df.groupby("method")[metric_cols].agg(["mean", "std"])

    # Flatten column names
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]

    return summary.sort_values(f"precision@{df['n_informative'].mode().iloc[0]}_mean", ascending=False)


def summarize_by_dataset_type(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize results by dataset type (standard, bias, nonlinear, etc.)."""
    if df.empty:
        return df

    # Categorize datasets
    df = df.copy()
    if "dataset_type" not in df.columns:
        df["dataset_type"] = df.apply(lambda row: dataset_type_from_config(row), axis=1)

    metric_cols = [
        c
        for c in df.columns
        if c.startswith(
            (
                "precision@",
                "recall@",
                "precision_ir@",
                "recall_ir@",
                "f1_ir@",
                "confounder_rate@",
            )
        )
    ]

    return df.groupby(["dataset_type", "method"])[metric_cols].mean()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run synthetic analysis."""
    parser = argparse.ArgumentParser(description="Analyze synthetic experiment results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("paper/results/rankings/classification"),
        help="Directory with ranking results",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("paper/data/classification/synthetic"),
        help="Directory with synthetic datasets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/results/synthetic_analysis.parquet"),
        help="Output file for full results",
    )
    args = parser.parse_args()

    logger.info(f"Analyzing results from {args.results_dir}")

    df = analyze_results(args.results_dir, args.data_dir)

    if df.empty:
        logger.warning("No results to analyze!")
        return

    logger.info(f"Analyzed {len(df)} result rows")

    # Save full results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output)
    logger.info(f"Saved results to {args.output}")

    # Print summaries
    print("\n=== SUMMARY BY METHOD ===")
    print(summarize_by_method(df).to_string())

    print("\n=== SUMMARY BY DATASET TYPE ===")
    print(summarize_by_dataset_type(df).to_string())


if __name__ == "__main__":
    main()
