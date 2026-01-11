"""Classifier metrics - ANALYSIS.

This script processes experiment results from DynamoDB exports and generates
summary statistics and visualizations for the paper.

Usage:
    # Format raw DynamoDB data
    DATA_DIR=/path/to/data GET_DATA=1 python clf_cv_analysis.py

    # Run analysis on formatted data
    DATA_DIR=/path/to/data python clf_cv_analysis.py
"""

import json
import os
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from boto3.dynamodb.types import TypeDeserializer
from loguru import logger

DATA_DIR = Path(os.environ.get("DATA_DIR", ".")).resolve()


class DecimalEncoder(json.JSONEncoder):
    """Handle decimal data for JSON serialization."""

    def default(self, obj: Any) -> str:
        """Cast decimal types to string."""
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)


def format_raw_data() -> None:
    """Format raw data dump from DynamoDB and save as CSV files."""
    deserde = TypeDeserializer()
    results = defaultdict(list)
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]

    for file in files:
        logger.info(f"Processing file ({file})")
        with open(DATA_DIR / file) as f:
            for j, line in enumerate(f, 1):
                if j % 25_000 == 0:
                    logger.info(f"Processing row ({j})")
                row = json.loads(line)["Item"]
                row.pop("feature_ranks", None)
                for key, value in row.items():
                    row[key] = deserde.deserialize(value)
                row = json.loads(json.dumps(row, cls=DecimalEncoder))
                for key, value in row.get("metrics", {}).items():
                    if key == "feature_ranks":
                        continue
                    dtype = int if key == "n_features_used" else float
                    row["metrics"][key] = list(map(dtype, value))
                results[row["method"]].append(row)

    total = sum([len(results[key]) for key in results.keys()])
    logger.info(f"{total} total configurations processed for feature selection")

    keys = list(results.keys())
    for key in keys:
        logger.info(f"Writing dataset ({key}) to disk")
        df = pd.json_normalize(results.pop(key)).fillna("None")
        df.to_csv(DATA_DIR / (key + ".csv"), index=False)


def load_results() -> dict[str, pd.DataFrame]:
    """Load all CSV result files into a dictionary of DataFrames."""
    results = {}
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

    for file in csv_files:
        method = file.replace(".csv", "")
        logger.info(f"Loading {method}")
        results[method] = pd.read_csv(DATA_DIR / file)

    return results


def compute_summary_statistics(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute summary statistics across methods and datasets."""
    summaries = []

    for method, df in results.items():
        for dataset in df["dataset"].unique():
            subset = df[df["dataset"] == dataset]

            # Get metrics at different feature counts
            for metric in ["accuracy_mean", "f1_mean", "auc_mean"]:
                if metric not in subset.columns:
                    continue

                # Find best performance across hyperparameters
                best_idx = subset[metric].idxmax()
                best_row = subset.loc[best_idx]

                summaries.append(
                    {
                        "method": method,
                        "dataset": dataset,
                        "metric": metric.replace("_mean", ""),
                        "best_value": best_row[metric],
                        "best_hyperparameters": str(best_row.get("hyperparameters", {})),
                    }
                )

    return pd.DataFrame(summaries)


def compute_ranking_table(
    results: dict[str, pd.DataFrame], metric: str = "accuracy"
) -> pd.DataFrame:
    """Compute method rankings per dataset."""
    rankings = []

    datasets = set()
    for df in results.values():
        datasets.update(df["dataset"].unique())

    for dataset in datasets:
        dataset_results = []
        for method, df in results.items():
            subset = df[df["dataset"] == dataset]
            if len(subset) == 0:
                continue

            metric_col = f"{metric}_mean"
            if metric_col not in subset.columns:
                continue

            best_value = subset[metric_col].max()
            dataset_results.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "value": best_value,
                }
            )

        # Rank methods for this dataset
        dataset_df = pd.DataFrame(dataset_results)
        if len(dataset_df) > 0:
            dataset_df["rank"] = dataset_df["value"].rank(ascending=False)
            rankings.extend(dataset_df.to_dict("records"))

    return pd.DataFrame(rankings)


def print_summary(results: dict[str, pd.DataFrame]) -> None:
    """Print summary of results to console."""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print(f"\nMethods: {list(results.keys())}")
    print(f"Total methods: {len(results)}")

    for method, df in results.items():
        print(f"\n--- {method} ---")
        print(f"  Configurations: {len(df)}")
        print(f"  Datasets: {df['dataset'].nunique()}")

        for metric in ["accuracy_mean", "f1_mean", "auc_mean"]:
            if metric in df.columns:
                print(f"  {metric}: {df[metric].mean():.4f} ± {df[metric].std():.4f}")


def run_analysis() -> None:
    """Run the full analysis pipeline."""
    logger.info("Loading results...")
    results = load_results()

    if not results:
        logger.warning("No CSV files found. Run with GET_DATA=1 first to format raw data.")
        return

    print_summary(results)

    logger.info("Computing summary statistics...")
    summary_df = compute_summary_statistics(results)
    summary_path = DATA_DIR / "summary_statistics.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")

    logger.info("Computing rankings...")
    ranking_df = compute_ranking_table(results, metric="accuracy")
    ranking_path = DATA_DIR / "method_rankings.csv"
    ranking_df.to_csv(ranking_path, index=False)
    logger.info(f"Rankings saved to {ranking_path}")

    # Print average rank per method
    print("\n" + "=" * 60)
    print("AVERAGE RANK BY METHOD (lower is better)")
    print("=" * 60)
    avg_ranks = ranking_df.groupby("method")["rank"].mean().sort_values()
    print(avg_ranks.round(2))


if __name__ == "__main__":
    if bool(os.environ.get("GET_DATA")):
        format_raw_data()
    else:
        run_analysis()
