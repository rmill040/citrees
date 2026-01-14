#!/usr/bin/env python3
"""Quick progress checker using S3 listing (no DynamoDB scan needed).

Usage:
    # Check feature selection progress (Stage 1)
    AWS_PROFILE=personal uv run python paper/scripts/check_progress.py --stage rankings

    # Check downstream eval progress (Stage 2)
    AWS_PROFILE=personal uv run python paper/scripts/check_progress.py --stage metrics

    # Show progress by method
    AWS_PROFILE=personal uv run python paper/scripts/check_progress.py --stage rankings --by-method

    # Show progress by dataset
    AWS_PROFILE=personal uv run python paper/scripts/check_progress.py --stage rankings --by-dataset
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import boto3
from loguru import logger

from paper.scripts.constants import CLF_METHODS, N_SEEDS, REG_METHODS, S3_BUCKET, AWS_REGION


def get_datasets(task_type: str = "classification") -> list[str]:
    """Get dataset names from paper/data directory."""
    data_dir = Path(__file__).resolve().parents[1] / "data"
    prefix = "clf_" if task_type == "classification" else "reg_"
    datasets = []
    for f in data_dir.glob(f"{prefix}*.parquet"):
        name = f.stem.replace(prefix, "").replace(".snappy", "")
        datasets.append(name)
    return sorted(datasets)


def list_s3_completed(stage: str, task_type: str = "classification") -> dict[str, set[tuple[str, int]]]:
    """List completed items from S3, grouped by dataset.

    Returns dict: dataset -> set of (method, seed) tuples
    """
    s3 = boto3.client("s3", region_name=AWS_REGION)
    completed: dict[str, set[tuple[str, int]]] = defaultdict(set)
    prefix = f"{stage}/{task_type}/"

    logger.info(f"Listing s3://{S3_BUCKET}/{prefix}...")

    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            parts = key.split("/")
            if len(parts) >= 4:
                dataset = parts[2]
                filename = parts[3]
                method_seed = filename.replace(".parquet", "")
                if "_seed" in method_seed:
                    method, seed_str = method_seed.rsplit("_seed", 1)
                    seed = int(seed_str)
                    completed[dataset].add((method, seed))
                    count += 1

    logger.info(f"Found {count} completed items across {len(completed)} datasets")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Check experiment progress from S3")
    parser.add_argument(
        "--stage",
        choices=["rankings", "metrics"],
        default="rankings",
        help="Which stage to check (rankings=Stage1, metrics=Stage2)",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
        help="Task type",
    )
    parser.add_argument(
        "--by-method",
        action="store_true",
        help="Show progress grouped by method",
    )
    parser.add_argument(
        "--by-dataset",
        action="store_true",
        help="Show progress grouped by dataset",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Only show synthetic datasets",
    )
    args = parser.parse_args()

    # Get expected datasets and methods
    datasets = get_datasets(args.task)
    methods = CLF_METHODS if args.task == "classification" else REG_METHODS

    if args.synthetic_only:
        datasets = [d for d in datasets if d.startswith("synthetic_")]

    total_expected = len(datasets) * len(methods) * N_SEEDS
    logger.info(f"Expected: {len(datasets)} datasets × {len(methods)} methods × {N_SEEDS} seeds = {total_expected}")

    # Get completed from S3
    completed = list_s3_completed(args.stage, args.task)

    # Flatten to total count
    total_completed = sum(len(items) for items in completed.values())
    pct = 100 * total_completed / total_expected if total_expected > 0 else 0

    print(f"\n{'='*60}")
    print(f"PROGRESS: {args.stage.upper()} ({args.task})")
    print(f"{'='*60}")
    print(f"Completed: {total_completed:,} / {total_expected:,} ({pct:.1f}%)")
    print(f"Remaining: {total_expected - total_completed:,}")

    if args.by_method:
        print(f"\n{'='*60}")
        print("BY METHOD:")
        print(f"{'='*60}")
        method_counts: dict[str, int] = defaultdict(int)
        for dataset_items in completed.values():
            for method, _ in dataset_items:
                method_counts[method] += 1

        expected_per_method = len(datasets) * N_SEEDS
        for method in methods:
            count = method_counts.get(method, 0)
            pct = 100 * count / expected_per_method if expected_per_method > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {method:12} {bar} {count:5} / {expected_per_method} ({pct:5.1f}%)")

    if args.by_dataset:
        print(f"\n{'='*60}")
        print("BY DATASET:")
        print(f"{'='*60}")
        expected_per_dataset = len(methods) * N_SEEDS

        # Show incomplete datasets first
        incomplete = []
        complete = []
        for dataset in datasets:
            count = len(completed.get(dataset, set()))
            if count < expected_per_dataset:
                incomplete.append((dataset, count))
            else:
                complete.append((dataset, count))

        if incomplete:
            print("\nINCOMPLETE:")
            for dataset, count in sorted(incomplete, key=lambda x: x[1]):
                pct = 100 * count / expected_per_dataset
                print(f"  {dataset:50} {count:4} / {expected_per_dataset} ({pct:5.1f}%)")

        print(f"\nCOMPLETE: {len(complete)} datasets")
        if len(complete) <= 10:
            for dataset, count in complete:
                print(f"  ✓ {dataset}")


if __name__ == "__main__":
    main()
