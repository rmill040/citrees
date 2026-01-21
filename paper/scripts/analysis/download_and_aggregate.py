"""Download and aggregate S3 experiment artifacts to local parquet files.

This script:
1. Lists S3 keys under rankings/ and metrics/ prefixes
2. Downloads parquet files locally
3. Concatenates to canonical output files for analysis

Usage:
    # Download both rankings and metrics for classification
    S3_BUCKET=my-bucket uv run python paper/scripts/analysis/download_and_aggregate.py \
        --task-type classification

    # Dry run (list files without downloading)
    S3_BUCKET=my-bucket uv run python paper/scripts/analysis/download_and_aggregate.py --dry-run

    # Download only metrics
    S3_BUCKET=my-bucket uv run python paper/scripts/analysis/download_and_aggregate.py --stage metrics
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Literal

import boto3
import pandas as pd
from loguru import logger

from paper.scripts.experiments._common import get_s3_bucket, get_s3_client
from paper.scripts.infra.config import load_config

OUTPUT_DIR = Path(__file__).parent.parent.parent / "results"


def list_s3_objects(
    bucket: str,
    prefix: str,
    *,
    region_name: str | None = None,
) -> list[str]:
    """List all object keys under a prefix."""
    client = get_s3_client(region_name=region_name)
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key and key.endswith(".parquet"):
                keys.append(key)

    return keys


def download_and_concat(
    bucket: str,
    keys: list[str],
    *,
    region_name: str | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Download parquet files from S3 and concatenate into a single DataFrame."""
    if not keys:
        return pd.DataFrame()

    client = get_s3_client(region_name=region_name)
    dfs: list[pd.DataFrame] = []

    for i, key in enumerate(keys):
        if show_progress and (i + 1) % 100 == 0:
            logger.info(f"  Downloaded {i + 1}/{len(keys)} files...")

        try:
            response = client.get_object(Bucket=bucket, Key=key)
            df = pd.read_parquet(io.BytesIO(response["Body"].read()))
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to download {key}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def aggregate_stage(
    bucket: str,
    stage: Literal["rankings", "metrics"],
    task_type: Literal["classification", "regression"],
    output_path: Path,
    *,
    region_name: str | None = None,
    dry_run: bool = False,
) -> int:
    """Aggregate all S3 files for a stage/task_type to a single parquet file.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    stage : {"rankings", "metrics"}
        Which stage to aggregate.
    task_type : {"classification", "regression"}
        Task type filter.
    output_path : Path
        Where to write the aggregated parquet file.
    region_name : str, optional
        AWS region.
    dry_run : bool
        If True, only list files without downloading.

    Returns
    -------
    int
        Number of files processed.
    """
    prefix = f"{stage}/{task_type}/"
    logger.info(f"Listing objects under s3://{bucket}/{prefix}")

    keys = list_s3_objects(bucket, prefix, region_name=region_name)
    logger.info(f"Found {len(keys)} parquet files")

    if dry_run:
        for key in keys[:20]:
            logger.info(f"  {key}")
        if len(keys) > 20:
            logger.info(f"  ... and {len(keys) - 20} more")
        return len(keys)

    if not keys:
        logger.warning(f"No files found for {stage}/{task_type}")
        return 0

    logger.info(f"Downloading and concatenating {len(keys)} files...")
    df = download_and_concat(bucket, keys, region_name=region_name)

    if df.empty:
        logger.warning("No data after concatenation")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Wrote {output_path} ({len(df)} rows)")

    return len(keys)


def main():
    parser = argparse.ArgumentParser(
        description="Download and aggregate S3 experiment artifacts"
    )
    parser.add_argument(
        "--stage",
        choices=["rankings", "metrics", "all"],
        default="all",
        help="Which stage to aggregate (default: all)",
    )
    parser.add_argument(
        "--task-type",
        choices=["classification", "regression", "all"],
        default="all",
        help="Task type filter (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for aggregated files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without downloading",
    )
    args = parser.parse_args()

    config = load_config()
    bucket = get_s3_bucket()
    region_name = config.region

    logger.info(f"S3 bucket: {bucket}")
    logger.info(f"Region: {region_name}")
    logger.info(f"Output dir: {args.output_dir}")

    # Determine stages and task types to process
    stages: list[Literal["rankings", "metrics"]] = (
        ["rankings", "metrics"] if args.stage == "all" else [args.stage]
    )
    task_types: list[Literal["classification", "regression"]] = (
        ["classification", "regression"] if args.task_type == "all" else [args.task_type]
    )

    # Output file naming convention
    output_files = {
        ("rankings", "classification"): "clf_rankings.parquet",
        ("rankings", "regression"): "reg_rankings.parquet",
        ("metrics", "classification"): "clf_evaluation.parquet",
        ("metrics", "regression"): "reg_evaluation.parquet",
    }

    total_files = 0
    for stage in stages:
        for task_type in task_types:
            output_path = args.output_dir / output_files[(stage, task_type)]
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {stage}/{task_type} -> {output_path.name}")
            logger.info(f"{'='*60}")

            n_files = aggregate_stage(
                bucket=bucket,
                stage=stage,
                task_type=task_type,
                output_path=output_path,
                region_name=region_name,
                dry_run=args.dry_run,
            )
            total_files += n_files

    logger.info(f"\nTotal files processed: {total_files}")


if __name__ == "__main__":
    main()
