"""Download and aggregate experiment artifacts to local parquet files.

This script:
1. Builds the method_id set from paper/scripts/pipeline/config.py
2. Filters files to ONLY method_ids in the current grid (rejects stale configs)
3. Concatenates to canonical output files for analysis
4. Runs sanity checks: method count, missing configs, unexpected IDs

Usage:
    # From local directory (primary path):
    uv run python paper/scripts/analysis/aggregate.py \
        --local-dir ../data --task all

    # From S3:
    S3_BUCKET=my-bucket uv run python paper/scripts/analysis/aggregate.py \
        --task classification

    # Dry run (list files without downloading)
    uv run python paper/scripts/analysis/aggregate.py \
        --local-dir ../data --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger

from paper.scripts.pipeline.config import get_configs

OUTPUT_DIR = Path(__file__).parent.parent.parent / "results"


# =============================================================================
# Grid helpers
# =============================================================================


def _build_grid_method_ids(task: str) -> set[str]:
    """Build the set of method_ids from the current config grid."""
    grid = get_configs(task)
    ids: set[str] = set()
    for method_name, config_list in grid.items():
        for params in config_list:
            p = {k: v for k, v in sorted(params.items()) if k not in ("method", "random_state")}
            h = hashlib.md5(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[:16]
            ids.add(f"{method_name}__{h}")
    return ids


def _method_id_from_filename(filename: str) -> str | None:
    """Extract method_id from a parquet filename like '{method_id}_seed{n}.parquet'."""
    stem = Path(filename).stem
    parts = stem.rsplit("_seed", 1)
    if len(parts) == 2:
        return parts[0]
    return None


def _filter_files_to_grid(files: list[Path], grid_ids: set[str]) -> list[Path]:
    """Keep only files whose method_id is in the grid."""
    kept: list[Path] = []
    for f in files:
        mid = _method_id_from_filename(f.name)
        if mid is not None and mid in grid_ids:
            kept.append(f)
    return kept


def _filter_keys_to_grid(keys: list[str], grid_ids: set[str]) -> list[str]:
    """Keep only S3 keys whose method_id is in the grid."""
    kept: list[str] = []
    for key in keys:
        filename = key.rsplit("/", 1)[-1]
        mid = _method_id_from_filename(filename)
        if mid is not None and mid in grid_ids:
            kept.append(key)
    return kept


def _validate(
    df: pd.DataFrame,
    grid_ids: set[str],
    stage: str,
    task: str,
) -> pd.DataFrame:
    """Validate that all method IDs in the DataFrame match the grid.

    Raises
    ------
    SystemExit
        If unexpected method_ids are found.
    """
    id_col = "method_id" if "method_id" in df.columns else "method"

    data_ids = set(df[id_col].unique())
    unexpected = data_ids - grid_ids
    missing = grid_ids - data_ids

    if unexpected:
        logger.error(f"  UNEXPECTED method_ids in {stage}/{task}: {unexpected}")
        logger.error("  Data contains configs not in the current grid. Aborting.")
        sys.exit(1)

    if missing:
        logger.warning(f"  Missing method_ids in {stage}/{task} ({len(missing)}):")
        for mid in sorted(missing):
            logger.warning(f"    {mid}")

    logger.info(f"  Validated: {len(data_ids)} method_ids, {len(missing)} missing from grid of {len(grid_ids)}")
    return df


# =============================================================================
# Local aggregation
# =============================================================================


def aggregate_local(
    local_dir: Path,
    stage: Literal["rankings", "metrics"],
    task: Literal["classification", "regression"],
    output_path: Path,
    grid_ids: set[str],
    *,
    dry_run: bool = False,
) -> int:
    """Aggregate local parquet files for a stage/task, filtered to the grid."""
    search_dir = local_dir / stage / task
    if not search_dir.exists():
        logger.warning(f"Directory not found: {search_dir}")
        return 0

    all_files = sorted(search_dir.rglob("*.parquet"))
    grid_files = _filter_files_to_grid(all_files, grid_ids)
    n_stale = len(all_files) - len(grid_files)
    logger.info(f"Found {len(all_files)} files, {len(grid_files)} match grid (filtered {n_stale} stale)")

    if dry_run:
        for f in grid_files[:20]:
            logger.info(f"  {f.relative_to(local_dir)}")
        if len(grid_files) > 20:
            logger.info(f"  ... and {len(grid_files) - 20} more")
        return len(grid_files)

    if not grid_files:
        logger.warning(f"No grid files found for {stage}/{task}")
        return 0

    logger.info(f"Reading {len(grid_files)} files...")
    dfs: list[pd.DataFrame] = []
    errors = 0
    for i, f in enumerate(grid_files):
        if (i + 1) % 500 == 0:
            logger.info(f"  Read {i + 1}/{len(grid_files)} files...")
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            errors += 1
            logger.warning(f"Failed to read {f}: {e}")

    if errors:
        logger.warning(f"  {errors} read errors")

    if not dfs:
        logger.warning("No data after concatenation")
        return 0

    df = pd.concat(dfs, ignore_index=True)
    df = _validate(df, grid_ids, stage, task)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Wrote {output_path} ({len(df):,} rows, {len(grid_files)} files)")

    return len(grid_files)


# =============================================================================
# S3 aggregation
# =============================================================================


def list_s3_objects(
    bucket: str,
    prefix: str,
    *,
    region_name: str | None = None,
) -> list[str]:
    """List all object keys under a prefix."""
    from paper.scripts.adapters import get_s3_client

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

    from paper.scripts.adapters import get_s3_client

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
    task: Literal["classification", "regression"],
    output_path: Path,
    grid_ids: set[str],
    *,
    region_name: str | None = None,
    dry_run: bool = False,
) -> int:
    """Aggregate S3 files for a stage/task, filtered to the grid."""
    prefix = f"{stage}/{task}/"
    logger.info(f"Listing objects under s3://{bucket}/{prefix}")

    all_keys = list_s3_objects(bucket, prefix, region_name=region_name)
    grid_keys = _filter_keys_to_grid(all_keys, grid_ids)
    n_stale = len(all_keys) - len(grid_keys)
    logger.info(f"Found {len(all_keys)} files, {len(grid_keys)} match grid (filtered {n_stale} stale)")

    if dry_run:
        for key in grid_keys[:20]:
            logger.info(f"  {key}")
        if len(grid_keys) > 20:
            logger.info(f"  ... and {len(grid_keys) - 20} more")
        return len(grid_keys)

    if not grid_keys:
        logger.warning(f"No grid files found for {stage}/{task}")
        return 0

    logger.info(f"Downloading {len(grid_keys)} files...")
    df = download_and_concat(bucket, grid_keys, region_name=region_name)

    if df.empty:
        logger.warning("No data after concatenation")
        return 0

    df = _validate(df, grid_ids, stage, task)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Wrote {output_path} ({len(df):,} rows)")

    return len(grid_keys)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Download and aggregate experiment artifacts (grid-filtered)")
    parser.add_argument(
        "--stage",
        choices=["rankings", "metrics", "all"],
        default="all",
        help="Which stage to aggregate (default: all)",
    )
    parser.add_argument(
        "--task",
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
        "--local-dir",
        type=Path,
        default=None,
        help="Local directory with rankings/ and metrics/ subdirs (skip S3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without downloading/reading",
    )
    args = parser.parse_args()

    stages: list[Literal["rankings", "metrics"]] = (
        ["rankings", "metrics"] if args.stage == "all" else [args.stage]
    )
    tasks: list[Literal["classification", "regression"]] = (
        ["classification", "regression"] if args.task == "all" else [args.task]
    )

    output_files = {
        ("rankings", "classification"): "clf_rankings.parquet",
        ("rankings", "regression"): "reg_rankings.parquet",
        ("metrics", "classification"): "clf_evaluation.parquet",
        ("metrics", "regression"): "reg_evaluation.parquet",
    }

    total_files = 0

    for task in tasks:
        grid_ids = _build_grid_method_ids(task)
        logger.info(f"Grid for {task}: {len(grid_ids)} method_ids")

        for stage in stages:
            output_path = args.output_dir / output_files[(stage, task)]
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing {stage}/{task} -> {output_path.name}")
            logger.info(f"{'=' * 60}")

            if args.local_dir is not None:
                n_files = aggregate_local(
                    local_dir=args.local_dir.resolve(),
                    stage=stage,
                    task=task,
                    output_path=output_path,
                    grid_ids=grid_ids,
                    dry_run=args.dry_run,
                )
            else:
                from paper.scripts.adapters import get_s3_bucket
                from paper.scripts.config import load_config

                config = load_config()
                bucket = get_s3_bucket()

                n_files = aggregate_stage(
                    bucket=bucket,
                    stage=stage,
                    task=task,
                    output_path=output_path,
                    grid_ids=grid_ids,
                    region_name=config.aws_region,
                    dry_run=args.dry_run,
                )
            total_files += n_files

    logger.info(f"\nTotal files processed: {total_files}")


if __name__ == "__main__":
    main()
