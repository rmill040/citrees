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
import io
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger

from paper.scripts.pipeline.grid import ExperimentGrid

OUTPUT_DIR = Path(__file__).parent.parent.parent / "results"


# =============================================================================
# Grid helpers
# =============================================================================


def _build_grid_config_keys(task: str) -> set[tuple[str, str, int]]:
    """Build the full set of expected (method_id, dataset, seed) tuples."""
    grid = ExperimentGrid.from_cli(task, source="all")
    return {cfg.key for cfg in grid}


def _method_id_from_filename(filename: str) -> str | None:
    """Extract method_id from a parquet filename like '{method_id}_seed{n}.parquet'."""
    stem = Path(filename).stem
    parts = stem.rsplit("_seed", 1)
    if len(parts) == 2:
        return parts[0]
    return None


def _seed_from_filename(filename: str) -> int | None:
    """Extract seed from a parquet filename like '{method_id}_seed{n}.parquet'."""
    stem = Path(filename).stem
    parts = stem.rsplit("_seed", 1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _artifact_key_from_path(path: Path) -> tuple[str, str, int] | None:
    """Extract (method_id, dataset, seed) from a local artifact path."""
    method_id = _method_id_from_filename(path.name)
    seed = _seed_from_filename(path.name)
    dataset = path.parent.name
    if method_id is None or seed is None or not dataset:
        return None
    return (method_id, dataset, seed)


def _artifact_key_from_s3_key(key: str) -> tuple[str, str, int] | None:
    """Extract (method_id, dataset, seed) from an S3 key."""
    parts = key.strip("/").split("/")
    if len(parts) < 4:
        return None
    method_id = _method_id_from_filename(parts[-1])
    seed = _seed_from_filename(parts[-1])
    dataset = parts[-2]
    if method_id is None or seed is None or not dataset:
        return None
    return (method_id, dataset, seed)


def _filter_files_to_grid(
    files: list[Path], grid_keys: set[tuple[str, str, int]]
) -> list[Path]:
    """Keep only files whose (method_id, dataset, seed) tuple is in the grid."""
    kept: list[Path] = []
    for f in files:
        artifact_key = _artifact_key_from_path(f)
        if artifact_key is not None and artifact_key in grid_keys:
            kept.append(f)
    return kept


def _filter_keys_to_grid(
    keys: list[str], grid_keys: set[tuple[str, str, int]]
) -> list[str]:
    """Keep only S3 keys whose (method_id, dataset, seed) tuple is in the grid."""
    kept: list[str] = []
    for key in keys:
        artifact_key = _artifact_key_from_s3_key(key)
        if artifact_key is not None and artifact_key in grid_keys:
            kept.append(key)
    return kept


def _validate(
    df: pd.DataFrame,
    grid_keys: set[tuple[str, str, int]],
    stage: str,
    task: str,
) -> pd.DataFrame:
    """Validate that all artifact keys in the DataFrame match the grid.

    Raises
    ------
    SystemExit
        If unexpected artifact keys are found.
    """
    key_col = "method_id" if "method_id" in df.columns else "method"
    data_keys = {
        (str(method_id), str(dataset), int(seed))
        for method_id, dataset, seed in df[[key_col, "dataset", "seed"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    unexpected = data_keys - grid_keys
    missing = grid_keys - data_keys

    if unexpected:
        sample = sorted(unexpected)[:10]
        logger.error(f"  UNEXPECTED artifact keys in {stage}/{task}: {sample}")
        logger.error("  Data contains dataset/config/seed tuples not in the current grid. Aborting.")
        sys.exit(1)

    if missing:
        logger.warning(f"  Missing artifact keys in {stage}/{task} ({len(missing)}):")
        for artifact_key in sorted(missing)[:20]:
            logger.warning(f"    {artifact_key}")
        if len(missing) > 20:
            logger.warning(f"    ... and {len(missing) - 20} more")

    logger.info(
        f"  Validated: {len(data_keys)} artifact keys, {len(missing)} missing from grid of {len(grid_keys)}"
    )
    return df


# =============================================================================
# Local aggregation
# =============================================================================


def aggregate_local(
    local_dir: Path,
    stage: Literal["rankings", "metrics"],
    task: Literal["classification", "regression"],
    output_path: Path,
    grid_keys: set[tuple[str, str, int]],
    *,
    dry_run: bool = False,
) -> int:
    """Aggregate local parquet files for a stage/task, filtered to the grid."""
    search_dir = local_dir / stage / task
    if not search_dir.exists():
        logger.warning(f"Directory not found: {search_dir}")
        return 0

    all_files = sorted(search_dir.rglob("*.parquet"))
    grid_files = _filter_files_to_grid(all_files, grid_keys)
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
    df = _validate(df, grid_keys, stage, task)

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
    grid_keys: set[tuple[str, str, int]],
    *,
    region_name: str | None = None,
    dry_run: bool = False,
) -> int:
    """Aggregate S3 files for a stage/task, filtered to the grid."""
    prefix = f"{stage}/{task}/"
    logger.info(f"Listing objects under s3://{bucket}/{prefix}")

    all_keys = list_s3_objects(bucket, prefix, region_name=region_name)
    matching_keys = _filter_keys_to_grid(all_keys, grid_keys)
    n_stale = len(all_keys) - len(matching_keys)
    logger.info(f"Found {len(all_keys)} files, {len(matching_keys)} match grid (filtered {n_stale} stale)")

    if dry_run:
        for key in matching_keys[:20]:
            logger.info(f"  {key}")
        if len(matching_keys) > 20:
            logger.info(f"  ... and {len(matching_keys) - 20} more")
        return len(matching_keys)

    if not matching_keys:
        logger.warning(f"No grid files found for {stage}/{task}")
        return 0

    logger.info(f"Downloading {len(matching_keys)} files...")
    df = download_and_concat(bucket, matching_keys, region_name=region_name)

    if df.empty:
        logger.warning("No data after concatenation")
        return 0

    df = _validate(df, grid_keys, stage, task)

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
        grid_keys = _build_grid_config_keys(task)
        logger.info(f"Grid for {task}: {len(grid_keys)} artifact keys")

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
                    grid_keys=grid_keys,
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
                    grid_keys=grid_keys,
                    region_name=config.aws_region,
                    dry_run=args.dry_run,
                )
            total_files += n_files

    logger.info(f"\nTotal files processed: {total_files}")


if __name__ == "__main__":
    main()
