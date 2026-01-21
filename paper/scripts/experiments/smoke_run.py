"""Deterministic end-to-end smoke run for the Ray experiments pipeline.

This script runs a minimal subset of the grid:
Stage 1 (rankings) → Stage 2 (metrics), then downloads the resulting artifacts from S3 and asserts their schemas.

Use this before scaling to the full grid to verify:
- Ray scheduling works (selection/evaluation resources + CPU reservations)
- S3 read/write paths are correct
- Artifact schemas are self-describing (no inference needed)
- Provenance is populated (`git_sha` is not "unknown")
"""

from __future__ import annotations

import argparse
from typing import Literal, cast

import ray
from loguru import logger

from paper.scripts.experiments import ray_eval, ray_feature_selection
from paper.scripts.experiments._common import (
    download_parquet_from_s3,
    get_dataset_shape,
    get_datasets,
    get_s3_bucket,
    metrics_s3_path,
    parse_csv_list,
)
from paper.scripts.experiments._common import rankings_s3_path as rankings_s3_key
from paper.scripts.infra.config import load_config
from paper.scripts.utils.constants import N_SPLITS
from paper.scripts.utils.experiment_configs import config_label, expand_method_configs

DataSource = Literal["real", "synthetic", "all"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke run: Stage 1 → Stage 2 with assertions")
    parser.add_argument(
        "--ray-address",
        default="auto",
        help="Ray address (default: auto). Use 'local' for local mode.",
    )
    parser.add_argument("--task-type", choices=["classification", "regression"], default=None)
    parser.add_argument(
        "--source",
        choices=["real", "synthetic", "all"],
        default="real",
        help="Dataset source filter",
    )
    parser.add_argument(
        "--dataset", default=None, help="Dataset name (default: choose smallest in source)"
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated base selection methods (default: mc,rf for clf; pc,rf for reg)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed index (default: 0)")
    return parser.parse_args()


def _choose_dataset(task_type: str, source: DataSource, preferred: str | None) -> str:
    datasets = get_datasets(task_type, source=source)
    if not datasets:
        raise ValueError(f"No datasets found for task_type={task_type!r}, source={source!r}")

    if preferred is not None:
        if preferred not in datasets:
            raise ValueError(
                f"Dataset {preferred!r} not found in {datasets[:10]}{'...' if len(datasets) > 10 else ''}"
            )
        return preferred

    # Deterministic: pick smallest dataset by (n_samples * n_features), tie-break by name.
    shapes = {d: get_dataset_shape(d, task_type) for d in datasets}
    return sorted(datasets, key=lambda d: (shapes[d][0] * shapes[d][1], d))[0]


def _assert_rankings_schema(
    df, *, dataset: str, task_type: str, seed: int, method_id: str, method_base: str
) -> None:
    required = {
        "fold_idx",
        "feature_ranking",
        "dataset",
        "task_type",
        "seed",
        "method_id",
        "method",
        "method_base",
        "artifact_version",
        "n_samples",
        "n_features",
        "selection_cpus",
        "elapsed_seconds",
        "created_at_utc",
        "git_sha",
    }
    missing = required - set(df.columns)
    if missing:
        raise AssertionError(f"Rankings missing columns: {sorted(missing)}")

    if df["dataset"].nunique() != 1 or df["dataset"].iloc[0] != dataset:
        raise AssertionError("Rankings 'dataset' column mismatch")
    if df["task_type"].nunique() != 1 or df["task_type"].iloc[0] != task_type:
        raise AssertionError("Rankings 'task_type' column mismatch")
    if df["seed"].nunique() != 1 or int(df["seed"].iloc[0]) != seed:
        raise AssertionError("Rankings 'seed' column mismatch")
    if df["method_id"].nunique() != 1 or df["method_id"].iloc[0] != method_id:
        raise AssertionError("Rankings 'method_id' column mismatch")
    if df["method_base"].nunique() != 1 or df["method_base"].iloc[0] != method_base:
        raise AssertionError("Rankings 'method_base' column mismatch")
    if (df["git_sha"] == "unknown").any():
        raise AssertionError(
            "Rankings git_sha is 'unknown' (expected env var or git checkout to provide a SHA)"
        )

    fold_idxs = sorted(int(v) for v in df["fold_idx"].tolist())
    if fold_idxs != list(range(N_SPLITS)):
        raise AssertionError(
            f"Rankings fold_idx mismatch: got {fold_idxs}, expected {list(range(N_SPLITS))}"
        )


def _assert_metrics_schema(
    df, *, dataset: str, task_type: str, seed: int, method_id: str, method_base: str
) -> None:
    required = {
        "dataset",
        "task_type",
        "seed",
        "method_id",
        "method",
        "method_base",
        "artifact_version",
        "fold_idx",
        "k",
        "downstream_model",
        "n_features_selected",
        "n_samples",
        "n_features",
        "evaluation_cpus",
        "elapsed_seconds",
        "created_at_utc",
        "git_sha",
    }
    missing = required - set(df.columns)
    if missing:
        raise AssertionError(f"Metrics missing columns: {sorted(missing)}")

    if df["dataset"].nunique() != 1 or df["dataset"].iloc[0] != dataset:
        raise AssertionError("Metrics 'dataset' column mismatch")
    if df["task_type"].nunique() != 1 or df["task_type"].iloc[0] != task_type:
        raise AssertionError("Metrics 'task_type' column mismatch")
    if df["seed"].nunique() != 1 or int(df["seed"].iloc[0]) != seed:
        raise AssertionError("Metrics 'seed' column mismatch")
    if df["method_id"].nunique() != 1 or df["method_id"].iloc[0] != method_id:
        raise AssertionError("Metrics 'method_id' column mismatch")
    if df["method_base"].nunique() != 1 or df["method_base"].iloc[0] != method_base:
        raise AssertionError("Metrics 'method_base' column mismatch")
    if (df["git_sha"] == "unknown").any():
        raise AssertionError(
            "Metrics git_sha is 'unknown' (expected env var or git checkout to provide a SHA)"
        )

    fold_idxs = sorted(set(int(v) for v in df["fold_idx"].tolist()))
    if fold_idxs != list(range(N_SPLITS)):
        raise AssertionError(
            f"Metrics fold_idx mismatch: got {fold_idxs}, expected {list(range(N_SPLITS))}"
        )


def main() -> None:
    args = _parse_args()
    cfg = load_config()

    if args.ray_address == "local":
        ray.init(ignore_reinit_error=True)
    else:
        ray.init(address=args.ray_address, ignore_reinit_error=True)

    task_type = args.task_type or cfg.experiment.type
    source = cast(DataSource, args.source)
    dataset = _choose_dataset(task_type, source, args.dataset)
    seed = int(args.seed)

    default_methods = ["mc", "rf"] if task_type == "classification" else ["pc", "rf"]
    methods = parse_csv_list(args.methods) or default_methods

    logger.info(
        "Smoke run config: task_type={}, source={}, dataset={}, seed={}, methods={}",
        task_type,
        args.source,
        dataset,
        seed,
        methods,
    )
    logger.info("S3 bucket: {}", get_s3_bucket())

    # -----------------------------------------------------------------------------
    # Stage 1: rankings
    # -----------------------------------------------------------------------------
    method_configs = expand_method_configs(methods)
    n_samples, n_features = get_dataset_shape(dataset, task_type)
    git_sha = ray_feature_selection.get_git_sha()

    stage1_futures = []
    for method_cfg in method_configs:
        method = method_cfg["method"]
        selection_cpus = ray_feature_selection.selection_num_cpus(
            method, n_samples=n_samples, n_features=n_features
        )
        stage1_futures.append(
            ray_feature_selection.process_config.options(num_cpus=selection_cpus).remote(
                method_cfg, dataset, seed, task_type, selection_cpus, git_sha
            )
        )

    stage1_results = ray.get(stage1_futures)
    failures = [r for r in stage1_results if r.get("status") != "done"]
    if failures:
        raise RuntimeError(f"Stage 1 failures: {failures}")

    # -----------------------------------------------------------------------------
    # Stage 2: metrics
    # -----------------------------------------------------------------------------
    evaluation_cpus = ray_eval.evaluation_num_cpus(task_type)

    stage2_futures = []
    for method_cfg in method_configs:
        stage2_futures.append(
            ray_eval.process_config.options(num_cpus=evaluation_cpus).remote(
                method_cfg, dataset, seed, task_type, evaluation_cpus, git_sha
            )
        )

    stage2_results = ray.get(stage2_futures)
    failures = [r for r in stage2_results if r.get("status") != "done"]
    if failures:
        raise RuntimeError(f"Stage 2 failures: {failures}")

    # -----------------------------------------------------------------------------
    # Validate artifacts by downloading from S3
    # -----------------------------------------------------------------------------
    for method_cfg in method_configs:
        method_base = method_cfg["method"]
        method_id = config_label(method_cfg)

        rankings_path = rankings_s3_key(task_type, dataset, method_id, seed)
        metrics_path = metrics_s3_path(task_type, dataset, method_id, seed)

        rankings_df = download_parquet_from_s3(rankings_path, region_name=cfg.region)
        metrics_df = download_parquet_from_s3(metrics_path, region_name=cfg.region)

        _assert_rankings_schema(
            rankings_df,
            dataset=dataset,
            task_type=task_type,
            seed=seed,
            method_id=method_id,
            method_base=method_base,
        )
        _assert_metrics_schema(
            metrics_df,
            dataset=dataset,
            task_type=task_type,
            seed=seed,
            method_id=method_id,
            method_base=method_base,
        )

        logger.info(
            "Validated artifacts: {} (rankings rows={}, metrics rows={})",
            method_id,
            len(rankings_df),
            len(metrics_df),
        )

    logger.info("Smoke run OK: Stage 1 + Stage 2 artifacts validated.")


if __name__ == "__main__":
    main()
