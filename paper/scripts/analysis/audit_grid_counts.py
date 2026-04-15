"""Audit raw local artifacts against the current experiment grid.

This script proves which current-grid configs are complete locally, which are
missing, and which raw files are stale relative to the current source-defined
grid.

Outputs:
  - paper/results/tables/grid_config_audit.csv
  - paper/results/tables/grid_dataset_audit.csv
  - paper/results/tables/grid_artifact_audit_summary.json

Usage:
  UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/audit_grid_counts.py --local-dir ../data
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

import pandas as pd

from paper.scripts.pipeline.grid import ExperimentGrid

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TABLES_DIR = RESULTS_DIR / "tables"


def _method_id_from_filename(filename: str) -> str | None:
    """Extract method_id from '{method_id}_seed{n}.parquet'."""
    stem = Path(filename).stem
    parts = stem.rsplit("_seed", 1)
    if len(parts) == 2:
        return parts[0]
    return None


def _seed_from_filename(filename: str) -> int | None:
    """Extract seed from '{method_id}_seed{n}.parquet'."""
    stem = Path(filename).stem
    parts = stem.rsplit("_seed", 1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _artifact_key_from_path(path: Path) -> tuple[str, str, int] | None:
    """Extract (method_id, dataset, seed) from a raw artifact path."""
    method_id = _method_id_from_filename(path.name)
    seed = _seed_from_filename(path.name)
    dataset = path.parent.name
    if method_id is None or seed is None or not dataset:
        return None
    return (method_id, dataset, seed)


def _collect_keys(
    local_dir: Path,
    stage: Literal["rankings", "metrics"],
    task: Literal["classification", "regression"],
) -> set[tuple[str, str, int]]:
    """Collect raw local artifact keys for a stage/task."""
    search_dir = local_dir / stage / task
    keys: set[tuple[str, str, int]] = set()
    for path in search_dir.rglob("*.parquet"):
        artifact_key = _artifact_key_from_path(path)
        if artifact_key is not None:
            keys.add(artifact_key)
    return keys


def _format_seed_list(seeds: set[int]) -> str:
    """Format a seed set for CSV output."""
    if not seeds:
        return ""
    return ",".join(str(seed) for seed in sorted(seeds))


def _audit_task(
    local_dir: Path,
    task: Literal["classification", "regression"],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Audit one task against the current ExperimentGrid."""
    grid = ExperimentGrid.from_cli(task, source="all")
    expected_keys = {cfg.key for cfg in grid}
    ranking_keys = _collect_keys(local_dir, "rankings", task)
    metric_keys = _collect_keys(local_dir, "metrics", task)

    matched_ranking_keys = ranking_keys & expected_keys
    matched_metric_keys = metric_keys & expected_keys
    extra_ranking_keys = ranking_keys - expected_keys
    extra_metric_keys = metric_keys - expected_keys

    expected_by_config: dict[str, set[tuple[str, str, int]]] = defaultdict(set)
    ranking_by_config: dict[str, set[tuple[str, str, int]]] = defaultdict(set)
    metric_by_config: dict[str, set[tuple[str, str, int]]] = defaultdict(set)

    for method_id, dataset, seed in expected_keys:
        expected_by_config[method_id].add((method_id, dataset, seed))
    for method_id, dataset, seed in matched_ranking_keys:
        ranking_by_config[method_id].add((method_id, dataset, seed))
    for method_id, dataset, seed in matched_metric_keys:
        metric_by_config[method_id].add((method_id, dataset, seed))

    config_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []

    for method_id in sorted(expected_by_config):
        method_base = method_id.split("__", 1)[0]
        expected = expected_by_config[method_id]
        observed_rank = ranking_by_config.get(method_id, set())
        observed_metric = metric_by_config.get(method_id, set())
        missing_rank = expected - observed_rank
        missing_metric = expected - observed_metric

        expected_datasets = sorted({dataset for _, dataset, _ in expected})
        dataset_rank_counts: dict[str, set[int]] = defaultdict(set)
        dataset_metric_counts: dict[str, set[int]] = defaultdict(set)
        dataset_expected_seeds: dict[str, set[int]] = defaultdict(set)

        for _, dataset, seed in expected:
            dataset_expected_seeds[dataset].add(seed)
        for _, dataset, seed in observed_rank:
            dataset_rank_counts[dataset].add(seed)
        for _, dataset, seed in observed_metric:
            dataset_metric_counts[dataset].add(seed)

        ranking_complete_datasets = 0
        metric_complete_datasets = 0
        ranking_partial_datasets = 0
        metric_partial_datasets = 0

        for dataset in expected_datasets:
            expected_seeds = dataset_expected_seeds[dataset]
            rank_seeds = dataset_rank_counts.get(dataset, set())
            metric_seeds = dataset_metric_counts.get(dataset, set())
            missing_rank_seeds = expected_seeds - rank_seeds
            missing_metric_seeds = expected_seeds - metric_seeds

            if rank_seeds == expected_seeds:
                ranking_complete_datasets += 1
            elif rank_seeds:
                ranking_partial_datasets += 1

            if metric_seeds == expected_seeds:
                metric_complete_datasets += 1
            elif metric_seeds:
                metric_partial_datasets += 1

            dataset_rows.append(
                {
                    "task": task,
                    "method_base": method_base,
                    "method_id": method_id,
                    "dataset": dataset,
                    "expected_seeds": len(expected_seeds),
                    "ranking_seeds": len(rank_seeds),
                    "metric_seeds": len(metric_seeds),
                    "missing_ranking_seeds": _format_seed_list(missing_rank_seeds),
                    "missing_metric_seeds": _format_seed_list(missing_metric_seeds),
                }
            )

        config_rows.append(
            {
                "task": task,
                "method_base": method_base,
                "method_id": method_id,
                "expected_jobs": len(expected),
                "ranking_jobs": len(observed_rank),
                "metric_jobs": len(observed_metric),
                "missing_ranking_jobs": len(missing_rank),
                "missing_metric_jobs": len(missing_metric),
                "expected_datasets": len(expected_datasets),
                "ranking_complete_datasets": ranking_complete_datasets,
                "metric_complete_datasets": metric_complete_datasets,
                "ranking_partial_datasets": ranking_partial_datasets,
                "metric_partial_datasets": metric_partial_datasets,
                "missing_ranking_cells": ";".join(
                    f"{dataset}/seed{seed}" for _, dataset, seed in sorted(missing_rank)
                ),
                "missing_metric_cells": ";".join(
                    f"{dataset}/seed{seed}" for _, dataset, seed in sorted(missing_metric)
                ),
            }
        )

    summary = {
        "task": task,
        "expected_artifact_keys": len(expected_keys),
        "raw_ranking_keys": len(ranking_keys),
        "raw_metric_keys": len(metric_keys),
        "matched_ranking_keys": len(matched_ranking_keys),
        "matched_metric_keys": len(matched_metric_keys),
        "extra_ranking_keys": len(extra_ranking_keys),
        "extra_metric_keys": len(extra_metric_keys),
        "extra_ranking_sample": [list(key) for key in sorted(extra_ranking_keys)[:20]],
        "extra_metric_sample": [list(key) for key in sorted(extra_metric_keys)[:20]],
        "configs_with_missing_rankings": sum(1 for row in config_rows if row["missing_ranking_jobs"] > 0),
        "configs_with_missing_metrics": sum(1 for row in config_rows if row["missing_metric_jobs"] > 0),
    }

    return (
        pd.DataFrame(config_rows).sort_values(["task", "method_base", "method_id"]).reset_index(drop=True),
        pd.DataFrame(dataset_rows).sort_values(["task", "method_base", "method_id", "dataset"]).reset_index(drop=True),
        summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw local artifacts against the current experiment grid")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("../data"),
        help="Local directory with rankings/ and metrics/ subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TABLES_DIR,
        help="Directory for audit outputs",
    )
    args = parser.parse_args()

    local_dir = args.local_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_frames: list[pd.DataFrame] = []
    dataset_frames: list[pd.DataFrame] = []
    summaries: dict[str, object] = {}

    for task in ("classification", "regression"):
        config_df, dataset_df, summary = _audit_task(local_dir, task)  # type: ignore[arg-type]
        config_frames.append(config_df)
        dataset_frames.append(dataset_df)
        summaries[task] = summary

    config_out = output_dir / "grid_config_audit.csv"
    dataset_out = output_dir / "grid_dataset_audit.csv"
    summary_out = output_dir / "grid_artifact_audit_summary.json"

    pd.concat(config_frames, ignore_index=True).to_csv(config_out, index=False)
    pd.concat(dataset_frames, ignore_index=True).to_csv(dataset_out, index=False)
    summary_out.write_text(json.dumps(summaries, indent=2) + "\n")

    print(f"Saved {config_out}")
    print(f"Saved {dataset_out}")
    print(f"Saved {summary_out}")


if __name__ == "__main__":
    main()
