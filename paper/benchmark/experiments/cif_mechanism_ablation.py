#!/usr/bin/env python3
"""CIF component-ablation runner for the paper.

This runner answers two reviewer-facing mechanism questions:

    Does the paper's split-importance ranking outperform a simpler split-count
    readout?

    How much of the CIF-vs-CIT gap remains when bootstrap sampling, feature
    muting, or forest aggregation is changed one component at a time?

Each work item is one task/dataset/seed/fold CIF fit for one model variant. The
fitted forest is then read one or two ways:

* ``split_importance``: the current paper CIF ranking.
* ``split_count``: count accepted split uses per feature, used for the readout
  ablation on the default fitted forest.

Practical controls such as adaptive stopping, histogram thresholds, and feature
scanning remain exactly as selected in the paper benchmark.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import re
import socket
import sys
import time
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import boto3
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
)
from paper.benchmark.adapters.data import (
    get_cv_splitter,
    get_dataset_metadata,
    get_datasets,
    load_dataset,
)
from paper.benchmark.config.constants import N_SPLITS
from paper.benchmark.pipeline.stage2 import (
    evaluate_fold,
    get_requested_evaluation_k_values,
)
from paper.benchmark.utils.env import get_git_sha, get_library_versions, utc_now_iso

ROOT: Final[Path] = Path(__file__).resolve().parents[3]
TABLES_DIR: Final[Path] = ROOT / "paper" / "results" / "tables"
DEFAULT_OUTPUT_URI: Final[str] = str(ROOT / "scratch" / "cif_mechanism_ablation")
SELECTED_CONFIG_PATH: Final[Path] = TABLES_DIR / "paper_benchmark_selected_config_details.csv"
EXPERIMENT_NAME: Final[str] = "cif_component_ablation"
MODEL_VARIANTS: Final[tuple[str, ...]] = (
    "cif_default",
    "cif_no_mute",
    "cif_one_tree",
    "cif_no_bootstrap",
)
DEFAULT_MODEL_VARIANTS: Final[tuple[str, ...]] = ("cif_default",)
RANKING_VARIANTS: Final[tuple[str, ...]] = ("split_importance", "split_count")
DEFAULT_FOLDS: Final[tuple[int, ...]] = tuple(range(N_SPLITS))

PARAM_COLUMNS: Final[tuple[str, ...]] = (
    "selector",
    "splitter",
    "alpha_selector",
    "alpha_splitter",
    "adjust_alpha_selector",
    "adjust_alpha_splitter",
    "n_resamples_selector",
    "n_resamples_splitter",
    "early_stopping_selector",
    "early_stopping_splitter",
    "early_stopping_confidence_selector",
    "early_stopping_confidence_splitter",
    "feature_muting",
    "feature_scanning",
    "max_features",
    "threshold_method",
    "threshold_scanning",
    "max_thresholds",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "min_impurity_decrease",
    "honesty",
    "honesty_fraction",
    "bootstrap",
    "sampling_method",
    "max_samples",
    "oob_score",
    "n_estimators",
    "n_jobs",
    "verbose",
)


@dataclass(frozen=True)
class WorkItem:
    """One independently runnable fold-level diagnostic."""

    task: str
    dataset: str
    seed: int
    fold_idx: int
    model_variant: str
    ranking_variants: tuple[str, ...]

    @property
    def file_stem(self) -> str:
        return f"{self.model_variant}_seed{self.seed}_fold{self.fold_idx}"


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _split_int_csv(value: str | None) -> tuple[int, ...]:
    return tuple(int(part) for part in _split_csv(value))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return bool(pd.isna(value)) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False


def _coerce_config_value(value: Any) -> Any:
    """Convert CSV-loaded values into estimator-friendly Python objects."""
    if _is_missing(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        if value == "" or value.lower() == "nan":
            return None
        if value == "True":
            return True
        if value == "False":
            return False
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)", value):
            number = float(value)
            return int(number) if number.is_integer() else number
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def load_selected_cif_params(task: str, *, n_jobs: int | None) -> dict[str, Any]:
    """Load the paper-selected CIF configuration for one task."""
    if not SELECTED_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Selected configuration table not found: {SELECTED_CONFIG_PATH}")

    selected = pd.read_csv(SELECTED_CONFIG_PATH)
    row = selected[(selected["task"] == task) & (selected["method_base"] == "cif")]
    if len(row) != 1:
        raise ValueError(f"Expected one selected CIF row for task={task!r}, found {len(row)}")

    params: dict[str, Any] = {}
    record = row.iloc[0]
    for column in PARAM_COLUMNS:
        if column not in row.columns:
            continue
        value = _coerce_config_value(record[column])
        if value is not None:
            params[column] = value

    # The regression forest constructor does not accept sampling_method.
    if task == "regression":
        params.pop("sampling_method", None)

    if n_jobs is not None:
        params["n_jobs"] = n_jobs
    params["verbose"] = 0
    return params


def apply_model_variant(params: dict[str, Any], model_variant: str) -> dict[str, Any]:
    """Apply one diagnostic model edit to the selected CIF configuration."""
    variant_params = dict(params)
    if model_variant == "cif_default":
        return variant_params
    if model_variant == "cif_no_mute":
        variant_params["feature_muting"] = False
        return variant_params
    if model_variant == "cif_one_tree":
        variant_params["n_estimators"] = 1
        return variant_params
    if model_variant == "cif_no_bootstrap":
        variant_params["bootstrap"] = False
        variant_params["max_samples"] = None
        variant_params["oob_score"] = False
        if "sampling_method" in variant_params:
            variant_params["sampling_method"] = None
        return variant_params
    raise ValueError(f"Unknown model variant: {model_variant}")


def build_cif(task: str, params: dict[str, Any], random_state: int, model_variant: str):
    """Build a selected CIF estimator for a task."""
    model_params = {
        **apply_model_variant(params, model_variant),
        "random_state": random_state,
    }
    if task == "classification":
        return ConditionalInferenceForestClassifier(**model_params)
    model_params.pop("sampling_method", None)
    return ConditionalInferenceForestRegressor(**model_params)


def _iter_internal_nodes(tree: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield internal nodes from a fitted CIT tree dict."""
    if "value" in tree:
        return
    yield {"feature": int(tree["feature"])}
    left = tree.get("left_child")
    right = tree.get("right_child")
    if isinstance(left, dict):
        yield from _iter_internal_nodes(left)
    if isinstance(right, dict):
        yield from _iter_internal_nodes(right)


def _rank_from_scores(scores: np.ndarray) -> np.ndarray:
    """Return descending-score ranking with deterministic feature-index tie-breaks."""
    features = np.arange(scores.shape[0])
    return np.lexsort((features, -scores)).astype(int)


def compute_ranking_scores(model: Any, n_features: int) -> dict[str, np.ndarray]:
    """Compute the two supported ranking score vectors from one fitted forest."""
    scores = {
        "split_importance": np.asarray(model.feature_importances_, dtype=float).copy(),
        "split_count": np.zeros(n_features, dtype=float),
    }

    for estimator in model.estimators_:
        tree = getattr(estimator, "tree_", None)
        if not isinstance(tree, dict):
            continue
        for node in _iter_internal_nodes(tree):
            scores["split_count"][int(node["feature"])] += 1.0

    return scores


def rankings_from_model(
    model: Any,
    n_features: int,
    ranking_variants: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return feature rankings for selected readout variants."""
    score_map = compute_ranking_scores(model, n_features)
    unknown = sorted(set(ranking_variants) - set(score_map))
    if unknown:
        raise ValueError(f"Unknown ranking variant(s): {unknown}")
    return {name: _rank_from_scores(score_map[name]) for name in ranking_variants}


def _s3_client() -> Any:
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    return boto3.client("s3", region_name=region)


def _split_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {uri}")
    bucket_key = uri[len("s3://") :]
    bucket, _, key = bucket_key.partition("/")
    if not bucket or not key:
        raise ValueError(f"Expected s3://bucket/key, got: {uri}")
    return bucket, key.rstrip("/")


def artifact_uri(output_uri: str, stage: str, item: WorkItem) -> str:
    """Build the output artifact URI for one fold-level work item."""
    rel = f"{stage}/{item.task}/{item.dataset}/{item.file_stem}.parquet"
    if output_uri.startswith("s3://"):
        return f"{output_uri.rstrip('/')}/{rel}"
    return str(Path(output_uri) / rel)


def artifact_exists(uri: str) -> bool:
    """Return True if a local or S3 artifact already exists."""
    if uri.startswith("s3://"):
        bucket, key = _split_s3_uri(uri)
        try:
            _s3_client().head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False
    return Path(uri).exists()


def read_frame(uri: str) -> pd.DataFrame:
    """Read a parquet DataFrame from a local path or S3 URI."""
    if uri.startswith("s3://"):
        bucket, key = _split_s3_uri(uri)
        body = _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
        return pd.read_parquet(io.BytesIO(body))
    return pd.read_parquet(uri)


def save_frame(df: pd.DataFrame, uri: str) -> None:
    """Save a DataFrame as parquet to a local path or S3 URI."""
    if uri.startswith("s3://"):
        bucket, key = _split_s3_uri(uri)
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        _s3_client().put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        return

    path = Path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def artifact_covers_rankings(uri: str, ranking_variants: Sequence[str]) -> bool:
    """Return True when an existing artifact contains all requested rankings."""
    if not artifact_exists(uri):
        return False
    try:
        df = read_frame(uri)
    except Exception:
        return False
    if "ranking_variant" not in df.columns:
        return False
    return set(ranking_variants).issubset(set(df["ranking_variant"].dropna()))


def item_is_complete(
    *,
    metrics_uri: str,
    rankings_uri: str,
    fits_uri: str,
    ranking_variants: Sequence[str],
) -> bool:
    """Return True when all fold-level outputs are present and complete."""
    return (
        artifact_exists(fits_uri)
        and artifact_covers_rankings(rankings_uri, ranking_variants)
        and artifact_covers_rankings(metrics_uri, ranking_variants)
    )


def run_item(
    item: WorkItem,
    *,
    output_uri: str,
    n_jobs: int | None,
    downstream_n_jobs: int,
    force: bool,
) -> dict[str, Any]:
    """Run one fold-level work item and write ranking/metric artifacts."""
    metrics_uri = artifact_uri(output_uri, "metrics", item)
    rankings_uri = artifact_uri(output_uri, "rankings", item)
    fits_uri = artifact_uri(output_uri, "fits", item)
    if not force and item_is_complete(
        metrics_uri=metrics_uri,
        rankings_uri=rankings_uri,
        fits_uri=fits_uri,
        ranking_variants=item.ranking_variants,
    ):
        return {
            "status": "skipped",
            "task": item.task,
            "dataset": item.dataset,
            "seed": item.seed,
            "fold_idx": item.fold_idx,
            "model_variant": item.model_variant,
        }

    started = time.perf_counter()
    created_at_utc = utc_now_iso()
    git_sha = get_git_sha()
    library_versions = get_library_versions()
    hostname = socket.gethostname()

    X, y = load_dataset(item.dataset, item.task)
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
    dataset_meta = get_dataset_metadata(item.dataset, item.task)
    params = load_selected_cif_params(item.task, n_jobs=n_jobs)

    cv = get_cv_splitter(item.task, N_SPLITS, item.seed)
    splits = list(cv.split(X, y))
    if item.fold_idx < 0 or item.fold_idx >= len(splits):
        raise ValueError(f"Invalid fold_idx={item.fold_idx}; expected 0..{len(splits) - 1}")

    train_idx, test_idx = splits[item.fold_idx]
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)

    random_state = item.seed * 1000 + item.fold_idx
    model = build_cif(
        item.task,
        params,
        random_state=random_state,
        model_variant=item.model_variant,
    )
    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_elapsed = time.perf_counter() - fit_start

    k_values = get_requested_evaluation_k_values(n_features)
    rankings = rankings_from_model(model, n_features, item.ranking_variants)

    common = {
        "experiment": EXPERIMENT_NAME,
        "task": item.task,
        "dataset": item.dataset,
        "seed": item.seed,
        "fold_idx": item.fold_idx,
        "model_variant": item.model_variant,
        "artifact_version": 2,
        "n_samples": n_samples,
        "n_features": n_features,
        "fit_elapsed_seconds": float(fit_elapsed),
        "created_at_utc": created_at_utc,
        "git_sha": git_sha,
        "library_versions": library_versions,
        "dataset_source": dataset_meta.get("dataset_source"),
        "dataset_type": dataset_meta.get("dataset_type"),
        "dataset_family": dataset_meta.get("dataset_family"),
        "n_informative": dataset_meta.get("n_informative"),
        "hostname": hostname,
    }

    fit_df = pd.DataFrame(
        [
            {
                **common,
                "n_estimators_actual": int(len(model.estimators_)),
            }
        ]
    )

    ranking_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for ranking_variant, ranking in rankings.items():
        ranking_rows.append(
            {
                **common,
                "ranking_variant": ranking_variant,
                "feature_ranking": ranking.astype(int).tolist(),
            }
        )

        fold_metrics = evaluate_fold(
            X_train_raw,
            y_train,
            X_test_raw,
            y_test,
            ranking,
            item.task,
            random_state,
            k_values=k_values,
            n_jobs=downstream_n_jobs,
        )
        for row in fold_metrics:
            row.update({**common, "ranking_variant": ranking_variant})
            metric_rows.append(row)

    rankings_df = pd.DataFrame(ranking_rows)
    metrics_df = pd.DataFrame(metric_rows)

    save_frame(rankings_df, rankings_uri)
    save_frame(metrics_df, metrics_uri)
    save_frame(fit_df, fits_uri)

    return {
        "status": "done",
        "task": item.task,
        "dataset": item.dataset,
        "seed": item.seed,
        "fold_idx": item.fold_idx,
        "model_variant": item.model_variant,
        "elapsed_seconds": time.perf_counter() - started,
        "fit_elapsed_seconds": float(fit_elapsed),
        "n_metric_rows": len(metrics_df),
        "n_ranking_rows": len(rankings_df),
        "metrics_uri": metrics_uri,
    }


def build_work_items(
    *,
    tasks: tuple[str, ...],
    source: str,
    datasets: tuple[str, ...],
    seeds: tuple[int, ...],
    folds: tuple[int, ...],
    model_variants: tuple[str, ...],
    ranking_variants: tuple[str, ...],
) -> list[WorkItem]:
    """Build the full fold-level work list before sharding."""
    available_by_task = {
        task: get_datasets(task, source=source)  # type: ignore[arg-type]
        for task in tasks
    }
    if datasets:
        known = set().union(*(set(values) for values in available_by_task.values()))
        unknown = sorted(set(datasets) - known)
        if unknown:
            raise ValueError(f"Unknown dataset(s) for requested tasks: {unknown}")

    items: list[WorkItem] = []
    for task in tasks:
        task_datasets = available_by_task[task]
        if datasets:
            wanted = set(datasets)
            task_datasets = [dataset for dataset in task_datasets if dataset in wanted]
        for dataset in task_datasets:
            for seed in seeds:
                for fold_idx in folds:
                    for model_variant in model_variants:
                        items.append(
                            WorkItem(
                                task=task,
                                dataset=dataset,
                                seed=seed,
                                fold_idx=fold_idx,
                                model_variant=model_variant,
                                ranking_variants=ranking_variants,
                            )
                        )
    return items


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIF component ablations")
    parser.add_argument("--output-uri", default=DEFAULT_OUTPUT_URI, help="Local path or s3:// URI")
    parser.add_argument(
        "--tasks",
        default="classification,regression",
        help="Comma-separated tasks: classification,regression",
    )
    parser.add_argument(
        "--source",
        choices=("real", "synthetic", "all"),
        default="real",
        help="Dataset source to run",
    )
    parser.add_argument("--datasets", default="", help="Optional comma-separated dataset filter")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seed indices")
    parser.add_argument(
        "--folds",
        default=",".join(str(fold_idx) for fold_idx in DEFAULT_FOLDS),
        help="Comma-separated fold indices",
    )
    parser.add_argument(
        "--model-variants",
        default=",".join(DEFAULT_MODEL_VARIANTS),
        help="Model variants: cif_default,cif_no_mute,cif_one_tree,cif_no_bootstrap",
    )
    parser.add_argument(
        "--ranking-variants",
        default=",".join(RANKING_VARIANTS),
        help="Ranking readouts: split_importance,split_count",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--shard-index", type=int, default=0, help="This shard index")
    parser.add_argument("--n-jobs", type=int, default=-1, help="CIF n_jobs per fit")
    parser.add_argument("--downstream-n-jobs", type=int, default=1, help="Downstream n_jobs")
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Print work items without running")
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap after sharding")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tasks = _split_csv(args.tasks)
    datasets = _split_csv(args.datasets)
    model_variants = _split_csv(args.model_variants)
    ranking_variants = _split_csv(args.ranking_variants)
    seeds = _split_int_csv(args.seeds)
    folds = _split_int_csv(args.folds)

    invalid_tasks = sorted(set(tasks) - {"classification", "regression"})
    if invalid_tasks:
        raise ValueError(f"Unknown task(s): {invalid_tasks}")
    invalid_rankings = sorted(set(ranking_variants) - set(RANKING_VARIANTS))
    if invalid_rankings:
        raise ValueError(f"Unknown ranking variant(s): {invalid_rankings}")
    invalid_model_variants = sorted(set(model_variants) - set(MODEL_VARIANTS))
    if invalid_model_variants:
        raise ValueError(f"Unknown model variant(s): {invalid_model_variants}")
    invalid_folds = sorted(fold_idx for fold_idx in folds if fold_idx not in DEFAULT_FOLDS)
    if invalid_folds:
        raise ValueError(f"Unknown fold index(es): {invalid_folds}; expected {DEFAULT_FOLDS}")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")

    all_items = build_work_items(
        tasks=tasks,
        source=args.source,
        datasets=datasets,
        seeds=seeds,
        folds=folds,
        model_variants=model_variants,
        ranking_variants=ranking_variants,
    )
    shard_items = [
        item for idx, item in enumerate(all_items) if idx % args.num_shards == args.shard_index
    ]
    if args.max_items is not None:
        shard_items = shard_items[: args.max_items]

    print(
        f"{EXPERIMENT_NAME}: total_items={len(all_items)} shard_items={len(shard_items)} "
        f"shard={args.shard_index}/{args.num_shards} output={args.output_uri}",
        flush=True,
    )

    if args.dry_run:
        for item in shard_items:
            print(item)
        return 0

    failures = 0
    for idx, item in enumerate(shard_items, start=1):
        print(
            f"[{idx}/{len(shard_items)}] {item.task} {item.dataset} "
            f"seed={item.seed} fold={item.fold_idx}",
            flush=True,
        )
        try:
            result = run_item(
                item,
                output_uri=args.output_uri,
                n_jobs=args.n_jobs,
                downstream_n_jobs=args.downstream_n_jobs,
                force=args.force,
            )
            print(result, flush=True)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(
                {
                    "status": "failed",
                    "task": item.task,
                    "dataset": item.dataset,
                    "seed": item.seed,
                    "fold_idx": item.fold_idx,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                file=sys.stderr,
                flush=True,
            )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
