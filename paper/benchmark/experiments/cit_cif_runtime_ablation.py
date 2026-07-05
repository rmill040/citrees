#!/usr/bin/env python3
"""Runtime ablation for CIT and CIF fitting/ranking speed.

This script is narrower than the mirrored knob ablation: it times the model fit
that produces feature importances, records tree structure, and optionally
records synthetic top-k recovery metrics. It does not run downstream models.

Usage examples:
    uv run python -m paper.benchmark.experiments.cit_cif_runtime_ablation

    uv run python -m paper.benchmark.experiments.cit_cif_runtime_ablation \
        --tasks clf --dataset-sources synthetic --max-synthetic-datasets 1 \
        --seeds 0 --variants cit_default,cif_default,dt,rt \
        --output-dir scratch/runtime_smoke --output-name runtime_smoke
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)

from paper.benchmark.experiments.experiment_common import (
    CLF_ALL,
    N_ESTIMATORS,
    RANDOM_STATE,
    REAL_CLF_NAMES,
    REAL_REG_NAMES,
    REG_ALL,
    TABLES_DIR,
    SyntheticDataset,
    build_cif,
    build_cit,
    load_real_clf,
    load_real_reg,
    warmup_jit,
)
from paper.benchmark.utils.metrics import f1_at_k, precision_at_k, recall_at_k

EXPERIMENT_NAME = "cit_cif_runtime_ablation"
K_VALUES = (5, 10, 25, 50, 100)
TASKS = ("clf", "reg")
DATASET_SOURCES = ("synthetic", "real")
SUMMARY_NUMERIC_COLUMNS = (
    "elapsed_seconds",
    "elapsed_ratio_vs_family_default",
    "mean_depth",
    "max_depth",
    "mean_features_used_per_tree",
    "importance_nonzero_count",
    "precision_at_10",
    "recall_at_10",
    "f1_at_10",
)


@dataclass(frozen=True)
class VariantSpec:
    """One model configuration to time."""

    name: str
    method_family: str
    overrides: dict[str, Any]
    description: str
    changes_statistical_rule: bool = False
    changes_sampling_scheme: bool = False


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset entry before seed-specific materialization."""

    task: str
    dataset_source: str
    dataset_key: str
    dataset_type: str
    synthetic_factory: Callable[[int], SyntheticDataset] | None = None


CIT_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("cit_default", "cit", {}, "CIT with the paper default runtime controls"),
    VariantSpec(
        "cit_no_adaptive",
        "cit",
        {"early_stopping_selector": None, "early_stopping_splitter": None},
        "CIT with full permutation tests instead of adaptive stopping",
    ),
    VariantSpec(
        "cit_no_feature_scan",
        "cit",
        {"feature_scanning": False},
        "CIT without feature scan ordering",
    ),
    VariantSpec(
        "cit_no_threshold_scan",
        "cit",
        {"threshold_scanning": False},
        "CIT without threshold scan ordering",
    ),
    VariantSpec(
        "cit_no_feature_mute",
        "cit",
        {"feature_muting": False},
        "CIT without muting features after failed selector tests",
    ),
    VariantSpec(
        "cit_exact_thresholds",
        "cit",
        {"threshold_method": "exact", "max_thresholds": None},
        "CIT using all exact split thresholds",
    ),
    VariantSpec(
        "cit_no_bonferroni",
        "cit",
        {"adjust_alpha_selector": False, "adjust_alpha_splitter": False},
        "CIT without Bonferroni correction",
        changes_statistical_rule=True,
    ),
)

CIF_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("cif_default", "cif", {}, "CIF with the paper default runtime controls"),
    VariantSpec(
        "cif_no_adaptive",
        "cif",
        {"early_stopping_selector": None, "early_stopping_splitter": None},
        "CIF with full permutation tests instead of adaptive stopping",
    ),
    VariantSpec(
        "cif_no_feature_scan",
        "cif",
        {"feature_scanning": False},
        "CIF without feature scan ordering",
    ),
    VariantSpec(
        "cif_no_threshold_scan",
        "cif",
        {"threshold_scanning": False},
        "CIF without threshold scan ordering",
    ),
    VariantSpec(
        "cif_no_feature_mute",
        "cif",
        {"feature_muting": False},
        "CIF without muting features after failed selector tests",
    ),
    VariantSpec(
        "cif_no_bootstrap",
        "cif",
        {"bootstrap": False, "sampling_method": None},
        "CIF without bootstrap sampling",
        changes_sampling_scheme=True,
    ),
    VariantSpec(
        "cif_all_features",
        "cif",
        {"max_features": None},
        "CIF considering all remaining features at each split",
    ),
    VariantSpec(
        "cif_exact_thresholds",
        "cif",
        {"threshold_method": "exact", "max_thresholds": None},
        "CIF using all exact split thresholds",
    ),
    VariantSpec(
        "cif_no_bonferroni",
        "cif",
        {"adjust_alpha_selector": False, "adjust_alpha_splitter": False},
        "CIF without Bonferroni correction",
        changes_statistical_rule=True,
    ),
)

REFERENCE_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("dt", "reference", {}, "single deterministic CART-style decision tree"),
    VariantSpec("rt", "reference", {}, "single randomized tree"),
    VariantSpec("rf", "reference", {}, "random forest reference"),
    VariantSpec("et", "reference", {}, "extra-trees forest reference"),
)

ALL_VARIANTS: dict[str, VariantSpec] = {
    spec.name: spec for spec in (*CIT_VARIANTS, *CIF_VARIANTS, *REFERENCE_VARIANTS)
}


def _split_csv(value: str) -> tuple[str, ...]:
    """Parse a comma-separated CLI value."""
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _expand_choices(
    values: Sequence[str], *, all_value: str, allowed: Sequence[str], label: str
) -> tuple[str, ...]:
    """Expand 'all' and validate a CLI choice list."""
    if all_value in values:
        return tuple(allowed)

    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"Unknown {label}: {', '.join(invalid)}. Allowed: {', '.join(allowed)}")
    return tuple(dict.fromkeys(values))


def _parse_seed_offsets(value: str) -> tuple[int, ...]:
    """Parse comma-separated seed offsets."""
    offsets = tuple(int(part) for part in _split_csv(value))
    if not offsets:
        raise ValueError("At least one seed offset is required")
    return offsets


def _git_sha() -> str:
    """Return the current git SHA if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def _build_reference(
    method: str,
    task: str,
    seed: int,
    n_jobs: int | None,
    n_estimators: int,
) -> BaseEstimator:
    """Build one non-CIT/CIF reference estimator."""
    if method == "dt":
        if task == "clf":
            return DecisionTreeClassifier(random_state=seed)
        return DecisionTreeRegressor(random_state=seed)
    if method == "rt":
        if task == "clf":
            return ExtraTreeClassifier(splitter="random", random_state=seed)
        return ExtraTreeRegressor(splitter="random", random_state=seed)
    if method == "rf":
        if task == "clf":
            return RandomForestClassifier(
                n_estimators=n_estimators, n_jobs=n_jobs, random_state=seed
            )
        return RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, random_state=seed)
    if method == "et":
        if task == "clf":
            return ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=n_jobs, random_state=seed)
        return ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=n_jobs, random_state=seed)
    raise ValueError(f"Unknown reference method: {method}")


def _build_model(
    spec: VariantSpec,
    task: str,
    seed: int,
    n_jobs: int | None,
    n_estimators: int,
) -> BaseEstimator:
    """Build a model for one runtime row."""
    if spec.method_family == "cit":
        return build_cit(task, seed, **spec.overrides)
    if spec.method_family == "cif":
        return build_cif(task, seed, n_estimators=n_estimators, n_jobs=n_jobs, **spec.overrides)
    return _build_reference(spec.name, task, seed, n_jobs, n_estimators)


def _synthetic_specs(
    tasks: Sequence[str], dataset_names: set[str] | None, max_count: int | None
) -> list[DatasetSpec]:
    """Build synthetic dataset specs."""
    specs: list[DatasetSpec] = []
    for task, factories in (("clf", CLF_ALL), ("reg", REG_ALL)):
        if task not in tasks:
            continue
        for factory in factories:
            _, _, _, dataset_type = factory(RANDOM_STATE)
            if dataset_names and dataset_type not in dataset_names:
                continue
            specs.append(
                DatasetSpec(
                    task=task,
                    dataset_source="synthetic",
                    dataset_key=dataset_type,
                    dataset_type=dataset_type,
                    synthetic_factory=factory,
                )
            )

    if max_count is None:
        return specs

    limited: list[DatasetSpec] = []
    counts = {task: 0 for task in tasks}
    for spec in specs:
        if counts[spec.task] >= max_count:
            continue
        counts[spec.task] += 1
        limited.append(spec)
    return limited


def _real_specs(
    tasks: Sequence[str],
    dataset_names: set[str] | None,
    max_count: int | None,
    skip_openml: bool,
) -> list[DatasetSpec]:
    """Build real dataset specs."""
    specs: list[DatasetSpec] = []
    real_names: tuple[tuple[str, Sequence[str]], ...] = (
        ("clf", REAL_CLF_NAMES),
        ("reg", REAL_REG_NAMES),
    )
    for task, names in real_names:
        if task not in tasks:
            continue
        task_count = 0
        for name in names:
            if skip_openml and name.startswith("openml_"):
                continue
            dataset_type = f"real_{name}"
            if dataset_names and name not in dataset_names and dataset_type not in dataset_names:
                continue
            if max_count is not None and task_count >= max_count:
                continue
            specs.append(
                DatasetSpec(
                    task=task,
                    dataset_source="real",
                    dataset_key=name,
                    dataset_type=dataset_type,
                )
            )
            task_count += 1
    return specs


def _load_dataset(
    spec: DatasetSpec,
    seed: int,
    real_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, str]],
) -> tuple[np.ndarray, np.ndarray, list[int] | None, str]:
    """Load one dataset for one seed."""
    if spec.dataset_source == "synthetic":
        if spec.synthetic_factory is None:
            raise ValueError(f"Synthetic dataset {spec.dataset_type} has no factory")
        X, y, true_info, dataset_type = spec.synthetic_factory(seed)
        return X, y, true_info, dataset_type

    cache_key = (spec.task, spec.dataset_key)
    if cache_key not in real_cache:
        if spec.task == "clf":
            real_cache[cache_key] = load_real_clf(spec.dataset_key)
        else:
            real_cache[cache_key] = load_real_reg(spec.dataset_key)
    X, y, dataset_type = real_cache[cache_key]
    return X, y, None, dataset_type


def _tree_depth(estimator: Any) -> float:
    """Extract depth from either citrees or sklearn tree estimators."""
    if hasattr(estimator, "depth_"):
        return float(estimator.depth_)
    if hasattr(estimator, "get_depth"):
        return float(estimator.get_depth())
    tree = getattr(estimator, "tree_", None)
    if tree is not None and hasattr(tree, "max_depth"):
        return float(tree.max_depth)
    return np.nan


def _tree_stats(model: BaseEstimator) -> dict[str, float]:
    """Extract comparable tree stats from citrees and sklearn estimators."""
    estimators = getattr(model, "estimators_", None)
    if estimators is not None and len(estimators) > 0:
        depths = [_tree_depth(estimator) for estimator in estimators]
        features_used = [
            float(np.count_nonzero(getattr(estimator, "feature_importances_", np.array([]))))
            for estimator in estimators
        ]
        return {
            "mean_depth": float(np.nanmean(depths)),
            "max_depth": float(np.nanmax(depths)),
            "mean_features_used_per_tree": float(np.nanmean(features_used)),
            "n_estimators_actual": float(len(estimators)),
        }

    return {
        "mean_depth": _tree_depth(model),
        "max_depth": _tree_depth(model),
        "mean_features_used_per_tree": float(
            np.count_nonzero(getattr(model, "feature_importances_", np.array([])))
        ),
        "n_estimators_actual": 1.0,
    }


def _fit_runtime_metrics(
    X: np.ndarray,
    y: np.ndarray,
    true_info: list[int] | None,
    model: BaseEstimator,
) -> dict[str, Any]:
    """Fit one model and return runtime, ranking, and structure metrics."""
    start = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - start

    importances = np.asarray(model.feature_importances_, dtype=float)
    ranking = list(np.argsort(importances)[::-1])

    metrics: dict[str, Any] = {
        "elapsed_seconds": float(elapsed),
        "importance_sum": float(importances.sum()),
        "importance_nonzero_count": int(np.count_nonzero(importances)),
        "importance_max": float(np.max(importances)) if importances.size else np.nan,
    }
    metrics.update(_tree_stats(model))

    if true_info is not None:
        for k in K_VALUES:
            k_eff = min(k, len(ranking))
            metrics[f"precision_at_{k}"] = precision_at_k(ranking, true_info, k_eff)
            metrics[f"recall_at_{k}"] = recall_at_k(ranking, true_info, k_eff)
            metrics[f"f1_at_{k}"] = f1_at_k(ranking, true_info, k_eff)

    return metrics


def _row_context(
    spec: DatasetSpec,
    variant: VariantSpec,
    seed_offset: int,
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_jobs: int | None,
    n_estimators: int,
    git_sha: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """Build metadata shared by success and failure rows."""
    requested_trees = (
        n_estimators if variant.method_family == "cif" or variant.name in {"rf", "et"} else 1
    )
    return {
        "experiment": EXPERIMENT_NAME,
        "task": spec.task,
        "dataset_source": spec.dataset_source,
        "dataset_key": spec.dataset_key,
        "dataset_type": spec.dataset_type,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(np.unique(y).size) if spec.task == "clf" else np.nan,
        "seed_offset": int(seed_offset),
        "seed": int(seed),
        "variant": variant.name,
        "method_family": variant.method_family,
        "variant_description": variant.description,
        "changes_statistical_rule": bool(variant.changes_statistical_rule),
        "changes_sampling_scheme": bool(variant.changes_sampling_scheme),
        "n_jobs": n_jobs,
        "n_estimators_requested": int(requested_trees),
        "run_shard_id": os.environ.get("CITREES_RUN_SHARD", ""),
        "ec2_instance_id": os.environ.get("EC2_INSTANCE_ID", ""),
        "ec2_instance_type": os.environ.get("EC2_INSTANCE_TYPE", ""),
        "aws_region": os.environ.get("AWS_DEFAULT_REGION", ""),
        "host": platform.node(),
        "python": platform.python_version(),
        "git_sha": git_sha,
        "created_at_utc": created_at_utc,
    }


def _add_default_ratios(raw: pd.DataFrame) -> pd.DataFrame:
    """Add elapsed-time ratios against the same-family default row."""
    df = raw.copy()
    df["elapsed_ratio_vs_family_default"] = np.nan

    keys = ["task", "dataset_source", "dataset_type", "seed", "method_family"]
    for family, default_variant in (("cit", "cit_default"), ("cif", "cif_default")):
        defaults = df[
            (df["method_family"] == family)
            & (df["variant"] == default_variant)
            & (df["fit_success"])
            & (df["elapsed_seconds"].notna())
        ][[*keys, "elapsed_seconds"]].rename(
            columns={"elapsed_seconds": "_default_elapsed_seconds"}
        )
        if defaults.empty:
            continue

        family_idx = (
            (df["method_family"] == family) & (df["fit_success"]) & (df["elapsed_seconds"].notna())
        )
        merged = df.loc[family_idx, keys + ["elapsed_seconds"]].merge(defaults, on=keys, how="left")
        ratios = merged["elapsed_seconds"] / merged["_default_elapsed_seconds"]
        df.loc[family_idx, "elapsed_ratio_vs_family_default"] = ratios.to_numpy()

    return df


def _summarize_group(raw: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Summarize runtime rows for one grouping level."""
    grouped = raw.groupby(list(group_cols), dropna=False)
    summary = grouped.agg(
        n_runs=("fit_success", "size"),
        n_success=("fit_success", "sum"),
        n_datasets=("dataset_type", "nunique"),
        elapsed_seconds_mean=("elapsed_seconds", "mean"),
        elapsed_seconds_median=("elapsed_seconds", "median"),
        elapsed_seconds_std=("elapsed_seconds", "std"),
        elapsed_ratio_vs_family_default_mean=("elapsed_ratio_vs_family_default", "mean"),
        elapsed_ratio_vs_family_default_median=("elapsed_ratio_vs_family_default", "median"),
        mean_depth_mean=("mean_depth", "mean"),
        max_depth_mean=("max_depth", "mean"),
        mean_features_used_per_tree_mean=("mean_features_used_per_tree", "mean"),
        importance_nonzero_count_mean=("importance_nonzero_count", "mean"),
        precision_at_10_mean=("precision_at_10", "mean"),
        recall_at_10_mean=("recall_at_10", "mean"),
        f1_at_10_mean=("f1_at_10", "mean"),
    )
    return summary.reset_index()


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Build source-level and dataset-level summaries."""
    raw_with_ratios = _add_default_ratios(raw)
    for column in SUMMARY_NUMERIC_COLUMNS:
        if column not in raw_with_ratios:
            raw_with_ratios[column] = np.nan

    source_summary = _summarize_group(
        raw_with_ratios,
        ["task", "dataset_source", "method_family", "variant"],
    )
    source_summary.insert(0, "summary_scope", "dataset_source")
    source_summary.insert(3, "dataset_type", "__all__")

    dataset_summary = _summarize_group(
        raw_with_ratios,
        ["task", "dataset_source", "dataset_type", "method_family", "variant"],
    )
    dataset_summary.insert(0, "summary_scope", "dataset")

    return pd.concat([source_summary, dataset_summary], ignore_index=True, sort=False)


def run(
    dataset_specs: Sequence[DatasetSpec],
    variants: Sequence[VariantSpec],
    seed_offsets: Sequence[int],
    *,
    seed_base: int,
    n_jobs: int | None,
    n_estimators: int,
    dry_run: bool,
) -> pd.DataFrame:
    """Run the requested runtime grid."""
    total = len(dataset_specs) * len(variants) * len(seed_offsets)
    print(f"Planned rows: {total}")
    print("Datasets:")
    for spec in dataset_specs:
        print(f"  {spec.task:3s} {spec.dataset_source:9s} {spec.dataset_type}")
    print("Variants:")
    for variant in variants:
        print(f"  {variant.name:22s} family={variant.method_family}")

    if dry_run:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    real_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, str]] = {}
    git_sha = _git_sha()
    created_at_utc = datetime.now(UTC).isoformat()
    row_idx = 0

    for spec in dataset_specs:
        for seed_offset in seed_offsets:
            seed = seed_base + seed_offset
            X, y, true_info, loaded_dataset_type = _load_dataset(spec, seed, real_cache)
            spec_for_row = DatasetSpec(
                task=spec.task,
                dataset_source=spec.dataset_source,
                dataset_key=spec.dataset_key,
                dataset_type=loaded_dataset_type,
                synthetic_factory=spec.synthetic_factory,
            )
            for variant in variants:
                row_idx += 1
                print(
                    f"[{row_idx}/{total}] {spec_for_row.task} {spec_for_row.dataset_type} "
                    f"seed={seed} variant={variant.name}",
                    flush=True,
                )
                row = _row_context(
                    spec_for_row,
                    variant,
                    seed_offset,
                    seed,
                    X,
                    y,
                    n_jobs=n_jobs,
                    n_estimators=n_estimators,
                    git_sha=git_sha,
                    created_at_utc=created_at_utc,
                )
                try:
                    model = _build_model(variant, spec_for_row.task, seed, n_jobs, n_estimators)
                    row.update(_fit_runtime_metrics(X, y, true_info, model))
                    row["fit_success"] = True
                    row["error"] = ""
                    print(f"    elapsed={row['elapsed_seconds']:.3f}s", flush=True)
                except Exception as exc:
                    row["fit_success"] = False
                    row["error"] = repr(exc)
                    print(f"    ERROR {type(exc).__name__}: {exc}", flush=True)
                rows.append(row)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="all", help="Comma list from clf,reg,all")
    parser.add_argument(
        "--dataset-sources", default="all", help="Comma list from synthetic,real,all"
    )
    parser.add_argument(
        "--datasets", default="", help="Optional comma list of dataset keys or dataset_type values"
    )
    parser.add_argument("--variants", default="all", help="Comma list of variants, or all")
    parser.add_argument(
        "--seeds", default="0,1,2,3,4", help="Comma list of seed offsets added to --seed-base"
    )
    parser.add_argument(
        "--seed-base", type=int, default=RANDOM_STATE, help="Base seed for seed offsets"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=N_ESTIMATORS,
        help="Number of trees for CIF and forest references; default matches the paper",
    )
    parser.add_argument(
        "--max-synthetic-datasets", type=int, default=None, help="Limit synthetic datasets per task"
    )
    parser.add_argument(
        "--max-real-datasets", type=int, default=None, help="Limit real datasets per task"
    )
    parser.add_argument(
        "--skip-openml", action="store_true", help="Skip OpenML real classification datasets"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="n_jobs for forest references and CIF; use -1 for parallel wall-clock runs",
    )
    parser.add_argument("--no-warmup", action="store_true", help="Skip Numba warmup")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print planned grid without fitting models"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=TABLES_DIR, help="Directory for output CSV files"
    )
    parser.add_argument("--output-name", default=EXPERIMENT_NAME, help="Output file stem")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()

    tasks = _expand_choices(_split_csv(args.tasks), all_value="all", allowed=TASKS, label="tasks")
    dataset_sources = _expand_choices(
        _split_csv(args.dataset_sources),
        all_value="all",
        allowed=DATASET_SOURCES,
        label="dataset sources",
    )
    variant_names = _expand_choices(
        _split_csv(args.variants),
        all_value="all",
        allowed=tuple(ALL_VARIANTS),
        label="variants",
    )
    seed_offsets = _parse_seed_offsets(args.seeds)
    dataset_names = set(_split_csv(args.datasets)) or None

    dataset_specs: list[DatasetSpec] = []
    if "synthetic" in dataset_sources:
        dataset_specs.extend(_synthetic_specs(tasks, dataset_names, args.max_synthetic_datasets))
    if "real" in dataset_sources:
        dataset_specs.extend(
            _real_specs(tasks, dataset_names, args.max_real_datasets, args.skip_openml)
        )
    variants = [ALL_VARIANTS[name] for name in variant_names]

    if not dataset_specs:
        raise SystemExit("No datasets selected")
    if not variants:
        raise SystemExit("No variants selected")

    if not args.no_warmup and not args.dry_run:
        warmup_jit()

    raw = run(
        dataset_specs,
        variants,
        seed_offsets,
        seed_base=args.seed_base,
        n_jobs=args.n_jobs,
        n_estimators=args.n_estimators,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return

    summary = summarize(raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / f"{args.output_name}_raw.csv"
    summary_path = args.output_dir / f"{args.output_name}_summary.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved raw rows: {raw_path} ({len(raw)} rows)")
    print(f"Saved summary: {summary_path} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
