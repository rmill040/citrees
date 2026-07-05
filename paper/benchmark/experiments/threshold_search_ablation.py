#!/usr/bin/env python3
"""Threshold-search ablation: isolate effect of threshold_method and max_thresholds.

Tests 7 CIF variants spanning exact vs histogram threshold methods and different
max_thresholds budgets, plus 3 baselines, on all synthetic and real datasets.
Closes the last major dedicated-ablation gap in the practical regime.

Variants:
    exact_all      — threshold_method="exact", all exact midpoints
    histogram_32   — threshold_method="histogram", max_thresholds=32
    histogram_64   — threshold_method="histogram", max_thresholds=64
    histogram_128  — threshold_method="histogram", max_thresholds=128
    histogram_256  — threshold_method="histogram", max_thresholds=256  (paper default)
    histogram_512  — threshold_method="histogram", max_thresholds=512
    histogram_all  — threshold_method="histogram", max_thresholds=None

Note:
    The exact threshold generator currently ignores max_thresholds and always
    returns all exact midpoints, so a would-be exact_256 variant would be a
    duplicate of exact_all rather than a distinct configuration.

Usage:
    uv run python -m paper.benchmark.experiments.threshold_search_ablation
"""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd

from paper.benchmark.experiments.experiment_common import (
    BASELINES,
    CLF_ALL,
    N_SEEDS,
    RANDOM_STATE,
    REAL_CLF_NAMES,
    REAL_REG_NAMES,
    REG_ALL,
    aggregate_seeds,
    build_baseline,
    build_cif,
    fit_and_evaluate_with_structure,
    format_line_with_structure,
    load_real_clf,
    load_real_reg,
    save_results,
    warmup_jit,
)

EXPERIMENT_NAME = "threshold_search_ablation"

THRESHOLD_VARIANTS: dict[str, dict[str, Any]] = {
    "exact_all": dict(threshold_method="exact", max_thresholds=None),
    "histogram_32": dict(threshold_method="histogram", max_thresholds=32),
    "histogram_64": dict(threshold_method="histogram", max_thresholds=64),
    "histogram_128": dict(threshold_method="histogram", max_thresholds=128),
    "histogram_256": dict(threshold_method="histogram", max_thresholds=256),
    "histogram_512": dict(threshold_method="histogram", max_thresholds=512),
    "histogram_all": dict(threshold_method="histogram", max_thresholds=None),
}


def _run_synthetic(rows: list[dict[str, Any]]) -> None:
    """Run threshold variants + baselines on all synthetic datasets."""
    for task, datasets in [("clf", CLF_ALL), ("reg", REG_ALL)]:
        print(f"\n--- {task.upper()} SYNTHETIC ---")
        for ds_fn in datasets:
            X_base, y_base, info_base, dtype = ds_fn(RANDOM_STATE)
            is_conf = "confounder" in dtype
            n_base = X_base.shape[1] - 20 if is_conf else None
            print(f"\n  {dtype} (n={X_base.shape[0]}, p={X_base.shape[1]})")

            for vname, overrides in THRESHOLD_VARIANTS.items():
                seed_results: list[dict[str, float]] = []
                for s in range(N_SEEDS):
                    seed = RANDOM_STATE + s
                    X, y, info, _ = ds_fn(seed)
                    model = build_cif(task, seed, **overrides)
                    result = fit_and_evaluate_with_structure(
                        X,
                        y,
                        info,
                        model,
                        seed,
                        task,
                        is_conf,
                        n_base,
                    )
                    seed_results.append(result)

                base_row: dict[str, Any] = {
                    "experiment": EXPERIMENT_NAME,
                    "task": task,
                    "dataset_type": dtype,
                    "variant": vname,
                    "n_features": X_base.shape[1],
                    "n_samples": X_base.shape[0],
                }
                agg = aggregate_seeds(seed_results, base_row)
                rows.append(agg)
                print(format_line_with_structure(vname, agg, is_conf))

            for method in BASELINES:
                seed_results = []
                for s in range(N_SEEDS):
                    seed = RANDOM_STATE + s
                    X, y, info, _ = ds_fn(seed)
                    model = build_baseline(method, task, seed)
                    result = fit_and_evaluate_with_structure(
                        X,
                        y,
                        info,
                        model,
                        seed,
                        task,
                        is_conf,
                        n_base,
                    )
                    seed_results.append(result)

                base_row = {
                    "experiment": EXPERIMENT_NAME,
                    "task": task,
                    "dataset_type": dtype,
                    "variant": method,
                    "n_features": X_base.shape[1],
                    "n_samples": X_base.shape[0],
                }
                agg = aggregate_seeds(seed_results, base_row)
                rows.append(agg)
                print(format_line_with_structure(method, agg, is_conf))


def _run_real(rows: list[dict[str, Any]]) -> None:
    """Run threshold variants + baselines on real clf and reg datasets."""
    print("\n--- REAL CLF DATASETS ---")
    for ds_name in REAL_CLF_NAMES:
        try:
            X, y, dtype = load_real_clf(ds_name)
        except Exception as e:
            print(f"\n  SKIP {ds_name}: {e}")
            continue
        _run_on_real_dataset(X, y, dtype, "clf", rows)

    print("\n--- REAL REG DATASETS ---")
    for ds_name in REAL_REG_NAMES:
        try:
            X, y, dtype = load_real_reg(ds_name)
        except Exception as e:
            print(f"\n  SKIP {ds_name}: {e}")
            continue
        _run_on_real_dataset(X, y, dtype, "reg", rows)


def _run_on_real_dataset(
    X: pd.DataFrame,
    y: pd.DataFrame,
    dtype: str,
    task: str,
    rows: list[dict[str, Any]],
) -> None:
    """Run all threshold variants + baselines on a single real dataset."""
    print(f"\n  {dtype} (n={X.shape[0]}, p={X.shape[1]})")

    for vname, overrides in THRESHOLD_VARIANTS.items():
        seed_results: list[dict[str, float]] = []
        for s in range(N_SEEDS):
            seed = RANDOM_STATE + s
            model = build_cif(task, seed, **overrides)
            result = fit_and_evaluate_with_structure(X, y, None, model, seed, task)
            seed_results.append(result)
        base_row: dict[str, Any] = {
            "experiment": EXPERIMENT_NAME,
            "task": task,
            "dataset_type": dtype,
            "variant": vname,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
        }
        agg = aggregate_seeds(seed_results, base_row)
        rows.append(agg)
        print(format_line_with_structure(vname, agg))

    for method in BASELINES:
        seed_results = []
        for s in range(N_SEEDS):
            seed = RANDOM_STATE + s
            model = build_baseline(method, task, seed)
            result = fit_and_evaluate_with_structure(X, y, None, model, seed, task)
            seed_results.append(result)
        base_row = {
            "experiment": EXPERIMENT_NAME,
            "task": task,
            "dataset_type": dtype,
            "variant": method,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
        }
        agg = aggregate_seeds(seed_results, base_row)
        rows.append(agg)
        print(format_line_with_structure(method, agg))


def run() -> pd.DataFrame:
    """Run the full threshold-search ablation experiment."""
    rows: list[dict] = []
    _run_synthetic(rows)
    _run_real(rows)
    return pd.DataFrame(rows)


def main() -> None:
    """Entry point."""
    sys.stdout.reconfigure(line_buffering=True)
    print(f"=== {EXPERIMENT_NAME} ===")
    warmup_jit()
    df = run()
    path = save_results(df, EXPERIMENT_NAME)
    print(f"\nSaved: {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
