#!/usr/bin/env python3
"""Optimization ablation: isolate the effect of each CIF optimization.

Archived exploratory study; not part of the current paper-facing rebuild path.

Tests 9 CIF variants (scanning, muting, adaptive stopping, bootstrap, subsampling)
+ 3 baselines (RF, ET, CIT) on all synthetic and real datasets.
The largest ablation experiment — shows which optimizations matter most.

Usage:
    uv run python -m paper.scripts.archive.experiments.study_legacy_optimization_ablation
"""

from __future__ import annotations

import sys

import pandas as pd

from paper.scripts.experiments._common import (
    BASELINES,
    CLF_ALL,
    N_SEEDS,
    OPTIMIZATION_VARIANTS,
    RANDOM_STATE,
    REAL_CLF_NAMES,
    REG_ALL,
    aggregate_seeds,
    build_baseline,
    build_cif,
    fit_and_evaluate,
    format_line,
    load_real_clf,
    save_results,
    warmup_jit,
)

EXPERIMENT_NAME = "optimization_ablation"


def run() -> pd.DataFrame:
    """Run the full optimization ablation experiment."""
    rows: list[dict] = []

    for task, datasets in [("clf", CLF_ALL), ("reg", REG_ALL)]:
        print(f"\n--- {task.upper()} ---")
        for ds_fn in datasets:
            X_base, y_base, info_base, dtype = ds_fn(RANDOM_STATE)
            is_conf = "confounder" in dtype
            n_base = X_base.shape[1] - 20 if is_conf else None
            print(f"\n  {dtype} (n={X_base.shape[0]}, p={X_base.shape[1]})")

            for vname, overrides in OPTIMIZATION_VARIANTS.items():
                seed_results: list[dict] = []
                for s in range(N_SEEDS):
                    seed = RANDOM_STATE + s
                    X, y, info, _ = ds_fn(seed)
                    model = build_cif(task, seed, **overrides)
                    result = fit_and_evaluate(X, y, info, model, seed, task, is_conf, n_base)
                    seed_results.append(result)

                base_row = {
                    "experiment": EXPERIMENT_NAME,
                    "task": task,
                    "dataset_type": dtype,
                    "variant": vname,
                    "n_features": X_base.shape[1],
                    "n_samples": X_base.shape[0],
                }
                agg = aggregate_seeds(seed_results, base_row)
                rows.append(agg)
                print(format_line(vname, agg, is_conf))

            for method in BASELINES:
                seed_results = []
                for s in range(N_SEEDS):
                    seed = RANDOM_STATE + s
                    X, y, info, _ = ds_fn(seed)
                    model = build_baseline(method, task, seed)
                    result = fit_and_evaluate(X, y, info, model, seed, task, is_conf, n_base)
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
                print(format_line(method, agg, is_conf))

    # Real CLF datasets (no ground truth — downstream accuracy only)
    print("\n--- REAL CLF DATASETS ---")
    for ds_name in REAL_CLF_NAMES:
        try:
            X, y, dtype = load_real_clf(ds_name)
        except Exception as e:
            print(f"\n  SKIP {ds_name}: {e}")
            continue
        print(f"\n  {dtype} (n={X.shape[0]}, p={X.shape[1]})")

        for vname, overrides in OPTIMIZATION_VARIANTS.items():
            seed_results = []
            for s in range(N_SEEDS):
                seed = RANDOM_STATE + s
                model = build_cif("clf", seed, **overrides)
                result = fit_and_evaluate(X, y, None, model, seed, "clf")
                seed_results.append(result)
            base_row = {
                "experiment": EXPERIMENT_NAME,
                "task": "clf",
                "dataset_type": dtype,
                "variant": vname,
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
            }
            agg = aggregate_seeds(seed_results, base_row)
            rows.append(agg)
            print(format_line(vname, agg))

        for method in BASELINES:
            seed_results = []
            for s in range(N_SEEDS):
                seed = RANDOM_STATE + s
                model = build_baseline(method, "clf", seed)
                result = fit_and_evaluate(X, y, None, model, seed, "clf")
                seed_results.append(result)
            base_row = {
                "experiment": EXPERIMENT_NAME,
                "task": "clf",
                "dataset_type": dtype,
                "variant": method,
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
            }
            agg = aggregate_seeds(seed_results, base_row)
            rows.append(agg)
            print(format_line(method, agg))

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
