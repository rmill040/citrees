#!/usr/bin/env python3
"""n_estimators sweep: how does ensemble size affect ranking quality?

Archived exploratory study; not part of the current paper-facing rebuild path.

Tests 6 levels (1 to 100) on challenging synthetic datasets (clf + reg).
Shows diminishing returns after ~25 trees for ranking quality.

Usage:
    uv run python -m paper.scripts.backlog.experiments.study_forest_size_sweep
"""

from __future__ import annotations

import sys

import pandas as pd

from paper.scripts.experiments._common import (
    CLF_CHALLENGING,
    N_SEEDS,
    RANDOM_STATE,
    REG_CHALLENGING,
    aggregate_seeds,
    build_cif,
    fit_and_evaluate,
    format_line,
    save_results,
    warmup_jit,
)

EXPERIMENT_NAME = "n_estimators_sweep"


def run() -> pd.DataFrame:
    """Run the n_estimators sweep experiment."""
    n_est_grid = [1, 5, 10, 25, 50, 100]

    rows: list[dict] = []
    for task, datasets in [("clf", CLF_CHALLENGING), ("reg", REG_CHALLENGING)]:
        print(f"\n--- {task.upper()} ---")
        for ds_fn in datasets:
            X_base, y_base, info_base, dtype = ds_fn(RANDOM_STATE)
            is_conf = "confounder" in dtype
            n_base = X_base.shape[1] - 20 if is_conf else None
            print(f"\n  {dtype} (n={X_base.shape[0]}, p={X_base.shape[1]})")

            for n_est in n_est_grid:
                seed_results: list[dict] = []
                for s in range(N_SEEDS):
                    seed = RANDOM_STATE + s
                    X, y, info, _ = ds_fn(seed)
                    model = build_cif(task, seed, n_estimators=n_est)
                    result = fit_and_evaluate(X, y, info, model, seed, task, is_conf, n_base)
                    seed_results.append(result)

                vname = f"n_est={n_est}"
                base_row = {
                    "experiment": EXPERIMENT_NAME,
                    "task": task,
                    "dataset_type": dtype,
                    "variant": vname,
                    "n_estimators": n_est,
                    "n_features": X_base.shape[1],
                    "n_samples": X_base.shape[0],
                }
                agg = aggregate_seeds(seed_results, base_row)
                rows.append(agg)
                print(format_line(vname, agg, is_conf))

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
