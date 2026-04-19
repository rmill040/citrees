#!/usr/bin/env python3
"""Alpha threshold sweep: how does alpha_selector affect feature ranking quality?

Archived exploratory study; not part of the current paper-facing rebuild path.

Tests 6 alpha levels (0.001 to 0.50) on all synthetic datasets (clf + reg).
Permissive alpha helps weak-signal datasets but hurts precision on noisy ones.

Usage:
    uv run python -m paper.scripts.backlog.experiments.study_alpha_selector_sweep
"""

from __future__ import annotations

import sys

import pandas as pd

from paper.scripts.experiments._common import (
    CLF_ALL,
    N_SEEDS,
    RANDOM_STATE,
    REG_ALL,
    aggregate_seeds,
    build_cif,
    fit_and_evaluate,
    format_line,
    save_results,
    warmup_jit,
)

EXPERIMENT_NAME = "alpha_sweep"


def run() -> pd.DataFrame:
    """Run the alpha sweep experiment."""
    alpha_grid = [0.001, 0.01, 0.05, 0.10, 0.20, 0.50]

    rows = []
    for task, datasets in [("clf", CLF_ALL), ("reg", REG_ALL)]:
        print(f"\n--- {task.upper()} ---")
        for ds_fn in datasets:
            X_base, y_base, info_base, dtype = ds_fn(RANDOM_STATE)
            is_conf = "confounder" in dtype
            n_base = X_base.shape[1] - 20 if is_conf else None
            print(f"\n  {dtype} (n={X_base.shape[0]}, p={X_base.shape[1]})")

            for alpha in alpha_grid:
                seed_results = []
                for s in range(N_SEEDS):
                    seed = RANDOM_STATE + s
                    X, y, info, _ = ds_fn(seed)
                    model = build_cif(task, seed, alpha_selector=alpha)
                    result = fit_and_evaluate(X, y, info, model, seed, task, is_conf, n_base)
                    seed_results.append(result)

                vname = f"alpha={alpha}"
                base_row = {
                    "experiment": EXPERIMENT_NAME,
                    "task": task,
                    "dataset_type": dtype,
                    "variant": vname,
                    "alpha_selector": alpha,
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
