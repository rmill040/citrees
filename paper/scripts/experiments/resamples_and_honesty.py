#!/usr/bin/env python3
"""n_resamples and honesty sweep: does more permutation precision or honesty help?

Tests 5 n_resamples levels (49 to 999) and honesty on/off on challenging datasets.
Shows diminishing returns past B=199 and minor honesty effect on ranking quality.

Usage:
    uv run python -m paper.scripts.experiments.resamples_and_honesty
"""

from __future__ import annotations

import sys

import pandas as pd

from paper.scripts.experiments._common import (
    N_SEEDS,
    RANDOM_STATE,
    aggregate_seeds,
    build_cif,
    clf_confounder,
    clf_standard_easy,
    clf_toeplitz,
    clf_weak_signal,
    fit_and_evaluate,
    format_line,
    reg_toeplitz,
    reg_weak_signal,
    save_results,
    warmup_jit,
)

EXPERIMENT_NAME = "resamples_and_honesty"


def run() -> pd.DataFrame:
    """Run the n_resamples + honesty sweep experiment."""
    variants = [
        ("B=49", dict(n_resamples_selector=49, n_resamples_splitter=49)),
        ("B=99", dict(n_resamples_selector=99, n_resamples_splitter=99)),
        ("B=199", dict(n_resamples_selector=199, n_resamples_splitter=199)),
        ("B=499", dict(n_resamples_selector=499, n_resamples_splitter=499)),
        ("B=999", dict(n_resamples_selector=999, n_resamples_splitter=999)),
        ("honesty", dict(honesty=True)),
        ("no_honesty", dict(honesty=False)),
    ]

    clf_datasets = [clf_standard_easy, clf_weak_signal, clf_toeplitz, clf_confounder]
    reg_datasets = [reg_toeplitz, reg_weak_signal]

    rows: list[dict] = []
    for task, task_datasets in [("clf", clf_datasets), ("reg", reg_datasets)]:
        print(f"\n--- {task.upper()} ---")
        for ds_fn in task_datasets:
            X_base, y_base, info_base, dtype = ds_fn(RANDOM_STATE)
            is_conf = "confounder" in dtype
            n_base = X_base.shape[1] - 20 if is_conf else None
            print(f"\n  {dtype} (n={X_base.shape[0]}, p={X_base.shape[1]})")

            for vname, overrides in variants:
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
