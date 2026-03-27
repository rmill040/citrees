#!/usr/bin/env python3
"""Bootstrap vs feature subsampling: can feature subsampling replace bootstrap?

Tests 8 variants (boot/noboot × allfeats/sqrt/log2/half) on 6 clf datasets.
Shows bootstrap + feature subsampling is complementary, not redundant.

Usage:
    uv run python -m paper.scripts.experiments.bootstrap_vs_subsampling
"""

from __future__ import annotations

import sys

import pandas as pd

from paper.scripts.experiments._common import (
    N_SEEDS,
    RANDOM_STATE,
    aggregate_seeds,
    build_cif,
    clf_bias,
    clf_confounder,
    clf_standard_easy,
    clf_standard_hard,
    clf_toeplitz,
    clf_weak_signal,
    fit_and_evaluate,
    format_line,
    save_results,
    warmup_jit,
)

EXPERIMENT_NAME = "bootstrap_vs_subsampling"


def run() -> pd.DataFrame:
    """Run the bootstrap vs feature subsampling experiment."""
    variants = [
        ("boot_allfeats", dict()),
        ("boot_sqrt", dict(max_features="sqrt")),
        ("boot_log2", dict(max_features="log2")),
        ("boot_half", dict(max_features=0.5)),
        ("noboot_allfeats", dict(bootstrap=False, sampling_method=None)),
        ("noboot_sqrt", dict(bootstrap=False, sampling_method=None, max_features="sqrt")),
        ("noboot_log2", dict(bootstrap=False, sampling_method=None, max_features="log2")),
        ("noboot_half", dict(bootstrap=False, sampling_method=None, max_features=0.5)),
    ]

    datasets = [clf_standard_easy, clf_standard_hard, clf_weak_signal,
                clf_toeplitz, clf_confounder, clf_bias]

    rows: list[dict] = []
    for ds_fn in datasets:
        X_base, y_base, info_base, dtype = ds_fn(RANDOM_STATE)
        is_conf = "confounder" in dtype
        n_base = X_base.shape[1] - 20 if is_conf else None
        print(f"\n  {dtype} (n={X_base.shape[0]}, p={X_base.shape[1]})")

        for vname, overrides in variants:
            seed_results: list[dict] = []
            for s in range(N_SEEDS):
                seed = RANDOM_STATE + s
                X, y, info, _ = ds_fn(seed)
                model = build_cif("clf", seed, **overrides)
                result = fit_and_evaluate(X, y, info, model, seed, "clf", is_conf, n_base)
                seed_results.append(result)

            base_row = {
                "experiment": EXPERIMENT_NAME,
                "task": "clf",
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
