#!/usr/bin/env python3
"""Sample size curves: minimum viable n for CIF feature selection.

Tests 6 sample sizes (50 to 2000) on three base dataset configs.
Shows CIF needs ~200 samples to match RF precision on easy problems.

Usage:
    uv run python -m paper.scripts.experiments.sample_size_curves
"""

from __future__ import annotations

import sys

import pandas as pd
from sklearn.datasets import make_classification

from paper.scripts.experiments._common import (
    N_SEEDS,
    RANDOM_STATE,
    aggregate_seeds,
    build_baseline,
    build_cif,
    fit_and_evaluate,
    save_results,
    shuffle_columns,
    warmup_jit,
)

EXPERIMENT_NAME = "sample_size_curves"


def run() -> pd.DataFrame:
    """Run the sample size curves experiment."""
    n_grid = [50, 100, 200, 500, 1000, 2000]

    base_configs = [
        {"name": "easy", "n_features": 100, "n_informative": 10, "class_sep": 1.5},
        {"name": "moderate", "n_features": 100, "n_informative": 10, "class_sep": 0.5},
        {"name": "highdim", "n_features": 500, "n_informative": 5, "class_sep": 1.0},
    ]

    rows: list[dict] = []
    for base_cfg in base_configs:
        bname = base_cfg["name"]
        print(f"\n=== Base: {bname} (p={base_cfg['n_features']}, k={base_cfg['n_informative']}) ===")
        for variant in ["cif", "rf", "et"]:
            print(f"\n--- {variant.upper()} ---")
            for n in n_grid:
                seed_results: list[dict] = []
                for s in range(N_SEEDS):
                    seed = RANDOM_STATE + s
                    X, y = make_classification(
                        n_samples=n,
                        n_features=base_cfg["n_features"],
                        n_informative=base_cfg["n_informative"],
                        n_redundant=0,
                        class_sep=base_cfg["class_sep"],
                        flip_y=0.0,
                        random_state=seed,
                        shuffle=False,
                    )
                    X, info = shuffle_columns(X, base_cfg["n_informative"], seed)

                    if variant == "cif":
                        model = build_cif("clf", seed)
                    else:
                        model = build_baseline(variant, "clf", seed)

                    result = fit_and_evaluate(X, y, info, model, seed, "clf")
                    seed_results.append(result)

                base_row = {
                    "experiment": EXPERIMENT_NAME,
                    "task": "clf",
                    "variant": variant,
                    "dataset_type": bname,
                    "n_samples": n,
                    "n_features": base_cfg["n_features"],
                    "n_informative": base_cfg["n_informative"],
                }
                agg = aggregate_seeds(seed_results, base_row)
                rows.append(agg)
                print(f"  n={n:5d}: P@10={agg.get('precision_at_10_mean', 0):.3f} "
                      f"F1={agg.get('f1_at_10_mean', 0):.3f} "
                      f"ds={agg.get('lr_ba_mean', 0):.3f} "
                      f"t={agg.get('elapsed_seconds_mean', 0):.1f}s")

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
