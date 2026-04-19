#!/usr/bin/env python3
"""Scaling curves: CIF vs RF vs ET wall-clock time vs n and p.

Archived exploratory study; not part of the current paper-facing rebuild path.

Tests time scaling with n (200-5000, p=100) and p (20-1000, n=1000).
Shows CIF overhead relative to RF/ET as dataset size grows.

Usage:
    uv run python -m paper.scripts.backlog.experiments.study_runtime_scaling
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification

from paper.scripts.experiments._common import (
    RANDOM_STATE,
    build_baseline,
    build_cif,
    save_results,
    warmup_jit,
)

EXPERIMENT_NAME = "scaling_curves"

TIMING_REPEATS = 3


def _build_model(variant: str, task: str, seed: int) -> BaseEstimator:
    """Build a CIF, RF, or ET model for timing."""
    if variant == "cif":
        return build_cif(task, seed)
    return build_baseline(variant, task, seed)


def run() -> pd.DataFrame:
    """Run the scaling curves experiment."""
    n_grid = [200, 500, 1000, 2000, 5000]
    p_grid = [20, 50, 100, 500, 1000]

    rows: list[dict] = []

    # Time vs n (fixed p=100, k=10)
    print("\n--- Time vs n (p=100) ---")
    for n in n_grid:
        for variant in ["cif", "rf", "et"]:
            times: list[float] = []
            for rep in range(TIMING_REPEATS):
                seed = RANDOM_STATE + rep
                X, y = make_classification(
                    n_samples=n, n_features=100, n_informative=10,
                    random_state=seed, shuffle=False,
                )
                model = _build_model(variant, "clf", seed)
                t0 = time.perf_counter()
                model.fit(X, y)
                times.append(time.perf_counter() - t0)

            rows.append({
                "experiment": EXPERIMENT_NAME,
                "sweep": "n",
                "variant": variant,
                "n_samples": n,
                "n_features": 100,
                "elapsed_median": float(np.median(times)),
                "elapsed_std": float(np.std(times)),
            })
            print(f"  n={n:5d} {variant:4s}: {np.median(times):.2f}s")

    # Time vs p (fixed n=1000, k=min(10, p//2))
    print("\n--- Time vs p (n=1000) ---")
    for p in p_grid:
        k = min(10, p // 2)
        for variant in ["cif", "rf", "et"]:
            times = []
            for rep in range(TIMING_REPEATS):
                seed = RANDOM_STATE + rep
                X, y = make_classification(
                    n_samples=1000, n_features=p, n_informative=k,
                    random_state=seed, shuffle=False,
                )
                model = _build_model(variant, "clf", seed)
                t0 = time.perf_counter()
                model.fit(X, y)
                times.append(time.perf_counter() - t0)

            rows.append({
                "experiment": EXPERIMENT_NAME,
                "sweep": "p",
                "variant": variant,
                "n_samples": 1000,
                "n_features": p,
                "elapsed_median": float(np.median(times)),
                "elapsed_std": float(np.std(times)),
            })
            print(f"  p={p:5d} {variant:4s}: {np.median(times):.2f}s")

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
