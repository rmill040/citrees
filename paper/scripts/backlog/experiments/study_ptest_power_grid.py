#!/usr/bin/env python3
"""Type I/II error and power curves for CIF permutation tests.

Archived exploratory study; not part of the current paper-facing rebuild path.

Two experiments:
1. Type I error (null): no signal, measure rejection rate (should be <= alpha).
2. Power curves (alternative): varying signal strength, measure detection rate.

Compares adaptive early stopping vs fixed-B across multiple B and alpha values.
Archived calibration check only; adaptive stopping is not a theorem-level
fixed-B p-value guarantee.

Usage:
    uv run python -m paper.scripts.backlog.experiments.study_ptest_power_grid
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from citrees._selector import ptest_mc
from paper.scripts.experiments._common import (
    RANDOM_STATE,
    save_results,
    warmup_jit,
)

EXPERIMENT_NAME = "power_analysis"

# Power analysis config
N_SIMULATIONS = 500
N_SAMPLES = 200
N_FEATURES = 20
B_GRID = (49, 99, 199, 499, 999)
ALPHA_GRID = (0.01, 0.05, 0.10)
CLASS_SEP_GRID = (0.1, 0.5, 1.0, 2.0)


def _make_null_data(n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate pure noise data: no relationship between X and y."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = rng.integers(0, 2, size=n).astype(np.int64)
    return X, y


def _make_alt_data(
    n: int, p: int, class_sep: float, seed: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate data with signal in feature 0."""
    X, y = make_classification(
        n_samples=n, n_features=p, n_informative=1, n_redundant=0,
        n_clusters_per_class=1, class_sep=class_sep, flip_y=0.0,
        random_state=seed, shuffle=False,
    )
    y = y.astype(np.int64)
    return X, y, 0


def _run_ptest(
    x: np.ndarray, y: np.ndarray,
    n_resamples: int, alpha: float,
    early_stopping: str | None, random_state: int,
) -> float:
    """Run a single MC permutation test on one feature."""
    n_classes = int(len(np.unique(y)))
    return ptest_mc(
        x=x, y=y, n_classes=n_classes,
        n_resamples=n_resamples, early_stopping=early_stopping,
        alpha=alpha, random_state=random_state,
    )


def _run_null_experiment() -> pd.DataFrame:
    """Measure empirical Type I error rate under the null hypothesis."""
    print("=== Experiment 1: Type I Error (null distribution) ===")
    rows: list[dict] = []

    for alpha in ALPHA_GRID:
        for B in B_GRID:
            for stopping_label, stopping_val in [("adaptive", "adaptive"), ("fixed_B", None)]:
                rejections = 0
                pvalues: list[float] = []

                for sim in range(N_SIMULATIONS):
                    seed = RANDOM_STATE + sim
                    X, y = _make_null_data(N_SAMPLES, N_FEATURES, seed)
                    pval = _run_ptest(X[:, 0], y, B, alpha, stopping_val, seed)
                    pvalues.append(pval)
                    if pval <= alpha:
                        rejections += 1

                rejection_rate = rejections / N_SIMULATIONS
                rows.append({
                    "experiment": EXPERIMENT_NAME,
                    "task": "clf",
                    "experiment_type": "null",
                    "alpha": alpha,
                    "B": B,
                    "stopping": stopping_label,
                    "class_sep": 0.0,
                    "n_simulations": N_SIMULATIONS,
                    "rejections": rejections,
                    "rejection_rate": round(rejection_rate, 4),
                    "mean_pvalue": round(float(np.mean(pvalues)), 4),
                    "median_pvalue": round(float(np.median(pvalues)), 4),
                })

                flag = " ***" if rejection_rate > alpha + 0.02 else ""
                print(f"  alpha={alpha:.2f}, B={B:>4d}, {stopping_label:>10s}: "
                      f"reject={rejection_rate:.3f} (expected<={alpha:.2f}){flag}")

    print()
    return pd.DataFrame(rows)


def _run_power_experiment() -> pd.DataFrame:
    """Measure statistical power under alternative hypotheses."""
    print("=== Experiment 2: Power Curves (alternative) ===")
    rows: list[dict] = []

    for alpha in ALPHA_GRID:
        print(f"\n  --- alpha={alpha:.2f} ---")
        for class_sep in CLASS_SEP_GRID:
            for B in B_GRID:
                for stopping_label, stopping_val in [("adaptive", "adaptive"), ("fixed_B", None)]:
                    rejections = 0
                    pvalues: list[float] = []

                    for sim in range(N_SIMULATIONS):
                        seed = RANDOM_STATE + sim
                        X, y, info_idx = _make_alt_data(N_SAMPLES, N_FEATURES, class_sep, seed)
                        pval = _run_ptest(X[:, info_idx], y, B, alpha, stopping_val, seed)
                        pvalues.append(pval)
                        if pval <= alpha:
                            rejections += 1

                    power = rejections / N_SIMULATIONS
                    rows.append({
                        "experiment": EXPERIMENT_NAME,
                        "task": "clf",
                        "experiment_type": "alternative",
                        "alpha": alpha,
                        "B": B,
                        "stopping": stopping_label,
                        "class_sep": class_sep,
                        "n_simulations": N_SIMULATIONS,
                        "rejections": rejections,
                        "rejection_rate": round(power, 4),
                        "mean_pvalue": round(float(np.mean(pvalues)), 4),
                        "median_pvalue": round(float(np.median(pvalues)), 4),
                    })
                    print(f"    sep={class_sep:.1f}, B={B:>4d}, {stopping_label:>10s}: "
                          f"power={power:.3f}")

    print()
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    """Run both null and power experiments."""
    df_null = _run_null_experiment()
    df_power = _run_power_experiment()
    return pd.concat([df_null, df_power], ignore_index=True)


def main() -> None:
    """Entry point."""
    sys.stdout.reconfigure(line_buffering=True)
    print(f"=== {EXPERIMENT_NAME} ===")
    print(f"  n_simulations={N_SIMULATIONS}, n={N_SAMPLES}, p={N_FEATURES}")
    print(f"  B_grid={B_GRID}")
    print(f"  alpha_grid={ALPHA_GRID}")
    print(f"  class_sep_grid={CLASS_SEP_GRID}")
    print()
    warmup_jit()

    df = run()
    path = save_results(df, EXPERIMENT_NAME)
    print(f"\nSaved: {path} ({len(df)} rows)")

    # Summary
    print("\n=== Key Finding ===")
    null_rows = df[df["experiment_type"] == "null"]
    adaptive_null = null_rows[null_rows["stopping"] == "adaptive"]
    violations = adaptive_null[adaptive_null["rejection_rate"] > adaptive_null["alpha"] + 0.02]
    if violations.empty:
        print("  Adaptive stopping does NOT inflate Type I error beyond tolerance.")
    else:
        print("  WARNING: Adaptive stopping inflates Type I error in some settings:")
        print(violations[["alpha", "B", "rejection_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
