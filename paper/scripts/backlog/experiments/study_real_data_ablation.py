#!/usr/bin/env python3
"""Real dataset ablation: CIF variants and baselines on real clf/reg datasets.

Archived exploratory study; not part of the current paper-facing rebuild path.

Tests 6 CIF ablation variants + 3 baselines (RF, ET, CIT) on 7 clf and 2 reg
real-world datasets. No ground truth — evaluates downstream accuracy only.

Usage:
    uv run python -m paper.scripts.backlog.experiments.study_real_data_ablation
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd

from paper.scripts.experiments._common import (
    BASELINES,
    N_SEEDS,
    RANDOM_STATE,
    REAL_ABLATION_VARIANTS,
    REAL_CLF_NAMES,
    REAL_REG_NAMES,
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

EXPERIMENT_NAME = "real_dataset_ablation"


def _run_on_dataset(
    X: np.ndarray, y: np.ndarray, dtype: str, task: str, rows: list[dict[str, Any]]
) -> None:
    """Run all CIF variants + baselines on a single dataset."""
    print(f"\n  {dtype} (n={X.shape[0]}, p={X.shape[1]})")

    for vname, overrides in REAL_ABLATION_VARIANTS.items():
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
    """Run the real dataset ablation experiment."""
    rows: list[dict] = []

    print("\n--- REAL CLF DATASETS ---")
    for ds_name in REAL_CLF_NAMES:
        try:
            X, y, dtype = load_real_clf(ds_name)
        except Exception as e:
            print(f"\n  SKIP {ds_name}: {e}")
            continue
        _run_on_dataset(X, y, dtype, "clf", rows)

    print("\n--- REAL REG DATASETS ---")
    for ds_name in REAL_REG_NAMES:
        try:
            X, y, dtype = load_real_reg(ds_name)
        except Exception as e:
            print(f"\n  SKIP {ds_name}: {e}")
            continue
        _run_on_dataset(X, y, dtype, "reg", rows)

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
