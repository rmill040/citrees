#!/usr/bin/env python3
"""Strictness continuum: CIF from conservative to RF-like.

Tests 9 CIF configs along the strictness spectrum (strict default to wide-open
alpha=0.99) plus RF/ET baselines and optional R baselines (r_ctree, r_cforest).
Captures tree structure stats (depth, n_features_used) to show the causal chain.

Usage:
    uv run python -m paper.scripts.experiments.strictness_continuum
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from paper.scripts.experiments._common import (
    CLF_ALL,
    K,
    N_SEEDS,
    RANDOM_STATE,
    aggregate_seeds,
    build_baseline,
    build_cif,
    compute_confounder_rate,
    compute_spread,
    downstream_clf,
    downstream_reg,
    fit_and_evaluate_with_structure,
    format_line,
    format_line_with_structure,
    reg_friedman,
    reg_linear,
    reg_toeplitz,
    reg_weak_signal,
    save_results,
    warmup_jit,
)
from paper.scripts.utils.metrics import f1_at_k, precision_at_k, recall_at_k

EXPERIMENT_NAME = "strictness_continuum"

CIF_CONFIGS: list[tuple[str, dict]] = [
    ("strict_default",
     dict(alpha_selector=0.05, adjust_alpha_selector=True,
          early_stopping_selector="adaptive")),
    ("no_bonf_a05",
     dict(alpha_selector=0.05, adjust_alpha_selector=False,
          early_stopping_selector="adaptive")),
    ("no_bonf_a10",
     dict(alpha_selector=0.10, adjust_alpha_selector=False,
          early_stopping_selector="adaptive")),
    ("no_bonf_a20",
     dict(alpha_selector=0.20, adjust_alpha_selector=False,
          early_stopping_selector="adaptive")),
    ("no_bonf_a50",
     dict(alpha_selector=0.50, adjust_alpha_selector=False,
          early_stopping_selector="adaptive")),
    ("wide_open_a99",
     dict(alpha_selector=0.99, adjust_alpha_selector=False,
          early_stopping_selector="adaptive")),
    ("no_bonf_a20_no_es",
     dict(alpha_selector=0.20, adjust_alpha_selector=False,
          early_stopping_selector=None, early_stopping_splitter=None)),
    ("no_bonf_a05_no_boot",
     dict(alpha_selector=0.05, adjust_alpha_selector=False,
          bootstrap=False, sampling_method=None)),
    ("optimized_a20",
     dict(alpha_selector=0.20, adjust_alpha_selector=False,
          max_samples=0.5)),
]


def run() -> pd.DataFrame:
    """Run the strictness continuum experiment."""
    clf_datasets = CLF_ALL
    reg_datasets = [reg_friedman, reg_linear, reg_toeplitz, reg_weak_signal]

    rows: list[dict] = []

    for task, datasets in [("clf", clf_datasets), ("reg", reg_datasets)]:
        print(f"\n--- {task.upper()} ---")
        for ds_fn in datasets:
            X_base, y_base, info_base, dtype = ds_fn(RANDOM_STATE)
            is_conf = "confounder" in dtype
            n_base = X_base.shape[1] - 20 if is_conf else None
            print(f"\n  {dtype} (n={X_base.shape[0]}, p={X_base.shape[1]})")

            # CIF variants
            for vname, overrides in CIF_CONFIGS:
                seed_results: list[dict] = []
                for s in range(N_SEEDS):
                    seed = RANDOM_STATE + s
                    X, y, info, _ = ds_fn(seed)
                    model = build_cif(task, seed, **overrides)
                    result = fit_and_evaluate_with_structure(
                        X, y, info, model, seed, task, is_conf, n_base)
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
                print(format_line_with_structure(vname, agg, is_conf))

            # RF and ET baselines (with tree stats)
            for method in ["rf", "et"]:
                seed_results = []
                for s in range(N_SEEDS):
                    seed = RANDOM_STATE + s
                    X, y, info, _ = ds_fn(seed)
                    model = build_baseline(method, task, seed)
                    result = fit_and_evaluate_with_structure(
                        X, y, info, model, seed, task, is_conf, n_base)
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

            # R baselines (if available)
            try:
                from paper.scripts.pipeline.r_methods import r_ctree_ranking, r_cforest_ranking

                r_task = "classification" if task == "clf" else "regression"
                for r_method, r_func, r_params in [
                    ("r_ctree_bonf", r_ctree_ranking, dict(testtype="Bonferroni")),
                    ("r_ctree_mc", r_ctree_ranking, dict(testtype="MonteCarlo", nresample=199)),
                    ("r_cforest_bonf", r_cforest_ranking,
                     dict(testtype="Bonferroni", ntree=100)),
                    ("r_cforest_mc", r_cforest_ranking,
                     dict(testtype="MonteCarlo", nresample=199, ntree=100)),
                ]:
                    seed_results = []
                    for s in range(N_SEEDS):
                        seed = RANDOM_STATE + s
                        X, y, info, _ = ds_fn(seed)
                        t0 = time.perf_counter()
                        ranking = list(r_func(X, y, task=r_task, **r_params))
                        elapsed = time.perf_counter() - t0

                        result: dict = {
                            "elapsed_seconds": elapsed,
                            "spread_at_10": compute_spread(ranking, X.shape[1], K),
                        }
                        if info is not None:
                            result["precision_at_10"] = precision_at_k(ranking, info, K)
                            result["recall_at_10"] = recall_at_k(ranking, info, K)
                            result["f1_at_10"] = f1_at_k(ranking, info, K)
                        if task == "clf":
                            result.update(downstream_clf(X, y, ranking, K, seed))
                        else:
                            result.update(downstream_reg(X, y, ranking, K, seed))
                        if is_conf and n_base is not None:
                            result["confounder_rate_at_5"] = compute_confounder_rate(
                                ranking, info, n_base, 5)
                            result["confounder_rate_at_10"] = compute_confounder_rate(
                                ranking, info, n_base, 10)
                        seed_results.append(result)

                    base_row = {
                        "experiment": EXPERIMENT_NAME,
                        "task": task,
                        "dataset_type": dtype,
                        "variant": r_method,
                        "n_features": X_base.shape[1],
                        "n_samples": X_base.shape[0],
                    }
                    agg = aggregate_seeds(seed_results, base_row)
                    rows.append(agg)
                    print(format_line(r_method, agg, is_conf))

            except ImportError:
                print("  [R methods unavailable -- skipping r_ctree/r_cforest]")

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
