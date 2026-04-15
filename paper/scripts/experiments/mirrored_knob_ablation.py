#!/usr/bin/env python3
"""Mirrored knob ablation: symmetric CIF knob study across CLF and REG.

Tests 7 CIF variants (default, no_adaptive, no_scan, no_mute, no_bootstrap,
no_bonferroni, all_off) + 3 baselines (RF, ET, CIT) on all synthetic and real
datasets for BOTH classification and regression. Evaluates all downstream models
(LR, SVM, KNN for clf; Ridge, SVR, KNN for reg) at multiple k values.

Closes the asymmetry gap where some knobs were only studied on one task.

Usage:
    uv run python -m paper.scripts.experiments.mirrored_knob_ablation
"""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold

from paper.scripts.experiments._common import (
    BASELINES,
    CLF_ALL,
    N_SEEDS,
    RANDOM_STATE,
    REAL_CLF_NAMES,
    REAL_REG_NAMES,
    REG_ALL,
    build_baseline,
    build_cif,
    get_tree_stats,
    load_real_clf,
    load_real_reg,
    make_clf_downstream,
    make_reg_downstream,
    save_results,
    warmup_jit,
)
from paper.scripts.utils.metrics import f1_at_k, precision_at_k, recall_at_k

EXPERIMENT_NAME = "mirrored_knob_ablation"

KNOB_VARIANTS: dict[str, dict[str, Any]] = {
    "cif_default": dict(),
    "cif_no_adaptive": dict(
        early_stopping_selector=None, early_stopping_splitter=None,
    ),
    "cif_no_scan": dict(feature_scanning=False),
    "cif_no_threshold_scan": dict(threshold_scanning=False),
    "cif_no_mute": dict(feature_muting=False),
    "cif_no_bootstrap": dict(bootstrap=False, sampling_method=None),
    "cif_no_bonferroni": dict(adjust_alpha_selector=False, adjust_alpha_splitter=False),
    "cif_all_off": dict(
        early_stopping_selector=None,
        early_stopping_splitter=None,
        feature_scanning=False,
        threshold_scanning=False,
        feature_muting=False,
        adjust_alpha_selector=False,
        adjust_alpha_splitter=False,
    ),
}

K_VALUES = [5, 10, 25, 50, 100]


def _downstream_metrics(
    X: np.ndarray,
    y: np.ndarray,
    ranking: list[int],
    task: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Evaluate all downstream models at all k values. Returns one row per (model, k)."""
    metric_name = "balanced_accuracy" if task == "clf" else "r2"
    scorer = balanced_accuracy_score if task == "clf" else r2_score
    n_splits = 3

    if task == "clf":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        top_k = ranking[:k]
        models = make_clf_downstream(seed) if task == "clf" else make_reg_downstream(seed)
        if len(top_k) == 0:
            for m_name in models:
                rows.append({"downstream_model": m_name, "k": k, "metric": metric_name, "value": np.nan})
            continue
        X_sel = X[:, top_k]
        for m_name, est in models.items():
            scores = []
            for tr, te in cv.split(X_sel, y):
                est.fit(X_sel[tr], y[tr])
                scores.append(scorer(y[te], est.predict(X_sel[te])))
            rows.append({"downstream_model": m_name, "k": k, "metric": metric_name, "value": float(np.mean(scores))})
    return rows


def _run_one(
    X: np.ndarray,
    y: np.ndarray,
    true_info: list[int] | None,
    model: BaseEstimator,
    seed: int,
    task: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Fit model, extract ranking, compute top-k metrics and downstream metrics."""
    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - t0

    fi = model.feature_importances_
    ranking = list(np.argsort(fi)[::-1])

    top_k_metrics: dict[str, float] = {"elapsed_seconds": elapsed}
    if true_info is not None:
        for k in K_VALUES:
            top_k_metrics[f"precision_at_{k}"] = precision_at_k(ranking, true_info, k)
            top_k_metrics[f"recall_at_{k}"] = recall_at_k(ranking, true_info, k)
            top_k_metrics[f"f1_at_{k}"] = f1_at_k(ranking, true_info, k)

    top_k_metrics.update(get_tree_stats(model))
    downstream_rows = _downstream_metrics(X, y, ranking, task, seed)
    return top_k_metrics, downstream_rows


def _process_variant(
    variant_name: str,
    task: str,
    dtype: str,
    ds_fn: Any,
    overrides: dict[str, Any],
    is_baseline: bool,
    rows: list[dict[str, Any]],
) -> None:
    """Run one variant across all seeds and append aggregated rows."""
    all_top_k: list[dict[str, float]] = []
    all_downstream: list[dict[str, Any]] = []

    for s in range(N_SEEDS):
        seed = RANDOM_STATE + s
        if callable(ds_fn):
            X, y, info, _ = ds_fn(seed)
        else:
            X, y, info = ds_fn
            info = None

        if is_baseline:
            model = build_baseline(variant_name, task, seed)
        else:
            model = build_cif(task, seed, **overrides)

        top_k, downstream = _run_one(X, y, info, model, seed, task)
        all_top_k.append(top_k)
        for d in downstream:
            d["seed"] = seed
        all_downstream.extend(downstream)

    # Aggregate top-k metrics across seeds
    base_row: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "task": task,
        "dataset_type": dtype,
        "variant": variant_name,
    }
    if callable(ds_fn):
        X_base, _, _, _ = ds_fn(RANDOM_STATE)
    else:
        X_base = ds_fn[0]
    base_row["n_features"] = X_base.shape[1]
    base_row["n_samples"] = X_base.shape[0]

    agg_row = dict(base_row)
    all_metric_keys = set()
    for d in all_top_k:
        all_metric_keys.update(d.keys())
    for metric_key in sorted(all_metric_keys):
        vals = [d[metric_key] for d in all_top_k if metric_key in d and not np.isnan(d.get(metric_key, np.nan))]
        if vals:
            agg_row[f"{metric_key}_mean"] = float(np.mean(vals))
            agg_row[f"{metric_key}_std"] = float(np.std(vals))

    # Aggregate downstream metrics across seeds, per (model, k)
    df_ds = pd.DataFrame(all_downstream)
    for (m_name, k_val), grp in df_ds.groupby(["downstream_model", "k"]):
        col_prefix = f"ds_{m_name}_k{k_val}"
        agg_row[f"{col_prefix}_mean"] = float(grp["value"].mean())
        agg_row[f"{col_prefix}_std"] = float(grp["value"].std())

    rows.append(agg_row)

    # Progress line
    p10 = agg_row.get("precision_at_10_mean", None)
    t = agg_row.get("elapsed_seconds_mean", 0)
    depth = agg_row.get("mean_depth_mean", 0)
    ds_lr_10 = agg_row.get("ds_lr_k10_mean", agg_row.get("ds_ridge_k10_mean", 0))
    p_str = f"P@10={p10:.3f}" if p10 is not None else "P@10=N/A"
    print(f"  {variant_name:22s}: {p_str} ds@10={ds_lr_10:.3f} depth={depth:.1f} t={t:.1f}s")


def run() -> pd.DataFrame:
    """Run the full mirrored knob ablation experiment."""
    rows: list[dict] = []

    # Synthetic datasets
    for task, datasets in [("clf", CLF_ALL), ("reg", REG_ALL)]:
        print(f"\n--- {task.upper()} SYNTHETIC ---")
        for ds_fn in datasets:
            X_base, _, _, dtype = ds_fn(RANDOM_STATE)
            print(f"\n  {dtype} (n={X_base.shape[0]}, p={X_base.shape[1]})")

            for vname, overrides in KNOB_VARIANTS.items():
                _process_variant(vname, task, dtype, ds_fn, overrides, False, rows)
            for method in BASELINES:
                _process_variant(method, task, dtype, ds_fn, {}, True, rows)

    # Real datasets
    for task, ds_names, loader in [
        ("clf", REAL_CLF_NAMES, load_real_clf),
        ("reg", REAL_REG_NAMES, load_real_reg),
    ]:
        print(f"\n--- REAL {task.upper()} DATASETS ---")
        for ds_name in ds_names:
            try:
                X, y, dtype = loader(ds_name)
            except Exception as e:
                print(f"\n  SKIP {ds_name}: {e}")
                continue
            print(f"\n  {dtype} (n={X.shape[0]}, p={X.shape[1]})")

            ds_tuple = (X, y, None)
            for vname, overrides in KNOB_VARIANTS.items():
                _process_variant(vname, task, dtype, ds_tuple, overrides, False, rows)
            for method in BASELINES:
                _process_variant(method, task, dtype, ds_tuple, {}, True, rows)

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
