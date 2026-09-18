#!/usr/bin/env python3
"""Empirical fitting-time scaling of CIT and CIF in the sample size and the predictor count.

Gaussian predictors with a planted weak linear signal in five of them; the
classification response thresholds the latent variable at its median. One tree
(CIT) and a 100-tree forest (CIF, ``n_jobs=-1``) are fit at every (n, p) cell with
the paper's recommended controls, once with adaptive stopping and once with the
full permutation budget, so the curves can be compared with the cost accounting
of the complexity appendix. Fit-only wall-clock time, median over repeats, after
one untimed identical fit per cell.

Usage:
    uv run python -m paper.benchmark.experiments.scaling_curves [--shard i --num-shards n] [--assemble]
"""

from __future__ import annotations

import argparse
import sys
import time
from io import TextIOWrapper
from typing import Any

import numpy as np
import pandas as pd

from paper.benchmark.experiments.experiment_common import (
    RANDOM_STATE,
    assemble_checkpoints,
    build_cif,
    build_cit,
    load_checkpoint,
    save_checkpoint,
    save_results,
    shard_indices,
    warmup_jit,
)

EXPERIMENT_NAME = "scaling_curves"
SAMPLE_SIZES = (500, 2000, 8000, 32000)
PREDICTOR_COUNTS = (20, 100, 500)
REPEATS = 2
VARIANTS: dict[str, dict[str, Any]] = {
    "default": {},
    "no_adaptive": {"early_stopping_selector": None, "early_stopping_splitter": None},
}


def make_data(task: str, n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    active = min(5, p)
    latent = X[:, :active] @ np.linspace(1.0, 0.25, active) + rng.standard_normal(n)
    if task == "clf":
        return X, (latent >= np.median(latent)).astype(np.int64)
    return X, latent


def _plan() -> list[tuple[str, int, int]]:
    plan = []
    for task in ("clf", "reg"):
        for n in SAMPLE_SIZES:
            for p in PREDICTOR_COUNTS:
                if n * p > 32000 * 100:  # 32,000 x 500 is beyond the head-to-head grid
                    continue
                plan.append((task, n, p))
    return plan


def run_cell(task: str, n: int, p: int) -> list[dict[str, Any]]:
    rows = []
    for model in ("cit", "cif"):
        for variant, overrides in VARIANTS.items():
            # One untimed fit of this exact cell first: the tiny warm-up in
            # experiment_common compiles only the adaptive forest path at n = 30,
            # so the first timed repeat of a cell used to include the just-in-time
            # compilation of kernels chosen by variant or size (the 2026-09-13 run
            # shows first repeats of 5.1 s against 0.7 s for the same CIT cell).
            X, y = make_data(task, n, p, RANDOM_STATE)
            if model == "cit":
                build_cit(task, RANDOM_STATE, **overrides).fit(X, y)
            else:
                build_cif(task, RANDOM_STATE, n_estimators=100, n_jobs=-1, **overrides).fit(X, y)
            seconds = []
            for r in range(REPEATS):
                X, y = make_data(task, n, p, RANDOM_STATE + r)
                if model == "cit":
                    est = build_cit(task, RANDOM_STATE + r, **overrides)
                else:
                    est = build_cif(
                        task, RANDOM_STATE + r, n_estimators=100, n_jobs=-1, **overrides
                    )
                t0 = time.perf_counter()
                est.fit(X, y)
                seconds.append(time.perf_counter() - t0)
            rows.append(
                {
                    "experiment": EXPERIMENT_NAME,
                    "task": task,
                    "n_samples": n,
                    "n_features": p,
                    "model": model,
                    "variant": variant,
                    "repeats": REPEATS,
                    "seconds_median": float(np.median(seconds)),
                    "seconds_min": float(np.min(seconds)),
                    "seconds_max": float(np.max(seconds)),
                }
            )
            print(
                f"  {task} n={n:>6} p={p:>4} {model} {variant:12s}: {np.median(seconds):8.2f}s",
                flush=True,
            )
    return rows


def run(shard: int = 0, num_shards: int = 1) -> pd.DataFrame:
    plan = _plan()
    mine = shard_indices(len(plan), shard, num_shards)
    rows: list[dict[str, Any]] = []
    for i, (task, n, p) in enumerate(plan):
        if i not in mine:
            continue
        key = f"n{n}_p{p}"
        cached = load_checkpoint(EXPERIMENT_NAME, task, key)
        if cached is not None:
            rows.extend(cached)
            continue
        cell = run_cell(task, n, p)
        save_checkpoint(EXPERIMENT_NAME, task, key, cell)
        rows.extend(cell)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--assemble", action="store_true")
    args = parser.parse_args()
    if isinstance(sys.stdout, TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    print(f"=== {EXPERIMENT_NAME} ===")
    if args.assemble:
        df = assemble_checkpoints(EXPERIMENT_NAME)
    else:
        warmup_jit()
        df = run(args.shard, args.num_shards)
        if args.num_shards > 1:
            print(f"\nShard {args.shard + 1}/{args.num_shards} done: {len(df)} rows checkpointed")
            return
    path = save_results(df, EXPERIMENT_NAME)
    print(f"\nSaved: {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
