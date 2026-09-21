#!/usr/bin/env python3
"""Random forest ranked by out-of-bag permutation importance, as a benchmark comparator.

The manuscripts state that an ordinary random forest ranked by out-of-bag
permutation importance removes CART's high-cardinality preference at a fraction
of the cost of conditional inference (NHANES application, weak-signal designs).
This experiment adds that ranker to the real-data benchmark: on every Stage 1
fold it fits the benchmark's random forest (100 trees, scikit-learn defaults),
ranks features by partykit-style out-of-bag permutation importance (one
permutation of each column on each tree's out-of-bag rows, error increase
averaged over trees; misclassification rate in classification, mean squared
error in regression), and scores the ranking with the Stage 2 fold evaluator,
producing rows that join the evaluation surface under ``method_base =
"rf_oobperm"``. Each fit is parallel over trees; the timing column records the
forest fit and the importance separately.

Usage:
    uv run python -m paper.benchmark.experiments.rf_oobperm_benchmark \
        --datasets gamma,spam --data-dir paper/data --output-dir out/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from paper.benchmark.adapters.data import _load_parquet_to_arrays, get_cv_splitter
from paper.benchmark.config.constants import N_SEEDS, N_SPLITS
from paper.benchmark.pipeline.stage2 import evaluate_fold, get_requested_evaluation_k_values

EXPERIMENT_NAME = "rf_oobperm_benchmark"
METHOD_BASE = "rf_oobperm"
METHOD_ID = "rf_oobperm__default"
N_TREES = 100


def _tree_oob_importance(
    tree: Any, oob: np.ndarray, X: np.ndarray, y: np.ndarray, task: str, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X_oob, y_oob = X[oob], y[oob]

    def loss(pred: np.ndarray) -> float:
        if task == "classification":
            return float(np.mean(pred != y_oob))
        return float(np.mean((pred - y_oob) ** 2))

    base = loss(tree.predict(X_oob))
    out = np.zeros(X.shape[1])
    for f in range(X.shape[1]):
        Xp = X_oob.copy()
        Xp[:, f] = rng.permutation(Xp[:, f])
        out[f] = loss(tree.predict(Xp)) - base
    return out


def oob_permutation_importance(
    rf: Any, X: np.ndarray, y: np.ndarray, task: str, seed: int
) -> np.ndarray:
    """One permutation of each column on each tree's out-of-bag rows, averaged over trees."""
    n = X.shape[0]
    jobs = []
    for j, (tree, in_bag) in enumerate(zip(rf.estimators_, rf.estimators_samples_, strict=True)):
        oob = np.setdiff1d(np.arange(n), in_bag)
        if oob.size:
            jobs.append(delayed(_tree_oob_importance)(tree, oob, X, y, task, seed * 1000 + j))
    return np.mean(np.vstack(Parallel(n_jobs=-1, prefer="threads")(jobs)), axis=0)


def run_dataset(
    task: str, name: str, data_dir: Path, output_dir: Path, seeds: tuple[int, ...] | None
) -> None:
    seed_list = tuple(range(N_SEEDS)) if seeds is None else tuple(seeds)
    suffix = "" if seeds is None else "__seeds" + "-".join(str(x) for x in seed_list)
    out = output_dir / f"{task}__{name}{suffix}.parquet"
    if out.exists():
        print(f"skip {task} {name}: exists", flush=True)
        return
    prefix = "clf" if task == "classification" else "reg"
    X, y = _load_parquet_to_arrays(
        (data_dir / task / "real" / f"{prefix}_{name}.parquet").read_bytes(), task
    )
    k_values = get_requested_evaluation_k_values(X.shape[1])
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for seed in seed_list:
        for fold_idx, (tr, te) in enumerate(get_cv_splitter(task, N_SPLITS, seed).split(X, y)):
            X_tr = StandardScaler().fit_transform(X[tr])
            rs = seed * 1000 + fold_idx
            forest_cls = (
                RandomForestClassifier if task == "classification" else RandomForestRegressor
            )
            start = time.perf_counter()
            rf = forest_cls(n_estimators=N_TREES, n_jobs=-1, random_state=rs).fit(X_tr, y[tr])
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            imp = oob_permutation_importance(rf, X_tr, y[tr], task, rs)
            importance_seconds = time.perf_counter() - start
            ranking = np.lexsort((np.arange(X.shape[1]), -imp))
            impurity_rank = np.argsort(rf.feature_importances_)[::-1]
            for res in evaluate_fold(X[tr], y[tr], X[te], y[te], ranking, task, rs, k_values, 1):
                res.update(
                    dataset=name,
                    task=task,
                    seed=seed,
                    fold_idx=fold_idx,
                    method_base=METHOD_BASE,
                    method_id=METHOD_ID,
                    dataset_source="real",
                    n_samples=int(X.shape[0]),
                    n_features=int(X.shape[1]),
                    fit_seconds=fit_seconds,
                    importance_seconds=importance_seconds,
                    oobperm_top10=json.dumps(ranking[:10].tolist()),
                    impurity_top10=json.dumps(impurity_rank[:10].tolist()),
                )
                rows.append(res)
            print(
                f"  {task} {name} seed={seed} fold={fold_idx}: fit {fit_seconds:.1f}s, "
                f"importance {importance_seconds:.1f}s",
                flush=True,
            )
    df = pd.DataFrame(rows)
    df["elapsed_seconds_dataset"] = time.perf_counter() - t0
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(
        f"{task} {name}: shape {X.shape}, {time.perf_counter() - t0:.0f}s, {len(df)} rows",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--data-dir", type=Path, default=Path("paper/data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="comma-separated task/name pairs, e.g. classification/gamma,regression/facebook",
    )
    parser.add_argument("--seeds", type=str, default=None, help="comma-separated benchmark seeds")
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else None
    for item in args.datasets.split(","):
        task, name = item.split("/")
        run_dataset(task, name, args.data_dir, args.output_dir, seeds)


if __name__ == "__main__":
    main()
