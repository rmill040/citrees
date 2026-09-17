#!/usr/bin/env python3
"""Weak signal with high-cardinality noise: where impurity importance fills the top-k.

Two parts. ``designs``: n = 1,000 with 10 informative Gaussian features, 25 noise
columns with 500 levels, 25 binary noise columns, and 10 Gaussian noise columns,
at three signal strengths, both tasks; each ranking method is scored by the share
of informative, high-cardinality-noise, and binary-noise features in its top-k,
by the Stage 2 downstream protocol at k in {5, 10, 25}, and by its support size
(features with nonzero importance) against the true count of 10. ``support``:
the support size of CIF, CIT, and RF on the synthetic benchmark datasets whose
informative features are known. Rankers: CIF and CIT (selected benchmark
configurations with the linear selectors), RF impurity importance, the same RF
ranked by out-of-bag permutation importance, XGBoost, and LightGBM.

Usage:
    uv run python -m paper.benchmark.experiments.weak_signal_cardinality \
        --part designs --task classification --output-dir out/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler

from paper.benchmark.adapters.data import _load_parquet_to_arrays, get_cv_splitter
from paper.benchmark.config.constants import N_SEEDS, N_SPLITS
from paper.benchmark.pipeline.selectors import get_embedding_model
from paper.benchmark.pipeline.stage2 import evaluate_fold

EXPERIMENT_NAME = "weak_signal_cardinality"
K_VALUES = (5, 10, 25)
N_INFORMATIVE = 10
SIGNALS = {"classification": (0.3, 0.5, 1.0), "regression": (0.25, 0.5, 1.0)}
CIF = {
    "classification": dict(
        adjust_alpha_selector=True,
        adjust_alpha_splitter=True,
        alpha_selector=0.05,
        alpha_splitter=0.05,
        bootstrap=True,
        early_stopping_confidence_selector=0.95,
        early_stopping_confidence_splitter=0.95,
        early_stopping_selector="adaptive",
        early_stopping_splitter="adaptive",
        feature_muting=True,
        feature_scanning=True,
        honesty=False,
        honesty_fraction=0.5,
        max_samples=None,
        max_thresholds=256,
        n_estimators=100,
        n_resamples_selector="minimum",
        n_resamples_splitter="minimum",
        sampling_method="stratified",
        selector="mc",
        splitter="gini",
        threshold_method="histogram",
        threshold_scanning=True,
    ),
    "regression": dict(
        adjust_alpha_selector=True,
        adjust_alpha_splitter=True,
        alpha_selector=0.05,
        alpha_splitter=0.05,
        bootstrap=True,
        early_stopping_confidence_selector=0.95,
        early_stopping_confidence_splitter=0.95,
        early_stopping_selector="adaptive",
        early_stopping_splitter="adaptive",
        feature_muting=True,
        feature_scanning=True,
        honesty=True,
        honesty_fraction=0.5,
        max_samples=None,
        max_thresholds=256,
        n_estimators=100,
        n_resamples_selector="minimum",
        n_resamples_splitter="minimum",
        selector="pc",
        splitter="mse",
        threshold_method="histogram",
        threshold_scanning=True,
    ),
}
CIT = {
    t: {
        k: v
        for k, v in p.items()
        if k not in ("bootstrap", "max_samples", "n_estimators", "sampling_method")
    }
    for t, p in CIF.items()
}


def design(
    task: str, signal: float, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]]]:
    rng = np.random.default_rng(seed)
    n = 1000
    if task == "classification":
        X_inf, y = make_classification(
            n_samples=n,
            n_features=N_INFORMATIVE,
            n_informative=N_INFORMATIVE,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=2,
            class_sep=signal,
            flip_y=0.05,
            shuffle=False,
            random_state=seed,
        )
        y = y.astype(np.int64)
    else:
        X_inf, y = make_regression(
            n_samples=n,
            n_features=N_INFORMATIVE,
            n_informative=N_INFORMATIVE,
            noise=1.0,
            shuffle=False,
            random_state=seed,
        )
        y = (signal * (y - y.mean()) / y.std() + rng.normal(size=n)).astype(np.float64)
    highcard = rng.integers(0, 500, size=(n, 25)).astype(float)
    binary = rng.integers(0, 2, size=(n, 25)).astype(float)
    gauss = rng.normal(size=(n, 10))
    X = np.hstack([X_inf, highcard, binary, gauss])
    groups = {
        "informative": list(range(0, 10)),
        "noise_highcard": list(range(10, 35)),
        "noise_binary": list(range(35, 60)),
        "noise_gaussian": list(range(60, 70)),
    }
    return X, y, groups


def _tree_oob(
    tree: Any, oob: np.ndarray, X: np.ndarray, y: np.ndarray, classifier: bool, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Xo, yo = X[oob], y[oob]

    def err(pred: np.ndarray) -> float:
        return float(np.mean(pred != yo)) if classifier else float(np.mean((pred - yo) ** 2))

    base = err(tree.predict(Xo))
    out = np.zeros(X.shape[1])
    for f in range(X.shape[1]):
        Xp = Xo.copy()
        Xp[:, f] = rng.permutation(Xp[:, f])
        out[f] = err(tree.predict(Xp)) - base
    return out


def rf_oob_importance(
    rf: Any, X: np.ndarray, y: np.ndarray, classifier: bool, seed: int
) -> np.ndarray:
    n = X.shape[0]
    jobs = []
    for j, (tree, in_bag) in enumerate(zip(rf.estimators_, rf.estimators_samples_, strict=True)):
        oob = np.setdiff1d(np.arange(n), in_bag)
        if oob.size:
            jobs.append(delayed(_tree_oob)(tree, oob, X, y, classifier, seed * 1000 + j))
    return np.mean(np.vstack(Parallel(n_jobs=-1, prefer="threads")(jobs)), axis=0)


def rankers(
    task: str, X_tr: np.ndarray, y_tr: np.ndarray, rs: int
) -> dict[str, tuple[np.ndarray, int, float]]:
    """Return {method: (importance, support size, fit seconds)}."""
    out = {}
    classifier = task == "classification"
    for name, method, params in (
        ("cif", "cif", CIF[task]),
        ("cit", "cit", CIT[task]),
        ("rf", "rf", {}),
        ("xgb", "xgb", {}),
        ("lgbm", "lgbm", {}),
    ):
        t0 = time.perf_counter()
        model = get_embedding_model(method, task, rs, n_jobs=-1, params=params)
        model.fit(X_tr, y_tr)
        imp = np.asarray(model.feature_importances_, dtype=float)
        out[name] = (imp, int(np.sum(imp > 0)), time.perf_counter() - t0)
        if name == "rf":
            t0 = time.perf_counter()
            oob = rf_oob_importance(model, X_tr, y_tr, classifier, rs)
            out["rf_oobperm"] = (oob, int(np.sum(oob > 0)), time.perf_counter() - t0)
    return out


def run_designs(task: str, output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for signal in SIGNALS[task]:
        for seed in range(N_SEEDS):
            X, y, groups = design(task, signal, seed)
            for fold_idx, (tr, te) in enumerate(get_cv_splitter(task, N_SPLITS, seed).split(X, y)):
                X_tr = StandardScaler().fit_transform(X[tr])
                rs = seed * 1000 + fold_idx
                for name, (imp, support, secs) in rankers(task, X_tr, y[tr], rs).items():
                    rank = np.lexsort((np.arange(X.shape[1]), -imp))
                    base = dict(
                        task=task,
                        signal=signal,
                        seed=seed,
                        fold_idx=fold_idx,
                        method=name,
                        support_size=support,
                        n_informative=N_INFORMATIVE,
                        fit_seconds=secs,
                    )
                    for k in K_VALUES:
                        top = set(int(i) for i in rank[:k])
                        for g, idx in groups.items():
                            base[f"{g}_share_at_{k}"] = len(top & set(idx)) / k
                    for res in evaluate_fold(
                        X[tr], y[tr], X[te], y[te], rank, task, rs, list(K_VALUES), 1
                    ):
                        rows.append({**base, **res})
            print(f"{task} signal={signal} seed={seed} done", flush=True)
    pd.DataFrame(rows).to_parquet(output_dir / f"designs__{task}.parquet", index=False)


def run_support(task: str, data_dir: Path, output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for f in sorted((data_dir / task / "synthetic").glob("*.parquet")):
        meta = pq.read_schema(f).metadata
        informative = [int(i) for i in json.loads(meta[b"informative_indices"].decode())]
        X, y = _load_parquet_to_arrays(f.read_bytes(), task)
        for seed in range(N_SEEDS):
            for fold_idx, (tr, _te) in enumerate(get_cv_splitter(task, N_SPLITS, seed).split(X, y)):
                X_tr = StandardScaler().fit_transform(X[tr])
                rs = seed * 1000 + fold_idx
                for name, method, params in (
                    ("cif", "cif", CIF[task]),
                    ("cit", "cit", CIT[task]),
                    ("rf", "rf", {}),
                ):
                    model = get_embedding_model(method, task, rs, n_jobs=-1, params=params)
                    model.fit(X_tr, y[tr])
                    imp = np.asarray(model.feature_importances_, dtype=float)
                    used = set(int(i) for i in np.flatnonzero(imp > 0))
                    rows.append(
                        dict(
                            task=task,
                            dataset=f.stem.split("_", 1)[1],
                            seed=seed,
                            fold_idx=fold_idx,
                            method=name,
                            n_features=int(X.shape[1]),
                            n_informative=len(informative),
                            support_size=len(used),
                            informative_in_support=len(used & set(informative)),
                        )
                    )
        print(f"{task} {f.stem} support done", flush=True)
    pd.DataFrame(rows).to_parquet(output_dir / f"support__{task}.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--part", choices=("designs", "support"), required=True)
    parser.add_argument("--task", choices=("classification", "regression"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("paper/data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.part == "designs":
        run_designs(args.task, args.output_dir)
    else:
        run_support(args.task, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
