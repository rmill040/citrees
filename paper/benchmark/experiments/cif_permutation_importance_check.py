#!/usr/bin/env python3
"""Mechanism-matched check: CIF ranked by permutation importance.

The benchmark ranks cforest by partykit's out-of-bag permutation importance and CIF
by split importance, so the CIF-versus-cforest margin mixes the importance
mechanism with the split procedure. This check refits the protocol-selected CIF
configuration on each Stage 1 fold, ranks features by permutation importance, and evaluates the
ranking with the Stage 2 fold evaluator, producing rows comparable to the
benchmark evaluation surface. The default importance is partykit-style
out-of-bag permutation importance (one permutation per tree on that tree's
out-of-bag rows, averaged over trees; ``method_base = "cif_oobperm"``), the
mechanism cforest is ranked by. ``--importance train`` instead uses scikit-learn
permutation importance on the training rows (``cif_perm``). The split
importance ranking of the same fit is kept for a reproduction check against the
stored Stage 1 ranking. Locked cells (isolet, gisette, letter) are never run.

Usage:
    uv run python -m paper.benchmark.experiments.cif_permutation_importance_check \
        --shard 0 --num-shards 8 --data-dir paper/data --output-dir out/
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
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from citrees._utils import (
    classic_bootstrap_sample,
    oversample_bootstrap_sample,
    stratified_bootstrap_sample,
    undersample_bootstrap_sample,
)
from paper.benchmark.adapters.data import _load_parquet_to_arrays, get_cv_splitter
from paper.benchmark.config.constants import N_SEEDS, N_SPLITS
from paper.benchmark.pipeline.selectors import get_embedding_model
from paper.benchmark.pipeline.stage2 import evaluate_fold, get_requested_evaluation_k_values

EXPERIMENT_NAME = "cif_permutation_importance_check"
N_REPEATS = 5
LOCKED = frozenset({"isolet", "gisette", "letter"})

# Protocol-selected CIF configurations of the benchmark (method_id in comments).
SELECTED: dict[str, tuple[str, dict[str, Any]]] = {
    "classification": (
        "cif__4ee6725d43ec8e5a",
        {
            "adjust_alpha_selector": True,
            "adjust_alpha_splitter": True,
            "alpha_selector": 0.05,
            "alpha_splitter": 0.05,
            "bootstrap": True,
            "early_stopping_confidence_selector": 0.95,
            "early_stopping_confidence_splitter": 0.95,
            "early_stopping_selector": "adaptive",
            "early_stopping_splitter": "adaptive",
            "feature_muting": True,
            "feature_scanning": True,
            "honesty": False,
            "honesty_fraction": 0.5,
            "max_samples": None,
            "max_thresholds": 256,
            "n_estimators": 100,
            "n_resamples_selector": "minimum",
            "n_resamples_splitter": "minimum",
            "sampling_method": "stratified",
            "selector": "rdc",
            "splitter": "gini",
            "threshold_method": "histogram",
            "threshold_scanning": True,
        },
    ),
    "regression": (
        "cif__40426895fc45caef",
        {
            "adjust_alpha_selector": True,
            "adjust_alpha_splitter": True,
            "alpha_selector": 0.05,
            "alpha_splitter": 0.05,
            "bootstrap": True,
            "early_stopping_confidence_selector": 0.95,
            "early_stopping_confidence_splitter": 0.95,
            "early_stopping_selector": "adaptive",
            "early_stopping_splitter": "adaptive",
            "feature_muting": True,
            "feature_scanning": True,
            "honesty": True,
            "honesty_fraction": 0.5,
            "max_samples": None,
            "max_thresholds": 256,
            "n_estimators": 100,
            "n_resamples_selector": "minimum",
            "n_resamples_splitter": "minimum",
            "selector": "rdc",
            "splitter": "mse",
            "threshold_method": "histogram",
            "threshold_scanning": True,
        },
    ),
}

# Real datasets of the CIF-versus-cforest pairwise panel, heaviest first so that
# round-robin sharding balances the permutation-importance cost (n_repeats x p).
PLAN: list[tuple[str, str]] = [
    ("classification", "dexter"),
    ("classification", "CLL_SUB_111"),
    ("classification", "orlraws10P"),
    ("classification", "arcene"),
    ("classification", "pixraw10P"),
    ("classification", "ALLAML"),
    ("regression", "coepra1"),
    ("regression", "coepra3"),
    ("classification", "TOX_171"),
    ("regression", "coepra2"),
    ("regression", "comm_violence"),
    ("regression", "community_crime"),
    ("classification", "warpPIE10P"),
    ("classification", "warpAR10P"),
    ("classification", "ORL"),
    ("classification", "Yale"),
    ("classification", "madelon"),
    ("classification", "musk"),
    ("classification", "gamma"),
    ("classification", "pendigits"),
    ("classification", "spam"),
    ("classification", "page-blocks"),
    ("regression", "residential"),
    ("classification", "vowel-context"),
    ("regression", "imports-85"),
    ("regression", "facebook"),
    ("classification", "glass"),
    ("classification", "wine"),
]


def _boot_idx(model: Any, tree: Any, y: np.ndarray) -> np.ndarray:
    """Reproduce the bootstrap indices the forest used to fit ``tree``."""
    kwargs = {"y": y, "max_samples": model._max_samples, "random_state": tree.random_state}
    if getattr(model, "_estimator_type", None) is not None and hasattr(model, "classes_"):
        method = str(getattr(model, "_sampling_method", "") or "")
        if method == "stratified":
            return stratified_bootstrap_sample(**kwargs)
        if method == "undersample":
            return undersample_bootstrap_sample(**kwargs)
        if method == "oversample":
            return oversample_bootstrap_sample(**kwargs)
    return classic_bootstrap_sample(**kwargs)


def _tree_oob_importance(
    tree: Any, oob_idx: np.ndarray, X: np.ndarray, y: np.ndarray, classifier: bool, seed: int
) -> np.ndarray:
    """Error increase per feature when it is permuted on the tree's out-of-bag rows."""
    rng = np.random.default_rng(seed)
    X_oob, y_oob = X[oob_idx], y[oob_idx]

    def err(pred: np.ndarray) -> float:
        return float(np.mean(pred != y_oob)) if classifier else float(np.mean((pred - y_oob) ** 2))

    base = err(tree.predict(X_oob))
    out = np.zeros(X.shape[1])
    for f in range(X.shape[1]):
        Xp = X_oob.copy()
        Xp[:, f] = rng.permutation(Xp[:, f])
        out[f] = err(tree.predict(Xp)) - base
    return out


def oob_permutation_importance(model: Any, X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """partykit-style unconditional permutation importance: one permutation per tree,
    scored on that tree's out-of-bag rows, averaged over trees."""
    classifier = hasattr(model, "classes_")
    n = X.shape[0]
    jobs = []
    for j, tree in enumerate(model.estimators_):
        oob = np.setdiff1d(np.arange(n), _boot_idx(model, tree, y))
        if oob.size == 0:
            continue
        jobs.append(delayed(_tree_oob_importance)(tree, oob, X, y, classifier, seed * 1000 + j))
    per_tree = Parallel(n_jobs=-1, backend="loky")(jobs)
    return np.mean(np.vstack(per_tree), axis=0)


def run_dataset(
    task: str,
    name: str,
    data_dir: Path,
    output_dir: Path,
    rankings_dir: Path | None,
    seeds: tuple[int, ...] | None = None,
    importance: str = "oob",
) -> None:
    """Refit, rank by permutation importance, evaluate, and write one parquet per dataset.

    ``seeds`` restricts the run to a subset of the benchmark seeds (the output file
    name then carries the subset) so that a slow dataset can be spread over boxes.
    """
    seed_list = tuple(range(N_SEEDS)) if seeds is None else tuple(seeds)
    suffix = "" if seeds is None else "__seeds" + "-".join(str(x) for x in seed_list)
    if importance == "oob":
        suffix = "__oob" + suffix
    label = "cif_oobperm" if importance == "oob" else "cif_perm"
    out = output_dir / f"{task}__{name}{suffix}.parquet"
    if out.exists():
        print(f"skip {task} {name}: exists", flush=True)
        return
    prefix = "clf" if task == "classification" else "reg"
    X, y = _load_parquet_to_arrays(
        (data_dir / task / "real" / f"{prefix}_{name}.parquet").read_bytes(), task
    )
    method_id, params = SELECTED[task]
    k_values = get_requested_evaluation_k_values(X.shape[1])
    rows: list[dict[str, Any]] = []
    agreement: list[float] = []
    t0 = time.perf_counter()
    for seed in seed_list:
        stored = None
        if rankings_dir is not None:
            f = rankings_dir / task / name / f"{method_id}_seed{seed}.parquet"
            stored = pd.read_parquet(f) if f.exists() else None
        for fold_idx, (tr, te) in enumerate(get_cv_splitter(task, N_SPLITS, seed).split(X, y)):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            rs = seed * 1000 + fold_idx
            model = get_embedding_model("cif", task, rs, n_jobs=-1, params=params)
            model.fit(X_tr, y[tr])
            split_rank = np.argsort(model.feature_importances_)[::-1]
            # Predict serially inside permutation_importance: a parallel forest predict
            # starts a worker pool per call, and permutation importance calls predict
            # n_repeats x p times; the parallelism belongs to the feature loop below.
            model.n_jobs = 1
            if stored is not None and (stored["fold_idx"] == fold_idx).any():
                stored_rank = np.array(
                    stored.loc[stored["fold_idx"] == fold_idx, "feature_ranking"].iloc[0]
                )
                agreement.append(float(np.mean(stored_rank[:10] == split_rank[:10])))
            if importance == "oob":
                imp = oob_permutation_importance(model, X_tr, y[tr], rs)
            else:
                pi = permutation_importance(
                    model, X_tr, y[tr], n_repeats=N_REPEATS, random_state=rs, n_jobs=-1
                )
                imp = pi.importances_mean
            perm_rank = np.lexsort((np.arange(X.shape[1]), -imp))
            for res in evaluate_fold(X[tr], y[tr], X[te], y[te], perm_rank, task, rs, k_values, 1):
                res.update(
                    dataset=name,
                    task=task,
                    seed=seed,
                    fold_idx=fold_idx,
                    method_base=label,
                    method_id=label + "__" + method_id.split("__")[1],
                    importance=importance,
                    n_samples=int(X.shape[0]),
                    n_features=int(X.shape[1]),
                    n_repeats=N_REPEATS,
                    perm_top10=json.dumps(perm_rank[:10].tolist()),
                    split_top10=json.dumps(split_rank[:10].tolist()),
                )
                rows.append(res)
    df = pd.DataFrame(rows)
    df["top10_agreement_with_stored_split_ranking"] = (
        float(np.mean(agreement)) if agreement else np.nan
    )
    df["elapsed_seconds_dataset"] = time.perf_counter() - t0
    df.to_parquet(out, index=False)
    print(
        f"{task} {name}: shape {X.shape}, {time.perf_counter() - t0:.0f}s, {len(df)} rows",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=Path("paper/data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rankings-dir", type=Path, default=None)
    parser.add_argument("--datasets", type=str, default=None, help="comma-separated dataset names")
    parser.add_argument("--seeds", type=str, default=None, help="comma-separated benchmark seeds")
    parser.add_argument(
        "--importance",
        choices=["oob", "train"],
        default="oob",
        help="oob: partykit-style out-of-bag permutation importance; train: scikit-learn permutation importance on training rows",
    )
    args = parser.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(",")) if args.seeds else None
    only = set(args.datasets.split(",")) if args.datasets else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, (task, name) in enumerate(PLAN):
        if only is not None:
            if name not in only:
                continue
        elif i % args.num_shards != args.shard:
            continue
        if name in LOCKED:
            raise RuntimeError(f"locked cell in plan: {name}")
        run_dataset(
            task, name, args.data_dir, args.output_dir, args.rankings_dir, seeds, args.importance
        )


if __name__ == "__main__":
    main()
