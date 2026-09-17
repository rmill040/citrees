#!/usr/bin/env python3
"""CIF-all recovery: does evaluating every candidate feature change the ranking?

The mechanism section of the arXiv paper shows that CIF with square-root feature
sampling rarely has an informative feature among its candidates on sparse
high-dimensional designs, while CIF-all (every nonconstant feature evaluated at
every node) does. This experiment measures the consequence for the ranking: the
selected CIF configuration and the same configuration with ``max_features=None``
are fit on the synthetic benchmark datasets (whose informative features are
known) and on a sparse grid (n = 250, p = 1,000, 2, 5, or 10 informative
features), per benchmark seed and Stage 1 fold, and scored by top-k precision and
recall against the planted features and by the Stage 2 downstream protocol.

Usage:
    uv run python -m paper.benchmark.experiments.cif_all_recovery \
        --shard 0 --num-shards 4 --data-dir paper/data --output-dir out/
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
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler

from paper.benchmark.adapters.data import _load_parquet_to_arrays, get_cv_splitter
from paper.benchmark.config.constants import N_SEEDS, N_SPLITS
from paper.benchmark.experiments.cif_permutation_importance_check import SELECTED
from paper.benchmark.pipeline.selectors import get_embedding_model
from paper.benchmark.pipeline.stage2 import evaluate_fold, get_requested_evaluation_k_values
from paper.benchmark.utils.metrics import precision_at_k, recall_at_k

EXPERIMENT_NAME = "cif_all_recovery"
K_VALUES = (5, 10, 25, 50, 100)
VARIANTS: dict[str, dict[str, Any]] = {"cif": {}, "cif_all": {"max_features": None}}
SPARSE_GRID = [(250, 1000, 2), (250, 1000, 5), (250, 1000, 10)]


def _synthetic_files(data_dir: Path) -> list[tuple[str, str, Path]]:
    out = []
    for task in ("classification", "regression"):
        for f in sorted((data_dir / task / "synthetic").glob("*.parquet")):
            out.append((task, f.stem.split("_", 1)[1], f))
    return out


def _load_synthetic(path: Path, task: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    meta = pq.read_schema(path).metadata
    informative = json.loads(meta[b"informative_indices"].decode())
    X, y = _load_parquet_to_arrays(path.read_bytes(), task)
    return X, y, [int(i) for i in informative]


def _sparse_design(
    task: str, n: int, p: int, k: int, seed: int
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if task == "classification":
        X, y = make_classification(
            n_samples=n,
            n_features=p,
            n_informative=k,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            class_sep=1.0,
            shuffle=False,
            random_state=seed,
        )
        y = y.astype(np.int64)
    else:
        X, y = make_regression(
            n_samples=n, n_features=p, n_informative=k, noise=1.0, shuffle=False, random_state=seed
        )
        y = y.astype(np.float64)
    return X, y, list(range(k))  # shuffle=False puts the informative features first


def plan(data_dir: Path) -> list[tuple[str, str, Any]]:
    items: list[tuple[str, str, Any]] = [
        (t, name, ("file", f)) for t, name, f in _synthetic_files(data_dir)
    ]
    for task in ("classification", "regression"):
        for n, p, k in SPARSE_GRID:
            items.append((task, f"sparse_n{n}_p{p}_k{k}", ("grid", (n, p, k))))
    return items


def run_item(task: str, name: str, source: Any, output_dir: Path) -> None:
    out = output_dir / f"{task}__{name}.parquet"
    if out.exists():
        print(f"skip {task} {name}", flush=True)
        return
    method_id, params = SELECTED[task]
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for seed in range(N_SEEDS):
        if source[0] == "file":
            X, y, informative = _load_synthetic(source[1], task)
        else:
            X, y, informative = _sparse_design(task, *source[1], seed)
        k_values = get_requested_evaluation_k_values(X.shape[1])
        for fold_idx, (tr, te) in enumerate(get_cv_splitter(task, N_SPLITS, seed).split(X, y)):
            X_tr = StandardScaler().fit_transform(X[tr])
            rs = seed * 1000 + fold_idx
            for variant, overrides in VARIANTS.items():
                model = get_embedding_model(
                    "cif", task, rs, n_jobs=-1, params={**params, **overrides}
                )
                start = time.perf_counter()
                model.fit(X_tr, y[tr])
                fit_seconds = time.perf_counter() - start
                ranking = np.argsort(model.feature_importances_)[::-1]
                rank_list = [int(i) for i in ranking]
                recovery = {
                    f"precision_at_{k}": precision_at_k(rank_list, informative, k)
                    for k in K_VALUES
                    if k <= X.shape[1]
                }
                recovery.update(
                    {
                        f"recall_at_{k}": recall_at_k(rank_list, informative, k)
                        for k in K_VALUES
                        if k <= X.shape[1]
                    }
                )
                n_used = int(np.sum(model.feature_importances_ > 0))
                for res in evaluate_fold(
                    X[tr], y[tr], X[te], y[te], ranking, task, rs, k_values, 1
                ):
                    res.update(
                        dataset=name,
                        task=task,
                        seed=seed,
                        fold_idx=fold_idx,
                        method_base=variant,
                        method_id=f"{variant}__{method_id.split('__')[1]}",
                        n_samples=int(X.shape[0]),
                        n_features=int(X.shape[1]),
                        n_informative=len(informative),
                        fit_seconds=fit_seconds,
                        n_features_used=n_used,
                        **recovery,
                    )
                    rows.append(res)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"{task} {name}: {time.perf_counter() - t0:.0f}s, {len(rows)} rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=Path("paper/data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, (task, name, source) in enumerate(plan(args.data_dir)):
        if i % args.num_shards == args.shard:
            run_item(task, name, source, args.output_dir)


if __name__ == "__main__":
    main()
