"""Decompose the recommended forest's fitting time at the reference condition.

Table 5 of the JSS article reports the recommended single tree at 0.05 s and
the recommended 100-tree forest at 21 s, while the exhaustive tree and forest
take 1.6 and 6.3 s. This script fits the reference condition of the first
performance study (1,000 observations, 50 Gaussian predictors, signal in the
first five) under variants that isolate the sources of the forest's cost:

* the recommended tree with all predictors and with square-root sampling;
* the recommended forest fit serially (``n_jobs=1``) and in parallel;
* the recommended forest without the threshold test, without the Bonferroni
  adjustment over candidate thresholds, and with the max-type test;
* the exhaustive forest of Table 5 for reference.

Every fit records wall time, mean tree depth, and mean internal node count,
so the per-tree work can
be compared with the single tree. Three repeats per variant after one warm-up.

Usage: ``python -m paper.jss.replication.performance_decomposition --output-dir /out/decomp``
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pandas as pd

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
)

N_SAMPLES, N_FEATURES, N_TREES, REPEATS = 1000, 50, 100, 3
MAX_DEPTH, MIN_SAMPLES_SPLIT, MIN_SAMPLES_LEAF, ALPHA = 3, 20, 7, 0.05

RECOMMENDED: dict[str, Any] = {
    "alpha_selector": ALPHA,
    "adjust_alpha_selector": True,
    "n_resamples_selector": "minimum",
    "early_stopping_selector": "adaptive",
    "early_stopping_confidence_selector": 0.95,
    "n_resamples_splitter": "minimum",
    "adjust_alpha_splitter": True,
    "early_stopping_splitter": "adaptive",
    "early_stopping_confidence_splitter": 0.95,
    "feature_muting": True,
    "feature_scanning": True,
    "threshold_scanning": True,
    "threshold_method": "histogram",
    "max_thresholds": 256,
    "max_depth": MAX_DEPTH,
    "min_samples_split": MIN_SAMPLES_SPLIT,
    "min_samples_leaf": MIN_SAMPLES_LEAF,
    "verbose": 0,
}
EXHAUSTIVE: dict[str, Any] = {
    "alpha_selector": ALPHA,
    "adjust_alpha_selector": True,
    "n_resamples_selector": 999,
    "scale_resamples_with_tests": False,
    "early_stopping_selector": None,
    "n_resamples_splitter": None,
    "adjust_alpha_splitter": False,
    "early_stopping_splitter": None,
    "feature_muting": False,
    "feature_scanning": False,
    "threshold_scanning": False,
    "threshold_method": "exact",
    "max_thresholds": None,
    "max_depth": MAX_DEPTH,
    "min_samples_split": MIN_SAMPLES_SPLIT,
    "min_samples_leaf": MIN_SAMPLES_LEAF,
    "verbose": 0,
}

# (variant, estimator kind, parameter overrides, forest n_jobs)
VARIANTS: tuple[tuple[str, str, dict[str, Any], int | None], ...] = (
    ("recommended tree, all predictors", "tree", {}, None),
    ("recommended tree, sqrt predictors", "tree", {"max_features": "sqrt"}, None),
    ("recommended forest, serial", "forest", {}, 1),
    ("recommended forest, parallel", "forest", {}, -1),
    (
        "recommended forest, parallel, no threshold test",
        "forest",
        {"n_resamples_splitter": None},
        -1,
    ),
    ("recommended forest, parallel, no adjustment", "forest", {"adjust_alpha_splitter": False}, -1),
    ("recommended forest, parallel, max-type", "forest", {"threshold_test": "maxt"}, -1),
    ("exhaustive tree, all predictors", "exhaustive-tree", {}, None),
    ("exhaustive forest, parallel", "exhaustive-forest", {}, -1),
)


def make_dataset(task: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.ascontiguousarray(rng.standard_normal((N_SAMPLES, N_FEATURES)), dtype=np.float64)
    coefficients = np.linspace(1.0, 0.25, 5, dtype=np.float64)
    signal = X[:, :5] @ coefficients
    if task == "classification":
        latent = signal + rng.standard_normal(N_SAMPLES)
        return X, (latent >= np.median(latent)).astype(np.int64)
    y = signal + 0.5 * np.sin(X[:, 0]) + rng.standard_normal(N_SAMPLES)
    return X, np.ascontiguousarray(y, dtype=np.float64)


def _internal_nodes(node: dict[str, Any]) -> int:
    if not node.get("left_child"):
        return 0
    return 1 + _internal_nodes(node["left_child"]) + _internal_nodes(node["right_child"])


def _depth(node: dict[str, Any], d: int = 0) -> int:
    children = [node[c] for c in ("left_child", "right_child") if node.get(c)]
    return max([_depth(c, d + 1) for c in children] + [d])


def build(task: str, kind: str, overrides: dict[str, Any], n_jobs: int | None, seed: int) -> Any:
    base = dict(EXHAUSTIVE if kind.startswith("exhaustive") else RECOMMENDED)
    base.update(overrides)
    base["selector"] = "mc" if task == "classification" else "pc"
    base["random_state"] = seed
    clf = task == "classification"
    if kind.endswith("tree"):
        return (ConditionalInferenceTreeClassifier if clf else ConditionalInferenceTreeRegressor)(
            **base
        )
    forest = ConditionalInferenceForestClassifier if clf else ConditionalInferenceForestRegressor
    extra: dict[str, Any] = {"n_estimators": N_TREES, "n_jobs": n_jobs}
    if kind.startswith("exhaustive"):
        extra.update(max_features=None, bootstrap=True, max_samples=None)
        if clf:
            extra["sampling_method"] = None
    return forest(**base, **extra)


def summarize(model: Any, kind: str, seconds: float) -> dict[str, Any]:
    trees = [model] if kind.endswith("tree") else list(model.estimators_)
    depths = [_depth(t.tree_) for t in trees]
    nodes = [_internal_nodes(t.tree_) for t in trees]
    return {
        "seconds": seconds,
        "trees": len(trees),
        "depth_mean": float(np.mean(depths)),
        "internal_nodes_mean": float(np.mean(nodes)),
        "internal_nodes_total": int(np.sum(nodes)),
        "seconds_per_internal_node": seconds / max(1, int(np.sum(nodes))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1718)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in ("classification", "regression"):
        X, y = make_dataset(task, args.seed)
        for variant, kind, overrides, n_jobs in VARIANTS:
            build(task, kind, overrides, n_jobs, args.seed).fit(X, y)  # warm-up
            for rep in range(REPEATS):
                model = build(task, kind, overrides, n_jobs, args.seed + rep)
                start = time.perf_counter()
                model.fit(X, y)
                row = {
                    "task": task,
                    "variant": variant,
                    "kind": kind,
                    "n_jobs": n_jobs,
                    "repeat": rep,
                }
                row.update(summarize(model, kind, time.perf_counter() - start))
                rows.append(row)
                print(
                    f"{task[:3]} {variant}: {row['seconds']:.3f}s depth {row['depth_mean']:.2f} "
                    f"nodes {row['internal_nodes_mean']:.2f}",
                    flush=True,
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "performance_decomposition.csv", index=False)
    meta = {
        "numba_threads": numba.get_num_threads(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
        "n_trees": N_TREES,
        "repeats": REPEATS,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(frame.groupby(["task", "variant"], sort=False)["seconds"].median().to_string())


if __name__ == "__main__":
    main()
