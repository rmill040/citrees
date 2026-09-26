"""Decompose the benchmark forest's fitting time at the performance reference condition.

The condition is the reference cell of the controlled performance study (1,000
observations, 50 Gaussian predictors, signal in the first five). The benchmark
configuration (minimum budgets, 256-bin histogram, per-threshold Bonferroni
test) is fit under variants that isolate the sources of the forest's cost:

* the benchmark tree with all predictors and with square-root sampling;
* the benchmark forest fit serially (``n_jobs=1``) and in parallel;
* the benchmark forest without the threshold test, without the Bonferroni
  adjustment over candidate thresholds, and with the max-type test;
* the exhaustive tree and forest of the performance reference table.

Every fit records wall time, mean tree depth (edges from the root to the deepest
leaf, so a root-only tree has depth 0), and mean internal node count. Each
variant is fit once as a warm-up and then timed ``repeats`` times. The ``full``
profile is the manuscript protocol and requires a host with at least
``REFERENCE_LOGICAL_CPUS`` logical CPUs; ``quick`` and ``smoke`` reduce the
workload to validate the pipeline and make no timing claim.

Usage: ``python -m paper.jss.replication.performance_decomposition --profile full
--output-dir <new directory>``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Final

import numba
import numpy as np
import pandas as pd

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
)
from paper.benchmark.utils import get_hardware_metadata

ANALYSIS: Final = "performance_decomposition"
MODULE_PATH: Final = "paper/jss/replication/performance_decomposition.py"
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
REFERENCE_LOGICAL_CPUS: Final = 32
MAX_DEPTH, MIN_SAMPLES_SPLIT, MIN_SAMPLES_LEAF, ALPHA = 3, 20, 7, 0.05

PROFILES: Final[dict[str, dict[str, int]]] = {
    "full": {"n_samples": 1000, "n_features": 50, "n_trees": 100, "repeats": 3},
    "quick": {"n_samples": 1000, "n_features": 50, "n_trees": 20, "repeats": 1},
    "smoke": {"n_samples": 200, "n_features": 10, "n_trees": 4, "repeats": 1},
}

BENCHMARK: Final[dict[str, Any]] = {
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
EXHAUSTIVE: Final[dict[str, Any]] = {
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

# (variant, estimator kind, parameter overrides, forest n_jobs). The variant labels
# are the row keys of the tracked table paper/results/tables/paper_performance_decomposition.csv.
VARIANTS: Final[tuple[tuple[str, str, dict[str, Any], int | None], ...]] = (
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
SUMMARY_KEYS: Final = ["task", "variant", "kind", "n_jobs"]


def make_dataset(
    task: str, n_samples: int, n_features: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian predictors with a linear signal in the first five columns."""
    rng = np.random.default_rng(seed)
    X = np.ascontiguousarray(rng.standard_normal((n_samples, n_features)), dtype=np.float64)
    coefficients = np.linspace(1.0, 0.25, 5, dtype=np.float64)
    signal = X[:, :5] @ coefficients
    if task == "classification":
        latent = signal + rng.standard_normal(n_samples)
        return X, (latent >= np.median(latent)).astype(np.int64)
    y = signal + 0.5 * np.sin(X[:, 0]) + rng.standard_normal(n_samples)
    return X, np.ascontiguousarray(y, dtype=np.float64)


def _internal_nodes(node: dict[str, Any]) -> int:
    """Number of internal (split) nodes below and including ``node``."""
    if not node.get("left_child"):
        return 0
    return 1 + _internal_nodes(node["left_child"]) + _internal_nodes(node["right_child"])


def _depth(node: dict[str, Any], d: int = 0) -> int:
    """Edges from ``node`` to its deepest leaf."""
    children = [node[c] for c in ("left_child", "right_child") if node.get(c)]
    return max([_depth(c, d + 1) for c in children] + [d])


def build(
    task: str, kind: str, overrides: dict[str, Any], n_jobs: int | None, n_trees: int, seed: int
) -> Any:
    """Construct the estimator for one variant."""
    base = dict(EXHAUSTIVE if kind.startswith("exhaustive") else BENCHMARK)
    base.update(overrides)
    base["selector"] = "mc" if task == "classification" else "pc"
    base["random_state"] = seed
    clf = task == "classification"
    if kind.endswith("tree"):
        return (ConditionalInferenceTreeClassifier if clf else ConditionalInferenceTreeRegressor)(
            **base
        )
    forest = ConditionalInferenceForestClassifier if clf else ConditionalInferenceForestRegressor
    extra: dict[str, Any] = {"n_estimators": n_trees, "n_jobs": n_jobs}
    if kind.startswith("exhaustive"):
        extra.update(max_features=None, bootstrap=True, max_samples=None)
        if clf:
            extra["sampling_method"] = None
    return forest(**base, **extra)


def summarize_fit(model: Any, kind: str, seconds: float) -> dict[str, Any]:
    """Per-fit wall time and tree-structure summaries."""
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


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Median, range, and structure per variant: the tracked decomposition table's schema."""
    return raw.groupby(SUMMARY_KEYS, as_index=False, sort=False, dropna=False).agg(
        seconds_median=("seconds", "median"),
        seconds_min=("seconds", "min"),
        seconds_max=("seconds", "max"),
        depth_mean=("depth_mean", "mean"),
        internal_nodes_mean=("internal_nodes_mean", "mean"),
        repeats=("seconds", "size"),
    )


def run(profile: dict[str, int], seed: int) -> pd.DataFrame:
    """Fit every variant for both tasks and return one row per timed fit."""
    rows = []
    for task in ("classification", "regression"):
        X, y = make_dataset(task, profile["n_samples"], profile["n_features"], seed)
        for variant, kind, overrides, n_jobs in VARIANTS:
            build(task, kind, overrides, n_jobs, profile["n_trees"], seed).fit(X, y)  # warm-up
            for rep in range(profile["repeats"]):
                model = build(task, kind, overrides, n_jobs, profile["n_trees"], seed + rep)
                start = time.perf_counter()
                model.fit(X, y)
                row: dict[str, Any] = {
                    "task": task,
                    "variant": variant,
                    "kind": kind,
                    "n_jobs": n_jobs,
                    "repeat": rep,
                }
                row.update(summarize_fit(model, kind, time.perf_counter() - start))
                rows.append(row)
                print(
                    f"{task[:3]} {variant}: {row['seconds']:.3f}s depth {row['depth_mean']:.2f} "
                    f"nodes {row['internal_nodes_mean']:.2f}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def write_receipt(
    output_dir: Path,
    profile_name: str,
    seed: int,
    hardware: dict[str, Any],
    elapsed_seconds: float,
    artifacts: list[Path],
) -> None:
    """Write the suite receipt: revision, environment, sources, and artifact hashes."""
    receipt = {
        "analysis": ANALYSIS,
        "profile": profile_name,
        "seed": seed,
        "design": PROFILES[profile_name],
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version,
        "platform": platform.platform(),
        "hardware": {**hardware, "numba_threads": numba.get_num_threads()},
        "elapsed_seconds": elapsed_seconds,
        "versions": {
            p: metadata.version(p)
            for p in ("citrees", "numpy", "pandas", "scikit-learn", "scipy", "numba")
        },
        "source_sha256": {
            MODULE_PATH: _sha256(REPO_ROOT / MODULE_PATH),
            "pyproject.toml": _sha256(REPO_ROOT / "pyproject.toml"),
            "uv.lock": _sha256(REPO_ROOT / "uv.lock"),
        },
        "artifacts": {p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in artifacts},
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="ascii"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="full")
    parser.add_argument("--seed", type=int, default=1718)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    hardware = get_hardware_metadata()
    if args.profile == "full" and int(hardware["logical_cpus"]) < REFERENCE_LOGICAL_CPUS:
        raise RuntimeError(
            f"the full profile requires {REFERENCE_LOGICAL_CPUS} logical CPUs; "
            f"this host has {hardware['logical_cpus']}"
        )
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    started = time.perf_counter()
    raw = run(PROFILES[args.profile], args.seed)
    args.output_dir.mkdir(parents=True)
    raw_path = args.output_dir / "performance_decomposition.csv"
    summary_path = args.output_dir / "performance_decomposition_summary.csv"
    raw.to_csv(raw_path, index=False)
    summary = summarize(raw)
    summary.to_csv(summary_path, index=False)
    write_receipt(
        args.output_dir,
        args.profile,
        args.seed,
        hardware,
        time.perf_counter() - started,
        [raw_path, summary_path],
    )
    print(summary.set_index(["task", "variant"])["seconds_median"].to_string())


if __name__ == "__main__":
    main()
