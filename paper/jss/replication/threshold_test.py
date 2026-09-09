"""Bonferroni versus max-type threshold test: calibration, scaling, and real data.

Three studies with the mc (classification) and pc (regression) selectors only.

1. Null calibration: response independent of the predictors; the false-split
   rate of a depth-1 tree over a grid of sample sizes and candidate counts.
2. Synthetic scaling: fitting time per tree on Gaussian predictors with a
   planted signal, over sample size and candidate count, both stopping modes.
3. Real subset: six mid-size benchmark datasets, both tests, both stopping
   modes, five folds; held-out score, depth, seconds per fit, and the Spearman
   agreement between the two tests' feature importances.

Timings are only comparable across cells within one run on one machine; the
receipt records the platform. Manuscript timings come from the cloud class the
performance section uses.
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

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from citrees import ConditionalInferenceTreeClassifier, ConditionalInferenceTreeRegressor

MODULE_PATH: Final = "paper/jss/replication/threshold_test.py"
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR: Final = REPO_ROOT / "paper" / "jss" / "results" / "threshold-test"
ANALYSIS: Final = "threshold_test"
ALPHA: Final = 0.05
TESTS: Final = ("bonferroni", "maxt")
STOPPING: Final = ("adaptive", None)
REAL_DATASETS: Final = {
    "classification": ("clf_vowel-context", "clf_spam", "clf_page-blocks"),
    "regression": ("reg_imports-85", "reg_residential", "reg_facebook"),
}
PROFILES: Final = {
    "full": {
        "null_reps": 2000,
        "null_n": (100, 200, 500),
        "null_k": (16, 64, 256),
        "scale_n": (1000, 4000, 16000),
        "scale_k": (16, 64, 256),
        "scale_reps": 3,
        "real_folds": 5,
        "real_k": 256,
    },
    "quick": {
        "null_reps": 300,
        "null_n": (200,),
        "null_k": (16, 64),
        "scale_n": (1000, 4000),
        "scale_k": (16, 64),
        "scale_reps": 1,
        "real_folds": 3,
        "real_k": 64,
    },
    "smoke": {
        "null_reps": 20,
        "null_n": (100,),
        "null_k": (16,),
        "scale_n": (500,),
        "scale_k": (16,),
        "scale_reps": 1,
        "real_folds": 2,
        "real_k": 16,
    },
}


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


def _estimator(task: str, test: str, stopping: str | None, k: int, seed: int, **extra: Any) -> Any:
    common = dict(
        threshold_method="histogram",
        max_thresholds=k,
        threshold_test=test,
        early_stopping_selector=stopping,
        early_stopping_splitter=stopping,
        alpha_selector=ALPHA,
        alpha_splitter=ALPHA,
        random_state=seed,
        **extra,
    )
    if task == "classification":
        return ConditionalInferenceTreeClassifier(selector="mc", **common)
    return ConditionalInferenceTreeRegressor(selector="pc", **common)


def _split_made(tree: dict[str, Any]) -> bool:
    return bool(tree.get("left_child"))


def _depth(node: dict[str, Any], d: int = 0) -> int:
    children = [node[c] for c in ("left_child", "right_child") if node.get(c)]
    return max([_depth(c, d + 1) for c in children] + [d])


def _wilson(rate: float, n: int) -> tuple[float, float]:
    half = 1.96 * np.sqrt(rate * (1 - rate) / n)
    return max(0.0, rate - half), min(1.0, rate + half)


def null_calibration(profile: dict[str, Any], seed: int) -> pd.DataFrame:
    rows = []
    for task in ("classification", "regression"):
        for n in profile["null_n"]:
            for k in profile["null_k"]:
                for test in TESTS:
                    for stopping in STOPPING:
                        rng = np.random.default_rng(seed)
                        rejections = 0
                        start = time.perf_counter()
                        for r in range(profile["null_reps"]):
                            X = rng.normal(size=(n, 5))
                            y = (
                                rng.integers(0, 2, n)
                                if task == "classification"
                                else rng.normal(size=n)
                            )
                            tree = _estimator(task, test, stopping, k, seed + r, max_depth=1).fit(
                                X, y
                            )
                            rejections += _split_made(tree.tree_)
                        rate = rejections / profile["null_reps"]
                        low, high = _wilson(rate, profile["null_reps"])
                        rows.append(
                            {
                                "study": "null_calibration",
                                "task": task,
                                "n": n,
                                "k": k,
                                "threshold_test": test,
                                "stopping": stopping or "exhaustive",
                                "replicates": profile["null_reps"],
                                "false_split_rate": rate,
                                "ci_low": low,
                                "ci_high": high,
                                "seconds_per_fit": (time.perf_counter() - start)
                                / profile["null_reps"],
                            }
                        )
                        print(
                            f"  null {task[:3]} n={n} k={k} {test} {stopping or 'exhaustive'}: {rate:.4f}",
                            flush=True,
                        )
    return pd.DataFrame(rows)


def synthetic_scaling(profile: dict[str, Any], seed: int) -> pd.DataFrame:
    rows = []
    for task in ("classification", "regression"):
        for n in profile["scale_n"]:
            rng = np.random.default_rng(seed)
            X = rng.normal(size=(n, 10))
            signal = X[:, 0] + 0.5 * X[:, 1] - 0.5 * X[:, 2]
            y = (
                (signal + rng.normal(size=n) > 0).astype(int)
                if task == "classification"
                else signal + rng.normal(size=n)
            )
            for k in profile["scale_k"]:
                for test in TESTS:
                    for stopping in STOPPING:
                        _estimator(task, test, stopping, k, seed).fit(X[:200], y[:200])  # warm JIT
                        seconds, depths = [], []
                        for r in range(profile["scale_reps"]):
                            start = time.perf_counter()
                            tree = _estimator(task, test, stopping, k, seed + r).fit(X, y)
                            seconds.append(time.perf_counter() - start)
                            depths.append(_depth(tree.tree_))
                        rows.append(
                            {
                                "study": "synthetic_scaling",
                                "task": task,
                                "n": n,
                                "k": k,
                                "threshold_test": test,
                                "stopping": stopping or "exhaustive",
                                "repeats": profile["scale_reps"],
                                "seconds_per_fit": float(np.median(seconds)),
                                "seconds_min": float(np.min(seconds)),
                                "seconds_max": float(np.max(seconds)),
                                "depth": float(np.mean(depths)),
                            }
                        )
                        print(
                            f"  scale {task[:3]} n={n} k={k} {test} {stopping or 'exhaustive'}: {np.median(seconds):.2f}s",
                            flush=True,
                        )
    return pd.DataFrame(rows)


def _load_real(task: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    frame = (
        pd.read_parquet(REPO_ROOT / "paper" / "data" / task / "real" / f"{name}.parquet")
        .select_dtypes("number")
        .dropna()
    )
    X = frame.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = frame.iloc[:, -1].to_numpy()
    if task == "classification":
        y = LabelEncoder().fit_transform(y)
    return X, y.astype(np.float64) if task == "regression" else y


def real_subset(profile: dict[str, Any], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, agreement = [], []
    k = profile["real_k"]
    for task, names in REAL_DATASETS.items():
        for name in names:
            X, y = _load_real(task, name)
            splitter = (StratifiedKFold if task == "classification" else KFold)(
                profile["real_folds"], shuffle=True, random_state=seed
            )
            for stopping in STOPPING:
                importances: dict[str, list[np.ndarray]] = {t: [] for t in TESTS}
                for test in TESTS:
                    scores, depths, seconds = [], [], []
                    for train, test_idx in splitter.split(X, y):
                        start = time.perf_counter()
                        tree = _estimator(task, test, stopping, k, seed).fit(X[train], y[train])
                        seconds.append(time.perf_counter() - start)
                        scores.append(tree.score(X[test_idx], y[test_idx]))
                        depths.append(_depth(tree.tree_))
                        importances[test].append(tree.feature_importances_)
                    rows.append(
                        {
                            "study": "real_subset",
                            "task": task,
                            "dataset": name,
                            "n": X.shape[0],
                            "p": X.shape[1],
                            "k": k,
                            "threshold_test": test,
                            "stopping": stopping or "exhaustive",
                            "folds": profile["real_folds"],
                            "score_mean": float(np.mean(scores)),
                            "score_sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                            "depth_mean": float(np.mean(depths)),
                            "seconds_per_fit": float(np.median(seconds)),
                        }
                    )
                    print(
                        f"  real {name} {test} {stopping or 'exhaustive'}: score {np.mean(scores):.3f} {np.median(seconds):.2f}s",
                        flush=True,
                    )
                rhos = [
                    spearmanr(a, b)[0]
                    for a, b in zip(importances["bonferroni"], importances["maxt"], strict=True)
                ]
                agreement.append(
                    {
                        "study": "real_subset_agreement",
                        "task": task,
                        "dataset": name,
                        "stopping": stopping or "exhaustive",
                        "k": k,
                        "folds": profile["real_folds"],
                        "importance_spearman_mean": float(np.nanmean(rhos)),
                        "importance_spearman_min": float(np.nanmin(rhos)),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(agreement)


def write_receipt(output_dir: Path, profile_name: str, artifacts: dict[str, Path]) -> None:
    versions = {
        p: metadata.version(p)
        for p in ("citrees", "numpy", "pandas", "scikit-learn", "scipy", "numba")
    }
    receipt = {
        "analysis": ANALYSIS,
        "profile": profile_name,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
        "source_sha256": {
            MODULE_PATH: _sha256(REPO_ROOT / MODULE_PATH),
            "pyproject.toml": _sha256(REPO_ROOT / "pyproject.toml"),
            "uv.lock": _sha256(REPO_ROOT / "uv.lock"),
        },
        "artifacts": {
            p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in artifacts.values()
        },
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="ascii"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="full")
    parser.add_argument("--seed", type=int, default=1718)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="Skip the synthetic scaling study (timings belong on the cloud class).",
    )
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    print(f"profile={args.profile} seed={args.seed}", flush=True)

    frames = {"null_calibration": null_calibration(profile, args.seed)}
    if not args.skip_timing:
        frames["synthetic_scaling"] = synthetic_scaling(profile, args.seed)
    real, agreement = real_subset(profile, args.seed)
    frames["real_subset"] = real
    frames["real_subset_agreement"] = agreement

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.iterdir():
        if stale.is_file():
            stale.unlink()
    artifacts: dict[str, Path] = {}
    for name, frame in frames.items():
        for suffix in ("parquet", "csv"):
            path = args.output_dir / f"{name}.{suffix}"
            frame.to_parquet(path, index=False) if suffix == "parquet" else frame.to_csv(
                path, index=False
            )
            artifacts[path.name] = path
    write_receipt(args.output_dir, args.profile, artifacts)
    print("\n== null calibration ==")
    print(
        frames["null_calibration"][
            [
                "task",
                "n",
                "k",
                "threshold_test",
                "stopping",
                "false_split_rate",
                "ci_low",
                "ci_high",
            ]
        ].to_string(index=False)
    )
    if "synthetic_scaling" in frames:
        print("\n== synthetic scaling (seconds per fit) ==")
        print(
            frames["synthetic_scaling"]
            .pivot_table(
                index=["task", "n", "k"],
                columns=["stopping", "threshold_test"],
                values="seconds_per_fit",
            )
            .round(3)
            .to_string()
        )
    print("\n== real subset ==")
    print(
        real[
            ["dataset", "threshold_test", "stopping", "score_mean", "depth_mean", "seconds_per_fit"]
        ].to_string(index=False)
    )
    print(agreement.to_string(index=False))


if __name__ == "__main__":
    main()
