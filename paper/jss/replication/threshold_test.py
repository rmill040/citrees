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
import concurrent.futures
import hashlib
import json
import multiprocessing
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from citrees import ConditionalInferenceTreeClassifier, ConditionalInferenceTreeRegressor

MODULE_PATH: Final = "paper/jss/replication/threshold_test.py"
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR: Final = REPO_ROOT / "paper" / "jss" / "results" / "threshold-test"
ANALYSIS: Final = "threshold_test"
ALPHA: Final = 0.05
DEFAULT_FIT_TIMEOUT: Final = 1200.0
TESTS: Final = ("bonferroni", "maxt")
# bonferroni_fixed: the per-threshold Bonferroni rule at a fixed budget of FIXED_B
# permutations per candidate (scale_resamples_with_tests=False), the control arm that
# separates the budget rule from the test; it cannot reject once alpha/K < 1/(FIXED_B+1).
# bonferroni_unadjusted: the per-threshold rule at level alpha for every candidate,
# the "No adjustment" column of the JSS timing tables (adjust_alpha_splitter=False).
TEST_CHOICES: Final = ("bonferroni", "maxt", "bonferroni_fixed", "bonferroni_unadjusted")
FIXED_B: Final = 999
# Location of the planted step in the power studies as a standard-normal quantile
# (0.5 is the centred cut at zero); set from --step-quantile.
STEP_QUANTILE: float = 0.5
STEP_Z: float = 0.0
# Nominal Stage B levels for the Stage-B-only studies; the realized size at each
# level lets power be read at a matched realized size.
STAGEB_ALPHAS: Final = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50)
STOPPING: Final = ("adaptive", None)
REAL_DATASETS: Final = {
    "classification": ("clf_vowel-context", "clf_spam", "clf_page-blocks"),
    "regression": ("reg_imports-85", "reg_residential", "reg_facebook"),
}
PROFILES: Final = {
    "full": {
        "stageb_size_reps": 500,
        "stageb_power_reps": 200,
        "power_reps": 500,
        "power_effects_classification": (0.05, 0.10, 0.20),
        "power_effects_regression": (0.2, 0.4, 0.8),
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
        "stageb_size_reps": 200,
        "stageb_power_reps": 100,
        "power_reps": 100,
        "power_effects_classification": (0.10, 0.20),
        "power_effects_regression": (0.4, 0.8),
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
        "stageb_size_reps": 10,
        "stageb_power_reps": 10,
        "power_reps": 10,
        "power_effects_classification": (0.20,),
        "power_effects_regression": (0.8,),
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
        threshold_test="bonferroni" if test.startswith("bonferroni") else test,
        early_stopping_selector=stopping,
        early_stopping_splitter=stopping,
        alpha_selector=ALPHA,
        alpha_splitter=ALPHA,
        random_state=seed,
    )
    if test == "bonferroni_fixed":
        common.update(n_resamples_splitter=FIXED_B, scale_resamples_with_tests=False)
    if test == "bonferroni_unadjusted":
        common.update(adjust_alpha_splitter=False)
    common.update(extra)
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


def _fit_worker(args: tuple[Any, ...]) -> dict[str, Any]:
    task, test, stopping, k, seed, extra, X, y = args
    # The child is a fresh process, so an identical untimed fit first excludes
    # compilation (or cache loading) of the kernels this cell selects.
    _estimator(task, test, stopping, k, seed, **extra).fit(X, y)
    start = time.perf_counter()
    tree = _estimator(task, test, stopping, k, seed, **extra).fit(X, y)
    return {
        "seconds": time.perf_counter() - start,
        "depth": _depth(tree.tree_),
        "importances": np.asarray(tree.feature_importances_),
        "tree": tree,
    }


def timed_fit(
    task: str,
    test: str,
    stopping: str | None,
    k: int,
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    timeout: float,
    **extra: Any,
) -> dict[str, Any] | None:
    """Fit in a child process; return None when the fit exceeds ``timeout`` seconds.

    A fit that cannot finish within the cap is censored, the same rule the
    performance section applies to its 48-hour cells. The child is killed so a
    runaway Bonferroni cell cannot stall the study.
    """
    # Spawn, never fork: the parent has already started Numba/OpenMP threads and
    # forking such a process is unsafe on Linux (the child is killed on start).
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    )
    future = executor.submit(_fit_worker, (task, test, stopping, k, seed, extra, X, y))
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        for proc in getattr(executor, "_processes", {}).values():
            proc.kill()
        return None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def null_calibration(
    profile: dict[str, Any], seed: int, tests: tuple[str, ...] = TESTS
) -> pd.DataFrame:
    rows = []
    for task in ("classification", "regression"):
        for n in profile["null_n"]:
            for k in profile["null_k"]:
                for test in tests:
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


def power_study(profile: dict[str, Any], seed: int, tests: tuple[str, ...] = TESTS) -> pd.DataFrame:
    """Detection of a planted root split by the two constructions.

    Five Gaussian predictors; the response depends on the first through a step at
    zero: in classification P(y=1) = 1/2 + delta on one side and 1/2 - delta on the
    other, in regression y = delta * 1[x_1 > 0] + noise. The same grid of sample
    sizes and candidate counts as the null calibration. Recorded per cell: the
    rate of any root split (power), the rate of a split on the planted predictor,
    and the median absolute error of the chosen threshold.
    """
    rows = []
    for task in ("classification", "regression"):
        effects = profile[f"power_effects_{task}"]
        for delta in effects:
            for n in profile["null_n"]:
                for k in profile["null_k"]:
                    for test in tests:
                        for stopping in STOPPING:
                            rng = np.random.default_rng(seed)
                            detected = correct = 0
                            errors: list[float] = []
                            start = time.perf_counter()
                            for r in range(profile["power_reps"]):
                                X = rng.normal(size=(n, 5))
                                side = X[:, 0] > STEP_Z
                                if task == "classification":
                                    prob = np.where(side, 0.5 + delta, 0.5 - delta)
                                    y = (rng.uniform(size=n) < prob).astype(np.int64)
                                else:
                                    y = delta * side.astype(float) + rng.normal(size=n)
                                tree = _estimator(
                                    task, test, stopping, k, seed + r, max_depth=1
                                ).fit(X, y)
                                node = tree.tree_
                                if _split_made(node):
                                    detected += 1
                                    if int(node["feature"]) == 0:
                                        correct += 1
                                        errors.append(abs(float(node["threshold"])))
                            reps = profile["power_reps"]
                            rows.append(
                                {
                                    "study": "power",
                                    "task": task,
                                    "effect": delta,
                                    "step_quantile": STEP_QUANTILE,
                                    "n": n,
                                    "k": k,
                                    "threshold_test": test,
                                    "stopping": stopping or "exhaustive",
                                    "replicates": reps,
                                    "power": detected / reps,
                                    "correct_feature_rate": correct / reps,
                                    "threshold_abs_error_median": (
                                        float(np.median(errors)) if errors else float("nan")
                                    ),
                                    "seconds_per_fit": (time.perf_counter() - start) / reps,
                                }
                            )
                            print(
                                f"power {task} delta={delta} n={n} k={k} {test} "
                                f"{stopping or 'exhaustive'}: {detected / reps:.3f}",
                                flush=True,
                            )
    return pd.DataFrame(rows)


def _stageb_only_rate(
    task: str,
    test: str,
    stopping: str | None,
    k: int,
    n: int,
    seed: int,
    reps: int,
    alpha_split: float,
    delta: float,
) -> tuple[float, float]:
    """Split rate of a depth-one tree on ONE predictor with the Stage A gate disabled.

    ``alpha_selector=1`` lets the single predictor through Stage A unconditionally, so
    the split decision is the Stage B test alone at nominal level ``alpha_split``,
    with the feature fixed independently of the response, which is the setting of the
    fixed-node Stage B corollary. ``delta`` = 0 gives the null (size); ``delta`` > 0
    plants a step at zero (power). Returns (rate, seconds per fit).
    """
    rng = np.random.default_rng(seed)
    splits = 0
    start = time.perf_counter()
    for r in range(reps):
        X = rng.normal(size=(n, 1))
        side = X[:, 0] > STEP_Z
        if task == "classification":
            prob = np.where(side, 0.5 + delta, 0.5 - delta)
            y = (rng.uniform(size=n) < prob).astype(np.int64)
        else:
            y = delta * side.astype(float) + rng.normal(size=n)
        tree = _estimator(
            task,
            test,
            stopping,
            k,
            seed + r,
            max_depth=1,
            alpha_selector=1.0,
            alpha_splitter=alpha_split,
        ).fit(X, y)
        splits += _split_made(tree.tree_)
    return splits / reps, (time.perf_counter() - start) / reps


def stageb_only(
    profile: dict[str, Any],
    seed: int,
    tasks: tuple[str, ...],
    sizes: tuple[int, ...] | None = None,
    tests: tuple[str, ...] = TESTS,
) -> pd.DataFrame:
    """Stage-B-only size and power over a grid of nominal levels.

    For each task, sample size, bin setting, construction, stopping mode, and nominal
    Stage B level, the null split rate (size) and the split rate under each planted
    effect (power) of the Stage B test alone. Reading power at the level whose
    realized size is 0.05 gives size-matched power.
    """
    rows = []
    for task in tasks:
        effects = (0.0,) + tuple(profile[f"power_effects_{task}"])
        for n in sizes or profile["null_n"]:
            for k in profile["null_k"]:
                for test in tests:
                    for stopping in STOPPING:
                        for alpha_split in STAGEB_ALPHAS:
                            for delta in effects:
                                reps = (
                                    profile["stageb_size_reps"]
                                    if delta == 0.0
                                    else profile["stageb_power_reps"]
                                )
                                rate, sec = _stageb_only_rate(
                                    task, test, stopping, k, n, seed, reps, alpha_split, delta
                                )
                                low, high = _wilson(rate, reps)
                                rows.append(
                                    {
                                        "study": "stageb_only",
                                        "task": task,
                                        "n": n,
                                        "k": k,
                                        "threshold_test": test,
                                        "stopping": stopping or "exhaustive",
                                        "nominal_alpha": alpha_split,
                                        "effect": delta,
                                        "step_quantile": STEP_QUANTILE,
                                        "replicates": reps,
                                        "split_rate": rate,
                                        "ci_low": low,
                                        "ci_high": high,
                                        "seconds_per_fit": sec,
                                    }
                                )
                                print(
                                    f"stageb {task} n={n} k={k} {test} {stopping or 'exhaustive'} "
                                    f"alpha={alpha_split} delta={delta}: {rate:.4f}",
                                    flush=True,
                                )
    return pd.DataFrame(rows)


def synthetic_scaling(
    profile: dict[str, Any],
    seed: int,
    timeout: float = DEFAULT_FIT_TIMEOUT,
    tests: tuple[str, ...] = TESTS,
) -> pd.DataFrame:
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
                for test in tests:
                    for stopping in STOPPING:
                        _estimator(task, test, stopping, k, seed).fit(X[:200], y[:200])  # warm JIT
                        seconds, depths, censored = [], [], False
                        for r in range(profile["scale_reps"]):
                            result = timed_fit(task, test, stopping, k, seed + r, X, y, timeout)
                            if result is None:
                                censored = True
                                break
                            seconds.append(result["seconds"])
                            depths.append(result["depth"])
                        rows.append(
                            {
                                "study": "synthetic_scaling",
                                "task": task,
                                "n": n,
                                "k": k,
                                "threshold_test": test,
                                "stopping": stopping or "exhaustive",
                                "repeats": len(seconds),
                                "censored": censored,
                                "timeout_seconds": timeout,
                                "seconds_per_fit": float(np.median(seconds))
                                if seconds
                                else float("nan"),
                                "seconds_min": float(np.min(seconds)) if seconds else float("nan"),
                                "seconds_max": float(np.max(seconds)) if seconds else float("nan"),
                                "depth": float(np.mean(depths)) if depths else float("nan"),
                            }
                        )
                        shown = "censored" if censored else f"{np.median(seconds):.2f}s"
                        print(
                            f"  scale {task[:3]} n={n} k={k} {test} {stopping or 'exhaustive'}: {shown}",
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


def real_subset(
    profile: dict[str, Any],
    seed: int,
    timeout: float = DEFAULT_FIT_TIMEOUT,
    tests: tuple[str, ...] = TESTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                censored_any = False
                for test in tests:
                    scores, depths, seconds, censored = [], [], [], False
                    for train, test_idx in splitter.split(X, y):
                        result = timed_fit(
                            task, test, stopping, k, seed, X[train], y[train], timeout
                        )
                        if result is None:
                            censored = True
                            break
                        seconds.append(result["seconds"])
                        scores.append(result["tree"].score(X[test_idx], y[test_idx]))
                        depths.append(result["depth"])
                        importances[test].append(result["importances"])
                    censored_any = censored_any or censored
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
                            "folds": len(scores),
                            "censored": censored,
                            "timeout_seconds": timeout,
                            "score_mean": float(np.mean(scores)) if scores else float("nan"),
                            "score_sd": float(np.std(scores, ddof=1))
                            if len(scores) > 1
                            else float("nan"),
                            "depth_mean": float(np.mean(depths)) if depths else float("nan"),
                            "seconds_per_fit": float(np.median(seconds))
                            if seconds
                            else float("nan"),
                        }
                    )
                    shown = (
                        "censored"
                        if censored
                        else f"score {np.mean(scores):.3f} {np.median(seconds):.2f}s"
                    )
                    print(f"  real {name} {test} {stopping or 'exhaustive'}: {shown}", flush=True)
                pairs = list(zip(importances["bonferroni"], importances["maxt"], strict=False))
                rhos = [spearmanr(a, b)[0] for a, b in pairs]
                agreement.append(
                    {
                        "study": "real_subset_agreement",
                        "task": task,
                        "dataset": name,
                        "stopping": stopping or "exhaustive",
                        "k": k,
                        "folds_compared": len(pairs),
                        "censored": censored_any,
                        "importance_spearman_mean": float(np.nanmean(rhos))
                        if rhos
                        else float("nan"),
                        "importance_spearman_min": float(np.nanmin(rhos)) if rhos else float("nan"),
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
    parser.add_argument(
        "--studies",
        nargs="+",
        choices=("calibration", "scaling", "real", "power", "stageb"),
        default=("calibration", "scaling", "real"),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("classification", "regression"),
        default=("classification", "regression"),
        help="Tasks for the Stage-B-only study (for sharding).",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=None,
        help="Sample sizes for the Stage-B-only study (for sharding).",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=TEST_CHOICES,
        default=TESTS,
        help=(
            "Stage B constructions to run (bonferroni_fixed is the fixed-budget control arm, "
            "bonferroni_unadjusted the per-threshold rule without the Bonferroni adjustment)."
        ),
    )
    parser.add_argument(
        "--step-quantile",
        type=float,
        default=0.5,
        help="Location of the planted step as a standard-normal quantile (0.5 = at zero).",
    )
    parser.add_argument(
        "--stageb-size-reps", type=int, default=None, help="Override profile null reps."
    )
    parser.add_argument(
        "--stageb-power-reps", type=int, default=None, help="Override profile power reps."
    )
    parser.add_argument(
        "--fit-timeout",
        type=float,
        default=DEFAULT_FIT_TIMEOUT,
        help="Seconds allowed per fit before the cell is censored.",
    )
    args = parser.parse_args()
    studies = set(args.studies)
    if args.skip_timing:
        studies.discard("scaling")
    profile = dict(PROFILES[args.profile])
    if args.stageb_size_reps:
        profile["stageb_size_reps"] = args.stageb_size_reps
    if args.stageb_power_reps:
        profile["stageb_power_reps"] = args.stageb_power_reps
    global STEP_QUANTILE, STEP_Z
    STEP_QUANTILE = float(args.step_quantile)
    STEP_Z = float(stats.norm.ppf(STEP_QUANTILE))
    tests = tuple(args.tests)
    print(
        f"profile={args.profile} seed={args.seed} tests={tests} step_quantile={STEP_QUANTILE}",
        flush=True,
    )

    frames: dict[str, pd.DataFrame] = {}
    if "calibration" in studies:
        frames["null_calibration"] = null_calibration(profile, args.seed, tests)
    if "power" in studies:
        frames["power"] = power_study(profile, args.seed, tests)
    if "stageb" in studies:
        frames["stageb_only"] = stageb_only(
            profile, args.seed, tuple(args.tasks), tuple(args.sizes) if args.sizes else None, tests
        )
    if "scaling" in studies:
        frames["synthetic_scaling"] = synthetic_scaling(profile, args.seed, args.fit_timeout, tests)
    if "real" in studies:
        real, agreement = real_subset(profile, args.seed, args.fit_timeout, tests)
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
    if "null_calibration" in frames:
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
    if "real_subset" in frames:
        print("\n== real subset ==")
        print(
            frames["real_subset"][
                [
                    "dataset",
                    "threshold_test",
                    "stopping",
                    "censored",
                    "score_mean",
                    "depth_mean",
                    "seconds_per_fit",
                ]
            ].to_string(index=False)
        )
        print(frames["real_subset_agreement"].to_string(index=False))


if __name__ == "__main__":
    main()
