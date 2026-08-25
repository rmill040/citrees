"""Measure fit time and peak memory for matched tree and forest implementations.

Each measured cell runs in a fresh subprocess with each implementation's native
full-machine parallelism (citrees via Numba threading, scikit-learn forests via
n_jobs=-1, partykit as shipped). The fresh subprocess isolates the process peak
resident-set size, excludes one-time imports and JIT compilation from fit
timing, and prevents an earlier large fit from contaminating a later cell's
memory high-water mark.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict, cast

import numpy as np
import pandas as pd

Task = Literal["classification", "regression"]
Profile = Literal["smoke", "quick", "full"]
ModelFamily = Literal["tree", "forest"]
Method = Literal["citrees", "partykit", "sklearn"]
Axis = Literal[
    "reference",
    "sample_size",
    "predictor_count",
    "selector",
    "permutation_budget",
    "forest_size",
]

TASKS: tuple[Task, ...] = ("classification", "regression")
MODEL_FAMILIES: tuple[ModelFamily, ...] = ("tree", "forest")
METHODS: tuple[Method, ...] = ("citrees", "partykit", "sklearn")

BASE_SEED = 1718
ALPHA = 0.05
MAX_DEPTH = 3
MIN_SAMPLES_SPLIT = 20
MIN_SAMPLES_LEAF = 7
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "performance"
# BLAS pools stay at one thread to avoid nested oversubscription; Numba carries
# the citrees parallelism and scikit-learn forests parallelize over trees.
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "32",
    "NUMBA_DISABLE_JIT": "0",
}

PERFORMANCE_RAW_SCHEMA = (
    "cell_id",
    "profile",
    "task",
    "model_family",
    "method",
    "axis",
    "axis_value",
    "repeat",
    "data_seed",
    "model_seed",
    "input_sha256",
    "n_samples",
    "n_features",
    "selector",
    "n_resamples",
    "n_estimators",
    "worker_pid",
    "elapsed_seconds",
    "baseline_peak_rss_bytes",
    "peak_rss_bytes",
    "incremental_peak_rss_bytes",
    "fit_result_size",
)

PERFORMANCE_SUMMARY_SCHEMA = (
    "profile",
    "task",
    "model_family",
    "method",
    "axis",
    "axis_value",
    "n_samples",
    "n_features",
    "selector",
    "n_resamples",
    "n_estimators",
    "n_repeats",
    "median_elapsed_seconds",
    "lower_quartile_elapsed_seconds",
    "upper_quartile_elapsed_seconds",
    "median_peak_rss_bytes",
    "lower_quartile_peak_rss_bytes",
    "upper_quartile_peak_rss_bytes",
    "median_incremental_peak_rss_bytes",
    "elapsed_ratio_to_reference",
    "peak_rss_ratio_to_reference",
)


@dataclass(frozen=True)
class PerformanceSettings:
    """Controlled dimensions for one replication profile."""

    repeats: int
    baseline_n_samples: int
    baseline_n_features: int
    baseline_n_resamples: int
    baseline_n_estimators: int
    sample_sizes: tuple[int, ...]
    predictor_counts: tuple[int, ...]
    permutation_budgets: tuple[int, ...]
    forest_sizes: tuple[int, ...]


@dataclass(frozen=True)
class PerformanceCell:
    """Complete immutable specification for one isolated fit."""

    cell_id: str
    profile: Profile
    task: Task
    model_family: ModelFamily
    method: Method
    axis: Axis
    axis_value: str
    repeat: int
    data_seed: int
    model_seed: int
    n_samples: int
    n_features: int
    selector: str
    n_resamples: int
    n_estimators: int


class CitreesControl(TypedDict):
    """Typed controls shared by citrees trees and forests."""

    selector: str
    alpha_selector: float
    adjust_alpha_selector: bool
    n_resamples_selector: int
    early_stopping_selector: str | None
    n_resamples_splitter: int | None
    adjust_alpha_splitter: bool
    early_stopping_splitter: str | None
    feature_muting: bool
    feature_scanning: bool
    threshold_scanning: bool
    threshold_method: str
    max_thresholds: int | None
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    random_state: int
    verbose: int


class CitreesForestControl(CitreesControl):
    """Typed controls added by citrees forests."""

    n_estimators: int
    max_features: int | float | str | None
    bootstrap: bool
    max_samples: int | float | None
    n_jobs: int


class PartykitControl(TypedDict):
    """Typed controls shared by partykit trees and forests."""

    task: str
    teststat: str
    testtype: str | tuple[str, ...]
    mincriterion: float
    nresample: int
    maxdepth: int
    minsplit: int
    minbucket: int
    random_state: int


CellRunner = Callable[[PerformanceCell], dict[str, object]]


def _settings(profile: Profile) -> PerformanceSettings:
    if profile == "smoke":
        return PerformanceSettings(
            repeats=1,
            baseline_n_samples=96,
            baseline_n_features=6,
            baseline_n_resamples=39,
            baseline_n_estimators=3,
            sample_sizes=(),
            predictor_counts=(),
            permutation_budgets=(),
            forest_sizes=(),
        )
    if profile == "quick":
        return PerformanceSettings(
            repeats=2,
            baseline_n_samples=500,
            baseline_n_features=20,
            baseline_n_resamples=199,
            baseline_n_estimators=25,
            sample_sizes=(250, 1_000),
            predictor_counts=(10, 50),
            permutation_budgets=(99, 499),
            forest_sizes=(10, 50),
        )
    if profile == "full":
        return PerformanceSettings(
            repeats=10,
            baseline_n_samples=1_000,
            baseline_n_features=50,
            baseline_n_resamples=999,
            baseline_n_estimators=100,
            sample_sizes=(250, 4_000),
            predictor_counts=(10, 200),
            permutation_budgets=(99, 9_999),
            forest_sizes=(25, 500),
        )
    raise ValueError(f"unknown performance profile: {profile}")


def _default_selector(task: Task) -> str:
    return "mc" if task == "classification" else "pc"


def _selector_alternatives(task: Task) -> tuple[str, ...]:
    return ("mi", "rdc") if task == "classification" else ("dc", "rdc")


def _stable_seed(base_seed: int, stream: str, values: tuple[object, ...]) -> int:
    descriptor = json.dumps(
        [stream, *values],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(descriptor.encode("ascii")).digest()
    key = int.from_bytes(digest[:4], byteorder="little", signed=False)
    sequence = np.random.SeedSequence([base_seed, key])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _cell_identity(values: dict[str, object]) -> str:
    payload = json.dumps(
        values,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _new_cell(
    *,
    profile: Profile,
    task: Task,
    model_family: ModelFamily,
    method: Method,
    axis: Axis,
    axis_value: str,
    repeat: int,
    n_samples: int,
    n_features: int,
    selector: str,
    n_resamples: int,
    n_estimators: int,
    base_seed: int,
) -> PerformanceCell:
    paired_values = (task, n_samples, n_features, repeat)
    values: dict[str, object] = {
        "profile": profile,
        "task": task,
        "model_family": model_family,
        "method": method,
        "axis": axis,
        "axis_value": axis_value,
        "repeat": repeat,
        "data_seed": _stable_seed(base_seed, "data", paired_values),
        "model_seed": _stable_seed(base_seed, "model", paired_values),
        "n_samples": n_samples,
        "n_features": n_features,
        "selector": selector,
        "n_resamples": n_resamples,
        "n_estimators": n_estimators,
    }
    return PerformanceCell(cell_id=_cell_identity(values), **values)  # type: ignore[arg-type]


def _method_selector(task: Task, method: Method) -> str:
    if method == "citrees":
        return _default_selector(task)
    return "native"


def build_performance_grid(
    profile: Profile,
    *,
    base_seed: int = BASE_SEED,
) -> tuple[PerformanceCell, ...]:
    """Build the exact paired cell inventory for one profile."""
    settings = _settings(profile)
    if settings.baseline_n_samples in settings.sample_sizes:
        raise ValueError("sample-size alternatives must exclude the reference")
    if settings.baseline_n_features in settings.predictor_counts:
        raise ValueError("predictor-count alternatives must exclude the reference")
    if settings.baseline_n_resamples in settings.permutation_budgets:
        raise ValueError("permutation-budget alternatives must exclude the reference")
    if settings.baseline_n_estimators in settings.forest_sizes:
        raise ValueError("forest-size alternatives must exclude the reference")

    cells: list[PerformanceCell] = []

    def append(
        *,
        task: Task,
        family: ModelFamily,
        method: Method,
        axis: Axis,
        axis_value: str,
        repeat: int,
        n_samples: int = settings.baseline_n_samples,
        n_features: int = settings.baseline_n_features,
        selector: str | None = None,
        n_resamples: int | None = None,
        n_estimators: int | None = None,
    ) -> None:
        cells.append(
            _new_cell(
                profile=profile,
                task=task,
                model_family=family,
                method=method,
                axis=axis,
                axis_value=axis_value,
                repeat=repeat,
                n_samples=n_samples,
                n_features=n_features,
                selector=_method_selector(task, method) if selector is None else selector,
                n_resamples=(
                    0
                    if method == "sklearn"
                    else (settings.baseline_n_resamples if n_resamples is None else n_resamples)
                ),
                n_estimators=(
                    1
                    if family == "tree"
                    else (settings.baseline_n_estimators if n_estimators is None else n_estimators)
                ),
                base_seed=base_seed,
            )
        )

    for repeat in range(settings.repeats):
        for task in TASKS:
            for family in MODEL_FAMILIES:
                for method in METHODS:
                    append(
                        task=task,
                        family=family,
                        method=method,
                        axis="reference",
                        axis_value="reference",
                        repeat=repeat,
                    )

                    for n_samples in settings.sample_sizes:
                        append(
                            task=task,
                            family=family,
                            method=method,
                            axis="sample_size",
                            axis_value=str(n_samples),
                            repeat=repeat,
                            n_samples=n_samples,
                        )

                    for n_features in settings.predictor_counts:
                        append(
                            task=task,
                            family=family,
                            method=method,
                            axis="predictor_count",
                            axis_value=str(n_features),
                            repeat=repeat,
                            n_features=n_features,
                        )

                for n_resamples in settings.permutation_budgets:
                    for method in cast(tuple[Method, ...], ("citrees", "partykit")):
                        append(
                            task=task,
                            family=family,
                            method=method,
                            axis="permutation_budget",
                            axis_value=str(n_resamples),
                            repeat=repeat,
                            n_resamples=n_resamples,
                        )

                for selector in _selector_alternatives(task):
                    append(
                        task=task,
                        family=family,
                        method="citrees",
                        axis="selector",
                        axis_value=selector,
                        repeat=repeat,
                        selector=selector,
                    )

            for n_estimators in settings.forest_sizes:
                for method in METHODS:
                    append(
                        task=task,
                        family="forest",
                        method=method,
                        axis="forest_size",
                        axis_value=str(n_estimators),
                        repeat=repeat,
                        n_estimators=n_estimators,
                    )

    if len({cell.cell_id for cell in cells}) != len(cells):
        raise RuntimeError("performance grid contains duplicate cell identities")
    return tuple(cells)


def _array_sha256(X: np.ndarray, y: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (np.ascontiguousarray(X), np.ascontiguousarray(y)):
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _make_dataset(cell: PerformanceCell) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cell.data_seed)
    X = np.ascontiguousarray(
        rng.standard_normal((cell.n_samples, cell.n_features)),
        dtype=np.float64,
    )
    active = min(5, cell.n_features)
    coefficients = np.linspace(1.0, 0.25, active, dtype=np.float64)
    signal = X[:, :active] @ coefficients
    if cell.task == "classification":
        latent = signal + rng.standard_normal(cell.n_samples)
        y: np.ndarray = (latent >= np.median(latent)).astype(np.int64)
    else:
        y = signal + 0.5 * np.sin(X[:, 0]) + rng.standard_normal(cell.n_samples)
        y = np.asarray(y, dtype=np.float64)
    return X, np.ascontiguousarray(y)


def _fit_citrees(cell: PerformanceCell, X: np.ndarray, y: np.ndarray) -> int:
    from citrees import (
        ConditionalInferenceForestClassifier,
        ConditionalInferenceForestRegressor,
        ConditionalInferenceTreeClassifier,
        ConditionalInferenceTreeRegressor,
    )

    common: CitreesControl = {
        "selector": cell.selector,
        "alpha_selector": ALPHA,
        "adjust_alpha_selector": True,
        "n_resamples_selector": cell.n_resamples,
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
        "random_state": cell.model_seed,
        "verbose": 0,
    }
    if cell.model_family == "tree":
        model = (
            ConditionalInferenceTreeClassifier(**common)
            if cell.task == "classification"
            else ConditionalInferenceTreeRegressor(**common)
        )
        model.fit(X, y)
        return int(model.depth_) + 1

    forest_common: CitreesForestControl = {
        **common,
        "n_estimators": cell.n_estimators,
        "max_features": None,
        "bootstrap": True,
        "max_samples": None,
        "n_jobs": 1,
    }
    if cell.task == "classification":
        forest = ConditionalInferenceForestClassifier(
            sampling_method=None,
            **forest_common,
        )
    else:
        forest = ConditionalInferenceForestRegressor(**forest_common)
    forest.fit(X, y)
    return len(forest.estimators_)


def _fit_partykit(cell: PerformanceCell, X: np.ndarray, y: np.ndarray) -> int:
    from paper.benchmark.pipeline.r_methods import r_cforest_fit, r_ctree_fit

    common: PartykitControl = {
        "task": cell.task,
        "teststat": "quadratic",
        "testtype": ("Bonferroni", "MonteCarlo"),
        "mincriterion": 1.0 - ALPHA,
        "nresample": cell.n_resamples,
        "maxdepth": MAX_DEPTH,
        "minsplit": MIN_SAMPLES_SPLIT,
        "minbucket": MIN_SAMPLES_LEAF,
        "random_state": cell.model_seed,
    }
    if cell.model_family == "tree":
        r_ctree_fit(X, y, **common)
        return 1
    r_cforest_fit(
        X,
        y,
        ntree=cell.n_estimators,
        mtry="all",
        replace=True,
        fraction=1.0,
        cores=1,
        **common,
    )
    return cell.n_estimators


def _fit_sklearn(cell: PerformanceCell, X: np.ndarray, y: np.ndarray) -> int:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    common = {
        "max_depth": MAX_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "random_state": cell.model_seed,
    }
    if cell.model_family == "tree":
        model = (
            DecisionTreeClassifier(**common)
            if cell.task == "classification"
            else DecisionTreeRegressor(**common)
        )
        model.fit(X, y)
        return int(model.get_depth()) + 1

    forest_common = {
        **common,
        "n_estimators": cell.n_estimators,
        "max_features": None,
        "bootstrap": True,
        "n_jobs": -1,
    }
    forest = (
        RandomForestClassifier(**forest_common)
        if cell.task == "classification"
        else RandomForestRegressor(**forest_common)
    )
    forest.fit(X, y)
    return len(forest.estimators_)


def _fit_model(cell: PerformanceCell, X: np.ndarray, y: np.ndarray) -> int:
    if cell.method == "citrees":
        return _fit_citrees(cell, X, y)
    if cell.method == "partykit":
        return _fit_partykit(cell, X, y)
    return _fit_sklearn(cell, X, y)


def _peak_rss_bytes(
    raw_peak_rss: int | float | None = None,
    *,
    system: str | None = None,
) -> int:
    """Normalize ru_maxrss to bytes on the supported execution platforms."""
    raw = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if raw_peak_rss is None else raw_peak_rss
    )
    if raw < 0:
        raise ValueError("ru_maxrss must be nonnegative")
    observed_system = platform.system() if system is None else system
    if observed_system == "Darwin":
        return int(raw)
    if observed_system == "Linux":
        return int(raw * 1024)
    raise RuntimeError(f"unsupported peak-RSS platform: {observed_system}")


def _warmup_cell(cell: PerformanceCell) -> PerformanceCell:
    return replace(
        cell,
        n_samples=64,
        n_features=min(cell.n_features, 6),
        n_resamples=(0 if cell.method == "sklearn" else min(cell.n_resamples, 39)),
        n_estimators=(1 if cell.model_family == "tree" else min(cell.n_estimators, 3)),
    )


def run_worker_cell(cell: PerformanceCell) -> dict[str, object]:
    """Run one warmup and one measured fit in the current isolated process."""
    warmup = _warmup_cell(cell)
    warmup_X, warmup_y = _make_dataset(warmup)
    _fit_model(warmup, warmup_X, warmup_y)
    del warmup_X, warmup_y
    gc.collect()

    X, y = _make_dataset(cell)
    input_sha256 = _array_sha256(X, y)
    gc.collect()
    baseline_peak_rss = _peak_rss_bytes()
    started = time.perf_counter()
    fit_result_size = _fit_model(cell, X, y)
    elapsed = time.perf_counter() - started
    peak_rss = _peak_rss_bytes()
    if elapsed <= 0.0:
        raise RuntimeError("measured fit elapsed time must be positive")
    if fit_result_size < 1:
        raise RuntimeError("measured fit did not produce a valid model")

    return {
        **asdict(cell),
        "worker_pid": os.getpid(),
        "input_sha256": input_sha256,
        "elapsed_seconds": elapsed,
        "baseline_peak_rss_bytes": baseline_peak_rss,
        "peak_rss_bytes": peak_rss,
        "incremental_peak_rss_bytes": max(0, peak_rss - baseline_peak_rss),
        "fit_result_size": fit_result_size,
    }


def _cell_from_payload(payload: dict[str, object]) -> PerformanceCell:
    expected = set(PerformanceCell.__dataclass_fields__)
    if set(payload) != expected:
        missing = sorted(expected.difference(payload))
        extra = sorted(set(payload).difference(expected))
        raise ValueError(f"performance worker cell differs: missing={missing}, extra={extra}")
    cell = PerformanceCell(**payload)  # type: ignore[arg-type]
    identity_values = asdict(cell)
    observed_identity = cast(str, identity_values.pop("cell_id"))
    expected_identity = _cell_identity(identity_values)
    if observed_identity != expected_identity:
        raise ValueError("performance worker cell identity does not match its contents")
    return cell


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    return environment


def _run_cell_subprocess(cell: PerformanceCell) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory(prefix="citrees-jss-performance-") as temporary:
        temporary_path = Path(temporary)
        input_path = temporary_path / "cell.json"
        output_path = temporary_path / "result.json"
        input_path.write_text(
            json.dumps(asdict(cell), separators=(",", ":"), sort_keys=True) + "\n",
            encoding="ascii",
        )
        command = [
            sys.executable,
            "-m",
            "paper.jss.replication.performance",
            "--worker-input",
            str(input_path),
            "--worker-output",
            str(output_path),
        ]
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=_worker_environment(),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"performance cell {cell.cell_id} failed with exit code "
                f"{completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n"
                f"{completed.stderr}"
            )
        if not output_path.is_file():
            raise RuntimeError(f"performance cell {cell.cell_id} wrote no result")
        result = json.loads(output_path.read_text(encoding="ascii"))
    if not isinstance(result, dict):
        raise TypeError("performance worker result must be a JSON object")
    return result


def summarize_performance(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate isolated repeats and compare each cell with its method reference."""
    group_columns = [
        "profile",
        "task",
        "model_family",
        "method",
        "axis",
        "axis_value",
        "n_samples",
        "n_features",
        "selector",
        "n_resamples",
        "n_estimators",
    ]
    summary = (
        raw.groupby(group_columns, sort=False, dropna=False)
        .agg(
            n_repeats=("repeat", "nunique"),
            median_elapsed_seconds=("elapsed_seconds", "median"),
            lower_quartile_elapsed_seconds=("elapsed_seconds", lambda x: x.quantile(0.25)),
            upper_quartile_elapsed_seconds=("elapsed_seconds", lambda x: x.quantile(0.75)),
            median_peak_rss_bytes=("peak_rss_bytes", "median"),
            lower_quartile_peak_rss_bytes=("peak_rss_bytes", lambda x: x.quantile(0.25)),
            upper_quartile_peak_rss_bytes=("peak_rss_bytes", lambda x: x.quantile(0.75)),
            median_incremental_peak_rss_bytes=("incremental_peak_rss_bytes", "median"),
        )
        .reset_index()
    )
    references = summary[summary["axis"].eq("reference")][
        [
            "task",
            "model_family",
            "method",
            "median_elapsed_seconds",
            "median_peak_rss_bytes",
        ]
    ].rename(
        columns={
            "median_elapsed_seconds": "reference_elapsed_seconds",
            "median_peak_rss_bytes": "reference_peak_rss_bytes",
        }
    )
    if references.duplicated(["task", "model_family", "method"]).any():
        raise ValueError("performance summary contains duplicate method references")
    summary = summary.merge(
        references,
        on=["task", "model_family", "method"],
        how="left",
        validate="many_to_one",
    )
    if summary[["reference_elapsed_seconds", "reference_peak_rss_bytes"]].isna().any(axis=None):
        raise ValueError("performance summary is missing a method reference")
    summary["elapsed_ratio_to_reference"] = (
        summary["median_elapsed_seconds"] / summary["reference_elapsed_seconds"]
    )
    summary["peak_rss_ratio_to_reference"] = (
        summary["median_peak_rss_bytes"] / summary["reference_peak_rss_bytes"]
    )
    summary = summary.drop(columns=["reference_elapsed_seconds", "reference_peak_rss_bytes"])
    return summary.loc[:, PERFORMANCE_SUMMARY_SCHEMA]


def validate_performance_raw(
    raw: pd.DataFrame,
    cells: Sequence[PerformanceCell],
    *,
    profile: Profile,
    require_unique_worker_pids: bool = True,
) -> None:
    """Validate one exact performance-cell inventory."""
    if tuple(raw.columns) != PERFORMANCE_RAW_SCHEMA:
        raise ValueError(
            "performance raw schema differs: "
            f"expected={PERFORMANCE_RAW_SCHEMA}, observed={tuple(raw.columns)}"
        )
    if not cells:
        raise ValueError("performance raw validation requires at least one cell")
    expected_ids = {cell.cell_id for cell in cells}
    expected_specs = {cell.cell_id: asdict(cell) for cell in cells}
    observed_ids = set(raw["cell_id"].astype(str))
    if observed_ids != expected_ids or len(raw) != len(cells):
        missing = sorted(expected_ids.difference(observed_ids))
        extra = sorted(observed_ids.difference(expected_ids))
        raise ValueError(f"performance raw inventory differs: missing={missing}, extra={extra}")
    if raw["cell_id"].duplicated().any():
        raise ValueError("performance raw cells must be unique")
    cell_columns = tuple(PerformanceCell.__dataclass_fields__)
    for observed in raw.loc[:, cell_columns].to_dict(orient="records"):
        cell_id = str(observed["cell_id"])
        if observed != expected_specs[cell_id]:
            raise ValueError(f"performance cell specification differs for {cell_id}")
    if not raw["profile"].eq(profile).all():
        raise ValueError("performance raw profile differs from the requested profile")
    if not raw["input_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("performance raw input hashes are malformed")
    if (
        not np.isfinite(raw["elapsed_seconds"].to_numpy(dtype=np.float64)).all()
        or not raw["elapsed_seconds"].gt(0.0).all()
    ):
        raise ValueError("performance elapsed times must be finite and positive")
    if not raw["worker_pid"].ge(1).all():
        raise ValueError("performance worker PIDs must be positive")
    if require_unique_worker_pids and raw["worker_pid"].nunique() != len(raw):
        raise ValueError("performance cells must come from distinct worker processes")
    for column in (
        "baseline_peak_rss_bytes",
        "peak_rss_bytes",
        "incremental_peak_rss_bytes",
        "fit_result_size",
    ):
        if not raw[column].ge(0).all():
            raise ValueError(f"performance {column} must be nonnegative")
    if not raw["fit_result_size"].ge(1).all():
        raise ValueError("performance fits must produce nonempty results")
    if not raw["peak_rss_bytes"].ge(raw["baseline_peak_rss_bytes"]).all():
        raise ValueError("performance peak RSS must be at least its baseline")
    expected_increment = (raw["peak_rss_bytes"] - raw["baseline_peak_rss_bytes"]).clip(lower=0)
    if not raw["incremental_peak_rss_bytes"].eq(expected_increment).all():
        raise ValueError("performance incremental peak RSS is inconsistent")

    paired = raw.groupby(
        ["task", "n_samples", "n_features", "repeat"],
        sort=False,
    ).agg(
        data_seeds=("data_seed", "nunique"),
        model_seeds=("model_seed", "nunique"),
        input_hashes=("input_sha256", "nunique"),
    )
    if not paired.eq(1).all(axis=None):
        raise ValueError("performance methods do not share paired data and model seeds")


def validate_performance_results(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    profile: Profile,
    base_seed: int,
    require_unique_worker_pids: bool = True,
) -> None:
    """Validate the complete raw-cell and aggregate result contract."""
    if tuple(summary.columns) != PERFORMANCE_SUMMARY_SCHEMA:
        raise ValueError(
            "performance summary schema differs: "
            f"expected={PERFORMANCE_SUMMARY_SCHEMA}, observed={tuple(summary.columns)}"
        )
    validate_performance_raw(
        raw,
        build_performance_grid(profile, base_seed=base_seed),
        profile=profile,
        require_unique_worker_pids=require_unique_worker_pids,
    )
    numeric_summary = summary.select_dtypes(include=[np.number])
    if not np.isfinite(numeric_summary.to_numpy(dtype=np.float64)).all():
        raise ValueError("performance summary contains non-finite numeric values")
    if not summary["n_repeats"].eq(_settings(profile).repeats).all():
        raise ValueError("performance summary repeat counts are incomplete")
    expected_summary = summarize_performance(raw)
    try:
        pd.testing.assert_frame_equal(
            summary.reset_index(drop=True),
            expected_summary.reset_index(drop=True),
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError("performance summary differs from the raw-cell aggregates") from error


def run_performance(
    profile: Profile,
    *,
    base_seed: int = BASE_SEED,
    runner: CellRunner = _run_cell_subprocess,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every profile cell in a separate process and return validated tables."""
    rows = [
        runner(cell)
        for cell in build_performance_grid(
            profile,
            base_seed=base_seed,
        )
    ]
    raw = pd.DataFrame(rows).loc[:, PERFORMANCE_RAW_SCHEMA]
    summary = summarize_performance(raw)
    validate_performance_results(
        raw,
        summary,
        profile=profile,
        base_seed=base_seed,
    )
    return raw, summary


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _r_environment() -> dict[str, str]:
    from paper.benchmark.pipeline.r_methods import get_r_runtime_versions

    return get_r_runtime_versions()


def performance_source_files() -> tuple[Path, ...]:
    """Return the complete source inventory for performance measurements."""
    repo_root = Path(__file__).resolve().parents[3]
    return (
        Path(__file__).resolve(),
        repo_root / "paper" / "benchmark" / "pipeline" / "r_methods.py",
        repo_root / "citrees" / "_forest.py",
        repo_root / "citrees" / "_permutation.py",
        repo_root / "citrees" / "_selector.py",
        repo_root / "citrees" / "_sequential.py",
        repo_root / "citrees" / "_splitter.py",
        repo_root / "citrees" / "_threshold_method.py",
        repo_root / "citrees" / "_tree.py",
        repo_root / "citrees" / "_types.py",
        repo_root / "pyproject.toml",
        repo_root / "uv.lock",
    )


def write_results(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    profile: Profile,
    base_seed: int,
    elapsed_seconds: float,
    require_unique_worker_pids: bool = True,
) -> None:
    """Write performance tables and a machine-readable execution receipt."""
    validate_performance_results(
        raw,
        summary,
        profile=profile,
        base_seed=base_seed,
        require_unique_worker_pids=require_unique_worker_pids,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "performance_raw.parquet"
    summary_path = output_dir / "performance_summary.parquet"
    summary_csv_path = output_dir / "performance_summary.csv"
    raw.to_parquet(raw_path, index=False)
    summary.to_parquet(summary_path, index=False)
    summary.to_csv(summary_csv_path, index=False)

    written_raw = pd.read_parquet(raw_path)
    written_summary = pd.read_parquet(summary_path)
    validate_performance_results(
        written_raw,
        written_summary,
        profile=profile,
        base_seed=base_seed,
        require_unique_worker_pids=require_unique_worker_pids,
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(summary_csv_path).convert_dtypes(),
        summary.convert_dtypes(),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

    repo_root = Path(__file__).resolve().parents[3]
    source_files = performance_source_files()
    versions = {
        package: importlib.metadata.version(package)
        for package in ("citrees", "numpy", "pandas", "scikit-learn", "scipy")
    }
    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
        versions["rpy2"] = importlib.metadata.version("rpy2")
    artifacts = (raw_path, summary_path, summary_csv_path)
    receipt = {
        "analysis": "performance",
        "profile": profile,
        "base_seed": base_seed,
        "settings": asdict(_settings(profile)),
        "controls": {
            "alpha": ALPHA,
            "max_depth": MAX_DEPTH,
            "min_samples_split": MIN_SAMPLES_SPLIT,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "candidate_features": "all",
            "bootstrap": True,
            "timed_operation": "model_construction_and_fit",
            "warmup": "same_method_tiny_fit_before_measurement",
            "memory": "process_peak_rss_with_pre_fit_high_water_subtracted",
            "process_isolation": "one_fresh_process_per_measured_cell",
            "thread_environment": THREAD_ENVIRONMENT,
        },
        "created_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "r_environment": _r_environment(),
        "source_sha256": {str(path.relative_to(repo_root)): _sha256(path) for path in source_files},
        "versions": versions,
        "tables": {
            "performance_raw": {
                "rows": len(raw),
                "columns": list(raw.columns),
            },
            "performance_summary": {
                "rows": len(summary),
                "columns": list(summary.columns),
            },
        },
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifacts
        },
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("smoke", "quick", "full"),
        default="quick",
        help="Replication workload.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Parquet, CSV, and receipt outputs.",
    )
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Base random seed.")
    parser.add_argument("--worker-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _run_worker(input_path: Path, output_path: Path) -> None:
    payload = json.loads(input_path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise TypeError("performance worker input must be a JSON object")
    cell = _cell_from_payload(payload)
    result = run_worker_cell(cell)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(output_path)


def main() -> None:
    args = _parse_args()
    if (args.worker_input is None) != (args.worker_output is None):
        raise ValueError("--worker-input and --worker-output must be supplied together")
    if args.worker_input is not None:
        _run_worker(args.worker_input, args.worker_output)
        return
    if args.profile == "full" and _git_dirty():
        raise RuntimeError("The full performance profile requires a clean source tree")
    started = time.perf_counter()
    raw, summary = run_performance(args.profile, base_seed=args.seed)
    elapsed = time.perf_counter() - started
    write_results(
        raw,
        summary,
        args.output_dir,
        profile=args.profile,
        base_seed=args.seed,
        elapsed_seconds=elapsed,
    )
    print(
        f"Wrote {len(raw)} isolated performance cells to {args.output_dir} "
        f"in {elapsed:.2f} seconds."
    )


if __name__ == "__main__":
    main()
