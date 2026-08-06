"""Stage 1: Feature selection using various methods.

This module implements the feature selection stage of the experiment pipeline.
It computes feature rankings using different methods and saves them to S3.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import socket
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.preprocessing import StandardScaler

from paper.benchmark.adapters.data import (
    get_cv_splitter,
    get_dataset_metadata,
    load_dataset,
)
from paper.benchmark.adapters.store import Store
from paper.benchmark.config.constants import (
    N_SPLITS,
    PIPELINE_ARTIFACT_VERSION,
    STAGE1_SELECTION_TIMEOUT_SECONDS,
)
from paper.benchmark.pipeline.types import ExperimentConfig, Result, TaskType
from paper.benchmark.pipeline.validation import (
    validate_artifact_provenance,
    validate_ranking_artifact,
)
from paper.benchmark.utils.env import (
    get_available_cpu_ids,
    get_benchmark_scope,
    get_container_image,
    get_git_sha,
    get_hardware_metadata,
    get_library_versions,
    partition_cpu_ids,
    utc_now_iso,
)

_R_METHODS = frozenset({"r_ctree", "r_cforest"})
_PROCESS_PARALLEL_FOLD_METHODS = frozenset({"cit", "ptest_mc", "ptest_pc", "ptest_dc", "ptest_rdc"})
_R_PROCESS_POLL_INTERVAL_SECONDS = 0.05
_R_PROCESS_START_TIMEOUT_SECONDS = 30.0
_R_PROCESS_EXIT_GRACE_SECONDS = 5.0


@dataclass
class _RFoldProcess:
    """Parent-owned process and result channel for one R fold."""

    fold_idx: int
    process: Any
    receiver: Connection
    launch_event: Any | None = None
    process_group_id: int | None = None


# =============================================================================
# Selection methods
# =============================================================================


def _encode_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_encoded, classes) with labels mapped to 0..K-1."""
    y_arr = np.asarray(y).ravel()
    classes, y_encoded = np.unique(y_arr, return_inverse=True)
    return y_encoded.astype(np.int64), classes


def filter_selector(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    task: TaskType,
    random_state: int,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Compute feature ranking using filter methods."""
    from citrees._selector import ClassifierSelectors, RegressorSelectors

    n_features = X.shape[1]
    selectors = ClassifierSelectors if task == "classification" else RegressorSelectors
    selector_fn: Callable[..., float] = selectors[method]
    selector_kwargs = (
        {"n_projections": (params or {}).get("rdc_n_projections", 10)} if method == "rdc" else {}
    )
    scores = np.zeros(n_features)
    rng = np.random.default_rng(random_state)

    if task == "classification":
        y_enc, classes = _encode_labels(y)
        n_classes = len(classes)
        for j in range(n_features):
            scores[j] = selector_fn(
                X[:, j],
                y_enc,
                n_classes,
                random_state=rng.integers(0, 2**31),
                **selector_kwargs,
            )
    else:
        use_abs = method == "pc"
        for j in range(n_features):
            score = selector_fn(
                X[:, j],
                y,
                standardize=True,
                random_state=rng.integers(0, 2**31),
                **selector_kwargs,
            )
            scores[j] = abs(score) if use_abs else score

    return np.argsort(scores)[::-1]


def _resolve_n_resamples(n_resamples: int | str, alpha: float) -> int:
    """Resolve n_resamples string values to integers.

    Uses the same formulas as citrees._tree:
    - "minimum": ceil(1 / alpha)
    - "maximum": ceil(1 / (4 * alpha^2))
    - "auto": max(lower_limit, ceil(z^2 * (1-alpha) / alpha))

    Raises
    ------
    ValueError
        If n_resamples is a string other than "minimum", "maximum", or "auto".
    """
    from math import ceil

    from scipy.stats import norm

    if isinstance(n_resamples, int):
        return n_resamples

    lower_limit = ceil(1 / alpha)
    if n_resamples == "minimum":
        return lower_limit
    elif n_resamples == "maximum":
        return ceil(1 / (4 * alpha * alpha))
    elif n_resamples == "auto":
        z = norm.ppf(1 - alpha)
        upper_limit = ceil(z * z * (1 - alpha) / alpha)
        return max(lower_limit, upper_limit)
    else:
        raise ValueError(
            f"n_resamples must be int or one of 'minimum', 'maximum', 'auto', got: {n_resamples!r}"
        )


def permutation_selector(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    task: TaskType,
    random_state: int,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Compute feature ranking using permutation tests.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target array.
    method : str
        Method name (e.g., "ptest_mc", "ptest_pc").
    task : str
        "classification" or "regression".
    random_state : int
        Random seed.
    params : dict, optional
        Parameters for the permutation test:
        - alpha: significance level (default 0.05)
        - n_resamples: number of permutations (default 1000, or "minimum"/"maximum"/"auto")
        - early_stopping: "adaptive", "simple", or None (default None)
    """
    from citrees._selector import (
        ClassifierSelectors,
        ClassifierSelectorTests,
        RegressorSelectors,
        RegressorSelectorTests,
    )

    params = params or {}
    alpha = params.get("alpha", 0.05)
    n_resamples_raw = params.get("n_resamples", 1000)
    n_resamples = _resolve_n_resamples(n_resamples_raw, alpha)
    early_stopping = params.get("early_stopping")
    rdc_kwargs = (
        {"n_projections": params.get("rdc_n_projections", 10)} if method == "ptest_rdc" else {}
    )

    base_method = method.replace("ptest_", "")
    n_features = X.shape[1]
    selectors = ClassifierSelectors if task == "classification" else RegressorSelectors
    selector_tests = ClassifierSelectorTests if task == "classification" else RegressorSelectorTests
    selector_fn: Callable[..., float] = selectors[base_method]
    test_fn: Callable[..., float] = selector_tests[base_method]

    scores = np.zeros(n_features)
    pvalues = np.ones(n_features)
    rng = np.random.default_rng(random_state)

    if task == "classification":
        y_enc, classes = _encode_labels(y)
        n_classes = len(classes)
        for j in range(n_features):
            rs = rng.integers(0, 2**31)
            scores[j] = selector_fn(
                X[:, j],
                y_enc,
                n_classes,
                random_state=rs,
                **rdc_kwargs,
            )
            pvalues[j] = test_fn(
                x=X[:, j],
                y=y_enc,
                n_classes=n_classes,
                alpha=alpha,
                n_resamples=n_resamples,
                early_stopping=early_stopping,
                random_state=rs,
                **rdc_kwargs,
            )
    else:
        use_abs = base_method == "pc"
        for j in range(n_features):
            rs = rng.integers(0, 2**31)
            score = selector_fn(
                X[:, j],
                y,
                standardize=True,
                random_state=rs,
                **rdc_kwargs,
            )
            scores[j] = abs(score) if use_abs else score
            pvalues[j] = test_fn(
                x=X[:, j],
                y=y,
                standardize=True,
                alpha=alpha,
                n_resamples=n_resamples,
                early_stopping=early_stopping,
                random_state=rs,
                **rdc_kwargs,
            )

    return np.lexsort((-scores, pvalues))


def embedding_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    method: str,
    task: TaskType,
    random_state: int,
    params: dict[str, Any] | None = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute feature ranking using tree-based embedding methods."""
    from paper.benchmark.pipeline.selectors import get_embedding_model

    params = params or {}
    model = get_embedding_model(method, task, random_state, n_jobs=n_jobs, params=params)
    model.fit(X_train, y_train)
    return np.argsort(model.feature_importances_)[::-1]


def wrapper_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str,
    task: TaskType,
    random_state: int,
    n_jobs: int = 1,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Compute feature ranking using wrapper methods.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target array.
    method : str
        Method name: "boruta", "pi", "cpi", "mrmr", "rfe".
    task : str
        "classification" or "regression".
    random_state : int
        Random seed.
    n_jobs : int
        Number of parallel jobs.
    params : dict, optional
        Method-specific parameters. See individual selector functions.
    """
    from paper.benchmark.pipeline.selectors import (
        boruta_selector,
        cpi_selector,
        mrmr_selector,
        pi_selector,
        rfe_selector,
    )

    params = params or {}

    if method == "boruta":
        return boruta_selector(X_train, y_train, task, random_state, n_jobs=n_jobs, params=params)
    if method == "pi":
        return pi_selector(X_train, y_train, task, random_state, n_jobs=n_jobs, params=params)
    if method == "cpi":
        return cpi_selector(X_train, y_train, task, random_state, n_jobs=n_jobs, params=params)
    if method == "mrmr":
        return mrmr_selector(X_train, y_train, task)
    if method == "rfe":
        return rfe_selector(
            X_train,
            y_train,
            task,
            random_state,
            n_jobs=n_jobs,
            params=params,
        )

    raise ValueError(f"Unknown wrapper method: {method}")


def _validate_fold_ranking(
    ranking: Any,
    *,
    fold_idx: int,
    method: str,
    n_features: int,
) -> np.ndarray:
    """Return one validated complete feature permutation."""
    values = np.asarray(ranking)
    if values.ndim != 1 or values.shape[0] != n_features:
        raise ValueError(
            f"{method} returned ranking shape {values.shape} for "
            f"{n_features} features in fold {fold_idx}"
        )
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{method} returned a non-integer ranking in fold {fold_idx}")
    if not np.array_equal(np.sort(values), np.arange(n_features)):
        raise ValueError(
            f"{method} did not return a permutation of all feature indices in fold {fold_idx}"
        )
    return values


def _run_selection_fold(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    task: TaskType,
    seed: int,
    params: dict[str, Any],
    n_jobs: int,
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    r_cforest_cores: int | None = None,
) -> dict[str, Any]:
    """Run and validate one fold without sharing estimator state."""
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    rs = seed * 1000 + fold_idx

    if method in ["mc", "mi", "rdc", "pc", "dc"]:
        ranking = filter_selector(X_train, y_train, method, task, rs, params=params)
    elif method.startswith("ptest_"):
        ranking = permutation_selector(X_train, y_train, method, task, rs, params=params)
    elif method in ["dt", "rt", "rf", "et", "xgb", "lgbm", "cat", "cit", "cif"]:
        ranking = embedding_selector(
            X_train,
            y_train,
            X_test,
            y_test,
            method,
            task,
            rs,
            params=params,
            n_jobs=n_jobs,
        )
    elif method in ["boruta", "pi", "cpi", "mrmr", "rfe"]:
        ranking = wrapper_selector(X_train, y_train, method, task, rs, n_jobs=n_jobs, params=params)
    elif method == "r_ctree":
        from paper.benchmark.pipeline.r_methods import r_ctree_ranking

        ranking = r_ctree_ranking(
            X_train,
            y_train,
            task=task,
            random_state=rs,
            **params,
        )
    elif method == "r_cforest":
        from paper.benchmark.pipeline.r_methods import r_cforest_ranking

        r_params = dict(params)
        if r_cforest_cores is not None:
            if "cores" in r_params:
                raise ValueError("r_cforest CPU count belongs to the runtime, not method params")
            r_params["cores"] = r_cforest_cores
        ranking = r_cforest_ranking(
            X_train,
            y_train,
            task=task,
            random_state=rs,
            **r_params,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    values = _validate_fold_ranking(
        ranking,
        fold_idx=fold_idx,
        method=method,
        n_features=X_train.shape[1],
    )
    return {
        "fold_idx": fold_idx,
        "fold_random_state": rs,
        "feature_ranking": values.tolist(),
        "fold_cpu_affinity": list(get_available_cpu_ids()),
    }


def _pin_current_process(cpu_ids: tuple[int, ...]) -> tuple[int, ...]:
    """Pin one spawned fold process and return its observed CPU affinity."""
    set_affinity = getattr(os, "sched_setaffinity", None)
    get_affinity = getattr(os, "sched_getaffinity", None)
    if set_affinity is None or get_affinity is None:
        raise RuntimeError("parallel R selection requires verifiable CPU affinity")
    expected = set(cpu_ids)
    set_affinity(0, expected)
    observed = set(get_affinity(0))
    if observed != expected:
        raise RuntimeError(
            f"fold process CPU affinity differs: expected={sorted(expected)}, "
            f"observed={sorted(observed)}"
        )
    return tuple(sorted(observed))


def _start_r_fold_process_group() -> int:
    """Create an isolated process group before embedded R can fork workers."""
    setsid = getattr(os, "setsid", None)
    getpgrp = getattr(os, "getpgrp", None)
    if setsid is None or getpgrp is None:
        raise RuntimeError("parallel R selection requires POSIX process groups")
    setsid()
    process_group_id = int(getpgrp())
    if process_group_id != os.getpid():
        raise RuntimeError(
            "R fold process group differs from its process ID: "
            f"pid={os.getpid()}, process_group_id={process_group_id}"
        )
    return process_group_id


def _run_r_selection_fold_process(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    task: TaskType,
    seed: int,
    params: dict[str, Any],
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cpu_ids: tuple[int, ...],
) -> dict[str, Any]:
    """Run one R fold in a fresh process bound to its CPU partition."""
    observed_cpu_ids = _pin_current_process(cpu_ids)
    row = _run_selection_fold(
        X,
        y,
        method,
        task,
        seed,
        params,
        1,
        fold_idx,
        train_idx,
        test_idx,
        r_cforest_cores=len(cpu_ids) if method == "r_cforest" else None,
    )
    row["fold_cpu_affinity"] = list(observed_cpu_ids)
    return row


def _run_r_selection_fold_entry(
    sender: Connection,
    launch_event: Any,
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    task: TaskType,
    seed: int,
    params: dict[str, Any],
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cpu_ids: tuple[int, ...],
) -> None:
    """Send one fold result or a serializable child failure to the parent."""
    try:
        process_group_id = _start_r_fold_process_group()
        sender.send(
            {
                "status": "ready",
                "process_group_id": process_group_id,
            }
        )
        if not launch_event.wait(_R_PROCESS_START_TIMEOUT_SECONDS):
            raise TimeoutError("parent did not authorize R fold execution")
        row = _run_r_selection_fold_process(
            X,
            y,
            method,
            task,
            seed,
            params,
            fold_idx,
            train_idx,
            test_idx,
            cpu_ids,
        )
        sender.send(
            {
                "status": "ok",
                "row": row,
            }
        )
    except BaseException as exc:
        sender.send(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        sender.close()


def _receive_r_fold_process_ready(
    handle: _RFoldProcess,
    *,
    deadline: float | None = None,
) -> int:
    """Require process-group readiness before the fold can launch R workers."""
    readiness_deadline = (
        time.monotonic() + _R_PROCESS_START_TIMEOUT_SECONDS if deadline is None else deadline
    )
    while True:
        remaining = readiness_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"R fold {handle.fold_idx} did not establish its process group "
                f"within {_R_PROCESS_START_TIMEOUT_SECONDS:g} seconds"
            )
        ready = wait(
            (handle.receiver,),
            timeout=min(remaining, _R_PROCESS_POLL_INTERVAL_SECONDS),
        )
        if handle.receiver in ready:
            break
        if not handle.process.is_alive() and not handle.receiver.poll():
            raise RuntimeError(
                f"R fold {handle.fold_idx} exited before process-group readiness "
                f"(exitcode={handle.process.exitcode})"
            )
    try:
        message = handle.receiver.recv()
    except EOFError as exc:
        handle.process.join(timeout=0)
        raise RuntimeError(
            f"R fold {handle.fold_idx} exited before process-group readiness "
            f"(exitcode={handle.process.exitcode})"
        ) from exc
    if not isinstance(message, dict):
        raise RuntimeError(f"R fold {handle.fold_idx} returned an invalid readiness message")
    if message.get("status") == "error":
        raise RuntimeError(
            f"R fold {handle.fold_idx} failed before readiness with "
            f"{message.get('error_type')}: {message.get('error')}\n"
            f"{message.get('traceback')}"
        )
    process_group_id = message.get("process_group_id")
    if (
        message.get("status") != "ready"
        or type(process_group_id) is not int
        or process_group_id <= 0
        or process_group_id != handle.process.pid
    ):
        raise RuntimeError(f"R fold {handle.fold_idx} returned an invalid readiness message")
    return process_group_id


def _r_process_group_exists(process_group_id: int) -> bool:
    """Return whether any process remains in one owned R process group."""
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_r_fold_process(handle: _RFoldProcess, signal_number: signal.Signals) -> None:
    """Signal one whole R fold process group, or its pre-readiness process."""
    if handle.process_group_id is not None:
        try:
            os.killpg(handle.process_group_id, signal_number)
        except ProcessLookupError:
            return
        return
    if not handle.process.is_alive():
        return
    if signal_number == signal.SIGTERM:
        handle.process.terminate()
    elif signal_number == signal.SIGKILL:
        handle.process.kill()
    else:
        raise ValueError(f"unsupported R fold cleanup signal: {signal_number}")


def _r_fold_process_remains(handle: _RFoldProcess) -> bool:
    """Return whether the direct fold or any process-group descendant remains."""
    return handle.process.is_alive() or (
        handle.process_group_id is not None and _r_process_group_exists(handle.process_group_id)
    )


def _wait_for_r_fold_process_exit(
    handles: list[_RFoldProcess],
    *,
    deadline: float,
) -> None:
    """Wait until every direct process and owned process group has exited."""
    while True:
        for handle in handles:
            handle.process.join(timeout=0)
        if not any(_r_fold_process_remains(handle) for handle in handles):
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.01, remaining))


def _terminate_r_fold_processes(handles: list[_RFoldProcess]) -> None:
    """Bound teardown of every fold process, descendant, and result channel."""
    try:
        for handle in handles:
            _signal_r_fold_process(handle, signal.SIGTERM)
        _wait_for_r_fold_process_exit(
            handles,
            deadline=time.monotonic() + _R_PROCESS_EXIT_GRACE_SECONDS,
        )
        for handle in handles:
            if _r_fold_process_remains(handle):
                _signal_r_fold_process(handle, signal.SIGKILL)
        _wait_for_r_fold_process_exit(
            handles,
            deadline=time.monotonic() + _R_PROCESS_EXIT_GRACE_SECONDS,
        )
        survivors = [handle.fold_idx for handle in handles if _r_fold_process_remains(handle)]
        if survivors:
            raise RuntimeError(f"R fold process groups survived cleanup: {survivors}")
    finally:
        for handle in handles:
            handle.receiver.close()


def _collect_r_fold_processes(
    handles: list[_RFoldProcess],
    *,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    """Collect all child results fail-fast within one absolute deadline."""
    deadline = time.monotonic() + timeout_seconds
    pending = {handle.receiver: handle for handle in handles}
    results: list[dict[str, Any]] = []
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pending_folds = sorted(handle.fold_idx for handle in pending.values())
            raise TimeoutError(f"parallel R selection timed out with pending folds {pending_folds}")
        ready = wait(
            tuple(pending),
            timeout=min(remaining, _R_PROCESS_POLL_INTERVAL_SECONDS),
        )
        for receiver in tuple(pending):
            if receiver not in ready:
                continue
            handle = pending.pop(receiver)
            try:
                message = receiver.recv()
            except EOFError as exc:
                handle.process.join(timeout=0)
                raise RuntimeError(
                    f"R fold {handle.fold_idx} exited without a result "
                    f"(exitcode={handle.process.exitcode})"
                ) from exc
            if not isinstance(message, dict) or message.get("status") not in {"ok", "error"}:
                raise RuntimeError(f"R fold {handle.fold_idx} returned an invalid process message")
            if message["status"] == "error":
                raise RuntimeError(
                    f"R fold {handle.fold_idx} failed with "
                    f"{message.get('error_type')}: {message.get('error')}\n"
                    f"{message.get('traceback')}"
                )
            row = message.get("row")
            if not isinstance(row, dict):
                raise RuntimeError(f"R fold {handle.fold_idx} returned an invalid result")
            results.append(row)
        for handle in tuple(pending.values()):
            if handle.process.is_alive() or handle.receiver.poll():
                continue
            raise RuntimeError(
                f"R fold {handle.fold_idx} exited without a result "
                f"(exitcode={handle.process.exitcode})"
            )

    exit_deadline = time.monotonic() + _R_PROCESS_EXIT_GRACE_SECONDS
    for handle in handles:
        handle.process.join(timeout=max(0.0, exit_deadline - time.monotonic()))
        if handle.process.is_alive():
            raise TimeoutError(f"R fold {handle.fold_idx} did not exit after returning its result")
        if handle.process.exitcode != 0:
            raise RuntimeError(
                f"R fold {handle.fold_idx} exited with code {handle.process.exitcode}"
            )
    return sorted(results, key=lambda row: int(row["fold_idx"]))


def run_r_selection_parallel(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    task: TaskType,
    seed: int,
    params: dict[str, Any] | None = None,
    timeout_seconds: float = STAGE1_SELECTION_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Run five R folds concurrently in independent embedded-R processes."""
    if method not in _R_METHODS:
        raise ValueError(f"parallel R selection does not support method {method!r}")
    normalized_params = dict(params or {})
    if "cores" in normalized_params:
        raise ValueError("r_cforest CPU count belongs to the runtime, not method params")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not np.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("parallel R selection timeout must be positive and finite")
    if sys.platform.startswith("linux") and os.getpid() == 1:
        raise RuntimeError("parallel R selection requires a container init process")

    deadline = time.monotonic() + timeout_seconds
    splits = tuple(get_cv_splitter(task, N_SPLITS, seed).split(X, y))
    if len(splits) != N_SPLITS:
        raise RuntimeError(f"expected {N_SPLITS} CV folds, observed {len(splits)}")
    cpu_partitions = partition_cpu_ids(get_available_cpu_ids(), N_SPLITS)
    context = multiprocessing.get_context("spawn")
    handles: list[_RFoldProcess] = []
    try:
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            receiver, sender = context.Pipe(duplex=False)
            launch_event = context.Event()
            process = context.Process(
                target=_run_r_selection_fold_entry,
                args=(
                    sender,
                    launch_event,
                    X,
                    y,
                    method,
                    task,
                    seed,
                    normalized_params,
                    fold_idx,
                    train_idx,
                    test_idx,
                    cpu_partitions[fold_idx],
                ),
                name=f"citrees-r-fold-{fold_idx}",
            )
            try:
                process.start()
            except BaseException:
                sender.close()
                receiver.close()
                raise
            sender.close()
            handle = _RFoldProcess(
                fold_idx=fold_idx,
                process=process,
                receiver=receiver,
                launch_event=launch_event,
            )
            handles.append(handle)
            readiness_deadline = min(
                deadline,
                time.monotonic() + _R_PROCESS_START_TIMEOUT_SECONDS,
            )
            handle.process_group_id = _receive_r_fold_process_ready(
                handle,
                deadline=readiness_deadline,
            )
            if handle.launch_event is None:
                raise RuntimeError(f"R fold {handle.fold_idx} has no launch event")
            handle.launch_event.set()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"parallel R selection timed out with pending folds {list(range(N_SPLITS))}"
            )
        return _collect_r_fold_processes(
            handles,
            timeout_seconds=remaining,
        )
    finally:
        _terminate_r_fold_processes(handles)


def run_selection(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    task: TaskType,
    seed: int,
    params: dict[str, Any] | None = None,
    n_jobs: int = 1,
    timeout_seconds: float = STAGE1_SELECTION_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Run feature selection across cross-validation folds."""
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not np.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("selection timeout must be positive and finite")
    normalized_params = dict(params or {})
    splits = tuple(get_cv_splitter(task, N_SPLITS, seed).split(X, y))
    if len(splits) != N_SPLITS:
        raise RuntimeError(f"expected {N_SPLITS} CV folds, observed {len(splits)}")

    if method in _PROCESS_PARALLEL_FOLD_METHODS:
        fold_workers = min(N_SPLITS, len(get_available_cpu_ids()))
        if fold_workers > 1:
            fold_calls = (
                delayed(_run_selection_fold)(
                    X,
                    y,
                    method,
                    task,
                    seed,
                    normalized_params,
                    1,
                    fold_idx,
                    train_idx,
                    test_idx,
                )
                for fold_idx, (train_idx, test_idx) in enumerate(splits)
            )
            with parallel_config(
                backend="loky",
                n_jobs=fold_workers,
                inner_max_num_threads=1,
            ):
                with Parallel(timeout=timeout_seconds) as parallel:
                    results: list[dict[str, Any]] = parallel(fold_calls)
                return _require_ordered_selection_folds(results)

    inner_jobs = 1 if method in _PROCESS_PARALLEL_FOLD_METHODS else n_jobs
    results = [
        _run_selection_fold(
            X,
            y,
            method,
            task,
            seed,
            normalized_params,
            inner_jobs,
            fold_idx,
            train_idx,
            test_idx,
        )
        for fold_idx, (train_idx, test_idx) in enumerate(splits)
    ]
    return _require_ordered_selection_folds(results)


def _require_ordered_selection_folds(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Require one result for each fold in deterministic order."""
    observed: list[int] = []
    for row in rows:
        fold_idx = row.get("fold_idx")
        if isinstance(fold_idx, (bool, np.bool_)) or not isinstance(fold_idx, (int, np.integer)):
            raise RuntimeError("selection fold result has a non-integer fold_idx")
        observed.append(int(fold_idx))
    expected = list(range(N_SPLITS))
    if observed != expected:
        raise RuntimeError(
            f"selection fold order differs: expected={expected}, observed={observed}"
        )
    return rows


def _run_selection(cfg: ExperimentConfig, store: Store) -> Result:
    """Run feature selection for a single configuration."""
    method = cfg.method.name
    params = cfg.method.params_dict
    dataset = cfg.dataset
    seed = cfg.seed
    task = cfg.task

    hostname = socket.gethostname()
    git_sha = get_git_sha()

    try:
        benchmark_scope = get_benchmark_scope()
        container_image = get_container_image()
        expected_provenance = {
            **benchmark_scope,
            "container_image": container_image,
            "git_sha": git_sha,
        }
        if store.exists("rankings", cfg):
            existing = store.load("rankings", cfg)
            validate_ranking_artifact(existing, cfg)
            validate_artifact_provenance(existing, expected_provenance)
            return Result(
                config=cfg,
                status="skipped",
                data=existing,
                hostname=hostname,
            )

        # Load dataset
        X, y = load_dataset(dataset, task, identity=cfg.dataset_identity)
        n_samples, n_features = int(X.shape[0]), int(X.shape[1])
        n_jobs = -1

        # Get dataset metadata
        dataset_meta = get_dataset_metadata(
            dataset,
            task,
            identity=cfg.dataset_identity,
        )

        # Run selection
        created_at_utc = utc_now_iso()
        tic = time.perf_counter()
        if method in _R_METHODS:
            fold_results = run_r_selection_parallel(
                X,
                y,
                method,
                task,
                seed,
                params=params,
            )
        else:
            fold_results = run_selection(
                X,
                y,
                method,
                task,
                seed,
                params=params,
                n_jobs=n_jobs,
            )
        elapsed = time.perf_counter() - tic
        library_versions = get_library_versions()
        if method in {"r_ctree", "r_cforest"}:
            from paper.benchmark.pipeline.r_methods import get_r_runtime_versions

            library_versions.update(get_r_runtime_versions())
        hardware = get_hardware_metadata()
        cpu_affinity = hardware["cpu_affinity"]
        if not isinstance(cpu_affinity, list):
            raise TypeError("hardware cpu_affinity must be a list")

        # Enrich results with metadata
        for row in fold_results:
            fold_cpu_affinity = row.get("fold_cpu_affinity")
            if not isinstance(fold_cpu_affinity, list):
                raise RuntimeError("selection fold did not report its CPU affinity")
            row.update(
                {
                    "dataset": dataset,
                    "task": task,
                    "seed": seed,
                    "method_id": cfg.method.label,
                    "method": cfg.method.label,
                    "method_base": method,
                    "method_params_json": json.dumps(
                        params,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "artifact_version": PIPELINE_ARTIFACT_VERSION,
                    "dataset_sha256": cfg.dataset_identity.sha256,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "elapsed_seconds": float(elapsed),
                    "created_at_utc": created_at_utc,
                    "git_sha": git_sha,
                    "container_image": container_image,
                    "library_versions": library_versions,
                    "hardware": hardware,
                    **benchmark_scope,
                    "dataset_source": dataset_meta.get("dataset_source"),
                    "dataset_type": dataset_meta.get("dataset_type"),
                    "dataset_family": dataset_meta.get("dataset_family"),
                    "n_informative": dataset_meta.get("n_informative"),
                }
            )

        # Save to store
        df = pd.DataFrame(fold_results)
        validate_ranking_artifact(df, cfg)
        validate_artifact_provenance(df, expected_provenance)
        try:
            s3_path = store.save("rankings", cfg, df)
        except FileExistsError:
            existing = store.load("rankings", cfg)
            validate_ranking_artifact(existing, cfg)
            validate_artifact_provenance(existing, expected_provenance)
            return Result(
                config=cfg,
                status="skipped",
                elapsed_seconds=elapsed,
                data=existing,
                hostname=hostname,
            )

        return Result(
            config=cfg,
            status="done",
            elapsed_seconds=elapsed,
            data=df,
            s3_path=s3_path,
            hostname=hostname,
        )

    except Exception as e:
        return Result(
            config=cfg,
            status="failed",
            error=str(e),
            error_type=type(e).__name__,
            traceback=traceback.format_exc(),
            hostname=hostname,
        )
