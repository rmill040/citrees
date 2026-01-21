"""Ray-based distributed downstream evaluation."""

from __future__ import annotations

import os
import socket
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
import ray
from loguru import logger
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from paper.scripts.experiments._common import (
    download_parquet_from_s3,
    get_dataset_metadata,
    get_datasets,
    get_git_sha,
    get_library_versions,
    list_s3_completed,
    load_dataset,
    metrics_s3_path,
    rankings_s3_path,
    s3_file_exists,
    upload_parquet_to_s3,
    utc_now_iso,
)
from paper.scripts.experiments._driver import (
    build_common_parser,
    init_ray,
    iter_grid,
    log_dry_run,
    log_failures,
    resolve_grid,
    run_futures,
)
from paper.scripts.infra.config import load_config
from paper.scripts.utils.constants import (
    CLF_DOWNSTREAM_MODELS,
    CLF_METHODS,
    N_SPLITS,
    REG_DOWNSTREAM_MODELS,
    REG_METHODS,
)
from paper.scripts.utils.experiment_configs import config_label, expand_method_configs

config = load_config()


# =============================================================================
# Ray resource planning (Stage 2 evaluation CPU needs)
# =============================================================================


def evaluation_num_cpus(task_type: str, *, downstream_models: list[str] | None = None) -> int:
    """Return Ray CPU reservation for a single (method, dataset, seed) evaluation config.

    Stage 2 runs all downstream models sequentially inside one Ray task.
    If you add heavy downstream models later (e.g., xgb/lgbm), reserve enough CPUs here so Ray packs work correctly.
    """
    exp = config.experiment

    models = downstream_models
    if models is None:
        models = CLF_DOWNSTREAM_MODELS if task_type == "classification" else REG_DOWNSTREAM_MODELS

    cpus = int(exp.evaluation_cpus_default)
    for model_name in models:
        override = exp.evaluation_cpus_overrides.get(model_name)
        if override is not None:
            cpus = max(cpus, int(override))

    return cpus


def evaluation_memory_bytes(task_type: str, *, downstream_models: list[str] | None = None) -> int:
    """Return Ray memory reservation (in bytes) for a single evaluation config."""
    exp = config.experiment

    models = downstream_models
    if models is None:
        models = CLF_DOWNSTREAM_MODELS if task_type == "classification" else REG_DOWNSTREAM_MODELS

    memory_gb = exp.evaluation_memory_gb_default
    for model_name in models:
        override = exp.evaluation_memory_gb_overrides.get(model_name)
        if override is not None:
            memory_gb = max(memory_gb, override)

    return int(memory_gb * 1024 * 1024 * 1024)


def get_clf_models(random_state: int, *, evaluation_cpus: int) -> dict[str, Any]:
    return {
        "lr": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "svm": SVC(class_weight="balanced", probability=True, random_state=random_state),
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=evaluation_cpus),
    }


def get_reg_models(random_state: int, *, evaluation_cpus: int) -> dict[str, Any]:
    return {
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "svr": SVR(),
        "knn": KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=evaluation_cpus),
    }


def compute_roc_auc(y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray | None) -> float:
    """Compute ROC AUC with explicit label/proba alignment.

    - Binary: uses the probability column corresponding to classes[1].
    - Multiclass: uses sklearn's OVR + weighted average with explicit labels ordering.
    - Undefined cases (single-class y_true or missing classes) return NaN.
    """
    unique = np.unique(y_true)
    if unique.size < 2:
        logger.debug("ROC AUC undefined: y_true has only {} unique class(es)", unique.size)
        return float(np.nan)

    if y_proba.ndim == 1:
        if classes is None or len(classes) < 2:
            logger.debug("ROC AUC undefined: 1D y_proba with insufficient classes")
            return float(np.nan)
        y_bin = (y_true == classes[1]).astype(int)
        return float(roc_auc_score(y_bin, y_proba))

    if y_proba.shape[1] == 2:
        if classes is None or len(classes) < 2:
            logger.debug("ROC AUC undefined: binary y_proba with insufficient classes")
            return float(np.nan)
        y_bin = (y_true == classes[1]).astype(int)
        return float(roc_auc_score(y_bin, y_proba[:, 1]))

    # Multiclass: require all classes to be present to avoid undefined metrics.
    if classes is None:
        logger.debug("ROC AUC undefined: multiclass y_proba with no classes array")
        return float(np.nan)
    if y_proba.shape[1] != len(classes):
        raise ValueError(
            f"y_proba has {y_proba.shape[1]} columns but classes has {len(classes)} entries"
        )
    if np.unique(y_true).size < len(classes):
        logger.debug("ROC AUC undefined: y_true has fewer unique classes than expected")
        return float(np.nan)

    return float(
        roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted", labels=classes)
    )


def evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ranking: np.ndarray,
    task_type: str,
    random_state: int,
    evaluation_cpus: int,
) -> list[dict[str, Any]]:
    n_features = X_train.shape[1]
    k_values = [5, 10, 25, 50, 100, n_features]
    k_values = sorted(set(k for k in k_values if k <= n_features))

    downstream_models = (
        CLF_DOWNSTREAM_MODELS if task_type == "classification" else REG_DOWNSTREAM_MODELS
    )
    model_factory = get_clf_models if task_type == "classification" else get_reg_models

    results = []
    for k in k_values:
        top_k_features = ranking[:k]
        X_train_k = X_train[:, top_k_features]
        X_test_k = X_test[:, top_k_features]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_k)
        X_test_scaled = scaler.transform(X_test_k)

        models = model_factory(random_state, evaluation_cpus=evaluation_cpus)
        for model_name in downstream_models:
            model = models[model_name]
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            if task_type == "classification":
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred, average="weighted"),
                    "f1_macro": f1_score(y_test, y_pred, average="macro"),
                    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                }
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test_scaled)
                    classes = model.classes_ if hasattr(model, "classes_") else None
                    roc_auc = compute_roc_auc(y_test, y_proba, classes)
                    metrics["roc_auc"] = roc_auc
                    metrics["auc"] = roc_auc
            else:
                metrics = {
                    "r2": r2_score(y_test, y_pred),
                    "mse": mean_squared_error(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "mae": mean_absolute_error(y_test, y_pred),
                }

            results.append(
                {
                    "k": k,
                    "n_features_selected": k,
                    "downstream_model": model_name,
                    **metrics,
                }
            )

    return results


def run_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    rankings_df: pd.DataFrame,
    dataset: str,
    method: str,
    method_id: str,
    task_type: str,
    seed: int,
    evaluation_cpus: int,
) -> list[dict[str, Any]]:
    results = []
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])

    # Reconstruct deterministic CV splits instead of storing per-sample indices in artifacts.
    if task_type == "classification":
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    splits = list(cv.split(X, y))

    for _, row in rankings_df.iterrows():
        fold_idx = int(row["fold_idx"])
        try:
            train_idx, test_idx = splits[fold_idx]
        except IndexError as e:
            raise ValueError(f"Invalid fold_idx={fold_idx}; expected 0..{len(splits) - 1}") from e
        ranking = np.array(row["feature_ranking"])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rs = seed * 1000 + fold_idx  # Avoid collision: seed=0/fold=4 vs seed=4/fold=0
        fold_results = evaluate_fold(
            X_train, y_train, X_test, y_test, ranking, task_type, rs, evaluation_cpus
        )

        for res in fold_results:
            res["dataset"] = dataset
            res["method_id"] = method_id
            res["method"] = method_id
            res["method_base"] = method
            res["task_type"] = task_type
            res["seed"] = seed
            res["fold_idx"] = fold_idx
            res["artifact_version"] = 2
            res["n_samples"] = n_samples
            res["n_features"] = n_features
            res["evaluation_cpus"] = evaluation_cpus
            results.append(res)

    return results


@ray.remote(resources={"evaluation": 1})
def process_config(
    method_config: dict[str, Any],
    dataset: str,
    seed: int,
    task_type: str,
    evaluation_cpus: int,
    git_sha: str,
    skip_existing: bool = False,
) -> dict[str, Any]:
    method = method_config["method"]
    method_id = config_label(method_config)
    rankings_path = rankings_s3_path(task_type, dataset, method_id, seed)
    metrics_path = metrics_s3_path(task_type, dataset, method_id, seed)
    created_at_utc = utc_now_iso()
    runtime: dict[str, Any] = {"hostname": socket.gethostname(), "pid": os.getpid()}
    try:
        rctx = ray.get_runtime_context()
        runtime["ray_node_id"] = rctx.get_node_id()
        runtime["ray_task_id"] = str(rctx.get_task_id())
    except Exception:
        pass

    # Skip if output exists (per-task check for extra safety)
    if skip_existing:
        try:
            if s3_file_exists(metrics_path, region_name=config.region):
                return {
                    "status": "skipped",
                    "method": method_id,
                    "dataset": dataset,
                    "seed": seed,
                    "evaluation_cpus": evaluation_cpus,
                    "rankings_path": rankings_path,
                    "metrics_path": metrics_path,
                    "reason": "output_exists",
                    **runtime,
                }
        except Exception:
            pass  # Continue if check fails

    try:
        rankings_exist = s3_file_exists(rankings_path, region_name=config.region)
    except Exception as e:
        return {
            "status": "failed",
            "method": method_id,
            "dataset": dataset,
            "seed": seed,
            "evaluation_cpus": evaluation_cpus,
            "rankings_path": rankings_path,
            "metrics_path": metrics_path,
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
            **runtime,
        }

    if not rankings_exist:
        return {
            "status": "no_rankings",
            "method": method_id,
            "dataset": dataset,
            "seed": seed,
            "evaluation_cpus": evaluation_cpus,
            "rankings_path": rankings_path,
            "metrics_path": metrics_path,
            **runtime,
        }

    try:
        rankings_df = download_parquet_from_s3(rankings_path, region_name=config.region)
        X, y = load_dataset(dataset, task_type)

        # Get dataset metadata for provenance
        dataset_meta = get_dataset_metadata(dataset, task_type)

        tic = time.perf_counter()
        results = run_evaluation(
            X, y, rankings_df, dataset, method, method_id, task_type, seed, evaluation_cpus
        )
        elapsed = time.perf_counter() - tic
        for row in results:
            row["elapsed_seconds"] = float(elapsed)
            row["created_at_utc"] = created_at_utc
            row["git_sha"] = git_sha
            row["library_versions"] = get_library_versions()
            # Add dataset metadata
            row["dataset_source"] = dataset_meta.get("dataset_source")
            row["dataset_type"] = dataset_meta.get("dataset_type")
            row["dataset_family"] = dataset_meta.get("dataset_family")
            row["n_informative"] = dataset_meta.get("n_informative")
        upload_parquet_to_s3(
            results,
            metrics_path,
            region_name=config.region,
            validate=config.experiment.s3_validate_uploads,
        )
        return {
            "status": "done",
            "method": method_id,
            "dataset": dataset,
            "seed": seed,
            "evaluation_cpus": evaluation_cpus,
            "elapsed_seconds": float(elapsed),
            "rankings_path": rankings_path,
            "metrics_path": metrics_path,
            **runtime,
        }
    except Exception as e:
        return {
            "status": "failed",
            "method": method_id,
            "dataset": dataset,
            "seed": seed,
            "evaluation_cpus": evaluation_cpus,
            "rankings_path": rankings_path,
            "metrics_path": metrics_path,
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
            **runtime,
        }


def main() -> None:
    args = build_common_parser("Ray Stage 2: rankings → metrics").parse_args()
    init_ray(args.ray_address)

    task_type = args.task_type or config.experiment.type
    methods = CLF_METHODS if task_type == "classification" else REG_METHODS
    method_configs = expand_method_configs(methods)
    datasets = get_datasets(task_type, source=args.source)
    n_seeds = config.experiment.n_seeds
    git_sha = get_git_sha()

    method_configs, datasets, seeds = resolve_grid(
        method_configs=method_configs,
        datasets=datasets,
        n_seeds=n_seeds,
        datasets_csv=args.datasets,
        methods_csv=args.methods,
        seeds_csv=args.seeds,
    )

    total_expected = len(method_configs) * len(datasets) * len(seeds)
    grid = list(iter_grid(method_configs, datasets, seeds))
    if args.only_missing:
        completed_metrics = list_s3_completed("metrics", task_type, region_name=config.region)
        completed_rankings = list_s3_completed("rankings", task_type, region_name=config.region)
        pending: list[tuple[dict[str, Any], str, int]] = []
        missing_rankings = 0
        for method_cfg, dataset, seed in grid:
            method_id = config_label(method_cfg)
            key = (method_id, dataset, seed)
            if key in completed_metrics:
                continue
            if key not in completed_rankings:
                missing_rankings += 1
                continue
            pending.append((method_cfg, dataset, seed))
        logger.info(
            "Stage 2 only-missing: expected={}, completed_metrics={}, pending={}, missing_rankings={}",
            total_expected,
            len(completed_metrics),
            len(pending),
            missing_rankings,
        )
    else:
        pending = grid

    logger.info(
        "Submitting {} configs ({} methods × {} datasets × {} seeds)",
        len(pending),
        len(method_configs),
        len(datasets),
        len(seeds),
    )

    evaluation_cpus = evaluation_num_cpus(task_type)
    evaluation_memory = evaluation_memory_bytes(task_type)
    logger.info(
        "Evaluation resource config: cpus={}, memory={}GB",
        evaluation_cpus,
        evaluation_memory / (1024 * 1024 * 1024),
    )

    if args.dry_run:
        log_dry_run(
            pending,
            stage="stage2",
            limit=args.dry_run_limit,
            describe=lambda m,
            d,
            s: f"method={config_label(m)}, dataset={d}, seed={s}, num_cpus={evaluation_cpus}",
        )
        return

    futures = [
        process_config.options(num_cpus=evaluation_cpus, memory=evaluation_memory).remote(
            m, d, s, task_type, evaluation_cpus, git_sha, skip_existing=args.skip_existing
        )
        for m, d, s in pending
    ]

    _counts, failures, _elapsed, _results = run_futures(
        futures,
        stage="stage2",
        success_statuses={"done"},
        skip_statuses={"no_rankings", "skipped"},
    )
    log_failures(failures, stage="stage2")


if __name__ == "__main__":
    main()
