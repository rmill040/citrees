"""Ray-based distributed downstream evaluation."""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import pandas as pd
import ray
from loguru import logger
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from paper.scripts.experiments._common import (
    download_parquet_from_s3,
    get_git_sha,
    get_datasets,
    load_dataset,
    metrics_s3_path,
    rankings_s3_path,
    s3_file_exists,
    utc_now_iso,
    upload_parquet_to_s3,
)
from paper.scripts.utils.constants import (
    CLF_DOWNSTREAM_MODELS,
    CLF_METHODS,
    N_SPLITS,
    REG_DOWNSTREAM_MODELS,
    REG_METHODS,
)
from paper.scripts.utils.experiment_configs import config_label, expand_method_configs
from paper.scripts.infra.config import load_config

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


def safe_roc_auc_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute ROC AUC, returning NaN when undefined (e.g., single-class fold)."""
    try:
        if y_proba.ndim == 1:
            return float(roc_auc_score(y_true, y_proba))
        if y_proba.shape[1] == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
    except Exception:
        return float(np.nan)


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

    downstream_models = CLF_DOWNSTREAM_MODELS if task_type == "classification" else REG_DOWNSTREAM_MODELS
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
                }
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test_scaled)
                    metrics["roc_auc"] = safe_roc_auc_score(y_test, y_proba)
            else:
                metrics = {
                    "r2": r2_score(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "mae": mean_absolute_error(y_test, y_pred),
                }

            results.append({
                "k": k,
                "n_features_selected": k,
                "downstream_model": model_name,
                **metrics,
            })

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

        rs = seed + fold_idx
        fold_results = evaluate_fold(X_train, y_train, X_test, y_test, ranking, task_type, rs, evaluation_cpus)

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
    config: dict[str, Any],
    dataset: str,
    seed: int,
    task_type: str,
    evaluation_cpus: int,
    git_sha: str,
) -> dict[str, Any]:
    method = config["method"]
    method_id = config_label(config)
    rankings_path = rankings_s3_path(task_type, dataset, method_id, seed)
    metrics_path = metrics_s3_path(task_type, dataset, method_id, seed)
    created_at_utc = utc_now_iso()

    try:
        rankings_exist = s3_file_exists(rankings_path, region_name=config.region)
    except Exception as e:
        return {"status": "failed", "method": method_id, "dataset": dataset, "seed": seed, "error": str(e)}

    if not rankings_exist:
        return {"status": "no_rankings", "method": method_id, "dataset": dataset, "seed": seed}

    try:
        rankings_df = download_parquet_from_s3(rankings_path, region_name=config.region)
        X, y = load_dataset(dataset, task_type)
        tic = time.perf_counter()
        results = run_evaluation(X, y, rankings_df, dataset, method, method_id, task_type, seed, evaluation_cpus)
        elapsed = time.perf_counter() - tic
        for row in results:
            row["elapsed_seconds"] = float(elapsed)
            row["created_at_utc"] = created_at_utc
            row["git_sha"] = git_sha
        upload_parquet_to_s3(results, metrics_path, region_name=config.region)
        return {"status": "done", "method": method_id, "dataset": dataset, "seed": seed, "elapsed": elapsed}
    except Exception as e:
        return {"status": "failed", "method": method_id, "dataset": dataset, "seed": seed, "error": str(e)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ray Stage 2: rankings → metrics")
    parser.add_argument(
        "--ray-address",
        default="auto",
        help="Ray address (default: auto). Use 'local' for local mode.",
    )
    parser.add_argument("--task-type", choices=["classification", "regression"], default=None)
    parser.add_argument("--source", choices=["all", "real", "synthetic"], default="all", help="Dataset source filter")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names (default: all)")
    parser.add_argument("--methods", default=None, help="Comma-separated base method names (default: all)")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed indices (default: 0..n_seeds-1)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned configs and exit")
    return parser.parse_args()


def _parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",")]
    return [v for v in items if v]


def _parse_csv_ints(value: str | None) -> list[int] | None:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return [int(v) for v in items]


def main() -> None:
    args = _parse_args()
    if args.ray_address == "local":
        ray.init(ignore_reinit_error=True)
    else:
        ray.init(address=args.ray_address, ignore_reinit_error=True)

    task_type = args.task_type or config.experiment.type
    methods = CLF_METHODS if task_type == "classification" else REG_METHODS
    method_configs = expand_method_configs(methods)
    datasets = get_datasets(task_type, source=args.source)
    n_seeds = config.experiment.n_seeds
    git_sha = get_git_sha()

    datasets_filter = set(_parse_csv_list(args.datasets) or [])
    methods_filter = set(_parse_csv_list(args.methods) or [])
    seeds_filter = _parse_csv_ints(args.seeds)

    if datasets_filter:
        datasets = [d for d in datasets if d in datasets_filter]
    if methods_filter:
        method_configs = [c for c in method_configs if c.get("method") in methods_filter]
    seeds = list(range(n_seeds)) if seeds_filter is None else seeds_filter

    configs = [(m, d, s) for m in method_configs for d in datasets for s in seeds]
    logger.info(
        "Submitting {} configs ({} methods × {} datasets × {} seeds)",
        len(configs),
        len(method_configs),
        len(datasets),
        len(seeds),
    )

    evaluation_cpus = evaluation_num_cpus(task_type)
    logger.info(
        "Evaluation CPU config: default={} (effective={}), overrides={}",
        config.experiment.evaluation_cpus_default,
        evaluation_cpus,
        config.experiment.evaluation_cpus_overrides,
    )

    if args.dry_run:
        for method_cfg, dataset, seed in configs[:50]:
            logger.info(
                "DRY RUN: method={}, dataset={}, seed={}, num_cpus={}",
                config_label(method_cfg),
                dataset,
                seed,
                evaluation_cpus,
            )
        if len(configs) > 50:
            logger.info("DRY RUN: ... ({} more configs)", len(configs) - 50)
        return

    futures = [
        process_config.options(num_cpus=evaluation_cpus).remote(m, d, s, task_type, evaluation_cpus, git_sha)
        for m, d, s in configs
    ]
    results = ray.get(futures)

    done = sum(1 for r in results if r["status"] == "done")
    no_rankings = sum(1 for r in results if r["status"] == "no_rankings")
    failed = sum(1 for r in results if r["status"] == "failed")

    logger.info(f"Done: {done}, No rankings: {no_rankings}, Failed: {failed}")

    if failed > 0:
        for r in results:
            if r["status"] == "failed":
                logger.error(f"Failed: {r['method']}/{r['dataset']}/seed{r['seed']}: {r['error']}")


if __name__ == "__main__":
    main()
