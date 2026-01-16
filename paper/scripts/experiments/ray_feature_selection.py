"""Ray-based distributed feature selection."""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import ray
import shap
from boruta import BorutaPy
from loguru import logger
from mrmr import mrmr_classif, mrmr_regression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
)
from citrees._selector import (
    ClassifierSelectors,
    ClassifierSelectorTests,
    RegressorSelectors,
    RegressorSelectorTests,
)
from paper.scripts.utils.constants import CLF_METHODS, N_SPLITS, REG_METHODS
from paper.scripts.utils.experiment_configs import config_label, expand_method_configs, extract_params
from paper.scripts.infra.config import load_config

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

config = load_config()

# Data directory - works both locally and when rsynced to remote
# On remote, script is at ~/ray_feature_selection.py but data is at ~/citrees/paper/data/
DATA_DIR = (
    Path("/home/ubuntu/citrees/paper/data")
    if Path("/home/ubuntu/citrees").exists()
    else Path(__file__).resolve().parents[2] / "data"
)


# =============================================================================
# Ray resource planning (per-method CPU needs)
# =============================================================================

# Many selection methods are effectively single-threaded, while others (tree ensembles) can use multi-core parallelism.
# Request Ray `num_cpus` per task to avoid oversubscription and to pack lightweight tasks per node.
#
# Tune these values based on your instance types and dataset sizes.
_DEFAULT_N_JOBS = 1
_THREADED_SELECTION_METHODS = {
    # Embedding methods
    "rf",
    "et",
    "xgb",
    "lgbm",
    "cat",
    # Wrapper methods (typically train tree ensembles internally)
    "boruta",
    "pi",
    "cpi",
    "shap",
    "rfe",
}


def selection_num_cpus(method: str, *, n_samples: int | None = None, n_features: int | None = None) -> int:
    """Return Ray CPU reservation for a single (method, dataset, seed) config.

    This value is also passed down as the thread count (`n_jobs` / `thread_count`) for thread-parallel methods so that
    Ray scheduling and internal library parallelism stay aligned.
    """
    exp = config.experiment

    override = exp.selection_cpus_overrides.get(method)
    if override is not None:
        return int(override)

    if method == "cif":
        if n_samples is not None and n_features is not None:
            complexity = n_samples * n_features
            if complexity >= exp.selection_cif_large_threshold:
                return exp.selection_cpus_cif_large
        return exp.selection_cpus_cif

    if method in _THREADED_SELECTION_METHODS:
        return exp.selection_cpus_threaded

    return exp.selection_cpus_default


def get_s3_client():
    """Create S3 client lazily (avoid serialization issues with Ray)."""
    return boto3.client("s3", region_name=config.region)


def get_datasets(task_type: str) -> list[str]:
    data_dir = DATA_DIR
    prefix = "clf_" if task_type == "classification" else "reg_"
    datasets = []
    for f in data_dir.glob(f"{prefix}*.parquet"):
        name = f.stem.replace(prefix, "").replace(".snappy", "")
        datasets.append(name)
    return sorted(datasets)


def get_dataset_path(name: str, task_type: str) -> Path:
    prefix = "clf_" if task_type == "classification" else "reg_"
    path = DATA_DIR / f"{prefix}{name}.parquet"
    if not path.exists():
        path = DATA_DIR / f"{prefix}{name}.snappy.parquet"
    return path


def get_dataset_shape(name: str, task_type: str) -> tuple[int, int]:
    """Get (n_samples, n_features) from parquet metadata, without loading the full dataset."""
    path = get_dataset_path(name, task_type)

    try:
        pf = pq.ParquetFile(path)
        n_samples = pf.metadata.num_rows
        feature_columns = [c for c in pf.schema_arrow.names if c != "y"]
        return int(n_samples), int(len(feature_columns))
    except Exception:
        X, _y = load_dataset(name, task_type)
        return int(X.shape[0]), int(X.shape[1])


def load_dataset(name: str, task_type: str) -> tuple[np.ndarray, np.ndarray]:
    path = get_dataset_path(name, task_type)
    df = pd.read_parquet(path)
    y = df.pop("y").values
    if task_type == "classification":
        y = y.astype(np.int64)
    else:
        y = y.astype(np.float64)
    X = df.values.astype(np.float64)
    return X, y


def s3_file_exists(s3_path: str) -> bool:
    parts = s3_path.replace("s3://", "").split("/", 1)
    try:
        get_s3_client().head_object(Bucket=parts[0], Key=parts[1])
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def upload_to_s3(data: list[dict], s3_path: str) -> None:
    df = pd.DataFrame(data)
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    parts = s3_path.replace("s3://", "").split("/", 1)
    get_s3_client().put_object(Bucket=parts[0], Key=parts[1], Body=buffer.getvalue())


def filter_selector(X: np.ndarray, y: np.ndarray, method: str, task_type: str, random_state: int) -> np.ndarray:
    n_features = X.shape[1]
    selectors = ClassifierSelectors if task_type == "classification" else RegressorSelectors
    selector_fn = selectors[method]
    scores = np.zeros(n_features)
    rng = np.random.default_rng(random_state)

    if task_type == "classification":
        n_classes = len(np.unique(y))
        for j in range(n_features):
            scores[j] = selector_fn(X[:, j], y, n_classes, random_state=rng.integers(0, 2**31))
    else:
        use_abs = method == "pc"
        for j in range(n_features):
            score = selector_fn(X[:, j], y, standardize=True, random_state=rng.integers(0, 2**31))
            scores[j] = abs(score) if use_abs else score

    return np.argsort(scores)[::-1]


def permutation_selector(X: np.ndarray, y: np.ndarray, method: str, task_type: str, random_state: int) -> np.ndarray:
    base_method = method.replace("ptest_", "")
    n_features = X.shape[1]
    selectors = ClassifierSelectors if task_type == "classification" else RegressorSelectors
    selector_tests = ClassifierSelectorTests if task_type == "classification" else RegressorSelectorTests
    selector_fn = selectors[base_method]
    test_fn = selector_tests[base_method]

    scores = np.zeros(n_features)
    pvalues = np.ones(n_features)
    rng = np.random.default_rng(random_state)

    if task_type == "classification":
        n_classes = len(np.unique(y))
        for j in range(n_features):
            rs = rng.integers(0, 2**31)
            scores[j] = selector_fn(X[:, j], y, n_classes, random_state=rs)
            pvalues[j] = test_fn(
                x=X[:, j],
                y=y,
                n_classes=n_classes,
                alpha=0.05,
                n_resamples=1000,
                early_stopping=None,
                random_state=rs,
            )
    else:
        use_abs = base_method == "pc"
        for j in range(n_features):
            rs = rng.integers(0, 2**31)
            score = selector_fn(X[:, j], y, standardize=True, random_state=rs)
            scores[j] = abs(score) if use_abs else score
            pvalues[j] = test_fn(
                x=X[:, j],
                y=y,
                standardize=True,
                alpha=0.05,
                n_resamples=1000,
                early_stopping=None,
                random_state=rs,
            )

    return np.lexsort((-scores, pvalues))


def get_embedding_model(
    method: str,
    task_type: str,
    random_state: int,
    n_jobs: int = _DEFAULT_N_JOBS,
    params: dict[str, Any] | None = None,
):
    params = params or {}

    if task_type == "classification":
        if method == "rf":
            base = {"n_estimators": 100, "n_jobs": n_jobs, "random_state": random_state}
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return RandomForestClassifier(**model_params)
        if method == "et":
            base = {"n_estimators": 100, "n_jobs": n_jobs, "random_state": random_state}
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return ExtraTreesClassifier(**model_params)
        if method == "cit":
            model_params = {**params, "random_state": random_state}
            return ConditionalInferenceTreeClassifier(**model_params)
        if method == "cif":
            base = {"n_estimators": 100, "n_jobs": n_jobs, "random_state": random_state}
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return ConditionalInferenceForestClassifier(**model_params)
        if method == "xgb":
            if not HAS_XGB:
                raise ValueError("XGBoost is not installed")
            base = {"n_estimators": 100, "n_jobs": n_jobs, "random_state": random_state, "verbosity": 0}
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return XGBClassifier(**model_params)
        if method == "lgbm":
            if not HAS_LGBM:
                raise ValueError("LightGBM is not installed")
            base = {"n_estimators": 100, "n_jobs": n_jobs, "random_state": random_state, "verbosity": -1}
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return LGBMClassifier(**model_params)
        if method == "cat":
            if not HAS_CAT:
                raise ValueError("CatBoost is not installed")
            base = {"n_estimators": 100, "random_state": random_state, "verbose": 0, "thread_count": n_jobs}
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return CatBoostClassifier(**model_params)
    else:
        if method == "rf":
            base = {"n_estimators": 100, "n_jobs": n_jobs, "random_state": random_state}
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return RandomForestRegressor(**model_params)
        if method == "et":
            base = {"n_estimators": 100, "n_jobs": n_jobs, "random_state": random_state}
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return ExtraTreesRegressor(**model_params)
        if method == "cit":
            model_params = {**params, "random_state": random_state}
            return ConditionalInferenceTreeRegressor(**model_params)
        if method == "cif":
            base = {"n_estimators": 100, "n_jobs": n_jobs, "random_state": random_state}
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return ConditionalInferenceForestRegressor(**model_params)
        if method == "xgb":
            if not HAS_XGB:
                raise ValueError("XGBoost is not installed")
            base = {"n_estimators": 100, "n_jobs": n_jobs, "random_state": random_state, "verbosity": 0}
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return XGBRegressor(**model_params)
        if method == "lgbm":
            if not HAS_LGBM:
                raise ValueError("LightGBM is not installed")
            base = {"n_estimators": 100, "n_jobs": n_jobs, "random_state": random_state, "verbosity": -1}
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return LGBMRegressor(**model_params)
        if method == "cat":
            if not HAS_CAT:
                raise ValueError("CatBoost is not installed")
            base = {"n_estimators": 100, "random_state": random_state, "verbose": 0, "thread_count": n_jobs}
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return CatBoostRegressor(**model_params)

    raise ValueError(f"Unknown embedding method: {method}")


def embedding_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    method: str,
    task_type: str,
    random_state: int,
    params: dict[str, Any] | None = None,
    n_jobs: int = _DEFAULT_N_JOBS,
) -> tuple[np.ndarray, dict[str, Any]]:
    model = get_embedding_model(method, task_type, random_state, n_jobs=n_jobs, params=params)
    model.fit(X_train, y_train)
    ranking = np.argsort(model.feature_importances_)[::-1]

    embedding_data = {
        "train_preds": model.predict(X_train).tolist(),
        "test_preds": model.predict(X_test).tolist(),
    }
    if task_type == "classification" and hasattr(model, "predict_proba"):
        embedding_data["train_proba"] = model.predict_proba(X_train).tolist()
        embedding_data["test_proba"] = model.predict_proba(X_test).tolist()

    return ranking, embedding_data


def boruta_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    random_state: int,
    n_jobs: int = _DEFAULT_N_JOBS,
) -> np.ndarray:
    if task_type == "classification":
        base_model = RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=random_state)
    else:
        base_model = RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=random_state)

    boruta = BorutaPy(base_model, n_estimators="auto", random_state=random_state, verbose=0)
    boruta.fit(X_train, y_train)
    return np.argsort(boruta.ranking_)


def pi_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    random_state: int,
    n_jobs: int = _DEFAULT_N_JOBS,
    val_fraction: float = 0.2,
) -> np.ndarray:
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=random_state)
        scoring = "accuracy"
    else:
        model = RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=random_state)
        scoring = "r2"

    stratify = y_train if task_type == "classification" else None
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_fraction,
        random_state=random_state,
        stratify=stratify,
    )

    model.fit(X_fit, y_fit)
    result = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=10,
        random_state=random_state,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    return np.argsort(result.importances_mean)[::-1]


def shap_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    random_state: int,
    n_jobs: int = _DEFAULT_N_JOBS,
) -> np.ndarray:
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=random_state)
    else:
        model = RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=random_state)

    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)

    if X_train.shape[0] > 1000:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X_train.shape[0], 1000, replace=False)
        X_sample = X_train[idx]
    else:
        X_sample = X_train

    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        importances = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        importances = np.abs(shap_values).mean(axis=0)

    return np.argsort(importances)[::-1]


def cpi_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    random_state: int,
    n_jobs: int = _DEFAULT_N_JOBS,
    val_fraction: float = 0.2,
    n_repeats: int = 10,
    correlation_threshold: float = 0.5,
) -> np.ndarray:
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=random_state)
        scoring_fn = lambda m, X, y: (m.predict(X) == y).mean()
    else:
        model = RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=random_state)
        scoring_fn = lambda m, X, y: 1 - ((y - m.predict(X)) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    stratify = y_train if task_type == "classification" else None
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_fraction,
        random_state=random_state,
        stratify=stratify,
    )

    model.fit(X_fit, y_fit)
    rng = np.random.default_rng(random_state)
    n_features = X_val.shape[1]
    baseline = scoring_fn(model, X_val, y_val)
    importances = np.zeros(n_features)

    for j in range(n_features):
        scores = []
        for rep in range(n_repeats):
            X_perm = X_val.copy()
            corr_features = _find_correlated(X_val, j, correlation_threshold)

            if len(corr_features) > 0:
                strata = _create_strata(X_val[:, corr_features])
                for stratum in np.unique(strata):
                    mask = strata == stratum
                    idx = np.where(mask)[0]
                    if len(idx) > 1:
                        X_perm[idx, j] = rng.permutation(X_perm[idx, j])
            else:
                X_perm[:, j] = rng.permutation(X_perm[:, j])

            scores.append(scoring_fn(model, X_perm, y_val))
        importances[j] = baseline - np.mean(scores)

    return np.argsort(importances)[::-1]


def _find_correlated(X: np.ndarray, j: int, threshold: float) -> list[int]:
    """Find features correlated with feature j above threshold."""
    correlated = []
    for k in range(X.shape[1]):
        if k != j:
            if np.std(X[:, j]) == 0 or np.std(X[:, k]) == 0:
                continue
            corr = np.abs(np.corrcoef(X[:, j], X[:, k])[0, 1])
            if not np.isnan(corr) and corr > threshold:
                correlated.append(k)
    return correlated


def _create_strata(X: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Create strata for conditional permutation."""
    if X.shape[1] == 0:
        return np.zeros(X.shape[0], dtype=int)
    strata = np.zeros(X.shape[0], dtype=int)
    for j in range(X.shape[1]):
        if np.std(X[:, j]) == 0:
            continue
        bins = np.unique(np.percentile(X[:, j], np.linspace(0, 100, n_bins + 1)))
        if len(bins) > 1:
            binned = np.digitize(X[:, j], bins[1:-1])
            strata = strata * n_bins + binned
    return strata


def mrmr_selector(X_train: np.ndarray, y_train: np.ndarray, task_type: str) -> np.ndarray:
    df = pd.DataFrame(X_train)
    y_series = pd.Series(y_train)
    n_features = X_train.shape[1]

    if task_type == "classification":
        selected = mrmr_classif(df, y_series, K=n_features, show_progress=False)
    else:
        selected = mrmr_regression(df, y_series, K=n_features, show_progress=False)

    return np.array(selected)


def rfe_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    random_state: int,
    n_jobs: int = _DEFAULT_N_JOBS,
) -> np.ndarray:
    if task_type == "classification":
        base_model = RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=random_state)
    else:
        base_model = RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=random_state)

    rfe = RFE(base_model, n_features_to_select=1, step=1)
    rfe.fit(X_train, y_train)
    return np.argsort(rfe.ranking_)


def run_selection(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    task_type: str,
    seed: int,
    params: dict[str, Any] | None = None,
    n_jobs: int = _DEFAULT_N_JOBS,
) -> list[dict[str, Any]]:
    params = params or {}

    if task_type == "classification":
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    results = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit preprocessing only on the training fold to avoid CV leakage.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        rs = seed + fold_idx
        embedding_data = None

        if method in ["mc", "mi", "rdc", "pc", "dc"]:
            ranking = filter_selector(X_train, y_train, method, task_type, rs)
        elif method.startswith("ptest_"):
            ranking = permutation_selector(X_train, y_train, method, task_type, rs)
        elif method in ["rf", "et", "xgb", "lgbm", "cat", "cit", "cif"]:
            ranking, embedding_data = embedding_selector(
                X_train, y_train, X_test, y_test, method, task_type, rs, params=params, n_jobs=n_jobs
            )
        elif method == "boruta":
            ranking = boruta_selector(X_train, y_train, task_type, rs, n_jobs=n_jobs)
        elif method == "pi":
            ranking = pi_selector(X_train, y_train, task_type, rs, n_jobs=n_jobs)
        elif method == "shap":
            ranking = shap_selector(X_train, y_train, task_type, rs, n_jobs=n_jobs)
        elif method == "cpi":
            ranking = cpi_selector(X_train, y_train, task_type, rs, n_jobs=n_jobs)
        elif method == "mrmr":
            ranking = mrmr_selector(X_train, y_train, task_type)
        elif method == "rfe":
            ranking = rfe_selector(X_train, y_train, task_type, rs, n_jobs=n_jobs)
        else:
            raise ValueError(f"Unknown method: {method}")

        fold_result = {
            "fold_idx": fold_idx,
            "feature_ranking": ranking.tolist(),
        }
        if embedding_data:
            fold_result.update({f"embedding_{k}": v for k, v in embedding_data.items()})

        results.append(fold_result)

    return results


@ray.remote(resources={"selection": 1})
def process_config(config: dict[str, Any], dataset: str, seed: int, task_type: str, selection_cpus: int) -> dict[str, Any]:
    method = config["method"]
    params = extract_params(config)
    method_id = config_label(config)
    s3_path = f"s3://{os.environ['S3_BUCKET']}/rankings/{task_type}/{dataset}/{method_id}_seed{seed}.parquet"

    try:
        X, y = load_dataset(dataset, task_type)
        tic = time.perf_counter()
        results = run_selection(X, y, method, task_type, seed, params=params, n_jobs=selection_cpus)
        elapsed = time.perf_counter() - tic
        upload_to_s3(results, s3_path)
        return {"status": "done", "method": method_id, "dataset": dataset, "seed": seed, "elapsed": elapsed}
    except Exception as e:
        return {"status": "failed", "method": method_id, "dataset": dataset, "seed": seed, "error": str(e)}


def main():
    ray.init(address="auto", ignore_reinit_error=True)

    task_type = config.experiment.type
    methods = CLF_METHODS if task_type == "classification" else REG_METHODS
    method_configs = expand_method_configs(methods)
    datasets = get_datasets(task_type)
    n_seeds = config.experiment.n_seeds

    configs = [(m, d, s) for m in method_configs for d in datasets for s in range(n_seeds)]
    logger.info(
        f"Submitting {len(configs)} configs ({len(method_configs)} methods × {len(datasets)} datasets × {n_seeds} seeds)"
    )

    exp = config.experiment
    logger.info(
        "Selection CPU config: default={}, threaded={}, cif={}, cif_large={} (threshold={} on n_samples*n_features)",
        exp.selection_cpus_default,
        exp.selection_cpus_threaded,
        exp.selection_cpus_cif,
        exp.selection_cpus_cif_large,
        exp.selection_cif_large_threshold,
    )

    dataset_shapes = {d: get_dataset_shape(d, task_type) for d in datasets}

    futures = []
    for method_cfg, dataset, seed in configs:
        method = method_cfg["method"]
        n_samples, n_features = dataset_shapes[dataset]
        selection_cpus = selection_num_cpus(method, n_samples=n_samples, n_features=n_features)
        futures.append(
            process_config.options(num_cpus=selection_cpus).remote(method_cfg, dataset, seed, task_type, selection_cpus)
        )
    results = ray.get(futures)

    done = sum(1 for r in results if r["status"] == "done")
    failed = sum(1 for r in results if r["status"] == "failed")

    logger.info(f"Done: {done}, Failed: {failed}")

    if failed > 0:
        for r in results:
            if r["status"] == "failed":
                logger.error(f"Failed: {r['method']}/{r['dataset']}/seed{r['seed']}: {r['error']}")


if __name__ == "__main__":
    main()
