"""Ray-based distributed feature selection."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
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
from sklearn.model_selection import KFold, StratifiedKFold
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
from paper.scripts.constants import CLF_METHODS, N_SPLITS, REG_METHODS
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
s3 = boto3.client("s3", region_name=config.region)


def get_datasets(task_type: str) -> list[str]:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    prefix = "clf_" if task_type == "classification" else "reg_"
    datasets = []
    for f in data_dir.glob(f"{prefix}*.parquet"):
        name = f.stem.replace(prefix, "").replace(".snappy", "")
        datasets.append(name)
    return sorted(datasets)


def load_dataset(name: str, task_type: str) -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    prefix = "clf_" if task_type == "classification" else "reg_"
    df = pd.read_parquet(data_dir / f"{prefix}{name}.parquet")
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
        s3.head_object(Bucket=parts[0], Key=parts[1])
        return True
    except Exception:
        return False


def upload_to_s3(data: list[dict], s3_path: str) -> None:
    df = pd.DataFrame(data)
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    parts = s3_path.replace("s3://", "").split("/", 1)
    s3.put_object(Bucket=parts[0], Key=parts[1], Body=buffer.getvalue())


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
        for j in range(n_features):
            scores[j] = selector_fn(X[:, j], y, standardize=True, random_state=rng.integers(0, 2**31))

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
            pvalues[j] = test_fn(X[:, j], y, n_classes, alpha=0.05, n_resamples=1000, early_stopping=None, random_state=rs)
    else:
        for j in range(n_features):
            rs = rng.integers(0, 2**31)
            scores[j] = selector_fn(X[:, j], y, standardize=True, random_state=rs)
            pvalues[j] = test_fn(X[:, j], y, standardize=True, alpha=0.05, n_resamples=1000, early_stopping=None, random_state=rs)

    return np.lexsort((-scores, pvalues))


def get_embedding_model(method: str, task_type: str, random_state: int):
    if task_type == "classification":
        models = {
            "rf": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state),
            "et": ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=random_state),
            "cit": ConditionalInferenceTreeClassifier(random_state=random_state),
            "cif": ConditionalInferenceForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state),
        }
        if HAS_XGB:
            models["xgb"] = XGBClassifier(n_estimators=100, n_jobs=-1, random_state=random_state, verbosity=0)
        if HAS_LGBM:
            models["lgbm"] = LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=random_state, verbosity=-1)
        if HAS_CAT:
            models["cat"] = CatBoostClassifier(n_estimators=100, random_state=random_state, verbose=0)
    else:
        models = {
            "rf": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=random_state),
            "et": ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=random_state),
            "cit": ConditionalInferenceTreeRegressor(random_state=random_state),
            "cif": ConditionalInferenceForestRegressor(n_estimators=100, n_jobs=-1, random_state=random_state),
        }
        if HAS_XGB:
            models["xgb"] = XGBRegressor(n_estimators=100, n_jobs=-1, random_state=random_state, verbosity=0)
        if HAS_LGBM:
            models["lgbm"] = LGBMRegressor(n_estimators=100, n_jobs=-1, random_state=random_state, verbosity=-1)
        if HAS_CAT:
            models["cat"] = CatBoostRegressor(n_estimators=100, random_state=random_state, verbose=0)

    return models[method]


def embedding_selector(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                       method: str, task_type: str, random_state: int) -> tuple[np.ndarray, dict[str, Any]]:
    model = get_embedding_model(method, task_type, random_state)
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


def boruta_selector(X_train: np.ndarray, y_train: np.ndarray, task_type: str, random_state: int) -> np.ndarray:
    if task_type == "classification":
        base_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state)
    else:
        base_model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=random_state)

    boruta = BorutaPy(base_model, n_estimators="auto", random_state=random_state, verbose=0)
    boruta.fit(X_train, y_train)
    return np.argsort(boruta.ranking_)


def pi_selector(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                task_type: str, random_state: int) -> np.ndarray:
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state)
        scoring = "accuracy"
    else:
        model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=random_state)
        scoring = "r2"

    model.fit(X_train, y_train)
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=random_state, scoring=scoring, n_jobs=-1)
    return np.argsort(result.importances_mean)[::-1]


def shap_selector(X_train: np.ndarray, y_train: np.ndarray, task_type: str, random_state: int) -> np.ndarray:
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state)
    else:
        model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=random_state)

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


def mrmr_selector(X_train: np.ndarray, y_train: np.ndarray, task_type: str) -> np.ndarray:
    df = pd.DataFrame(X_train)
    y_series = pd.Series(y_train)
    n_features = X_train.shape[1]

    if task_type == "classification":
        selected = mrmr_classif(df, y_series, K=n_features, show_progress=False)
    else:
        selected = mrmr_regression(df, y_series, K=n_features, show_progress=False)

    return np.array(selected)


def rfe_selector(X_train: np.ndarray, y_train: np.ndarray, task_type: str, random_state: int) -> np.ndarray:
    if task_type == "classification":
        base_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state)
    else:
        base_model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=random_state)

    rfe = RFE(base_model, n_features_to_select=1, step=1)
    rfe.fit(X_train, y_train)
    return np.argsort(rfe.ranking_)


def run_selection(X: np.ndarray, y: np.ndarray, method: str, task_type: str, seed: int) -> list[dict[str, Any]]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if task_type == "classification":
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    results = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        rs = seed + fold_idx
        embedding_data = None

        if method in ["mc", "mi", "rdc", "pc", "dc"]:
            ranking = filter_selector(X_train, y_train, method, task_type, rs)
        elif method.startswith("ptest_"):
            ranking = permutation_selector(X_train, y_train, method, task_type, rs)
        elif method in ["rf", "et", "xgb", "lgbm", "cat", "cit", "cif"]:
            ranking, embedding_data = embedding_selector(X_train, y_train, X_test, y_test, method, task_type, rs)
        elif method == "boruta":
            ranking = boruta_selector(X_train, y_train, task_type, rs)
        elif method == "pi":
            ranking = pi_selector(X_train, y_train, X_test, y_test, task_type, rs)
        elif method == "shap":
            ranking = shap_selector(X_train, y_train, task_type, rs)
        elif method == "mrmr":
            ranking = mrmr_selector(X_train, y_train, task_type)
        elif method == "rfe":
            ranking = rfe_selector(X_train, y_train, task_type, rs)
        else:
            raise ValueError(f"Unknown method: {method}")

        fold_result = {
            "fold_idx": fold_idx,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "feature_ranking": ranking.tolist(),
        }
        if embedding_data:
            fold_result.update({f"embedding_{k}": v for k, v in embedding_data.items()})

        results.append(fold_result)

    return results


@ray.remote(resources={"selection": 1})
def process_config(method: str, dataset: str, seed: int, task_type: str) -> dict[str, Any]:
    s3_path = f"s3://{config.bucket_name}/rankings/{task_type}/{dataset}/{method}_seed{seed}.parquet"

    if s3_file_exists(s3_path):
        return {"status": "skipped", "method": method, "dataset": dataset, "seed": seed}

    try:
        X, y = load_dataset(dataset, task_type)
        tic = time.perf_counter()
        results = run_selection(X, y, method, task_type, seed)
        elapsed = time.perf_counter() - tic
        upload_to_s3(results, s3_path)
        return {"status": "done", "method": method, "dataset": dataset, "seed": seed, "elapsed": elapsed}
    except Exception as e:
        return {"status": "failed", "method": method, "dataset": dataset, "seed": seed, "error": str(e)}


def main():
    ray.init(address="auto", ignore_reinit_error=True)

    task_type = config.experiment.type
    methods = CLF_METHODS if task_type == "classification" else REG_METHODS
    datasets = get_datasets(task_type)
    n_seeds = config.experiment.n_seeds

    configs = [(m, d, s) for m in methods for d in datasets for s in range(n_seeds)]
    logger.info(f"Submitting {len(configs)} configs ({len(methods)} methods × {len(datasets)} datasets × {n_seeds} seeds)")

    futures = [process_config.remote(m, d, s, task_type) for m, d, s in configs]
    results = ray.get(futures)

    done = sum(1 for r in results if r["status"] == "done")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")

    logger.info(f"Done: {done}, Skipped: {skipped}, Failed: {failed}")

    if failed > 0:
        for r in results:
            if r["status"] == "failed":
                logger.error(f"Failed: {r['method']}/{r['dataset']}/seed{r['seed']}: {r['error']}")


if __name__ == "__main__":
    main()
