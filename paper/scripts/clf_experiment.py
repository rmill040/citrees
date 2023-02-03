"""Classifier experiments."""
import inspect
import json
import os
import time
from copy import deepcopy
from math import ceil
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List

from joblib import delayed, Parallel
import numpy as np
import pandas as pd
from pydantic import BaseModel
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from loguru import logger
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from citrees import ConditionalInferenceForestClassifier, ConditionalInferenceTreeClassifier
from citrees._registry import Registry
from citrees._selector import ClassifierSelectors, ClassifierSelectorTests


HERE = Path(__file__).resolve()
DATA_DIR = HERE.parents[1] / "data"
FILES = [f for f in os.listdir(DATA_DIR) if f.startswith("clf_")]

METHODS = Registry("Methods")
RESULTS = []
RANDOM_STATE = 1718


class Result(BaseModel):
    """Data structure to hold single experiment result."""

    method: str
    hyperparameters: Dict[str, Any] = {}
    feature_ranks: List[int]
    dataset: str
    n_samples: int
    n_features: int
    n_classes: int


def sort_features(*, scores: np.ndarray, higher_is_better: bool) -> List[int]:
    """Sort features based on score."""
    ranks = np.argsort(scores).tolist()
    if higher_is_better:
        ranks = ranks[::-1]
    return ranks


##################
# FILTER METHODS #
##################


def _filter_method_selector(
    *,
    method: str,
    key: str,
    dataset: str,
    n_samples: int,
    n_features: int,
    n_classes: int,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """Filter method as feature selector."""
    scores = np.zeros(n_features)
    for j in range(n_features):
        scores[j] = ClassifierSelectors[key](x=X[:, j], y=y, n_classes=n_classes, random_state=RANDOM_STATE)

    feature_ranks = sort_features(scores=scores, higher_is_better=True)

    RESULTS.append(
        Result(
            method=method,
            hyperparameters={},
            dataset=dataset,
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            feature_ranks=feature_ranks,
        ).dict()
    )


def _filter_permutation_method_selector(
    *,
    method: str,
    key: str,
    dataset: str,
    n_samples: int,
    n_features: int,
    n_classes: int,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """Filter permutation method feature selector."""

    def f(*, x: np.ndarray, y: np.ndarray, hyperparameters: Dict[str, Any]) -> float:
        """Calculate achieved significance level."""
        return ClassifierSelectorTests[key](x=x, y=y, **hyperparameters)

    # Generate parameter combinations
    params = []
    for alpha in [0.10, 0.05, 0.01]:
        for n_resamples in ["minimum", "maximum", "auto"]:
            for early_stopping in [True, False]:
                params.append(
                    {
                        "n_classes": n_classes,
                        "alpha": alpha,
                        "n_resamples": n_resamples,
                        "early_stopping": early_stopping,
                        "random_state": RANDOM_STATE,
                    }
                )

    # Evaluate parameter combinations
    n_params = len(params)
    with Parallel(n_jobs=min(X.shape[1], cpu_count()), verbose=0, backend="loky") as parallel:
        for i, hyperparameters in enumerate(params, 1):
            # Convert string n_resamples into an integer, but make sure to use the string value for results
            _hyperparameters = deepcopy(hyperparameters)
            if hyperparameters["n_resamples"] == "minimum":
                _hyperparameters["n_resamples"] = ceil(1 / hyperparameters["alpha"])
            elif hyperparameters["n_resamples"] == "maximum":
                _hyperparameters["n_resamples"] = ceil(1 / (4 * hyperparameters["alpha"] * hyperparameters["alpha"]))
            else:
                z = norm.ppf(1 - hyperparameters["alpha"])
                _hyperparameters["n_resamples"] = ceil(
                    z * z * (1 - hyperparameters["alpha"]) / hyperparameters["alpha"]
                )

            logger.info(f"Evaluating hyperparameter combination {i}/{n_params}:\n{_hyperparameters}")
            scores = parallel(delayed(f)(x=X[:, j], y=y, hyperparameters=_hyperparameters) for j in range(n_features))
            feature_ranks = sort_features(scores=scores, higher_is_better=False)

            RESULTS.append(
                Result(
                    method=method,
                    hyperparameters=hyperparameters,
                    dataset=dataset,
                    n_samples=n_samples,
                    n_features=n_features,
                    n_classes=n_classes,
                    feature_ranks=feature_ranks,
                )
            )


@METHODS.register("mi")
def mi(*, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray) -> None:
    """Mutual information as feature selector."""
    method = inspect.currentframe().f_code.co_name
    _filter_method_selector(
        method=method,
        key="mi",
        dataset=dataset,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        X=X,
        y=y,
    )


@METHODS.register("mc")
def mc(*, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray) -> None:
    """Multiple correlation as feature selector."""
    method = inspect.currentframe().f_code.co_name
    _filter_method_selector(
        method=method,
        key="mc",
        dataset=dataset,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        X=X,
        y=y,
    )


@METHODS.register("ptest_mi")
def ptest_mi(*, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray) -> None:
    """Permutation testing with mutual information as feature selector."""
    method = inspect.currentframe().f_code.co_name
    _filter_permutation_method_selector(
        method=method,
        key="mi",
        dataset=dataset,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        X=X,
        y=y,
    )


@METHODS.register("ptest_mc")
def ptest_mc(*, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray) -> None:
    """Permutation testing with multiple correlation as feature selector."""
    method = inspect.currentframe().f_code.co_name
    _filter_permutation_method_selector(
        method=method,
        key="mc",
        dataset=dataset,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        X=X,
        y=y,
    )


@METHODS.register("ptest_hybrid")
def ptest_hybrid(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    """Permutation testing with multiple correlation or mutual information as feature selector."""
    method = inspect.currentframe().f_code.co_name
    _filter_permutation_method_selector(
        method=method,
        key="hybrid",
        dataset=dataset,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        X=X,
        y=y,
    )


####################
# EMBEDDED METHODS #
####################


def _embedding_method(
    *,
    n_jobs: int,
    estimator: Any,
    method: str,
    params: Dict[str, Any],
    dataset: str,
    n_samples: int,
    n_features: int,
    n_classes: int,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """Embedding method as feature selector."""

    def f(
        *, X: np.ndarray, y: np.ndarray, estimator: Any, hyperparameters: Dict[str, Any], n_params: int, i: int
    ) -> List[int]:
        """Rank features using estimator."""
        logger.info(f"Evaluating hyperparameter combination {i}/{n_params}:\n{hyperparameters}")
        clf = estimator(**hyperparameters).fit(X, y)
        if hasattr(clf, "feature_importances_"):
            scores = clf.feature_importances_
        else:
            scores = abs(clf.coef_.ravel())
        return sort_features(scores=scores, higher_is_better=False)

    n_params = len(params)
    results = Parallel(n_jobs=n_jobs, verbose=0, backend="loky")(
        delayed(f)(X=X, y=y, estimator=estimator, hyperparameters=hyperparameters, n_params=n_params, i=i)
        for i, hyperparameters in enumerate(params, 1)
    )

    for feature_ranks, hyperparameters in zip(results, params):
        RESULTS.append(
            Result(
                method=method,
                hyperparameters=hyperparameters,
                dataset=dataset,
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                feature_ranks=feature_ranks,
            )
        )


@METHODS.register("lr")
def lr(*, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray) -> None:
    """Logistic regression for feature selection."""
    method = inspect.currentframe().f_code.co_name
    params = []
    for class_weight in [None, "balanced"]:
        params.append(
            {
                "penalty": None,
                "random_state": RANDOM_STATE,
                "class_weight": class_weight,
                "solver": "lbfgs",
            }
        )

    _embedding_method(
        estimator=LogisticRegression,
        n_jobs=min(len(params), cpu_count()),
        method=method,
        params=params,
        dataset=dataset,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        X=X,
        y=y,
    )


@METHODS.register("lr_l1")
def lr_l1(*, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray) -> None:
    """Logistic regression with L1 norm as feature selector."""
    method = inspect.currentframe().f_code.co_name
    params = []
    for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        for class_weight in [None, "balanced"]:
            params.append(
                {
                    "C": C,
                    "penalty": "l1",
                    "random_state": RANDOM_STATE,
                    "class_weight": class_weight,
                    "solver": "liblinear",
                }
            )

    _embedding_method(
        estimator=LogisticRegression,
        n_jobs=min(len(params), cpu_count()),
        method=method,
        params=params,
        dataset=dataset,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        X=X,
        y=y,
    )


@METHODS.register("lr_l2")
def lr_l2(*, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray) -> None:
    """Logistic regression with L2 norm as feature selector."""
    method = inspect.currentframe().f_code.co_name
    params = []
    for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        for class_weight in [None, "balanced"]:
            params.append(
                {
                    "C": C,
                    "penalty": "l2",
                    "random_state": RANDOM_STATE,
                    "class_weight": class_weight,
                    "solver": "liblinear",
                }
            )

    _embedding_method(
        estimator=LogisticRegression,
        n_jobs=min(len(params), cpu_count()),
        method=method,
        params=params,
        dataset=dataset,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        X=X,
        y=y,
    )


@METHODS.register("xgb")
def xgb(*, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray) -> None:
    """XGBOOST classifier as feature selector."""
    method = inspect.currentframe().f_code.co_name

    params = []
    for max_depth in [1, 3, 5]:
        for learning_rate in [0.01, 0.1]:
            for subsample in [0.8, 1.0]:
                for colsample_bytree in [0.8, 1.0]:
                    for reg_alpha in [0.01, None]:
                        for reg_lambda in [0.01, None]:
                            params.append(
                                {
                                    "max_depth": max_depth,
                                    "learning_rate": learning_rate,
                                    "subsample": subsample,
                                    "colsample_bytree": colsample_bytree,
                                    "reg_alpha": reg_alpha,
                                    "reg_lambda": reg_lambda,
                                    "n_jobs": 1,
                                    "random_state": RANDOM_STATE,
                                    "importance_type": "gain",
                                }
                            )

    _embedding_method(
        estimator=XGBClassifier,
        n_jobs=-1,
        method=method,
        params=params,
        dataset=dataset,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        X=X,
        y=y,
    )


@METHODS.register("lightgbm")
def lightgbm(*, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray) -> None:
    """LightGBM classifier as feature selector."""
    method = inspect.currentframe().f_code.co_name


@METHODS.register("catboost")
def catboost(*, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray) -> None:
    """CatBoost classifier as feature selector."""
    method = inspect.currentframe().f_code.co_name


@METHODS.register("dt")
def dt(*, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray) -> None:
    """Decision tree classifier as feature selector."""
    method = inspect.currentframe().f_code.co_name


@METHODS.register("rt")
def rt(*, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray) -> None:
    """Random decision tree classifier as feature selector."""
    method = inspect.currentframe().f_code.co_name


@METHODS.register("rf")
def rf(*, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray) -> None:
    """Random forest classifier as feature selector."""
    method = inspect.currentframe().f_code.co_name


@METHODS.register("et")
def et(*, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray) -> None:
    """Extra trees classifier as feature selector."""
    method = inspect.currentframe().f_code.co_name


@METHODS.register("cit")
def cit(*, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray) -> None:
    """Conditional inference tree classifier as feature selector."""
    method = inspect.currentframe().f_code.co_name

    params = []
    for adjust_alpha_selector in [True, False]:
        for adjust_alpha_splitter in [True, False]:
            for n_resamples_selector in ["minimum", "maximum", "auto"]:
                for n_resamples_splitter in ["minimum", "maximum", "auto"]:
                    for threshold_method in ["exact", "random", "percentile", "histogram"]:
                        hyperparameters = {
                            "adjust_alpha_selector": adjust_alpha_selector,
                            "adjust_alpha_splitter": adjust_alpha_splitter,
                            "n_resamples_selector": n_resamples_selector,
                            "n_resamples_splitter": n_resamples_splitter,
                            "threshold_method": threshold_method,
                            "random_state": RANDOM_STATE,
                        }
                        if threshold_method == "exact":
                            for max_thresholds in [None]:
                                hyperparameters = deepcopy(hyperparameters)
                                hyperparameters["max_thresholds"] = max_thresholds
                                params.append(hyperparameters)
                        elif threshold_method == "random":
                            for max_thresholds in [0.8]:
                                hyperparameters = deepcopy(hyperparameters)
                                hyperparameters["max_thresholds"] = max_thresholds
                                params.append(hyperparameters)
                        elif threshold_method == "percentile":
                            for max_thresholds in [0.8]:
                                hyperparameters = deepcopy(hyperparameters)
                                hyperparameters["max_thresholds"] = max_thresholds
                                params.append(hyperparameters)
                        else:
                            for max_thresholds in [0.8]:
                                hyperparameters = deepcopy(hyperparameters)
                                hyperparameters["max_thresholds"] = max_thresholds
                                params.append(hyperparameters)

    _embedding_method(
        estimator=ConditionalInferenceTreeClassifier,
        n_jobs=-1,
        method=method,
        params=params,
        dataset=dataset,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        X=X,
        y=y,
    )


@METHODS.register("cif")
def cif(*, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray) -> None:
    """Conditional inference forest classifier as feature selector."""
    method = inspect.currentframe().f_code.co_name


def main() -> None:
    """Classifier experiments."""
    n_files = len(FILES)
    for j, f in enumerate(FILES, 1):
        X = pd.read_parquet(os.path.join(DATA_DIR, f))
        y = X.pop("y").astype(int).values
        X = X.astype(float).values

        # Standardize features
        X = StandardScaler().fit_transform(X)

        dataset = f.replace("clf_", "").replace(".snappy.parquet", "")
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        logger.info("=" * 100)
        logger.info(
            f"Processing dataset {dataset} ({j}/{n_files}): # Samples = {n_samples} | # Features = {n_features} | "
            f"# Classes = {n_classes}"
        )
        logger.info("=" * 100)

        for key in METHODS.keys():
            logger.info(f"Running feature selection method ({key})")

            tic = time.time()

            METHODS[key](dataset=dataset, n_samples=n_samples, n_features=n_features, n_classes=n_classes, X=X, y=y)

            total = round((time.time() - tic) / 60, 2)
            logger.info(f"Finished feature selection method in ({total}) minutes")


if __name__ == "__main__":
    main()
