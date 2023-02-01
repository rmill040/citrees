"""Classifier experiments."""
import inspect
import os
import time
from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Any, Dict, List

from joblib import delayed, Parallel
import numpy as np
import pandas as pd
from pydantic import BaseModel
from boruta import BorutaPy
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from loguru import logger
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
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


def sort_features(*, scores: np.ndarray, higher_is_better: bool) -> List[int]:
    """Sort features based on score."""
    ranks = np.argsort(scores).tolist()
    if higher_is_better:
        ranks = ranks[::-1]
    return ranks


class Result(BaseModel):
    """Data structure to hold single experiment result."""

    method: str
    hyperparameters: Dict[str, Any] = {}
    feature_ranks: List[int]
    dataset: str
    n_samples: int
    n_features: int
    n_classes: int


##################
# FILTER METHODS #
##################


@METHODS.register("mutual_information_selector")
def mutual_information_selector(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    """Mutual information feature selection."""

    method = inspect.currentframe().f_code.co_name

    scores = np.zeros(n_features)
    for j in range(n_features):
        scores[j] = ClassifierSelectors["mi"](x=X[:, j], y=y, n_classes=n_classes, random_state=RANDOM_STATE)

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


@METHODS.register("multiple_correlation_selector")
def multiple_correlation_selector(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    """Multiple correlation feature selection."""
    
    method = inspect.currentframe().f_code.co_name

    scores = np.zeros(n_features)
    for j in range(n_features):
        scores[j] = ClassifierSelectors["mc"](x=X[:, j], y=y, n_classes=n_classes, random_state=RANDOM_STATE)

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
    

@METHODS.register("permutation_test_mutual_information_selector")
def permutation_test_mutual_information_selector(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    """Permutation testing with mutual information for feature selection."""

    def f(x, y, hyperparameters):
        return ClassifierSelectorTests["mi"](x=x, y=y, **hyperparameters)

    method = inspect.currentframe().f_code.co_name

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
    with Parallel(n_jobs=1, verbose=0, backend="loky") as parallel:
        for i, hyperparameters in enumerate(params, 1):
            # Convert string n_resamples into an integer, but make sure to use the string value for results
            _hyperparameters = deepcopy(hyperparameters)
            if hyperparameters["n_resamples"] == "minimum":
                _hyperparameters["n_resamples"] = ceil(1 / hyperparameters["alpha"])
            elif hyperparameters["n_resamples"] == "maximum":
                _hyperparameters["n_resamples"] = ceil(1 / (4 * hyperparameters["alpha"] * hyperparameters["alpha"]))
            else:
                z = norm.ppf(1 - hyperparameters["alpha"])
                _hyperparameters["n_resamples"] = ceil(z * z * (1 - hyperparameters["alpha"]) / hyperparameters["alpha"])

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


@METHODS.register("permutation_test_multiple_correlation_selector")
def permutation_test_multiple_correlation_selector(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    """Permutation testing with multiple correlation for feature selection."""

    def f(x, y, hyperparameters):
        return ClassifierSelectorTests["mc"](x=x, y=y, **hyperparameters)

    method = inspect.currentframe().f_code.co_name

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
    with Parallel(n_jobs=1, verbose=0, backend="loky") as parallel:
        for i, hyperparameters in enumerate(params, 1):
            # Convert string n_resamples into an integer, but make sure to use the string value for results
            _hyperparameters = deepcopy(hyperparameters)
            if hyperparameters["n_resamples"] == "minimum":
                _hyperparameters["n_resamples"] = ceil(1 / hyperparameters["alpha"])
            elif hyperparameters["n_resamples"] == "maximum":
                _hyperparameters["n_resamples"] = ceil(1 / (4 * hyperparameters["alpha"] * hyperparameters["alpha"]))
            else:
                z = norm.ppf(1 - hyperparameters["alpha"])
                _hyperparameters["n_resamples"] = ceil(z * z * (1 - hyperparameters["alpha"]) / hyperparameters["alpha"])

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


@METHODS.register("permutation_test_hybrid_classifier")
def permutation_test_hybrid_classifier(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    """Permutation testing with multiple correlation or mutual information for feature selection."""

    def f(x, y, hyperparameters):
        return ClassifierSelectorTests["hybrid"](x=x, y=y, **hyperparameters)

    method = inspect.currentframe().f_code.co_name

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
    with Parallel(n_jobs=1, verbose=0, backend="loky") as parallel:
        for i, hyperparameters in enumerate(params, 1):
            # Convert string n_resamples into an integer, but make sure to use the string value for results
            _hyperparameters = deepcopy(hyperparameters)
            if hyperparameters["n_resamples"] == "minimum":
                _hyperparameters["n_resamples"] = ceil(1 / hyperparameters["alpha"])
            elif hyperparameters["n_resamples"] == "maximum":
                _hyperparameters["n_resamples"] = ceil(1 / (4 * hyperparameters["alpha"] * hyperparameters["alpha"]))
            else:
                z = norm.ppf(1 - hyperparameters["alpha"])
                _hyperparameters["n_resamples"] = ceil(z * z * (1 - hyperparameters["alpha"]) / hyperparameters["alpha"])

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


###################
# WRAPPER METHODS #
###################


@METHODS.register("recursive_feature_elimination")
def recursive_feature_elimination(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    pass


@METHODS.register("permutation_importance_selector")
def permutation_importance_selector(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    pass


@METHODS.register("boruta_selector")
def boruta_selector(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    pass


####################
# EMBEDDED METHODS #
####################


@METHODS.register("logistic_regression_l1")
def logistic_regression_l1(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    pass


@METHODS.register("logistic_regression_l2")
def logistic_regression_l2(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    pass


@METHODS.register("logistic_regression_l1l2")
def logistic_regression_l1l2(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    pass


@METHODS.register("xgboost_classifier")
def xgboost_classifier(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    pass


@METHODS.register("lightgbm_classifier")
def lightgbm_classifier(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    pass


@METHODS.register("catboost_classifier")
def catboost_classifier(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    pass


@METHODS.register("decision_tree_classifier")
def decision_tree_classifier(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    pass


@METHODS.register("random_tree_classifier")
def random_tree_classifier(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    pass


@METHODS.register("random_forest_classifier")
def random_forest_classifier(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    pass


@METHODS.register("extra_trees_classifier")
def extra_trees_classifier(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    pass


@METHODS.register("conditional_inference_tree_classifier")
def conditional_inference_tree_classifier(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    pass


@METHODS.register("conditonal_inference_forest_classifier")
def conditonal_inference_forest_classifier(
    *, dataset: str, n_samples: int, n_features: int, n_classes: int, X: np.ndarray, y: np.ndarray
) -> None:
    pass


def main() -> None:
    """Main script to run experiments for classifiers."""
    for f in FILES:
        X = pd.read_parquet(os.path.join(DATA_DIR, f))
        y = X.pop("y").astype(int).values
        X = X.astype(float).values
        
        # FOR TESTING
        # if X.shape[1] > 3:
        #     X = X[:, :3]

        dataset = f.replace("clf_", "").replace(".snappy.parquet", "")
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        logger.info(
            f"Processing dataset {dataset}: # Samples = {n_samples} | # Features = {n_features} | # Classes = "
            f"{n_classes}"
        )

        for key in METHODS.keys():
            logger.info(f"Running feature selection method ({key})")
            
            tic = time.time()
            
            METHODS[key](
                dataset=dataset, n_samples=n_samples, n_features=n_features, n_classes=n_classes, X=X, y=y
            )
            
            total = round((time.time() - tic) / 60, 2)
            logger.info(f"Finished feature selection method in ({total}) minutes")


if __name__ == "__main__":
    main()
