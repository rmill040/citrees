"""Shared classifier definitions for experiments.

This module provides standardized classifier configurations used across
synthetic and real dataset experiments.

Note: The ray_eval.py script has its own inline model factories that accept
random_state and n_jobs parameters. This module provides simpler versions
for other scripts that may need them.
"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR


def get_clf_models(random_state: int, *, n_jobs: int = 1) -> dict[str, Any]:
    """Get classification downstream models.

    Parameters
    ----------
    random_state : int
        Random seed for reproducibility.
    n_jobs : int, default=1
        Number of parallel jobs for KNN.

    Returns
    -------
    dict[str, Any]
        Dictionary mapping model names to sklearn classifiers.
        - lr: Logistic Regression with class balancing
        - svm: SVC with probability estimates and class balancing
        - knn: k-Nearest Neighbors with distance weighting
    """
    return {
        "lr": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "svm": SVC(class_weight="balanced", probability=True, random_state=random_state),
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=n_jobs),
    }


def get_reg_models(random_state: int, *, n_jobs: int = 1) -> dict[str, Any]:
    """Get regression downstream models.

    Parameters
    ----------
    random_state : int
        Random seed for reproducibility.
    n_jobs : int, default=1
        Number of parallel jobs for KNN.

    Returns
    -------
    dict[str, Any]
        Dictionary mapping model names to sklearn regressors.
        - ridge: Ridge regression with L2 regularization
        - svr: Support Vector Regression
        - knn: k-Nearest Neighbors regressor with distance weighting
    """
    return {
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "svr": SVR(),
        "knn": KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=n_jobs),
    }
