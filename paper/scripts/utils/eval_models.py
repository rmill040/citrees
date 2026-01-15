"""Shared classifier definitions for experiments.

This module provides standardized classifier configurations used across
synthetic and real dataset experiments.
"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from paper.scripts.utils.constants import RANDOM_STATE


def get_clf_models() -> dict[str, Any]:
    """Get classification downstream models.

    Returns
    -------
    dict[str, Any]
        Dictionary mapping model names to sklearn classifiers.
        - lr: Logistic Regression with class balancing
        - svm: SVC with probability estimates and class balancing
        - knn: k-Nearest Neighbors with distance weighting
    """
    return {
        "lr": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "svm": SVC(class_weight="balanced", probability=True, random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    }


def get_reg_models() -> dict[str, Any]:
    """Get regression downstream models.

    Returns
    -------
    dict[str, Any]
        Dictionary mapping model names to sklearn regressors.
        - ridge: Ridge regression with L2 regularization
        - svr: Support Vector Regression
        - knn: k-Nearest Neighbors regressor with distance weighting
    """
    return {
        "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "svr": SVR(),
        "knn": KNeighborsRegressor(n_neighbors=5, weights="distance"),
    }
