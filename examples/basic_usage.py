"""Basic citrees classifier, regressor, and forest example."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_WARNINGS", "0")

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
    MaxValuesMethod,
)


def top_features(importances: np.ndarray, n: int = 3) -> list[int]:
    """Return indices of the largest feature-importance values."""
    return np.argsort(importances)[::-1][:n].tolist()


def classification_tree_example() -> None:
    X, y = make_classification(
        n_samples=240,
        n_features=12,
        n_informative=4,
        n_redundant=2,
        random_state=42,
        shuffle=False,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    tree = ConditionalInferenceTreeClassifier(
        selector="mc",
        splitter="gini",
        n_resamples_selector=20,
        n_resamples_splitter=20,
        max_depth=4,
        random_state=42,
        verbose=0,
    )
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_test)

    print("classification tree accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("classification tree top features:", top_features(tree.feature_importances_))


def regression_tree_example() -> None:
    X, y = make_regression(
        n_samples=240,
        n_features=12,
        n_informative=4,
        noise=8.0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    tree = ConditionalInferenceTreeRegressor(
        selector="pc",
        splitter="mse",
        n_resamples_selector=20,
        n_resamples_splitter=20,
        max_depth=4,
        random_state=42,
        verbose=0,
    )
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_test)

    print("regression tree r2:", round(r2_score(y_test, predictions), 3))
    print("regression tree top features:", top_features(tree.feature_importances_))


def forest_ranking_example() -> None:
    X, y = make_classification(
        n_samples=260,
        n_features=16,
        n_informative=5,
        n_redundant=3,
        random_state=7,
        shuffle=False,
    )

    forest = ConditionalInferenceForestClassifier(
        n_estimators=5,
        max_features=MaxValuesMethod.SQRT,
        n_resamples_selector=20,
        n_resamples_splitter=20,
        max_depth=4,
        n_jobs=1,
        random_state=7,
        verbose=0,
    )
    forest.fit(X, y)

    print("forest top features:", top_features(forest.feature_importances_, n=5))


if __name__ == "__main__":
    classification_tree_example()
    regression_tree_example()
    forest_ranking_example()
