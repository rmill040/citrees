"""Tests for paper/scripts/pipeline/stage2.py (evaluation)."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from paper.scripts.pipeline.stage2 import compute_roc_auc, evaluate_fold

pytestmark = pytest.mark.paper


class TestComputeRocAuc:
    """Tests for compute_roc_auc function."""

    def test_binary_labels(self):
        """Test ROC AUC computation with binary labels {1, 2}."""
        y_true = np.array([1, 2, 1, 2, 2, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
        classes = np.array([1, 2])

        expected = compute_roc_auc(y_true, y_proba, classes)
        # Manually binarize using class 2 as positive
        y_bin = (y_true == 2).astype(int)
        manual = roc_auc_score(y_bin, y_proba)

        assert np.isfinite(expected)
        assert expected == manual

    def test_single_class_returns_nan(self):
        """Test that single class in y_true returns NaN."""
        y_true = np.array([1, 1, 1, 1])
        y_proba = np.array([0.2, 0.3, 0.1, 0.4])
        classes = np.array([1, 2])

        result = compute_roc_auc(y_true, y_proba, classes)
        assert np.isnan(result)

    def test_multiclass_missing_class_returns_nan(self):
        """Test that missing class in multiclass returns NaN."""
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.6, 0.3, 0.1],
                [0.2, 0.7, 0.1],
            ]
        )
        classes = np.array([0, 1, 2])

        result = compute_roc_auc(y_true, y_proba, classes)
        assert np.isnan(result)


class TestEvaluateFold:
    """Tests for evaluate_fold function."""

    def test_classification_metrics_schema(self):
        """Test that classification evaluation returns required metrics."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 5))
        y = (X[:, 0] + rng.normal(scale=0.1, size=40) > 0).astype(int)

        X_train, X_test = X[:30], X[30:]
        y_train, y_test = y[:30], y[30:]
        ranking = np.arange(X.shape[1])

        results = evaluate_fold(
            X_train,
            y_train,
            X_test,
            y_test,
            ranking,
            task="classification",
            random_state=0,
            n_jobs=1,
        )

        row = results[0]
        required = {
            "accuracy",
            "f1",
            "f1_macro",
            "balanced_accuracy",
            "roc_auc",
            "auc",
        }
        missing = required - set(row.keys())
        assert not missing, f"Missing metrics: {missing}"

    def test_regression_metrics_schema(self):
        """Test that regression evaluation returns required metrics."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 5))
        y = X[:, 0] * 0.5 + rng.normal(scale=0.1, size=40)

        X_train, X_test = X[:30], X[30:]
        y_train, y_test = y[:30], y[30:]
        ranking = np.arange(X.shape[1])

        results = evaluate_fold(
            X_train,
            y_train,
            X_test,
            y_test,
            ranking,
            task="regression",
            random_state=0,
            n_jobs=1,
        )

        row = results[0]
        required = {"r2", "mse", "rmse", "mae"}
        missing = required - set(row.keys())
        assert not missing, f"Missing metrics: {missing}"
