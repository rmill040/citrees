"""Tests for citrees._conformal module."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from citrees import (
    CQR,
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
    ConformalClassifier,
    ConformalRegressor,
)


@pytest.fixture
def classification_data():
    """Generate classification dataset."""
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    return X, y


@pytest.fixture
def regression_data():
    """Generate regression dataset."""
    X, y = make_regression(
        n_samples=300,
        n_features=10,
        n_informative=5,
        noise=10.0,
        random_state=42,
    )
    return X, y


class TestConformalClassifier:
    """Tests for ConformalClassifier."""

    def test_fit_predict(self, classification_data):
        """Test basic fit and predict."""
        X, y = classification_data
        base = ConditionalInferenceForestClassifier(n_estimators=10, random_state=42, verbose=0)
        clf = ConformalClassifier(base, alpha=0.1, random_state=42)
        clf.fit(X, y)

        # Check fitted attributes
        assert hasattr(clf, "estimator_")
        assert hasattr(clf, "qhat_")
        assert hasattr(clf, "classes_")

        # Check predictions
        preds = clf.predict(X)
        assert preds.shape == y.shape

    def test_predict_set(self, classification_data):
        """Test prediction set generation."""
        X, y = classification_data
        base = ConditionalInferenceForestClassifier(n_estimators=10, random_state=42, verbose=0)
        clf = ConformalClassifier(base, alpha=0.1, random_state=42)
        clf.fit(X, y)

        pred_sets = clf.predict_set(X)
        assert len(pred_sets) == len(X)
        assert all(isinstance(ps, set) for ps in pred_sets)
        # Each set should contain at least one class
        assert all(len(ps) >= 1 for ps in pred_sets)

    def test_coverage_guarantee(self, classification_data):
        """Test coverage is approximately 1-alpha."""
        X, y = classification_data
        base = ConditionalInferenceForestClassifier(n_estimators=20, random_state=42, verbose=0)
        alpha = 0.1
        clf = ConformalClassifier(base, alpha=alpha, random_state=42)
        clf.fit(X, y)

        coverage = clf.coverage(X, y)
        # Coverage should be at least 1-alpha (with some tolerance for randomness)
        assert coverage >= (1 - alpha) - 0.05, f"Coverage {coverage} too low"

    def test_average_set_size(self, classification_data):
        """Test average set size calculation."""
        X, y = classification_data
        base = ConditionalInferenceForestClassifier(n_estimators=10, random_state=42, verbose=0)
        clf = ConformalClassifier(base, alpha=0.1, random_state=42)
        clf.fit(X, y)

        avg_size = clf.average_set_size(X)
        assert avg_size >= 1.0
        assert avg_size <= len(np.unique(y))

    def test_predict_proba(self, classification_data):
        """Test predict_proba passthrough."""
        X, y = classification_data
        base = ConditionalInferenceForestClassifier(n_estimators=10, random_state=42, verbose=0)
        clf = ConformalClassifier(base, alpha=0.1, random_state=42)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_invalid_estimator(self, classification_data):
        """Test that estimator without predict_proba raises error."""
        from sklearn.svm import SVC

        X, y = classification_data
        # SVC without probability=True doesn't have predict_proba
        base = SVC()
        with pytest.raises(ValueError, match="predict_proba"):
            ConformalClassifier(base, alpha=0.1)

    def test_invalid_alpha(self, classification_data):
        """Test invalid alpha values."""
        X, y = classification_data
        base = ConditionalInferenceForestClassifier(n_estimators=5, verbose=0)

        with pytest.raises(ValueError, match="alpha"):
            ConformalClassifier(base, alpha=0.0)

        with pytest.raises(ValueError, match="alpha"):
            ConformalClassifier(base, alpha=1.0)

    def test_different_alpha_values(self, classification_data):
        """Test that different alpha values produce different set sizes."""
        X, y = classification_data
        base = ConditionalInferenceForestClassifier(n_estimators=15, random_state=42, verbose=0)

        clf_low = ConformalClassifier(base, alpha=0.05, random_state=42)
        clf_low.fit(X, y)
        size_low = clf_low.average_set_size(X)

        clf_high = ConformalClassifier(base, alpha=0.20, random_state=42)
        clf_high.fit(X, y)
        size_high = clf_high.average_set_size(X)

        # Lower alpha (higher confidence) should give larger sets
        assert size_low >= size_high


class TestConformalRegressor:
    """Tests for ConformalRegressor."""

    def test_fit_predict(self, regression_data):
        """Test basic fit and predict."""
        X, y = regression_data
        base = ConditionalInferenceForestRegressor(n_estimators=10, random_state=42, verbose=0)
        reg = ConformalRegressor(base, alpha=0.1, random_state=42)
        reg.fit(X, y)

        # Check fitted attributes
        assert hasattr(reg, "estimator_")
        assert hasattr(reg, "qhat_")

        # Check predictions
        preds = reg.predict(X)
        assert preds.shape == y.shape

    def test_predict_interval(self, regression_data):
        """Test prediction interval generation."""
        X, y = regression_data
        base = ConditionalInferenceForestRegressor(n_estimators=10, random_state=42, verbose=0)
        reg = ConformalRegressor(base, alpha=0.1, random_state=42)
        reg.fit(X, y)

        lower, upper = reg.predict_interval(X)
        assert lower.shape == y.shape
        assert upper.shape == y.shape
        # Upper should be greater than lower
        assert (upper > lower).all()

    def test_coverage_guarantee(self, regression_data):
        """Test coverage is approximately 1-alpha."""
        X, y = regression_data
        base = ConditionalInferenceForestRegressor(n_estimators=20, random_state=42, verbose=0)
        alpha = 0.1
        reg = ConformalRegressor(base, alpha=alpha, random_state=42)
        reg.fit(X, y)

        coverage = reg.coverage(X, y)
        # Coverage should be at least 1-alpha (with some tolerance)
        assert coverage >= (1 - alpha) - 0.05, f"Coverage {coverage} too low"

    def test_average_interval_width(self, regression_data):
        """Test average interval width calculation."""
        X, y = regression_data
        base = ConditionalInferenceForestRegressor(n_estimators=10, random_state=42, verbose=0)
        reg = ConformalRegressor(base, alpha=0.1, random_state=42)
        reg.fit(X, y)

        width = reg.average_interval_width(X)
        assert width > 0

    def test_invalid_alpha(self, regression_data):
        """Test invalid alpha values."""
        X, y = regression_data
        base = ConditionalInferenceForestRegressor(n_estimators=5, verbose=0)

        with pytest.raises(ValueError, match="alpha"):
            ConformalRegressor(base, alpha=0.0)

        with pytest.raises(ValueError, match="alpha"):
            ConformalRegressor(base, alpha=1.5)


class TestCQR:
    """Tests for Conformalized Quantile Regression."""

    def test_fit_predict(self, regression_data):
        """Test basic fit and predict."""
        X, y = regression_data
        base = ConditionalInferenceForestRegressor(n_estimators=10, random_state=42, verbose=0)
        cqr = CQR(base, alpha=0.1, random_state=42)
        cqr.fit(X, y)

        # Check fitted attributes
        assert hasattr(cqr, "estimator_")
        assert hasattr(cqr, "qhat_")

    def test_predict_interval(self, regression_data):
        """Test CQR prediction intervals."""
        X, y = regression_data
        base = ConditionalInferenceForestRegressor(n_estimators=10, random_state=42, verbose=0)
        cqr = CQR(base, alpha=0.1, random_state=42)
        cqr.fit(X, y)

        lower, upper = cqr.predict_interval(X)
        assert lower.shape == y.shape
        assert upper.shape == y.shape
        assert (upper > lower).all()

    def test_coverage(self, regression_data):
        """Test CQR coverage."""
        X, y = regression_data
        base = ConditionalInferenceForestRegressor(n_estimators=20, random_state=42, verbose=0)
        alpha = 0.1
        cqr = CQR(base, alpha=alpha, random_state=42)
        cqr.fit(X, y)

        coverage = cqr.coverage(X, y)
        # Coverage should be at least 1-alpha
        assert coverage >= (1 - alpha) - 0.05

    def test_adaptive_intervals(self, regression_data):
        """Test that CQR produces adaptive intervals."""
        X, y = regression_data
        base = ConditionalInferenceForestRegressor(n_estimators=20, random_state=42, verbose=0)
        cqr = CQR(base, alpha=0.1, random_state=42)
        cqr.fit(X, y)

        lower, upper = cqr.predict_interval(X)
        widths = upper - lower

        # CQR intervals should have some variability (adaptive)
        # unlike split conformal which has constant width
        assert np.std(widths) > 0
