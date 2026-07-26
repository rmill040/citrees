"""Integration tests for edge cases, boundary conditions, and special modes."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
)

# Fast parameters for integration tests
FAST_PARAMS = {
    "n_resamples_selector": "minimum",
    "n_resamples_splitter": "minimum",
    "verbose": 0,
    "random_state": 42,
}


@pytest.fixture
def classification_data():
    """Generate classification dataset."""
    X, y = make_classification(
        n_samples=80,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=2,
        random_state=42,
    )
    return X, y


@pytest.fixture
def regression_data():
    """Generate regression dataset."""
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=80,
        n_features=10,
        n_informative=5,
        noise=10.0,
        random_state=42,
    )
    return X, y


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_feature(self):
        """Test with single feature."""
        X = np.random.randn(100, 1)
        y = (X[:, 0] > 0).astype(int)

        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_single_class(self):
        """Test with single class (should still work)."""
        X = np.random.randn(100, 5)
        y = np.zeros(100, dtype=int)

        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert (preds == 0).all()

    def test_constant_feature(self, classification_data):
        """Test with constant feature."""
        X, y = classification_data
        X[:, 0] = 1.0

        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_high_dimensional(self):
        """Test with more features than samples."""
        X = np.random.randn(50, 100)
        y = np.random.randint(0, 2, 50)

        clf = ConditionalInferenceTreeClassifier(max_features="sqrt", **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_max_depth_1(self, classification_data):
        """Test with max_depth=1 (stump)."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(max_depth=1, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_small_sample(self):
        """Test with small sample size."""
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, 20)

        clf = ConditionalInferenceTreeClassifier(min_samples_split=2, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape


def _count_splits(node):
    """Count internal (non-leaf) nodes in a fitted tree."""
    if node is None or "left_child" not in node:
        return 0
    return 1 + _count_splits(node.get("left_child")) + _count_splits(node.get("right_child"))


class TestStatisticalCorrectness:
    """The permutation tests must gate splitting, not merely run.

    Selector-level p-value validity is asserted directly in
    tests/unit/test_statistical_validity.py. These tests cover the property
    that matters at the estimator level: a tree built on pure noise should not
    split, because no feature clears the significance threshold.
    """

    @pytest.mark.slow
    def test_no_splits_on_pure_noise(self):
        """A tree fit on independent noise must stay a single leaf."""
        rng = np.random.default_rng(1718)
        X = rng.standard_normal((150, 10))
        y = rng.integers(0, 2, 150)

        clf = ConditionalInferenceTreeClassifier(
            n_resamples_selector="auto",
            n_resamples_splitter="auto",
            alpha_selector=0.05,
            random_state=1718,
            verbose=0,
        )
        clf.fit(X, y)

        n_splits = _count_splits(clf.tree_)
        assert n_splits == 0, (
            f"Tree split {n_splits} times on pure noise; permutation gating is not working"
        )

    @pytest.mark.slow
    def test_splits_on_real_signal(self):
        """The same configuration must still split when signal is present."""
        rng = np.random.default_rng(1718)
        X = rng.standard_normal((150, 10))
        y = (X[:, 0] > 0).astype(int)

        clf = ConditionalInferenceTreeClassifier(
            n_resamples_selector="auto",
            n_resamples_splitter="auto",
            alpha_selector=0.05,
            random_state=1718,
            verbose=0,
        )
        clf.fit(X, y)

        assert _count_splits(clf.tree_) > 0, "Tree failed to split on a clean deterministic signal"
        assert clf.tree_["feature"] == 0, (
            f"Tree split on feature {clf.tree_['feature']} instead of the informative one"
        )


class TestHonestyMode:
    """Honest estimation must actually split the sample, not just accept the flag."""

    def test_honesty_changes_the_fitted_tree(self, classification_data):
        """Honest and non-honest fits on identical data and seed must differ.

        Honesty holds out a fraction of the data for leaf estimation, so the
        structure is learned from fewer samples. An identical tree would mean
        the flag is being ignored.
        """
        X, y = classification_data

        params = {**FAST_PARAMS, "max_depth": 3}
        honest = ConditionalInferenceTreeClassifier(
            honesty=True, honesty_fraction=0.5, **params
        ).fit(X, y)
        standard = ConditionalInferenceTreeClassifier(honesty=False, **params).fit(X, y)

        assert honest.tree_ != standard.tree_, (
            "honesty=True produced an identical tree to honesty=False"
        )

    def test_honesty_reduces_in_sample_optimism(self):
        """Honest leaf estimates should be less optimistic on training data.

        Leaves are estimated from samples not used to choose the splits, so the
        training-set accuracy of an honest tree should not exceed that of a
        standard tree. Averaged over seeds to avoid asserting on one draw.
        """
        honest_scores = []
        standard_scores = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            X = rng.standard_normal((200, 8))
            y = (X[:, 0] + rng.standard_normal(200) * 1.5 > 0).astype(int)

            params = {
                "n_resamples_selector": "minimum",
                "n_resamples_splitter": "minimum",
                "verbose": 0,
                "random_state": seed,
                "max_depth": 4,
            }
            honest = ConditionalInferenceTreeClassifier(
                honesty=True, honesty_fraction=0.5, **params
            ).fit(X, y)
            standard = ConditionalInferenceTreeClassifier(honesty=False, **params).fit(X, y)

            honest_scores.append((honest.predict(X) == y).mean())
            standard_scores.append((standard.predict(X) == y).mean())

        assert np.mean(honest_scores) <= np.mean(standard_scores), (
            f"Honest tree was more optimistic in-sample ({np.mean(honest_scores):.3f}) "
            f"than the standard tree ({np.mean(standard_scores):.3f})"
        )

    def test_honest_regressor_fits_and_predicts(self, regression_data):
        """Honest regression must produce finite predictions of the right shape."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        reg = ConditionalInferenceTreeRegressor(honesty=True, honesty_fraction=0.5, **FAST_PARAMS)
        reg.fit(X_train, y_train)

        y_pred = reg.predict(X_test)
        assert y_pred.shape == y_test.shape
        assert np.all(np.isfinite(y_pred)), "Honest regressor produced non-finite predictions"

    def test_honest_forest_classifier(self, classification_data):
        """Test honest forest classifier."""
        X, y = classification_data

        clf = ConditionalInferenceForestClassifier(
            n_estimators=5, honesty=True, honesty_fraction=0.5, **FAST_PARAMS
        )
        clf.fit(X, y)

        assert len(clf.estimators_) == 5
        y_pred = clf.predict(X)
        assert y_pred.shape == y.shape

    def test_honest_forest_regressor(self, regression_data):
        """Test honest forest regressor."""
        X, y = regression_data

        reg = ConditionalInferenceForestRegressor(
            n_estimators=5, honesty=True, honesty_fraction=0.5, **FAST_PARAMS
        )
        reg.fit(X, y)

        assert len(reg.estimators_) == 5
        y_pred = reg.predict(X)
        assert y_pred.shape == y.shape

    def test_honesty_fraction_values(self, classification_data):
        """Test different honesty_fraction values."""
        X, y = classification_data

        for fraction in [0.3, 0.5, 0.7]:
            clf = ConditionalInferenceTreeClassifier(
                honesty=True, honesty_fraction=fraction, **FAST_PARAMS
            )
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape
