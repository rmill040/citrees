"""Integration tests for sklearn compatibility."""

import pickle
import tempfile

import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
        n_samples=100,
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
        n_samples=100,
        n_features=10,
        n_informative=5,
        noise=10.0,
        random_state=42,
    )
    return X, y


# =============================================================================
# ESTIMATOR TYPE CHECKS
# =============================================================================


class TestEstimatorTypes:
    """Test sklearn estimator type detection."""

    def test_tree_classifier_is_classifier(self):
        """Test tree classifier is detected as classifier."""
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        assert is_classifier(clf)
        assert not is_regressor(clf)

    def test_tree_regressor_is_regressor(self):
        """Test tree regressor is detected as regressor."""
        reg = ConditionalInferenceTreeRegressor(**FAST_PARAMS)
        assert is_regressor(reg)
        assert not is_classifier(reg)

    def test_forest_classifier_is_classifier(self):
        """Test forest classifier is detected as classifier."""
        clf = ConditionalInferenceForestClassifier(n_estimators=5, **FAST_PARAMS)
        assert is_classifier(clf)
        assert not is_regressor(clf)

    def test_forest_regressor_is_regressor(self):
        """Test forest regressor is detected as regressor."""
        reg = ConditionalInferenceForestRegressor(n_estimators=5, **FAST_PARAMS)
        assert is_regressor(reg)
        assert not is_classifier(reg)


# =============================================================================
# CLONE TESTS
# =============================================================================


class TestClone:
    """Test sklearn clone functionality."""

    def test_clone_tree_classifier(self, classification_data):
        """Test cloning tree classifier."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(alpha_selector=0.1, **FAST_PARAMS)
        clf.fit(X, y)

        clf2 = clone(clf)

        # Cloned estimator should have same parameters
        assert clf2.alpha_selector == 0.1
        assert clf2.random_state == 42

        # Cloned estimator should not be fitted
        assert not hasattr(clf2, "classes_")
        assert not hasattr(clf2, "tree_")

        # Cloned estimator should be fittable
        clf2.fit(X, y)
        assert hasattr(clf2, "classes_")

    def test_clone_tree_regressor(self, regression_data):
        """Test cloning tree regressor."""
        X, y = regression_data
        reg = ConditionalInferenceTreeRegressor(alpha_selector=0.1, **FAST_PARAMS)
        reg.fit(X, y)

        reg2 = clone(reg)

        assert reg2.alpha_selector == 0.1
        assert not hasattr(reg2, "tree_")

        reg2.fit(X, y)
        assert hasattr(reg2, "tree_")

    def test_clone_forest_classifier(self, classification_data):
        """Test cloning forest classifier."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=5, **FAST_PARAMS)
        clf.fit(X, y)

        clf2 = clone(clf)

        assert clf2.n_estimators == 5
        assert not hasattr(clf2, "estimators_")

        clf2.fit(X, y)
        assert len(clf2.estimators_) == 5

    def test_clone_forest_regressor(self, regression_data):
        """Test cloning forest regressor."""
        X, y = regression_data
        reg = ConditionalInferenceForestRegressor(n_estimators=5, **FAST_PARAMS)
        reg.fit(X, y)

        reg2 = clone(reg)

        assert reg2.n_estimators == 5
        assert not hasattr(reg2, "estimators_")

        reg2.fit(X, y)
        assert len(reg2.estimators_) == 5


# =============================================================================
# GET/SET PARAMS TESTS
# =============================================================================


class TestGetSetParams:
    """Test get_params and set_params functionality."""

    def test_get_params_tree_classifier(self):
        """Test get_params for tree classifier."""
        clf = ConditionalInferenceTreeClassifier(
            alpha_selector=0.1,
            max_depth=5,
            **FAST_PARAMS,
        )
        params = clf.get_params()

        assert params["alpha_selector"] == 0.1
        assert params["max_depth"] == 5
        assert params["random_state"] == 42

    def test_set_params_tree_classifier(self):
        """Test set_params for tree classifier."""
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)

        clf.set_params(alpha_selector=0.2, max_depth=3)

        assert clf.alpha_selector == 0.2
        assert clf.max_depth == 3

    def test_get_params_forest_classifier(self):
        """Test get_params for forest classifier."""
        clf = ConditionalInferenceForestClassifier(
            n_estimators=10,
            max_depth=5,
            **FAST_PARAMS,
        )
        params = clf.get_params()

        assert params["n_estimators"] == 10
        assert params["max_depth"] == 5

    def test_set_params_forest_classifier(self):
        """Test set_params for forest classifier."""
        clf = ConditionalInferenceForestClassifier(n_estimators=5, **FAST_PARAMS)

        clf.set_params(n_estimators=10, max_depth=3)

        assert clf.n_estimators == 10
        assert clf.max_depth == 3


# =============================================================================
# PIPELINE TESTS
# =============================================================================


class TestPipeline:
    """Test sklearn Pipeline integration."""

    def test_pipeline_tree_classifier(self, classification_data):
        """Test tree classifier in pipeline."""
        X, y = classification_data

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", ConditionalInferenceTreeClassifier(**FAST_PARAMS)),
            ]
        )

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert preds.shape == y.shape

    def test_pipeline_tree_regressor(self, regression_data):
        """Test tree regressor in pipeline."""
        X, y = regression_data

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", ConditionalInferenceTreeRegressor(**FAST_PARAMS)),
            ]
        )

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert preds.shape == y.shape

    def test_pipeline_forest_classifier(self, classification_data):
        """Test forest classifier in pipeline."""
        X, y = classification_data

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", ConditionalInferenceForestClassifier(n_estimators=5, **FAST_PARAMS)),
            ]
        )

        pipe.fit(X, y)
        preds = pipe.predict(X)
        proba = pipe.predict_proba(X)

        assert preds.shape == y.shape
        assert proba.shape == (len(y), 2)

    def test_pipeline_forest_regressor(self, regression_data):
        """Test forest regressor in pipeline."""
        X, y = regression_data

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", ConditionalInferenceForestRegressor(n_estimators=5, **FAST_PARAMS)),
            ]
        )

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert preds.shape == y.shape


# =============================================================================
# GRID SEARCH TESTS
# =============================================================================


class TestGridSearch:
    """Test sklearn GridSearchCV integration."""

    def test_gridsearch_tree_classifier(self, classification_data):
        """Test GridSearchCV with tree classifier."""
        X, y = classification_data

        clf = ConditionalInferenceTreeClassifier(
            n_resamples_selector="minimum",
            n_resamples_splitter="minimum",
            verbose=0,
        )

        param_grid = {
            "alpha_selector": [0.05, 0.1],
            "max_depth": [3, 5],
        }

        grid = GridSearchCV(clf, param_grid, cv=3, scoring="accuracy")
        grid.fit(X, y)

        assert hasattr(grid, "best_params_")
        assert hasattr(grid, "best_score_")
        assert grid.best_params_["alpha_selector"] in [0.05, 0.1]
        assert grid.best_params_["max_depth"] in [3, 5]

    def test_gridsearch_tree_regressor(self, regression_data):
        """Test GridSearchCV with tree regressor."""
        X, y = regression_data

        reg = ConditionalInferenceTreeRegressor(
            n_resamples_selector="minimum",
            n_resamples_splitter="minimum",
            verbose=0,
        )

        param_grid = {
            "alpha_selector": [0.05, 0.1],
            "max_depth": [3, 5],
        }

        grid = GridSearchCV(reg, param_grid, cv=3, scoring="r2")
        grid.fit(X, y)

        assert hasattr(grid, "best_params_")
        assert hasattr(grid, "best_score_")

    def test_gridsearch_forest_classifier(self, classification_data):
        """Test GridSearchCV with forest classifier."""
        X, y = classification_data

        clf = ConditionalInferenceForestClassifier(
            n_resamples_selector="minimum",
            n_resamples_splitter="minimum",
            verbose=0,
        )

        param_grid = {
            "n_estimators": [3, 5],
            "max_depth": [2, 3],
        }

        grid = GridSearchCV(clf, param_grid, cv=2, scoring="accuracy")
        grid.fit(X, y)

        assert hasattr(grid, "best_params_")


# =============================================================================
# CROSS VALIDATION TESTS
# =============================================================================


class TestCrossValidation:
    """Test sklearn cross_val_score integration."""

    def test_cross_val_tree_classifier(self, classification_data):
        """Test cross_val_score with tree classifier."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)

        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")

        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_cross_val_tree_regressor(self, regression_data):
        """Test cross_val_score with tree regressor."""
        X, y = regression_data
        reg = ConditionalInferenceTreeRegressor(**FAST_PARAMS)

        scores = cross_val_score(reg, X, y, cv=3, scoring="r2")

        assert len(scores) == 3

    def test_cross_val_forest_classifier(self, classification_data):
        """Test cross_val_score with forest classifier."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=5, **FAST_PARAMS)

        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")

        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_cross_val_forest_regressor(self, regression_data):
        """Test cross_val_score with forest regressor."""
        X, y = regression_data
        reg = ConditionalInferenceForestRegressor(n_estimators=5, **FAST_PARAMS)

        scores = cross_val_score(reg, X, y, cv=3, scoring="r2")

        assert len(scores) == 3


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestSerialization:
    """Test pickle serialization."""

    def test_pickle_tree_classifier(self, classification_data):
        """Test pickling tree classifier."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X, y)

        # Pickle and unpickle
        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(clf, f)
            f.seek(0)
            clf2 = pickle.load(f)

        # Predictions should match
        preds1 = clf.predict(X)
        preds2 = clf2.predict(X)
        assert np.array_equal(preds1, preds2)

        # Probabilities should match
        proba1 = clf.predict_proba(X)
        proba2 = clf2.predict_proba(X)
        assert np.allclose(proba1, proba2)

    def test_pickle_tree_regressor(self, regression_data):
        """Test pickling tree regressor."""
        X, y = regression_data
        reg = ConditionalInferenceTreeRegressor(**FAST_PARAMS)
        reg.fit(X, y)

        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(reg, f)
            f.seek(0)
            reg2 = pickle.load(f)

        preds1 = reg.predict(X)
        preds2 = reg2.predict(X)
        assert np.allclose(preds1, preds2)

    def test_pickle_forest_classifier(self, classification_data):
        """Test pickling forest classifier."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=5, **FAST_PARAMS)
        clf.fit(X, y)

        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(clf, f)
            f.seek(0)
            clf2 = pickle.load(f)

        preds1 = clf.predict(X)
        preds2 = clf2.predict(X)
        assert np.array_equal(preds1, preds2)

    def test_pickle_forest_regressor(self, regression_data):
        """Test pickling forest regressor."""
        X, y = regression_data
        reg = ConditionalInferenceForestRegressor(n_estimators=5, **FAST_PARAMS)
        reg.fit(X, y)

        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(reg, f)
            f.seek(0)
            reg2 = pickle.load(f)

        preds1 = reg.predict(X)
        preds2 = reg2.predict(X)
        assert np.allclose(preds1, preds2)

    def test_pickle_unfitted_estimator(self):
        """Test pickling unfitted estimator."""
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)

        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(clf, f)
            f.seek(0)
            clf2 = pickle.load(f)

        # Parameters should be preserved
        assert clf2.random_state == 42
        assert clf2.alpha_selector == clf.alpha_selector


# =============================================================================
# FEATURE NAMES TESTS
# =============================================================================


class TestFeatureNames:
    """Test feature names handling."""

    def test_feature_names_in_tree(self, classification_data):
        """Test feature_names_in_ attribute."""
        import pandas as pd

        X, y = classification_data
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)

        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X_df, y)

        assert hasattr(clf, "feature_names_in_")
        assert list(clf.feature_names_in_) == feature_names

    def test_feature_names_in_forest(self, classification_data):
        """Test feature_names_in_ attribute for forest."""
        import pandas as pd

        X, y = classification_data
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)

        clf = ConditionalInferenceForestClassifier(n_estimators=5, **FAST_PARAMS)
        clf.fit(X_df, y)

        assert hasattr(clf, "feature_names_in_")
        assert list(clf.feature_names_in_) == feature_names
