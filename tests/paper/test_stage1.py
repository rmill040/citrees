"""Tests for paper/benchmark/pipeline/stage1.py (feature selection)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from citrees._selector import (
    ClassifierSelectors,
    ClassifierSelectorTests,
    RegressorSelectors,
    RegressorSelectorTests,
)

pytestmark = pytest.mark.paper


class TestFilterSelector:
    """Tests for filter_selector function."""

    def test_label_invariance(self) -> None:
        """Filter selector should be invariant to label shift."""
        from paper.benchmark.pipeline.stage1 import filter_selector

        n_per = 5
        x0 = np.vstack(
            [
                np.full((n_per, 1), 1.0),
                np.full((n_per, 1), 0.0),
                np.full((n_per, 1), 0.0),
            ]
        )
        x1 = np.vstack(
            [
                np.full((n_per, 1), 0.0),
                np.full((n_per, 1), 0.0),
                np.full((n_per, 1), 100.0),
            ]
        )
        X = np.hstack([x0, x1])
        y = np.array([0] * n_per + [1] * n_per + [2] * n_per)

        rank_base = filter_selector(X, y, method="mc", task="classification", random_state=0)
        rank_shift = filter_selector(X, y + 5, method="mc", task="classification", random_state=0)

        assert np.array_equal(rank_base, rank_shift)


class TestPermutationSelector:
    """Tests for permutation_selector function."""

    def test_kw_only_classifier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ptest_* selector tests should be called with keyword-only args (classification)."""
        from paper.benchmark.pipeline.stage1 import permutation_selector

        def selector_fn(x, y, n_classes, random_state=None):
            return 0.0

        def test_fn(*, x, y, n_classes, alpha, n_resamples, early_stopping, random_state):
            return 0.5

        monkeypatch.setitem(ClassifierSelectors._registry, "kw", selector_fn)
        monkeypatch.setitem(ClassifierSelectorTests._registry, "kw", test_fn)

        X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=float)
        y = np.array([0, 1, 0], dtype=int)

        ranking = permutation_selector(
            X, y, method="ptest_kw", task="classification", random_state=0
        )
        assert ranking.shape == (X.shape[1],)

    def test_kw_only_regression(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ptest_* selector tests should be called with keyword-only args (regression)."""
        from paper.benchmark.pipeline.stage1 import permutation_selector

        def selector_fn(x, y, standardize=True, random_state=None):
            return 0.0

        def test_fn(*, x, y, standardize, alpha, n_resamples, early_stopping, random_state):
            return 0.5

        monkeypatch.setitem(RegressorSelectors._registry, "kw", selector_fn)
        monkeypatch.setitem(RegressorSelectorTests._registry, "kw", test_fn)

        X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=float)
        y = np.array([0.1, -0.2, 0.3], dtype=float)

        ranking = permutation_selector(X, y, method="ptest_kw", task="regression", random_state=0)
        assert ranking.shape == (X.shape[1],)


class TestEmbeddingSelector:
    """Tests for tree-based embedding selector wiring."""

    def test_single_tree_factories_use_sklearn_tree_estimators(self) -> None:
        """dt and rt should be single sklearn trees, not ensemble estimators."""
        from sklearn.tree import (
            DecisionTreeClassifier,
            DecisionTreeRegressor,
            ExtraTreeClassifier,
            ExtraTreeRegressor,
        )

        from paper.benchmark.pipeline.selectors import get_embedding_model

        clf_dt = get_embedding_model("dt", "classification", random_state=1718)
        clf_rt = get_embedding_model("rt", "classification", random_state=1718)
        reg_dt = get_embedding_model("dt", "regression", random_state=1718)
        reg_rt = get_embedding_model("rt", "regression", random_state=1718)

        assert isinstance(clf_dt, DecisionTreeClassifier)
        assert isinstance(clf_rt, ExtraTreeClassifier)
        assert isinstance(reg_dt, DecisionTreeRegressor)
        assert isinstance(reg_rt, ExtraTreeRegressor)
        assert clf_dt.splitter == "best"
        assert clf_rt.splitter == "random"
        assert not hasattr(clf_rt, "n_estimators")
        assert not hasattr(reg_rt, "n_estimators")

    def test_run_selection_accepts_dt_and_rt(self) -> None:
        """Stage 1 should route dt and rt through embedding selection."""
        from paper.benchmark.pipeline.stage1 import run_selection

        rng = np.random.default_rng(1718)
        X = rng.normal(size=(30, 4))
        y = np.array([0, 1] * 15, dtype=int)

        for method in ("dt", "rt"):
            results = run_selection(
                X,
                y,
                method=method,
                task="classification",
                seed=0,
                n_jobs=1,
            )

            assert len(results) == 5
            assert all(row["feature_ranking"] for row in results)


class TestWrapperSelectors:
    """Tests for wrapper selector scoring behavior."""

    def test_pi_selector_uses_balanced_accuracy_for_classification(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PI should align its internal scorer with the benchmark headline metric."""
        from paper.benchmark.pipeline import selectors

        X = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.1, 0.9],
                [0.9, 0.1],
            ],
            dtype=float,
        )
        y = np.array([0, 1, 0, 1, 0, 1], dtype=int)
        captured: dict[str, object] = {}

        def fake_permutation_importance(model, X_val, y_val, **kwargs):
            captured["scoring"] = kwargs["scoring"]
            return SimpleNamespace(importances_mean=np.array([0.2, 0.1], dtype=float))

        monkeypatch.setattr(selectors, "permutation_importance", fake_permutation_importance)

        ranking = selectors.pi_selector(
            X,
            y,
            task="classification",
            random_state=0,
            n_jobs=1,
            val_fraction=0.5,
            params={"n_repeats": 1},
        )

        assert ranking.tolist() == [0, 1]
        assert captured["scoring"] == "balanced_accuracy"

    def test_cpi_selector_uses_balanced_accuracy_for_classification(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CPI should rank minority-recall signal ahead of majority-only signal."""
        from paper.benchmark.pipeline import selectors

        class DummyParallel:
            def __init__(self, n_jobs: int):
                self.n_jobs = n_jobs

            def __call__(self, tasks):
                return [fn(*args, **kwargs) for fn, args, kwargs in tasks]

        def dummy_delayed(fn):
            def wrapped(*args, **kwargs):
                return fn, args, kwargs

            return wrapped

        class DummyRF:
            def __init__(self, *args, **kwargs):
                pass

            def fit(self, X, y):
                return self

            def predict(self, X):
                x0 = X[:, 0]
                x1 = X[:, 1]
                return np.where(x0 > 0.5, 0, np.where(x1 > 0.5, 1, 0)).astype(int)

        class DummyRng:
            def __init__(self, seed: int):
                self.seed = seed

            def permutation(self, values):
                if self.seed == 1718:
                    return np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 1], dtype=values.dtype)
                if self.seed == 1719:
                    return np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=values.dtype)
                raise AssertionError(f"Unexpected seed {self.seed}")

        X_fit = np.zeros((4, 2), dtype=float)
        y_fit = np.array([0, 0, 1, 1], dtype=int)
        X_val = np.array([[1.0, 1.0]] * 8 + [[0.0, 1.0], [0.0, 1.0]], dtype=float)
        y_val = np.array([0] * 8 + [1, 1], dtype=int)

        def fake_train_test_split(X, y, **kwargs):
            return X_fit.copy(), X_val.copy(), y_fit.copy(), y_val.copy()

        monkeypatch.setattr(selectors, "RandomForestClassifier", DummyRF)
        monkeypatch.setattr(selectors, "Parallel", DummyParallel)
        monkeypatch.setattr(selectors, "delayed", dummy_delayed)
        monkeypatch.setattr(selectors, "train_test_split", fake_train_test_split)
        monkeypatch.setattr(selectors.np.random, "default_rng", lambda seed: DummyRng(seed))

        ranking = selectors.cpi_selector(
            np.zeros((1, 2), dtype=float),
            np.array([0], dtype=int),
            task="classification",
            random_state=1718,
            n_jobs=1,
            params={"n_repeats": 1},
        )

        assert ranking.tolist() == [1, 0]


# ---------------------------------------------------------------------------
# r_ctree / r_cforest
# ---------------------------------------------------------------------------

try:
    from paper.benchmark.pipeline.r_methods import _get_partykit, r_ctree_ranking

    _get_partykit()
    _has_r = True
except (ImportError, OSError):
    _has_r = False

_skip_no_r = pytest.mark.skipif(not _has_r, reason="R / rpy2 / partykit not available")


class TestRCtreeRanking:
    """Regression tests for r_ctree_ranking (partykit node traversal)."""

    @_skip_no_r
    def test_ranking_not_identity_classification(self) -> None:
        """r_ctree must produce a non-identity ranking on separable data."""
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42,
        )
        ranking = r_ctree_ranking(X, y, task="classification")
        assert ranking.shape == (10,)
        assert not np.array_equal(ranking, np.arange(10)), (
            "ranking is identity — split extraction broken"
        )

    @_skip_no_r
    def test_ranking_not_identity_regression(self) -> None:
        """r_ctree must produce a non-identity ranking on regression data."""
        from sklearn.datasets import make_regression

        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=5,
            random_state=42,
        )
        ranking = r_ctree_ranking(X, y, task="regression")
        assert ranking.shape == (10,)
        assert not np.array_equal(ranking, np.arange(10)), (
            "ranking is identity — split extraction broken"
        )

    @_skip_no_r
    def test_stump_on_noise(self) -> None:
        """Pure noise should yield a stump (no splits) and identity ranking."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 20))
        y = rng.integers(0, 2, 50)
        ranking = r_ctree_ranking(X, y, task="classification")
        assert ranking.shape == (20,)
        # Identity is the expected fallback when no splits are made
        assert np.array_equal(ranking, np.arange(20))

    @_skip_no_r
    def test_deterministic(self) -> None:
        """Same input must give the same ranking (ctree is deterministic)."""
        from sklearn.datasets import load_iris

        iris = load_iris()
        r1 = r_ctree_ranking(iris.data, iris.target, task="classification")
        r2 = r_ctree_ranking(iris.data, iris.target, task="classification")
        assert np.array_equal(r1, r2)

    @_skip_no_r
    def test_top_features_are_split_variables(self) -> None:
        """Top-ranked features should be those actually used in splits."""
        from sklearn.datasets import load_wine

        wine = load_wine()
        ranking = r_ctree_ranking(wine.data, wine.target, task="classification")
        # Wine tree uses several features; rank-1 must not be a zero-count feature.
        # If var_counts were all-zero, ranking[0] would be 0 (identity fallback).
        # Wine's known important features include proline (12), flavanoids (6),
        # color_intensity (9), etc. — feature 0 (alcohol) is rarely the sole
        # top split, so a non-zero first entry is strong evidence.
        assert ranking[0] != 0, "rank-1 feature is 0 — likely identity fallback"
