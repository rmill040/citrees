"""Tests for paper/scripts/pipeline/stage1.py (feature selection)."""

from __future__ import annotations

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
        from paper.scripts.pipeline.stage1 import filter_selector

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
        from paper.scripts.pipeline.stage1 import permutation_selector

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
        from paper.scripts.pipeline.stage1 import permutation_selector

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


# ---------------------------------------------------------------------------
# r_ctree / r_cforest
# ---------------------------------------------------------------------------

try:
    from paper.scripts.pipeline.r_methods import r_ctree_ranking, r_cforest_ranking

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
            n_samples=200, n_features=10, n_informative=5,
            n_redundant=0, n_clusters_per_class=1, random_state=42,
        )
        ranking = r_ctree_ranking(X, y, task="classification")
        assert ranking.shape == (10,)
        assert not np.array_equal(ranking, np.arange(10)), "ranking is identity — split extraction broken"

    @_skip_no_r
    def test_ranking_not_identity_regression(self) -> None:
        """r_ctree must produce a non-identity ranking on regression data."""
        from sklearn.datasets import make_regression

        X, y = make_regression(
            n_samples=200, n_features=10, n_informative=5, random_state=42,
        )
        ranking = r_ctree_ranking(X, y, task="regression")
        assert ranking.shape == (10,)
        assert not np.array_equal(ranking, np.arange(10)), "ranking is identity — split extraction broken"

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
