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
