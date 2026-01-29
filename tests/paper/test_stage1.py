"""Tests for paper/scripts/pipeline/stage1.py (feature selection)."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import numpy as np
import pytest

from citrees._selector import (
    ClassifierSelectors,
    ClassifierSelectorTests,
    RegressorSelectors,
    RegressorSelectorTests,
)

pytestmark = pytest.mark.paper


def _fake_ray_module() -> ModuleType:
    """Create a fake ray module for testing without Ray dependency."""
    fake_ray = ModuleType("ray")

    def remote(*args, **kwargs):
        def decorator(fn):
            return fn

        if args and len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return decorator

    fake_ray.remote = remote  # type: ignore[attr-defined]
    fake_ray.init = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    fake_ray.get = lambda value: value  # type: ignore[attr-defined]
    return fake_ray


def _load_stage1_module(monkeypatch: pytest.MonkeyPatch):
    """Load stage1 module with mocked Ray dependency."""
    monkeypatch.setitem(sys.modules, "ray", _fake_ray_module())
    monkeypatch.setitem(sys.modules, "shap", ModuleType("shap"))

    module_name = "paper.scripts.pipeline.stage1"
    if module_name in sys.modules:
        del sys.modules[module_name]

    return importlib.import_module(module_name)


class TestFilterSelector:
    """Tests for filter_selector function."""

    def test_label_invariance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Filter selector should be invariant to label shift."""
        module = _load_stage1_module(monkeypatch)

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

        rank_base = module.filter_selector(
            X, y, method="mc", task="classification", random_state=0
        )
        rank_shift = module.filter_selector(
            X, y + 5, method="mc", task="classification", random_state=0
        )

        assert np.array_equal(rank_base, rank_shift)


class TestPermutationSelector:
    """Tests for permutation_selector function."""

    def test_kw_only_classifier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ptest_* selector tests should be called with keyword-only args (classification)."""
        module = _load_stage1_module(monkeypatch)

        def selector_fn(x, y, n_classes, random_state=None):
            return 0.0

        def test_fn(*, x, y, n_classes, alpha, n_resamples, early_stopping, random_state):
            return 0.5

        monkeypatch.setitem(ClassifierSelectors._registry, "kw", selector_fn)
        monkeypatch.setitem(ClassifierSelectorTests._registry, "kw", test_fn)

        X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=float)
        y = np.array([0, 1, 0], dtype=int)

        ranking = module.permutation_selector(
            X, y, method="ptest_kw", task="classification", random_state=0
        )
        assert ranking.shape == (X.shape[1],)

    def test_kw_only_regression(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ptest_* selector tests should be called with keyword-only args (regression)."""
        module = _load_stage1_module(monkeypatch)

        def selector_fn(x, y, standardize=True, random_state=None):
            return 0.0

        def test_fn(*, x, y, standardize, alpha, n_resamples, early_stopping, random_state):
            return 0.5

        monkeypatch.setitem(RegressorSelectors._registry, "kw", selector_fn)
        monkeypatch.setitem(RegressorSelectorTests._registry, "kw", test_fn)

        X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=float)
        y = np.array([0.1, -0.2, 0.3], dtype=float)

        ranking = module.permutation_selector(
            X, y, method="ptest_kw", task="regression", random_state=0
        )
        assert ranking.shape == (X.shape[1],)
