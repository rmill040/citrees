"""Tests for paper/scripts/experiments/ray_feature_selection.py."""

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

pytestmark = pytest.mark.other


def _fake_ray_module() -> ModuleType:
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


def _load_ray_feature_selection(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "ray", _fake_ray_module())
    monkeypatch.setitem(sys.modules, "shap", ModuleType("shap"))

    module_name = "paper.scripts.experiments.ray_feature_selection"
    if module_name in sys.modules:
        del sys.modules[module_name]

    return importlib.import_module(module_name)


def test_permutation_selector_kw_only_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """ptest_* selector tests should be called with keyword-only args (classification)."""
    module = _load_ray_feature_selection(monkeypatch)

    def selector_fn(x, y, n_classes, random_state=None):
        return 0.0

    def test_fn(*, x, y, n_classes, alpha, n_resamples, early_stopping, random_state):
        return 0.5

    monkeypatch.setitem(ClassifierSelectors._registry, "kw", selector_fn)
    monkeypatch.setitem(ClassifierSelectorTests._registry, "kw", test_fn)

    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=float)
    y = np.array([0, 1, 0], dtype=int)

    ranking = module.permutation_selector(
        X, y, method="ptest_kw", task_type="classification", random_state=0
    )
    assert ranking.shape == (X.shape[1],)


def test_permutation_selector_kw_only_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    """ptest_* selector tests should be called with keyword-only args (regression)."""
    module = _load_ray_feature_selection(monkeypatch)

    def selector_fn(x, y, standardize=True, random_state=None):
        return 0.0

    def test_fn(*, x, y, standardize, alpha, n_resamples, early_stopping, random_state):
        return 0.5

    monkeypatch.setitem(RegressorSelectors._registry, "kw", selector_fn)
    monkeypatch.setitem(RegressorSelectorTests._registry, "kw", test_fn)

    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=float)
    y = np.array([0.1, -0.2, 0.3], dtype=float)

    ranking = module.permutation_selector(
        X, y, method="ptest_kw", task_type="regression", random_state=0
    )
    assert ranking.shape == (X.shape[1],)
