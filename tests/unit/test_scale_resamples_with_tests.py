"""An explicit permutation budget is scaled by the Bonferroni test count unless disabled."""

import numpy as np

from citrees import ConditionalInferenceForestClassifier, ConditionalInferenceTreeClassifier


def _fit(**kwargs):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = (X[:, 0] > 0).astype(np.int64)
    tree = ConditionalInferenceTreeClassifier(
        selector="mc", n_resamples_selector=999, n_resamples_splitter=None, random_state=0, **kwargs
    )
    tree.fit(X, y)
    return tree


def test_explicit_budget_is_scaled_by_default() -> None:
    tree = _fit()
    assert tree._n_resamples_selector == 999 * 5


def test_explicit_budget_kept_when_scaling_disabled() -> None:
    tree = _fit(scale_resamples_with_tests=False)
    assert tree._n_resamples_selector == 999
    assert tree._alpha_selector == 0.05 / 5


def test_forest_passes_flag_to_trees() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = (X[:, 0] > 0).astype(np.int64)
    forest = ConditionalInferenceForestClassifier(
        n_estimators=2,
        selector="mc",
        n_resamples_selector=999,
        n_resamples_splitter=None,
        scale_resamples_with_tests=False,
        n_jobs=1,
        random_state=0,
    ).fit(X, y)
    assert all(t.scale_resamples_with_tests is False for t in forest.estimators_)
