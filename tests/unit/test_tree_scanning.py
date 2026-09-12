"""Scanning (candidate ordering, stop at first rejection) is independent of early stopping."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from citrees import ConditionalInferenceTreeClassifier


def _data(seed: int = 1718, n: int = 400, p: int = 12) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (X[:, 3] + 0.8 * X[:, 7] + rng.normal(size=n) > 0).astype(int)
    return X, y


@pytest.mark.tree
def test_scanning_without_early_stopping_orders_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y = _data()
    clf = ConditionalInferenceTreeClassifier(
        early_stopping_selector=None,
        early_stopping_splitter=None,
        feature_scanning=True,
        threshold_scanning=True,
        n_resamples_selector="minimum",
        n_resamples_splitter="minimum",
        max_depth=1,
        random_state=1718,
    )
    scanned: list[int] = []
    original = ConditionalInferenceTreeClassifier._scan_features

    def spy(self, X_, y_, features):
        ordered = original(self, X_, y_, features)
        scanned.append(len(ordered))
        return ordered

    monkeypatch.setattr(ConditionalInferenceTreeClassifier, "_scan_features", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "unused hyperparameter" warning for this combination
        clf.fit(X, y)
    assert scanned, "feature scanning must run when early stopping is off"
    counts = clf.realized_permutation_counts_
    # Root node: 12 features at alpha/12 => 240 permutations per test. Scanning stops at the
    # first rejecting feature (the strongest, tested first), so far fewer than 12 tests run.
    assert counts["selector"] < 12 * 240


@pytest.mark.tree
def test_no_scanning_tests_every_feature_regardless_of_early_stopping() -> None:
    X, y = _data()
    counts = {}
    for es in ("adaptive", None):
        clf = ConditionalInferenceTreeClassifier(
            early_stopping_selector=es,
            early_stopping_splitter=es,
            feature_scanning=False,
            threshold_scanning=False,
            n_resamples_selector="minimum",
            n_resamples_splitter="minimum",
            max_depth=1,
            max_thresholds=8,
            threshold_method="histogram",
            random_state=1718,
        ).fit(X, y)
        counts[es] = clf.realized_permutation_counts_["selector"]
    # Root node: 12 features tested exhaustively at 240 permutations each; the minimum budget
    # equals the stopping floor, so the adaptive rule cannot stop early either.
    assert counts["adaptive"] == counts[None] == 12 * 240


@pytest.mark.tree
def test_default_configuration_still_scans_and_stops() -> None:
    X, y = _data()
    clf = ConditionalInferenceTreeClassifier(max_depth=1, random_state=1718).fit(X, y)
    assert clf.realized_permutation_counts_["selector"] < 12 * 240
