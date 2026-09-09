"""Fitted results must not depend on the Numba thread count or on ``n_jobs``.

Inside an ``@njit(parallel=True)`` kernel Numba auto-parallelizes top-level array
reductions, and under fastmath their summation order depends on the thread
count. The observed statistic was computed that way while the permuted
statistics inside ``prange`` were reduced serially, so an exact tie between the
two could flip across ``>=`` at some thread counts and not others. The forest
divides the thread pool across its workers, so the bug surfaced as importances
that changed with ``n_jobs``. The kernels now compute both statistics with the
same serial helpers; these tests pin that.
"""

from __future__ import annotations

import numba
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
)

pytestmark = pytest.mark.tree


@pytest.fixture(scope="module")
def glass() -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet("tests/data/clf_glass.snappy.parquet")
    y = LabelEncoder().fit_transform(frame.iloc[:, -1])
    return frame.iloc[:, :-1].to_numpy(dtype=np.float64), y


@pytest.fixture(scope="module")
def regression() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1718)
    X = rng.normal(size=(150, 6))
    # Duplicated rows make exact statistic ties, the case that used to flip.
    X = np.vstack([X, X[:40]])
    y = X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=X.shape[0])
    return X, y


@pytest.fixture
def restore_threads():
    before = numba.get_num_threads()
    yield
    numba.set_num_threads(before)


def _fit_at_threads(make, X, y, counts):
    signatures = []
    for count in counts:
        numba.set_num_threads(min(count, numba.config.NUMBA_NUM_THREADS))
        model = make().fit(X, y)
        signatures.append(np.round(model.feature_importances_, 12).tobytes())
    return signatures


THREAD_COUNTS = (1, 2, 3, 4)


class TestTreeThreadInvariance:
    @pytest.mark.parametrize("early_stopping", ["adaptive", None])
    def test_classifier_tree_is_thread_count_invariant(
        self, glass, restore_threads, early_stopping
    ) -> None:
        X, y = glass
        make = lambda: ConditionalInferenceTreeClassifier(  # noqa: E731
            selector="mc",
            threshold_method="histogram",
            max_thresholds=32,
            early_stopping_selector=early_stopping,
            early_stopping_splitter=early_stopping,
            random_state=1718,
        )
        assert len(set(_fit_at_threads(make, X, y, THREAD_COUNTS))) == 1

    @pytest.mark.parametrize("splitter", ["gini", "entropy"])
    def test_classifier_splitters_are_thread_count_invariant(
        self, glass, restore_threads, splitter
    ) -> None:
        X, y = glass
        make = lambda: ConditionalInferenceTreeClassifier(  # noqa: E731
            selector="mc",
            splitter=splitter,
            threshold_method="histogram",
            max_thresholds=16,
            early_stopping_selector=None,
            early_stopping_splitter=None,
            random_state=3,
        )
        assert len(set(_fit_at_threads(make, X, y, THREAD_COUNTS))) == 1

    @pytest.mark.parametrize("splitter", ["mse", "mae"])
    def test_regressor_tree_is_thread_count_invariant(
        self, regression, restore_threads, splitter
    ) -> None:
        X, y = regression
        make = lambda: ConditionalInferenceTreeRegressor(  # noqa: E731
            selector="pc",
            splitter=splitter,
            threshold_method="histogram",
            max_thresholds=16,
            early_stopping_selector=None,
            early_stopping_splitter=None,
            random_state=11,
        )
        assert len(set(_fit_at_threads(make, X, y, THREAD_COUNTS))) == 1


class TestForestNJobsInvariance:
    def test_classifier_forest_importances_do_not_depend_on_n_jobs(self, glass) -> None:
        X, y = glass
        results = [
            ConditionalInferenceForestClassifier(
                n_estimators=6,
                selector="mc",
                threshold_method="histogram",
                max_thresholds=32,
                n_jobs=n_jobs,
                random_state=1718,
            )
            .fit(X, y)
            .feature_importances_
            for n_jobs in (1, 2, 3)
        ]
        for other in results[1:]:
            np.testing.assert_array_equal(results[0], other)

    def test_regressor_forest_importances_do_not_depend_on_n_jobs(self, regression) -> None:
        X, y = regression
        results = [
            ConditionalInferenceForestRegressor(
                n_estimators=6,
                selector="pc",
                threshold_method="histogram",
                max_thresholds=16,
                n_jobs=n_jobs,
                random_state=1718,
            )
            .fit(X, y)
            .feature_importances_
            for n_jobs in (1, 2, 3)
        ]
        for other in results[1:]:
            np.testing.assert_array_equal(results[0], other)
