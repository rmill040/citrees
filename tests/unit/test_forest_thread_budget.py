"""Forest workers must share, not multiply, the Numba thread pool."""

import numba
import numpy as np

from citrees import ConditionalInferenceForestClassifier
from citrees._forest import _threads_per_worker


def test_threads_per_worker_divides_pool():
    total = numba.get_num_threads()
    assert _threads_per_worker(1) == total
    assert _threads_per_worker(None) == total
    assert _threads_per_worker(2) == max(1, total // 2)
    assert _threads_per_worker(10_000) == 1


def test_parallel_forest_matches_serial():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6))
    y = (X[:, 0] + 0.3 * rng.normal(size=120) > 0).astype(int)
    kwargs = dict(n_estimators=4, max_depth=2, random_state=7, verbose=0)
    serial = ConditionalInferenceForestClassifier(n_jobs=1, **kwargs).fit(X, y)
    parallel = ConditionalInferenceForestClassifier(n_jobs=2, **kwargs).fit(X, y)
    np.testing.assert_allclose(serial.feature_importances_, parallel.feature_importances_)
