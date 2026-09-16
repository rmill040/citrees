"""Unit test for the out-of-bag permutation importance used by the CIF check."""

import numpy as np
import pytest

from citrees import ConditionalInferenceForestClassifier, ConditionalInferenceForestRegressor
from paper.benchmark.experiments.cif_permutation_importance_check import oob_permutation_importance

pytestmark = pytest.mark.paper


def _data(seed: int, classification: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(80, 6))
    signal = 2.0 * X[:, 0] - 1.5 * X[:, 1]
    y = (signal > 0).astype(np.int64) if classification else signal + 0.3 * rng.normal(size=80)
    return X, y


@pytest.mark.parametrize("classification", [True, False])
def test_oob_importance_ranks_signal_features_first(classification: bool) -> None:
    X, y = _data(0, classification)
    cls = (
        ConditionalInferenceForestClassifier
        if classification
        else ConditionalInferenceForestRegressor
    )
    model = cls(
        n_estimators=8,
        n_jobs=1,
        random_state=0,
        selector="mc" if classification else "pc",
        max_thresholds=32,
    )
    model.fit(X, y)
    imp = oob_permutation_importance(model, X, y, seed=0)
    assert imp.shape == (6,)
    assert set(np.argsort(imp)[::-1][:2]) == {0, 1}
    assert np.all(imp[2:] <= imp[:2].min())
