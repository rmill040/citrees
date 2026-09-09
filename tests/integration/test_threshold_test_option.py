"""The ``threshold_test`` option: Bonferroni per threshold versus one max-type test."""

from __future__ import annotations

import numba
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.preprocessing import LabelEncoder

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
    ThresholdTest,
)

pytestmark = pytest.mark.tree


@pytest.fixture(scope="module")
def glass() -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet("tests/data/clf_glass.snappy.parquet")
    return frame.iloc[:, :-1].to_numpy(dtype=np.float64), LabelEncoder().fit_transform(
        frame.iloc[:, -1]
    )


@pytest.fixture(scope="module")
def regression() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1718)
    X = rng.normal(size=(240, 5))
    return X, 2.0 * (X[:, 0] > 0) + X[:, 1] + 0.4 * rng.normal(size=240)


COMMON = dict(threshold_method="histogram", max_thresholds=16, random_state=1718)


class TestOption:
    def test_default_is_bonferroni(self) -> None:
        assert ConditionalInferenceTreeClassifier().threshold_test == ThresholdTest.BONFERRONI
        assert ConditionalInferenceForestRegressor().threshold_test == ThresholdTest.BONFERRONI

    def test_invalid_value_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ConditionalInferenceTreeClassifier(threshold_test="holm")

    def test_forest_forwards_the_option_to_its_trees(self, glass) -> None:
        X, y = glass
        forest = ConditionalInferenceForestClassifier(
            n_estimators=2, selector="mc", threshold_test="maxt", **COMMON
        ).fit(X, y)
        assert all(tree.threshold_test == "maxt" for tree in forest.estimators_)


class TestFits:
    @pytest.mark.parametrize("splitter", ["gini", "entropy"])
    @pytest.mark.parametrize("early_stopping", ["adaptive", None])
    def test_classifier_tree_fits_and_predicts(self, glass, splitter, early_stopping) -> None:
        X, y = glass
        tree = ConditionalInferenceTreeClassifier(
            selector="mc",
            splitter=splitter,
            threshold_test="maxt",
            early_stopping_splitter=early_stopping,
            **COMMON,
        ).fit(X, y)
        assert tree.score(X, y) > 0.8
        assert len(tree.tree_) > 1

    @pytest.mark.parametrize("splitter", ["mse", "mae"])
    def test_regressor_tree_fits_and_predicts(self, regression, splitter) -> None:
        X, y = regression
        tree = ConditionalInferenceTreeRegressor(
            selector="pc", splitter=splitter, threshold_test="maxt", **COMMON
        ).fit(X, y)
        assert tree.score(X, y) > 0.4

    def test_split_thresholds_come_from_the_candidate_set(self, regression) -> None:
        X, y = regression
        tree = ConditionalInferenceTreeRegressor(
            selector="pc", threshold_test="maxt", **COMMON
        ).fit(X, y)

        def thresholds(node):
            if "threshold" in node and node.get("threshold") is not None:
                yield node["feature"], node["threshold"]
            for key in ("left_child", "right_child"):
                if key in node and node[key]:
                    yield from thresholds(node[key])

        for feature, threshold in thresholds(tree.tree_):
            assert X[:, feature].min() <= threshold <= X[:, feature].max()

    def test_maxt_finds_the_planted_split(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.normal(size=(300, 4))
        y = 2.0 * (X[:, 2] > 0.3) + 0.3 * rng.normal(size=300)
        tree = ConditionalInferenceTreeRegressor(
            selector="pc", threshold_test="maxt", max_depth=1, **COMMON
        ).fit(X, y)
        assert tree.tree_["feature"] == 2
        # the histogram candidate nearest the planted step at 0.3
        assert abs(tree.tree_["threshold"] - 0.3) < 0.35

    def test_forest_regressor_fits(self, regression) -> None:
        X, y = regression
        forest = ConditionalInferenceForestRegressor(
            n_estimators=4, selector="pc", threshold_test="maxt", n_jobs=2, **COMMON
        ).fit(X, y)
        assert forest.feature_importances_.argmax() in (0, 1)


class TestInvariance:
    @pytest.fixture
    def restore_threads(self):
        before = numba.get_num_threads()
        yield
        numba.set_num_threads(before)

    def test_maxt_tree_is_thread_count_invariant(self, glass, restore_threads) -> None:
        X, y = glass
        signatures = set()
        for count in (1, 2, 4):
            numba.set_num_threads(min(count, numba.config.NUMBA_NUM_THREADS))
            tree = ConditionalInferenceTreeClassifier(
                selector="mc", threshold_test="maxt", early_stopping_splitter=None, **COMMON
            ).fit(X, y)
            signatures.add(np.round(tree.feature_importances_, 12).tobytes())
        assert len(signatures) == 1

    def test_maxt_forest_is_n_jobs_invariant(self, glass) -> None:
        X, y = glass
        results = [
            ConditionalInferenceForestClassifier(
                n_estimators=4, selector="mc", threshold_test="maxt", n_jobs=n, **COMMON
            )
            .fit(X, y)
            .feature_importances_
            for n in (1, 2)
        ]
        np.testing.assert_array_equal(results[0], results[1])
