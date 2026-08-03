"""Tests for exact permutation-count diagnostics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from citrees import ConditionalInferenceTreeClassifier, _splitter
from citrees._permutation import collect_permutation_counts
from citrees._selector import (
    _ADAPTIVE_BATCH_SIZE as SELECTOR_ADAPTIVE_BATCH_SIZE,
)
from citrees._selector import (
    _ptest_mc_parallel_batched_result,
    _ptest_mc_parallel_result,
    _ptest_multi_result,
    ptest_dc,
    ptest_mc,
    ptest_mi,
    ptest_pc,
    ptest_rdc_classifier,
    ptest_rdc_regressor,
)
from citrees._selector import _ptest_result as _selector_ptest_result
from citrees._splitter import (
    _ptest_mse_parallel_result,
    ptest_entropy,
    ptest_gini,
    ptest_mae,
    ptest_mse,
)
from citrees._splitter import _ptest_result as _splitter_ptest_result


@pytest.mark.parametrize("stopping", [None, "adaptive", "simple"])
def test_serial_selector_count_matches_statistic_invocations(
    stopping: str | None,
) -> None:
    calls = 0

    def statistic(
        x: np.ndarray,
        y: np.ndarray,
        standardize: bool,
        random_state: int,
    ) -> float:
        nonlocal calls
        calls += 1
        return float(np.corrcoef(x, y)[0, 1])

    rng = np.random.default_rng(1718)
    x = rng.normal(size=60)
    y = rng.normal(size=60)
    _, n_permutations = _selector_ptest_result(
        func=statistic,
        func_arg=True,
        x=x,
        y=y,
        n_resamples=39,
        early_stopping=stopping,
        alpha=0.05,
        random_state=1718,
    )

    assert n_permutations == calls - 1


@pytest.mark.parametrize("stopping", [None, "adaptive", "simple"])
def test_serial_splitter_count_matches_impurity_invocations(
    stopping: str | None,
) -> None:
    calls = 0

    def impurity(y: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return float(np.var(y))

    rng = np.random.default_rng(1718)
    x = rng.normal(size=60)
    y = rng.normal(size=60)
    _, n_permutations = _splitter_ptest_result(
        func=impurity,
        x=x,
        y=y,
        threshold=0.0,
        n_resamples=39,
        early_stopping=stopping,
        alpha=0.05,
        random_state=1718,
    )

    assert n_permutations == (calls - 2) // 2


def test_serial_splitter_adaptive_stops_at_first_eligible_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked_counts: list[int] = []
    impurity_calls = 0

    def stop_at_first_check(alpha: float, a: float, b: float) -> float:
        assert alpha == 0.05
        checked_counts.append(int(a + b - 2))
        return 1.0

    def impurity(y: np.ndarray) -> float:
        nonlocal impurity_calls
        impurity_calls += 1
        return float(np.var(y))

    monkeypatch.setattr(_splitter, "_beta_cdf", stop_at_first_check)
    rng = np.random.default_rng(1718)
    x = rng.normal(size=60)
    y = rng.normal(size=60)
    _, n_permutations = _splitter_ptest_result(
        func=impurity,
        x=x,
        y=y,
        threshold=0.0,
        n_resamples=95,
        early_stopping="adaptive",
        alpha=0.05,
        random_state=1718,
    )

    assert checked_counts == [32]
    assert n_permutations == 32
    assert impurity_calls == 2 + (2 * n_permutations)


def test_serial_splitter_adaptive_reports_partial_final_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked_counts: list[int] = []

    def never_stop(alpha: float, a: float, b: float) -> float:
        assert alpha == 0.05
        checked_counts.append(int(a + b - 2))
        return 0.5

    monkeypatch.setattr(_splitter, "_beta_cdf", never_stop)
    rng = np.random.default_rng(1718)
    x = rng.normal(size=60)
    y = rng.normal(size=60)
    _, n_permutations = _splitter_ptest_result(
        func=np.var,
        x=x,
        y=y,
        threshold=0.0,
        n_resamples=95,
        early_stopping="adaptive",
        alpha=0.05,
        random_state=1718,
    )

    assert checked_counts == [32, 64, 95]
    assert n_permutations == 95


def test_serial_splitter_strong_signal_has_exact_adaptive_count() -> None:
    x = np.concatenate([np.zeros(50), np.ones(50)])
    y = x.astype(np.int64)

    p_value, n_permutations = _splitter_ptest_result(
        func=_splitter.gini,
        x=x,
        y=y,
        threshold=0.5,
        n_resamples=1_000,
        early_stopping="adaptive",
        alpha=0.05,
        random_state=1718,
    )

    assert n_permutations == 64
    assert p_value == pytest.approx(1 / 65)


def test_multi_selector_count_matches_joint_statistic_invocations() -> None:
    calls = [0, 0]

    def first(
        x: np.ndarray,
        y: np.ndarray,
        standardize: bool,
        random_state: int,
    ) -> float:
        calls[0] += 1
        return float(np.corrcoef(x, y)[0, 1])

    def second(
        x: np.ndarray,
        y: np.ndarray,
        standardize: bool,
        random_state: int,
    ) -> float:
        calls[1] += 1
        return float(abs(np.mean(x * y)))

    rng = np.random.default_rng(1718)
    x = rng.normal(size=60)
    y = rng.normal(size=60)
    _, n_permutations = _ptest_multi_result(
        funcs=[first, second],
        func_args=[True, True],
        x=x,
        y=y,
        n_resamples=39,
        early_stopping="adaptive",
        alpha=0.05,
        random_state=1718,
    )

    assert calls == [n_permutations + 1, n_permutations + 1]


def test_parallel_selector_results_report_fixed_and_adaptive_counts() -> None:
    rng = np.random.default_rng(1718)
    x = rng.normal(size=80)
    y = (x + rng.normal(size=80) > 0).astype(np.int64)

    fixed = _ptest_mc_parallel_result(x, y, 2, 200, 1718)
    adaptive = _ptest_mc_parallel_batched_result(x, y, 2, 200, 1718, 0.05, 0.95)

    assert fixed[1] == 200
    assert adaptive[1] == 2 * SELECTOR_ADAPTIVE_BATCH_SIZE
    assert adaptive[0] == pytest.approx(1 / (adaptive[1] + 1))


def test_parallel_splitter_result_reports_fixed_count() -> None:
    rng = np.random.default_rng(1718)
    x = rng.normal(size=80)
    y = x + rng.normal(size=80)

    _, n_permutations = _ptest_mse_parallel_result(x, y, 0.0, 200, 1718)

    assert n_permutations == 200


@pytest.mark.parametrize(
    ("test", "kwargs"),
    [
        (
            ptest_mc,
            {"n_classes": 2},
        ),
        (
            ptest_mi,
            {"n_classes": 2},
        ),
        (
            ptest_pc,
            {"standardize": True},
        ),
        (
            ptest_dc,
            {"standardize": True},
        ),
        (
            ptest_rdc_classifier,
            {"n_classes": 2},
        ),
        (
            ptest_rdc_regressor,
            {"standardize": True},
        ),
    ],
)
def test_public_selector_ptests_return_float(
    test: Callable[..., float],
    kwargs: dict[str, object],
) -> None:
    rng = np.random.default_rng(1718)
    x = rng.normal(size=40)
    y = (rng.normal(size=40) > 0).astype(np.int64) if "n_classes" in kwargs else rng.normal(size=40)

    result = test(
        x=x,
        y=y,
        n_resamples=39,
        early_stopping="adaptive",
        alpha=0.05,
        random_state=1718,
        **kwargs,
    )

    assert type(result) is float


@pytest.mark.parametrize(
    ("test", "classification"),
    [
        (ptest_gini, True),
        (ptest_entropy, True),
        (ptest_mse, False),
        (ptest_mae, False),
    ],
)
def test_public_splitter_ptests_return_float(
    test: Callable[..., float],
    classification: bool,
) -> None:
    rng = np.random.default_rng(1718)
    x = rng.normal(size=40)
    y = (rng.normal(size=40) > 0).astype(np.int64) if classification else rng.normal(size=40)

    result = test(
        x=x,
        y=y,
        threshold=0.0,
        n_resamples=39,
        early_stopping="simple",
        alpha=0.05,
        random_state=1718,
    )

    assert type(result) is float


def test_collector_totals_parallel_selector_and_splitter_results() -> None:
    rng = np.random.default_rng(1718)
    x = rng.normal(size=80)
    y_class = (x + rng.normal(size=80) > 0).astype(np.int64)
    y_regression = x + rng.normal(size=80)
    expected_selector = _ptest_mc_parallel_batched_result(x, y_class, 2, 200, 1718, 0.05, 0.95)[1]
    expected_splitter = _ptest_mse_parallel_result(x, y_regression, 0.0, 200, 1718)[1]

    with collect_permutation_counts() as counts:
        ptest_mc(
            x=x,
            y=y_class,
            n_classes=2,
            n_resamples=200,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=1718,
        )
        ptest_mse(
            x=x,
            y=y_regression,
            threshold=0.0,
            n_resamples=200,
            early_stopping=None,
            alpha=0.05,
            random_state=1718,
        )

    assert counts == {
        "selector": expected_selector,
        "splitter": expected_splitter,
    }


def test_estimator_counts_multi_selector_path_and_resets_on_refit() -> None:
    rng = np.random.default_rng(1718)
    X = rng.normal(size=(80, 2))
    y = (X[:, 0] + rng.normal(size=80) > 0).astype(np.int64)
    model = ConditionalInferenceTreeClassifier(
        selector=["mc", "rdc"],
        n_resamples_selector=39,
        n_resamples_splitter=None,
        early_stopping_selector="adaptive",
        early_stopping_splitter=None,
        feature_muting=False,
        feature_scanning=False,
        threshold_scanning=False,
        max_depth=1,
        random_state=1718,
    )

    model.fit(X, y)
    first = dict(model.realized_permutation_counts_)
    model.set_params(n_resamples_selector=None, early_stopping_selector=None)
    model.fit(X, y)

    assert first["selector"] > 0
    assert first["splitter"] == 0
    assert model.realized_permutation_counts_ == {"selector": 0, "splitter": 0}
