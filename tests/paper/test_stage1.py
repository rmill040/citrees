"""Tests for paper/benchmark/pipeline/stage1.py (feature selection)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from citrees._selector import (
    ClassifierSelectors,
    ClassifierSelectorTests,
    RegressorSelectors,
    RegressorSelectorTests,
)
from paper.benchmark.pipeline.r_methods import (
    _MAX_SUPPORTED_RANDOM_STATE,
    _normalize_r_seed,
    _ranking_from_named_scores,
    _resolve_cores,
    _resolve_mtry,
)

pytestmark = pytest.mark.paper


def test_concurrent_valid_ranking_upload_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that loses an immutable upload race must validate and skip."""
    from paper.benchmark.pipeline import stage1
    from paper.benchmark.pipeline.types import DatasetIdentity, ExperimentConfig, MethodConfig

    config = ExperimentConfig(
        method=MethodConfig("dt"),
        dataset="fixture",
        seed=0,
        task="classification",
        dataset_identity=DatasetIdentity("d" * 64, n_samples=20, n_features=4),
    )
    X = np.arange(80, dtype=float).reshape(20, 4)
    y = np.array([0, 1] * 10)

    class ConcurrentStore:
        def __init__(self) -> None:
            self.saved: pd.DataFrame | None = None

        def exists(self, stage: str, cfg: ExperimentConfig) -> bool:
            return False

        def save(
            self,
            stage: str,
            cfg: ExperimentConfig,
            frame: pd.DataFrame,
        ) -> str:
            self.saved = frame.copy()
            raise FileExistsError("another worker uploaded first")

        def load(self, stage: str, cfg: ExperimentConfig) -> pd.DataFrame:
            assert self.saved is not None
            return self.saved.copy()

    monkeypatch.setattr(
        stage1,
        "load_dataset",
        lambda dataset, task, *, identity: (X, y),
    )
    monkeypatch.setattr(
        stage1,
        "get_dataset_metadata",
        lambda dataset, task, *, identity: {
            "dataset_source": "fixture",
            "dataset_type": "real",
            "dataset_family": "test",
            "n_informative": 1,
        },
    )
    monkeypatch.setattr(
        stage1,
        "run_selection",
        lambda *args, **kwargs: [
            {
                "fold_idx": fold,
                "fold_random_state": fold,
                "feature_ranking": list(range(X.shape[1])),
            }
            for fold in range(5)
        ],
    )
    monkeypatch.setattr(stage1, "get_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(stage1, "get_library_versions", lambda: {"python": "3.12.7"})
    monkeypatch.setattr(stage1, "get_hardware_metadata", lambda: {"logical_cpus": 1})
    monkeypatch.setattr(
        stage1,
        "get_container_image",
        lambda: "repository@sha256:" + "a" * 64,
    )
    monkeypatch.setattr(
        stage1,
        "get_benchmark_scope",
        lambda: {
            "artifact_prefix": "repairs/run-001",
            "campaign_sha256": "e" * 64,
            "manifest_sha256": "b" * 64,
            "aws_account_id": "123456789012",
        },
    )

    result = stage1._run_selection(config, ConcurrentStore())  # type: ignore[arg-type]

    assert result.is_skipped
    assert result.data is not None


class TestFilterSelector:
    """Tests for filter_selector function."""

    def test_label_invariance(self) -> None:
        """Filter selector should be invariant to label shift."""
        from paper.benchmark.pipeline.stage1 import filter_selector

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
        from paper.benchmark.pipeline.stage1 import permutation_selector

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
        from paper.benchmark.pipeline.stage1 import permutation_selector

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


class TestEmbeddingSelector:
    """Tests for tree-based embedding selector wiring."""

    def test_single_tree_factories_use_sklearn_tree_estimators(self) -> None:
        """dt and rt should be single sklearn trees, not ensemble estimators."""
        from sklearn.tree import (
            DecisionTreeClassifier,
            DecisionTreeRegressor,
            ExtraTreeClassifier,
            ExtraTreeRegressor,
        )

        from paper.benchmark.pipeline.selectors import get_embedding_model

        clf_dt = get_embedding_model("dt", "classification", random_state=1718)
        clf_rt = get_embedding_model("rt", "classification", random_state=1718)
        reg_dt = get_embedding_model("dt", "regression", random_state=1718)
        reg_rt = get_embedding_model("rt", "regression", random_state=1718)

        assert isinstance(clf_dt, DecisionTreeClassifier)
        assert isinstance(clf_rt, ExtraTreeClassifier)
        assert isinstance(reg_dt, DecisionTreeRegressor)
        assert isinstance(reg_rt, ExtraTreeRegressor)
        assert clf_dt.splitter == "best"
        assert clf_rt.splitter == "random"
        assert not hasattr(clf_rt, "n_estimators")
        assert not hasattr(reg_rt, "n_estimators")

    def test_run_selection_accepts_dt_and_rt(self) -> None:
        """Stage 1 should route dt and rt through embedding selection."""
        from paper.benchmark.pipeline.stage1 import run_selection

        rng = np.random.default_rng(1718)
        X = rng.normal(size=(30, 4))
        y = np.array([0, 1] * 15, dtype=int)

        for method in ("dt", "rt"):
            results = run_selection(
                X,
                y,
                method=method,
                task="classification",
                seed=0,
                n_jobs=1,
            )

            assert len(results) == 5
            assert all(row["feature_ranking"] for row in results)

    @pytest.mark.parametrize("method", ["dt", "rt"])
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("signal_index", [0, 4])
    def test_single_tree_rankings_preserve_known_feature_identity(
        self,
        method: str,
        task: str,
        signal_index: int,
    ) -> None:
        """DT and RT must rank the original signal column first in every fold."""
        from paper.benchmark.pipeline.stage1 import run_selection

        signal = np.linspace(-2.0, 2.0, 200)
        X = np.zeros((signal.size, 5), dtype=np.float64)
        X[:, signal_index] = signal
        y = (signal > 0).astype(np.int64) if task == "classification" else signal

        results = run_selection(
            X,
            y,
            method=method,
            task=task,
            seed=0,
            n_jobs=1,
        )

        assert len(results) == 5
        for row in results:
            ranking = row["feature_ranking"]
            assert ranking[0] == signal_index
            assert sorted(ranking) == list(range(X.shape[1]))


class TestWrapperSelectors:
    """Tests for wrapper selector scoring behavior."""

    def test_pi_selector_uses_balanced_accuracy_for_classification(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PI should align its internal scorer with the benchmark headline metric."""
        from paper.benchmark.pipeline import selectors

        X = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.1, 0.9],
                [0.9, 0.1],
            ],
            dtype=float,
        )
        y = np.array([0, 1, 0, 1, 0, 1], dtype=int)
        captured: dict[str, object] = {}

        def fake_permutation_importance(model, X_val, y_val, **kwargs):
            captured["scoring"] = kwargs["scoring"]
            return SimpleNamespace(importances_mean=np.array([0.2, 0.1], dtype=float))

        monkeypatch.setattr(selectors, "permutation_importance", fake_permutation_importance)

        ranking = selectors.pi_selector(
            X,
            y,
            task="classification",
            random_state=0,
            n_jobs=1,
            val_fraction=0.5,
            params={"n_repeats": 1},
        )

        assert ranking.tolist() == [0, 1]
        assert captured["scoring"] == "balanced_accuracy"

    def test_cpi_selector_uses_balanced_accuracy_for_classification(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CPI should rank minority-recall signal ahead of majority-only signal."""
        from paper.benchmark.pipeline import selectors

        class DummyParallel:
            def __init__(self, n_jobs: int):
                self.n_jobs = n_jobs

            def __call__(self, tasks):
                return [fn(*args, **kwargs) for fn, args, kwargs in tasks]

        def dummy_delayed(fn):
            def wrapped(*args, **kwargs):
                return fn, args, kwargs

            return wrapped

        class DummyRF:
            def __init__(self, *args, **kwargs):
                pass

            def fit(self, X, y):
                return self

            def predict(self, X):
                x0 = X[:, 0]
                x1 = X[:, 1]
                return np.where(x0 > 0.5, 0, np.where(x1 > 0.5, 1, 0)).astype(int)

        class DummyRng:
            def __init__(self, seed: int):
                self.seed = seed

            def permutation(self, values):
                if self.seed == 1718:
                    return np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 1], dtype=values.dtype)
                if self.seed == 1719:
                    return np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=values.dtype)
                raise AssertionError(f"Unexpected seed {self.seed}")

        X_fit = np.zeros((4, 2), dtype=float)
        y_fit = np.array([0, 0, 1, 1], dtype=int)
        X_val = np.array([[1.0, 1.0]] * 8 + [[0.0, 1.0], [0.0, 1.0]], dtype=float)
        y_val = np.array([0] * 8 + [1, 1], dtype=int)

        def fake_train_test_split(X, y, **kwargs):
            return X_fit.copy(), X_val.copy(), y_fit.copy(), y_val.copy()

        monkeypatch.setattr(selectors, "RandomForestClassifier", DummyRF)
        monkeypatch.setattr(selectors, "Parallel", DummyParallel)
        monkeypatch.setattr(selectors, "delayed", dummy_delayed)
        monkeypatch.setattr(selectors, "train_test_split", fake_train_test_split)
        monkeypatch.setattr(selectors.np.random, "default_rng", lambda seed: DummyRng(seed))

        ranking = selectors.cpi_selector(
            np.zeros((1, 2), dtype=float),
            np.array([0], dtype=int),
            task="classification",
            random_state=1718,
            n_jobs=1,
            params={"n_repeats": 1},
        )

        assert ranking.tolist() == [1, 0]

    def test_boruta_uses_importance_history_within_decision_tiers(self) -> None:
        """Tied Boruta decisions must not collapse to original column order."""
        from sklearn.datasets import load_wine

        from paper.benchmark.pipeline.selectors import boruta_selector

        wine = load_wine()
        permutation = np.arange(wine.data.shape[1])[::-1]
        original = boruta_selector(
            wine.data,
            wine.target,
            task="classification",
            random_state=1718,
            n_jobs=1,
            params={"n_estimators": "auto", "max_iter": 200},
        )
        permuted = boruta_selector(
            wine.data[:, permutation],
            wine.target,
            task="classification",
            random_state=1718,
            n_jobs=1,
            params={"n_estimators": "auto", "max_iter": 200},
        )

        assert set(original[:5]) == set(permutation[permuted[:5]])

    def test_boruta_orders_each_decision_tier_by_descending_importance(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Boruta ties must have an exact, importance-descending total order."""
        import boruta

        from paper.benchmark.pipeline import selectors

        class DummyBoruta:
            def __init__(self, *args: object, **kwargs: object) -> None:
                del args, kwargs

            def fit(self, X: np.ndarray, y: np.ndarray) -> DummyBoruta:
                del X, y
                self.ranking_ = np.array([1, 1, 2, 2])
                self.importance_history_ = np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan],
                        [0.2, 0.8, 0.1, 0.6],
                        [0.4, 1.0, 0.3, 0.8],
                    ]
                )
                return self

        monkeypatch.setattr(boruta, "BorutaPy", DummyBoruta)
        ranking = selectors.boruta_selector(
            np.zeros((8, 4)),
            np.array([0, 1] * 4),
            task="classification",
            random_state=1718,
            n_jobs=1,
        )

        assert ranking.tolist() == [1, 0, 3, 2]

    def test_rfe_step_is_explicit_in_method_identity(self) -> None:
        """RFE artifacts must identify the one-at-a-time elimination algorithm."""
        from paper.benchmark.pipeline.methods import get_full_method_configs

        for task in ("classification", "regression"):
            configs = get_full_method_configs(["rfe"], task)
            assert len(configs) == 1
            assert configs[0].params_dict == {"step": 1}

    def test_cforest_core_count_is_explicit_in_method_identity(self) -> None:
        """Production cforest variants must not depend on worker CPU count."""
        from paper.benchmark.pipeline.methods import get_full_method_configs

        for task in ("classification", "regression"):
            configs = get_full_method_configs(["r_cforest"], task)
            assert configs
            assert all(config.params_dict["cores"] == 1 for config in configs)

    def test_rfe_uses_and_requires_single_feature_steps(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """RFE must produce a total elimination order and reject batched steps."""
        from paper.benchmark.pipeline import selectors

        captured: dict[str, object] = {}

        class DummyRFE:
            def __init__(self, estimator, *, n_features_to_select: int, step: int):
                del estimator
                captured["n_features_to_select"] = n_features_to_select
                captured["step"] = step

            def fit(self, X: np.ndarray, y: np.ndarray):
                del y
                self.ranking_ = np.arange(1, X.shape[1] + 1)
                return self

        monkeypatch.setattr(selectors, "RFE", DummyRFE)
        X = np.arange(24, dtype=float).reshape(6, 4)
        y = np.array([0, 1, 0, 1, 0, 1])

        ranking = selectors.rfe_selector(
            X,
            y,
            task="classification",
            random_state=1718,
            n_jobs=1,
            params={"step": 1},
        )

        assert ranking.tolist() == [0, 1, 2, 3]
        assert captured == {"n_features_to_select": 1, "step": 1}
        with pytest.raises(ValueError, match="requires RFE step=1"):
            selectors.rfe_selector(
                X,
                y,
                task="classification",
                random_state=1718,
                n_jobs=1,
                params={"step": 0.1},
            )


# ---------------------------------------------------------------------------
# r_ctree / r_cforest
# ---------------------------------------------------------------------------

try:
    from paper.benchmark.pipeline.r_methods import (
        _get_partykit,
        r_cforest_ranking,
        r_ctree_ranking,
        r_ctree_root_feature,
    )

    _get_partykit()
    _has_r = True
except (ImportError, OSError):
    _has_r = False

_skip_no_r = pytest.mark.skipif(not _has_r, reason="R / rpy2 / partykit not available")


class TestRBoundaryHelpers:
    """Unit tests for values crossing between Python and partykit."""

    def test_seed_mapping_covers_supported_boundaries_without_collisions(self) -> None:
        random_states = [0, 1, 2_147_483_646, 2_147_483_647, 2_147_483_648, 4_294_967_294]

        r_seeds = [_normalize_r_seed(value) for value in random_states]

        assert r_seeds == [0, 1, 2_147_483_646, 2_147_483_647, -2_147_483_647, -1]
        assert len(set(r_seeds)) == len(r_seeds)

    @pytest.mark.parametrize("random_state", [-1, _MAX_SUPPORTED_RANDOM_STATE + 1])
    def test_seed_mapping_rejects_out_of_range_values(self, random_state: int) -> None:
        with pytest.raises(ValueError, match="random_state must be between"):
            _normalize_r_seed(random_state)

    @pytest.mark.parametrize("random_state", [True, 1.5, "1718"])
    def test_seed_mapping_rejects_non_integer_values(self, random_state: object) -> None:
        with pytest.raises(TypeError, match="random_state must be an integer"):
            _normalize_r_seed(random_state)  # type: ignore[arg-type]

    def test_sparse_named_scores_restore_original_feature_indices(self) -> None:
        ranking = _ranking_from_named_scores(
            ["X3", "X1", "X4"],
            np.array([2.0, 2.0, -1.0]),
            n_features=5,
        )

        assert ranking.tolist() == [1, 3, 0, 2, 4]

    @pytest.mark.parametrize(
        ("feature_names", "scores", "match"),
        [
            (["X0"], np.array([1.0, 2.0]), "different numbers"),
            (["X0", "X1"], np.array([[1.0, 2.0]]), "non-vector"),
            (["X0"], np.array([np.nan]), "non-finite"),
            (["feature0"], np.array([1.0]), "Unexpected"),
            (["X01"], np.array([1.0]), "Unexpected"),
            (["X5"], np.array([1.0]), "out of range"),
            (["X1", "X1"], np.array([1.0, 2.0]), "duplicate"),
        ],
    )
    def test_named_scores_reject_malformed_r_output(
        self,
        feature_names: list[str],
        scores: np.ndarray,
        match: str,
    ) -> None:
        with pytest.raises(RuntimeError, match=match):
            _ranking_from_named_scores(feature_names, scores, n_features=5)

    @pytest.mark.parametrize(
        ("mtry", "n_features", "expected"),
        [
            (None, 10, 10),
            ("all", 10, 10),
            ("sqrt", 10, 4),
            ("log", 10, 4),
            ("log", 1, 1),
            (3, 10, 3),
        ],
    )
    def test_mtry_resolution(self, mtry: int | str | None, n_features: int, expected: int) -> None:
        assert _resolve_mtry(mtry, n_features) == expected

    @pytest.mark.parametrize("mtry", [0, 6, "bogus", 1.5, True])
    def test_mtry_rejects_invalid_values(self, mtry: object) -> None:
        with pytest.raises((TypeError, ValueError)):
            _resolve_mtry(mtry, 5)  # type: ignore[arg-type]

    def test_default_core_resolution_uses_detected_cpu_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("paper.benchmark.pipeline.r_methods.os.cpu_count", lambda: 8)
        assert _resolve_cores(-1) == 8

        monkeypatch.setattr("paper.benchmark.pipeline.r_methods.os.cpu_count", lambda: None)
        assert _resolve_cores(-1) == 1

    @pytest.mark.parametrize("cores", [0, -2])
    def test_core_resolution_rejects_invalid_counts(self, cores: int) -> None:
        with pytest.raises(ValueError, match="cores must be -1 or a positive integer"):
            _resolve_cores(cores)

    @_skip_no_r
    def test_r_runtime_versions_are_reported(self) -> None:
        from paper.benchmark.pipeline.r_methods import get_r_runtime_versions

        versions = get_r_runtime_versions()

        assert versions["r"].startswith("R version ")
        assert versions["partykit"]


class TestRCtreeRanking:
    """Regression tests for r_ctree_ranking (partykit node traversal)."""

    @_skip_no_r
    def test_ranking_not_identity_classification(self) -> None:
        """r_ctree must produce a non-identity ranking on separable data."""
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42,
        )
        ranking = r_ctree_ranking(X, y, task="classification")
        assert ranking.shape == (10,)
        assert not np.array_equal(ranking, np.arange(10)), (
            "ranking is identity — split extraction broken"
        )

    @_skip_no_r
    def test_ranking_not_identity_regression(self) -> None:
        """r_ctree must produce a non-identity ranking on regression data."""
        from sklearn.datasets import make_regression

        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=5,
            random_state=42,
        )
        ranking = r_ctree_ranking(X, y, task="regression")
        assert ranking.shape == (10,)
        assert not np.array_equal(ranking, np.arange(10)), (
            "ranking is identity — split extraction broken"
        )

    @_skip_no_r
    def test_stump_on_noise(self) -> None:
        """Pure noise should yield a stump (no splits) and identity ranking."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 20))
        y = rng.integers(0, 2, 50)
        ranking = r_ctree_ranking(X, y, task="classification")
        assert ranking.shape == (20,)
        # Identity is the expected fallback when no splits are made
        assert np.array_equal(ranking, np.arange(20))

    @_skip_no_r
    def test_deterministic(self) -> None:
        """Same input must give the same ranking (ctree is deterministic)."""
        from sklearn.datasets import load_iris

        iris = load_iris()
        r1 = r_ctree_ranking(iris.data, iris.target, task="classification")
        r2 = r_ctree_ranking(iris.data, iris.target, task="classification")
        assert np.array_equal(r1, r2)

    @_skip_no_r
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("signal_index", [0, 2, 4])
    def test_feature_index_matches_known_signal(self, task: str, signal_index: int) -> None:
        """R model-frame variable IDs must map to the original Python feature."""
        rng = np.random.default_rng(1718)
        signal = rng.standard_normal(240)
        X = rng.standard_normal((signal.size, 5))
        X[:, signal_index] = signal
        y = (signal > 0).astype(int) if task == "classification" else signal

        ranking = r_ctree_ranking(
            X,
            y,
            task=task,
            testtype="Univariate",
            alpha=0.05,
            minsplit=2,
            minbucket=1,
        )

        assert ranking.shape == (X.shape[1],)
        assert np.array_equal(np.sort(ranking), np.arange(X.shape[1]))
        assert ranking[0] == signal_index

    @_skip_no_r
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("random_state", [1718, 3_238_245_004])
    def test_root_feature_uses_zero_based_index(self, task: str, random_state: int) -> None:
        """The root-feature helper should return the predictive second column."""
        rng = np.random.default_rng(1718)
        signal = rng.standard_normal(200)
        X = np.column_stack([rng.standard_normal(200), signal])
        y = (signal > 0).astype(int) if task == "classification" else signal

        root_feature = r_ctree_root_feature(X, y, task=task, random_state=random_state)

        assert root_feature == 1

    @_skip_no_r
    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_root_feature_returns_minus_one_when_tree_has_no_split(self, task: str) -> None:
        """A terminal root must not be passed to partykit's split-node accessor."""
        X = np.zeros((40, 4), dtype=np.float64)
        y = (
            np.array([0, 1] * 20, dtype=np.int64)
            if task == "classification"
            else np.arange(40, dtype=np.float64)
        )

        root_feature = r_ctree_root_feature(
            X,
            y,
            task=task,
            mincriterion=1.0,
        )

        assert root_feature == -1

    @_skip_no_r
    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_monte_carlo_same_seed_is_deterministic(self, task: str) -> None:
        """Monte Carlo ctree rankings must reproduce under an identical seed."""
        rng = np.random.default_rng(1718)
        signal = rng.standard_normal(120)
        X = np.column_stack([rng.standard_normal((signal.size, 3)), signal])
        y = (signal > 0).astype(int) if task == "classification" else signal
        kwargs = {
            "task": task,
            "testtype": "MonteCarlo",
            "nresample": 99,
            "minsplit": 10,
            "minbucket": 4,
            "random_state": 42,
        }

        first = r_ctree_ranking(X, y, **kwargs)
        second = r_ctree_ranking(X, y, **kwargs)

        assert np.array_equal(first, second)


class TestRCforestRanking:
    """Regression tests for named partykit cforest importance values."""

    @_skip_no_r
    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_sparse_named_importance_maps_to_complete_ranking(self, task: str) -> None:
        """A last-column signal must retain its original Python feature index."""
        rng = np.random.default_rng(1718)
        signal = rng.standard_normal(180)
        X = np.column_stack([np.zeros((signal.size, 4)), signal])
        y = (signal > 0).astype(int) if task == "classification" else signal

        ranking = r_cforest_ranking(
            X,
            y,
            task=task,
            ntree=50,
            mtry="all",
            cores=2,
            random_state=1718,
        )

        assert ranking.shape == (X.shape[1],)
        assert np.array_equal(np.sort(ranking), np.arange(X.shape[1]))
        assert ranking[0] == X.shape[1] - 1

    @_skip_no_r
    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_parallel_same_seed_is_deterministic(self, task: str) -> None:
        """The same fold seed must reproduce with multi-core forest execution."""
        rng = np.random.default_rng(1718)
        X = rng.standard_normal((160, 6))
        signal = X[:, 5] + 0.25 * X[:, 2]
        y = (signal > 0).astype(int) if task == "classification" else signal
        kwargs = {
            "task": task,
            "ntree": 50,
            "mtry": "sqrt",
            "cores": 2,
            "random_state": 42,
        }

        rankings = [r_cforest_ranking(X, y, **kwargs) for _ in range(3)]

        assert all(np.array_equal(rankings[0], ranking) for ranking in rankings[1:])

    @_skip_no_r
    def test_default_core_path_is_reproducible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Production's cores=-1 path must use deterministic parallel streams."""
        monkeypatch.setattr("paper.benchmark.pipeline.r_methods.os.cpu_count", lambda: 2)
        rng = np.random.default_rng(1718)
        X = rng.standard_normal((160, 6))
        y = (X[:, 5] + 0.25 * X[:, 2] > 0).astype(int)
        kwargs = {
            "task": "classification",
            "ntree": 50,
            "mtry": "sqrt",
            "cores": -1,
            "random_state": 42,
        }

        first = r_cforest_ranking(X, y, **kwargs)
        second = r_cforest_ranking(X, y, **kwargs)

        assert np.array_equal(first, second)

    @_skip_no_r
    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_every_production_grid_variant_runs_through_r(
        self, task: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every configured R variant must return a semantically valid ranking."""
        from paper.benchmark.pipeline.methods import get_full_method_configs

        monkeypatch.setattr("paper.benchmark.pipeline.r_methods.os.cpu_count", lambda: 2)
        rng = np.random.default_rng(1718)
        signal = rng.standard_normal(80)
        X = np.column_stack([rng.standard_normal((signal.size, 3)), signal])
        y = (signal > 0).astype(int) if task == "classification" else signal

        configs = get_full_method_configs(["r_ctree", "r_cforest"], task)
        assert len(configs) == 6
        for config in configs:
            params = config.params_dict
            ranking = (
                r_ctree_ranking(X, y, task=task, random_state=42, **params)
                if config.name == "r_ctree"
                else r_cforest_ranking(
                    X,
                    y,
                    task=task,
                    random_state=42,
                    **params,
                )
            )

            assert np.array_equal(np.sort(ranking), np.arange(X.shape[1])), config.label
            assert ranking[0] == X.shape[1] - 1, config.label


def test_cforest_production_config_ignores_worker_cpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured cforest core count must override host CPU discovery."""
    from paper.benchmark.pipeline import r_methods
    from paper.benchmark.pipeline.methods import get_full_method_configs
    from paper.benchmark.pipeline.stage1 import run_selection

    observed_cores: list[int] = []

    def ranking(
        X: np.ndarray,
        y: np.ndarray,
        *,
        cores: int,
        **kwargs: object,
    ) -> np.ndarray:
        del y, kwargs
        observed_cores.append(cores)
        return np.arange(X.shape[1])

    monkeypatch.setattr(r_methods, "r_cforest_ranking", ranking)
    config = get_full_method_configs(["r_cforest"], "classification")[0]
    rng = np.random.default_rng(1718)
    X = rng.standard_normal((50, 4))
    y = np.array([0, 1] * 25)

    for cpu_count in (2, 64):
        monkeypatch.setattr(
            "paper.benchmark.pipeline.r_methods.os.cpu_count",
            lambda value=cpu_count: value,
        )
        run_selection(
            X,
            y,
            "r_cforest",
            "classification",
            seed=0,
            params=config.params_dict,
        )

    assert observed_cores == [1] * 10


@pytest.mark.parametrize("method", ["r_ctree", "r_cforest"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_r_methods_receive_fold_seed(
    method: str,
    task: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 1 must pass its deterministic fold seed into each R method."""
    from paper.benchmark.pipeline import r_methods
    from paper.benchmark.pipeline.stage1 import run_selection

    observed: list[int] = []

    def ranking(
        X: np.ndarray,
        y: np.ndarray,
        *,
        random_state: int,
        **kwargs: object,
    ) -> np.ndarray:
        observed.append(random_state)
        return np.arange(X.shape[1])

    monkeypatch.setattr(r_methods, f"{method}_ranking", ranking)
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 4))
    y = np.array([0, 1] * 25) if task == "classification" else rng.standard_normal(50)

    run_selection(X, y, method, task, seed=7)

    assert observed == [7000, 7001, 7002, 7003, 7004]


@pytest.mark.parametrize(
    ("ranking", "match"),
    [
        (np.array([], dtype=int), "ranking shape"),
        (np.array([0, 1, 2], dtype=int), "ranking shape"),
        (np.array([[0, 1, 2, 3]], dtype=int), "ranking shape"),
        (np.array([0.0, 1.0, 2.0, 3.0]), "non-integer"),
        (np.array([0, 1, 1, 3]), "permutation"),
        (np.array([-1, 0, 1, 2]), "permutation"),
        (np.array([0, 1, 2, 4]), "permutation"),
    ],
)
def test_stage1_rejects_malformed_rankings(
    ranking: np.ndarray,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No method may emit an incomplete or ambiguous feature ranking."""
    from paper.benchmark.pipeline import r_methods
    from paper.benchmark.pipeline.stage1 import run_selection

    def malformed_ranking(X: np.ndarray, y: np.ndarray, **kwargs: object) -> np.ndarray:
        return ranking

    monkeypatch.setattr(r_methods, "r_ctree_ranking", malformed_ranking)
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 4))
    y = np.array([0, 1] * 25)

    with pytest.raises(ValueError, match=match):
        run_selection(X, y, "r_ctree", "classification", seed=0)
