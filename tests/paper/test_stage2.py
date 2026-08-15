"""Tests for paper/benchmark/pipeline/stage2.py (evaluation)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from paper.benchmark.adapters.store import LoadedArtifact
from paper.benchmark.config.constants import LOGISTIC_REGRESSION_MAX_ITER
from paper.benchmark.pipeline.stage2 import (
    compute_roc_auc,
    evaluate_fold,
    get_clf_models,
    get_requested_evaluation_k_values,
    metrics_cover_requested_k_values,
    resolve_evaluation_k_values,
    run_evaluation,
)

pytestmark = pytest.mark.paper


class TestComputeRocAuc:
    """Tests for compute_roc_auc function."""

    def test_binary_labels(self):
        """Test ROC AUC computation with binary labels {1, 2}."""
        y_true = np.array([1, 2, 1, 2, 2, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
        classes = np.array([1, 2])

        expected = compute_roc_auc(y_true, y_proba, classes)
        # Manually binarize using class 2 as positive
        y_bin = (y_true == 2).astype(int)
        manual = roc_auc_score(y_bin, y_proba)

        assert np.isfinite(expected)
        assert expected == manual

    def test_single_class_returns_nan(self):
        """Test that single class in y_true returns NaN."""
        y_true = np.array([1, 1, 1, 1])
        y_proba = np.array([0.2, 0.3, 0.1, 0.4])
        classes = np.array([1, 2])

        result = compute_roc_auc(y_true, y_proba, classes)
        assert np.isnan(result)

    def test_multiclass_missing_class_returns_nan(self):
        """Test that missing class in multiclass returns NaN."""
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.6, 0.3, 0.1],
                [0.2, 0.7, 0.1],
            ]
        )
        classes = np.array([0, 1, 2])

        result = compute_roc_auc(y_true, y_proba, classes)
        assert np.isnan(result)


class TestEvaluateFold:
    """Tests for evaluate_fold function."""

    def test_classification_metrics_schema(self):
        """Test that classification evaluation returns required metrics."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 5))
        y = (X[:, 0] + rng.normal(scale=0.1, size=40) > 0).astype(int)

        X_train, X_test = X[:30], X[30:]
        y_train, y_test = y[:30], y[30:]
        ranking = np.arange(X.shape[1])

        results = evaluate_fold(
            X_train,
            y_train,
            X_test,
            y_test,
            ranking,
            task="classification",
            random_state=0,
            n_jobs=1,
        )

        row = results[0]
        required = {
            "accuracy",
            "f1",
            "f1_macro",
            "balanced_accuracy",
            "roc_auc",
            "auc",
        }
        missing = required - set(row.keys())
        assert not missing, f"Missing metrics: {missing}"

    def test_logistic_regression_uses_convergence_safe_iteration_budget(self):
        """The canonical LR model must use the native-audited iteration budget."""
        model = get_clf_models(random_state=1718)["lr"]

        assert model.max_iter == LOGISTIC_REGRESSION_MAX_ITER == 5_000

    def test_regression_metrics_schema(self):
        """Test that regression evaluation returns required metrics."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 5))
        y = X[:, 0] * 0.5 + rng.normal(scale=0.1, size=40)

        X_train, X_test = X[:30], X[30:]
        y_train, y_test = y[:30], y[30:]
        ranking = np.arange(X.shape[1])

        results = evaluate_fold(
            X_train,
            y_train,
            X_test,
            y_test,
            ranking,
            task="regression",
            random_state=0,
            n_jobs=1,
        )

        row = results[0]
        required = {"r2", "mse", "rmse", "mae"}
        missing = required - set(row.keys())
        assert not missing, f"Missing metrics: {missing}"

    def test_custom_k_values_are_honored(self):
        """Evaluation should use the requested k schedule rather than the hard-coded defaults."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 8))
        y = (X[:, 0] + rng.normal(scale=0.1, size=40) > 0).astype(int)

        results = evaluate_fold(
            X[:30],
            y[:30],
            X[30:],
            y[30:],
            np.arange(X.shape[1]),
            task="classification",
            random_state=0,
            k_values=[3, 8],
            n_jobs=1,
        )

        observed_k = sorted({row["k"] for row in results})
        assert observed_k == [3, 8]

    def test_short_ranking_records_actual_selected_feature_count(self):
        """A requested budget above ranking length must report the indexed columns."""
        rng = np.random.default_rng(1718)
        X = rng.normal(size=(40, 5))
        y = X[:, 0] - 0.5 * X[:, 2] + rng.normal(scale=0.1, size=40)

        results = evaluate_fold(
            X[:30],
            y[:30],
            X[30:],
            y[30:],
            np.array([2, 0, 1]),
            task="regression",
            random_state=0,
            k_values=[5],
            n_jobs=1,
        )

        assert {row["k"] for row in results} == {5}
        assert {row["n_features_selected"] for row in results} == {3}


class TestEvaluationKBudgets:
    """Tests for Stage 2 k-budget scheduling helpers."""

    def test_resolve_k_values_clips_deduplicates_and_adds_endpoint(self):
        """The k schedule should merge defaults, extras, fractions, and endpoint cleanly."""
        result = resolve_evaluation_k_values(
            80,
            base_k_values=[5, 10, 25],
            extra_k_values=[10, 40, 120],
            extra_k_fractions=[0.5, 1.0],
        )
        assert result == [5, 10, 25, 40, 80]

    def test_metrics_cover_requested_k_values_detects_missing_budget(self):
        """Existing metrics should be considered incomplete when any requested k is absent."""
        import pandas as pd

        metrics = pd.DataFrame({"k": [5, 10, 25, 50, 100]})
        assert metrics_cover_requested_k_values(metrics, [5, 10, 25])
        assert not metrics_cover_requested_k_values(metrics, [5, 10, 250])

    def test_high_p_defaults_add_bridge_without_manual_overrides(self):
        """High-p datasets should automatically add bridge budgets above the standard schedule."""
        assert get_requested_evaluation_k_values(80) == [5, 10, 25, 50, 80]
        assert get_requested_evaluation_k_values(1200) == [
            5,
            10,
            25,
            50,
            100,
            150,
            200,
            300,
            500,
            600,
            750,
            900,
            1000,
            1200,
        ]


def test_stage2_keys_rankings_by_fold_id_when_rows_are_shuffled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ranking row order must not change the fold-to-ranking assignment."""
    from paper.benchmark.pipeline import stage2
    from paper.benchmark.pipeline.types import (
        DatasetIdentity,
        ExperimentConfig,
        MethodConfig,
    )

    config = ExperimentConfig(
        method=MethodConfig("rf"),
        dataset="fixture",
        seed=3,
        task="classification",
        dataset_identity=DatasetIdentity("d" * 64, n_samples=50, n_features=5),
    )
    rng = np.random.default_rng(1718)
    X = rng.normal(size=(50, 5))
    y = np.array([0, 1] * 25)
    rankings = pd.DataFrame(
        {
            "fold_idx": list(range(5)),
            "feature_ranking": [np.roll(np.arange(5), fold).tolist() for fold in range(5)],
        }
    )

    def fake_evaluate_fold(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        ranking: np.ndarray,
        task: str,
        random_state: int,
        k_values: list[int],
        n_jobs: int,
    ) -> list[dict[str, object]]:
        del X_train, y_train, X_test, y_test, task, k_values, n_jobs
        return [
            {
                "k": 1,
                "n_features_selected": 1,
                "downstream_model": "fixture",
                "first_feature": int(ranking[0]),
                "evaluation_random_state": random_state,
            }
        ]

    monkeypatch.setattr(stage2, "evaluate_fold", fake_evaluate_fold)
    ordered = run_evaluation(X, y, rankings, config, n_jobs=1)
    shuffled = run_evaluation(
        X,
        y,
        rankings.sample(frac=1, random_state=99).reset_index(drop=True),
        config,
        n_jobs=1,
    )

    def fold_key(row: dict[str, object]) -> int:
        return int(row["fold_idx"])

    assert sorted(ordered, key=fold_key) == sorted(shuffled, key=fold_key)
    assert {int(row["fold_idx"]): int(row["first_feature"]) for row in shuffled} == {
        fold: int(np.roll(np.arange(5), fold)[0]) for fold in range(5)
    }


def test_failed_evaluation_records_elapsed_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed Stage 2 result must retain its measured wall-clock duration."""
    from paper.benchmark.pipeline import stage2
    from paper.benchmark.pipeline.types import (
        DatasetIdentity,
        ExperimentConfig,
        MethodConfig,
    )

    config = ExperimentConfig(
        method=MethodConfig("rf"),
        dataset="fixture",
        seed=0,
        task="classification",
        dataset_identity=DatasetIdentity("d" * 64, n_samples=20, n_features=4),
    )
    X = np.arange(80, dtype=float).reshape(20, 4)
    y = np.array([0, 1] * 10)

    class RankingStore:
        def exists(self, stage: str, cfg: ExperimentConfig) -> bool:
            del cfg
            return stage == "rankings"

        def load_with_payload_sha256(
            self,
            stage: str,
            cfg: ExperimentConfig,
        ) -> LoadedArtifact:
            del stage, cfg
            return LoadedArtifact(
                frame=pd.DataFrame({"fold_idx": [0]}),
                payload_sha256="c" * 64,
            )

    def fail_evaluation(*args: object, **kwargs: object) -> list[dict[str, object]]:
        del args, kwargs
        raise RuntimeError("deterministic evaluation failure")

    clock = iter([100.0, 101.0, 103.5])
    monkeypatch.setattr(stage2.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(stage2, "get_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(
        stage2,
        "get_benchmark_scope",
        lambda: {
            "artifact_prefix": "repairs/test",
            "campaign_sha256": "b" * 64,
            "canonical_manifest_sha256": "c" * 64,
            "gate_receipt_sha256": "d" * 64,
            "manifest_sha256": "e" * 64,
            "runtime_contract_sha256": "f" * 64,
            "aws_account_id": "123456789012",
        },
    )
    monkeypatch.setattr(
        stage2,
        "get_container_image",
        lambda: "repository@sha256:" + "1" * 64,
    )
    monkeypatch.setattr(stage2, "validate_ranking_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr(stage2, "validate_artifact_provenance", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        stage2,
        "load_dataset",
        lambda dataset, task, *, identity: (X, y),
    )
    monkeypatch.setattr(
        stage2,
        "get_dataset_metadata",
        lambda dataset, task, *, identity: {
            "dataset_source": "fixture",
            "dataset_type": "real",
            "dataset_family": "test",
            "n_informative": 1,
        },
    )
    monkeypatch.setattr(stage2, "run_evaluation", fail_evaluation)

    result = stage2._run_evaluation(config, RankingStore())  # type: ignore[arg-type]

    assert result.is_failure
    assert result.error_type == "RuntimeError"
    assert result.elapsed_seconds == pytest.approx(3.5)


@pytest.mark.parametrize(
    ("duplicate_upload", "expected_status"),
    [(False, "done"), (True, "skipped")],
)
def test_run_evaluation_validates_and_round_trips_complete_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    duplicate_upload: bool,
    expected_status: str,
) -> None:
    """The real Stage 2 entry point must save a validator-complete artifact."""
    from paper.benchmark.config.constants import PIPELINE_ARTIFACT_VERSION
    from paper.benchmark.pipeline import stage2
    from paper.benchmark.pipeline.types import (
        DatasetIdentity,
        ExperimentConfig,
        MethodConfig,
    )
    from paper.benchmark.pipeline.validation import validate_metrics_artifact

    config = ExperimentConfig(
        method=MethodConfig("rf", params=(("n_estimators", 100),)),
        dataset="fixture",
        seed=2,
        task="classification",
        dataset_identity=DatasetIdentity("d" * 64, n_samples=80, n_features=5),
    )
    rng = np.random.default_rng(1718)
    X = rng.normal(size=(80, 5))
    y = (X[:, 0] + 0.25 * X[:, 1] > 0).astype(int)
    common = {
        "artifact_version": PIPELINE_ARTIFACT_VERSION,
        "artifact_prefix": "repairs/run-001",
        "aws_account_id": "123456789012",
        "campaign_sha256": "e" * 64,
        "canonical_manifest_sha256": "c" * 64,
        "container_image": "repository@sha256:" + "a" * 64,
        "created_at_utc": "2026-08-03T12:00:00+00:00",
        "dataset": config.dataset,
        "dataset_sha256": config.dataset_identity.sha256,
        "gate_receipt_sha256": "d" * 64,
        "git_sha": "a" * 40,
        "hardware": {"logical_cpus": 32, "cpu_affinity": list(range(32))},
        "library_versions": {"python": "3.12.7"},
        "method": config.method.label,
        "method_base": config.method.name,
        "method_id": config.method.label,
        "method_params_json": json.dumps(
            config.method.params_dict,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "manifest_sha256": "b" * 64,
        "runtime_contract_sha256": "f" * 64,
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "seed": config.seed,
        "task": config.task,
    }
    rankings = pd.DataFrame(
        [
            {
                **common,
                "feature_ranking": list(range(X.shape[1])),
                "fold_cpu_affinity": list(range(32)),
                "fold_idx": fold,
                "fold_random_state": config.seed * 1000 + fold,
            }
            for fold in range(5)
        ]
    )

    class RoundTripStore:
        def __init__(self) -> None:
            self.saved: pd.DataFrame | None = None

        def exists(self, stage: str, cfg: ExperimentConfig) -> bool:
            del cfg
            return stage == "rankings"

        def load(self, stage: str, cfg: ExperimentConfig) -> pd.DataFrame:
            del cfg
            if stage == "rankings":
                return rankings.copy()
            if stage == "metrics" and self.saved is not None:
                return self.saved.copy()
            raise AssertionError(f"unexpected load: {stage}")

        def load_with_payload_sha256(
            self,
            stage: str,
            cfg: ExperimentConfig,
        ) -> LoadedArtifact:
            return LoadedArtifact(
                frame=self.load(stage, cfg),
                payload_sha256="c" * 64,
            )

        def save(self, stage: str, cfg: ExperimentConfig, frame: pd.DataFrame) -> str:
            del cfg
            assert stage == "metrics"
            path = tmp_path / "metrics.parquet"
            frame.to_parquet(path, index=False)
            self.saved = pd.read_parquet(path)
            if duplicate_upload:
                raise FileExistsError("another worker uploaded first")
            return str(path)

    store = RoundTripStore()
    monkeypatch.setattr(
        stage2,
        "load_dataset",
        lambda dataset, task, *, identity: (X, y),
    )
    monkeypatch.setattr(
        stage2,
        "get_dataset_metadata",
        lambda dataset, task, *, identity: {
            "dataset_source": "fixture",
            "dataset_type": "real",
            "dataset_family": "test",
            "n_informative": 2,
        },
    )
    monkeypatch.setattr(stage2, "get_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(stage2, "get_library_versions", lambda: {"python": "3.12.7"})
    monkeypatch.setattr(stage2, "get_hardware_metadata", lambda: {"logical_cpus": 32})
    monkeypatch.setattr(
        stage2,
        "get_container_image",
        lambda: "repository@sha256:" + "a" * 64,
    )
    monkeypatch.setattr(
        stage2,
        "get_benchmark_scope",
        lambda: {
            "artifact_prefix": "repairs/run-001",
            "campaign_sha256": "e" * 64,
            "canonical_manifest_sha256": "c" * 64,
            "gate_receipt_sha256": "d" * 64,
            "manifest_sha256": "b" * 64,
            "runtime_contract_sha256": "f" * 64,
            "aws_account_id": "123456789012",
        },
    )
    monkeypatch.setattr(stage2, "utc_now_iso", lambda: "2026-08-03T13:00:00+00:00")

    result = stage2._run_evaluation(config, store)  # type: ignore[arg-type]

    assert result.status == expected_status, result.error
    assert store.saved is not None
    assert store.saved["dataset_sha256"].unique().tolist() == ["d" * 64]
    assert store.saved["canonical_manifest_sha256"].unique().tolist() == ["c" * 64]
    assert store.saved["gate_receipt_sha256"].unique().tolist() == ["d" * 64]
    assert store.saved["ranking_dataset_sha256"].unique().tolist() == ["d" * 64]
    assert store.saved["ranking_canonical_manifest_sha256"].unique().tolist() == ["c" * 64]
    assert store.saved["ranking_gate_receipt_sha256"].unique().tolist() == ["d" * 64]
    assert store.saved["ranking_payload_sha256"].unique().tolist() == ["c" * 64]
    validate_metrics_artifact(store.saved, config)
