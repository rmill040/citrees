"""Tests for exact manifest-to-artifact reconciliation."""

from __future__ import annotations

import json
from itertools import product

import pandas as pd
import pytest
from typer.testing import CliRunner

from paper.benchmark.adapters.store import LoadedArtifact
from paper.benchmark.cli import manifest as manifest_cli
from paper.benchmark.cli.app import app as root_cli
from paper.benchmark.config.constants import (
    CLF_DOWNSTREAM_MODELS,
    PIPELINE_ARTIFACT_VERSION,
)
from paper.benchmark.pipeline.manifest import ManifestCell, RerunManifest
from paper.benchmark.pipeline.reconcile import (
    ArtifactIssue,
    ReconciliationReport,
    reconcile_manifest_artifacts,
)
from paper.benchmark.pipeline.types import (
    DatasetIdentity,
    ExperimentConfig,
    MethodConfig,
    StageType,
    TaskType,
)

pytestmark = pytest.mark.paper

ACCOUNT_ID = "123456789012"
ARTIFACT_PREFIX = "repairs/run-001"
CONTAINER_IMAGE = "repository@sha256:" + "a" * 64
GIT_SHA = "a" * 40
MANIFEST_SHA256 = "b" * 64
CAMPAIGN_SHA256 = "e" * 64
RANKING_PAYLOAD_SHA256 = "c" * 64
N_FEATURES = 4
N_SAMPLES = 50


def _config(
    seed: int,
    *,
    dataset: str = "glass",
    task: TaskType = "classification",
    dataset_sha256: str = "d" * 64,
) -> ExperimentConfig:
    return ExperimentConfig(
        method=MethodConfig("rf"),
        dataset=dataset,
        seed=seed,
        task=task,
        dataset_identity=DatasetIdentity(
            dataset_sha256,
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
        ),
    )


def _cell(
    seed: int,
    *,
    task: TaskType = "classification",
    dataset_sha256: str = "d" * 64,
    stage1_required: bool = True,
    stage2_required: bool = True,
) -> ManifestCell:
    return ManifestCell(
        config=_config(
            seed,
            task=task,
            dataset_sha256=dataset_sha256,
        ),
        target_aws_account_id=ACCOUNT_ID,
        dataset_source="real",
        rerun_reason="adapter_correction",
        historically_omitted=False,
        stage1_required=stage1_required,
        stage2_required=stage2_required,
    )


def _manifest(*cells: ManifestCell) -> RerunManifest:
    return RerunManifest(
        sha256=MANIFEST_SHA256,
        campaign_sha256=CAMPAIGN_SHA256,
        cells=tuple(cells),
    )


def _provenance() -> dict[str, str]:
    return {
        "artifact_prefix": ARTIFACT_PREFIX,
        "aws_account_id": ACCOUNT_ID,
        "campaign_sha256": CAMPAIGN_SHA256,
        "container_image": CONTAINER_IMAGE,
        "git_sha": GIT_SHA,
        "manifest_sha256": MANIFEST_SHA256,
    }


def _common(config: ExperimentConfig) -> dict[str, object]:
    return {
        "artifact_version": PIPELINE_ARTIFACT_VERSION,
        "artifact_prefix": ARTIFACT_PREFIX,
        "aws_account_id": ACCOUNT_ID,
        "campaign_sha256": CAMPAIGN_SHA256,
        "container_image": CONTAINER_IMAGE,
        "created_at_utc": "2026-08-03T12:00:00+00:00",
        "dataset": config.dataset,
        "dataset_sha256": config.dataset_identity.sha256,
        "git_sha": GIT_SHA,
        "hardware": {"logical_cpus": 32},
        "library_versions": {"python": "3.12.7", "sklearn": "1.8.0"},
        "method": config.method.label,
        "method_base": config.method.name,
        "method_id": config.method.label,
        "method_params_json": json.dumps(
            config.method.params_dict,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "manifest_sha256": MANIFEST_SHA256,
        "n_features": N_FEATURES,
        "n_samples": N_SAMPLES,
        "seed": config.seed,
        "task": config.task,
    }


def _rankings(config: ExperimentConfig) -> pd.DataFrame:
    common = _common(config)
    return pd.DataFrame(
        [
            {
                **common,
                "feature_ranking": list(range(N_FEATURES)),
                "fold_idx": fold,
                "fold_random_state": config.seed * 1000 + fold,
                "selection_cpus": 32,
            }
            for fold in range(5)
        ]
    )


def _metrics(config: ExperimentConfig) -> pd.DataFrame:
    common = _common(config)
    return pd.DataFrame(
        [
            {
                **common,
                "accuracy": 0.8,
                "auc": 0.85,
                "balanced_accuracy": 0.79,
                "downstream_model": model,
                "evaluation_cpus": 32,
                "f1": 0.8,
                "f1_macro": 0.79,
                "fold_idx": fold,
                "fold_random_state": config.seed * 1000 + fold,
                "k": N_FEATURES,
                "n_features_selected": N_FEATURES,
                "ranking_artifact_version": PIPELINE_ARTIFACT_VERSION,
                "ranking_artifact_prefix": ARTIFACT_PREFIX,
                "ranking_aws_account_id": ACCOUNT_ID,
                "ranking_container_image": CONTAINER_IMAGE,
                "ranking_dataset_sha256": config.dataset_identity.sha256,
                "ranking_git_sha": GIT_SHA,
                "ranking_manifest_sha256": MANIFEST_SHA256,
                "ranking_payload_sha256": RANKING_PAYLOAD_SHA256,
                "roc_auc": 0.85,
            }
            for fold, model in product(range(5), CLF_DOWNSTREAM_MODELS)
        ]
    )


class _Store:
    artifact_prefix = ARTIFACT_PREFIX

    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        keys: dict[StageType, tuple[str, ...]] | None = None,
        load_error: Exception | None = None,
    ) -> None:
        self.frames = frames
        self.keys = keys or {
            stage: tuple(sorted(key for key in frames if f"/{stage}/" in key))
            for stage in ("rankings", "metrics")
        }
        self.load_error = load_error
        self.loaded: list[str] = []

    def artifact_key(self, stage: StageType, config: ExperimentConfig) -> str:
        return (
            f"{self.artifact_prefix}/{stage}/{config.task}/{config.dataset}/"
            f"{config.method.label}_seed{config.seed}.parquet"
        )

    def list_stage_keys(self, stage: StageType) -> tuple[str, ...]:
        return self.keys.get(stage, ())

    def load(self, stage: StageType, config: ExperimentConfig) -> pd.DataFrame:
        key = self.artifact_key(stage, config)
        self.loaded.append(key)
        if self.load_error is not None:
            raise self.load_error
        return self.frames[key].copy()

    def load_with_payload_sha256(
        self,
        stage: StageType,
        config: ExperimentConfig,
    ) -> LoadedArtifact:
        key = self.artifact_key(stage, config)
        self.loaded.append(key)
        if self.load_error is not None:
            raise self.load_error
        frame = self.frames.get(key, _rankings(config))
        return LoadedArtifact(
            frame=frame.copy(),
            payload_sha256=RANKING_PAYLOAD_SHA256,
        )


def test_reconciliation_accepts_exact_valid_stage_requirements() -> None:
    cells = (
        _cell(0),
        _cell(1, stage2_required=False),
        _cell(2, stage1_required=False),
    )
    manifest = _manifest(*cells)
    store = _Store(
        {
            store_key: frame
            for cell in cells
            for stage, frame in (
                ("rankings", _rankings(cell.config)),
                ("metrics", _metrics(cell.config)),
            )
            if cell.required_for(stage)
            for store_key in (
                (
                    f"{ARTIFACT_PREFIX}/{stage}/{cell.config.task}/"
                    f"{cell.config.dataset}/{cell.config.method.label}_seed"
                    f"{cell.config.seed}.parquet"
                ),
            )
        }
    )

    report = reconcile_manifest_artifacts(store, manifest, _provenance())

    assert report.is_complete
    assert report.counts == {
        "expected": 4,
        "valid": 4,
        "missing": 0,
        "extra": 0,
        "malformed": 0,
        "invalid": 0,
        "provenance_mismatch": 0,
    }
    ranking_dependency = store.artifact_key("rankings", cells[2].config)
    assert tuple(sorted(store.loaded)) == tuple(sorted((*report.expected_keys, ranking_dependency)))


def test_reconciliation_keeps_same_named_tasks_in_separate_cache_entries() -> None:
    cells = (
        _cell(0),
        _cell(
            0,
            task="regression",
            dataset_sha256="f" * 64,
            stage2_required=False,
        ),
    )
    store = _Store({})
    store.frames = {
        store.artifact_key("rankings", cell.config): _rankings(cell.config) for cell in cells
    }
    store.keys = {
        "rankings": tuple(store.frames),
        "metrics": (),
    }

    report = reconcile_manifest_artifacts(
        store,
        _manifest(*cells),
        _provenance(),
        stages=("rankings",),
    )

    assert report.is_complete
    assert report.counts["valid"] == 2
    assert sorted(store.loaded) == sorted(store.frames)


def test_reconciliation_classifies_every_issue_without_loading_unapproved_keys() -> None:
    valid_cell, missing_cell, invalid_cell, provenance_cell = (
        _cell(0),
        _cell(1),
        _cell(2),
        _cell(3),
    )
    manifest = _manifest(valid_cell, missing_cell, invalid_cell, provenance_cell)
    valid_key = _Store({}).artifact_key("rankings", valid_cell.config)
    invalid_key = _Store({}).artifact_key("rankings", invalid_cell.config)
    provenance_key = _Store({}).artifact_key("rankings", provenance_cell.config)
    extra_key = _Store({}).artifact_key("rankings", _config(99))
    malformed_key = f"{ARTIFACT_PREFIX}/rankings/classification/glass/nested/object.parquet"
    frames = {
        valid_key: _rankings(valid_cell.config),
        invalid_key: _rankings(invalid_cell.config).assign(fold_idx=0),
        provenance_key: _rankings(provenance_cell.config).assign(manifest_sha256="c" * 64),
    }
    store = _Store(
        frames,
        keys={
            "rankings": (
                malformed_key,
                extra_key,
                valid_key,
                invalid_key,
                provenance_key,
            ),
            "metrics": (),
        },
    )

    report = reconcile_manifest_artifacts(
        store,
        manifest,
        _provenance(),
        stages=("rankings",),
    )

    assert not report.is_complete
    assert report.counts == {
        "expected": 4,
        "valid": 1,
        "missing": 1,
        "extra": 1,
        "malformed": 1,
        "invalid": 1,
        "provenance_mismatch": 1,
    }
    assert report.extra_keys == (extra_key,)
    assert report.malformed_keys == (malformed_key,)
    assert {issue.key for issue in report.invalid_artifacts} == {invalid_key}
    assert {issue.key for issue in report.provenance_mismatches} == {provenance_key}
    assert set(store.loaded) == {valid_key, invalid_key, provenance_key}


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("artifact_prefix", "other/run", "artifact prefix"),
        ("aws_account_id", "210987654321", "account binding"),
        ("manifest_sha256", "c" * 64, "manifest digest"),
    ],
)
def test_reconciliation_rejects_scope_mismatch_before_listing(
    field: str,
    value: str,
    match: str,
) -> None:
    provenance = _provenance()
    provenance[field] = value
    store = _Store({})

    with pytest.raises(ValueError, match=match):
        reconcile_manifest_artifacts(
            store,
            _manifest(_cell(0)),
            provenance,
            stages=("rankings",),
        )

    assert store.loaded == []


def test_reconciliation_propagates_unclassified_storage_failure() -> None:
    cell = _cell(0)
    key = _Store({}).artifact_key("rankings", cell.config)
    store = _Store(
        {key: _rankings(cell.config)},
        load_error=RuntimeError("S3 transport failed"),
    )

    with pytest.raises(RuntimeError, match="S3 transport failed"):
        reconcile_manifest_artifacts(
            store,
            _manifest(cell),
            _provenance(),
            stages=("rankings",),
        )


@pytest.mark.parametrize(
    ("report", "expected_exit_code"),
    [
        (
            ReconciliationReport(
                expected_keys=("expected",),
                valid_keys=("expected",),
                missing_keys=(),
                extra_keys=(),
                malformed_keys=(),
                invalid_artifacts=(),
                provenance_mismatches=(),
            ),
            0,
        ),
        (
            ReconciliationReport(
                expected_keys=("expected",),
                valid_keys=(),
                missing_keys=("expected",),
                extra_keys=("extra",),
                malformed_keys=("malformed",),
                invalid_artifacts=(ArtifactIssue("invalid", "bad schema"),),
                provenance_mismatches=(ArtifactIssue("provenance", "wrong digest"),),
            ),
            1,
        ),
    ],
)
def test_reconciliation_cli_is_a_failing_gate(
    report: ReconciliationReport,
    expected_exit_code: int,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from paper.benchmark.adapters.store import S3Store
    from paper.benchmark.pipeline import reconcile as reconcile_module
    from paper.benchmark.utils import env

    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("fixture")
    manifest = _manifest(_cell(0))
    store = _Store({})
    store.bucket = "citrees-test"

    monkeypatch.setattr(
        manifest_cli,
        "parse_rerun_manifest",
        lambda payload: manifest,
    )
    monkeypatch.setattr(
        S3Store,
        "from_env",
        classmethod(lambda cls, validate_uploads=False: store),
    )
    monkeypatch.setattr(
        reconcile_module,
        "reconcile_manifest_artifacts",
        lambda *args, **kwargs: report,
    )
    monkeypatch.setattr(
        env,
        "get_benchmark_scope",
        lambda: {
            "artifact_prefix": ARTIFACT_PREFIX,
            "aws_account_id": ACCOUNT_ID,
            "manifest_sha256": MANIFEST_SHA256,
        },
    )
    monkeypatch.setattr(env, "get_container_image", lambda: CONTAINER_IMAGE)
    monkeypatch.setattr(env, "get_git_sha", lambda: GIT_SHA)

    manifest_result = CliRunner().invoke(
        manifest_cli.app,
        ["reconcile", "--manifest", str(manifest_path)],
    )
    check_result = CliRunner().invoke(
        root_cli,
        ["check", "--manifest", str(manifest_path)],
    )

    assert manifest_result.exit_code == expected_exit_code
    assert check_result.exit_code == expected_exit_code
