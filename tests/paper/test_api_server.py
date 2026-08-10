"""Tests for API queue completion logic."""

from __future__ import annotations

import asyncio
import io
import json
from dataclasses import replace
from itertools import product
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
from botocore.exceptions import ClientError
from fastapi import HTTPException

from paper.benchmark.adapters.data import get_dataset_payload_identity, get_dataset_s3_key
from paper.benchmark.adapters.store import ControlReceipt, LoadedArtifact
from paper.benchmark.api import server
from paper.benchmark.api.server import (
    QueueState,
    _configured_lease_seconds,
    _configured_max_cell_attempts,
    _configured_stage,
    _load_durable_attempt_state,
    _needs_metrics_work,
    _preflight_manifest_datasets,
    _validated_ranking_feature_count,
)
from paper.benchmark.api.worker import _deserialize_config, _validate_assignment
from paper.benchmark.config.constants import (
    CLF_DOWNSTREAM_MODELS,
    MIN_ASSIGNMENT_LEASE_SECONDS,
    PIPELINE_ARTIFACT_VERSION,
)
from paper.benchmark.experiments.r_cforest_reproducibility import (
    gate_receipt_s3_key,
)
from paper.benchmark.pipeline.campaign_gate import (
    ApprovedCampaignGate,
    CampaignGateIdentity,
)
from paper.benchmark.pipeline.manifest import ManifestCell, RerunManifest
from paper.benchmark.pipeline.reconcile import ArtifactIssue, ReconciliationReport
from paper.benchmark.pipeline.types import DatasetIdentity, ExperimentConfig, MethodConfig
from paper.benchmark.pipeline.validation import derive_assignment_id

pytestmark = pytest.mark.paper
REQUEST_ID = "1" * 32
WORKER_ID = "2" * 32
CANONICAL_MANIFEST_SHA256 = "c" * 64
CANONICAL_MANIFEST_S3_KEY = f"canonical-rerun-manifests/{CANONICAL_MANIFEST_SHA256}.csv"
GATE_RECEIPT_SHA256 = "d" * 64
GATE_RECEIPT_S3_KEY = gate_receipt_s3_key(GATE_RECEIPT_SHA256)
RUNTIME_CONTRACT_SHA256 = "f" * 64
RUNTIME_CONTRACT_S3_KEY = f"runtime-contracts/{RUNTIME_CONTRACT_SHA256}.json"


def _config(
    method: str = "cif",
    *,
    dataset: str = "arcene",
    seed: int = 0,
    n_samples: int = 100,
    n_features: int = 4,
) -> ExperimentConfig:
    return ExperimentConfig(
        method=MethodConfig(method),
        dataset=dataset,
        seed=seed,
        task="classification",
        dataset_identity=DatasetIdentity(
            "d" * 64,
            n_samples=n_samples,
            n_features=n_features,
        ),
    )


def _expected_provenance() -> dict[str, str]:
    return {
        "artifact_prefix": "repairs/r-baselines/run-001",
        "aws_account_id": "123456789012",
        "campaign_sha256": "e" * 64,
        "canonical_manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "container_image": "repository@sha256:" + "a" * 64,
        "gate_receipt_sha256": GATE_RECEIPT_SHA256,
        "git_sha": "a" * 40,
        "manifest_sha256": "a" * 64,
        "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
    }


def _configure_gate(
    monkeypatch: pytest.MonkeyPatch,
    manifest: RerunManifest,
) -> ApprovedCampaignGate:
    identity = CampaignGateIdentity(
        account_id="123456789012",
        campaign_sha256=manifest.campaign_sha256,
        canonical_manifest_s3_key=CANONICAL_MANIFEST_S3_KEY,
        canonical_manifest_sha256=CANONICAL_MANIFEST_SHA256,
        gate_receipt_s3_key=GATE_RECEIPT_S3_KEY,
        gate_receipt_sha256=GATE_RECEIPT_SHA256,
        manifest_s3_key=f"rerun-manifests/{manifest.sha256}.csv",
        manifest_sha256=manifest.sha256,
        runtime_contract_s3_key=RUNTIME_CONTRACT_S3_KEY,
        runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
    )
    approved = ApprovedCampaignGate(
        identity=identity,
        canonical_manifest=manifest,
        manifest=manifest,
        runtime_contract={},
        gate_receipt={},
    )
    monkeypatch.setattr(
        "paper.benchmark.pipeline.campaign_gate.configured_campaign_gate_identity",
        lambda: identity,
    )
    monkeypatch.setattr(
        "paper.benchmark.pipeline.campaign_gate.load_approved_campaign_gate",
        lambda store, configured_identity: approved,
    )
    return approved


def _assignment_id(
    config: ExperimentConfig,
    attempt: int,
    *,
    stage: str = "rankings",
) -> str:
    return derive_assignment_id(
        config,
        stage=stage,
        attempt=attempt,
        expected_provenance=_expected_provenance(),
    )


def _attempt_receipt(
    config: ExperimentConfig,
    assignment_id: str,
    attempt: int,
) -> dict[str, object]:
    return {
        **_expected_provenance(),
        "artifact_version": PIPELINE_ARTIFACT_VERSION,
        "assignment_id": assignment_id,
        "attempt": attempt,
        "created_at_utc": "2026-08-03T12:00:00+00:00",
        "dataset": config.dataset,
        "dataset_sha256": config.dataset_identity.sha256,
        "lease_seconds": 30,
        "method_base": config.method.name,
        "method_id": config.method.label,
        "method_params_json": "{}",
        "n_features": config.dataset_identity.n_features,
        "n_samples": config.dataset_identity.n_samples,
        "seed": config.seed,
        "stage": "rankings",
        "status": "assigned",
        "task": config.task,
        "request_id": REQUEST_ID,
        "worker_id": WORKER_ID,
    }


def _failure_receipt(
    config: ExperimentConfig,
    assignment_id: str,
    attempt: int,
) -> dict[str, object]:
    receipt = _attempt_receipt(config, assignment_id, attempt)
    receipt.pop("lease_seconds")
    receipt.update(
        {
            "elapsed_seconds": 1.0,
            "error": "fit failed",
            "error_type": "RuntimeError",
            "hardware": {"logical_cpus": 8},
            "hostname": "worker-1",
            "library_versions": {"python": "3.12.7"},
            "status": "failed",
            "traceback": "RuntimeError: fit failed",
        }
    )
    return receipt


def _activate_lease(
    queue: QueueState,
    config: ExperimentConfig,
    assignment_id: str,
    attempt: int,
    lease_seconds: int,
    *,
    now: float | None = None,
    request_id: str = REQUEST_ID,
    worker_id: str = WORKER_ID,
) -> None:
    queue.offer_next(
        assignment_id,
        attempt,
        worker_id,
        request_id,
        60,
        now=now,
    )
    assert (
        queue.begin_start(
            assignment_id,
            worker_id=worker_id,
            request_id=request_id,
            now=now,
        )
        is not None
    )
    queue.activate_offer(assignment_id, lease_seconds, now=now)


class _FakeStore:
    def __init__(self, rankings_df: pd.DataFrame, metrics_df: pd.DataFrame):
        self._rankings_df = rankings_df
        self._metrics_df = metrics_df

    def load(self, stage: str, config: ExperimentConfig) -> pd.DataFrame:
        del config
        if stage == "rankings":
            return self._rankings_df
        if stage == "metrics":
            return self._metrics_df
        raise AssertionError(f"unexpected stage: {stage}")

    def load_with_payload_sha256(
        self,
        stage: str,
        config: ExperimentConfig,
    ) -> LoadedArtifact:
        return LoadedArtifact(
            frame=self.load(stage, config),
            payload_sha256="c" * 64,
        )


def _artifact_common(
    config: ExperimentConfig,
    *,
    n_features: int,
) -> dict[str, object]:
    return {
        "artifact_version": PIPELINE_ARTIFACT_VERSION,
        "artifact_prefix": "repairs/run-001",
        "aws_account_id": "123456789012",
        "campaign_sha256": "e" * 64,
        "canonical_manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "container_image": "repository@sha256:" + "a" * 64,
        "created_at_utc": "2026-08-03T12:00:00+00:00",
        "dataset": config.dataset,
        "dataset_sha256": config.dataset_identity.sha256,
        "gate_receipt_sha256": GATE_RECEIPT_SHA256,
        "git_sha": "a" * 40,
        "hardware": {"logical_cpus": 32, "cpu_affinity": list(range(32))},
        "library_versions": {"python": "3.12.7"},
        "method": config.method.label,
        "method_base": config.method.name,
        "method_id": config.method.label,
        "method_params_json": "{}",
        "manifest_sha256": "b" * 64,
        "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
        "n_features": n_features,
        "n_samples": 100,
        "seed": config.seed,
        "task": config.task,
    }


def _valid_rankings(
    config: ExperimentConfig,
    *,
    n_features: int,
) -> pd.DataFrame:
    common = _artifact_common(config, n_features=n_features)
    return pd.DataFrame(
        [
            {
                **common,
                "feature_ranking": list(range(n_features)),
                "fold_cpu_affinity": list(range(32)),
                "fold_idx": fold,
                "fold_random_state": config.seed * 1000 + fold,
            }
            for fold in range(5)
        ]
    )


def _metrics(
    config: ExperimentConfig,
    *,
    n_features: int,
    k_values: list[int],
) -> pd.DataFrame:
    common = _artifact_common(config, n_features=n_features)
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
                "k": k,
                "n_features_selected": k,
                "ranking_artifact_version": PIPELINE_ARTIFACT_VERSION,
                "ranking_artifact_prefix": "repairs/run-001",
                "ranking_aws_account_id": "123456789012",
                "ranking_canonical_manifest_sha256": CANONICAL_MANIFEST_SHA256,
                "ranking_container_image": "repository@sha256:" + "a" * 64,
                "ranking_dataset_sha256": config.dataset_identity.sha256,
                "ranking_gate_receipt_sha256": GATE_RECEIPT_SHA256,
                "ranking_git_sha": "a" * 40,
                "ranking_manifest_sha256": "b" * 64,
                "ranking_payload_sha256": "c" * 64,
                "ranking_runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
                "roc_auc": 0.85,
            }
            for fold, k, model in product(range(5), k_values, CLF_DOWNSTREAM_MODELS)
        ]
    )


@pytest.mark.parametrize(
    "raw",
    ["rankings", "metrics"],
)
def test_configured_stage_is_validated(
    raw: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CITREES_STAGE", raw)
    assert _configured_stage() == raw


@pytest.mark.parametrize("raw", ["", " ", "rankings,metrics", "bogus"])
def test_configured_stage_rejects_empty_or_unknown_values(
    raw: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CITREES_STAGE", raw)
    with pytest.raises(RuntimeError):
        _configured_stage()


@pytest.mark.parametrize(
    "raw",
    [str(MIN_ASSIGNMENT_LEASE_SECONDS), "900", "86400"],
)
def test_configured_lease_seconds_is_heartbeat_safe(
    raw: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CITREES_LEASE_SECONDS", raw)
    assert _configured_lease_seconds() == int(raw)


@pytest.mark.parametrize(
    "raw",
    ["", "0", "-1", "01", "1.5", "invalid", str(MIN_ASSIGNMENT_LEASE_SECONDS - 1)],
)
def test_configured_lease_seconds_rejects_invalid_values(
    raw: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CITREES_LEASE_SECONDS", raw)
    with pytest.raises(RuntimeError, match=f">= {MIN_ASSIGNMENT_LEASE_SECONDS}"):
        _configured_lease_seconds()


@pytest.mark.parametrize("raw", ["1", "3", "10"])
def test_configured_max_cell_attempts_is_positive(
    raw: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CITREES_MAX_CELL_ATTEMPTS", raw)
    assert _configured_max_cell_attempts() == int(raw)


@pytest.mark.parametrize("raw", ["", "0", "-1", "01", "1.5", "invalid"])
def test_configured_max_cell_attempts_rejects_invalid_values(
    raw: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CITREES_MAX_CELL_ATTEMPTS", raw)
    with pytest.raises(RuntimeError, match="positive integer"):
        _configured_max_cell_attempts()


def _dataset_payload(*, offset: int = 0) -> bytes:
    frame = pd.DataFrame(
        {
            **{f"x{index}": [value + offset for value in range(10)] for index in range(4)},
            "y": [0, 1] * 5,
        }
    )
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def test_manifest_dataset_preflight_validates_each_unique_object_once() -> None:
    payload = _dataset_payload()
    identity = get_dataset_payload_identity(payload)
    first = replace(_config(), dataset_identity=identity)
    second = replace(first, seed=1)
    cells = tuple(
        ManifestCell(
            config=config,
            target_aws_account_id="123456789012",
            dataset_source="real",
            rerun_reason="adapter_correction",
            historically_omitted=False,
            stage1_required=True,
            stage2_required=True,
        )
        for config in (first, second)
    )
    client = MagicMock()
    client.get_object.return_value = {"Body": io.BytesIO(payload)}
    store = SimpleNamespace(bucket="citrees-test", client=client)

    assert (
        _preflight_manifest_datasets(
            store,  # type: ignore[arg-type]
            RerunManifest(
                sha256="a" * 64,
                campaign_sha256="e" * 64,
                runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
                cells=cells,
            ),
        )
        == 1
    )
    client.get_object.assert_called_once_with(
        Bucket="citrees-test",
        Key=get_dataset_s3_key(
            first.dataset,
            first.task,
            "real",
            identity,
        ),
    )


@pytest.mark.parametrize(
    "response",
    [
        ClientError(
            {
                "Error": {"Code": "NoSuchKey", "Message": "missing"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "GetObject",
        ),
        {"Body": io.BytesIO(_dataset_payload(offset=1))},
    ],
)
def test_manifest_dataset_preflight_rejects_missing_or_mismatched_objects(
    response: object,
) -> None:
    identity = get_dataset_payload_identity(_dataset_payload())
    config = replace(_config(), dataset_identity=identity)
    cell = ManifestCell(
        config=config,
        target_aws_account_id="123456789012",
        dataset_source="real",
        rerun_reason="adapter_correction",
        historically_omitted=False,
        stage1_required=True,
        stage2_required=True,
    )
    client = MagicMock()
    if isinstance(response, Exception):
        client.get_object.side_effect = response
    else:
        client.get_object.return_value = response
    store = SimpleNamespace(bucket="citrees-test", client=client)

    with pytest.raises((FileNotFoundError, ValueError)):
        _preflight_manifest_datasets(
            store,  # type: ignore[arg-type]
            RerunManifest(
                sha256="a" * 64,
                campaign_sha256="e" * 64,
                runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
                cells=(cell,),
            ),
        )


def test_existing_ranking_is_validated_before_queue_exclusion() -> None:
    cfg = _config()
    valid = _FakeStore(
        rankings_df=_valid_rankings(cfg, n_features=4),
        metrics_df=pd.DataFrame(),
    )

    assert _validated_ranking_feature_count(valid, cfg, set()) is None
    assert _validated_ranking_feature_count(valid, cfg, {cfg.key}) == 4

    invalid = _FakeStore(
        rankings_df=pd.DataFrame({"feature_ranking": [[0, 1, 2, 3]]}),
        metrics_df=pd.DataFrame(),
    )
    with pytest.raises(ValueError, match="must contain 5 folds"):
        _validated_ranking_feature_count(invalid, cfg, {cfg.key})


def test_queue_build_uses_only_manifest_cells_for_active_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("r_ctree", dataset="fixture")

    class FakeS3Store:
        bucket = "citrees-fixture"
        artifact_prefix = "repairs/r-baselines/run-001"

        def list_completed(self, stage: str, task: str):
            return set()

        def list_control_receipts(self, receipt_type: str, stage: str, task: str):
            return ()

    cell = ManifestCell(
        config=config,
        target_aws_account_id="123456789012",
        dataset_source="real",
        rerun_reason="adapter_correction",
        historically_omitted=False,
        stage1_required=True,
        stage2_required=True,
    )
    manifest = RerunManifest(
        sha256="a" * 64,
        campaign_sha256="e" * 64,
        runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
        cells=(cell,),
    )

    monkeypatch.setenv("CITREES_STAGE", "rankings")
    monkeypatch.setenv("CITREES_LEASE_SECONDS", "900")
    monkeypatch.setenv("CITREES_MAX_CELL_ATTEMPTS", "3")
    monkeypatch.setenv("CITREES_ARTIFACT_PREFIX", "repairs/r-baselines/run-001")
    monkeypatch.setenv("CITREES_CAMPAIGN_SHA256", "e" * 64)
    monkeypatch.setenv("CITREES_CANONICAL_MANIFEST_SHA256", CANONICAL_MANIFEST_SHA256)
    monkeypatch.setenv("CITREES_GATE_RECEIPT_SHA256", GATE_RECEIPT_SHA256)
    monkeypatch.setenv("CITREES_MANIFEST_SHA256", "a" * 64)
    monkeypatch.setenv("CITREES_RUNTIME_CONTRACT_SHA256", RUNTIME_CONTRACT_SHA256)
    monkeypatch.setattr(
        "paper.benchmark.infra.aws.get_aws_account_id",
        lambda: "123456789012",
    )
    monkeypatch.setenv("CITREES_IMAGE_URI", "repository@sha256:" + "a" * 64)
    monkeypatch.setenv("GIT_SHA", "a" * 40)
    monkeypatch.setattr(
        "paper.benchmark.adapters.store.S3Store.from_env",
        lambda validate_uploads: FakeS3Store(),
    )
    _configure_gate(monkeypatch, manifest)
    monkeypatch.setattr(server, "_preflight_manifest_datasets", lambda store, value: 1)

    server._build_queues()

    assert set(server._queues) == {"rankings/classification", "rankings/regression"}
    assert server._queues["rankings/classification"].initial == 1
    assert server._queues["rankings/regression"].initial == 0
    assert server._active_stage == "rankings"
    assert server._active_methods == ("r_ctree",)
    assert server._manifest is manifest


def _ranking_reconciliation_report(
    config: ExperimentConfig,
    *,
    issue: str | None = None,
) -> ReconciliationReport:
    key = (
        "repairs/r-baselines/run-001/rankings/classification/"
        f"{config.dataset}/{config.method.label}_seed{config.seed}.parquet"
    )
    expected = (key,)
    valid = expected
    missing: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()
    malformed: tuple[str, ...] = ()
    invalid: tuple[ArtifactIssue, ...] = ()
    provenance_mismatches: tuple[ArtifactIssue, ...] = ()

    if issue == "missing":
        valid = ()
        missing = expected
    elif issue == "extra":
        extra = (f"{key}.extra",)
    elif issue == "malformed":
        malformed = ("repairs/r-baselines/run-001/rankings/malformed",)
    elif issue == "invalid":
        valid = ()
        invalid = (ArtifactIssue(key=key, detail="invalid ranking artifact"),)
    elif issue == "provenance_mismatch":
        valid = ()
        provenance_mismatches = (ArtifactIssue(key=key, detail="ranking provenance mismatch"),)
    elif issue is not None:
        raise AssertionError(f"unknown reconciliation issue: {issue}")

    return ReconciliationReport(
        expected_keys=expected,
        valid_keys=valid,
        missing_keys=missing,
        extra_keys=extra,
        malformed_keys=malformed,
        invalid_artifacts=invalid,
        provenance_mismatches=provenance_mismatches,
    )


def _configure_metrics_queue_build(
    monkeypatch: pytest.MonkeyPatch,
    config: ExperimentConfig,
    report: ReconciliationReport,
) -> tuple[object, RerunManifest, MagicMock]:
    class FakeS3Store:
        bucket = "citrees-fixture"
        artifact_prefix = "repairs/r-baselines/run-001"

        def __init__(self) -> None:
            self.list_completed_calls: list[tuple[str, str]] = []

        def list_completed(self, stage: str, task: str):
            self.list_completed_calls.append((stage, task))
            if stage == "rankings" and task == "classification":
                return {config.key}
            return set()

        def list_control_receipts(self, receipt_type: str, stage: str, task: str):
            return ()

    cell = ManifestCell(
        config=config,
        target_aws_account_id="123456789012",
        dataset_source="real",
        rerun_reason="adapter_correction",
        historically_omitted=False,
        stage1_required=True,
        stage2_required=True,
    )
    manifest = RerunManifest(
        sha256="a" * 64,
        campaign_sha256="e" * 64,
        runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
        cells=(cell,),
    )
    store = FakeS3Store()
    reconcile = MagicMock(return_value=report)

    monkeypatch.setenv("CITREES_STAGE", "metrics")
    monkeypatch.setenv("CITREES_LEASE_SECONDS", "900")
    monkeypatch.setenv("CITREES_MAX_CELL_ATTEMPTS", "3")
    monkeypatch.setenv("CITREES_ARTIFACT_PREFIX", store.artifact_prefix)
    monkeypatch.setenv("CITREES_CAMPAIGN_SHA256", manifest.campaign_sha256)
    monkeypatch.setenv("CITREES_CANONICAL_MANIFEST_SHA256", CANONICAL_MANIFEST_SHA256)
    monkeypatch.setenv("CITREES_GATE_RECEIPT_SHA256", GATE_RECEIPT_SHA256)
    monkeypatch.setenv("CITREES_MANIFEST_SHA256", manifest.sha256)
    monkeypatch.setenv("CITREES_RUNTIME_CONTRACT_SHA256", RUNTIME_CONTRACT_SHA256)
    monkeypatch.setattr(
        "paper.benchmark.infra.aws.get_aws_account_id",
        lambda: "123456789012",
    )
    monkeypatch.setenv("CITREES_IMAGE_URI", "repository@sha256:" + "a" * 64)
    monkeypatch.setenv("GIT_SHA", "a" * 40)
    monkeypatch.setattr(
        "paper.benchmark.adapters.store.S3Store.from_env",
        lambda validate_uploads: store,
    )
    monkeypatch.setattr(
        "paper.benchmark.pipeline.reconcile.reconcile_manifest_artifacts",
        reconcile,
    )
    _configure_gate(monkeypatch, manifest)
    monkeypatch.setattr(server, "_preflight_manifest_datasets", lambda active_store, value: 1)
    monkeypatch.setattr(
        server,
        "_validated_ranking_feature_count",
        lambda active_store, cfg, completed, *, expected_provenance: (
            cfg.dataset_identity.n_features if cfg.key in completed else None
        ),
    )
    return store, manifest, reconcile


def test_metrics_queue_build_requires_complete_exact_ranking_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("r_ctree", dataset="fixture")
    report = _ranking_reconciliation_report(config)
    store, manifest, reconcile = _configure_metrics_queue_build(
        monkeypatch,
        config,
        report,
    )

    server._build_queues()

    reconcile.assert_called_once_with(
        store,
        manifest,
        _expected_provenance(),
        stages=("rankings",),
    )
    assert set(server._queues) == {"metrics/classification", "metrics/regression"}
    assert server._queues["metrics/classification"].initial == 1
    assert server._queues["metrics/regression"].initial == 0


@pytest.mark.parametrize(
    "issue",
    [
        "missing",
        "extra",
        "malformed",
        "invalid",
        "provenance_mismatch",
    ],
)
def test_metrics_queue_build_rejects_any_ranking_reconciliation_issue(
    issue: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("r_ctree", dataset="fixture")
    report = _ranking_reconciliation_report(config, issue=issue)
    store, _manifest, reconcile = _configure_metrics_queue_build(
        monkeypatch,
        config,
        report,
    )
    queue_factory = MagicMock(side_effect=AssertionError("metrics queue construction began"))
    monkeypatch.setattr(QueueState, "from_configs", queue_factory)

    with pytest.raises(RuntimeError, match=rf"{issue}=1"):
        server._build_queues()

    reconcile.assert_called_once()
    assert store.list_completed_calls == []  # type: ignore[attr-defined]
    queue_factory.assert_not_called()
    assert server._queues == {}


def test_next_assignment_includes_artifact_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("r_ctree", dataset="fixture")
    server._queues.clear()
    server._queues["rankings/classification"] = QueueState.from_configs(
        [config],
        max_attempts=3,
    )
    server._active_stage = "rankings"
    server._lease_seconds = 30
    save_attempt = MagicMock()
    server._store = SimpleNamespace(  # type: ignore[assignment]
        artifact_prefix="repairs/r-baselines/run-001",
        save_attempt=save_attempt,
    )
    server._manifest = RerunManifest(
        sha256="a" * 64,
        campaign_sha256="e" * 64,
        runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
        cells=(),
    )
    server._expected_provenance = _expected_provenance()
    monkeypatch.setattr(server, "_validate_assignment_artifact", lambda config: None)

    request = server.AssignmentRequest(request_id=REQUEST_ID, worker_id=WORKER_ID)
    response = asyncio.run(server.next_config(request))
    body = json.loads(response.body)

    assert body["artifact_prefix"] == "repairs/r-baselines/run-001"
    assert body["aws_account_id"] == "123456789012"
    assert body["campaign_sha256"] == "e" * 64
    assert body["canonical_manifest_sha256"] == CANONICAL_MANIFEST_SHA256
    assert body["container_image"] == "repository@sha256:" + "a" * 64
    assert body["gate_receipt_sha256"] == GATE_RECEIPT_SHA256
    assert body["git_sha"] == "a" * 40
    assert body["manifest_sha256"] == "a" * 64
    assert body["runtime_contract_sha256"] == RUNTIME_CONTRACT_SHA256
    assert len(body["assignment_id"]) == 32
    assert body["attempt"] == 1
    assert body["lease_seconds"] == 30
    assert body["request_id"] == REQUEST_ID
    assert body["worker_id"] == WORKER_ID
    assert body["stage"] == "rankings"
    assert body["method"]["name"] == "r_ctree"
    assert body["dataset_identity"] == {
        "sha256": "d" * 64,
        "n_samples": 100,
        "n_features": 4,
    }
    assert _deserialize_config(body) == config
    assert server._queues["rankings/classification"].attempts == 0
    update = server.AssignmentUpdate(
        assignment_id=body["assignment_id"],
        request_id=REQUEST_ID,
        worker_id=WORKER_ID,
    )
    assert asyncio.run(server.start_assignment(update)) == {"started": True}
    save_attempt.assert_called_once()
    heartbeat = asyncio.run(server.heartbeat_assignment(update))
    completed = asyncio.run(server.complete_assignment(update))
    assert heartbeat == {"extended": True}
    assert completed == {"completed": True}
    assert server._queues["rankings/classification"].completed == 1


def test_queue_leases_heartbeat_and_acknowledges_completion() -> None:
    config = _config("r_ctree", dataset="fixture")
    queue = QueueState.from_configs([config], max_attempts=3)
    assignment_id = _assignment_id(config, 1)

    _activate_lease(queue, config, assignment_id, 1, 10, now=100.0)
    assert queue.available == 0
    assert queue.in_flight == 1
    assert queue.pending == 1

    assert queue.heartbeat(
        assignment_id,
        WORKER_ID,
        REQUEST_ID,
        10,
        now=105.0,
    )
    assert queue.requeue_expired(now=114.9) == 0
    assert (
        queue.begin_finalization(
            assignment_id,
            worker_id=WORKER_ID,
            request_id=REQUEST_ID,
            now=114.9,
        )
        is not None
    )
    assert queue.finish_finalization(assignment_id, failed=False)
    assert queue.completed == 1
    assert queue.failed == 0
    assert queue.pending == 0
    assert not queue.finish_finalization(assignment_id, failed=False)


def test_queue_offer_does_not_consume_attempt_before_start() -> None:
    config = _config("r_ctree", dataset="fixture")
    queue = QueueState.from_configs([config], max_attempts=3)
    assignment_id = _assignment_id(config, 1)

    serialized = queue.offer_next(
        assignment_id,
        1,
        WORKER_ID,
        REQUEST_ID,
        60,
        now=100.0,
    )
    assert serialized["dataset"] == config.dataset
    assert queue.available == 0
    assert queue.reserved == 1
    assert queue.in_flight == 0
    assert queue.pending == 1
    assert queue.attempts == 0

    assert queue.cancel_offer(assignment_id)
    assert queue.available == 1
    assert queue.reserved == 0
    assert queue.next_candidate() == (config, 1)


def test_assignment_id_is_deterministic_per_campaign_cell_and_attempt() -> None:
    config = _config("r_ctree", dataset="fixture")
    server._expected_provenance = _expected_provenance()

    first = server._deterministic_assignment_id(config, stage="rankings", attempt=1)
    assert first == server._deterministic_assignment_id(
        config,
        stage="rankings",
        attempt=1,
    )
    assert len(first) == 32
    assert first != server._deterministic_assignment_id(
        config,
        stage="rankings",
        attempt=2,
    )
    assert first != server._deterministic_assignment_id(
        config,
        stage="metrics",
        attempt=1,
    )


def test_lost_next_response_replays_one_uncharged_offer() -> None:
    config = _config("r_ctree", dataset="fixture")
    queue = QueueState.from_configs([config], max_attempts=1)
    server._queues = {"rankings/classification": queue}
    server._active_stage = "rankings"
    server._expected_provenance = _expected_provenance()

    first = asyncio.run(server._pick_next(WORKER_ID, REQUEST_ID))
    replay = asyncio.run(server._pick_next(WORKER_ID, REQUEST_ID))

    assert replay == first
    assert queue.attempts == 0
    assert queue.reserved == 1
    assert queue.in_flight == 0


def test_lost_next_response_replays_before_cross_task_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classification_config = _config("r_ctree", dataset="classification-fixture")
    regression_config = ExperimentConfig(
        method=MethodConfig("r_ctree"),
        dataset="regression-fixture",
        seed=0,
        task="regression",
        dataset_identity=DatasetIdentity(
            "d" * 64,
            n_samples=100,
            n_features=4,
        ),
    )
    classification = QueueState.from_configs([classification_config], max_attempts=1)
    regression = QueueState.from_configs([regression_config], max_attempts=1)
    server._queues = {
        "rankings/classification": classification,
        "rankings/regression": regression,
    }
    server._active_stage = "rankings"
    server._expected_provenance = _expected_provenance()
    task_orders = iter(
        [
            ["classification", "regression"],
            ["regression", "classification"],
        ]
    )
    monkeypatch.setattr(
        server.random,
        "shuffle",
        lambda tasks: tasks.__setitem__(slice(None), next(task_orders)),
    )

    first = asyncio.run(server._pick_next(WORKER_ID, REQUEST_ID))
    replay = asyncio.run(server._pick_next(WORKER_ID, REQUEST_ID))

    assert replay == first
    assert classification.reserved == 1
    assert regression.reserved == 0


def test_cross_task_duplicate_request_ownership_is_rejected() -> None:
    classification_config = _config("r_ctree", dataset="classification-fixture")
    regression_config = ExperimentConfig(
        method=MethodConfig("r_ctree"),
        dataset="regression-fixture",
        seed=0,
        task="regression",
        dataset_identity=DatasetIdentity(
            "d" * 64,
            n_samples=100,
            n_features=4,
        ),
    )
    classification = QueueState.from_configs([classification_config], max_attempts=1)
    regression = QueueState.from_configs([regression_config], max_attempts=1)
    server._queues = {
        "rankings/classification": classification,
        "rankings/regression": regression,
    }
    server._active_stage = "rankings"
    server._expected_provenance = _expected_provenance()
    classification_candidate = classification.next_candidate()
    regression_candidate = regression.next_candidate()
    assert classification_candidate is not None
    assert regression_candidate is not None
    for queue, (config, attempt) in (
        (classification, classification_candidate),
        (regression, regression_candidate),
    ):
        queue.offer_next(
            server._deterministic_assignment_id(
                config,
                stage="rankings",
                attempt=attempt,
            ),
            attempt,
            WORKER_ID,
            REQUEST_ID,
            offer_seconds=30,
        )

    with pytest.raises(RuntimeError, match="multiple task queues"):
        asyncio.run(server._pick_next(WORKER_ID, REQUEST_ID))


def test_ambiguous_attempt_write_loads_and_validates_existing_receipt() -> None:
    config = _config("r_ctree", dataset="fixture")
    server._queues = {
        "rankings/classification": QueueState.from_configs(
            [config],
            max_attempts=3,
        )
    }
    server._active_stage = "rankings"
    server._lease_seconds = 900
    server._expected_provenance = _expected_provenance()
    server._manifest = RerunManifest(
        sha256="a" * 64,
        campaign_sha256="e" * 64,
        runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
        cells=(),
    )
    assignment_id = server._deterministic_assignment_id(
        config,
        stage="rankings",
        attempt=1,
    )
    captured: dict[str, object] = {}

    def commit_then_disconnect(
        stage: str,
        cfg: ExperimentConfig,
        observed_assignment_id: str,
        receipt: dict[str, object],
        *,
        expected_provenance: dict[str, str],
    ) -> None:
        del stage, cfg, observed_assignment_id, expected_provenance
        captured.update(receipt)
        raise RuntimeError("response lost")

    load_attempt = MagicMock(side_effect=lambda *args: dict(captured))
    server._store = SimpleNamespace(  # type: ignore[assignment]
        artifact_prefix="repairs/r-baselines/run-001",
        save_attempt=commit_then_disconnect,
        load_attempt=load_attempt,
    )

    request = server.AssignmentRequest(request_id=REQUEST_ID, worker_id=WORKER_ID)
    response = asyncio.run(server.next_config(request))
    body = json.loads(response.body)
    update = server.AssignmentUpdate(
        assignment_id=body["assignment_id"],
        request_id=REQUEST_ID,
        worker_id=WORKER_ID,
    )
    assert asyncio.run(server.start_assignment(update)) == {"started": True}

    assert body["assignment_id"] == assignment_id
    assert body["attempt"] == 1
    assert server._queues["rankings/classification"].attempts == 1
    assert server._queues["rankings/classification"].in_flight == 1
    load_attempt.assert_called_once_with("rankings", config, assignment_id)


def test_attempt_receipt_from_another_owner_is_rejected() -> None:
    config = _config("r_ctree", dataset="fixture")
    server._active_stage = "rankings"
    server._lease_seconds = 900
    server._expected_provenance = _expected_provenance()
    assignment_id = _assignment_id(config, 1)
    intended = server._attempt_receipt(
        config,
        stage="rankings",
        assignment_id=assignment_id,
        attempt=1,
        lease_seconds=900,
        request_id=REQUEST_ID,
        worker_id=WORKER_ID,
    )
    other = {
        **intended,
        "request_id": "3" * 32,
        "worker_id": "4" * 32,
    }
    server._store = SimpleNamespace(  # type: ignore[assignment]
        save_attempt=MagicMock(side_effect=RuntimeError("precondition failed")),
        load_attempt=MagicMock(return_value=other),
    )

    with pytest.raises(server.AttemptOwnershipError):
        asyncio.run(
            server._persist_attempt_receipt(
                config,
                stage="rankings",
                assignment_id=assignment_id,
                attempt=1,
                receipt=intended,
            )
        )


def test_failed_attempt_write_releases_reservation_without_consuming_attempt() -> None:
    config = _config("r_ctree", dataset="fixture")
    queue = QueueState.from_configs([config], max_attempts=3)
    server._queues = {"rankings/classification": queue}
    server._active_stage = "rankings"
    server._lease_seconds = 900
    server._expected_provenance = _expected_provenance()
    server._manifest = RerunManifest(
        sha256="a" * 64,
        campaign_sha256="e" * 64,
        runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
        cells=(),
    )
    server._store = SimpleNamespace(  # type: ignore[assignment]
        artifact_prefix="repairs/r-baselines/run-001",
        save_attempt=MagicMock(side_effect=RuntimeError("write failed")),
        load_attempt=MagicMock(side_effect=FileNotFoundError("missing")),
    )

    request = server.AssignmentRequest(request_id=REQUEST_ID, worker_id=WORKER_ID)
    response = asyncio.run(server.next_config(request))
    body = json.loads(response.body)
    update = server.AssignmentUpdate(
        assignment_id=body["assignment_id"],
        request_id=REQUEST_ID,
        worker_id=WORKER_ID,
    )
    with pytest.raises(RuntimeError, match="write failed"):
        asyncio.run(server.start_assignment(update))

    assert queue.available == 0
    assert queue.reserved == 1
    assert queue.in_flight == 0
    assert queue.attempts == 0


def test_queue_rejects_heartbeat_and_terminal_update_at_expiry() -> None:
    config = _config("r_ctree", dataset="fixture")
    heartbeat_queue = QueueState.from_configs([config], max_attempts=2)
    heartbeat_assignment = _assignment_id(config, 1)
    _activate_lease(heartbeat_queue, config, heartbeat_assignment, 1, 10, now=100.0)

    assert not heartbeat_queue.heartbeat(
        heartbeat_assignment,
        WORKER_ID,
        REQUEST_ID,
        10,
        now=110.0,
    )

    terminal_queue = QueueState.from_configs([config], max_attempts=2)
    terminal_assignment = _assignment_id(config, 1)
    _activate_lease(terminal_queue, config, terminal_assignment, 1, 10, now=100.0)

    assert (
        terminal_queue.begin_finalization(
            terminal_assignment,
            worker_id=WORKER_ID,
            request_id=REQUEST_ID,
            now=110.0,
        )
        is None
    )


def test_expiration_and_explicit_failure_share_one_attempt_budget() -> None:
    config = _config("r_ctree", dataset="fixture")
    queue = QueueState.from_configs([config], max_attempts=3)

    first = _assignment_id(config, 1)
    _activate_lease(queue, config, first, 1, 10, now=100.0)
    assert queue.requeue_expired(now=110.0) == 1
    assert queue.next_candidate() == (config, 2)

    second = _assignment_id(config, 2)
    _activate_lease(queue, config, second, 2, 10, now=111.0)
    assert (
        queue.begin_finalization(
            second,
            worker_id=WORKER_ID,
            request_id=REQUEST_ID,
            now=112.0,
        )
        is not None
    )
    assert queue.finish_finalization(second, failed=True)
    assert queue.next_candidate() == (config, 3)

    third = _assignment_id(config, 3)
    _activate_lease(queue, config, third, 3, 10, now=113.0)
    assert queue.requeue_expired(now=123.0) == 1
    assert queue.attempts == 3
    assert queue.failed == 1
    assert queue.pending == 0


def test_restart_reconstructs_attempts_from_immutable_receipts() -> None:
    config = _config("r_ctree", dataset="fixture")
    first_assignment = _assignment_id(config, 1)
    second_assignment = _assignment_id(config, 2)
    attempts = (
        ControlReceipt(
            key="attempt-1",
            dataset=config.dataset,
            method_label=config.method.label,
            seed=config.seed,
            assignment_id=first_assignment,
            payload=_attempt_receipt(config, first_assignment, 1),
        ),
        ControlReceipt(
            key="attempt-2",
            dataset=config.dataset,
            method_label=config.method.label,
            seed=config.seed,
            assignment_id=second_assignment,
            payload=_attempt_receipt(config, second_assignment, 2),
        ),
    )
    failures = (
        ControlReceipt(
            key="failure-2",
            dataset=config.dataset,
            method_label=config.method.label,
            seed=config.seed,
            assignment_id=second_assignment,
            payload=_failure_receipt(config, second_assignment, 2),
        ),
    )
    store = SimpleNamespace(
        list_control_receipts=lambda receipt_type, stage, task: (
            attempts if receipt_type == "attempts" else failures
        )
    )
    server._expected_provenance = _expected_provenance()

    durable = _load_durable_attempt_state(
        store,  # type: ignore[arg-type]
        (config,),
        stage="rankings",
        task="classification",
        max_attempts=3,
    )
    restarted = QueueState.from_configs(
        [config],
        max_attempts=3,
        attempts=durable.counts,
    )

    assert durable.counts == {config.key: 2}
    assert durable.open_attempts == {}
    assert restarted.next_candidate() == (config, 3)


def test_restart_rejects_assignment_id_unbound_to_cell_attempt() -> None:
    config = _config("r_ctree", dataset="fixture")
    arbitrary_assignment_id = "f" * 32
    attempts = (
        ControlReceipt(
            key="attempt-1",
            dataset=config.dataset,
            method_label=config.method.label,
            seed=config.seed,
            assignment_id=arbitrary_assignment_id,
            payload=_attempt_receipt(config, arbitrary_assignment_id, 1),
        ),
    )
    store = SimpleNamespace(
        list_control_receipts=lambda receipt_type, stage, task: (
            attempts if receipt_type == "attempts" else ()
        )
    )
    server._expected_provenance = _expected_provenance()

    with pytest.raises(ValueError, match="assignment_id"):
        _load_durable_attempt_state(
            store,  # type: ignore[arg-type]
            (config,),
            stage="rankings",
            task="classification",
            max_attempts=3,
        )


def test_completion_validation_holds_lease_across_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("r_ctree", dataset="fixture")
    queue = QueueState.from_configs([config], max_attempts=1)
    assignment_id = _assignment_id(config, 1)
    _activate_lease(queue, config, assignment_id, 1, 30)
    server._queues = {"rankings/classification": queue}
    server._active_stage = "rankings"
    server._store = SimpleNamespace()  # type: ignore[assignment]

    def validate(cfg: ExperimentConfig) -> None:
        assert cfg == config
        assert queue.requeue_expired(now=float("inf")) == 0

    monkeypatch.setattr(server, "_validate_assignment_artifact", validate)
    update = server.AssignmentUpdate(
        assignment_id=assignment_id,
        request_id=REQUEST_ID,
        worker_id=WORKER_ID,
    )

    assert asyncio.run(server.complete_assignment(update)) == {"completed": True}
    assert queue.completed == 1
    assert queue.failed == 0


def test_terminal_acknowledgements_require_durable_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("r_ctree", dataset="fixture")
    completion_queue = QueueState.from_configs([config], max_attempts=1)
    completion_assignment = _assignment_id(config, 1)
    _activate_lease(completion_queue, config, completion_assignment, 1, 30)
    server._queues = {"rankings/classification": completion_queue}
    server._active_stage = "rankings"
    server._expected_provenance = _expected_provenance()
    server._store = SimpleNamespace()  # type: ignore[assignment]
    update = server.AssignmentUpdate(
        assignment_id=completion_assignment,
        request_id=REQUEST_ID,
        worker_id=WORKER_ID,
    )
    monkeypatch.setattr(
        server,
        "_validate_assignment_artifact",
        lambda cfg: (_ for _ in ()).throw(FileNotFoundError("missing artifact")),
    )

    with pytest.raises(HTTPException) as completion_error:
        asyncio.run(server.complete_assignment(update))
    assert completion_error.value.status_code == 409
    assert completion_queue.in_flight == 1

    failure_queue = QueueState.from_configs([config], max_attempts=1)
    failure_assignment = _assignment_id(config, 1)
    _activate_lease(failure_queue, config, failure_assignment, 1, 30)
    server._queues = {"rankings/classification": failure_queue}
    failure_update = server.AssignmentUpdate(
        assignment_id=failure_assignment,
        request_id=REQUEST_ID,
        worker_id=WORKER_ID,
    )
    server._store = SimpleNamespace(  # type: ignore[assignment]
        load_failure=lambda stage, cfg, assignment_id: (_ for _ in ()).throw(
            FileNotFoundError("missing receipt")
        )
    )

    with pytest.raises(HTTPException) as failure_error:
        asyncio.run(server.fail_assignment(failure_update))
    assert failure_error.value.status_code == 409
    assert failure_queue.in_flight == 1

    server._store = SimpleNamespace(  # type: ignore[assignment]
        load_failure=lambda stage, cfg, assignment_id: _failure_receipt(
            config,
            failure_assignment,
            1,
        )
    )
    assert asyncio.run(server.fail_assignment(failure_update)) == {"failed": True}
    assert failure_queue.failed == 1


def test_worker_rejects_namespace_manifest_and_stage_mismatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SimpleNamespace(artifact_prefix="repairs/r-baselines/run-001")
    config = _config("r_ctree", dataset="fixture")
    assignment = {
        "artifact_prefix": "repairs/r-baselines/run-001",
        "aws_account_id": "123456789012",
        "campaign_sha256": "e" * 64,
        "canonical_manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "container_image": "repository@sha256:" + "a" * 64,
        "gate_receipt_sha256": GATE_RECEIPT_SHA256,
        "git_sha": "a" * 40,
        "manifest_sha256": "a" * 64,
        "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
        "stage": "rankings",
        "attempt": 1,
        "assignment_id": _assignment_id(config, 1),
        "request_id": REQUEST_ID,
        "worker_id": WORKER_ID,
        **server._serialize_config(config),
    }
    monkeypatch.setenv("CITREES_STAGE", "rankings")
    monkeypatch.setenv("CITREES_ARTIFACT_PREFIX", "repairs/r-baselines/run-001")
    monkeypatch.setenv("CITREES_CAMPAIGN_SHA256", "e" * 64)
    monkeypatch.setenv("CITREES_CANONICAL_MANIFEST_SHA256", CANONICAL_MANIFEST_SHA256)
    monkeypatch.setenv("CITREES_GATE_RECEIPT_SHA256", GATE_RECEIPT_SHA256)
    monkeypatch.setenv("CITREES_MANIFEST_SHA256", "a" * 64)
    monkeypatch.setenv("CITREES_RUNTIME_CONTRACT_SHA256", RUNTIME_CONTRACT_SHA256)
    monkeypatch.setattr(
        "paper.benchmark.infra.aws.get_aws_account_id",
        lambda: "123456789012",
    )
    monkeypatch.setenv("CITREES_IMAGE_URI", "repository@sha256:" + "a" * 64)
    monkeypatch.setenv("GIT_SHA", "a" * 40)

    def validate(payload: dict[str, object]) -> None:
        _validate_assignment(
            payload,
            store,  # type: ignore[arg-type]
            approved_configs={config.key: config},
            worker_id=WORKER_ID,
            request_id=REQUEST_ID,
        )

    validate(assignment)

    with pytest.raises(RuntimeError, match="provenance mismatch"):
        validate({**assignment, "artifact_prefix": "repairs/other-run"})
    with pytest.raises(RuntimeError, match="stage mismatch"):
        validate({**assignment, "stage": "metrics"})
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        validate({**assignment, "manifest_sha256": "b" * 64})
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        validate({**assignment, "aws_account_id": "999999999999"})

    outside = replace(config, dataset="outside")
    with pytest.raises(RuntimeError, match="outside the active manifest"):
        validate(
            {
                **assignment,
                "assignment_id": _assignment_id(outside, 1),
                **server._serialize_config(outside),
            }
        )


def test_needs_metrics_work_when_no_metrics_artifact_exists():
    cfg = _config()
    store = _FakeStore(
        rankings_df=pd.DataFrame({"feature_ranking": [list(range(1000))]}),
        metrics_df=pd.DataFrame({"k": []}),
    )
    assert _needs_metrics_work(store, cfg, completed_metrics=set())


def test_needs_metrics_work_when_existing_metrics_miss_high_p_bridge():
    cfg = _config(n_features=1200)
    observed_k = [5, 10, 25, 50, 100]
    store = _FakeStore(
        rankings_df=_valid_rankings(cfg, n_features=1200),
        metrics_df=_metrics(cfg, n_features=1200, k_values=observed_k),
    )
    with pytest.raises(ValueError, match="exactly one row"):
        _needs_metrics_work(store, cfg, completed_metrics={cfg.key})


def test_existing_metrics_covering_high_p_bridge_are_complete():
    cfg = _config(n_features=1200)
    required_k = [5, 10, 25, 50, 100, 150, 200, 300, 500, 600, 750, 900, 1000, 1200]
    store = _FakeStore(
        rankings_df=_valid_rankings(cfg, n_features=1200),
        metrics_df=_metrics(cfg, n_features=1200, k_values=required_k),
    )
    assert not _needs_metrics_work(store, cfg, completed_metrics={cfg.key})


def test_existing_metrics_with_only_global_k_coverage_are_incomplete() -> None:
    cfg = _config(n_features=1200)
    required_k = [5, 10, 25, 50, 100, 150, 200, 300, 500, 600, 750, 900, 1000, 1200]
    metrics = _metrics(cfg, n_features=1200, k_values=required_k)
    one_row_per_k = metrics.drop_duplicates("k")
    store = _FakeStore(
        rankings_df=_valid_rankings(cfg, n_features=1200),
        metrics_df=one_row_per_k,
    )

    with pytest.raises(ValueError, match="exactly one row"):
        _needs_metrics_work(store, cfg, completed_metrics={cfg.key})
