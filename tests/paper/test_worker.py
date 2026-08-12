"""Tests for durable worker assignment finalization."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from paper.benchmark.api import worker
from paper.benchmark.experiments.r_cforest_reproducibility import (
    gate_receipt_s3_key,
)
from paper.benchmark.pipeline.campaign_gate import (
    ApprovedCampaignGate,
    CampaignGateIdentity,
)
from paper.benchmark.pipeline.manifest import ManifestCell, RerunManifest
from paper.benchmark.pipeline.types import DatasetIdentity, ExperimentConfig, MethodConfig, Result
from paper.benchmark.pipeline.validation import derive_assignment_id

pytestmark = pytest.mark.paper
REQUEST_ID = "1" * 32
WORKER_ID = "2" * 32
CANONICAL_MANIFEST_SHA256 = "c" * 64
CANONICAL_MANIFEST_S3_KEY = f"canonical-rerun-manifests/{CANONICAL_MANIFEST_SHA256}.csv"
GATE_RECEIPT_SHA256 = "d" * 64
GATE_RECEIPT_S3_KEY = gate_receipt_s3_key(GATE_RECEIPT_SHA256)
MANIFEST_SHA256 = "b" * 64
MANIFEST_S3_KEY = f"rerun-manifests/{MANIFEST_SHA256}.csv"
RUNTIME_CONTRACT_SHA256 = "f" * 64
RUNTIME_CONTRACT_S3_KEY = f"runtime-contracts/{RUNTIME_CONTRACT_SHA256}.json"


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        method=MethodConfig("r_ctree"),
        dataset="glass",
        seed=0,
        task="classification",
        dataset_identity=DatasetIdentity("d" * 64, n_samples=214, n_features=9),
    )


def _assignment_id(*, stage: str = "rankings") -> str:
    return derive_assignment_id(
        _config(),
        stage=stage,
        attempt=1,
        expected_provenance={
            "artifact_prefix": "repairs/run-001",
            "aws_account_id": "123456789012",
            "campaign_sha256": "e" * 64,
            "canonical_manifest_sha256": CANONICAL_MANIFEST_SHA256,
            "container_image": "repository@sha256:" + "a" * 64,
            "gate_receipt_sha256": GATE_RECEIPT_SHA256,
            "git_sha": "a" * 40,
            "manifest_sha256": MANIFEST_SHA256,
            "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
        },
    )


def _approved_gate() -> ApprovedCampaignGate:
    config = _config()
    manifest = RerunManifest(
        sha256=MANIFEST_SHA256,
        campaign_sha256="e" * 64,
        runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
        cells=(
            ManifestCell(
                config=config,
                target_aws_account_id="123456789012",
                dataset_source="real",
                rerun_reason="adapter_correction",
                historically_omitted=False,
                stage1_required=True,
                stage2_required=True,
            ),
        ),
    )
    return ApprovedCampaignGate(
        identity=CampaignGateIdentity(
            account_id="123456789012",
            campaign_sha256="e" * 64,
            canonical_manifest_s3_key=CANONICAL_MANIFEST_S3_KEY,
            canonical_manifest_sha256=CANONICAL_MANIFEST_SHA256,
            gate_receipt_s3_key=GATE_RECEIPT_S3_KEY,
            gate_receipt_sha256=GATE_RECEIPT_SHA256,
            manifest_s3_key=MANIFEST_S3_KEY,
            manifest_sha256=MANIFEST_SHA256,
            runtime_contract_s3_key=RUNTIME_CONTRACT_S3_KEY,
            runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
        ),
        canonical_manifest=manifest,
        manifest=manifest,
        runtime_contract={},
        gate_receipt={},
    )


def test_approved_config_map_preserves_same_named_cells_across_tasks() -> None:
    classification = _config()
    regression = ExperimentConfig(
        method=classification.method,
        dataset=classification.dataset,
        seed=classification.seed,
        task="regression",
        dataset_identity=classification.dataset_identity,
    )
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
        for config in (classification, regression)
    )
    manifest = RerunManifest(
        sha256=MANIFEST_SHA256,
        campaign_sha256="e" * 64,
        runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
        cells=cells,
    )
    base_gate = _approved_gate()
    gate = ApprovedCampaignGate(
        identity=base_gate.identity,
        canonical_manifest=manifest,
        manifest=manifest,
        runtime_contract={},
        gate_receipt={},
    )

    approved = worker._approved_configs_for_stage(gate, "rankings")

    assert len(approved) == 2
    assert approved[classification.key] == classification
    assert approved[regression.key] == regression
    assert classification.key != regression.key


class _FailureStore:
    artifact_prefix = "repairs/run-001"

    def __init__(self, *, fail_save: bool = False) -> None:
        self.fail_save = fail_save
        self.receipt: dict[str, object] | None = None
        self.expected_provenance: dict[str, str] | None = None

    def save_failure(
        self,
        stage: str,
        config: ExperimentConfig,
        assignment_id: str,
        receipt: dict[str, object],
        *,
        expected_provenance: dict[str, str],
    ) -> str:
        if self.fail_save:
            raise RuntimeError("S3 unavailable")
        self.receipt = receipt
        self.expected_provenance = expected_provenance
        return "s3://bucket/repairs/run-001/failures/receipt.json"

    def load_failure(
        self,
        stage: str,
        config: ExperimentConfig,
        assignment_id: str,
    ) -> dict[str, object]:
        del stage, config, assignment_id
        if self.receipt is None:
            raise FileNotFoundError("missing")
        return self.receipt

    def bind_write_guard(self, guard: object) -> None:
        del guard

    def clear_write_guard(self) -> None:
        return None


def _response(status_code: int, payload: dict[str, object] | None = None) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("GET", "http://api.test/fixture"),
    )


def _scope_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "artifact_prefix": "repairs/run-001",
        "aws_account_id": "123456789012",
        "campaign_sha256": "e" * 64,
        "canonical_manifest_s3_key": CANONICAL_MANIFEST_S3_KEY,
        "canonical_manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "container_image": "repository@sha256:" + "a" * 64,
        "gate_receipt_s3_key": GATE_RECEIPT_S3_KEY,
        "gate_receipt_sha256": GATE_RECEIPT_SHA256,
        "git_sha": "a" * 40,
        "manifest_s3_key": MANIFEST_S3_KEY,
        "manifest_sha256": MANIFEST_SHA256,
        "runtime_contract_s3_key": RUNTIME_CONTRACT_S3_KEY,
        "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
        "stage": "rankings",
        "queues": {
            "rankings/classification": {
                "pending": 0,
            }
        },
    }
    payload.update(overrides)
    return payload


class _ScriptedClient:
    def __init__(
        self,
        *,
        gets: list[httpx.Response | Exception],
        posts: list[httpx.Response | Exception],
    ) -> None:
        self.gets = gets
        self.posts = posts
        self.closed = False
        self.post_calls = 0

    def __enter__(self) -> _ScriptedClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True

    @staticmethod
    def _pop(
        scripted: list[httpx.Response | Exception],
    ) -> httpx.Response:
        if not scripted:
            raise AssertionError("unexpected HTTP call")
        value = scripted.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def get(self, path: str) -> httpx.Response:
        assert path == "/status"
        return self._pop(self.gets)

    def post(self, path: str, **kwargs: object) -> httpx.Response:
        del kwargs
        assert path in {"/next", "/start", "/heartbeat", "/complete", "/fail"}
        self.post_calls += 1
        return self._pop(self.posts)


def _configure_worker(
    monkeypatch: pytest.MonkeyPatch,
    client: _ScriptedClient,
) -> None:
    _configure_provenance(monkeypatch)
    monkeypatch.setenv("CITREES_STAGE", "rankings")
    monkeypatch.setattr(worker, "_shutdown", False)
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    monkeypatch.setattr(worker, "_load_gate_approved_campaign", lambda store: _approved_gate())
    monkeypatch.setattr(
        worker.S3Store,
        "from_env",
        classmethod(
            lambda cls, validate_uploads=False: SimpleNamespace(artifact_prefix="repairs/run-001")
        ),
    )
    monkeypatch.setattr(worker.httpx, "Client", lambda **kwargs: client)


def _configure_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CITREES_ARTIFACT_PREFIX", "repairs/run-001")
    monkeypatch.setenv("CITREES_CAMPAIGN_SHA256", "e" * 64)
    monkeypatch.setenv("CITREES_CANONICAL_MANIFEST_S3_KEY", CANONICAL_MANIFEST_S3_KEY)
    monkeypatch.setenv("CITREES_CANONICAL_MANIFEST_SHA256", CANONICAL_MANIFEST_SHA256)
    monkeypatch.setenv("CITREES_GATE_RECEIPT_S3_KEY", GATE_RECEIPT_S3_KEY)
    monkeypatch.setenv("CITREES_GATE_RECEIPT_SHA256", GATE_RECEIPT_SHA256)
    monkeypatch.setenv("CITREES_MANIFEST_S3_KEY", MANIFEST_S3_KEY)
    monkeypatch.setenv("CITREES_MANIFEST_SHA256", MANIFEST_SHA256)
    monkeypatch.setenv("CITREES_RUNTIME_CONTRACT_S3_KEY", RUNTIME_CONTRACT_S3_KEY)
    monkeypatch.setenv("CITREES_RUNTIME_CONTRACT_SHA256", RUNTIME_CONTRACT_SHA256)
    monkeypatch.setenv("CITREES_IMAGE_URI", "repository@sha256:" + "a" * 64)
    monkeypatch.setenv("GIT_SHA", "a" * 40)
    monkeypatch.setattr(
        "paper.benchmark.infra.aws.get_aws_account_id",
        lambda: "123456789012",
    )


def test_worker_live_runtime_probe_mismatch_precedes_any_api_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provenance(monkeypatch)
    store = SimpleNamespace(artifact_prefix="repairs/run-001")
    approved = _approved_gate()
    gate_loader = MagicMock(return_value=approved)
    live_probe = MagicMock(side_effect=RuntimeError("running runtime contract mismatch"))
    api_client_factory = MagicMock(side_effect=AssertionError("API client was constructed"))
    monkeypatch.setattr(
        worker.S3Store,
        "from_env",
        classmethod(lambda cls, validate_uploads=False: store),
    )
    monkeypatch.setattr(
        "paper.benchmark.pipeline.campaign_gate.configured_campaign_gate_identity",
        lambda: approved.identity,
    )
    monkeypatch.setattr(
        "paper.benchmark.pipeline.campaign_gate.load_approved_campaign_gate",
        gate_loader,
    )
    monkeypatch.setattr(
        "paper.benchmark.experiments.r_cforest_reproducibility.require_running_runtime_contract",
        live_probe,
    )
    monkeypatch.setattr(worker.httpx, "Client", api_client_factory)

    with pytest.raises(RuntimeError, match="running runtime contract mismatch"):
        worker.run_worker(
            "http://api.test",
            poll_interval=0,
            max_api_failures=2,
        )

    gate_loader.assert_called_once_with(store, approved.identity)
    live_probe.assert_called_once_with(approved.runtime_contract)
    api_client_factory.assert_not_called()


def test_no_rankings_becomes_durable_failed_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provenance(monkeypatch)
    store = _FailureStore()
    response = MagicMock(status_code=200)
    client = MagicMock()
    client.post.return_value = response
    result = Result(config=_config(), status="no_rankings", elapsed_seconds=1.25)

    reusable = worker._finalize_assignment(
        client,
        store,  # type: ignore[arg-type]
        "metrics",
        "a" * 32,
        1,
        WORKER_ID,
        REQUEST_ID,
        result,
        poll_interval=0,
    )

    assert not reusable
    assert store.receipt is not None
    assert store.receipt["error_type"] == "MissingRankingArtifact"
    assert store.receipt["elapsed_seconds"] == pytest.approx(1.25)
    assert store.receipt["artifact_prefix"] == "repairs/run-001"
    assert store.receipt["canonical_manifest_sha256"] == CANONICAL_MANIFEST_SHA256
    assert store.receipt["gate_receipt_sha256"] == GATE_RECEIPT_SHA256
    assert store.receipt["manifest_sha256"] == MANIFEST_SHA256
    assert store.receipt["aws_account_id"] == "123456789012"
    assert store.receipt["campaign_sha256"] == "e" * 64
    assert store.receipt["attempt"] == 1
    assert store.expected_provenance == {
        "artifact_prefix": "repairs/run-001",
        "aws_account_id": "123456789012",
        "campaign_sha256": "e" * 64,
        "canonical_manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "container_image": "repository@sha256:" + "a" * 64,
        "gate_receipt_sha256": GATE_RECEIPT_SHA256,
        "git_sha": "a" * 40,
        "manifest_sha256": MANIFEST_SHA256,
        "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
    }
    client.post.assert_called_once_with(
        "/fail",
        json={
            "assignment_id": "a" * 32,
            "request_id": REQUEST_ID,
            "worker_id": WORKER_ID,
        },
    )


def test_receipt_failure_leaves_lease_unacknowledged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provenance(monkeypatch)
    store = _FailureStore(fail_save=True)
    client = MagicMock()
    result = Result(
        config=_config(),
        status="failed",
        error="fit failed",
        error_type="RuntimeError",
    )

    reusable = worker._finalize_assignment(
        client,
        store,  # type: ignore[arg-type]
        "rankings",
        "b" * 32,
        1,
        WORKER_ID,
        REQUEST_ID,
        result,
        poll_interval=0,
    )

    assert not reusable
    client.post.assert_not_called()


def test_ambiguous_failure_receipt_is_loaded_and_acknowledged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provenance(monkeypatch)

    class AmbiguousStore(_FailureStore):
        def __init__(self) -> None:
            super().__init__()
            self.load_calls = 0

        def save_failure(
            self,
            stage: str,
            config: ExperimentConfig,
            assignment_id: str,
            receipt: dict[str, object],
            *,
            expected_provenance: dict[str, str],
        ) -> str:
            self.receipt = receipt
            self.expected_provenance = expected_provenance
            raise RuntimeError("response lost after commit")

        def load_failure(
            self,
            stage: str,
            config: ExperimentConfig,
            assignment_id: str,
        ) -> dict[str, object]:
            self.load_calls += 1
            return super().load_failure(stage, config, assignment_id)

    store = AmbiguousStore()
    client = MagicMock()
    client.post.return_value = MagicMock(status_code=200)
    assignment_id = _assignment_id()

    reusable = worker._finalize_assignment(
        client,
        store,  # type: ignore[arg-type]
        "rankings",
        assignment_id,
        1,
        WORKER_ID,
        REQUEST_ID,
        Result(
            config=_config(),
            status="failed",
            error="fit failed",
            error_type="RuntimeError",
            traceback="RuntimeError: fit failed",
            hostname="worker-1",
        ),
        poll_interval=0,
    )

    assert not reusable
    assert store.load_calls == 1
    client.post.assert_called_once()


@pytest.mark.parametrize("status", ["done", "skipped"])
def test_acknowledgement_failure_does_not_escape_worker(
    status: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker,
        "_acknowledge_assignment",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("API unavailable")),
    )

    reusable = worker._finalize_assignment(
        MagicMock(),
        _FailureStore(),  # type: ignore[arg-type]
        "rankings",
        "c" * 32,
        1,
        WORKER_ID,
        REQUEST_ID,
        Result(config=_config(), status=status),  # type: ignore[arg-type]
        poll_interval=0,
    )
    assert not reusable


@pytest.mark.parametrize("status", ["done", "skipped"])
def test_successful_acknowledgement_allows_worker_reuse(
    status: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker, "_acknowledge_assignment", lambda *args, **kwargs: None)

    reusable = worker._finalize_assignment(
        MagicMock(),
        _FailureStore(),  # type: ignore[arg-type]
        "rankings",
        "c" * 32,
        1,
        WORKER_ID,
        REQUEST_ID,
        Result(config=_config(), status=status),  # type: ignore[arg-type]
        poll_interval=0,
    )

    assert reusable


def test_unexpected_execution_exception_is_recorded_and_does_not_escape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(worker, "_heartbeat_loop", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker,
        "_save_failure_receipt",
        lambda store, stage, assignment_id, attempt, worker_id, request_id, result: (
            captured.update(
                {
                    "attempt": attempt,
                    "elapsed_seconds": result.elapsed_seconds,
                    "error": result.error,
                    "error_type": result.error_type,
                    "traceback": result.traceback,
                }
            )
            or "s3://bucket/failure.json"
        ),
    )
    monkeypatch.setattr(worker, "_acknowledge_assignment", lambda *args, **kwargs: None)

    def crash(config: ExperimentConfig, store: object) -> Result:
        raise ValueError("unexpected cell failure")

    clock = iter([100.0, 103.5])
    monkeypatch.setattr(worker.time, "perf_counter", lambda: next(clock))

    reusable = worker._execute_assignment(
        MagicMock(),
        _FailureStore(),  # type: ignore[arg-type]
        "http://127.0.0.1:8000",
        "rankings",
        "d" * 32,
        2,
        WORKER_ID,
        REQUEST_ID,
        30,
        _config(),
        crash,
        poll_interval=0,
    )

    assert not reusable
    assert captured["attempt"] == 2
    assert captured["elapsed_seconds"] == pytest.approx(3.5)
    assert captured["error"] == "unexpected cell failure"
    assert captured["error_type"] == "ValueError"
    assert "ValueError: unexpected cell failure" in str(captured["traceback"])


def test_timeout_failure_allows_worker_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker, "_heartbeat_loop", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker,
        "_save_failure_receipt",
        lambda *args, **kwargs: "s3://bucket/failure.json",
    )
    monkeypatch.setattr(worker, "_acknowledge_assignment", lambda *args, **kwargs: None)

    reusable = worker._execute_assignment(
        MagicMock(),
        _FailureStore(),  # type: ignore[arg-type]
        "http://127.0.0.1:8000",
        "rankings",
        "d" * 32,
        1,
        WORKER_ID,
        REQUEST_ID,
        30,
        _config(),
        lambda config, store: Result(  # noqa: ARG005
            config=config,
            status="failed",
            error="selection exceeded its wall-clock limit",
            error_type="TimeoutError",
        ),
        poll_interval=0,
    )

    assert reusable


def test_non_reusable_assignment_stops_before_second_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedClient(
        gets=[_response(200, _scope_payload())],
        posts=[],
    )
    _configure_worker(monkeypatch, client)
    assignment = _response(
        200,
        {
            "stage": "rankings",
            "assignment_id": "a" * 32,
            "attempt": 1,
            "lease_seconds": 60,
        },
    )
    next_assignment = MagicMock(
        side_effect=[
            assignment,
            AssertionError("worker polled after a non-reusable outcome"),
        ]
    )
    monkeypatch.setattr(worker, "_next_assignment", next_assignment)
    monkeypatch.setattr(worker, "_validate_assignment", lambda *args, **kwargs: _config())
    monkeypatch.setattr(worker, "_start_assignment", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_execute_assignment", lambda *args, **kwargs: False)

    worker.run_worker(
        "http://api.test",
        poll_interval=0,
        max_api_failures=2,
    )

    assert next_assignment.call_count == 1
    assert client.closed


@pytest.mark.parametrize(
    "failure",
    [
        httpx.RemoteProtocolError("server disconnected"),
        httpx.ConnectError("connection refused"),
    ],
)
def test_worker_bounds_all_transport_failures_and_closes_client(
    failure: httpx.HTTPError,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedClient(
        gets=[_response(200, _scope_payload())],
        posts=[failure, failure],
    )
    _configure_worker(monkeypatch, client)

    with pytest.raises(RuntimeError, match="2 consecutive times"):
        worker.run_worker(
            "http://api.test",
            poll_interval=0,
            max_api_failures=2,
        )

    assert client.post_calls == 2
    assert client.closed


def test_worker_bounds_persistent_5xx_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedClient(
        gets=[_response(200, _scope_payload())],
        posts=[_response(503), _response(503)],
    )
    _configure_worker(monkeypatch, client)

    with pytest.raises(RuntimeError, match="assignment poll failed 2"):
        worker.run_worker(
            "http://api.test",
            poll_interval=0,
            max_api_failures=2,
        )

    assert client.post_calls == 2
    assert client.closed


def test_worker_validates_scope_before_leasing_and_exits_on_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedClient(
        gets=[_response(200, _scope_payload(manifest_sha256="c" * 64))],
        posts=[],
    )
    _configure_worker(monkeypatch, client)

    with pytest.raises(RuntimeError, match="queue scope mismatch"):
        worker.run_worker(
            "http://api.test",
            poll_interval=0,
            max_api_failures=2,
        )

    assert client.post_calls == 0
    assert client.closed


def test_worker_closes_client_after_clean_queue_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedClient(
        gets=[
            _response(200, _scope_payload()),
            _response(200, _scope_payload()),
        ],
        posts=[_response(204)],
    )
    _configure_worker(monkeypatch, client)

    worker.run_worker(
        "http://api.test",
        poll_interval=0,
        max_api_failures=2,
    )

    assert client.post_calls == 1
    assert client.closed


def test_successful_api_calls_reset_the_consecutive_failure_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope_failure = httpx.ConnectError("scope unavailable")
    poll_failure = httpx.ConnectError("poll unavailable")
    client = _ScriptedClient(
        gets=[
            scope_failure,
            _response(200, _scope_payload()),
            _response(200, _scope_payload()),
        ],
        posts=[
            poll_failure,
            _response(204),
        ],
    )
    _configure_worker(monkeypatch, client)

    worker.run_worker(
        "http://api.test",
        poll_interval=0,
        max_api_failures=2,
    )

    assert client.closed


def test_heartbeat_marks_a_missing_lease_as_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedClient(gets=[], posts=[_response(404)])
    monkeypatch.setattr(worker.httpx, "Client", lambda **kwargs: client)
    lease_lost = threading.Event()

    class ImmediateEvent:
        def wait(self, timeout: float) -> bool:
            del timeout
            return False

    worker._heartbeat_loop(
        "http://api.test",
        "e" * 32,
        WORKER_ID,
        REQUEST_ID,
        3,
        ImmediateEvent(),  # type: ignore[arg-type]
        lease_lost,
    )

    assert lease_lost.is_set()
    assert client.closed


def test_heartbeat_partition_marks_lease_lost_at_local_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedClient(
        gets=[],
        posts=[httpx.ConnectError("partition")],
    )
    monkeypatch.setattr(worker.httpx, "Client", lambda **kwargs: client)
    monkeypatch.setattr(
        worker.time,
        "monotonic",
        MagicMock(side_effect=[0.0, 0.0, 5.0, 35.0]),
    )
    lease_lost = threading.Event()

    class ImmediateEvent:
        def wait(self, timeout: float) -> bool:
            del timeout
            return False

    worker._heartbeat_loop(
        "http://api.test",
        "e" * 32,
        WORKER_ID,
        REQUEST_ID,
        30,
        ImmediateEvent(),  # type: ignore[arg-type]
        lease_lost,
    )

    assert lease_lost.is_set()
    assert client.closed


def test_terminal_queue_status_reuses_exact_campaign_scope_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = {
        "campaign_sha256": "wrong-campaign",
        "manifest_sha256": "wrong-manifest",
        "queues": {},
    }
    response = MagicMock()
    response.json.return_value = status
    client = MagicMock()
    client.get.return_value = response
    store = MagicMock()

    def reject_scope(data: dict[str, Any], observed_store: Any) -> None:
        assert data == status
        assert observed_store is store
        raise RuntimeError("queue scope mismatch")

    monkeypatch.setattr(worker, "_validate_api_scope", reject_scope)

    with pytest.raises(RuntimeError, match="queue scope mismatch"):
        worker._queue_has_outstanding_work(client, store)

    client.get.assert_called_once_with("/status")
    response.raise_for_status.assert_called_once_with()


def test_lost_lease_is_not_acknowledged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finalized = False

    def lose_lease(*args: object) -> None:
        lease_lost = args[6]
        assert isinstance(lease_lost, threading.Event)
        lease_lost.set()

    def finalize(*args: object, **kwargs: object) -> None:
        nonlocal finalized
        finalized = True

    monkeypatch.setattr(worker, "_heartbeat_loop", lose_lease)
    monkeypatch.setattr(worker, "_finalize_assignment", finalize)

    reusable = worker._execute_assignment(
        MagicMock(),
        _FailureStore(),  # type: ignore[arg-type]
        "http://api.test",
        "rankings",
        "f" * 32,
        1,
        WORKER_ID,
        REQUEST_ID,
        30,
        _config(),
        lambda config, store: Result(config=config, status="done"),
        poll_interval=0,
    )

    assert not finalized
    assert not reusable
