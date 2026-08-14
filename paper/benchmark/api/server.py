"""Manifest-bound FastAPI server for corrected benchmark orchestration.

On startup, the server downloads one content-addressed private manifest,
validates every cell against the current method and dataset inventories, and
builds queues for exactly one stage. Existing artifacts are validated before a
cell is excluded. Metrics execution cannot start until every approved ranking
artifact exists and passes validation.

Queues are shuffled so workers do not receive adjacent configurations in
manifest order. Configurations are serialized only when popped via ``/next``.

Start with:
    uvicorn paper.benchmark.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

from paper.benchmark.pipeline.manifest import RerunManifest
from paper.benchmark.pipeline.types import CellKey, StageType, TaskType

if TYPE_CHECKING:
    from paper.benchmark.adapters.store import S3Store
    from paper.benchmark.pipeline.types import ExperimentConfig


class AttemptOwnershipError(RuntimeError):
    """Raised when another API process owns a cell attempt receipt."""


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_config(cfg: Any) -> dict[str, Any]:
    """Serialize an ExperimentConfig to a JSON-safe dict."""
    return {
        "method": {
            "name": cfg.method.name,
            "params": dict(cfg.method.params),
        },
        "dataset": cfg.dataset,
        "dataset_identity": {
            "sha256": cfg.dataset_identity.sha256,
            "n_samples": cfg.dataset_identity.n_samples,
            "n_features": cfg.dataset_identity.n_features,
        },
        "seed": cfg.seed,
        "task": cfg.task,
    }


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_TASKS: tuple[TaskType, ...] = ("classification", "regression")
_VALID_STAGES: tuple[StageType, ...] = ("rankings", "metrics")

# Queue key: "{stage}/{task}" -> e.g. "rankings/classification"
QueueKey = str


@dataclass
class AssignmentLease:
    """One in-flight assignment and its expiry."""

    config: Any
    attempt: int
    expires_at: float
    request_id: str
    worker_id: str


@dataclass
class AssignmentOffer:
    """One uncharged assignment offered to a specific worker request."""

    config: Any
    attempt: int
    expires_at: float
    request_id: str
    worker_id: str
    starting: bool = False


@dataclass
class QueueState:
    """Available, in-flight, completed, and failed work for one queue."""

    _available: deque[Any]
    max_attempts: int
    _attempts: dict[CellKey, int] = field(default_factory=dict, repr=False)
    initial: int = 0
    served: int = 0
    completed: int = 0
    failed: int = 0
    _offers: dict[str, AssignmentOffer] = field(
        default_factory=dict,
        repr=False,
    )
    _leases: dict[str, AssignmentLease] = field(default_factory=dict, repr=False)
    _finalizing: dict[str, AssignmentLease] = field(default_factory=dict, repr=False)

    @classmethod
    def from_configs(
        cls,
        configs: list[Any],
        *,
        max_attempts: int,
        attempts: dict[CellKey, int] | None = None,
        initial: int | None = None,
        failed: int = 0,
    ) -> QueueState:
        """Create queue state from pending configs and durable attempt counts."""
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        durable_attempts = dict(attempts or {})
        for config in configs:
            count = durable_attempts.get(config.key, 0)
            if count < 0 or count >= max_attempts:
                raise ValueError(
                    f"pending config {config} has invalid durable attempt count {count}"
                )
        return cls(
            _available=deque(configs),
            max_attempts=max_attempts,
            _attempts=durable_attempts,
            initial=len(configs) + failed if initial is None else initial,
            failed=failed,
        )

    def requeue_expired(self, now: float | None = None) -> int:
        """Consume expired attempts and requeue cells below the fixed bound."""
        observed_at = time.monotonic() if now is None else now
        expired = [
            assignment_id
            for assignment_id, lease in self._leases.items()
            if lease.expires_at <= observed_at
        ]
        for assignment_id in expired:
            lease = self._leases.pop(assignment_id)
            if lease.attempt >= self.max_attempts:
                self.failed += 1
            else:
                self._available.append(lease.config)
        return len(expired)

    def requeue_expired_offers(self, now: float | None = None) -> int:
        """Return unconfirmed offers without consuming an attempt."""
        observed_at = time.monotonic() if now is None else now
        expired = [
            assignment_id
            for assignment_id, offer in self._offers.items()
            if not offer.starting and offer.expires_at <= observed_at
        ]
        for assignment_id in expired:
            offer = self._offers.pop(assignment_id)
            self._available.appendleft(offer.config)
        return len(expired)

    def next_candidate(self) -> tuple[Any, int] | None:
        """Return the next config and attempt number without mutating state."""
        if not self._available:
            return None
        config = self._available[0]
        attempt = self._attempts.get(config.key, 0) + 1
        if attempt > self.max_attempts:
            raise RuntimeError(f"attempt limit already exhausted for {config}")
        return config, attempt

    def offer_next(
        self,
        assignment_id: str,
        attempt: int,
        worker_id: str,
        request_id: str,
        offer_seconds: int,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Offer one candidate without recording or consuming an attempt."""
        observed_at = time.monotonic() if now is None else now
        if (
            assignment_id in self._offers
            or assignment_id in self._leases
            or assignment_id in self._finalizing
        ):
            raise ValueError(f"assignment already exists: {assignment_id}")
        self.requeue_expired_offers(observed_at)
        self.requeue_expired(observed_at)
        candidate = self.next_candidate()
        if candidate is None:
            raise RuntimeError("cannot reserve from an empty queue")
        config, expected_attempt = candidate
        if attempt != expected_attempt:
            raise ValueError(
                f"attempt {attempt} does not match next durable attempt {expected_attempt}"
            )
        reserved = self._available.popleft()
        if reserved is not config:
            raise RuntimeError("queue candidate changed during reservation")
        self._offers[assignment_id] = AssignmentOffer(
            config=config,
            attempt=attempt,
            expires_at=observed_at + offer_seconds,
            request_id=request_id,
            worker_id=worker_id,
        )
        self.served += 1
        return _serialize_config(config)

    def begin_start(
        self,
        assignment_id: str,
        *,
        worker_id: str,
        request_id: str,
        now: float | None = None,
    ) -> AssignmentOffer | None:
        """Hold a matching unexpired offer while its attempt receipt is written."""
        offer = self._offers.get(assignment_id)
        observed_at = time.monotonic() if now is None else now
        if (
            offer is None
            or offer.expires_at <= observed_at
            or offer.worker_id != worker_id
            or offer.request_id != request_id
            or offer.starting
        ):
            return None
        offer.starting = True
        return offer

    def activate_offer(
        self,
        assignment_id: str,
        lease_seconds: int,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Activate an offer only after its attempt receipt is durable."""
        offer = self._offers.get(assignment_id)
        if offer is None or not offer.starting:
            raise ValueError(f"assignment offer is not starting: {assignment_id}")
        expected_attempt = self._attempts.get(offer.config.key, 0) + 1
        if offer.attempt != expected_attempt:
            raise RuntimeError(
                f"offered attempt {offer.attempt} does not match "
                f"next durable attempt {expected_attempt}"
            )
        observed_at = time.monotonic() if now is None else now
        self._offers.pop(assignment_id)
        self._attempts[offer.config.key] = offer.attempt
        self._leases[assignment_id] = AssignmentLease(
            config=offer.config,
            attempt=offer.attempt,
            expires_at=observed_at + lease_seconds,
            request_id=offer.request_id,
            worker_id=offer.worker_id,
        )
        return _serialize_config(offer.config)

    def cancel_offer(self, assignment_id: str) -> bool:
        """Return an offer to the front of the queue without charging it."""
        offer = self._offers.pop(assignment_id, None)
        if offer is None:
            return False
        self._available.appendleft(offer.config)
        return True

    def release_start(self, assignment_id: str) -> bool:
        """Allow a failed receipt write to be retried for the same offer."""
        offer = self._offers.get(assignment_id)
        if offer is None:
            return False
        offer.starting = False
        return True

    def recover_lease(
        self,
        assignment_id: str,
        *,
        config: Any,
        attempt: int,
        worker_id: str,
        request_id: str,
        lease_seconds: int,
        now: float | None = None,
    ) -> None:
        """Restore one unmatched durable attempt after API restart."""
        if config in self._available:
            self._available.remove(config)
        if (
            assignment_id in self._offers
            or assignment_id in self._leases
            or assignment_id in self._finalizing
        ):
            raise ValueError(f"assignment already exists: {assignment_id}")
        if self._attempts.get(config.key) != attempt:
            raise ValueError("recovered attempt is not the cell's latest durable attempt")
        observed_at = time.monotonic() if now is None else now
        self._leases[assignment_id] = AssignmentLease(
            config=config,
            attempt=attempt,
            expires_at=observed_at + lease_seconds,
            request_id=request_id,
            worker_id=worker_id,
        )

    def heartbeat(
        self,
        assignment_id: str,
        worker_id: str,
        request_id: str,
        lease_seconds: int,
        *,
        now: float | None = None,
    ) -> bool:
        """Extend a lease only while it remains unexpired."""
        finalizing = self._finalizing.get(assignment_id)
        if finalizing is not None:
            if finalizing.worker_id != worker_id or finalizing.request_id != request_id:
                return False
            observed_at = time.monotonic() if now is None else now
            finalizing.expires_at = observed_at + lease_seconds
            return True
        lease = self._leases.get(assignment_id)
        if lease is None:
            return False
        observed_at = time.monotonic() if now is None else now
        if (
            lease.expires_at <= observed_at
            or lease.worker_id != worker_id
            or lease.request_id != request_id
        ):
            return False
        lease.expires_at = observed_at + lease_seconds
        return True

    def begin_finalization(
        self,
        assignment_id: str,
        *,
        worker_id: str,
        request_id: str,
        now: float | None = None,
    ) -> AssignmentLease | None:
        """Hold an owned lease so evidence validation cannot race expiry."""
        lease = self._leases.get(assignment_id)
        observed_at = time.monotonic() if now is None else now
        if (
            lease is None
            or lease.expires_at <= observed_at
            or lease.worker_id != worker_id
            or lease.request_id != request_id
        ):
            return None
        self._leases.pop(assignment_id)
        self._finalizing[assignment_id] = lease
        return lease

    def cancel_finalization(
        self,
        assignment_id: str,
        lease_seconds: int,
        *,
        now: float | None = None,
    ) -> bool:
        """Restore a held lease when terminal evidence is not valid."""
        lease = self._finalizing.pop(assignment_id, None)
        if lease is None:
            return False
        observed_at = time.monotonic() if now is None else now
        lease.expires_at = observed_at + lease_seconds
        self._leases[assignment_id] = lease
        return True

    def finish_finalization(self, assignment_id: str, *, failed: bool) -> bool:
        """Commit one terminal outcome after its durable evidence passes."""
        lease = self._finalizing.pop(assignment_id, None)
        if lease is None:
            return False
        if failed:
            self.failed += 1
        else:
            self.completed += 1
        return True

    def leased_assignment(
        self,
        assignment_id: str,
        *,
        worker_id: str | None = None,
        request_id: str | None = None,
        now: float | None = None,
    ) -> AssignmentLease | None:
        """Return an assignment only while its lease remains unexpired."""
        lease = self._leases.get(assignment_id)
        observed_at = time.monotonic() if now is None else now
        if (
            lease is None
            or lease.expires_at <= observed_at
            or (worker_id is not None and lease.worker_id != worker_id)
            or (request_id is not None and lease.request_id != request_id)
        ):
            return None
        return lease

    def assignment_for_request(
        self,
        worker_id: str,
        request_id: str,
    ) -> tuple[str, AssignmentOffer | AssignmentLease] | None:
        """Return an existing offer or lease for one idempotent worker request."""
        for assignments in (self._offers, self._leases, self._finalizing):
            for assignment_id, assignment in assignments.items():
                if assignment.worker_id == worker_id and assignment.request_id == request_id:
                    return assignment_id, assignment
        return None

    def owns_assignment(self, assignment_id: str) -> bool:
        """Return whether this queue holds any state for an assignment."""
        return (
            assignment_id in self._offers
            or assignment_id in self._leases
            or assignment_id in self._finalizing
        )

    @property
    def pending(self) -> int:
        """Number of available and in-flight cells."""
        return len(self._available) + len(self._offers) + len(self._leases) + len(self._finalizing)

    @property
    def available(self) -> int:
        """Number of cells ready to lease."""
        return len(self._available)

    @property
    def in_flight(self) -> int:
        """Number of cells with active leases."""
        return len(self._leases) + len(self._finalizing)

    @property
    def reserved(self) -> int:
        """Number of unconfirmed assignment offers."""
        return len(self._offers)

    @property
    def attempts(self) -> int:
        """Number of durable attempts recorded for this queue."""
        return sum(self._attempts.values())


class AssignmentUpdate(BaseModel):
    """Owned assignment accepted by start, heartbeat, and terminal endpoints."""

    assignment_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    worker_id: str = Field(pattern=r"^[0-9a-f]{32}$")


class AssignmentRequest(BaseModel):
    """Idempotent request for one uncharged assignment offer."""

    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    worker_id: str = Field(pattern=r"^[0-9a-f]{32}$")


_queues: dict[QueueKey, QueueState] = {}
_store: S3Store | None = None
_manifest: RerunManifest | None = None
_canonical_manifest_s3_key = ""
_canonical_manifest_sha256 = ""
_gate_receipt_s3_key = ""
_gate_receipt_sha256 = ""
_manifest_s3_key = ""
_runtime_contract_s3_key = ""
_active_stage: StageType = "rankings"
_active_methods: tuple[str, ...] = ()
_lease_seconds = 900
_max_cell_attempts = 0
_expected_provenance: dict[str, str] = {}
_queue_lock = asyncio.Lock()


def _key(stage: StageType, task: TaskType) -> QueueKey:
    return f"{stage}/{task}"


def _configured_stage() -> StageType:
    """Return the single explicitly configured queue stage."""
    raw = os.environ.get("CITREES_STAGE", "").strip()
    if raw not in _VALID_STAGES:
        raise RuntimeError(f"CITREES_STAGE must be one of {_VALID_STAGES}, got {raw!r}")
    return cast(StageType, raw)


def _configured_lease_seconds() -> int:
    """Return a lease long enough for heartbeat timeout and retry."""
    from paper.benchmark.config.constants import MIN_ASSIGNMENT_LEASE_SECONDS

    raw = os.environ.get("CITREES_LEASE_SECONDS", "").strip()
    try:
        lease_seconds = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"CITREES_LEASE_SECONDS must be an integer >= {MIN_ASSIGNMENT_LEASE_SECONDS}"
        ) from exc
    if lease_seconds < MIN_ASSIGNMENT_LEASE_SECONDS or str(lease_seconds) != raw:
        raise RuntimeError(
            f"CITREES_LEASE_SECONDS must be an integer >= {MIN_ASSIGNMENT_LEASE_SECONDS}"
        )
    return lease_seconds


def _configured_max_cell_attempts() -> int:
    """Return the explicit fixed attempt budget for each cell."""
    raw = os.environ.get("CITREES_MAX_CELL_ATTEMPTS", "").strip()
    try:
        max_attempts = int(raw)
    except ValueError as exc:
        raise RuntimeError("CITREES_MAX_CELL_ATTEMPTS must be a positive integer") from exc
    if max_attempts <= 0 or str(max_attempts) != raw:
        raise RuntimeError("CITREES_MAX_CELL_ATTEMPTS must be a positive integer")
    return max_attempts


def _preflight_manifest_datasets(store: S3Store, manifest: RerunManifest) -> int:
    """Validate every unique manifest dataset object before serving work."""
    from paper.benchmark.adapters.data import get_dataset_s3_payload

    bindings = sorted(
        {
            (
                cell.config.dataset,
                cell.config.task,
                cell.dataset_source,
                cell.config.dataset_identity,
            )
            for cell in manifest.cells
        },
        key=lambda binding: (binding[1], binding[2], binding[0], binding[3].sha256),
    )
    for name, task, source, identity in bindings:
        get_dataset_s3_payload(
            name,
            task,
            source,
            identity,
            bucket=store.bucket,
            client=store.client,
        )
    return len(bindings)


def _validated_ranking_feature_count(
    store: S3Store,
    cfg: ExperimentConfig,
    completed_rankings: set[CellKey],
    *,
    expected_provenance: dict[str, str] | None = None,
) -> int | None:
    """Return a completed ranking's feature count, or None when it is absent."""
    if cfg.key not in completed_rankings:
        return None

    from paper.benchmark.pipeline.validation import (
        validate_artifact_provenance,
        validate_ranking_artifact,
    )

    rankings_df = store.load("rankings", cfg)
    n_features = validate_ranking_artifact(rankings_df, cfg)
    if expected_provenance is not None:
        validate_artifact_provenance(rankings_df, expected_provenance)
    return n_features


def _needs_metrics_work(
    store: S3Store,
    cfg: ExperimentConfig,
    completed_metrics: set[CellKey],
    *,
    expected_provenance: dict[str, str] | None = None,
) -> bool:
    """Return True when Stage 2 is missing or violates the artifact contract."""
    if cfg.key not in completed_metrics:
        return True

    from paper.benchmark.pipeline.validation import (
        validate_artifact_provenance,
        validate_metrics_artifact,
        validate_metrics_ranking_payload,
        validate_ranking_artifact,
    )

    loaded_rankings = store.load_with_payload_sha256("rankings", cfg)
    rankings_df = loaded_rankings.frame
    validate_ranking_artifact(rankings_df, cfg)
    if expected_provenance is not None:
        validate_artifact_provenance(rankings_df, expected_provenance)

    metrics_df = store.load("metrics", cfg)
    validate_metrics_artifact(metrics_df, cfg)
    validate_metrics_ranking_payload(
        metrics_df,
        loaded_rankings.payload_sha256,
    )
    if expected_provenance is not None:
        from paper.benchmark.pipeline.validation import validate_artifact_provenance

        validate_artifact_provenance(
            metrics_df,
            expected_provenance,
            include_ranking_reference=True,
        )
    return False


def _attempt_receipt(
    config: ExperimentConfig,
    *,
    stage: StageType,
    assignment_id: str,
    attempt: int,
    lease_seconds: int,
    request_id: str,
    worker_id: str,
) -> dict[str, Any]:
    """Build the complete immutable receipt written before leasing work."""
    from paper.benchmark.config.constants import PIPELINE_ARTIFACT_VERSION
    from paper.benchmark.utils.env import utc_now_iso

    return {
        "artifact_version": PIPELINE_ARTIFACT_VERSION,
        "assignment_id": assignment_id,
        "attempt": attempt,
        "created_at_utc": utc_now_iso(),
        "dataset": config.dataset,
        "dataset_sha256": config.dataset_identity.sha256,
        "lease_seconds": lease_seconds,
        "method_base": config.method.name,
        "method_id": config.method.label,
        "method_params_json": json.dumps(
            config.method.params_dict,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "n_features": config.dataset_identity.n_features,
        "n_samples": config.dataset_identity.n_samples,
        "seed": config.seed,
        "stage": stage,
        "status": "assigned",
        "task": config.task,
        "request_id": request_id,
        "worker_id": worker_id,
        **_expected_provenance,
    }


@dataclass(frozen=True)
class RecoveredAttempt:
    """One latest durable attempt without terminal evidence."""

    assignment_id: str
    attempt: int
    config: ExperimentConfig
    request_id: str
    worker_id: str


@dataclass(frozen=True)
class DurableAttemptState:
    """Validated attempt counts and restartable in-flight assignments."""

    counts: dict[CellKey, int]
    open_attempts: dict[CellKey, RecoveredAttempt]
    terminal_failures: frozenset[CellKey]


def _load_durable_attempt_state(
    store: S3Store,
    configs: tuple[ExperimentConfig, ...],
    *,
    stage: StageType,
    task: TaskType,
    max_attempts: int,
) -> DurableAttemptState:
    """Validate receipts and reconstruct exact counts plus live retry state."""
    from paper.benchmark.pipeline.validation import (
        validate_attempt_receipt,
        validate_failure_receipt,
    )

    configs_by_key = {config.key: config for config in configs}
    assignment_attempts: dict[str, RecoveredAttempt] = {}
    attempt_numbers: dict[CellKey, set[int]] = {}
    for receipt in store.list_control_receipts("attempts", stage, task):
        key: CellKey = (task, receipt.dataset, receipt.method_label, receipt.seed)
        config = configs_by_key.get(key)
        if config is None:
            raise RuntimeError(f"attempt receipt is outside the active manifest: {receipt.key}")
        attempt = cast(int, receipt.payload.get("attempt"))
        request_id = cast(str, receipt.payload.get("request_id"))
        worker_id = cast(str, receipt.payload.get("worker_id"))
        validate_attempt_receipt(
            receipt.payload,
            config,
            stage=stage,
            assignment_id=receipt.assignment_id,
            attempt=attempt,
            request_id=request_id,
            worker_id=worker_id,
            expected_provenance=_expected_provenance,
        )
        if receipt.assignment_id in assignment_attempts:
            raise RuntimeError(f"duplicate assignment receipt {receipt.assignment_id}")
        assignment_attempts[receipt.assignment_id] = RecoveredAttempt(
            assignment_id=receipt.assignment_id,
            attempt=attempt,
            config=config,
            request_id=request_id,
            worker_id=worker_id,
        )
        numbers = attempt_numbers.setdefault(key, set())
        if attempt in numbers:
            raise RuntimeError(f"duplicate attempt {attempt} for {config}")
        numbers.add(attempt)

    for key, numbers in attempt_numbers.items():
        expected = set(range(1, len(numbers) + 1))
        if numbers != expected:
            raise RuntimeError(
                f"attempt receipts are not contiguous for {configs_by_key[key]}: "
                f"observed={sorted(numbers)}"
            )
        if len(numbers) > max_attempts:
            raise RuntimeError(
                f"attempt receipts exceed the configured bound for {configs_by_key[key]}"
            )

    failed_assignments: set[str] = set()
    terminal_failures: set[CellKey] = set()
    for receipt in store.list_control_receipts("failures", stage, task):
        key = (task, receipt.dataset, receipt.method_label, receipt.seed)
        config = configs_by_key.get(key)
        if config is None:
            raise RuntimeError(f"failure receipt is outside the active manifest: {receipt.key}")
        assignment = assignment_attempts.get(receipt.assignment_id)
        if assignment is None or assignment.config != config:
            raise RuntimeError(f"failure receipt has no matching attempt receipt: {receipt.key}")
        if receipt.assignment_id in failed_assignments:
            raise RuntimeError(f"duplicate failure receipt {receipt.assignment_id}")
        if key in terminal_failures:
            raise RuntimeError(f"multiple durable failures exist for {config}")
        validate_failure_receipt(
            receipt.payload,
            config,
            stage=stage,
            assignment_id=receipt.assignment_id,
            attempt=assignment.attempt,
            request_id=assignment.request_id,
            worker_id=assignment.worker_id,
            expected_provenance=_expected_provenance,
        )
        failed_assignments.add(receipt.assignment_id)
        terminal_failures.add(key)

    for assignment_id in failed_assignments:
        assignment = assignment_attempts[assignment_id]
        if assignment.attempt != max(attempt_numbers[assignment.config.key]):
            raise RuntimeError(
                f"attempt receipts continue after durable failure for {assignment.config}"
            )

    open_attempts: dict[CellKey, RecoveredAttempt] = {}
    for assignment_id, assignment in assignment_attempts.items():
        key = assignment.config.key
        if assignment_id in failed_assignments:
            continue
        if assignment.attempt == max(attempt_numbers[key]):
            open_attempts[key] = assignment

    return DurableAttemptState(
        counts={key: len(numbers) for key, numbers in attempt_numbers.items()},
        open_attempts=open_attempts,
        terminal_failures=frozenset(terminal_failures),
    )


def _build_queues() -> None:
    """Build the manifest-bound queue after validating existing artifacts."""
    global _active_methods, _active_stage, _canonical_manifest_s3_key, _canonical_manifest_sha256, _expected_provenance, _gate_receipt_s3_key, _gate_receipt_sha256, _lease_seconds, _manifest, _manifest_s3_key, _max_cell_attempts, _runtime_contract_s3_key, _store  # noqa: PLW0603
    from loguru import logger

    from paper.benchmark.adapters.store import S3Store as _S3Store

    logger.info("Connecting to S3...")
    store = _S3Store.from_env(validate_uploads=True)
    _store = store
    _active_stage = _configured_stage()
    _lease_seconds = _configured_lease_seconds()
    _max_cell_attempts = _configured_max_cell_attempts()
    from paper.benchmark.pipeline.campaign_gate import (
        configured_campaign_gate_identity,
        load_approved_campaign_gate,
    )

    approved_gate = load_approved_campaign_gate(
        store,
        configured_campaign_gate_identity(),
    )
    _manifest = approved_gate.manifest
    _canonical_manifest_s3_key = approved_gate.identity.canonical_manifest_s3_key
    _canonical_manifest_sha256 = approved_gate.identity.canonical_manifest_sha256
    _gate_receipt_s3_key = approved_gate.identity.gate_receipt_s3_key
    _gate_receipt_sha256 = approved_gate.identity.gate_receipt_sha256
    _manifest_s3_key = approved_gate.identity.manifest_s3_key
    _runtime_contract_s3_key = approved_gate.identity.runtime_contract_s3_key
    dataset_count = _preflight_manifest_datasets(store, _manifest)
    logger.info("Validated {} unique manifest datasets", dataset_count)
    from paper.benchmark.pipeline.validation import validate_expected_provenance
    from paper.benchmark.utils.env import (
        get_benchmark_scope,
        get_container_image,
        get_git_sha,
    )

    configured_scope = get_benchmark_scope()
    if configured_scope["artifact_prefix"] != store.artifact_prefix:
        raise RuntimeError("artifact prefix differs between store and provenance environment")
    if configured_scope["manifest_sha256"] != _manifest.sha256:
        raise RuntimeError("manifest digest differs between manifest and provenance environment")
    if configured_scope["campaign_sha256"] != _manifest.campaign_sha256:
        raise RuntimeError("campaign digest differs between manifest and provenance environment")
    if (
        configured_scope["canonical_manifest_sha256"]
        != approved_gate.identity.canonical_manifest_sha256
    ):
        raise RuntimeError(
            "canonical manifest digest differs between gate and provenance environment"
        )
    if configured_scope["gate_receipt_sha256"] != approved_gate.identity.gate_receipt_sha256:
        raise RuntimeError("gate receipt digest differs between gate and provenance environment")
    if configured_scope["runtime_contract_sha256"] != _manifest.runtime_contract_sha256:
        raise RuntimeError(
            "runtime contract digest differs between manifest and provenance environment"
        )
    if _manifest.account_ids != (configured_scope["aws_account_id"],):
        raise RuntimeError(
            f"manifest is bound to accounts {list(_manifest.account_ids)}, "
            f"not active account {configured_scope['aws_account_id']}"
        )
    _expected_provenance = validate_expected_provenance(
        {
            **configured_scope,
            "container_image": get_container_image(),
            "git_sha": get_git_sha(),
        }
    )
    _active_methods = _manifest.method_names
    _queues.clear()

    if _active_stage == "metrics":
        from paper.benchmark.pipeline.reconcile import reconcile_manifest_artifacts

        ranking_report = reconcile_manifest_artifacts(
            store,
            _manifest,
            _expected_provenance,
            stages=("rankings",),
        )
        if not ranking_report.is_complete:
            counts = ", ".join(
                f"{category}={count}" for category, count in ranking_report.counts.items()
            )
            raise RuntimeError(
                "Cannot launch metrics: exact Stage 1 namespace reconciliation failed "
                f"for account {_expected_provenance['aws_account_id']}, "
                f"prefix {store.artifact_prefix!r}, manifest {_manifest.sha256}: {counts}"
            )
        logger.info(
            "Exact Stage 1 namespace reconciliation passed for {} ranking artifacts",
            len(ranking_report.valid_keys),
        )

    for task in _TASKS:
        configs = _manifest.configs_for(task, _active_stage)
        logger.info(
            "{}: {} approved {} configs from manifest {}",
            task,
            len(configs),
            _active_stage,
            _manifest.sha256,
        )
        completed_rankings = store.list_completed("rankings", task)
        ranking_features: dict[CellKey, int] = {}
        pending: list[ExperimentConfig] = []
        for cfg in configs:
            n_features = _validated_ranking_feature_count(
                store,
                cfg,
                completed_rankings,
                expected_provenance=_expected_provenance,
            )
            if n_features is None:
                if _active_stage == "rankings":
                    pending.append(cfg)
            else:
                ranking_features[cfg.key] = n_features

        if _active_stage == "metrics":
            missing_rankings = [cfg for cfg in configs if cfg.key not in ranking_features]
            if missing_rankings:
                examples = ", ".join(str(cfg) for cfg in missing_rankings[:5])
                raise RuntimeError(
                    f"Cannot launch metrics/{task}: {len(missing_rankings)} approved "
                    f"ranking artifacts are missing; examples: {examples}"
                )
            completed_metrics = store.list_completed("metrics", task)
            pending = [
                cfg
                for cfg in configs
                if _needs_metrics_work(
                    store,
                    cfg,
                    completed_metrics,
                    expected_provenance=_expected_provenance,
                )
            ]

        incomplete = list(pending)
        attempt_state = _load_durable_attempt_state(
            store,
            configs,
            stage=_active_stage,
            task=task,
            max_attempts=_max_cell_attempts,
        )
        exhausted = [
            config
            for config in incomplete
            if config.key in attempt_state.terminal_failures
            or (
                attempt_state.counts.get(config.key, 0) >= _max_cell_attempts
                and config.key not in attempt_state.open_attempts
            )
        ]
        pending = [
            config
            for config in incomplete
            if config.key not in attempt_state.open_attempts
            and config.key not in attempt_state.terminal_failures
            and attempt_state.counts.get(config.key, 0) < _max_cell_attempts
        ]
        random.shuffle(pending)
        queue = QueueState.from_configs(
            pending,
            max_attempts=_max_cell_attempts,
            attempts=attempt_state.counts,
            initial=len(incomplete),
            failed=len(exhausted),
        )
        incomplete_keys = {config.key for config in incomplete}
        for key, recovered in attempt_state.open_attempts.items():
            if key not in incomplete_keys:
                continue
            queue.recover_lease(
                recovered.assignment_id,
                config=recovered.config,
                attempt=recovered.attempt,
                worker_id=recovered.worker_id,
                request_id=recovered.request_id,
                lease_seconds=_lease_seconds,
            )
        _queues[_key(_active_stage, task)] = queue
        logger.info(
            "{}/{}: {} approved, {} valid completed, {} exhausted, {} pending, {} recovered",
            _active_stage,
            task,
            len(configs),
            len(configs) - len(incomplete),
            len(exhausted),
            len(pending),
            queue.in_flight,
        )

    logger.info("All queues built. Server ready.")


# ---------------------------------------------------------------------------
# Queue selection logic
# ---------------------------------------------------------------------------


def _deterministic_assignment_id(
    config: ExperimentConfig,
    *,
    stage: StageType,
    attempt: int,
) -> str:
    """Derive one stable assignment ID from campaign and cell identity."""
    from paper.benchmark.pipeline.validation import derive_assignment_id

    return derive_assignment_id(
        config,
        stage=stage,
        attempt=attempt,
        expected_provenance=_expected_provenance,
    )


async def _persist_attempt_receipt(
    config: ExperimentConfig,
    *,
    stage: StageType,
    assignment_id: str,
    attempt: int,
    receipt: dict[str, Any],
) -> None:
    """Write a receipt or validate the exact object after an ambiguous write."""
    if _store is None:
        raise RuntimeError("Artifact store is not initialized")
    try:
        await run_in_threadpool(
            _store.save_attempt,
            stage,
            config,
            assignment_id,
            receipt,
            expected_provenance=_expected_provenance,
        )
        return
    except Exception as write_error:
        try:
            existing = await run_in_threadpool(
                _store.load_attempt,
                stage,
                config,
                assignment_id,
            )
        except FileNotFoundError:
            raise write_error from None

        from paper.benchmark.pipeline.validation import validate_attempt_receipt

        try:
            validate_attempt_receipt(
                existing,
                config,
                stage=stage,
                assignment_id=assignment_id,
                attempt=attempt,
                request_id=cast(str, receipt.get("request_id")),
                worker_id=cast(str, receipt.get("worker_id")),
                expected_provenance=_expected_provenance,
            )
        except ValueError as exc:
            raise AttemptOwnershipError(
                f"attempt receipt ownership conflict for assignment {assignment_id}"
            ) from exc
        if existing != receipt:
            raise AttemptOwnershipError(
                f"attempt receipt ownership conflict for assignment {assignment_id}"
            ) from write_error


async def _pick_next(
    worker_id: str,
    request_id: str,
) -> tuple[StageType, TaskType, str, int, dict[str, Any]] | None:
    """Create or replay one uncharged assignment offer."""
    from paper.benchmark.config.constants import ASSIGNMENT_OFFER_SECONDS

    async with _queue_lock:
        _requeue_expired_assignments()
        existing_assignments: list[tuple[TaskType, str, AssignmentOffer | AssignmentLease]] = []
        for task in _TASKS:
            queue = _queues.get(_key(_active_stage, task))
            if queue is None:
                continue
            existing = queue.assignment_for_request(worker_id, request_id)
            if existing is not None:
                assignment_id, assignment = existing
                existing_assignments.append((task, assignment_id, assignment))
        if len(existing_assignments) > 1:
            raise RuntimeError(
                "worker request is owned by multiple task queues: "
                f"worker_id={worker_id} request_id={request_id}"
            )
        if existing_assignments:
            task, assignment_id, assignment = existing_assignments[0]
            return (
                _active_stage,
                task,
                assignment_id,
                assignment.attempt,
                _serialize_config(assignment.config),
            )

        tasks = list(_TASKS)
        random.shuffle(tasks)
        for task in tasks:
            queue = _queues.get(_key(_active_stage, task))
            if queue is None:
                continue
            candidate = queue.next_candidate()
            if candidate is None:
                continue
            config, attempt = candidate
            assignment_id = _deterministic_assignment_id(
                config,
                stage=_active_stage,
                attempt=attempt,
            )
            serialized = queue.offer_next(
                assignment_id,
                attempt,
                worker_id,
                request_id,
                ASSIGNMENT_OFFER_SECONDS,
            )
            return _active_stage, task, assignment_id, attempt, serialized
    return None


def _assignment_queue(assignment_id: str) -> QueueState | None:
    """Return the queue that owns an offer, lease, or terminal hold."""
    for queue in _queues.values():
        if queue.owns_assignment(assignment_id):
            return queue
    return None


def _requeue_expired_assignments() -> int:
    """Return all expired offers and leases to their queues."""
    return sum(
        queue.requeue_expired_offers() + queue.requeue_expired() for queue in _queues.values()
    )


def _validate_assignment_artifact(config: ExperimentConfig) -> None:
    """Reload and validate the durable artifact for one completed assignment."""
    if _store is None:
        raise RuntimeError("Artifact store is not initialized")

    ranking_n_features = _validated_ranking_feature_count(
        _store,
        config,
        {config.key},
        expected_provenance=_expected_provenance,
    )
    if ranking_n_features is None:
        raise FileNotFoundError(f"Ranking artifact is missing for {config}")
    if _active_stage == "rankings":
        return

    if _needs_metrics_work(
        _store,
        config,
        {config.key},
        expected_provenance=_expected_provenance,
    ):
        raise FileNotFoundError(f"Metrics artifact is missing for {config}")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    _build_queues()
    yield


# Single uvicorn worker is the correct choice here: the /next handler is a
# sub-microsecond deque pop with no I/O, so one async process handles 10k+
# req/s. Queue state is module-level (in-process), which is fine since there
# is no need for multi-process parallelism at this throughput.
app = FastAPI(title="citrees experiment queue", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/next")
async def next_config(request: AssignmentRequest) -> Response:
    result = await _pick_next(request.worker_id, request.request_id)
    if result is None:
        return Response(status_code=204)
    stage, _task, assignment_id, attempt, config = result
    if _store is None or _manifest is None:
        raise RuntimeError("Queue server is not initialized")
    # config already contains "task" from _serialize_config
    return JSONResponse(
        status_code=200,
        content={
            "stage": stage,
            **_expected_provenance,
            "assignment_id": assignment_id,
            "attempt": attempt,
            "lease_seconds": _lease_seconds,
            "request_id": request.request_id,
            "worker_id": request.worker_id,
            **config,
        },
    )


@app.post("/start")
async def start_assignment(update: AssignmentUpdate) -> dict[str, bool]:
    """Durably record an offered attempt before the worker may execute it."""
    async with _queue_lock:
        _requeue_expired_assignments()
        queue = _assignment_queue(update.assignment_id)
        if queue is None:
            raise HTTPException(status_code=404, detail="assignment offer not found")
        existing = queue.leased_assignment(
            update.assignment_id,
            worker_id=update.worker_id,
            request_id=update.request_id,
        )
        if existing is not None:
            return {"started": True}
        offer = queue.begin_start(
            update.assignment_id,
            worker_id=update.worker_id,
            request_id=update.request_id,
        )
        if offer is None:
            raise HTTPException(status_code=404, detail="assignment offer not found")
        receipt = _attempt_receipt(
            offer.config,
            stage=_active_stage,
            assignment_id=update.assignment_id,
            attempt=offer.attempt,
            lease_seconds=_lease_seconds,
            request_id=offer.request_id,
            worker_id=offer.worker_id,
        )

    try:
        await _persist_attempt_receipt(
            offer.config,
            stage=_active_stage,
            assignment_id=update.assignment_id,
            attempt=offer.attempt,
            receipt=receipt,
        )
    except AttemptOwnershipError as exc:
        async with _queue_lock:
            queue.cancel_offer(update.assignment_id)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except BaseException:
        async with _queue_lock:
            queue.release_start(update.assignment_id)
        raise

    async with _queue_lock:
        queue.activate_offer(update.assignment_id, _lease_seconds)
    return {"started": True}


@app.post("/heartbeat")
async def heartbeat_assignment(update: AssignmentUpdate) -> dict[str, bool]:
    """Extend one active assignment lease."""
    async with _queue_lock:
        _requeue_expired_assignments()
        queue = _assignment_queue(update.assignment_id)
        if queue is None or not queue.heartbeat(
            update.assignment_id,
            update.worker_id,
            update.request_id,
            _lease_seconds,
        ):
            raise HTTPException(status_code=404, detail="assignment lease not found")
    return {"extended": True}


@app.post("/complete")
async def complete_assignment(update: AssignmentUpdate) -> dict[str, bool]:
    """Acknowledge one successfully persisted assignment."""
    async with _queue_lock:
        _requeue_expired_assignments()
        queue = _assignment_queue(update.assignment_id)
        lease = (
            queue.begin_finalization(
                update.assignment_id,
                worker_id=update.worker_id,
                request_id=update.request_id,
            )
            if queue is not None
            else None
        )
        if queue is None or lease is None:
            raise HTTPException(status_code=404, detail="assignment lease not found")
        config = lease.config
    try:
        await run_in_threadpool(_validate_assignment_artifact, config)
    except (FileNotFoundError, ValueError) as exc:
        async with _queue_lock:
            queue.cancel_finalization(update.assignment_id, _lease_seconds)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except BaseException:
        async with _queue_lock:
            queue.cancel_finalization(update.assignment_id, _lease_seconds)
        raise
    async with _queue_lock:
        if not queue.finish_finalization(update.assignment_id, failed=False):
            raise HTTPException(status_code=404, detail="assignment lease not found")
    return {"completed": True}


@app.post("/fail")
async def fail_assignment(update: AssignmentUpdate) -> dict[str, bool]:
    """Acknowledge one assignment with a durable failure receipt."""
    async with _queue_lock:
        _requeue_expired_assignments()
        queue = _assignment_queue(update.assignment_id)
        lease = (
            queue.begin_finalization(
                update.assignment_id,
                worker_id=update.worker_id,
                request_id=update.request_id,
            )
            if queue is not None
            else None
        )
        if queue is None or lease is None or _store is None:
            raise HTTPException(status_code=404, detail="assignment lease not found")
        config = lease.config
        attempt = lease.attempt

    artifact_valid = False
    try:
        await run_in_threadpool(_validate_assignment_artifact, config)
        artifact_valid = True
    except FileNotFoundError:
        pass
    except ValueError as exc:
        async with _queue_lock:
            queue.cancel_finalization(update.assignment_id, _lease_seconds)
        raise HTTPException(status_code=409, detail=f"invalid artifact: {exc}") from exc
    except BaseException:
        async with _queue_lock:
            queue.cancel_finalization(update.assignment_id, _lease_seconds)
        raise

    if not artifact_valid:
        try:
            receipt = await run_in_threadpool(
                _store.load_failure,
                _active_stage,
                config,
                update.assignment_id,
            )
            from paper.benchmark.pipeline.validation import validate_failure_receipt

            validate_failure_receipt(
                receipt,
                config,
                stage=_active_stage,
                assignment_id=update.assignment_id,
                attempt=attempt,
                request_id=update.request_id,
                worker_id=update.worker_id,
                expected_provenance=_expected_provenance,
            )
        except (FileNotFoundError, ValueError) as exc:
            async with _queue_lock:
                queue.cancel_finalization(update.assignment_id, _lease_seconds)
            raise HTTPException(status_code=409, detail=f"invalid failure receipt: {exc}") from exc
        except BaseException:
            async with _queue_lock:
                queue.cancel_finalization(update.assignment_id, _lease_seconds)
            raise

    async with _queue_lock:
        if not queue.finish_finalization(
            update.assignment_id,
            failed=not artifact_valid,
        ):
            raise HTTPException(status_code=404, detail="assignment lease not found")
    return {"failed": not artifact_valid}


@app.get("/status")
async def get_status() -> dict[str, Any]:
    async with _queue_lock:
        expired = _requeue_expired_assignments()
        return {
            "artifact_prefix": _expected_provenance.get("artifact_prefix"),
            "aws_account_id": _expected_provenance.get("aws_account_id"),
            "campaign_sha256": _expected_provenance.get("campaign_sha256"),
            "canonical_manifest_s3_key": _canonical_manifest_s3_key or None,
            "canonical_manifest_sha256": _canonical_manifest_sha256 or None,
            "container_image": _expected_provenance.get("container_image"),
            "gate_receipt_s3_key": _gate_receipt_s3_key or None,
            "gate_receipt_sha256": _gate_receipt_sha256 or None,
            "git_sha": _expected_provenance.get("git_sha"),
            "manifest_s3_key": _manifest_s3_key or None,
            "manifest_sha256": _expected_provenance.get("manifest_sha256"),
            "runtime_contract_sha256": _expected_provenance.get("runtime_contract_sha256"),
            "runtime_contract_s3_key": _runtime_contract_s3_key or None,
            "manifest_cells": len(_manifest.cells) if _manifest is not None else 0,
            "methods": list(_active_methods),
            "stage": _active_stage,
            "lease_seconds": _lease_seconds,
            "max_cell_attempts": _max_cell_attempts,
            "expired_requeued": expired,
            "queues": {
                k: {
                    "available": q.available,
                    "reserved": q.reserved,
                    "in_flight": q.in_flight,
                    "pending": q.pending,
                    "initial": q.initial,
                    "served": q.served,
                    "attempts": q.attempts,
                    "completed": q.completed,
                    "failed": q.failed,
                }
                for k, q in _queues.items()
            },
        }
