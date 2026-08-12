"""Pull-based worker that pops configs from the API server and executes them.

The worker takes no stage or task arguments — it reads the assigned stage
and task from the /next response. The server decides what to hand out.

Run as:
    python -m paper.benchmark.api.worker --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import threading
import time
import traceback
import uuid
from typing import TYPE_CHECKING, Any, cast

import httpx
from loguru import logger

from paper.benchmark.adapters.store import S3Store
from paper.benchmark.config.constants import (
    API_POLL_INTERVAL_SECONDS,
    MIN_ASSIGNMENT_LEASE_SECONDS,
    WORKER_CONTROL_TIMEOUT_SECONDS,
    WORKER_HEARTBEAT_TIMEOUT_SECONDS,
    WORKER_MAX_API_FAILURES,
)
from paper.benchmark.pipeline.types import (
    CellKey,
    DatasetIdentity,
    ExperimentConfig,
    MethodConfig,
    Result,
    StageType,
    TaskType,
)

if TYPE_CHECKING:
    from paper.benchmark.pipeline.campaign_gate import ApprovedCampaignGate

_shutdown = False
_ACK_ATTEMPTS = 5


def _handle_signal(signum: int, frame: Any) -> None:
    global _shutdown
    name = signal.Signals(signum).name
    if _shutdown:
        logger.warning("Received {} again, forcing exit", name)
        sys.exit(1)
    logger.warning("Received {}, finishing current task then exiting...", name)
    _shutdown = True


def _deserialize_config(d: dict[str, Any]) -> ExperimentConfig:
    """Reconstruct ExperimentConfig from JSON dict."""
    method_dict = d["method"]
    params = tuple(sorted(method_dict.get("params", {}).items()))
    method = MethodConfig(name=method_dict["name"], params=params)
    identity_dict = d["dataset_identity"]
    if not isinstance(identity_dict, dict) or set(identity_dict) != {
        "sha256",
        "n_samples",
        "n_features",
    }:
        raise ValueError("dataset_identity must contain sha256, n_samples, and n_features")
    if (
        not isinstance(identity_dict["sha256"], str)
        or type(identity_dict["n_samples"]) is not int
        or type(identity_dict["n_features"]) is not int
    ):
        raise ValueError("dataset_identity fields have invalid types")
    dataset_identity = DatasetIdentity(
        sha256=identity_dict["sha256"],
        n_samples=identity_dict["n_samples"],
        n_features=identity_dict["n_features"],
    )
    return ExperimentConfig(
        method=method,
        dataset=d["dataset"],
        seed=d["seed"],
        task=d["task"],
        dataset_identity=dataset_identity,
    )


def _validate_assignment(
    data: dict[str, Any],
    store: S3Store,
    *,
    approved_configs: dict[CellKey, ExperimentConfig],
    worker_id: str,
    request_id: str,
) -> ExperimentConfig:
    """Reject work issued outside the worker's immutable queue scope."""
    expected_stage = os.environ.get("CITREES_STAGE", "").strip()
    if not expected_stage or data.get("stage") != expected_stage:
        raise RuntimeError(
            f"API/worker stage mismatch: API={data.get('stage')!r}, worker={expected_stage!r}"
        )
    expected = _expected_worker_provenance(store)
    observed = {field: data.get(field) for field in expected}
    if observed != expected:
        raise RuntimeError(f"API/worker provenance mismatch: API={observed!r}, worker={expected!r}")
    attempt = data.get("attempt")
    if type(attempt) is not int or attempt <= 0:
        raise RuntimeError(f"API returned invalid attempt={attempt!r}")
    if data.get("worker_id") != worker_id or data.get("request_id") != request_id:
        raise RuntimeError("API returned an assignment owned by another worker request")
    config = _deserialize_config(data)
    approved_config = approved_configs.get(config.key)
    if approved_config != config:
        raise RuntimeError(f"API returned assignment outside the active manifest: {config}")
    from paper.benchmark.pipeline.validation import derive_assignment_id

    expected_assignment_id = derive_assignment_id(
        config,
        stage=expected_stage,
        attempt=attempt,
        expected_provenance=expected,
    )
    if data.get("assignment_id") != expected_assignment_id:
        raise RuntimeError(
            f"API returned assignment_id={data.get('assignment_id')!r}, "
            f"expected {expected_assignment_id!r}"
        )
    return config


def _approved_configs_for_stage(
    approved_gate: ApprovedCampaignGate,
    stage: StageType,
) -> dict[CellKey, ExperimentConfig]:
    """Build a collision-free approval map across both task queues."""
    configs = [
        config
        for task in ("classification", "regression")
        for config in approved_gate.manifest.configs_for(
            cast(TaskType, task),
            stage,
        )
    ]
    approved_configs = {config.key: config for config in configs}
    if len(approved_configs) != len(configs):
        raise RuntimeError("active manifest contains colliding experiment cell keys")
    return approved_configs


def _expected_worker_provenance(store: S3Store) -> dict[str, str]:
    """Return and validate the provenance of the running worker."""
    from paper.benchmark.pipeline.validation import validate_expected_provenance
    from paper.benchmark.utils.env import (
        get_benchmark_scope,
        get_container_image,
        get_git_sha,
    )

    expected = validate_expected_provenance(
        {
            **get_benchmark_scope(),
            "container_image": get_container_image(),
            "git_sha": get_git_sha(),
        }
    )
    if expected["artifact_prefix"] != store.artifact_prefix:
        raise RuntimeError("worker store artifact prefix does not match its provenance environment")
    return expected


def _load_gate_approved_campaign(store: S3Store) -> ApprovedCampaignGate:
    """Load the complete gate-approved campaign before leasing work."""
    from paper.benchmark.experiments.r_cforest_reproducibility import (
        require_running_runtime_contract,
    )
    from paper.benchmark.pipeline.campaign_gate import (
        configured_campaign_gate_identity,
        load_approved_campaign_gate,
    )

    approved = load_approved_campaign_gate(
        store,
        configured_campaign_gate_identity(),
    )
    require_running_runtime_contract(approved.runtime_contract)
    return approved


def _validate_api_scope(data: dict[str, Any], store: S3Store) -> None:
    """Reject an API server outside the worker's immutable queue scope."""
    expected = {
        **_expected_worker_provenance(store),
        "canonical_manifest_s3_key": os.environ.get(
            "CITREES_CANONICAL_MANIFEST_S3_KEY", ""
        ).strip(),
        "canonical_manifest_sha256": os.environ.get(
            "CITREES_CANONICAL_MANIFEST_SHA256", ""
        ).strip(),
        "gate_receipt_s3_key": os.environ.get("CITREES_GATE_RECEIPT_S3_KEY", "").strip(),
        "gate_receipt_sha256": os.environ.get("CITREES_GATE_RECEIPT_SHA256", "").strip(),
        "manifest_s3_key": os.environ.get("CITREES_MANIFEST_S3_KEY", "").strip(),
        "runtime_contract_s3_key": os.environ.get("CITREES_RUNTIME_CONTRACT_S3_KEY", "").strip(),
        "stage": os.environ.get("CITREES_STAGE", "").strip(),
    }
    observed = {field: data.get(field) for field in expected}
    if not expected["stage"] or observed != expected:
        raise RuntimeError(
            f"API/worker queue scope mismatch: API={observed!r}, worker={expected!r}"
        )


def _heartbeat_loop(
    api_url: str,
    assignment_id: str,
    worker_id: str,
    request_id: str,
    lease_seconds: int,
    stop_event: threading.Event,
    lease_lost_event: threading.Event,
) -> None:
    """Extend an assignment lease until the running cell terminates."""
    interval = max(1.0, lease_seconds / 3)
    retry_interval = max(1.0, min(30.0, interval / 3))
    next_interval = interval
    deadline = time.monotonic() + lease_seconds
    try:
        with httpx.Client(
            base_url=api_url,
            timeout=WORKER_HEARTBEAT_TIMEOUT_SECONDS,
        ) as client:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    lease_lost_event.set()
                    logger.error("Assignment lease {} expired locally", assignment_id)
                    return
                if stop_event.wait(min(next_interval, remaining)):
                    return
                if time.monotonic() >= deadline:
                    lease_lost_event.set()
                    logger.error("Assignment lease {} expired locally", assignment_id)
                    return
                try:
                    response = client.post(
                        "/heartbeat",
                        json={
                            "assignment_id": assignment_id,
                            "request_id": request_id,
                            "worker_id": worker_id,
                        },
                    )
                    if response.status_code == 404:
                        logger.error(
                            "Assignment lease {} was lost",
                            assignment_id,
                        )
                        lease_lost_event.set()
                        return
                    response.raise_for_status()
                    deadline = time.monotonic() + lease_seconds
                    next_interval = interval
                except httpx.HTTPError as exc:
                    if time.monotonic() >= deadline:
                        lease_lost_event.set()
                        logger.error(
                            "Assignment lease {} expired during heartbeat failure",
                            assignment_id,
                        )
                        return
                    next_interval = retry_interval
                    logger.warning(
                        "Heartbeat failed for assignment {}: {}",
                        assignment_id,
                        exc,
                    )
    except Exception:
        lease_lost_event.set()
        logger.exception(
            "Heartbeat loop crashed for assignment {}",
            assignment_id,
        )


def _record_api_failure(
    exc: httpx.HTTPError,
    *,
    operation: str,
    failure_count: int,
    max_api_failures: int,
    poll_interval: float,
) -> int:
    """Sleep after a control-plane failure or raise at the fixed bound."""
    updated_count = failure_count + 1
    logger.warning(
        "{} failed ({}/{}): {}",
        operation,
        updated_count,
        max_api_failures,
        exc,
    )
    if updated_count >= max_api_failures:
        raise RuntimeError(f"{operation} failed {updated_count} consecutive times") from exc
    time.sleep(poll_interval)
    return updated_count


def _get_api_scope(client: httpx.Client, store: S3Store) -> None:
    """Load and validate the server scope before leasing any assignment."""
    response = client.get("/status")
    response.raise_for_status()
    _validate_api_scope(response.json(), store)


def _next_assignment(
    client: httpx.Client,
    worker_id: str,
    request_id: str,
) -> httpx.Response:
    """Request one idempotent assignment offer."""
    response = client.post(
        "/next",
        json={"request_id": request_id, "worker_id": worker_id},
    )
    if response.status_code != 204:
        response.raise_for_status()
    return response


def _start_assignment(
    client: httpx.Client,
    assignment_id: str,
    worker_id: str,
    request_id: str,
) -> None:
    """Confirm an offer and wait for its durable attempt receipt."""
    response = client.post(
        "/start",
        json={
            "assignment_id": assignment_id,
            "request_id": request_id,
            "worker_id": worker_id,
        },
    )
    response.raise_for_status()


def _acknowledge_assignment(
    client: httpx.Client,
    assignment_id: str,
    worker_id: str,
    request_id: str,
    *,
    failed: bool,
    poll_interval: float,
) -> None:
    """Acknowledge an assignment after its artifact or receipt is durable."""
    endpoint = "/fail" if failed else "/complete"
    for attempt in range(1, _ACK_ATTEMPTS + 1):
        try:
            response = client.post(
                endpoint,
                json={
                    "assignment_id": assignment_id,
                    "request_id": request_id,
                    "worker_id": worker_id,
                },
            )
            if response.status_code == 404:
                logger.warning("Assignment lease {} no longer exists", assignment_id)
                return
            response.raise_for_status()
            return
        except httpx.HTTPError:
            if attempt == _ACK_ATTEMPTS:
                raise
            time.sleep(poll_interval)


def _save_failure_receipt(
    store: S3Store,
    stage: StageType,
    assignment_id: str,
    attempt: int,
    worker_id: str,
    request_id: str,
    result: Any,
) -> str:
    """Persist the complete provenance and traceback for a failed cell."""
    import json

    from paper.benchmark.config.constants import PIPELINE_ARTIFACT_VERSION
    from paper.benchmark.utils.env import (
        get_benchmark_scope,
        get_container_image,
        get_git_sha,
        get_hardware_metadata,
        get_library_versions,
        utc_now_iso,
    )

    config = result.config
    benchmark_scope = get_benchmark_scope()
    container_image = get_container_image()
    git_sha = get_git_sha()
    receipt = {
        "artifact_version": PIPELINE_ARTIFACT_VERSION,
        "assignment_id": assignment_id,
        "attempt": attempt,
        "container_image": container_image,
        "created_at_utc": utc_now_iso(),
        "dataset": config.dataset,
        "dataset_sha256": config.dataset_identity.sha256,
        "elapsed_seconds": result.elapsed_seconds,
        "error": result.error,
        "error_type": result.error_type,
        "git_sha": git_sha,
        "hardware": get_hardware_metadata(),
        "hostname": result.hostname,
        "library_versions": get_library_versions(),
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
        "status": "failed",
        "task": config.task,
        "traceback": result.traceback,
        "request_id": request_id,
        "worker_id": worker_id,
        **benchmark_scope,
    }
    expected_provenance = {
        **benchmark_scope,
        "container_image": container_image,
        "git_sha": git_sha,
    }
    try:
        return store.save_failure(
            stage,
            config,
            assignment_id,
            receipt,
            expected_provenance=expected_provenance,
        )
    except Exception as write_error:
        try:
            existing = store.load_failure(stage, config, assignment_id)
        except FileNotFoundError:
            raise write_error from None

        from paper.benchmark.pipeline.validation import validate_failure_receipt

        validate_failure_receipt(
            existing,
            config,
            stage=stage,
            assignment_id=assignment_id,
            attempt=attempt,
            request_id=request_id,
            worker_id=worker_id,
            expected_provenance=expected_provenance,
        )
        if existing != receipt:
            raise RuntimeError(
                f"failure receipt ownership conflict for assignment {assignment_id}"
            ) from write_error
        return f"durable failure receipt for {assignment_id}"


def _failed_result(
    config: ExperimentConfig,
    *,
    error: str,
    error_type: str,
    traceback_text: str | None = None,
    elapsed_seconds: float = 0.0,
) -> Result:
    """Create a terminal failed result for worker-level execution errors."""
    return Result(
        config=config,
        status="failed",
        elapsed_seconds=elapsed_seconds,
        error=error,
        error_type=error_type,
        traceback=traceback_text,
        hostname=socket.gethostname(),
    )


def _finalize_assignment(
    client: httpx.Client,
    store: S3Store,
    stage: StageType,
    assignment_id: str,
    attempt: int,
    worker_id: str,
    request_id: str,
    result: Result,
    *,
    poll_interval: float,
) -> None:
    """Persist terminal evidence and acknowledge without terminating the worker."""
    if result.is_no_rankings:
        result = _failed_result(
            result.config,
            error="ranking artifact was unavailable for a metrics assignment",
            error_type="MissingRankingArtifact",
            elapsed_seconds=result.elapsed_seconds,
        )

    if result.is_failure:
        try:
            receipt_path = _save_failure_receipt(
                store,
                stage,
                assignment_id,
                attempt,
                worker_id,
                request_id,
                result,
            )
        except Exception:
            logger.exception(
                "Could not persist failure receipt for assignment {}; "
                "leaving the lease unacknowledged",
                assignment_id,
            )
            return

        logger.error(
            "Failed: {} - {}: {} (receipt={})",
            result.config,
            result.error_type,
            result.error,
            receipt_path,
        )
        try:
            _acknowledge_assignment(
                client,
                assignment_id,
                worker_id,
                request_id,
                failed=True,
                poll_interval=poll_interval,
            )
        except Exception:
            logger.exception(
                "Could not acknowledge failed assignment {}; the durable receipt remains available",
                assignment_id,
            )
        return

    if result.status not in {"done", "skipped"}:
        _finalize_assignment(
            client,
            store,
            stage,
            assignment_id,
            attempt,
            worker_id,
            request_id,
            _failed_result(
                result.config,
                error=f"assignment returned non-terminal status {result.status!r}",
                error_type="InvalidResultStatus",
                elapsed_seconds=result.elapsed_seconds,
            ),
            poll_interval=poll_interval,
        )
        return

    try:
        _acknowledge_assignment(
            client,
            assignment_id,
            worker_id,
            request_id,
            failed=False,
            poll_interval=poll_interval,
        )
    except Exception:
        logger.exception(
            "Could not acknowledge completed assignment {}; "
            "the immutable artifact remains available",
            assignment_id,
        )
        return
    logger.info("Done: {} (status={})", result.config, result.status)


def _execute_assignment(
    client: httpx.Client,
    store: S3Store,
    api_url: str,
    stage: StageType,
    assignment_id: str,
    attempt: int,
    worker_id: str,
    request_id: str,
    lease_seconds: int,
    config: ExperimentConfig,
    run_fn: Any,
    *,
    poll_interval: float,
) -> None:
    """Execute and finalize one valid assignment while keeping the fleet alive."""
    heartbeat_stop = threading.Event()
    lease_lost = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(
            api_url,
            assignment_id,
            worker_id,
            request_id,
            lease_seconds,
            heartbeat_stop,
            lease_lost,
        ),
        daemon=True,
    )
    store.bind_write_guard(lambda: not lease_lost.is_set())
    heartbeat_thread.start()
    try:
        execution_started = time.perf_counter()
        try:
            result = run_fn(config, store)
        except Exception as exc:
            result = _failed_result(
                config,
                error=str(exc),
                error_type=type(exc).__name__,
                traceback_text=traceback.format_exc(),
                elapsed_seconds=time.perf_counter() - execution_started,
            )
        if lease_lost.is_set():
            logger.error(
                "Not acknowledging assignment {} because its lease was lost",
                assignment_id,
            )
            return
        _finalize_assignment(
            client,
            store,
            stage,
            assignment_id,
            attempt,
            worker_id,
            request_id,
            result,
            poll_interval=poll_interval,
        )
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=WORKER_HEARTBEAT_TIMEOUT_SECONDS + 5.0)
        if heartbeat_thread.is_alive():
            logger.error(
                "Heartbeat thread did not stop for assignment {}",
                assignment_id,
            )
        store.clear_write_guard()


def _queue_has_outstanding_work(client: httpx.Client, store: S3Store) -> bool:
    """Return whether any queue still has available or in-flight cells."""
    response = client.get("/status")
    response.raise_for_status()
    data = response.json()
    _validate_api_scope(data, store)
    queues = data.get("queues", {})
    return any(int(queue.get("pending", 0)) > 0 for queue in queues.values())


def run_worker(
    api_url: str,
    poll_interval: float = API_POLL_INTERVAL_SECONDS,
    max_api_failures: int = WORKER_MAX_API_FAILURES,
) -> None:
    """Main worker loop: pop config -> execute -> repeat.

    Parameters
    ----------
    api_url : str
        URL of the API server.
    poll_interval : float
        Seconds between polls when queue is empty.
    max_api_failures : int
        Exit after this many consecutive control-plane failures.
    """
    from paper.benchmark.pipeline.stage1 import _run_selection
    from paper.benchmark.pipeline.stage2 import _run_evaluation

    store = S3Store.from_env(validate_uploads=True)
    approved_gate = _load_gate_approved_campaign(store)
    configured_stage = os.environ.get("CITREES_STAGE", "").strip()
    if configured_stage not in {"rankings", "metrics"}:
        raise RuntimeError("CITREES_STAGE must be rankings or metrics")
    approved_configs = _approved_configs_for_stage(
        approved_gate,
        cast(StageType, configured_stage),
    )
    if poll_interval < 0:
        raise ValueError("poll_interval must be non-negative")
    if max_api_failures <= 0:
        raise ValueError("max_api_failures must be positive")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info(
        "Worker started: api_url={}, max_api_failures={}",
        api_url,
        max_api_failures,
    )

    worker_id = uuid.uuid4().hex
    request_id = uuid.uuid4().hex
    consecutive_api_failures = 0
    with httpx.Client(base_url=api_url, timeout=WORKER_CONTROL_TIMEOUT_SECONDS) as client:
        while not _shutdown:
            try:
                _get_api_scope(client, store)
            except httpx.HTTPError as exc:
                consecutive_api_failures = _record_api_failure(
                    exc,
                    operation="API scope check",
                    failure_count=consecutive_api_failures,
                    max_api_failures=max_api_failures,
                    poll_interval=poll_interval,
                )
                continue
            consecutive_api_failures = 0
            break

        while not _shutdown:
            try:
                resp = _next_assignment(client, worker_id, request_id)
            except httpx.HTTPError as exc:
                consecutive_api_failures = _record_api_failure(
                    exc,
                    operation="assignment poll",
                    failure_count=consecutive_api_failures,
                    max_api_failures=max_api_failures,
                    poll_interval=poll_interval,
                )
                continue
            consecutive_api_failures = 0

            if resp.status_code == 204:
                request_id = uuid.uuid4().hex
                try:
                    outstanding = _queue_has_outstanding_work(client, store)
                except httpx.HTTPError as exc:
                    consecutive_api_failures = _record_api_failure(
                        exc,
                        operation="queue status check",
                        failure_count=consecutive_api_failures,
                        max_api_failures=max_api_failures,
                        poll_interval=poll_interval,
                    )
                    continue
                consecutive_api_failures = 0
                if not outstanding:
                    logger.info("Queue complete, exiting")
                    break
                logger.debug("No assignment available; in-flight work remains")
                time.sleep(poll_interval)
                continue

            data = resp.json()
            cfg = _validate_assignment(
                data,
                store,
                approved_configs=approved_configs,
                worker_id=worker_id,
                request_id=request_id,
            )
            consecutive_api_failures = 0
            stage = data["stage"]
            if stage not in {"rankings", "metrics"}:
                raise RuntimeError(f"API returned invalid stage={stage!r}")
            stage = cast(StageType, stage)
            assignment_id = data["assignment_id"]
            attempt = data["attempt"]
            lease_seconds = int(data["lease_seconds"])
            if lease_seconds < MIN_ASSIGNMENT_LEASE_SECONDS:
                raise RuntimeError(
                    "API returned lease_seconds="
                    f"{lease_seconds}, minimum is {MIN_ASSIGNMENT_LEASE_SECONDS}"
                )
            while True:
                try:
                    _start_assignment(
                        client,
                        assignment_id,
                        worker_id,
                        request_id,
                    )
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code in {404, 409}:
                        raise RuntimeError("assignment offer could not be durably started") from exc
                    consecutive_api_failures = _record_api_failure(
                        exc,
                        operation="assignment start",
                        failure_count=consecutive_api_failures,
                        max_api_failures=max_api_failures,
                        poll_interval=poll_interval,
                    )
                    continue
                except httpx.HTTPError as exc:
                    consecutive_api_failures = _record_api_failure(
                        exc,
                        operation="assignment start",
                        failure_count=consecutive_api_failures,
                        max_api_failures=max_api_failures,
                        poll_interval=poll_interval,
                    )
                    continue
                consecutive_api_failures = 0
                break

            assignment_request_id = request_id
            request_id = uuid.uuid4().hex
            logger.info(
                "Running: assignment={} stage={} {}",
                assignment_id,
                stage,
                cfg,
            )

            run_fn = _run_selection if stage == "rankings" else _run_evaluation
            _execute_assignment(
                client,
                store,
                api_url,
                stage,
                assignment_id,
                attempt,
                worker_id,
                assignment_request_id,
                lease_seconds,
                cfg,
                run_fn,
                poll_interval=poll_interval,
            )

    logger.info("Worker shutting down")


def main() -> None:
    parser = argparse.ArgumentParser(description="citrees experiment worker")
    parser.add_argument("--api-url", required=True, help="API server URL")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=API_POLL_INTERVAL_SECONDS,
        help="Seconds between queue polls",
    )
    parser.add_argument(
        "--max-api-failures",
        type=int,
        default=WORKER_MAX_API_FAILURES,
        help="Exit after N consecutive control-plane failures",
    )
    args = parser.parse_args()
    run_worker(
        args.api_url,
        poll_interval=args.poll_interval,
        max_api_failures=args.max_api_failures,
    )


if __name__ == "__main__":
    main()
