"""Run and compare the fixed-family r_cforest reproducibility gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import timedelta
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_info

from paper.benchmark.adapters.data import load_dataset
from paper.benchmark.pipeline.instance_identity import (
    SUPPORTED_REGION,
    InstanceIdentityEvidence,
    OperatorInstanceReadback,
    collect_aws_caller_identity,
    collect_instance_identity,
    collect_operator_readback,
    get_assumed_role_identity,
    validate_instance_identity_record,
    validate_operator_readback,
)
from paper.benchmark.pipeline.manifest import (
    ManifestCell,
    RerunManifest,
    account_manifest_sha256_map,
    parse_rerun_manifest,
    validate_canonical_campaign,
)
from paper.benchmark.pipeline.methods import get_full_method_configs
from paper.benchmark.pipeline.operator_attestation import (
    create_operator_attestation,
    generate_operator_keypair,
    load_operator_private_key,
    load_operator_public_key,
    parse_utc_timestamp,
    utc_timestamp,
    validate_operator_attestation,
    validate_operator_public_key,
)
from paper.benchmark.pipeline.r_methods import get_r_runtime_versions
from paper.benchmark.pipeline.runtime_contract import (
    EXPECTED_THREAD_ENVIRONMENT,
    PYTHON_LIBRARY_NAMES,
    R_RUNTIME_FIELDS,
    RUNTIME_CONTRACT_PROFILE,
    RUNTIME_CONTRACT_SCHEMA_VERSION,
    RUNTIME_PROVENANCE_FIELDS,
    THREAD_ENVIRONMENT,
    parse_runtime_contract,
    runtime_contract_sha256,
    validate_cpu_affinity,
    validate_openssl_version,
    validate_runtime_contract,
)
from paper.benchmark.pipeline.stage1 import run_r_selection_parallel
from paper.benchmark.pipeline.stage2 import get_requested_evaluation_k_values
from paper.benchmark.pipeline.types import TaskType
from paper.benchmark.utils.env import get_available_cpu_ids, partition_cpu_ids

SCHEMA_VERSION = 7
GATE_RECEIPT_SCHEMA_VERSION = 8
GATE_RECEIPT_PROFILE = "r_cforest_reproducibility_gate"
GATE_RECEIPT_S3_PREFIX = "runtime-gate-receipts"
GATE_MARKET = "on-demand"
N_FOLDS = 5
N_EXPECTED_RUNS = 4
N_EXPECTED_HOSTS = 2
N_EXPECTED_GATE_ACCOUNTS = 1
EXPECTED_REPLACEMENT_CELLS = 940
EXPECTED_DATASET_TASK_PAIRS = 47
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_GATE_CELLS = 8
EXPECTED_GATE_DATASET_TASK_PAIRS = 4
GATE_HOST_SLOTS = ("arc-a", "arc-b")
GATE_REPEATS = (1, 2)
GATE_PANEL_SPECIFICATION = (
    ("compact", "Bonferroni", False, EXPECTED_SEEDS[0]),
    ("compact", "MonteCarlo", True, EXPECTED_SEEDS[-1]),
    ("high_dimensional", "Bonferroni", True, EXPECTED_SEEDS[-1]),
    ("high_dimensional", "MonteCarlo", False, EXPECTED_SEEDS[0]),
)
MAX_OPERATOR_READBACK_AGE_SECONDS = 300
MAX_OPERATOR_READBACK_CLOCK_SKEW_SECONDS = 30
OPENSSL_VERSION_TIMEOUT_SECONDS = 10
REPLACEMENT_REASON = "confirmed_r_adapter_mapping_and_seed_defect"
REQUIRED_GATE_ENVIRONMENT = (
    "CITREES_IMAGE_URI",
    "GIT_SHA",
)
STATIC_PROVENANCE_FIELDS = (
    "ami_id",
    "architecture",
    "cpu_affinity",
    "cpu_model",
    "git_sha",
    "kernel",
    "logical_cpus",
    "machine",
    "microcode",
    "openssl_version",
    "os_release",
    "python_libraries",
    "r_numerical_libraries",
    "r_runtime",
    "script_sha256",
    "thread_environment",
    "threadpools",
)
PROVENANCE_FIELDS = {
    "ami_id",
    "architecture",
    "availability_zone",
    "availability_zone_id",
    "aws_account_id",
    "boot_id",
    "container_image",
    "cpu_affinity",
    "cpu_model",
    "git_sha",
    "hostname",
    "instance_identity",
    "instance_id",
    "instance_type",
    "kernel",
    "logical_cpus",
    "machine",
    "microcode",
    "openssl_version",
    "os_release",
    "process_id",
    "process_start_ticks",
    "python_libraries",
    "r_numerical_libraries",
    "r_runtime",
    "run_id",
    "script_sha256",
    "thread_environment",
    "threadpools",
}
PAYLOAD_FIELDS = {
    "campaign_sha256",
    "elapsed_seconds",
    "manifest_sha256",
    "profile",
    "provenance",
    "results",
    "runtime_contract_sha256",
    "schema_version",
    "target_aws_account_ids",
}
DATASET_RESULT_FIELDS = {
    "configurations",
    "dataset",
    "dataset_source",
    "identity",
    "k_values",
    "task",
}
CONFIGURATION_RESULT_FIELDS = {
    "elapsed_seconds",
    "fold_cpu_affinity",
    "method",
    "params",
    "rankings",
    "ranking_sha256",
    "seed",
    "selected_set_sha256",
}
GATE_RECEIPT_FIELDS = {
    "account_manifest_sha256",
    "campaign_sha256",
    "created_at_utc",
    "manifest_sha256",
    "operator_readbacks",
    "profile",
    "report",
    "run_payloads",
    "runtime_contract_sha256",
    "schema_version",
}
_BOOT_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_GIT_SHA_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_IMAGE_DIGEST_PATTERN = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_GATE_IMAGE_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_GATE_LAUNCH_NONCE_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_GATE_SOURCE_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _run_payload_sha256s(payloads: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return canonical content digests for the exact gate run payloads."""
    digests = sorted(_sha256_bytes(_canonical_json_bytes(payload)) for payload in payloads)
    if len(digests) != N_EXPECTED_RUNS or len(set(digests)) != N_EXPECTED_RUNS:
        raise ValueError("gate run payloads must have four unique canonical digests")
    return digests


def _container_image_digest(image_uri: str) -> str:
    """Return the registry-independent digest of one immutable image URI."""
    if not _IMAGE_DIGEST_PATTERN.fullmatch(image_uri):
        raise ValueError("container image is not an immutable digest URI")
    return image_uri.rsplit("@", maxsplit=1)[1]


def gate_launch_identity(
    source_git_sha: str,
    image_digest: str,
    launch_nonce: str,
) -> str:
    """Return the exact identity of one fresh Arc on-demand gate attempt."""
    if not _GATE_SOURCE_GIT_SHA_PATTERN.fullmatch(source_git_sha):
        raise ValueError("gate source Git SHA must contain 40 lowercase hex digits")
    if not _GATE_IMAGE_DIGEST_PATTERN.fullmatch(image_digest):
        raise ValueError("gate image digest must be sha256 followed by 64 lowercase hex digits")
    if not _GATE_LAUNCH_NONCE_PATTERN.fullmatch(launch_nonce):
        raise ValueError("gate launch nonce must contain 32 lowercase hex digits")
    return _sha256_bytes(
        f"{source_git_sha}\0{image_digest}\0{launch_nonce}\0{GATE_MARKET}".encode("ascii")
    )


def gate_output_prefix(
    source_git_sha: str,
    image_digest: str,
    launch_nonce: str,
) -> str:
    """Return the immutable S3 prefix for one Arc on-demand gate attempt."""
    gate_launch_identity(source_git_sha, image_digest, launch_nonce)
    image_binding = image_digest.removeprefix("sha256:")[:16]
    return (
        "gates/r-cforest-reproducibility/"
        f"source-{source_git_sha}/image-{image_binding}/"
        f"attempt-{launch_nonce}-arc-on-demand2"
    )


def _integer_digest(values: Sequence[int]) -> str:
    canonical = np.asarray(values, dtype="<i8")
    return _sha256_bytes(canonical.tobytes())


def _linux_process_identity(
    *,
    boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id"),
    stat_path: Path = Path("/proc/self/stat"),
) -> tuple[str, int]:
    """Return the Linux boot ID and this process's kernel start ticks."""
    if platform.system() != "Linux" and (
        boot_id_path == Path("/proc/sys/kernel/random/boot_id")
        or stat_path == Path("/proc/self/stat")
    ):
        raise RuntimeError("the r_cforest reproducibility gate requires Linux")
    try:
        boot_id = boot_id_path.read_text(encoding="ascii").strip()
        stat = stat_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"cannot read Linux process identity: {exc}") from exc
    if not _BOOT_ID_PATTERN.fullmatch(boot_id):
        raise RuntimeError(f"invalid Linux boot ID {boot_id!r}")

    command_end = stat.rfind(") ")
    if command_end < 0:
        raise RuntimeError("invalid /proc/self/stat process record")
    fields_from_state = stat[command_end + 2 :].split()
    start_time_index = 22 - 3
    if len(fields_from_state) <= start_time_index:
        raise RuntimeError("incomplete /proc/self/stat process record")
    try:
        start_ticks = int(fields_from_state[start_time_index])
    except ValueError as exc:
        raise RuntimeError("invalid /proc/self/stat process start ticks") from exc
    if start_ticks <= 0:
        raise RuntimeError("process start ticks must be positive")
    return boot_id, start_ticks


def _cpu_field(name: str) -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        raise RuntimeError("/proc/cpuinfo is required by the reproducibility gate")
    values = {
        line.partition(":")[2].strip()
        for line in cpuinfo.read_text(encoding="ascii").splitlines()
        if line.partition(":")[0].strip() == name
    }
    if len(values) != 1:
        raise RuntimeError(f"expected one /proc/cpuinfo {name!r} value, observed {sorted(values)}")
    return next(iter(values))


def _os_release() -> dict[str, str]:
    try:
        values = platform.freedesktop_os_release()
    except OSError:
        return {"ID": "unknown", "VERSION_ID": "unknown"}
    return {
        "ID": values.get("ID", "unknown"),
        "VERSION_ID": values.get("VERSION_ID", "unknown"),
    }


def _python_libraries() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in PYTHON_LIBRARY_NAMES}


def _r_numerical_libraries() -> dict[str, str]:
    script = (
        "info <- sessionInfo(); "
        "cat(as.character(info$BLAS), '\\n', as.character(info$LAPACK), '\\n', sep='')"
    )
    lines = subprocess.check_output(
        ["Rscript", "--vanilla", "-e", script],
        text=True,
        stderr=subprocess.STDOUT,
    ).splitlines()
    if len(lines) != 2 or not all(lines):
        raise RuntimeError(f"unexpected R numerical-library output: {lines!r}")
    return {"blas": lines[0], "lapack": lines[1]}


def _openssl_version() -> str:
    """Return canonical output from the pinned OpenSSL CLI."""
    try:
        output = subprocess.check_output(
            ["openssl", "version"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=OPENSSL_VERSION_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("cannot execute openssl version") from exc
    lines = output.splitlines()
    if len(lines) != 1:
        raise RuntimeError(f"invalid OpenSSL CLI version output: {lines!r}")
    try:
        return validate_openssl_version(lines[0], source="OpenSSL CLI version")
    except ValueError as exc:
        raise RuntimeError(f"invalid OpenSSL CLI version output: {lines!r}") from exc


def _canonical_threadpools() -> list[dict[str, Any]]:
    pools = threadpool_info()
    if not pools:
        raise RuntimeError("threadpoolctl did not discover any numerical-library thread pools")
    normalized = sorted(
        pools,
        key=lambda item: (
            str(item.get("user_api", "")),
            str(item.get("internal_api", "")),
            str(item.get("prefix", "")),
            str(item.get("filepath", "")),
        ),
    )
    invalid = [
        {
            "internal_api": pool.get("internal_api"),
            "num_threads": pool.get("num_threads"),
            "filepath": pool.get("filepath"),
        }
        for pool in normalized
        if pool.get("num_threads") != 1
    ]
    if invalid:
        raise RuntimeError(f"numerical thread pools are not single-threaded: {invalid}")
    return normalized


def _thread_environment() -> dict[str, str]:
    values = {name: os.environ.get(name, "").strip() for name in THREAD_ENVIRONMENT}
    invalid = {
        name: value for name, value in values.items() if value != EXPECTED_THREAD_ENVIRONMENT[name]
    }
    if invalid:
        raise RuntimeError(f"thread environment differs from the frozen values: {invalid}")
    return values


def _required_environment() -> dict[str, str]:
    values = {name: os.environ.get(name, "").strip() for name in REQUIRED_GATE_ENVIRONMENT}
    missing = sorted(name for name, value in values.items() if not value)
    if missing:
        raise RuntimeError(f"missing required gate environment: {missing}")
    return values


def _provenance(
    run_id: str,
    *,
    instance_identity: InstanceIdentityEvidence | None = None,
) -> dict[str, Any]:
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    environment = _required_environment()
    evidence = collect_instance_identity() if instance_identity is None else instance_identity
    if not isinstance(evidence, InstanceIdentityEvidence):
        raise TypeError("instance_identity must be an InstanceIdentityEvidence")
    signed = evidence.identity
    boot_id, process_start_ticks = _linux_process_identity()
    script_path = Path(__file__).resolve()
    cpu_affinity = get_available_cpu_ids()
    return {
        "run_id": run_id,
        "hostname": socket.gethostname(),
        "process_id": os.getpid(),
        "boot_id": boot_id,
        "process_start_ticks": process_start_ticks,
        "instance_identity": evidence.to_record(),
        "instance_id": signed.instance_id,
        "availability_zone": signed.availability_zone,
        "availability_zone_id": evidence.availability_zone_id,
        "aws_account_id": signed.account_id,
        "instance_type": signed.instance_type,
        "ami_id": signed.image_id,
        "architecture": signed.architecture,
        "container_image": environment["CITREES_IMAGE_URI"],
        "git_sha": environment["GIT_SHA"],
        "machine": platform.machine(),
        "cpu_model": _cpu_field("model name"),
        "microcode": _cpu_field("microcode"),
        "logical_cpus": len(cpu_affinity),
        "cpu_affinity": list(cpu_affinity),
        "kernel": platform.release(),
        "openssl_version": _openssl_version(),
        "os_release": _os_release(),
        "python_libraries": _python_libraries(),
        "r_runtime": get_r_runtime_versions(),
        "r_numerical_libraries": _r_numerical_libraries(),
        "thread_environment": _thread_environment(),
        "threadpools": _canonical_threadpools(),
        "script_sha256": _sha256_bytes(script_path.read_bytes()),
    }


def summarize_rankings(
    rankings: Sequence[Sequence[int]],
    *,
    n_features: int,
    k_values: Sequence[int],
) -> dict[str, Any]:
    """Validate and summarize one configuration's complete fold rankings."""
    if len(rankings) != N_FOLDS:
        raise ValueError(f"expected {N_FOLDS} fold rankings, observed {len(rankings)}")
    expected = np.arange(n_features)
    normalized: list[list[int]] = []
    for fold_idx, ranking in enumerate(rankings):
        values = np.asarray(ranking)
        if values.ndim != 1 or values.shape[0] != n_features:
            raise ValueError(f"fold {fold_idx} has ranking shape {values.shape}")
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"fold {fold_idx} ranking is not integer-valued")
        if not np.array_equal(np.sort(values), expected):
            raise ValueError(f"fold {fold_idx} ranking is not a complete permutation")
        normalized.append(values.astype(int).tolist())

    selected_set_sha256 = {
        str(k): [_integer_digest(sorted(ranking[:k])) for ranking in normalized] for k in k_values
    }
    return {
        "rankings": normalized,
        "ranking_sha256": [_integer_digest(ranking) for ranking in normalized],
        "selected_set_sha256": selected_set_sha256,
    }


def _replacement_inventory(
    manifest: RerunManifest,
) -> dict[TaskType, dict[str, tuple[ManifestCell, ...]]]:
    """Validate and group the complete replacement manifest."""
    validate_canonical_campaign(manifest)
    if not _SHA256_PATTERN.fullmatch(manifest.sha256):
        raise ValueError("replacement manifest digest is invalid")
    replacement_cells = tuple(
        cell for cell in manifest.cells if cell.config.method.name == "r_cforest"
    )
    if len(replacement_cells) != EXPECTED_REPLACEMENT_CELLS:
        raise ValueError(
            "canonical manifest must contain "
            f"{EXPECTED_REPLACEMENT_CELLS} r_cforest cells, observed "
            f"{len(replacement_cells)}"
        )
    replacement_account_ids = {cell.target_aws_account_id for cell in replacement_cells}
    if len(replacement_account_ids) != N_EXPECTED_GATE_ACCOUNTS:
        raise ValueError(
            f"r_cforest replacement cells must bind exactly {N_EXPECTED_GATE_ACCOUNTS} AWS account"
        )
    grouped: dict[tuple[TaskType, str], list[ManifestCell]] = defaultdict(list)
    for cell in replacement_cells:
        if cell.rerun_reason != REPLACEMENT_REASON:
            raise ValueError("replacement manifest cell contract differs")
        grouped[(cell.config.task, cell.config.dataset)].append(cell)
    if len(grouped) != EXPECTED_DATASET_TASK_PAIRS:
        raise ValueError(
            "replacement manifest must contain "
            f"{EXPECTED_DATASET_TASK_PAIRS} dataset/task pairs, observed {len(grouped)}"
        )
    if {task for task, _dataset in grouped} != {"classification", "regression"}:
        raise ValueError("replacement manifest must contain both tasks")
    real_tasks = {cell.config.task for cell in replacement_cells if cell.dataset_source == "real"}
    if real_tasks != {"classification", "regression"}:
        raise ValueError("replacement manifest must contain real datasets for both tasks")

    inventory: dict[TaskType, dict[str, tuple[ManifestCell, ...]]] = {
        "classification": {},
        "regression": {},
    }
    for (task, dataset), cells in sorted(grouped.items()):
        expected_labels = set(_expected_configurations(task))
        expected_cells = set(product(expected_labels, EXPECTED_SEEDS))
        observed_cells = {(cell.config.method.label, cell.config.seed) for cell in cells}
        if observed_cells != expected_cells or len(cells) != len(expected_cells):
            raise ValueError(f"replacement manifest has an incomplete {task}/{dataset} grid")
        identities = {cell.config.dataset_identity for cell in cells}
        sources = {cell.dataset_source for cell in cells}
        if len(identities) != 1 or len(sources) != 1:
            raise ValueError(f"replacement manifest conflicts within {task}/{dataset}")
        inventory[task][dataset] = tuple(
            sorted(cells, key=lambda cell: (cell.config.method.label, cell.config.seed))
        )
    return inventory


def _gate_inventory(
    manifest: RerunManifest,
) -> dict[TaskType, dict[str, tuple[ManifestCell, ...]]]:
    """Select the fixed reproducibility panel from the complete manifest."""
    replacement_inventory = _replacement_inventory(manifest)
    panel: dict[TaskType, dict[str, tuple[ManifestCell, ...]]] = {
        "classification": {},
        "regression": {},
    }
    for task, task_inventory in replacement_inventory.items():
        real_datasets = {
            dataset: cells
            for dataset, cells in task_inventory.items()
            if cells[0].dataset_source == "real"
        }
        if len(real_datasets) < 2:
            raise ValueError(f"r_cforest gate requires two real {task} datasets")

        compact_dataset = min(
            real_datasets,
            key=lambda dataset: (
                real_datasets[dataset][0].config.dataset_identity.n_samples
                * real_datasets[dataset][0].config.dataset_identity.n_features,
                real_datasets[dataset][0].config.dataset_identity.n_features,
                real_datasets[dataset][0].config.dataset_identity.n_samples,
                dataset,
            ),
        )
        maximum_features = max(
            cells[0].config.dataset_identity.n_features for cells in real_datasets.values()
        )
        high_dimensional_dataset = min(
            (
                dataset
                for dataset, cells in real_datasets.items()
                if cells[0].config.dataset_identity.n_features == maximum_features
            ),
            key=lambda dataset: (
                real_datasets[dataset][0].config.dataset_identity.n_samples,
                dataset,
            ),
        )
        if compact_dataset == high_dimensional_dataset:
            raise ValueError(f"r_cforest gate {task} dimensional anchors must differ")

        datasets = {
            "compact": compact_dataset,
            "high_dimensional": high_dimensional_dataset,
        }
        selected: dict[str, list[ManifestCell]] = defaultdict(list)
        for profile, testtype, replace, seed in GATE_PANEL_SPECIFICATION:
            dataset = datasets[profile]
            matches = [
                cell
                for cell in real_datasets[dataset]
                if cell.config.seed == seed
                and cell.config.method.params_dict.get("testtype") == testtype
                and cell.config.method.params_dict.get("replace") is replace
            ]
            if len(matches) != 1:
                raise ValueError(
                    "r_cforest gate panel does not identify exactly one "
                    f"{task}/{dataset}/{testtype}/replace={replace}/seed={seed} cell"
                )
            selected[dataset].append(matches[0])
        panel[task] = {
            dataset: tuple(
                sorted(cells, key=lambda cell: (cell.config.method.label, cell.config.seed))
            )
            for dataset, cells in sorted(selected.items())
        }

    observed_dataset_task_pairs = sum(len(task_inventory) for task_inventory in panel.values())
    observed_cells = sum(
        len(cells) for task_inventory in panel.values() for cells in task_inventory.values()
    )
    if (
        observed_dataset_task_pairs != EXPECTED_GATE_DATASET_TASK_PAIRS
        or observed_cells != EXPECTED_GATE_CELLS
    ):
        raise RuntimeError(
            "r_cforest gate panel size differs: "
            f"dataset_task_pairs={observed_dataset_task_pairs}, cells={observed_cells}"
        )
    return panel


def _identity_payload(cell: ManifestCell) -> dict[str, Any]:
    identity = cell.config.dataset_identity
    return {
        "sha256": identity.sha256,
        "n_samples": identity.n_samples,
        "n_features": identity.n_features,
    }


def _configuration_key(cell: ManifestCell) -> str:
    """Return the unique result key for one manifest cell."""
    return f"{cell.config.method.label}_seed{cell.config.seed}"


def _load_manifest(path: Path) -> RerunManifest:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read replacement manifest {path}: {exc}") from exc
    manifest = parse_rerun_manifest(payload)
    _replacement_inventory(manifest)
    return manifest


def run_gate(
    run_id: str,
    runtime_contract: dict[str, Any],
    manifest: RerunManifest,
) -> dict[str, Any]:
    """Execute one fresh-process gate run."""
    started = time.monotonic()
    expected_runtime_sha256 = runtime_contract_sha256(runtime_contract)
    if manifest.runtime_contract_sha256 != expected_runtime_sha256:
        raise ValueError("replacement manifest is bound to a different runtime contract")
    provenance = _provenance(run_id)
    _require_runtime_match(provenance, runtime_contract, source="running process")
    replacement_inventory = _replacement_inventory(manifest)
    inventory = _gate_inventory(manifest)
    target_aws_account_ids = sorted(
        {
            cell.target_aws_account_id
            for task_inventory in replacement_inventory.values()
            for cells in task_inventory.values()
            for cell in cells
        }
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "profile": "r_cforest_reproducibility",
        "target_aws_account_ids": target_aws_account_ids,
        "campaign_sha256": manifest.campaign_sha256,
        "manifest_sha256": manifest.sha256,
        "runtime_contract_sha256": expected_runtime_sha256,
        "provenance": provenance,
        "results": {"classification": {}, "regression": {}},
    }

    for task, task_inventory in inventory.items():
        for dataset, cells in task_inventory.items():
            first_cell = cells[0]
            identity = first_cell.config.dataset_identity
            X, y = load_dataset(
                dataset,
                task,
                identity=identity,
                source=first_cell.dataset_source,
            )
            k_values = get_requested_evaluation_k_values(identity.n_features)
            dataset_result: dict[str, Any] = {
                "dataset": dataset,
                "dataset_source": first_cell.dataset_source,
                "task": task,
                "identity": _identity_payload(first_cell),
                "k_values": k_values,
                "configurations": {},
            }
            for cell in cells:
                config = cell.config.method
                config_started = time.monotonic()
                rows = run_r_selection_parallel(
                    X,
                    y,
                    "r_cforest",
                    task,
                    seed=cell.config.seed,
                    params=config.params_dict,
                )
                rankings = [list(map(int, row["feature_ranking"])) for row in rows]
                dataset_result["configurations"][_configuration_key(cell)] = {
                    "method": config.name,
                    "params": config.params_dict,
                    "seed": cell.config.seed,
                    "elapsed_seconds": time.monotonic() - config_started,
                    "fold_cpu_affinity": [row["fold_cpu_affinity"] for row in rows],
                    **summarize_rankings(
                        rankings,
                        n_features=identity.n_features,
                        k_values=k_values,
                    ),
                }
            payload["results"][task][dataset] = dataset_result

    payload["elapsed_seconds"] = time.monotonic() - started
    return payload


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON contains duplicate field {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r} is not allowed")


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read gate payload {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"gate payload is not an object: {path}")
    return payload


def _require_exact_fields(
    value: dict[str, Any],
    expected: set[str],
    *,
    source: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{source} fields differ: missing={missing}, extra={extra}")


def _require_nonempty_string(value: Any, *, source: str) -> str:
    if not isinstance(value, str) or not value.strip() or value == "unknown":
        raise ValueError(f"{source} must be a non-empty concrete string")
    return value


def _require_positive_number(value: Any, *, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{source} must be a positive number")
    return float(value)


def _validate_version_mapping(
    value: Any,
    *,
    expected_fields: set[str],
    source: str,
) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{source} must be an object")
    _require_exact_fields(value, expected_fields, source=source)
    for field, version in value.items():
        _require_nonempty_string(version, source=f"{source}.{field}")


def _machine_matches_architecture(machine: str, architecture: str) -> bool:
    """Return whether a kernel machine label matches signed EC2 architecture."""
    normalized_machine = {"aarch64": "arm64"}.get(machine, machine)
    return normalized_machine == architecture


def _validate_provenance(
    provenance: Any,
    *,
    source: str,
) -> InstanceIdentityEvidence:
    if not isinstance(provenance, dict):
        raise TypeError(f"{source} provenance is not an object")
    _require_exact_fields(provenance, PROVENANCE_FIELDS, source=f"{source} provenance")

    for field in (
        "ami_id",
        "architecture",
        "availability_zone",
        "availability_zone_id",
        "cpu_model",
        "hostname",
        "instance_id",
        "instance_type",
        "kernel",
        "machine",
        "microcode",
        "run_id",
    ):
        _require_nonempty_string(provenance[field], source=f"{source} provenance.{field}")
    identity_record = provenance["instance_identity"]
    if not isinstance(identity_record, Mapping):
        raise TypeError(f"{source} provenance.instance_identity is not an object")
    evidence = validate_instance_identity_record(identity_record)
    signed = evidence.identity
    expected_identity_fields = {
        "ami_id": signed.image_id,
        "architecture": signed.architecture,
        "availability_zone": signed.availability_zone,
        "availability_zone_id": evidence.availability_zone_id,
        "aws_account_id": signed.account_id,
        "instance_id": signed.instance_id,
        "instance_type": signed.instance_type,
    }
    mismatches = [
        field
        for field, expected in expected_identity_fields.items()
        if provenance[field] != expected
    ]
    if mismatches:
        raise ValueError(f"{source} provenance differs from signed EC2 identity: {mismatches}")
    if not _machine_matches_architecture(
        str(provenance["machine"]),
        signed.architecture,
    ):
        raise ValueError(f"{source} provenance.machine differs from signed EC2 architecture")
    if not _IMAGE_DIGEST_PATTERN.fullmatch(str(provenance["container_image"])):
        raise ValueError(f"{source} provenance.container_image is not immutable")
    if not _GIT_SHA_PATTERN.fullmatch(str(provenance["git_sha"])):
        raise ValueError(f"{source} provenance.git_sha is invalid")
    if not _SHA256_PATTERN.fullmatch(str(provenance["script_sha256"])):
        raise ValueError(f"{source} provenance.script_sha256 is invalid")
    if not _BOOT_ID_PATTERN.fullmatch(str(provenance["boot_id"])):
        raise ValueError(f"{source} provenance.boot_id is invalid")

    for field in (
        "logical_cpus",
        "process_id",
        "process_start_ticks",
    ):
        value = provenance[field]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{source} provenance.{field} must be a positive integer")
    validate_cpu_affinity(
        provenance["cpu_affinity"],
        logical_cpus=provenance["logical_cpus"],
        source=f"{source} provenance.cpu_affinity",
    )
    if len(provenance["cpu_affinity"]) < N_FOLDS:
        raise ValueError(f"{source} provenance.cpu_affinity cannot provide five fold partitions")

    _validate_version_mapping(
        provenance["os_release"],
        expected_fields={"ID", "VERSION_ID"},
        source=f"{source} provenance.os_release",
    )
    _validate_version_mapping(
        provenance["python_libraries"],
        expected_fields=set(PYTHON_LIBRARY_NAMES),
        source=f"{source} provenance.python_libraries",
    )
    _validate_version_mapping(
        provenance["r_runtime"],
        expected_fields=set(R_RUNTIME_FIELDS),
        source=f"{source} provenance.r_runtime",
    )
    _validate_version_mapping(
        provenance["r_numerical_libraries"],
        expected_fields={"blas", "lapack"},
        source=f"{source} provenance.r_numerical_libraries",
    )
    validate_openssl_version(
        provenance["openssl_version"],
        source=f"{source} provenance.openssl_version",
    )

    if provenance["thread_environment"] != EXPECTED_THREAD_ENVIRONMENT:
        raise ValueError(f"{source} provenance.thread_environment is not frozen")
    threadpools = provenance["threadpools"]
    if not isinstance(threadpools, list) or not threadpools:
        raise ValueError(f"{source} provenance.threadpools must be a non-empty list")
    for index, pool in enumerate(threadpools):
        if not isinstance(pool, dict):
            raise TypeError(f"{source} provenance.threadpools[{index}] is not an object")
        if pool.get("num_threads") != 1:
            raise ValueError(f"{source} provenance.threadpools[{index}] is not single-threaded")
        for field in ("filepath", "internal_api", "prefix", "user_api"):
            _require_nonempty_string(
                pool.get(field),
                source=f"{source} provenance.threadpools[{index}].{field}",
            )
    return evidence


def create_runtime_contract(
    provenance: dict[str, Any],
    operator_public_key: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the content-addressed runtime contract from one validated probe."""
    _validate_provenance(provenance, source="runtime probe")
    runtime = {
        field: provenance[field]
        for field in sorted(RUNTIME_PROVENANCE_FIELDS - {"container_image_digest"})
    }
    runtime["container_image_digest"] = _container_image_digest(provenance["container_image"])
    return validate_runtime_contract(
        {
            "schema_version": RUNTIME_CONTRACT_SCHEMA_VERSION,
            "profile": RUNTIME_CONTRACT_PROFILE,
            "operator_attestation_public_key": validate_operator_public_key(operator_public_key),
            "runtime": dict(sorted(runtime.items())),
        }
    )


def running_runtime_contract_sha256(runtime_contract: Mapping[str, Any]) -> str:
    """Probe and hash the current worker's frozen runtime contract."""
    normalized = validate_runtime_contract(runtime_contract)
    observed = create_runtime_contract(
        _provenance("worker-runtime-probe"),
        normalized["operator_attestation_public_key"],
    )
    return runtime_contract_sha256(observed)


def require_running_runtime_contract(runtime_contract: Mapping[str, Any]) -> None:
    """Reject a worker whose runtime differs from the gate-approved contract."""
    expected_sha256 = runtime_contract_sha256(runtime_contract)
    observed_sha256 = running_runtime_contract_sha256(runtime_contract)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            "running worker runtime contract differs from the gate-approved "
            f"digest: expected {expected_sha256}, observed {observed_sha256}"
        )


def _require_runtime_match(
    provenance: dict[str, Any],
    contract: dict[str, Any],
    *,
    source: str,
) -> None:
    runtime = validate_runtime_contract(contract)["runtime"]
    observed = {
        field: provenance[field]
        for field in sorted(RUNTIME_PROVENANCE_FIELDS - {"container_image_digest"})
    }
    observed["container_image_digest"] = _container_image_digest(provenance["container_image"])
    mismatches = [
        field for field in sorted(RUNTIME_PROVENANCE_FIELDS) if observed[field] != runtime[field]
    ]
    if mismatches:
        raise ValueError(f"{source} differs from the frozen runtime: {mismatches}")


def _load_runtime_contract(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read runtime contract {path}: {exc}") from exc
    return parse_runtime_contract(payload)


def _expected_configurations(task: TaskType) -> dict[str, Any]:
    configurations = get_full_method_configs(["r_cforest"], task)
    combinations = {
        (config.params_dict.get("testtype"), config.params_dict.get("replace"))
        for config in configurations
    }
    expected_combinations = {
        ("Bonferroni", False),
        ("Bonferroni", True),
        ("MonteCarlo", False),
        ("MonteCarlo", True),
    }
    if len(configurations) != 4 or combinations != expected_combinations:
        raise RuntimeError("r_cforest gate requires the four frozen production configurations")
    return {configuration.label: configuration for configuration in configurations}


def _validate_payload(
    payload: dict[str, Any],
    *,
    manifest: RerunManifest,
    runtime_contract: dict[str, Any],
    source: str,
) -> InstanceIdentityEvidence:
    replacement_inventory = _replacement_inventory(manifest)
    inventory = _gate_inventory(manifest)
    _require_exact_fields(payload, PAYLOAD_FIELDS, source=source)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{source} has an unexpected schema version")
    if payload.get("profile") != "r_cforest_reproducibility":
        raise ValueError(f"{source} has an unexpected profile")
    expected_manifest_fields = {
        "campaign_sha256": manifest.campaign_sha256,
        "manifest_sha256": manifest.sha256,
        "target_aws_account_ids": sorted(
            {
                cell.target_aws_account_id
                for task_inventory in replacement_inventory.values()
                for cells in task_inventory.values()
                for cell in cells
            }
        ),
    }
    for field, expected_value in expected_manifest_fields.items():
        if payload[field] != expected_value:
            raise ValueError(f"{source} has an invalid {field}")
    _require_positive_number(payload["elapsed_seconds"], source=f"{source} elapsed_seconds")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError(f"{source} provenance is not an object")
    evidence = _validate_provenance(provenance, source=source)
    expected_runtime_sha256 = runtime_contract_sha256(runtime_contract)
    if manifest.runtime_contract_sha256 != expected_runtime_sha256:
        raise ValueError(f"{source} manifest is bound to a different runtime contract")
    if payload["runtime_contract_sha256"] != expected_runtime_sha256:
        raise ValueError(f"{source} has an invalid runtime contract digest")
    _require_runtime_match(provenance, runtime_contract, source=source)
    results = payload.get("results")
    if not isinstance(results, dict) or set(results) != {
        "classification",
        "regression",
    }:
        raise ValueError(f"{source} does not contain both gate tasks")

    for task in ("classification", "regression"):
        task_results = results[task]
        if not isinstance(task_results, dict):
            raise TypeError(f"{source} {task} results are not an object")
        if set(task_results) != set(inventory[task]):
            raise ValueError(f"{source} has an invalid {task} dataset inventory")

        for dataset, cells in inventory[task].items():
            dataset_result = task_results[dataset]
            if not isinstance(dataset_result, dict):
                raise TypeError(f"{source} {task}/{dataset} result is not an object")
            _require_exact_fields(
                dataset_result,
                DATASET_RESULT_FIELDS,
                source=f"{source} {task}/{dataset} result",
            )
            first_cell = cells[0]
            identity = first_cell.config.dataset_identity
            if (
                dataset_result["task"] != task
                or dataset_result["dataset"] != dataset
                or dataset_result["dataset_source"] != first_cell.dataset_source
                or dataset_result["identity"] != _identity_payload(first_cell)
            ):
                raise ValueError(f"{source} has an invalid {task}/{dataset} identity")
            expected_k_values = get_requested_evaluation_k_values(identity.n_features)
            if dataset_result["k_values"] != expected_k_values:
                raise ValueError(f"{source} has an invalid {task}/{dataset} feature-count schedule")

            configurations = dataset_result["configurations"]
            expected_configurations = {_configuration_key(cell): cell for cell in cells}
            if not isinstance(configurations, dict) or set(configurations) != set(
                expected_configurations
            ):
                raise ValueError(
                    f"{source} does not contain the exact {task}/{dataset} configurations"
                )
            for label, expected_cell in expected_configurations.items():
                expected_configuration = expected_cell.config.method
                result = configurations[label]
                if not isinstance(result, dict):
                    raise TypeError(f"{source} {task}/{dataset}/{label} is not an object")
                _require_exact_fields(
                    result,
                    CONFIGURATION_RESULT_FIELDS,
                    source=f"{source} {task}/{dataset}/{label}",
                )
                if result["method"] != expected_configuration.name:
                    raise ValueError(f"{source} {task}/{dataset}/{label} has an invalid method")
                if result["params"] != expected_configuration.params_dict:
                    raise ValueError(f"{source} {task}/{dataset}/{label} has invalid parameters")
                if result["seed"] != expected_cell.config.seed:
                    raise ValueError(f"{source} {task}/{dataset}/{label} has an invalid seed")
                _require_positive_number(
                    result["elapsed_seconds"],
                    source=(f"{source} {task}/{dataset}/{label} elapsed_seconds"),
                )
                summary = summarize_rankings(
                    result["rankings"],
                    n_features=identity.n_features,
                    k_values=expected_k_values,
                )
                for field in ("ranking_sha256", "selected_set_sha256"):
                    if result[field] != summary[field]:
                        raise ValueError(
                            f"{source} {task}/{dataset}/{label} has an invalid {field}"
                        )
                expected_cpu_affinity = [
                    list(cpu_ids)
                    for cpu_ids in partition_cpu_ids(
                        tuple(provenance["cpu_affinity"]),
                        N_FOLDS,
                    )
                ]
                if result["fold_cpu_affinity"] != expected_cpu_affinity:
                    raise ValueError(
                        f"{source} {task}/{dataset}/{label} has invalid fold_cpu_affinity"
                    )
    return evidence


def _canonical_operator_readbacks(
    readbacks: Sequence[Mapping[str, Any]],
    *,
    evidence_by_instance: Mapping[str, InstanceIdentityEvidence],
    gate_launch_nonce: str,
    manifest: RerunManifest,
    payloads: Sequence[Mapping[str, Any]],
    host_slot_by_instance: Mapping[str, str],
    receipt_created_at_utc: str,
    runtime_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Verify and order one fresh signed operator readback per gate host."""
    if len(readbacks) != N_EXPECTED_HOSTS:
        raise ValueError(
            f"expected {N_EXPECTED_HOSTS} operator readbacks, observed {len(readbacks)}"
        )
    normalized_runtime = validate_runtime_contract(runtime_contract)
    runtime_sha256 = runtime_contract_sha256(normalized_runtime)
    run_payload_sha256s = _run_payload_sha256s(payloads)
    receipt_created_at = parse_utc_timestamp(
        receipt_created_at_utc,
        source="gate receipt created_at_utc",
    )
    validated: list[tuple[OperatorInstanceReadback, dict[str, Any]]] = []
    seen_instances: set[str] = set()
    for index, record in enumerate(readbacks):
        if not isinstance(record, Mapping):
            raise TypeError(f"operator attestation[{index}] is not an object")
        attestation = validate_operator_attestation(
            record,
            campaign_sha256=manifest.campaign_sha256,
            manifest_sha256=manifest.sha256,
            public_key=normalized_runtime["operator_attestation_public_key"],
            run_payload_sha256s=run_payload_sha256s,
            runtime_contract_sha256=runtime_sha256,
        )
        observed_at = parse_utc_timestamp(
            attestation["observed_at_utc"],
            source=f"operator attestation[{index}] observed_at_utc",
        )
        age = receipt_created_at - observed_at
        if age > timedelta(seconds=MAX_OPERATOR_READBACK_AGE_SECONDS):
            raise ValueError(f"operator attestation[{index}] is stale")
        if age < -timedelta(seconds=MAX_OPERATOR_READBACK_CLOCK_SKEW_SECONDS):
            raise ValueError(f"operator attestation[{index}] is from the future")
        readback_record = attestation["readback"]
        instance_id = readback_record.get("instance_id")
        if not isinstance(instance_id, str) or instance_id not in evidence_by_instance:
            raise ValueError(f"operator attestation[{index}] does not identify a gate instance")
        if instance_id in seen_instances:
            raise ValueError(f"duplicate operator readback for instance {instance_id}")
        seen_instances.add(instance_id)
        evidence = evidence_by_instance[instance_id]
        readback = validate_operator_readback(evidence, readback_record)
        if readback.state != "running":
            raise ValueError(
                f"operator readback for instance {instance_id} was not collected while running"
            )
        if readback.instance_lifecycle != GATE_MARKET:
            raise ValueError(f"operator readback for instance {instance_id} is not on-demand")
        tags = dict(readback.tags)
        source_git_sha = str(normalized_runtime["runtime"]["git_sha"])
        image_digest = str(normalized_runtime["runtime"]["container_image_digest"])
        try:
            gate_identity = gate_launch_identity(
                source_git_sha,
                image_digest,
                gate_launch_nonce,
            )
            output_prefix = gate_output_prefix(
                source_git_sha,
                image_digest,
                gate_launch_nonce,
            )
        except ValueError as exc:
            raise ValueError("the trusted gate launch nonce is invalid") from exc
        expected_tags = {
            "citrees-artifact-prefix": output_prefix,
            "citrees-gate-identity": gate_identity,
            "citrees-gate-launch-nonce": gate_launch_nonce,
            "citrees-host-slot": host_slot_by_instance[instance_id],
            "citrees-image-digest": image_digest,
            "citrees-market": GATE_MARKET,
            "citrees-role": "r-cforest-reproducibility-gate",
            "citrees-source-git-sha": source_git_sha,
        }
        if any(tags.get(key) != value for key, value in expected_tags.items()):
            raise ValueError(
                f"operator readback for instance {instance_id} lacks the exact gate launch tags"
            )
        instance_role = get_assumed_role_identity(evidence.sts_identity.arn)
        operator_role = get_assumed_role_identity(readback.operator_identity.arn)
        if (
            readback.operator_identity.arn == evidence.sts_identity.arn
            or (
                operator_role is not None
                and instance_role is not None
                and operator_role[0] == instance_role[0]
            )
            or (operator_role is not None and operator_role[1] in evidence_by_instance)
        ):
            raise ValueError(f"operator readback for instance {instance_id} is not independent")
        if readback_record != readback.to_record() or dict(record) != attestation:
            raise ValueError(f"operator attestation[{index}] is not canonical")
        validated.append((readback, attestation))
    if seen_instances != set(evidence_by_instance):
        raise ValueError("operator readbacks do not cover the exact gate hosts")
    return [
        attestation
        for readback, attestation in sorted(
            validated,
            key=lambda value: value[0].instance_id,
        )
    ]


def _gate_host_slots_by_instance(
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Bind the exact gate run IDs to two host slots and instance IDs."""
    expected_run_ids = {
        f"{slot_id}-repeat-{repeat}" for slot_id in GATE_HOST_SLOTS for repeat in GATE_REPEATS
    }
    observed_run_ids: set[str] = set()
    instance_by_slot: dict[str, str] = {}
    slot_by_instance: dict[str, str] = {}
    for payload in payloads:
        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping):
            raise TypeError("gate payload provenance is not an object")
        run_id = provenance.get("run_id")
        instance_id = provenance.get("instance_id")
        if not isinstance(run_id, str) or not isinstance(instance_id, str):
            raise ValueError("gate payload lacks a run ID or instance ID")
        if run_id not in expected_run_ids:
            raise ValueError("gate run IDs do not cover the exact Arc host slots and repeats")
        observed_run_ids.add(run_id)
        slot_id = run_id.partition("-repeat-")[0]
        prior_instance = instance_by_slot.setdefault(slot_id, instance_id)
        prior_slot = slot_by_instance.setdefault(instance_id, slot_id)
        if prior_instance != instance_id or prior_slot != slot_id:
            raise ValueError("gate host slots do not map one-to-one to instance IDs")
    if observed_run_ids != expected_run_ids:
        raise ValueError("gate run IDs do not cover the exact Arc host slots and repeats")
    if set(instance_by_slot) != set(GATE_HOST_SLOTS) or len(slot_by_instance) != N_EXPECTED_HOSTS:
        raise ValueError("gate host slots do not identify the exact two instances")
    return slot_by_instance


def collect_live_operator_readbacks(
    payloads: Sequence[dict[str, Any]],
    *,
    manifest: RerunManifest,
    operator_private_key_pem: bytes,
    operator_profiles: Sequence[str],
    runtime_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect and sign one live AWS control-plane readback per gate host."""
    if (
        len(operator_profiles) != N_EXPECTED_GATE_ACCOUNTS
        or len(set(operator_profiles)) != N_EXPECTED_GATE_ACCOUNTS
        or any(not profile or profile != profile.strip() for profile in operator_profiles)
    ):
        raise ValueError(
            f"operator_profiles must contain exactly {N_EXPECTED_GATE_ACCOUNTS} unique profile name"
        )
    if len(payloads) != N_EXPECTED_RUNS:
        raise ValueError(f"expected {N_EXPECTED_RUNS} gate payloads, observed {len(payloads)}")

    evidence_by_account: dict[str, dict[str, InstanceIdentityEvidence]] = {}
    evidence_by_instance: dict[str, InstanceIdentityEvidence] = {}
    for index, payload in enumerate(payloads):
        provenance = payload.get("provenance")
        evidence = _validate_provenance(
            provenance,
            source=f"payload[{index}]",
        )
        instance_id = evidence.identity.instance_id
        prior = evidence_by_instance.get(instance_id)
        if prior is not None and prior.to_record() != evidence.to_record():
            raise ValueError(f"instance {instance_id} has conflicting signed identity evidence")
        evidence_by_instance[instance_id] = evidence
        account_id = evidence.identity.account_id
        evidence_by_account.setdefault(account_id, {})[instance_id] = evidence
    if (
        len(evidence_by_account) != N_EXPECTED_GATE_ACCOUNTS
        or len(evidence_by_instance) != N_EXPECTED_HOSTS
    ):
        raise ValueError("gate payloads do not identify the exact account and host counts")

    import boto3

    normalized_runtime = validate_runtime_contract(runtime_contract)
    runtime_sha256 = runtime_contract_sha256(normalized_runtime)
    run_payload_sha256s = _run_payload_sha256s(payloads)
    readbacks: list[dict[str, Any]] = []
    observed_accounts: set[str] = set()
    for profile in operator_profiles:
        session = boto3.Session(
            profile_name=profile,
            region_name=SUPPORTED_REGION,
        )
        sts_client = session.client("sts")
        operator_identity = collect_aws_caller_identity(sts_client=sts_client)
        account_id = operator_identity.account_id
        if account_id in observed_accounts:
            raise ValueError(f"operator profiles duplicate AWS account {account_id}")
        account_evidence = evidence_by_account.get(account_id)
        if account_evidence is None:
            raise ValueError(
                f"operator profile {profile!r} targets account {account_id}, "
                "which is not a gate account"
            )
        observed_accounts.add(account_id)
        ec2_client = session.client("ec2")
        iam_client = session.client("iam")
        for instance_id in sorted(account_evidence):
            readback = collect_operator_readback(
                account_evidence[instance_id],
                ec2_client=ec2_client,
                iam_client=iam_client,
                sts_client=sts_client,
            )
            readbacks.append(
                create_operator_attestation(
                    readback.to_record(),
                    campaign_sha256=manifest.campaign_sha256,
                    manifest_sha256=manifest.sha256,
                    observed_at_utc=utc_timestamp(),
                    private_key_pem=operator_private_key_pem,
                    public_key=normalized_runtime["operator_attestation_public_key"],
                    run_payload_sha256s=run_payload_sha256s,
                    runtime_contract_sha256=runtime_sha256,
                )
            )
    if observed_accounts != set(evidence_by_account):
        raise ValueError("operator profiles do not cover the exact gate accounts")
    return readbacks


def compare_payloads(
    payloads: Sequence[dict[str, Any]],
    operator_readbacks: Sequence[Mapping[str, Any]],
    *,
    gate_launch_nonce: str,
    manifest: RerunManifest,
    receipt_created_at_utc: str | None = None,
    runtime_contract: dict[str, Any],
) -> dict[str, Any]:
    """Require four exact gate runs across two equivalent hosts and two zones."""
    created_at_utc = utc_timestamp() if receipt_created_at_utc is None else receipt_created_at_utc
    runtime = validate_runtime_contract(runtime_contract)["runtime"]
    if len(payloads) != N_EXPECTED_RUNS:
        raise ValueError(f"expected {N_EXPECTED_RUNS} gate payloads, observed {len(payloads)}")
    evidences: list[InstanceIdentityEvidence] = []
    for index, payload in enumerate(payloads):
        evidences.append(
            _validate_payload(
                payload,
                manifest=manifest,
                runtime_contract=runtime_contract,
                source=f"payload[{index}]",
            )
        )

    provenances = [payload["provenance"] for payload in payloads]
    host_slot_by_instance = _gate_host_slots_by_instance(payloads)

    instance_ids = [str(provenance.get("instance_id", "")) for provenance in provenances]
    instance_counts = {
        instance_id: instance_ids.count(instance_id) for instance_id in set(instance_ids)
    }
    if len(instance_counts) != N_EXPECTED_HOSTS or set(instance_counts.values()) != {2}:
        raise ValueError("gate requires two runs on each of two instance IDs")
    evidence_by_instance: dict[str, InstanceIdentityEvidence] = {}
    evidence_records_by_instance: dict[str, Mapping[str, Any]] = {}
    for provenance, evidence in zip(provenances, evidences, strict=True):
        instance_id = evidence.identity.instance_id
        record = provenance["instance_identity"]
        prior = evidence_records_by_instance.get(instance_id)
        if prior is not None and prior != record:
            raise ValueError(f"instance {instance_id} has conflicting signed identity evidence")
        evidence_by_instance[instance_id] = evidence
        evidence_records_by_instance[instance_id] = record
    canonical_readbacks = _canonical_operator_readbacks(
        operator_readbacks,
        evidence_by_instance=evidence_by_instance,
        gate_launch_nonce=gate_launch_nonce,
        manifest=manifest,
        payloads=payloads,
        host_slot_by_instance=host_slot_by_instance,
        receipt_created_at_utc=created_at_utc,
        runtime_contract=runtime_contract,
    )
    readback_records = [attestation["readback"] for attestation in canonical_readbacks]

    process_incarnations = {
        (
            provenance["instance_id"],
            provenance["boot_id"],
            provenance["process_start_ticks"],
        )
        for provenance in provenances
    }
    if len(process_incarnations) != N_EXPECTED_RUNS:
        raise ValueError("gate requires four distinct Linux process incarnations")
    host_boot_ids: set[str] = set()
    for instance_id in instance_counts:
        host_provenances = [
            provenance for provenance in provenances if provenance["instance_id"] == instance_id
        ]
        boot_ids = {str(provenance["boot_id"]) for provenance in host_provenances}
        start_ticks = {int(provenance["process_start_ticks"]) for provenance in host_provenances}
        if len(boot_ids) != 1 or len(start_ticks) != 2:
            raise ValueError(f"instance {instance_id} requires one boot and two fresh processes")
        host_boot_ids.update(boot_ids)
    if len(host_boot_ids) != N_EXPECTED_HOSTS:
        raise ValueError("gate requires independent boot identities for both hosts")

    expected_account_ids = {
        cell.target_aws_account_id
        for task_inventory in _replacement_inventory(manifest).values()
        for cells in task_inventory.values()
        for cell in cells
    }
    observed_account_ids = {str(provenance["aws_account_id"]) for provenance in provenances}
    if observed_account_ids != expected_account_ids:
        raise ValueError("gate hosts must run in the exact target AWS account")
    for instance_id in instance_counts:
        host_account_ids = {
            str(provenance["aws_account_id"])
            for provenance in provenances
            if provenance["instance_id"] == instance_id
        }
        if len(host_account_ids) != 1:
            raise ValueError(f"instance {instance_id} appears in multiple AWS accounts")

    availability_zone_ids = {str(readback["availability_zone_id"]) for readback in readback_records}
    if "" in availability_zone_ids or len(availability_zone_ids) != N_EXPECTED_HOSTS:
        raise ValueError("gate requires two distinct physical availability-zone IDs")
    for instance_id in instance_counts:
        host_zone_ids = {
            str(provenance.get("availability_zone_id", ""))
            for provenance in provenances
            if provenance.get("instance_id") == instance_id
        }
        if len(host_zone_ids) != 1:
            raise ValueError(f"instance {instance_id} appears in multiple availability-zone IDs")
    availability_zones = {str(readback["availability_zone"]) for readback in readback_records}

    for field in STATIC_PROVENANCE_FIELDS:
        values = [provenance.get(field) for provenance in provenances]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"gate provenance differs for {field}")
    image_digests = [
        _container_image_digest(str(provenance["container_image"])) for provenance in provenances
    ]
    if any(digest != image_digests[0] for digest in image_digests[1:]):
        raise ValueError("gate provenance differs for container_image_digest")

    reference = payloads[0]["results"]
    executed_cells = 0
    fold_rankings = 0
    selected_sets = 0
    mismatches: list[str] = []
    for task, reference_task in reference.items():
        for dataset, reference_dataset in reference_task.items():
            for payload_index, payload in enumerate(payloads[1:], start=1):
                candidate_dataset = payload["results"][task][dataset]
                for field in ("dataset_source", "identity", "k_values"):
                    if candidate_dataset[field] != reference_dataset[field]:
                        mismatches.append(f"payload[{payload_index}] {task}/{dataset} {field}")
            for label, reference_result in reference_dataset["configurations"].items():
                executed_cells += 1
                for fold_idx, reference_ranking in enumerate(reference_result["rankings"]):
                    fold_rankings += 1
                    for payload_index, payload in enumerate(payloads[1:], start=1):
                        candidate_result = payload["results"][task][dataset]["configurations"][
                            label
                        ]
                        if candidate_result["params"] != reference_result["params"]:
                            mismatches.append(
                                f"payload[{payload_index}] {task}/{dataset}/{label} params"
                            )
                        if candidate_result["rankings"][fold_idx] != reference_ranking:
                            mismatches.append(
                                f"payload[{payload_index}] "
                                f"{task}/{dataset}/{label} "
                                f"fold={fold_idx} ranking"
                            )
                    for k in reference_dataset["k_values"]:
                        selected_sets += 1
                        reference_set = sorted(reference_ranking[:k])
                        for payload_index, payload in enumerate(payloads[1:], start=1):
                            candidate_ranking = payload["results"][task][dataset]["configurations"][
                                label
                            ]["rankings"][fold_idx]
                            if sorted(candidate_ranking[:k]) != reference_set:
                                mismatches.append(
                                    f"payload[{payload_index}] "
                                    f"{task}/{dataset}/{label} "
                                    f"fold={fold_idx} k={k}"
                                )

    if mismatches:
        examples = ", ".join(mismatches[:20])
        remainder = len(mismatches) - min(len(mismatches), 20)
        suffix = f", plus {remainder} more" if remainder else ""
        raise RuntimeError(f"r_cforest reproducibility gate failed: {examples}{suffix}")
    return {
        "status": "GO",
        "runs": N_EXPECTED_RUNS,
        "hosts": N_EXPECTED_HOSTS,
        "dataset_task_pairs": EXPECTED_GATE_DATASET_TASK_PAIRS,
        "process_incarnations": len(process_incarnations),
        "availability_zones": sorted(availability_zones),
        "availability_zone_ids": sorted(availability_zone_ids),
        "aws_account_ids": sorted(observed_account_ids),
        "operator_readbacks": len(canonical_readbacks),
        "operator_observed_at_utc": sorted(
            str(attestation["observed_at_utc"]) for attestation in canonical_readbacks
        ),
        "executed_cells": executed_cells,
        "fold_rankings": fold_rankings,
        "selected_sets": selected_sets,
        "runtime_contract_sha256": runtime_contract_sha256(runtime_contract),
        "instance_type": runtime["instance_type"],
        "cpu_model": runtime["cpu_model"],
    }


def validate_gate_receipt_sha256(value: str) -> str:
    """Validate one exact lowercase hexadecimal gate-receipt digest."""
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError("gate receipt SHA-256 must be 64 lowercase hexadecimal characters")
    return value


def gate_receipt_s3_key(sha256: str) -> str:
    """Return the content-addressed S3 key for one gate receipt."""
    return f"{GATE_RECEIPT_S3_PREFIX}/{validate_gate_receipt_sha256(sha256)}.json"


def _canonical_gate_payloads(
    payloads: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return gate payloads in their canonical run-ID order."""
    if len(payloads) != N_EXPECTED_RUNS:
        raise ValueError(f"expected {N_EXPECTED_RUNS} gate payloads, observed {len(payloads)}")
    run_ids: list[str] = []
    for index, payload in enumerate(payloads):
        if not isinstance(payload, dict):
            raise TypeError(f"gate payload[{index}] is not an object")
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            raise TypeError(f"gate payload[{index}] provenance is not an object")
        run_id = provenance.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"gate payload[{index}] has an invalid run ID")
        run_ids.append(run_id)
    if len(set(run_ids)) != N_EXPECTED_RUNS:
        raise ValueError("gate run IDs must be unique")
    return [
        payload
        for _run_id, payload in sorted(
            zip(run_ids, payloads, strict=True),
            key=lambda item: item[0],
        )
    ]


def create_gate_receipt(
    payloads: Sequence[dict[str, Any]],
    operator_readbacks: Sequence[Mapping[str, Any]],
    *,
    gate_launch_nonce: str,
    manifest: RerunManifest,
    runtime_contract: dict[str, Any],
) -> dict[str, Any]:
    """Create one complete immutable receipt for a successful gate."""
    canonical_payloads = _canonical_gate_payloads(payloads)
    created_at_utc = utc_timestamp()
    report = compare_payloads(
        canonical_payloads,
        operator_readbacks,
        gate_launch_nonce=gate_launch_nonce,
        manifest=manifest,
        receipt_created_at_utc=created_at_utc,
        runtime_contract=runtime_contract,
    )
    canonical_readbacks = [
        dict(record)
        for record in sorted(
            operator_readbacks,
            key=lambda value: str(value["readback"]["instance_id"]),
        )
    ]
    return {
        "schema_version": GATE_RECEIPT_SCHEMA_VERSION,
        "profile": GATE_RECEIPT_PROFILE,
        "account_manifest_sha256": account_manifest_sha256_map(manifest),
        "campaign_sha256": manifest.campaign_sha256,
        "created_at_utc": created_at_utc,
        "manifest_sha256": manifest.sha256,
        "runtime_contract_sha256": runtime_contract_sha256(runtime_contract),
        "run_payloads": canonical_payloads,
        "operator_readbacks": canonical_readbacks,
        "report": report,
    }


def validate_gate_receipt(
    receipt: dict[str, Any],
    *,
    manifest: RerunManifest,
    runtime_contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one complete successful gate receipt."""
    if not isinstance(receipt, dict):
        raise TypeError("gate receipt is not an object")
    _require_exact_fields(receipt, GATE_RECEIPT_FIELDS, source="gate receipt")
    if receipt["schema_version"] != GATE_RECEIPT_SCHEMA_VERSION:
        raise ValueError("gate receipt has an unexpected schema version")
    if receipt["profile"] != GATE_RECEIPT_PROFILE:
        raise ValueError("gate receipt has an unexpected profile")
    expected_bindings = {
        "account_manifest_sha256": account_manifest_sha256_map(manifest),
        "campaign_sha256": manifest.campaign_sha256,
        "manifest_sha256": manifest.sha256,
        "runtime_contract_sha256": runtime_contract_sha256(runtime_contract),
    }
    for field, expected in expected_bindings.items():
        if receipt[field] != expected:
            raise ValueError(f"gate receipt has an invalid {field}")
    created_at_utc = receipt["created_at_utc"]
    parse_utc_timestamp(created_at_utc, source="gate receipt created_at_utc")
    payloads = receipt["run_payloads"]
    if not isinstance(payloads, list):
        raise TypeError("gate receipt run_payloads is not an array")
    canonical_payloads = _canonical_gate_payloads(payloads)
    if payloads != canonical_payloads:
        raise ValueError("gate receipt run payloads are not canonically ordered")
    operator_readbacks = receipt["operator_readbacks"]
    if not isinstance(operator_readbacks, list):
        raise TypeError("gate receipt operator_readbacks is not an array")
    canonical_readbacks = sorted(
        operator_readbacks,
        key=lambda value: (
            str(value.get("readback", {}).get("instance_id", ""))
            if isinstance(value, Mapping) and isinstance(value.get("readback"), Mapping)
            else ""
        ),
    )
    if operator_readbacks != canonical_readbacks:
        raise ValueError("gate receipt operator readbacks are not canonically ordered")
    launch_nonces: set[str] = set()
    for index, record in enumerate(canonical_readbacks):
        if not isinstance(record, Mapping):
            raise TypeError(f"operator attestation[{index}] is not an object")
        readback = record.get("readback")
        if not isinstance(readback, Mapping):
            raise TypeError(f"operator attestation[{index}] readback is not an object")
        tags = readback.get("tags")
        if not isinstance(tags, list):
            raise TypeError(f"operator attestation[{index}] readback tags are not an array")
        launch_nonce = next(
            (
                tag.get("value")
                for tag in tags
                if isinstance(tag, Mapping) and tag.get("key") == "citrees-gate-launch-nonce"
            ),
            None,
        )
        if not isinstance(launch_nonce, str):
            raise ValueError(f"operator attestation[{index}] lacks a gate launch nonce")
        launch_nonces.add(launch_nonce)
    if len(launch_nonces) != 1:
        raise ValueError("operator readbacks do not share one gate launch nonce")
    gate_launch_nonce = next(iter(launch_nonces))
    expected_report = compare_payloads(
        canonical_payloads,
        canonical_readbacks,
        gate_launch_nonce=gate_launch_nonce,
        manifest=manifest,
        receipt_created_at_utc=created_at_utc,
        runtime_contract=runtime_contract,
    )
    if receipt["report"] != expected_report:
        raise ValueError("gate receipt report differs from the embedded run payloads")
    return {
        "schema_version": GATE_RECEIPT_SCHEMA_VERSION,
        "profile": GATE_RECEIPT_PROFILE,
        **expected_bindings,
        "created_at_utc": created_at_utc,
        "run_payloads": canonical_payloads,
        "operator_readbacks": canonical_readbacks,
        "report": expected_report,
    }


def serialize_gate_receipt(
    receipt: dict[str, Any],
    *,
    manifest: RerunManifest,
    runtime_contract: dict[str, Any],
) -> bytes:
    """Serialize one validated gate receipt to canonical JSON bytes."""
    normalized = validate_gate_receipt(
        receipt,
        manifest=manifest,
        runtime_contract=runtime_contract,
    )
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def gate_receipt_sha256(
    receipt: dict[str, Any],
    *,
    manifest: RerunManifest,
    runtime_contract: dict[str, Any],
) -> str:
    """Return the SHA-256 digest of one canonical gate receipt."""
    return hashlib.sha256(
        serialize_gate_receipt(
            receipt,
            manifest=manifest,
            runtime_contract=runtime_contract,
        )
    ).hexdigest()


def parse_gate_receipt(
    payload: bytes,
    *,
    manifest: RerunManifest,
    runtime_contract: dict[str, Any],
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Parse canonical gate-receipt bytes and verify their complete evidence."""
    if not isinstance(payload, bytes):
        raise TypeError("gate receipt payload must be bytes")
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != validate_gate_receipt_sha256(expected_sha256):
        raise ValueError(
            f"gate receipt SHA-256 mismatch: expected {expected_sha256}, observed {digest}"
        )
    try:
        parsed = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid gate receipt JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("gate receipt must be an object")
    canonical = serialize_gate_receipt(
        parsed,
        manifest=manifest,
        runtime_contract=runtime_contract,
    )
    if payload != canonical:
        raise ValueError("gate receipt must use canonical JSON bytes")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    key_parser = subparsers.add_parser(
        "generate-operator-key",
        help="Create the local Ed25519 operator-attestation keypair",
    )
    key_parser.add_argument("--private-key", type=Path, required=True)
    key_parser.add_argument("--public-key", type=Path, required=True)

    freeze_parser = subparsers.add_parser(
        "freeze-runtime",
        help="Probe the EC2 runtime and bind the operator public key",
    )
    freeze_parser.add_argument("--operator-public-key", type=Path, required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Execute one complete r_cforest reproducibility run",
    )
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--runtime-contract", type=Path, required=True)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Collect signed readbacks and create the immutable GO receipt",
    )
    compare_parser.add_argument("--runs", nargs=N_EXPECTED_RUNS, type=Path, required=True)
    compare_parser.add_argument(
        "--operator-profiles",
        nargs=N_EXPECTED_GATE_ACCOUNTS,
        required=True,
    )
    compare_parser.add_argument("--manifest", type=Path, required=True)
    compare_parser.add_argument("--gate-launch-nonce", required=True)
    compare_parser.add_argument("--operator-private-key", type=Path, required=True)
    compare_parser.add_argument("--runtime-contract", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "generate-operator-key":
        public_key = generate_operator_keypair(args.private_key, args.public_key)
        print(json.dumps(public_key, sort_keys=True, separators=(",", ":")))
        return
    if args.command == "freeze-runtime":
        contract = create_runtime_contract(
            _provenance("runtime-freeze"),
            load_operator_public_key(args.operator_public_key),
        )
        print(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return
    manifest = _load_manifest(args.manifest)
    runtime_contract = _load_runtime_contract(args.runtime_contract)
    if args.command == "run":
        payload = run_gate(args.run_id, runtime_contract, manifest)
        print(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return
    payloads = [_load_payload(path) for path in args.runs]
    operator_readbacks = collect_live_operator_readbacks(
        payloads,
        manifest=manifest,
        operator_private_key_pem=load_operator_private_key(args.operator_private_key),
        operator_profiles=args.operator_profiles,
        runtime_contract=runtime_contract,
    )
    receipt = create_gate_receipt(
        payloads,
        operator_readbacks,
        gate_launch_nonce=args.gate_launch_nonce,
        manifest=manifest,
        runtime_contract=runtime_contract,
    )
    sys.stdout.buffer.write(
        serialize_gate_receipt(
            receipt,
            manifest=manifest,
            runtime_contract=runtime_contract,
        )
    )


if __name__ == "__main__":
    main()
