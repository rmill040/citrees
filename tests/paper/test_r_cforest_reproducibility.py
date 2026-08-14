"""Tests for the fixed-family r_cforest reproducibility gate."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import boto3
import numpy as np
import pytest

from paper.benchmark.experiments import r_cforest_reproducibility as gate
from paper.benchmark.pipeline.instance_identity import (
    AwsCallerIdentity,
    InstanceIdentityEvidence,
    validate_instance_identity,
    validate_instance_identity_record,
)
from paper.benchmark.pipeline.manifest import (
    ManifestCell,
    RerunManifest,
    compute_campaign_sha256,
    parse_rerun_manifest,
    partition_rerun_manifest_by_account,
    serialize_rerun_manifest,
)
from paper.benchmark.pipeline.methods import get_full_method_configs
from paper.benchmark.pipeline.operator_attestation import (
    create_operator_attestation,
)
from paper.benchmark.pipeline.stage2 import get_requested_evaluation_k_values
from paper.benchmark.pipeline.types import DatasetIdentity, ExperimentConfig, TaskType
from tests.paper.operator_attestation_fixtures import (
    OPERATOR_PRIVATE_KEY_PEM,
    OPERATOR_PUBLIC_KEY,
)
from tests.paper.test_instance_identity import _create_signer, _sign_document

pytestmark = pytest.mark.paper

CPU_MODEL = "AMD EPYC 7R13 Processor"
INSTANCE_TYPE = "c6a.8xlarge"
OPENSSL_VERSION = "OpenSSL 3.0.13 30 Jan 2024"
HOST_A = "i-aaaaaaaaaaaaaaaaa"
HOST_B = "i-bbbbbbbbbbbbbbbbb"
BOOT_A = "11111111-1111-4111-8111-111111111111"
BOOT_B = "22222222-2222-4222-8222-222222222222"
ACCOUNT_A = "123456789012"
ACCOUNT_B = "210987654321"
FEATURE_COUNTS = (9, 10, 13, 120, 166)
AMI_ID = "ami-" + "a" * 17
ZONE_ID_A = "use1-az1"
ZONE_ID_B = "use1-az2"
GATE_LAUNCH_NONCE = "c" * 32
TASK_DATASET_COUNTS: tuple[tuple[TaskType, int], ...] = (
    ("classification", 24),
    ("regression", 23),
)


def _accept_signature(
    raw_document: bytes,
    rsa2048_signature: bytes,
    certificate_pem: bytes,
) -> None:
    assert raw_document
    assert rsa2048_signature
    assert certificate_pem


@pytest.fixture(autouse=True)
def _accept_fixture_identity_signatures(monkeypatch: pytest.MonkeyPatch) -> None:
    def validate(record):
        return validate_instance_identity_record(
            record,
            signature_verifier=_accept_signature,
        )

    monkeypatch.setattr(gate, "validate_instance_identity_record", validate)


def _instance_caller(account_id: str, instance_id: str) -> AwsCallerIdentity:
    return AwsCallerIdentity(
        account_id=account_id,
        arn=(f"arn:aws:sts::{account_id}:assumed-role/citrees-gate-instance/{instance_id}"),
        user_id=f"AROAGATE:{instance_id}",
    )


def _operator_caller(account_id: str) -> AwsCallerIdentity:
    return AwsCallerIdentity(
        account_id=account_id,
        arn=f"arn:aws:sts::{account_id}:assumed-role/citrees-gate-operator/readback",
        user_id="AROAOPERATOR:readback",
    )


def _identity_evidence(
    *,
    instance_id: str,
    availability_zone: str,
    availability_zone_id: str,
    account_id: str,
    ami_id: str = AMI_ID,
) -> InstanceIdentityEvidence:
    document = {
        "accountId": account_id,
        "architecture": "x86_64",
        "availabilityZone": availability_zone,
        "billingProducts": None,
        "devpayProductCodes": None,
        "imageId": ami_id,
        "instanceId": instance_id,
        "instanceType": INSTANCE_TYPE,
        "kernelId": None,
        "marketplaceProductCodes": None,
        "pendingTime": "2026-08-05T12:34:56Z",
        "privateIp": "10.0.0.10",
        "ramdiskId": None,
        "region": "us-east-1",
        "version": "2017-09-30",
    }
    raw_document = (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    return validate_instance_identity(
        raw_document,
        b"AA==",
        availability_zone_id,
        _instance_caller(account_id, instance_id),
        signature_verifier=_accept_signature,
    )


def _manifest(runtime_contract_sha256: str = "f" * 64) -> RerunManifest:
    cells: list[ManifestCell] = []
    for task, dataset_count in TASK_DATASET_COUNTS:
        configurations = get_full_method_configs(["r_cforest"], task)
        for dataset_index in range(dataset_count):
            dataset = f"{task}_fixture_{dataset_index}"
            identity = DatasetIdentity(
                sha256=hashlib.sha256(f"{task}:{dataset}".encode()).hexdigest(),
                n_samples=1000,
                n_features=FEATURE_COUNTS[dataset_index % len(FEATURE_COUNTS)],
            )
            for configuration in configurations:
                for seed in gate.EXPECTED_SEEDS:
                    cells.append(
                        ManifestCell(
                            config=ExperimentConfig(
                                method=configuration,
                                dataset=dataset,
                                seed=seed,
                                task=task,
                                dataset_identity=identity,
                            ),
                            target_aws_account_id=ACCOUNT_A,
                            dataset_source="real",
                            rerun_reason=gate.REPLACEMENT_REASON,
                            historically_omitted=(task == "classification" and dataset_index == 0),
                            stage1_required=True,
                            stage2_required=True,
                        )
                    )
    other_method = get_full_method_configs(["rf"], "classification")[0]
    other_identity = cells[0].config.dataset_identity
    cells.append(
        ManifestCell(
            config=ExperimentConfig(
                method=other_method,
                dataset="classification_fixture_0",
                seed=0,
                task="classification",
                dataset_identity=other_identity,
            ),
            target_aws_account_id=ACCOUNT_A,
            dataset_source="real",
            rerun_reason="unrelated_campaign_cell",
            historically_omitted=False,
            stage1_required=True,
            stage2_required=True,
        )
    )
    frozen_cells = tuple(cells)
    return RerunManifest(
        sha256="a" * 64,
        campaign_sha256=compute_campaign_sha256(
            frozen_cells,
            runtime_contract_sha256=runtime_contract_sha256,
        ),
        runtime_contract_sha256=runtime_contract_sha256,
        cells=frozen_cells,
    )


def _rankings(n_features: int) -> list[list[int]]:
    values = list(range(n_features))
    return [values[fold:] + values[:fold] for fold in range(gate.N_FOLDS)]


def _configuration_result(
    cell: ManifestCell,
) -> dict[str, Any]:
    configuration = cell.config.method
    n_features = cell.config.dataset_identity.n_features
    k_values = get_requested_evaluation_k_values(n_features)
    return {
        "method": configuration.name,
        "params": configuration.params_dict,
        "elapsed_seconds": 1.0,
        "fold_cpu_affinity": [
            list(cpu_ids) for cpu_ids in gate.partition_cpu_ids(tuple(range(32)), gate.N_FOLDS)
        ],
        **gate.summarize_rankings(
            _rankings(n_features),
            n_features=n_features,
            k_values=k_values,
        ),
    }


def _dataset_result(cells: tuple[ManifestCell, ...]) -> dict[str, Any]:
    first_cell = cells[0]
    config = first_cell.config
    return {
        "dataset": config.dataset,
        "dataset_source": first_cell.dataset_source,
        "task": config.task,
        "identity": gate._identity_payload(first_cell),
        "k_values": get_requested_evaluation_k_values(config.dataset_identity.n_features),
        "configurations": {
            gate._configuration_key(cell): {
                "seed": cell.config.seed,
                **_configuration_result(cell),
            }
            for cell in cells
        },
    }


def _provenance(
    run_id: str,
    instance_id: str,
    availability_zone: str,
    availability_zone_id: str,
    aws_account_id: str,
    boot_id: str,
    process_start_ticks: int,
    *,
    ami_id: str = AMI_ID,
) -> dict[str, Any]:
    evidence = _identity_evidence(
        instance_id=instance_id,
        availability_zone=availability_zone,
        availability_zone_id=availability_zone_id,
        account_id=aws_account_id,
        ami_id=ami_id,
    )
    return {
        "ami_id": ami_id,
        "architecture": "x86_64",
        "availability_zone": availability_zone,
        "availability_zone_id": availability_zone_id,
        "aws_account_id": aws_account_id,
        "boot_id": boot_id,
        "container_image": (
            f"{aws_account_id}.dkr.ecr.us-east-1.amazonaws.com/"
            f"citrees-{aws_account_id}@sha256:" + "b" * 64
        ),
        "cpu_affinity": list(range(32)),
        "cpu_model": CPU_MODEL,
        "git_sha": "c" * 40,
        "hostname": instance_id,
        "instance_identity": evidence.to_record(),
        "instance_id": instance_id,
        "instance_type": INSTANCE_TYPE,
        "kernel": "6.1.0-fixture",
        "logical_cpus": 32,
        "machine": "x86_64",
        "microcode": "0x1000065",
        "openssl_version": OPENSSL_VERSION,
        "os_release": {"ID": "amzn", "VERSION_ID": "2023"},
        "process_id": 1,
        "process_start_ticks": process_start_ticks,
        "python_libraries": {name: "1.0" for name in gate.PYTHON_LIBRARY_NAMES},
        "r_numerical_libraries": {
            "blas": "/usr/local/lib/R/lib/libRblas.so",
            "lapack": "/usr/local/lib/R/lib/libRlapack.so",
        },
        "r_selection_timeout_seconds": gate.STAGE1_SELECTION_TIMEOUT_SECONDS,
        "r_runtime": {
            "r": "R version 4.5.2",
            "partykit": "1.2.24",
            "libcoin": "1.0.10",
            "mvtnorm": "1.3.3",
        },
        "run_id": run_id,
        "script_sha256": "d" * 64,
        "thread_environment": {
            name: gate.EXPECTED_THREAD_ENVIRONMENT[name] for name in gate.THREAD_ENVIRONMENT
        },
        "threadpools": [
            {
                "architecture": "Zen",
                "filepath": "/app/.venv/lib/libopenblas.so",
                "internal_api": "openblas",
                "num_threads": 1,
                "prefix": "libopenblas",
                "threading_layer": "pthreads",
                "user_api": "blas",
                "version": "0.3.30",
            }
        ],
    }


def _payload(
    provenance: dict[str, Any],
    runtime_contract: dict[str, Any],
    manifest: RerunManifest,
) -> dict[str, Any]:
    inventory = gate._gate_inventory(manifest)
    return {
        "schema_version": gate.SCHEMA_VERSION,
        "profile": "r_cforest_reproducibility",
        "target_aws_account_ids": [ACCOUNT_A],
        "campaign_sha256": manifest.campaign_sha256,
        "manifest_sha256": manifest.sha256,
        "runtime_contract_sha256": gate.runtime_contract_sha256(runtime_contract),
        "provenance": provenance,
        "results": {
            task: {dataset: _dataset_result(cells) for dataset, cells in task_inventory.items()}
            for task, task_inventory in inventory.items()
        },
        "elapsed_seconds": 2.0,
    }


def _payloads() -> tuple[dict[str, Any], RerunManifest, list[dict[str, Any]]]:
    provenances = [
        _provenance(
            "arc-a-repeat-1",
            HOST_A,
            "us-east-1a",
            ZONE_ID_A,
            ACCOUNT_A,
            BOOT_A,
            100,
        ),
        _provenance(
            "arc-a-repeat-2",
            HOST_A,
            "us-east-1a",
            ZONE_ID_A,
            ACCOUNT_A,
            BOOT_A,
            200,
        ),
        _provenance(
            "arc-b-repeat-1",
            HOST_B,
            "us-east-1b",
            ZONE_ID_B,
            ACCOUNT_A,
            BOOT_B,
            100,
        ),
        _provenance(
            "arc-b-repeat-2",
            HOST_B,
            "us-east-1b",
            ZONE_ID_B,
            ACCOUNT_A,
            BOOT_B,
            200,
        ),
    ]
    runtime_contract = gate.create_runtime_contract(provenances[0], OPERATOR_PUBLIC_KEY)
    manifest = _manifest(gate.runtime_contract_sha256(runtime_contract))
    return (
        runtime_contract,
        manifest,
        [_payload(provenance, runtime_contract, manifest) for provenance in provenances],
    )


def _gate_tags(
    runtime_contract: dict[str, Any],
    host_slot: str,
) -> list[dict[str, str]]:
    runtime = runtime_contract["runtime"]
    source_git_sha = str(runtime["git_sha"])
    image_digest = str(runtime["container_image_digest"])
    return [
        {"key": "Name", "value": "citrees-r-cforest-gate"},
        {
            "key": "citrees-artifact-prefix",
            "value": gate.gate_output_prefix(
                source_git_sha,
                image_digest,
                GATE_LAUNCH_NONCE,
            ),
        },
        {
            "key": "citrees-gate-identity",
            "value": gate.gate_launch_identity(
                source_git_sha,
                image_digest,
                GATE_LAUNCH_NONCE,
            ),
        },
        {"key": "citrees-gate-launch-nonce", "value": GATE_LAUNCH_NONCE},
        {"key": "citrees-host-slot", "value": host_slot},
        {"key": "citrees-image-digest", "value": image_digest},
        {"key": "citrees-market", "value": gate.GATE_MARKET},
        {
            "key": "citrees-role",
            "value": "r-cforest-reproducibility-gate",
        },
        {"key": "citrees-source-git-sha", "value": source_git_sha},
    ]


def _operator_readbacks(
    payloads: list[dict[str, Any]],
    runtime_contract: dict[str, Any],
    manifest: RerunManifest,
    *,
    observed_at_utc: str | None = None,
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    host_slot_by_instance = {HOST_A: "arc-a", HOST_B: "arc-b"}
    for payload in payloads:
        provenance = payload["provenance"]
        instance_id = provenance["instance_id"]
        if instance_id in records:
            continue
        evidence = validate_instance_identity_record(
            provenance["instance_identity"],
            signature_verifier=_accept_signature,
        )
        signed = evidence.identity
        records[instance_id] = {
            "architecture": signed.architecture,
            "availability_zone": signed.availability_zone,
            "availability_zone_id": evidence.availability_zone_id,
            "iam_instance_profile_arn": (
                f"arn:aws:iam::{signed.account_id}:instance-profile/citrees-gate-instance"
            ),
            "iam_role_arn": f"arn:aws:iam::{signed.account_id}:role/citrees-gate-instance",
            "image_id": signed.image_id,
            "instance_id": signed.instance_id,
            "instance_lifecycle": gate.GATE_MARKET,
            "instance_type": signed.instance_type,
            "operator_identity": _operator_caller(signed.account_id).to_record(),
            "owner_account_id": signed.account_id,
            "region": signed.region,
            "state": "running",
            "tags": _gate_tags(
                runtime_contract,
                host_slot_by_instance[instance_id],
            ),
        }
    return [
        _attest_operator_readback(
            records[instance_id],
            payloads,
            runtime_contract,
            manifest,
            observed_at_utc=observed_at_utc,
        )
        for instance_id in sorted(records)
    ]


def _attest_operator_readback(
    record: dict[str, Any],
    payloads: list[dict[str, Any]],
    runtime_contract: dict[str, Any],
    manifest: RerunManifest,
    *,
    observed_at_utc: str | None = None,
) -> dict[str, Any]:
    return create_operator_attestation(
        record,
        campaign_sha256=manifest.campaign_sha256,
        manifest_sha256=manifest.sha256,
        observed_at_utc=gate.utc_timestamp() if observed_at_utc is None else observed_at_utc,
        private_key_pem=OPERATOR_PRIVATE_KEY_PEM,
        public_key=runtime_contract["operator_attestation_public_key"],
        run_payload_sha256s=gate._run_payload_sha256s(payloads),
        runtime_contract_sha256=gate.runtime_contract_sha256(runtime_contract),
    )


def _compare(
    runtime_contract: dict[str, Any],
    manifest: RerunManifest,
    payloads: list[dict[str, Any]],
    operator_readbacks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return gate.compare_payloads(
        payloads,
        (
            _operator_readbacks(payloads, runtime_contract, manifest)
            if operator_readbacks is None
            else operator_readbacks
        ),
        gate_launch_nonce=GATE_LAUNCH_NONCE,
        manifest=manifest,
        runtime_contract=runtime_contract,
    )


def test_gate_attempt_identity_and_prefix_bind_fresh_nonce() -> None:
    source_git_sha = "a" * 40
    image_digest = "sha256:" + "b" * 64

    assert gate.gate_launch_identity(
        source_git_sha,
        image_digest,
        GATE_LAUNCH_NONCE,
    ) != gate.gate_launch_identity(
        source_git_sha,
        image_digest,
        "d" * 32,
    )
    assert gate.gate_output_prefix(
        source_git_sha,
        image_digest,
        GATE_LAUNCH_NONCE,
    ).endswith(f"attempt-{GATE_LAUNCH_NONCE}-arc-on-demand2")


@pytest.mark.parametrize(
    ("source_git_sha", "image_digest", "launch_nonce"),
    [
        ("invalid", "sha256:" + "b" * 64, "c" * 32),
        ("a" * 40, "invalid", "c" * 32),
        ("a" * 40, "sha256:" + "b" * 64, "invalid"),
    ],
)
def test_gate_attempt_identity_rejects_invalid_inputs(
    source_git_sha: str,
    image_digest: str,
    launch_nonce: str,
) -> None:
    with pytest.raises(ValueError):
        gate.gate_launch_identity(source_git_sha, image_digest, launch_nonce)


def test_gate_validates_the_full_scope_and_executes_a_stratified_panel() -> None:
    manifest = _manifest()
    replacement_inventory = gate._replacement_inventory(manifest)
    gate_inventory = gate._gate_inventory(manifest)

    replacement_cells = [cell for cell in manifest.cells if cell.config.method.name == "r_cforest"]
    assert len(manifest.cells) == 941
    assert len(replacement_cells) == 940
    assert {cell.target_aws_account_id for cell in replacement_cells} == {ACCOUNT_A}
    assert sum(cell.historically_omitted for cell in replacement_cells) == 20
    assert sum(len(datasets) for datasets in replacement_inventory.values()) == 47
    assert set(replacement_inventory) == {"classification", "regression"}
    for task, datasets in replacement_inventory.items():
        configurations = get_full_method_configs(["r_cforest"], task)
        assert len(configurations) == 4
        assert {
            (config.params_dict["testtype"], config.params_dict["replace"])
            for config in configurations
        } == {
            ("Bonferroni", False),
            ("Bonferroni", True),
            ("MonteCarlo", False),
            ("MonteCarlo", True),
        }
        for cells in datasets.values():
            assert len(cells) == 20
            assert {cell.config.seed for cell in cells} == set(gate.EXPECTED_SEEDS)

    assert sum(len(datasets) for datasets in gate_inventory.values()) == 4
    assert (
        sum(len(cells) for datasets in gate_inventory.values() for cells in datasets.values()) == 8
    )
    for task, datasets in gate_inventory.items():
        assert len(datasets) == 2
        assert all(cells[0].dataset_source == "real" for cells in datasets.values())
        assert {len(cells) for cells in datasets.values()} == {2}
        real_datasets = {
            dataset: cells
            for dataset, cells in replacement_inventory[task].items()
            if cells[0].dataset_source == "real"
        }
        expected_compact = min(
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
        expected_high_dimensional = min(
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
        assert set(datasets) == {expected_compact, expected_high_dimensional}
        selected_cells = [cell for cells in datasets.values() for cell in cells]
        assert {
            (
                cell.config.method.params_dict["testtype"],
                cell.config.method.params_dict["replace"],
            )
            for cell in selected_cells
        } == {
            ("Bonferroni", False),
            ("Bonferroni", True),
            ("MonteCarlo", False),
            ("MonteCarlo", True),
        }
        assert {cell.config.seed for cell in selected_cells} == {
            gate.EXPECTED_SEEDS[0],
            gate.EXPECTED_SEEDS[-1],
        }


def test_gate_accepts_a_complete_inventory_with_stage1_recovery_mask() -> None:
    manifest = _manifest()
    masked_cells = tuple(
        replace(
            cell,
            stage1_required=(
                index % 7 == 0 if cell.config.method.name == "r_cforest" else cell.stage1_required
            ),
        )
        for index, cell in enumerate(manifest.cells)
    )
    masked_manifest = replace(
        manifest,
        campaign_sha256=compute_campaign_sha256(
            masked_cells,
            runtime_contract_sha256=manifest.runtime_contract_sha256,
        ),
        cells=masked_cells,
    )

    inventory = gate._replacement_inventory(masked_manifest)
    replacement_cells = tuple(
        cell for cell in masked_cells if cell.config.method.name == "r_cforest"
    )

    assert any(cell.stage1_required for cell in replacement_cells)
    assert any(not cell.stage1_required for cell in replacement_cells)
    assert (
        sum(len(cells) for datasets in inventory.values() for cells in datasets.values())
        == gate.EXPECTED_REPLACEMENT_CELLS
    )


def test_gate_accepts_a_complete_inventory_with_execution_exclusions() -> None:
    manifest = _manifest()
    excluded_cells = tuple(
        replace(
            cell,
            stage1_required=False,
            stage2_required=False,
        )
        if index % 11 == 0 and cell.config.method.name == "r_cforest"
        else cell
        for index, cell in enumerate(manifest.cells)
    )
    excluded_manifest = replace(
        manifest,
        campaign_sha256=compute_campaign_sha256(
            excluded_cells,
            runtime_contract_sha256=manifest.runtime_contract_sha256,
        ),
        cells=excluded_cells,
    )

    inventory = gate._replacement_inventory(excluded_manifest)
    replacement_cells = tuple(
        cell for cell in excluded_cells if cell.config.method.name == "r_cforest"
    )

    assert any(not cell.stage1_required and not cell.stage2_required for cell in replacement_cells)
    assert (
        sum(len(cells) for datasets in inventory.values() for cells in datasets.values())
        == gate.EXPECTED_REPLACEMENT_CELLS
    )


def test_gate_rejects_changed_replacement_reason() -> None:
    manifest = _manifest()
    changed_one = False
    cells: list[ManifestCell] = []
    for cell in manifest.cells:
        if not changed_one and cell.config.method.name == "r_cforest":
            cell = replace(cell, rerun_reason="different")
            changed_one = True
        cells.append(cell)
    frozen_cells = tuple(cells)
    changed_manifest = replace(
        manifest,
        campaign_sha256=compute_campaign_sha256(
            frozen_cells,
            runtime_contract_sha256=manifest.runtime_contract_sha256,
        ),
        cells=frozen_cells,
    )

    with pytest.raises(ValueError, match="cell contract differs"):
        gate._replacement_inventory(changed_manifest)


def test_linux_process_identity_uses_boot_id_and_start_ticks(tmp_path: Path) -> None:
    boot_id = tmp_path / "boot_id"
    process_stat = tmp_path / "stat"
    boot_id.write_text(BOOT_A + "\n", encoding="ascii")
    process_stat.write_text(
        "1 (python worker) S " + " ".join(["0"] * 18 + ["987654"]) + "\n",
        encoding="ascii",
    )

    assert gate._linux_process_identity(
        boot_id_path=boot_id,
        stat_path=process_stat,
    ) == (BOOT_A, 987654)


def test_thread_environment_requires_every_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in gate.THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, gate.EXPECTED_THREAD_ENVIRONMENT[name])
    assert gate._thread_environment() == gate.EXPECTED_THREAD_ENVIRONMENT

    monkeypatch.delenv(next(iter(gate.THREAD_ENVIRONMENT)))
    with pytest.raises(RuntimeError, match="thread environment"):
        gate._thread_environment()


def test_openssl_version_records_the_exact_cli_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def check_output(command: list[str], **kwargs: Any) -> str:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return OPENSSL_VERSION + "\n"

    monkeypatch.setattr(gate.subprocess, "check_output", check_output)

    assert gate._openssl_version() == OPENSSL_VERSION
    assert observed == {
        "command": ["openssl", "version"],
        "kwargs": {
            "stderr": gate.subprocess.STDOUT,
            "text": True,
            "timeout": gate.OPENSSL_VERSION_TIMEOUT_SECONDS,
        },
    }


@pytest.mark.parametrize(
    "output",
    [
        "",
        "unknown\n",
        OPENSSL_VERSION + "\nextra output\n",
        " " + OPENSSL_VERSION + "\n",
    ],
)
def test_openssl_version_rejects_noncanonical_output(
    output: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gate.subprocess,
        "check_output",
        lambda *args, **kwargs: output,
    )

    with pytest.raises(RuntimeError, match="OpenSSL CLI version"):
        gate._openssl_version()


def test_openssl_version_fails_closed_when_the_cli_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*args: Any, **kwargs: Any) -> str:
        raise FileNotFoundError("openssl")

    monkeypatch.setattr(gate.subprocess, "check_output", missing)

    with pytest.raises(RuntimeError, match="cannot execute openssl version"):
        gate._openssl_version()


def test_provenance_derives_ec2_scope_from_signed_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _identity_evidence(
        instance_id=HOST_A,
        availability_zone="us-east-1a",
        availability_zone_id=ZONE_ID_A,
        account_id=ACCOUNT_A,
    )
    monkeypatch.setenv(
        "CITREES_IMAGE_URI",
        f"{ACCOUNT_A}.dkr.ecr.us-east-1.amazonaws.com/citrees@sha256:" + "b" * 64,
    )
    monkeypatch.setenv("GIT_SHA", "c" * 40)
    monkeypatch.setenv("AWS_ACCOUNT_ID", ACCOUNT_B)
    monkeypatch.setenv("EC2_INSTANCE_ID", HOST_B)
    monkeypatch.setattr(gate, "_linux_process_identity", lambda: (BOOT_A, 100))
    monkeypatch.setattr(
        gate, "_cpu_field", lambda field: CPU_MODEL if field == "model name" else "0x1"
    )
    monkeypatch.setattr(gate, "_os_release", lambda: {"ID": "amzn", "VERSION_ID": "2023"})
    monkeypatch.setattr(
        gate,
        "_python_libraries",
        lambda: {name: "1.0" for name in gate.PYTHON_LIBRARY_NAMES},
    )
    monkeypatch.setattr(
        gate,
        "get_r_runtime_versions",
        lambda: {
            "r": "R version 4.5.2",
            "partykit": "1.2.24",
            "libcoin": "1.0.10",
            "mvtnorm": "1.3.3",
        },
    )
    monkeypatch.setattr(
        gate,
        "_r_numerical_libraries",
        lambda: {"blas": "libRblas.so", "lapack": "libRlapack.so"},
    )
    monkeypatch.setattr(gate, "_openssl_version", lambda: OPENSSL_VERSION)
    monkeypatch.setattr(
        gate,
        "_thread_environment",
        lambda: dict(gate.EXPECTED_THREAD_ENVIRONMENT),
    )
    monkeypatch.setattr(
        gate,
        "_canonical_threadpools",
        lambda: [
            {
                "filepath": "/lib/libopenblas.so",
                "internal_api": "openblas",
                "num_threads": 1,
                "prefix": "libopenblas",
                "user_api": "blas",
            }
        ],
    )
    monkeypatch.setattr(gate.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(gate, "get_available_cpu_ids", lambda: tuple(range(32)))

    provenance = gate._provenance("signed-run", instance_identity=evidence)

    assert gate.REQUIRED_GATE_ENVIRONMENT == ("CITREES_IMAGE_URI", "GIT_SHA")
    assert provenance["aws_account_id"] == ACCOUNT_A
    assert provenance["instance_id"] == HOST_A
    assert provenance["availability_zone_id"] == ZONE_ID_A
    assert provenance["ami_id"] == AMI_ID
    assert provenance["cpu_affinity"] == list(range(32))
    assert provenance["logical_cpus"] == 32
    assert provenance["openssl_version"] == OPENSSL_VERSION
    assert provenance["r_selection_timeout_seconds"] == gate.STAGE1_SELECTION_TIMEOUT_SECONDS
    assert provenance["instance_identity"] == evidence.to_record()


def test_gate_cli_requires_frozen_public_and_local_private_keys() -> None:
    freeze = gate._parser().parse_args(
        [
            "freeze-runtime",
            "--operator-public-key",
            "operator-public.pem",
        ]
    )
    compare = gate._parser().parse_args(
        [
            "compare",
            "--runs",
            "run-1.json",
            "run-2.json",
            "run-3.json",
            "run-4.json",
            "--operator-profiles",
            "profile-arc",
            "--gate-launch-nonce",
            GATE_LAUNCH_NONCE,
            "--operator-private-key",
            "operator-private.pem",
            "--manifest",
            "manifest.csv",
            "--runtime-contract",
            "runtime.json",
        ]
    )

    assert freeze.operator_public_key == Path("operator-public.pem")
    assert compare.gate_launch_nonce == GATE_LAUNCH_NONCE
    assert compare.operator_private_key == Path("operator-private.pem")


def test_payload_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    payload_path = tmp_path / "duplicate.json"
    payload_path.write_text(
        '{"provenance":{},"provenance":{}}',
        encoding="ascii",
    )

    with pytest.raises(RuntimeError, match="duplicate field 'provenance'"):
        gate._load_payload(payload_path)


def test_compare_cli_requires_four_runs_and_one_operator_profile() -> None:
    args = gate._parser().parse_args(
        [
            "compare",
            "--runs",
            "run-1.json",
            "run-2.json",
            "run-3.json",
            "run-4.json",
            "--operator-profiles",
            "profile-arc",
            "--gate-launch-nonce",
            GATE_LAUNCH_NONCE,
            "--operator-private-key",
            "operator-private.pem",
            "--manifest",
            "manifest.csv",
            "--runtime-contract",
            "runtime.json",
        ]
    )

    assert len(args.runs) == gate.N_EXPECTED_RUNS
    assert len(args.operator_profiles) == gate.N_EXPECTED_GATE_ACCOUNTS


def test_runtime_contract_enforcement_rejects_live_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_contract, _manifest_value, _payload_values = _payloads()
    expected = gate.runtime_contract_sha256(runtime_contract)
    monkeypatch.setattr(
        gate,
        "running_runtime_contract_sha256",
        lambda contract: expected,
    )
    gate.require_running_runtime_contract(runtime_contract)

    monkeypatch.setattr(
        gate,
        "running_runtime_contract_sha256",
        lambda contract: "b" * 64,
    )
    with pytest.raises(RuntimeError, match="differs from the gate-approved"):
        gate.require_running_runtime_contract(runtime_contract)

    with pytest.raises(ValueError, match="fields differ"):
        gate.require_running_runtime_contract({})


def test_runtime_contract_is_content_addressed_and_excludes_host_identity() -> None:
    runtime_contract, _manifest_value, payloads = _payloads()

    assert len(gate.runtime_contract_sha256(runtime_contract)) == 64
    assert set(runtime_contract["runtime"]) == gate.RUNTIME_PROVENANCE_FIELDS
    assert "instance_id" not in runtime_contract["runtime"]
    assert "boot_id" not in runtime_contract["runtime"]
    assert "container_image" not in runtime_contract["runtime"]
    assert "script_sha256" not in runtime_contract["runtime"]
    assert runtime_contract["runtime"]["openssl_version"] == OPENSSL_VERSION
    assert runtime_contract["runtime"]["container_image_digest"] == "sha256:" + "b" * 64
    assert payloads[0]["runtime_contract_sha256"] == gate.runtime_contract_sha256(runtime_contract)


def test_runtime_contract_rejects_cpu_affinity_drift() -> None:
    runtime_contract, manifest, payloads = _payloads()
    payloads[0]["provenance"]["cpu_affinity"] = list(range(1, 33))

    with pytest.raises(ValueError, match=r"frozen runtime: \['cpu_affinity'\]"):
        _compare(runtime_contract, manifest, payloads)


def test_openssl_provenance_is_required_and_runtime_bound() -> None:
    runtime_contract, manifest, payloads = _payloads()

    missing = copy.deepcopy(payloads)
    missing[0]["provenance"].pop("openssl_version")
    with pytest.raises(ValueError, match=r"missing=\['openssl_version'\]"):
        _compare(runtime_contract, manifest, missing)

    mismatched = copy.deepcopy(payloads)
    mismatched[2]["provenance"]["openssl_version"] = "OpenSSL 3.0.14 4 Jun 2024"
    with pytest.raises(ValueError, match=r"frozen runtime: \['openssl_version'\]"):
        _compare(runtime_contract, manifest, mismatched)

    missing_contract_field = copy.deepcopy(runtime_contract)
    missing_contract_field["runtime"].pop("openssl_version")
    with pytest.raises(ValueError, match=r"missing=\['openssl_version'\]"):
        gate.runtime_contract_sha256(missing_contract_field)


def test_openssl_provenance_requires_exact_cross_host_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_contract, manifest, payloads = _payloads()
    payloads[2]["provenance"]["openssl_version"] = "OpenSSL 3.0.14 4 Jun 2024"
    monkeypatch.setattr(gate, "_require_runtime_match", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="gate provenance differs for openssl_version"):
        _compare(runtime_contract, manifest, payloads)


def test_gate_schema_versions_reject_prior_evidence() -> None:
    assert gate.RUNTIME_CONTRACT_SCHEMA_VERSION == 6
    assert gate.SCHEMA_VERSION == 7
    assert gate.GATE_RECEIPT_SCHEMA_VERSION == 8

    runtime_contract, manifest, payloads = _payloads()

    old_contract = copy.deepcopy(runtime_contract)
    old_contract["schema_version"] = 4
    with pytest.raises(ValueError, match="unexpected schema version"):
        gate.validate_runtime_contract(old_contract)

    old_payloads = copy.deepcopy(payloads)
    old_payloads[0]["schema_version"] = gate.SCHEMA_VERSION - 1
    with pytest.raises(ValueError, match="unexpected schema version"):
        _compare(runtime_contract, manifest, old_payloads)

    receipt = gate.create_gate_receipt(
        payloads,
        _operator_readbacks(payloads, runtime_contract, manifest),
        gate_launch_nonce=GATE_LAUNCH_NONCE,
        manifest=manifest,
        runtime_contract=runtime_contract,
    )
    receipt["schema_version"] = gate.GATE_RECEIPT_SCHEMA_VERSION - 1
    with pytest.raises(ValueError, match="unexpected schema version"):
        gate.serialize_gate_receipt(
            receipt,
            manifest=manifest,
            runtime_contract=runtime_contract,
        )


def test_compare_accepts_two_fresh_pid_one_processes_on_two_hosts() -> None:
    runtime_contract, manifest, payloads = _payloads()

    report = _compare(runtime_contract, manifest, payloads)
    expected_selected_sets = sum(
        len(get_requested_evaluation_k_values(cells[0].config.dataset_identity.n_features))
        * len(cells)
        * gate.N_FOLDS
        for datasets in gate._gate_inventory(manifest).values()
        for cells in datasets.values()
    )

    assert report["status"] == "GO"
    assert report["process_incarnations"] == 4
    assert report["dataset_task_pairs"] == 4
    assert report["executed_cells"] == 8
    assert report["fold_rankings"] == 40
    assert report["selected_sets"] == expected_selected_sets
    assert report["aws_account_ids"] == [ACCOUNT_A]
    assert report["availability_zones"] == ["us-east-1a", "us-east-1b"]
    assert report["availability_zone_ids"] == [ZONE_ID_A, ZONE_ID_B]
    assert report["operator_readbacks"] == 2
    assert report["instance_type"] == INSTANCE_TYPE
    assert report["cpu_model"] == CPU_MODEL


@pytest.mark.parametrize(
    "run_id",
    [
        "arc-a-repeat-3",
        "arc-c-repeat-1",
        "host-a-run-1",
    ],
)
def test_compare_rejects_run_ids_outside_exact_arc_host_repeats(
    run_id: str,
) -> None:
    runtime_contract, manifest, payloads = _payloads()
    payloads[0]["provenance"]["run_id"] = run_id

    with pytest.raises(
        ValueError,
        match="run IDs do not cover the exact Arc host slots and repeats",
    ):
        _compare(runtime_contract, manifest, payloads)


@pytest.mark.parametrize(
    "target_aws_account_ids",
    [
        [],
        [ACCOUNT_B],
        [ACCOUNT_A, ACCOUNT_B],
        [ACCOUNT_A, ACCOUNT_A],
    ],
)
def test_compare_rejects_invalid_target_account_declarations(
    target_aws_account_ids: list[str],
) -> None:
    runtime_contract, manifest, payloads = _payloads()
    payloads[0]["target_aws_account_ids"] = target_aws_account_ids

    with pytest.raises(ValueError, match="invalid target_aws_account_ids"):
        _compare(runtime_contract, manifest, payloads)


def test_compare_uses_physical_zone_ids_not_zone_names() -> None:
    runtime_contract, manifest, payloads = _payloads()
    for index in (2, 3):
        old = payloads[index]["provenance"]
        payloads[index]["provenance"] = _provenance(
            old["run_id"],
            HOST_B,
            "us-east-1b",
            ZONE_ID_A,
            ACCOUNT_A,
            old["boot_id"],
            old["process_start_ticks"],
        )

    with pytest.raises(ValueError, match="physical availability-zone IDs"):
        _compare(runtime_contract, manifest, payloads)


def test_live_operator_readbacks_cover_both_hosts_in_one_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_contract, manifest, payloads = _payloads()
    host_scope = {
        HOST_A: ("us-east-1a", ZONE_ID_A),
        HOST_B: ("us-east-1b", ZONE_ID_B),
    }
    observed_profiles: list[str] = []

    class StsClient:
        def __init__(self, account_id: str) -> None:
            self.account_id = account_id

        def get_caller_identity(self) -> dict[str, Any]:
            identity = _operator_caller(self.account_id)
            return {
                "Account": identity.account_id,
                "Arn": identity.arn,
                "UserId": identity.user_id,
                "ResponseMetadata": {"HTTPStatusCode": 200},
            }

    class Ec2Client:
        def __init__(self, account_id: str) -> None:
            self.account_id = account_id

        def describe_instances(self, **kwargs: Any) -> dict[str, Any]:
            instance_ids = kwargs["InstanceIds"]
            assert len(instance_ids) == 1
            instance_id = instance_ids[0]
            availability_zone, _zone_id = host_scope[instance_id]
            host_slot = {HOST_A: "arc-a", HOST_B: "arc-b"}[instance_id]
            return {
                "Reservations": [
                    {
                        "OwnerId": self.account_id,
                        "Instances": [
                            {
                                "Architecture": "x86_64",
                                "ImageId": AMI_ID,
                                "IamInstanceProfile": {
                                    "Arn": (
                                        f"arn:aws:iam::{self.account_id}:"
                                        "instance-profile/citrees-gate-instance"
                                    ),
                                    "Id": "AIPAGATE",
                                },
                                "InstanceId": instance_id,
                                "InstanceType": INSTANCE_TYPE,
                                "Placement": {"AvailabilityZone": availability_zone},
                                "State": {"Name": "running"},
                                "Tags": [
                                    {"Key": tag["key"], "Value": tag["value"]}
                                    for tag in _gate_tags(
                                        runtime_contract,
                                        host_slot,
                                    )
                                ],
                            }
                        ],
                    }
                ],
                "ResponseMetadata": {"HTTPStatusCode": 200},
            }

        def describe_availability_zones(self, **kwargs: Any) -> dict[str, Any]:
            zone_names = kwargs["ZoneNames"]
            assert len(zone_names) == 1
            availability_zone = zone_names[0]
            zone_id = next(
                zone_id
                for candidate_zone, zone_id in host_scope.values()
                if candidate_zone == availability_zone
            )
            return {
                "AvailabilityZones": [
                    {
                        "ZoneName": availability_zone,
                        "ZoneId": zone_id,
                        "RegionName": "us-east-1",
                    }
                ],
                "ResponseMetadata": {"HTTPStatusCode": 200},
            }

    class IamClient:
        def __init__(self, account_id: str) -> None:
            self.account_id = account_id

        def get_instance_profile(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs == {"InstanceProfileName": "citrees-gate-instance"}
            return {
                "InstanceProfile": {
                    "Arn": (
                        f"arn:aws:iam::{self.account_id}:instance-profile/citrees-gate-instance"
                    ),
                    "Roles": [
                        {"Arn": (f"arn:aws:iam::{self.account_id}:role/citrees-gate-instance")}
                    ],
                },
                "ResponseMetadata": {"HTTPStatusCode": 200},
            }

    class Session:
        def __init__(self, *, profile_name: str, region_name: str) -> None:
            assert region_name == "us-east-1"
            assert profile_name == "profile-arc"
            observed_profiles.append(profile_name)
            self.account_id = ACCOUNT_A

        def client(self, service: str):
            if service == "sts":
                return StsClient(self.account_id)
            if service == "ec2":
                return Ec2Client(self.account_id)
            if service == "iam":
                return IamClient(self.account_id)
            raise AssertionError(f"unexpected service {service}")

    monkeypatch.setattr(boto3, "Session", Session)

    readbacks = gate.collect_live_operator_readbacks(
        payloads,
        manifest=manifest,
        operator_private_key_pem=OPERATOR_PRIVATE_KEY_PEM,
        operator_profiles=("profile-arc",),
        runtime_contract=runtime_contract,
    )

    assert observed_profiles == ["profile-arc"]
    assert {record["readback"]["instance_id"] for record in readbacks} == {HOST_A, HOST_B}
    assert {record["readback"]["availability_zone_id"] for record in readbacks} == {
        ZONE_ID_A,
        ZONE_ID_B,
    }
    assert {record["readback"]["instance_lifecycle"] for record in readbacks} == {gate.GATE_MARKET}


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "state",
        "instance_lifecycle",
        "market_tag",
        "artifact_prefix_tag",
        "gate_identity_tag",
        "launch_nonce_tag",
        "host_slot_tag",
        "image_digest_tag",
        "source_git_sha_tag",
        "operator",
        "same_role",
    ],
)
def test_compare_requires_two_independent_contemporaneous_operator_readbacks(
    mutation: str,
) -> None:
    runtime_contract, manifest, payloads = _payloads()
    readbacks = _operator_readbacks(payloads, runtime_contract, manifest)
    if mutation == "missing":
        readbacks.pop()
    elif mutation == "duplicate":
        readbacks[1] = copy.deepcopy(readbacks[0])
    else:
        readback = copy.deepcopy(readbacks[0]["readback"])
        if mutation == "state":
            readback["state"] = "stopped"
        elif mutation == "instance_lifecycle":
            readback["instance_lifecycle"] = "spot"
        elif mutation == "market_tag":
            next(tag for tag in readback["tags"] if tag["key"] == "citrees-market")["value"] = (
                "spot"
            )
        elif mutation.endswith("_tag"):
            tag_mutations = {
                "artifact_prefix_tag": (
                    "citrees-artifact-prefix",
                    "gates/r-cforest-reproducibility/wrong",
                ),
                "gate_identity_tag": ("citrees-gate-identity", "d" * 64),
                "launch_nonce_tag": ("citrees-gate-launch-nonce", "d" * 32),
                "host_slot_tag": ("citrees-host-slot", "arc-b"),
                "image_digest_tag": ("citrees-image-digest", "sha256:" + "d" * 64),
                "source_git_sha_tag": ("citrees-source-git-sha", "d" * 40),
            }
            key, value = tag_mutations[mutation]
            next(tag for tag in readback["tags"] if tag["key"] == key)["value"] = value
        elif mutation == "operator":
            evidence = validate_instance_identity_record(
                payloads[0]["provenance"]["instance_identity"],
                signature_verifier=_accept_signature,
            )
            readback["operator_identity"] = evidence.sts_identity.to_record()
        else:
            readback["operator_identity"] = AwsCallerIdentity(
                account_id=ACCOUNT_A,
                arn=(
                    f"arn:aws:sts::{ACCOUNT_A}:assumed-role/citrees-gate-instance/readback-session"
                ),
                user_id="AROAGATE:readback-session",
            ).to_record()
        readbacks[0] = _attest_operator_readback(
            readback,
            payloads,
            runtime_contract,
            manifest,
        )

    with pytest.raises(ValueError):
        _compare(
            runtime_contract,
            manifest,
            payloads,
            operator_readbacks=readbacks,
        )


def test_compare_rejects_coordinated_resigned_alternate_gate_attempt() -> None:
    runtime_contract, manifest, payloads = _payloads()
    readbacks = _operator_readbacks(payloads, runtime_contract, manifest)
    runtime = runtime_contract["runtime"]
    alternate_nonce = "d" * 32
    alternate_values = {
        "citrees-artifact-prefix": gate.gate_output_prefix(
            runtime["git_sha"],
            runtime["container_image_digest"],
            alternate_nonce,
        ),
        "citrees-gate-identity": gate.gate_launch_identity(
            runtime["git_sha"],
            runtime["container_image_digest"],
            alternate_nonce,
        ),
        "citrees-gate-launch-nonce": alternate_nonce,
    }
    for index, attestation in enumerate(readbacks):
        readback = copy.deepcopy(attestation["readback"])
        for tag in readback["tags"]:
            if tag["key"] in alternate_values:
                tag["value"] = alternate_values[tag["key"]]
        readbacks[index] = _attest_operator_readback(
            readback,
            payloads,
            runtime_contract,
            manifest,
        )

    with pytest.raises(ValueError, match="exact gate launch tags"):
        _compare(
            runtime_contract,
            manifest,
            payloads,
            operator_readbacks=readbacks,
        )


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@pytest.mark.parametrize(
    "offset_seconds",
    [
        -(gate.MAX_OPERATOR_READBACK_AGE_SECONDS + 1),
        gate.MAX_OPERATOR_READBACK_CLOCK_SKEW_SECONDS + 1,
    ],
)
def test_compare_rejects_stale_or_future_operator_attestations(
    offset_seconds: int,
) -> None:
    runtime_contract, manifest, payloads = _payloads()
    created_at = datetime.now(UTC).replace(microsecond=0)
    observed_at = created_at + timedelta(seconds=offset_seconds)
    readbacks = _operator_readbacks(
        payloads,
        runtime_contract,
        manifest,
        observed_at_utc=_timestamp(observed_at),
    )

    with pytest.raises(ValueError, match="stale|future"):
        gate.compare_payloads(
            payloads,
            readbacks,
            gate_launch_nonce=GATE_LAUNCH_NONCE,
            manifest=manifest,
            receipt_created_at_utc=_timestamp(created_at),
            runtime_contract=runtime_contract,
        )


def test_operator_attestations_cannot_be_replayed_across_run_payloads() -> None:
    runtime_contract, manifest, payloads = _payloads()
    readbacks = _operator_readbacks(payloads, runtime_contract, manifest)
    changed_payloads = copy.deepcopy(payloads)
    changed_payloads[0]["elapsed_seconds"] = 3.0

    with pytest.raises(ValueError, match="bindings differ"):
        _compare(
            runtime_contract,
            manifest,
            changed_payloads,
            operator_readbacks=readbacks,
        )


def test_operator_signature_rejects_co_tampered_zone_evidence_and_readback() -> None:
    runtime_contract, manifest, payloads = _payloads()
    readbacks = _operator_readbacks(payloads, runtime_contract, manifest)
    changed_payloads = copy.deepcopy(payloads)
    changed_readbacks = copy.deepcopy(readbacks)
    for payload in changed_payloads:
        provenance = payload["provenance"]
        if provenance["instance_id"] == HOST_A:
            provenance["availability_zone_id"] = "use1-az9"
            provenance["instance_identity"]["availability_zone_id"] = "use1-az9"
    changed_readbacks[0]["readback"]["availability_zone_id"] = "use1-az9"

    with pytest.raises(ValueError, match="bindings differ|signature verification failed"):
        _compare(
            runtime_contract,
            manifest,
            changed_payloads,
            operator_readbacks=changed_readbacks,
        )


def test_gate_receipt_embeds_and_revalidates_complete_evidence() -> None:
    runtime_contract, manifest, payloads = _payloads()
    receipt = gate.create_gate_receipt(
        list(reversed(payloads)),
        list(reversed(_operator_readbacks(payloads, runtime_contract, manifest))),
        gate_launch_nonce=GATE_LAUNCH_NONCE,
        manifest=manifest,
        runtime_contract=runtime_contract,
    )
    serialized = gate.serialize_gate_receipt(
        receipt,
        manifest=manifest,
        runtime_contract=runtime_contract,
    )
    digest = gate.gate_receipt_sha256(
        receipt,
        manifest=manifest,
        runtime_contract=runtime_contract,
    )

    assert receipt["report"]["status"] == "GO"
    assert receipt["account_manifest_sha256"] == {
        account_id: hashlib.sha256(payload).hexdigest()
        for account_id, payload in partition_rerun_manifest_by_account(manifest).items()
    }
    assert [payload["provenance"]["run_id"] for payload in receipt["run_payloads"]] == sorted(
        payload["provenance"]["run_id"] for payload in payloads
    )
    assert [
        record["readback"]["instance_id"] for record in receipt["operator_readbacks"]
    ] == sorted((HOST_A, HOST_B))
    assert digest == hashlib.sha256(serialized).hexdigest()
    assert gate.gate_receipt_s3_key(digest) == f"runtime-gate-receipts/{digest}.json"
    assert (
        gate.parse_gate_receipt(
            serialized,
            manifest=manifest,
            runtime_contract=runtime_contract,
            expected_sha256=digest,
        )
        == receipt
    )


def test_gate_receipt_rechecks_real_pkcs7_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_contract, manifest, payloads = _payloads()
    signer = _create_signer(tmp_path / "gate-signer")
    evidence_by_instance: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        provenance = payload["provenance"]
        instance_id = provenance["instance_id"]
        if instance_id in evidence_by_instance:
            provenance["instance_identity"] = copy.deepcopy(evidence_by_instance[instance_id])
            continue
        record = provenance["instance_identity"]
        raw_document = base64.b64decode(
            record["raw_document_base64"],
            validate=True,
        )
        signature = _sign_document(
            signer,
            raw_document,
            name=provenance["run_id"],
        )
        evidence = validate_instance_identity(
            raw_document,
            signature,
            provenance["availability_zone_id"],
            _instance_caller(
                provenance["aws_account_id"],
                provenance["instance_id"],
            ),
            certificate_pem=signer.certificate,
        )
        evidence_record = evidence.to_record()
        evidence_by_instance[instance_id] = evidence_record
        provenance["instance_identity"] = copy.deepcopy(evidence_record)

    def validate(record):
        return validate_instance_identity_record(
            record,
            certificate_pem=signer.certificate,
        )

    monkeypatch.setattr(gate, "validate_instance_identity_record", validate)
    receipt = gate.create_gate_receipt(
        payloads,
        _operator_readbacks(payloads, runtime_contract, manifest),
        gate_launch_nonce=GATE_LAUNCH_NONCE,
        manifest=manifest,
        runtime_contract=runtime_contract,
    )
    serialized = gate.serialize_gate_receipt(
        receipt,
        manifest=manifest,
        runtime_contract=runtime_contract,
    )

    assert (
        gate.parse_gate_receipt(
            serialized,
            manifest=manifest,
            runtime_contract=runtime_contract,
        )
        == receipt
    )

    tampered = copy.deepcopy(receipt)
    identity_record = tampered["run_payloads"][0]["provenance"]["instance_identity"]
    raw_document = base64.b64decode(
        identity_record["raw_document_base64"],
        validate=True,
    )
    changed_document = raw_document.replace(
        b'"instanceType":"c6a.8xlarge"',
        b'"instanceType":"m6i.8xlarge"',
    )
    assert changed_document != raw_document
    identity_record["raw_document_base64"] = base64.b64encode(changed_document).decode("ascii")

    with pytest.raises(ValueError, match="signature|does not exactly match"):
        gate.serialize_gate_receipt(
            tampered,
            manifest=manifest,
            runtime_contract=runtime_contract,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda receipt: receipt.update(manifest_sha256="0" * 64),
        lambda receipt: receipt["account_manifest_sha256"].update({ACCOUNT_A: "0" * 64}),
        lambda receipt: receipt["report"].update(status="NO-GO"),
        lambda receipt: receipt["operator_readbacks"][0]["readback"].update(
            availability_zone_id=ZONE_ID_B
        ),
        lambda receipt: receipt["run_payloads"][0]["results"]["classification"].clear(),
        lambda receipt: receipt["run_payloads"].reverse(),
    ],
)
def test_gate_receipt_rejects_tampering(mutation) -> None:
    runtime_contract, manifest, payloads = _payloads()
    receipt = gate.create_gate_receipt(
        payloads,
        _operator_readbacks(payloads, runtime_contract, manifest),
        gate_launch_nonce=GATE_LAUNCH_NONCE,
        manifest=manifest,
        runtime_contract=runtime_contract,
    )
    mutation(receipt)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        gate.serialize_gate_receipt(
            receipt,
            manifest=manifest,
            runtime_contract=runtime_contract,
        )


def test_gate_receipt_parser_rejects_noncanonical_or_wrong_digest() -> None:
    runtime_contract, manifest, payloads = _payloads()
    receipt = gate.create_gate_receipt(
        payloads,
        _operator_readbacks(payloads, runtime_contract, manifest),
        gate_launch_nonce=GATE_LAUNCH_NONCE,
        manifest=manifest,
        runtime_contract=runtime_contract,
    )
    serialized = gate.serialize_gate_receipt(
        receipt,
        manifest=manifest,
        runtime_contract=runtime_contract,
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        gate.parse_gate_receipt(
            serialized,
            manifest=manifest,
            runtime_contract=runtime_contract,
            expected_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="canonical JSON"):
        gate.parse_gate_receipt(
            serialized + b"\n",
            manifest=manifest,
            runtime_contract=runtime_contract,
        )


def test_compare_cli_writes_canonical_receipt_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = b'{"report":{"status":"GO"}}'
    raw_stdout = io.BytesIO()
    stdout = io.TextIOWrapper(raw_stdout, encoding="ascii", write_through=True)
    args = SimpleNamespace(
        command="compare",
        gate_launch_nonce=GATE_LAUNCH_NONCE,
        manifest=Path("manifest.csv"),
        operator_private_key=Path("operator-private.pem"),
        operator_profiles=("arc",),
        runs=tuple(Path(f"run-{index}.json") for index in range(4)),
        runtime_contract=Path("runtime-contract.json"),
    )
    parser = SimpleNamespace(parse_args=lambda: args)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(gate, "_parser", lambda: parser)
    monkeypatch.setattr(gate, "_load_manifest", lambda _path: object())
    monkeypatch.setattr(gate, "_load_runtime_contract", lambda _path: {})
    monkeypatch.setattr(gate, "_load_payload", lambda _path: {})
    monkeypatch.setattr(gate, "load_operator_private_key", lambda _path: b"private")
    monkeypatch.setattr(gate, "collect_live_operator_readbacks", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(gate, "create_gate_receipt", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        gate,
        "serialize_gate_receipt",
        lambda *_args, **_kwargs: expected,
    )

    gate.main()
    stdout.flush()

    assert raw_stdout.getvalue() == expected


def test_run_gate_executes_every_panel_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = _provenance(
        "gate-run",
        HOST_A,
        "us-east-1a",
        ZONE_ID_A,
        ACCOUNT_A,
        BOOT_A,
        100,
    )
    runtime_contract = gate.create_runtime_contract(provenance, OPERATOR_PUBLIC_KEY)
    manifest = _manifest(gate.runtime_contract_sha256(runtime_contract))
    cells = tuple(
        cell
        for cell in manifest.cells
        if cell.config.method.name == "r_cforest"
        and cell.config.task == "classification"
        and cell.config.dataset == "classification_fixture_0"
        and (
            (
                cell.config.method.params_dict["testtype"],
                cell.config.method.params_dict["replace"],
                cell.config.seed,
            )
            in {
                ("Bonferroni", False, gate.EXPECTED_SEEDS[0]),
                ("MonteCarlo", True, gate.EXPECTED_SEEDS[-1]),
            }
        )
    )
    inventory = {
        "classification": {"classification_fixture_0": cells},
        "regression": {},
    }
    observed_seeds: list[int] = []

    monkeypatch.setattr(gate, "_provenance", lambda run_id: provenance)
    monkeypatch.setattr(gate, "_replacement_inventory", lambda value: inventory)
    monkeypatch.setattr(gate, "_gate_inventory", lambda value: inventory)
    monkeypatch.setattr(
        gate,
        "load_dataset",
        lambda *args, **kwargs: (
            np.arange(180, dtype=float).reshape(20, 9),
            np.repeat(np.arange(2), 10),
        ),
    )

    def run_r_selection_parallel(
        X: Any,
        y: Any,
        method: str,
        task: str,
        seed: int,
        params: dict[str, Any],
        timeout_seconds: float,
    ) -> list[dict[str, Any]]:
        del X, y, method, task, params
        assert timeout_seconds == gate.STAGE1_SELECTION_TIMEOUT_SECONDS
        observed_seeds.append(seed)
        cpu_partitions = gate.partition_cpu_ids(tuple(range(32)), gate.N_FOLDS)
        return [
            {
                "feature_ranking": list(range(9)),
                "fold_cpu_affinity": list(cpu_partitions[fold]),
            }
            for fold in range(gate.N_FOLDS)
        ]

    monkeypatch.setattr(gate, "run_r_selection_parallel", run_r_selection_parallel)

    payload = gate.run_gate("gate-run", runtime_contract, manifest)
    configurations = payload["results"]["classification"]["classification_fixture_0"][
        "configurations"
    ]

    assert observed_seeds == [gate.EXPECTED_SEEDS[0], gate.EXPECTED_SEEDS[-1]]
    assert len(configurations) == 2
    assert {result["seed"] for result in configurations.values()} == {
        gate.EXPECTED_SEEDS[0],
        gate.EXPECTED_SEEDS[-1],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("thread_environment", {name: None for name in gate.THREAD_ENVIRONMENT}),
        ("threadpools", [{"num_threads": 2}]),
        ("container_image", "repository:latest"),
        ("script_sha256", "not-a-digest"),
    ],
)
def test_compare_rejects_incomplete_runtime_provenance(
    field: str,
    value: Any,
) -> None:
    runtime_contract, manifest, payloads = _payloads()
    payloads[0]["provenance"][field] = value

    with pytest.raises((TypeError, ValueError)):
        _compare(runtime_contract, manifest, payloads)


def test_compare_rejects_duplicate_process_incarnation() -> None:
    runtime_contract, manifest, payloads = _payloads()
    payloads[1]["provenance"]["process_start_ticks"] = 100

    with pytest.raises(ValueError, match="process incarnations"):
        _compare(runtime_contract, manifest, payloads)


def test_compare_rejects_shared_boot_identity_across_hosts() -> None:
    runtime_contract, manifest, payloads = _payloads()
    payloads[2]["provenance"]["boot_id"] = BOOT_A
    payloads[3]["provenance"]["boot_id"] = BOOT_A

    with pytest.raises(ValueError, match="independent boot"):
        _compare(runtime_contract, manifest, payloads)


def test_compare_requires_both_gate_hosts_in_the_target_account() -> None:
    runtime_contract, manifest, payloads = _payloads()
    for index in (2, 3):
        old = payloads[index]["provenance"]
        payloads[index]["provenance"] = _provenance(
            old["run_id"],
            HOST_B,
            old["availability_zone"],
            old["availability_zone_id"],
            ACCOUNT_B,
            old["boot_id"],
            old["process_start_ticks"],
        )

    with pytest.raises(ValueError, match="exact target AWS account"):
        _compare(runtime_contract, manifest, payloads)


def test_gate_requires_one_target_account() -> None:
    _runtime_contract, manifest, _payloads_value = _payloads()
    changed_one = False
    cells: list[ManifestCell] = []
    for cell in manifest.cells:
        if not changed_one and cell.config.method.name == "r_cforest":
            cell = replace(cell, target_aws_account_id=ACCOUNT_B)
            changed_one = True
        cells.append(cell)
    frozen_cells = tuple(cells)
    campaign_sha256 = compute_campaign_sha256(
        frozen_cells,
        runtime_contract_sha256=manifest.runtime_contract_sha256,
    )
    changed = parse_rerun_manifest(
        serialize_rerun_manifest(
            frozen_cells,
            campaign_sha256=campaign_sha256,
            runtime_contract_sha256=manifest.runtime_contract_sha256,
        )
    )

    with pytest.raises(ValueError, match="exactly 1 AWS account"):
        gate._replacement_inventory(changed)


def test_compare_rejects_nonfrozen_runtime_even_when_all_runs_match() -> None:
    runtime_contract, manifest, payloads = _payloads()
    for payload in payloads:
        old = payload["provenance"]
        payload["provenance"] = _provenance(
            old["run_id"],
            old["instance_id"],
            old["availability_zone"],
            old["availability_zone_id"],
            old["aws_account_id"],
            old["boot_id"],
            old["process_start_ticks"],
            ami_id="ami-" + "e" * 17,
        )

    with pytest.raises(ValueError, match="frozen runtime"):
        _compare(runtime_contract, manifest, payloads)


def test_compare_rejects_missing_dynamic_feature_cutoff() -> None:
    runtime_contract, manifest, payloads = _payloads()
    dataset_result = next(iter(payloads[0]["results"]["classification"].values()))
    dataset_result["k_values"].pop()

    with pytest.raises(ValueError, match="feature-count schedule"):
        _compare(runtime_contract, manifest, payloads)


def test_compare_rejects_wrong_dataset_identity() -> None:
    runtime_contract, manifest, payloads = _payloads()
    dataset_result = next(iter(payloads[0]["results"]["regression"].values()))
    dataset_result["identity"]["sha256"] = "e" * 64

    with pytest.raises(ValueError, match="identity"):
        _compare(runtime_contract, manifest, payloads)


def test_compare_rejects_missing_or_modified_production_configuration() -> None:
    runtime_contract, manifest, payloads = _payloads()
    dataset_result = next(iter(payloads[0]["results"]["classification"].values()))
    configurations = dataset_result["configurations"]
    removed_label = next(iter(configurations))
    configurations.pop(removed_label)

    with pytest.raises(ValueError, match="exact classification/.+ configurations"):
        _compare(runtime_contract, manifest, payloads)

    runtime_contract, manifest, payloads = _payloads()
    dataset_result = next(iter(payloads[0]["results"]["classification"].values()))
    configurations = dataset_result["configurations"]
    configurations[next(iter(configurations))]["params"]["cores"] = 2
    with pytest.raises(ValueError, match="invalid parameters"):
        _compare(runtime_contract, manifest, payloads)


def test_compare_rejects_valid_but_different_complete_ranking() -> None:
    runtime_contract, manifest, payloads = _payloads()
    dataset_result = next(iter(payloads[3]["results"]["regression"].values()))
    result = next(iter(dataset_result["configurations"].values()))
    n_features = dataset_result["identity"]["n_features"]
    rankings = copy.deepcopy(result["rankings"])
    rankings[0][0], rankings[0][1] = rankings[0][1], rankings[0][0]
    result.update(
        gate.summarize_rankings(
            rankings,
            n_features=n_features,
            k_values=dataset_result["k_values"],
        )
    )

    with pytest.raises(RuntimeError, match="ranking"):
        _compare(runtime_contract, manifest, payloads)


def test_summarize_rankings_rejects_incomplete_permutation() -> None:
    rankings = _rankings(120)
    rankings[0][-1] = rankings[0][0]

    with pytest.raises(ValueError, match="complete permutation"):
        gate.summarize_rankings(
            rankings,
            n_features=120,
            k_values=[5, 10, 25, 50, 100, 120],
        )
