"""Tests for strict corrected-benchmark manifest handling."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from typing import Any

import pytest

from paper.benchmark.config.constants import R_SELECTION_TIMEOUT_SECONDS
from paper.benchmark.pipeline.manifest import (
    MANIFEST_COLUMNS,
    account_manifest_sha256_map,
    compute_campaign_sha256,
    manifest_s3_key,
    parse_rerun_manifest,
    partition_rerun_manifest_by_account,
    serialize_rerun_manifest,
    validate_canonical_campaign,
    validate_manifest_sha256,
    verify_account_manifest_shard,
    verify_account_manifest_shards,
)
from paper.benchmark.pipeline.methods import get_full_method_configs
from paper.benchmark.pipeline.runtime_contract import (
    EXPECTED_THREAD_VALUE,
    PYTHON_LIBRARY_NAMES,
    R_RUNTIME_FIELDS,
    RUNTIME_CONTRACT_PROFILE,
    RUNTIME_CONTRACT_SCHEMA_VERSION,
    THREAD_ENVIRONMENT,
    parse_runtime_contract,
    runtime_contract_s3_key,
    runtime_contract_sha256,
    serialize_runtime_contract,
    validate_runtime_contract_sha256,
)
from tests.paper.operator_attestation_fixtures import OPERATOR_PUBLIC_KEY

pytestmark = pytest.mark.paper
CAMPAIGN_SHA256 = "e" * 64
RUNTIME_CONTRACT_SHA256 = "c" * 64


def _runtime_contract(*, cpu_model: str = "AMD EPYC 9R14") -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_CONTRACT_SCHEMA_VERSION,
        "profile": RUNTIME_CONTRACT_PROFILE,
        "operator_attestation_public_key": OPERATOR_PUBLIC_KEY,
        "runtime": {
            "ami_id": "ami-" + "a" * 17,
            "container_image_digest": "sha256:" + "b" * 64,
            "cpu_affinity": list(range(32)),
            "cpu_model": cpu_model,
            "git_sha": "c" * 40,
            "instance_type": "m7i.8xlarge",
            "kernel": "6.1.0-fixture",
            "logical_cpus": 32,
            "machine": "x86_64",
            "microcode": "0x1000065",
            "openssl_version": "OpenSSL 3.0.13 30 Jan 2024",
            "os_release": {"ID": "amzn", "VERSION_ID": "2023"},
            "python_libraries": {name: "1.0" for name in PYTHON_LIBRARY_NAMES},
            "r_numerical_libraries": {
                "blas": "/usr/local/lib/R/lib/libRblas.so",
                "lapack": "/usr/local/lib/R/lib/libRlapack.so",
            },
            "r_selection_timeout_seconds": R_SELECTION_TIMEOUT_SECONDS,
            "r_runtime": {name: "1.0" for name in R_RUNTIME_FIELDS},
            "thread_environment": {name: EXPECTED_THREAD_VALUE for name in THREAD_ENVIRONMENT},
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
        },
    }


def _row(
    *,
    method_name: str = "r_ctree",
    dataset: str = "glass",
    seed: int = 0,
    target_aws_account_id: str = "123456789012",
) -> dict[str, Any]:
    method = get_full_method_configs([method_name], "classification")[0]
    return {
        "task": "classification",
        "target_aws_account_id": target_aws_account_id,
        "campaign_sha256": CAMPAIGN_SHA256,
        "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
        "dataset_source": "real",
        "dataset": dataset,
        "dataset_sha256": "d" * 64,
        "dataset_n_samples": 214,
        "dataset_n_features": 9,
        "method_base": method.name,
        "method_id": method.label,
        "method_params_json": json.dumps(
            method.params_dict,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "seed": seed,
        "rerun_reason": "adapter_correction",
        "historically_omitted": False,
        "stage1_required": True,
        "stage2_required": True,
        "status": "pending",
    }


def _payload(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _canonical_payload(rows: list[dict[str, Any]]) -> bytes:
    provisional = parse_rerun_manifest(_payload(rows))
    campaign_sha256 = compute_campaign_sha256(
        provisional.cells,
        runtime_contract_sha256=provisional.runtime_contract_sha256,
    )
    return _payload([{**row, "campaign_sha256": campaign_sha256} for row in rows])


def test_manifest_validates_current_grid_and_exact_digest() -> None:
    payload = _payload([_row()])
    digest = hashlib.sha256(payload).hexdigest()

    manifest = parse_rerun_manifest(payload, expected_sha256=digest)

    assert manifest.sha256 == digest
    assert manifest.campaign_sha256 == CAMPAIGN_SHA256
    assert manifest.runtime_contract_sha256 == RUNTIME_CONTRACT_SHA256
    assert len(manifest.cells) == 1
    assert manifest.method_names == ("r_ctree",)
    assert manifest.account_ids == ("123456789012",)
    assert manifest.method_counts("rankings") == {"r_ctree": 1}
    assert manifest.configs_for("classification", "metrics") == (manifest.cells[0].config,)
    assert manifest_s3_key(digest) == f"rerun-manifests/{digest}.csv"


def test_manifest_rejects_digest_identity_and_duplicate_errors() -> None:
    row = _row()
    payload = _payload([row])

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        parse_rerun_manifest(payload, expected_sha256="a" * 64)
    with pytest.raises(ValueError, match="duplicate cell identity"):
        parse_rerun_manifest(_payload([row, row]))

    invalid_identity = {**row, "method_id": "r_ctree__" + "0" * 16}
    with pytest.raises(ValueError, match="method identity"):
        parse_rerun_manifest(_payload([invalid_identity]))

    invalid_dataset = {**row, "dataset_sha256": "not-a-sha"}
    with pytest.raises(ValueError, match="invalid dataset identity"):
        parse_rerun_manifest(_payload([invalid_dataset]))

    conflicting_dataset = {
        **_row(seed=1),
        "dataset_sha256": "e" * 64,
    }
    with pytest.raises(ValueError, match="dataset identity conflicts"):
        parse_rerun_manifest(_payload([row, conflicting_dataset]))

    conflicting_campaign = {
        **_row(seed=1),
        "campaign_sha256": "f" * 64,
    }
    with pytest.raises(ValueError, match="campaign_sha256 conflicts"):
        parse_rerun_manifest(_payload([row, conflicting_campaign]))

    conflicting_runtime = {
        **_row(seed=1),
        "runtime_contract_sha256": "f" * 64,
    }
    with pytest.raises(ValueError, match="runtime_contract_sha256 conflicts"):
        parse_rerun_manifest(_payload([row, conflicting_runtime]))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset_source", "synthetic", "does not match dataset"),
        ("target_aws_account_id", "not-an-account", "12 decimal digits"),
        ("seed", 5, "seed must be in"),
        ("status", "done", "status must be pending"),
        ("stage1_required", "yes", "must be True or False"),
    ],
)
def test_manifest_rejects_invalid_cell_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    row = _row()
    row[field] = value

    with pytest.raises(ValueError, match=message):
        parse_rerun_manifest(_payload([row]))


def test_manifest_requires_canonical_parameter_json() -> None:
    row = _row()
    row["method_params_json"] = json.dumps(
        json.loads(row["method_params_json"]),
        sort_keys=True,
    )

    with pytest.raises(ValueError, match="canonical JSON"):
        parse_rerun_manifest(_payload([row]))


def test_runtime_contract_is_canonical_and_content_addressed() -> None:
    contract = _runtime_contract()
    payload = serialize_runtime_contract(contract)
    digest = runtime_contract_sha256(contract)

    assert digest == hashlib.sha256(payload).hexdigest()
    assert parse_runtime_contract(payload, expected_sha256=digest) == contract
    assert runtime_contract_s3_key(digest) == f"runtime-contracts/{digest}.json"
    assert payload == json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")

    with pytest.raises(ValueError, match="canonical JSON bytes"):
        parse_runtime_contract(payload + b"\n")


def test_campaign_digest_binds_runtime_contract() -> None:
    first_runtime = runtime_contract_sha256(_runtime_contract())
    second_runtime = runtime_contract_sha256(
        _runtime_contract(cpu_model="Intel Xeon Platinum 8488C")
    )
    row = {**_row(), "runtime_contract_sha256": first_runtime}
    cells = parse_rerun_manifest(_payload([row])).cells

    first_campaign = compute_campaign_sha256(
        cells,
        runtime_contract_sha256=first_runtime,
    )
    second_campaign = compute_campaign_sha256(
        cells,
        runtime_contract_sha256=second_runtime,
    )

    assert first_runtime != second_runtime
    assert first_campaign != second_campaign


def test_canonical_campaign_requires_exactly_two_accounts() -> None:
    one_account = parse_rerun_manifest(_canonical_payload([_row(dataset="glass")]))
    three_accounts = parse_rerun_manifest(
        _canonical_payload(
            [
                _row(dataset="glass", target_aws_account_id="123456789012"),
                _row(dataset="wine", target_aws_account_id="210987654321"),
                _row(
                    dataset="madelon",
                    target_aws_account_id="345678901234",
                ),
            ]
        )
    )
    two_accounts = parse_rerun_manifest(
        _canonical_payload(
            [
                _row(dataset="glass", target_aws_account_id="123456789012"),
                _row(dataset="wine", target_aws_account_id="210987654321"),
            ]
        )
    )

    for manifest in (one_account, three_accounts):
        with pytest.raises(ValueError, match="exactly 2 AWS accounts"):
            validate_canonical_campaign(manifest)
    validate_canonical_campaign(two_accounts)


def test_manifest_rejects_missing_or_mixed_runtime_contracts() -> None:
    missing = {**_row(), "runtime_contract_sha256": ""}
    with pytest.raises(ValueError, match="runtime contract SHA-256"):
        parse_rerun_manifest(_payload([missing]))

    mixed = [
        _row(seed=0),
        {**_row(seed=1), "runtime_contract_sha256": "f" * 64},
    ]
    with pytest.raises(ValueError, match="runtime_contract_sha256 conflicts"):
        parse_rerun_manifest(_payload(mixed))


def test_partition_is_deterministic_disjoint_and_complete() -> None:
    rows = [
        _row(
            method_name=method_name,
            dataset=dataset,
            seed=seed,
            target_aws_account_id=account_id,
        )
        for method_name, dataset, seed, account_id in (
            ("r_ctree", "glass", 0, "123456789012"),
            ("r_ctree", "wine", 1, "210987654321"),
            ("r_cforest", "glass", 2, "123456789012"),
            ("r_cforest", "wine", 3, "210987654321"),
            ("boruta", "glass", 4, "123456789012"),
        )
    ]
    manifest = parse_rerun_manifest(_canonical_payload(rows))

    first = partition_rerun_manifest_by_account(manifest)
    second = partition_rerun_manifest_by_account(manifest)
    parsed = [parse_rerun_manifest(payload) for payload in first.values()]
    identities = [{cell.identity for cell in shard.cells} for shard in parsed]

    assert first == second
    assert len(parsed[0].cells) == 3
    assert len(parsed[1].cells) == 2
    assert identities[0].isdisjoint(identities[1])
    assert identities[0] | identities[1] == {cell.identity for cell in manifest.cells}
    assert {shard.campaign_sha256 for shard in parsed} == {manifest.campaign_sha256}
    assert {shard.runtime_contract_sha256 for shard in parsed} == {manifest.runtime_contract_sha256}
    assert verify_account_manifest_shards(manifest, first) == {
        "123456789012": 3,
        "210987654321": 2,
    }
    assert account_manifest_sha256_map(manifest) == {
        account_id: hashlib.sha256(payload).hexdigest() for account_id, payload in first.items()
    }
    for account_id, payload in first.items():
        shard = verify_account_manifest_shard(
            manifest,
            account_id=account_id,
            payload=payload,
        )
        assert shard.account_ids == (account_id,)


def test_single_account_shard_must_match_canonical_partition() -> None:
    rows = [
        _row(dataset="glass", target_aws_account_id="123456789012"),
        _row(dataset="wine", target_aws_account_id="210987654321"),
    ]
    manifest = parse_rerun_manifest(_canonical_payload(rows))
    payload = partition_rerun_manifest_by_account(manifest)["123456789012"]
    changed = payload.replace(b"glass", b"wine", 1)

    with pytest.raises(ValueError, match="differs from the canonical partition"):
        verify_account_manifest_shard(
            manifest,
            account_id="123456789012",
            payload=changed,
        )


def test_shards_from_different_campaigns_cannot_be_combined() -> None:
    first_rows = [
        _row(dataset="glass", target_aws_account_id="123456789012"),
        _row(dataset="wine", target_aws_account_id="210987654321"),
    ]
    second_rows = [
        first_rows[0],
        {
            **first_rows[1],
            "rerun_reason": "historically_omitted",
        },
    ]
    first = parse_rerun_manifest(_canonical_payload(first_rows))
    second = parse_rerun_manifest(_canonical_payload(second_rows))
    first_shards = partition_rerun_manifest_by_account(first)
    second_shards = partition_rerun_manifest_by_account(second)
    mixed = {
        "123456789012": first_shards["123456789012"],
        "210987654321": second_shards["210987654321"],
    }

    with pytest.raises(ValueError, match="campaign SHA-256"):
        verify_account_manifest_shards(first, mixed)


def test_shard_with_different_runtime_contract_is_rejected() -> None:
    rows = [
        _row(dataset="glass", target_aws_account_id="123456789012"),
        _row(dataset="wine", target_aws_account_id="210987654321"),
    ]
    manifest = parse_rerun_manifest(_canonical_payload(rows))
    shards = partition_rerun_manifest_by_account(manifest)
    account_id = "123456789012"
    shard = parse_rerun_manifest(shards[account_id])
    shards[account_id] = serialize_rerun_manifest(
        shard.cells,
        campaign_sha256=shard.campaign_sha256,
        runtime_contract_sha256="f" * 64,
    )

    with pytest.raises(ValueError, match="runtime contract SHA-256"):
        verify_account_manifest_shards(manifest, shards)


def test_serialized_shard_preserves_parent_campaign_identity() -> None:
    manifest = parse_rerun_manifest(_canonical_payload([_row()]))
    payload = serialize_rerun_manifest(
        manifest.cells,
        campaign_sha256=manifest.campaign_sha256,
        runtime_contract_sha256=manifest.runtime_contract_sha256,
    )

    parsed = parse_rerun_manifest(payload)
    assert parsed.campaign_sha256 == manifest.campaign_sha256
    assert parsed.runtime_contract_sha256 == manifest.runtime_contract_sha256


@pytest.mark.parametrize("value", ["", "A" * 64, "0" * 63, "g" * 64])
def test_manifest_digest_validation_is_strict(value: str) -> None:
    with pytest.raises(ValueError, match="64 lowercase"):
        validate_manifest_sha256(value)


@pytest.mark.parametrize("value", ["", "A" * 64, "0" * 63, "g" * 64])
def test_runtime_contract_digest_validation_is_strict(value: str) -> None:
    with pytest.raises(ValueError, match="64 lowercase"):
        validate_runtime_contract_sha256(value)
