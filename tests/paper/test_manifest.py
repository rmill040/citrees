"""Tests for strict corrected-benchmark manifest handling."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from typing import Any

import pytest

from paper.benchmark.pipeline.manifest import (
    MANIFEST_COLUMNS,
    compute_campaign_sha256,
    manifest_s3_key,
    parse_rerun_manifest,
    partition_rerun_manifest_by_account,
    serialize_rerun_manifest,
    validate_manifest_sha256,
    verify_account_manifest_shards,
)
from paper.benchmark.pipeline.methods import get_full_method_configs

pytestmark = pytest.mark.paper
CAMPAIGN_SHA256 = "e" * 64


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
    campaign_sha256 = compute_campaign_sha256(provisional.cells)
    return _payload([{**row, "campaign_sha256": campaign_sha256} for row in rows])


def test_manifest_validates_current_grid_and_exact_digest() -> None:
    payload = _payload([_row()])
    digest = hashlib.sha256(payload).hexdigest()

    manifest = parse_rerun_manifest(payload, expected_sha256=digest)

    assert manifest.sha256 == digest
    assert manifest.campaign_sha256 == CAMPAIGN_SHA256
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
    assert verify_account_manifest_shards(manifest, first) == {
        "123456789012": 3,
        "210987654321": 2,
    }


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


def test_serialized_shard_preserves_parent_campaign_identity() -> None:
    manifest = parse_rerun_manifest(_canonical_payload([_row()]))
    payload = serialize_rerun_manifest(
        manifest.cells,
        campaign_sha256=manifest.campaign_sha256,
    )

    assert parse_rerun_manifest(payload).campaign_sha256 == manifest.campaign_sha256


@pytest.mark.parametrize("value", ["", "A" * 64, "0" * 63, "g" * 64])
def test_manifest_digest_validation_is_strict(value: str) -> None:
    with pytest.raises(ValueError, match="64 lowercase"):
        validate_manifest_sha256(value)
