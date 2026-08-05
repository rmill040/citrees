"""Runtime loading and validation for one gate-approved benchmark campaign."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from paper.benchmark.experiments.r_cforest_reproducibility import (
    GATE_RECEIPT_PROFILE,
    GATE_RECEIPT_SCHEMA_VERSION,
    gate_receipt_s3_key,
    parse_gate_receipt,
    validate_gate_receipt_sha256,
)
from paper.benchmark.pipeline.manifest import (
    RerunManifest,
    manifest_s3_key,
    parse_rerun_manifest,
    validate_canonical_campaign,
    validate_manifest_sha256,
    verify_account_manifest_shard,
)
from paper.benchmark.pipeline.runtime_contract import (
    RUNTIME_CONTRACT_PROFILE,
    RUNTIME_CONTRACT_SCHEMA_VERSION,
    parse_runtime_contract,
    runtime_contract_s3_key,
    validate_runtime_contract_sha256,
)


@dataclass(frozen=True)
class CampaignGateIdentity:
    """Content identities required to run one account shard."""

    account_id: str
    campaign_sha256: str
    canonical_manifest_s3_key: str
    canonical_manifest_sha256: str
    gate_receipt_s3_key: str
    gate_receipt_sha256: str
    manifest_s3_key: str
    manifest_sha256: str
    runtime_contract_s3_key: str
    runtime_contract_sha256: str


@dataclass(frozen=True)
class ApprovedCampaignGate:
    """Fully validated canonical campaign, account shard, and gate evidence."""

    identity: CampaignGateIdentity
    canonical_manifest: RerunManifest
    manifest: RerunManifest
    runtime_contract: dict[str, Any]
    gate_receipt: dict[str, Any]


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def configured_campaign_gate_identity() -> CampaignGateIdentity:
    """Load and validate the configured immutable campaign identities."""
    from paper.benchmark.infra.aws import get_aws_account_id

    account_id = get_aws_account_id()

    campaign_sha256 = validate_manifest_sha256(_required_environment("CITREES_CAMPAIGN_SHA256"))
    canonical_manifest_sha256 = validate_manifest_sha256(
        _required_environment("CITREES_CANONICAL_MANIFEST_SHA256")
    )
    manifest_sha256 = validate_manifest_sha256(_required_environment("CITREES_MANIFEST_SHA256"))
    gate_receipt_sha256 = validate_gate_receipt_sha256(
        _required_environment("CITREES_GATE_RECEIPT_SHA256")
    )
    runtime_contract_sha256 = validate_runtime_contract_sha256(
        _required_environment("CITREES_RUNTIME_CONTRACT_SHA256")
    )

    identity = CampaignGateIdentity(
        account_id=account_id,
        campaign_sha256=campaign_sha256,
        canonical_manifest_s3_key=_required_environment("CITREES_CANONICAL_MANIFEST_S3_KEY"),
        canonical_manifest_sha256=canonical_manifest_sha256,
        gate_receipt_s3_key=_required_environment("CITREES_GATE_RECEIPT_S3_KEY"),
        gate_receipt_sha256=gate_receipt_sha256,
        manifest_s3_key=_required_environment("CITREES_MANIFEST_S3_KEY"),
        manifest_sha256=manifest_sha256,
        runtime_contract_s3_key=_required_environment("CITREES_RUNTIME_CONTRACT_S3_KEY"),
        runtime_contract_sha256=runtime_contract_sha256,
    )
    expected_keys = {
        "canonical manifest": manifest_s3_key(canonical_manifest_sha256),
        "manifest": manifest_s3_key(manifest_sha256),
        "gate receipt": gate_receipt_s3_key(gate_receipt_sha256),
        "runtime contract": runtime_contract_s3_key(runtime_contract_sha256),
    }
    observed_keys = {
        "canonical manifest": identity.canonical_manifest_s3_key,
        "manifest": identity.manifest_s3_key,
        "gate receipt": identity.gate_receipt_s3_key,
        "runtime contract": identity.runtime_contract_s3_key,
    }
    for label, expected_key in expected_keys.items():
        if observed_keys[label] != expected_key:
            raise RuntimeError(
                f"configured {label} key must be {expected_key!r}, got {observed_keys[label]!r}"
            )
    return identity


def _load_object(store: Any, key: str, *, label: str) -> tuple[bytes, dict[str, str]]:
    from botocore.exceptions import ClientError

    try:
        response = store.client.get_object(Bucket=store.bucket, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            raise FileNotFoundError(f"{label} not found: s3://{store.bucket}/{key}") from exc
        raise
    return response["Body"].read(), response.get("Metadata", {})


def load_approved_campaign_gate(
    store: Any,
    identity: CampaignGateIdentity,
) -> ApprovedCampaignGate:
    """Fetch and fully validate one gate-approved active-account campaign."""
    runtime_payload, runtime_metadata = _load_object(
        store,
        identity.runtime_contract_s3_key,
        label="Runtime contract",
    )
    expected_runtime_metadata = {
        "profile": RUNTIME_CONTRACT_PROFILE,
        "schema-version": str(RUNTIME_CONTRACT_SCHEMA_VERSION),
        "sha256": identity.runtime_contract_sha256,
    }
    if runtime_metadata != expected_runtime_metadata:
        raise RuntimeError("runtime contract S3 metadata is invalid")
    runtime_contract = parse_runtime_contract(
        runtime_payload,
        expected_sha256=identity.runtime_contract_sha256,
    )

    canonical_payload, canonical_metadata = _load_object(
        store,
        identity.canonical_manifest_s3_key,
        label="Canonical manifest",
    )
    canonical_manifest = parse_rerun_manifest(
        canonical_payload,
        expected_sha256=identity.canonical_manifest_sha256,
    )
    validate_canonical_campaign(canonical_manifest)
    expected_canonical_metadata = {
        "campaign-sha256": identity.campaign_sha256,
        "cell-count": str(len(canonical_manifest.cells)),
        "profile": "canonical-campaign",
        "runtime-contract-key": identity.runtime_contract_s3_key,
        "runtime-contract-sha256": identity.runtime_contract_sha256,
        "sha256": identity.canonical_manifest_sha256,
        "target-aws-account-ids": ",".join(canonical_manifest.account_ids),
    }
    if canonical_metadata != expected_canonical_metadata:
        raise RuntimeError("canonical manifest S3 metadata is invalid")
    if (
        canonical_manifest.campaign_sha256 != identity.campaign_sha256
        or canonical_manifest.runtime_contract_sha256 != identity.runtime_contract_sha256
        or identity.account_id not in canonical_manifest.account_ids
    ):
        raise RuntimeError("canonical manifest scope is invalid")

    manifest_payload, manifest_metadata = _load_object(
        store,
        identity.manifest_s3_key,
        label="Account manifest",
    )
    manifest = verify_account_manifest_shard(
        canonical_manifest,
        account_id=identity.account_id,
        payload=manifest_payload,
    )
    if manifest.sha256 != identity.manifest_sha256:
        raise RuntimeError("account manifest digest is invalid")
    expected_manifest_metadata = {
        "campaign-sha256": identity.campaign_sha256,
        "canonical-manifest-key": identity.canonical_manifest_s3_key,
        "canonical-manifest-sha256": identity.canonical_manifest_sha256,
        "cell-count": str(len(manifest.cells)),
        "gate-receipt-key": identity.gate_receipt_s3_key,
        "gate-receipt-sha256": identity.gate_receipt_sha256,
        "runtime-contract-key": identity.runtime_contract_s3_key,
        "runtime-contract-sha256": identity.runtime_contract_sha256,
        "sha256": identity.manifest_sha256,
        "target-aws-account-id": identity.account_id,
    }
    if manifest_metadata != expected_manifest_metadata:
        raise RuntimeError("account manifest S3 metadata is invalid")

    receipt_payload, receipt_metadata = _load_object(
        store,
        identity.gate_receipt_s3_key,
        label="Gate receipt",
    )
    expected_receipt_metadata = {
        "campaign-sha256": identity.campaign_sha256,
        "manifest-sha256": identity.canonical_manifest_sha256,
        "profile": GATE_RECEIPT_PROFILE,
        "runtime-contract-sha256": identity.runtime_contract_sha256,
        "schema-version": str(GATE_RECEIPT_SCHEMA_VERSION),
        "sha256": identity.gate_receipt_sha256,
        "status": "GO",
    }
    if receipt_metadata != expected_receipt_metadata:
        raise RuntimeError("gate receipt S3 metadata is invalid")
    gate_receipt = parse_gate_receipt(
        receipt_payload,
        manifest=canonical_manifest,
        runtime_contract=runtime_contract,
        expected_sha256=identity.gate_receipt_sha256,
    )
    if (
        gate_receipt["report"]["status"] != "GO"
        or gate_receipt["account_manifest_sha256"].get(identity.account_id)
        != identity.manifest_sha256
    ):
        raise RuntimeError("gate receipt does not approve the active account manifest")

    return ApprovedCampaignGate(
        identity=identity,
        canonical_manifest=canonical_manifest,
        manifest=manifest,
        runtime_contract=runtime_contract,
        gate_receipt=gate_receipt,
    )
