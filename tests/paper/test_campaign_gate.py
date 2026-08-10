"""End-to-end tests for canonical campaign gate publication and loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from paper.benchmark.experiments import r_cforest_reproducibility as gate
from paper.benchmark.experiments.r_cforest_reproducibility import (
    create_gate_receipt,
    gate_receipt_s3_key,
    serialize_gate_receipt,
)
from paper.benchmark.infra import aws as aws_infra
from paper.benchmark.pipeline import campaign_gate as campaign_gate_module
from paper.benchmark.pipeline.campaign_gate import (
    CampaignGateIdentity,
    configured_campaign_gate_identity,
    load_approved_campaign_gate,
)
from paper.benchmark.pipeline.instance_identity import validate_instance_identity_record
from paper.benchmark.pipeline.manifest import (
    canonical_manifest_s3_key,
    compute_campaign_sha256,
    manifest_s3_key,
    parse_rerun_manifest,
    partition_rerun_manifest_by_account,
    serialize_rerun_manifest,
)
from paper.benchmark.pipeline.runtime_contract import (
    parse_runtime_contract,
    runtime_contract_s3_key,
    serialize_runtime_contract,
)
from tests.paper.test_infra import _MemoryS3
from tests.paper.test_r_cforest_reproducibility import (
    ACCOUNT_A,
    GATE_LAUNCH_NONCE,
    _accept_signature,
    _operator_readbacks,
    _payloads,
)

pytestmark = pytest.mark.paper

_REQUIRED_GATE_ENVIRONMENT = (
    "CITREES_CAMPAIGN_SHA256",
    "CITREES_CANONICAL_MANIFEST_SHA256",
    "CITREES_MANIFEST_SHA256",
    "CITREES_GATE_RECEIPT_SHA256",
    "CITREES_RUNTIME_CONTRACT_SHA256",
    "CITREES_CANONICAL_MANIFEST_S3_KEY",
    "CITREES_GATE_RECEIPT_S3_KEY",
    "CITREES_MANIFEST_S3_KEY",
    "CITREES_RUNTIME_CONTRACT_S3_KEY",
)


def _validate_fixture_identity(record):
    return validate_instance_identity_record(
        record,
        signature_verifier=_accept_signature,
    )


@pytest.fixture(autouse=True)
def _accept_fixture_identity_signatures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gate,
        "validate_instance_identity_record",
        _validate_fixture_identity,
    )


def _write_campaign_inputs(
    directory: Path,
) -> tuple[Path, Path, Path, Path]:
    runtime_contract, source_manifest, payloads = _payloads()
    canonical_payload = serialize_rerun_manifest(
        source_manifest.cells,
        campaign_sha256=source_manifest.campaign_sha256,
        runtime_contract_sha256=source_manifest.runtime_contract_sha256,
    )
    canonical = parse_rerun_manifest(canonical_payload)
    for payload in payloads:
        payload["manifest_sha256"] = canonical.sha256
    receipt = create_gate_receipt(
        payloads,
        _operator_readbacks(payloads, runtime_contract, canonical),
        gate_launch_nonce=GATE_LAUNCH_NONCE,
        manifest=canonical,
        runtime_contract=runtime_contract,
    )

    canonical_path = directory / "canonical.csv"
    canonical_path.write_bytes(canonical_payload)
    manifest_path = directory / "account.csv"
    manifest_path.write_bytes(partition_rerun_manifest_by_account(canonical)[ACCOUNT_A])
    runtime_path = directory / "runtime.json"
    runtime_path.write_bytes(serialize_runtime_contract(runtime_contract))
    receipt_path = directory / "receipt.json"
    receipt_path.write_bytes(
        serialize_gate_receipt(
            receipt,
            manifest=canonical,
            runtime_contract=runtime_contract,
        )
    )
    return manifest_path, canonical_path, runtime_path, receipt_path


def _publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_MemoryS3, dict[str, str | int]]:
    paths = _write_campaign_inputs(tmp_path)
    client = _MemoryS3()
    monkeypatch.setattr(aws_infra, "get_aws_account_id", lambda: ACCOUNT_A)
    monkeypatch.setattr(aws_infra, "ensure_s3_bucket", lambda region: "citrees-test")
    monkeypatch.setattr(aws_infra.boto3, "client", lambda *args, **kwargs: client)
    return client, aws_infra.publish_rerun_manifest(*paths)


@pytest.fixture(scope="module")
def _published_campaign_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[
    dict[str, tuple[bytes, dict[str, str]]],
    dict[str, str | int],
]:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        gate,
        "validate_instance_identity_record",
        _validate_fixture_identity,
    )
    try:
        client, publication = _publish(
            tmp_path_factory.mktemp("campaign-gate"),
            monkeypatch,
        )
        objects = {
            key: (payload, dict(metadata)) for key, (payload, metadata) in client.objects.items()
        }
    finally:
        monkeypatch.undo()
    return objects, dict(publication)


@pytest.fixture
def published_campaign(
    _published_campaign_template: tuple[
        dict[str, tuple[bytes, dict[str, str]]],
        dict[str, str | int],
    ],
) -> tuple[_MemoryS3, dict[str, str | int]]:
    objects, publication = _published_campaign_template
    client = _MemoryS3()
    client.objects = {
        key: (payload, dict(metadata)) for key, (payload, metadata) in objects.items()
    }
    return client, dict(publication)


def _identity(publication: dict[str, str | int]) -> CampaignGateIdentity:
    return CampaignGateIdentity(
        account_id=ACCOUNT_A,
        campaign_sha256=str(publication["campaign_sha256"]),
        canonical_manifest_s3_key=str(publication["canonical_manifest_s3_key"]),
        canonical_manifest_sha256=str(publication["canonical_manifest_sha256"]),
        gate_receipt_s3_key=str(publication["gate_receipt_s3_key"]),
        gate_receipt_sha256=str(publication["gate_receipt_sha256"]),
        manifest_s3_key=str(publication["key"]),
        manifest_sha256=str(publication["sha256"]),
        runtime_contract_s3_key=str(publication["runtime_contract_s3_key"]),
        runtime_contract_sha256=str(publication["runtime_contract_sha256"]),
    )


def _configure_identity_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> CampaignGateIdentity:
    identity = CampaignGateIdentity(
        account_id=ACCOUNT_A,
        campaign_sha256="e" * 64,
        canonical_manifest_s3_key=canonical_manifest_s3_key("c" * 64),
        canonical_manifest_sha256="c" * 64,
        gate_receipt_s3_key=gate_receipt_s3_key("d" * 64),
        gate_receipt_sha256="d" * 64,
        manifest_s3_key=manifest_s3_key("b" * 64),
        manifest_sha256="b" * 64,
        runtime_contract_s3_key=runtime_contract_s3_key("f" * 64),
        runtime_contract_sha256="f" * 64,
    )
    values = {
        "CITREES_CAMPAIGN_SHA256": identity.campaign_sha256,
        "CITREES_CANONICAL_MANIFEST_S3_KEY": identity.canonical_manifest_s3_key,
        "CITREES_CANONICAL_MANIFEST_SHA256": identity.canonical_manifest_sha256,
        "CITREES_GATE_RECEIPT_S3_KEY": identity.gate_receipt_s3_key,
        "CITREES_GATE_RECEIPT_SHA256": identity.gate_receipt_sha256,
        "CITREES_MANIFEST_S3_KEY": identity.manifest_s3_key,
        "CITREES_MANIFEST_SHA256": identity.manifest_sha256,
        "CITREES_RUNTIME_CONTRACT_S3_KEY": identity.runtime_contract_s3_key,
        "CITREES_RUNTIME_CONTRACT_SHA256": identity.runtime_contract_sha256,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(aws_infra, "get_aws_account_id", lambda: ACCOUNT_A)
    return identity


def _store(client: _MemoryS3) -> SimpleNamespace:
    return SimpleNamespace(bucket="citrees-test", client=client)


def _replace_canonical_manifest(
    client: _MemoryS3,
    identity: CampaignGateIdentity,
    *,
    cells: tuple[Any, ...],
    campaign_sha256: str,
) -> CampaignGateIdentity:
    payload = serialize_rerun_manifest(
        cells,
        campaign_sha256=campaign_sha256,
        runtime_contract_sha256=identity.runtime_contract_sha256,
    )
    digest = hashlib.sha256(payload).hexdigest()
    key = canonical_manifest_s3_key(digest)
    _, original_metadata = client.objects[identity.canonical_manifest_s3_key]
    account_ids = sorted({cell.target_aws_account_id for cell in cells})
    metadata = {
        **original_metadata,
        "campaign-sha256": campaign_sha256,
        "cell-count": str(len(cells)),
        "sha256": digest,
        "target-aws-account-ids": ",".join(account_ids),
    }
    client.objects[key] = (payload, metadata)
    return replace(
        identity,
        campaign_sha256=campaign_sha256,
        canonical_manifest_s3_key=key,
        canonical_manifest_sha256=digest,
    )


def _replace_gate_receipt(
    client: _MemoryS3,
    identity: CampaignGateIdentity,
    *,
    mutation: str,
) -> CampaignGateIdentity:
    payload, original_metadata = client.objects[identity.gate_receipt_s3_key]
    receipt = json.loads(payload.decode("ascii"))
    if mutation == "non-go":
        receipt["report"]["status"] = "NO-GO"
    elif mutation == "wrong-account-manifest":
        receipt["account_manifest_sha256"][identity.account_id] = "0" * 64
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(f"unknown gate receipt mutation {mutation!r}")
    changed_payload = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    digest = hashlib.sha256(changed_payload).hexdigest()
    key = gate_receipt_s3_key(digest)
    client.objects[key] = (
        changed_payload,
        {**original_metadata, "sha256": digest},
    )
    manifest_payload, manifest_metadata = client.objects[identity.manifest_s3_key]
    client.objects[identity.manifest_s3_key] = (
        manifest_payload,
        {
            **manifest_metadata,
            "gate-receipt-key": key,
            "gate-receipt-sha256": digest,
        },
    )
    return replace(
        identity,
        gate_receipt_s3_key=key,
        gate_receipt_sha256=digest,
    )


def test_publication_loads_only_the_exact_gate_approved_account_shard(
    published_campaign: tuple[_MemoryS3, dict[str, str | int]],
) -> None:
    client, publication = published_campaign
    identity = _identity(publication)

    approved = load_approved_campaign_gate(_store(client), identity)

    assert approved.identity == identity
    assert approved.canonical_manifest.sha256 == identity.canonical_manifest_sha256
    assert approved.canonical_manifest.campaign_sha256 == identity.campaign_sha256
    assert approved.canonical_manifest.account_ids == (ACCOUNT_A,)
    assert len(approved.canonical_manifest.cells) == 941
    assert approved.manifest.sha256 == identity.manifest_sha256
    assert len(approved.manifest.cells) == 941
    assert approved.manifest.account_ids == (ACCOUNT_A,)
    assert approved.runtime_contract["profile"] == "r_cforest_runtime"
    assert approved.gate_receipt["report"]["status"] == "GO"
    assert approved.gate_receipt["account_manifest_sha256"][ACCOUNT_A] == identity.manifest_sha256


@pytest.mark.parametrize(
    "environment_value",
    (None, " \t"),
    ids=("absent", "blank"),
)
@pytest.mark.parametrize("name", _REQUIRED_GATE_ENVIRONMENT)
def test_configured_gate_identity_rejects_every_missing_environment_value(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    environment_value: str | None,
) -> None:
    _configure_identity_environment(monkeypatch)
    if environment_value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, environment_value)

    with pytest.raises(RuntimeError, match=rf"{name} is required"):
        configured_campaign_gate_identity()


@pytest.mark.parametrize(
    ("name", "label", "wrong_key"),
    (
        (
            "CITREES_CANONICAL_MANIFEST_S3_KEY",
            "canonical manifest",
            manifest_s3_key("0" * 64),
        ),
        (
            "CITREES_MANIFEST_S3_KEY",
            "manifest",
            manifest_s3_key("0" * 64),
        ),
        (
            "CITREES_GATE_RECEIPT_S3_KEY",
            "gate receipt",
            gate_receipt_s3_key("0" * 64),
        ),
        (
            "CITREES_RUNTIME_CONTRACT_S3_KEY",
            "runtime contract",
            runtime_contract_s3_key("0" * 64),
        ),
    ),
)
def test_configured_gate_identity_rejects_every_content_addressed_key_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    label: str,
    wrong_key: str,
) -> None:
    _configure_identity_environment(monkeypatch)
    monkeypatch.setenv(name, wrong_key)

    with pytest.raises(
        RuntimeError,
        match=rf"configured {label} key must be",
    ):
        configured_campaign_gate_identity()


def test_configured_gate_identity_uses_live_sts_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _configure_identity_environment(monkeypatch)
    monkeypatch.setenv("AWS_ACCOUNT_ID", "999999999999")

    assert configured_campaign_gate_identity() == expected


@pytest.mark.parametrize(
    ("key_attribute", "label"),
    (
        ("runtime_contract_s3_key", "Runtime contract"),
        ("canonical_manifest_s3_key", "Canonical manifest"),
        ("manifest_s3_key", "Account manifest"),
        ("gate_receipt_s3_key", "Gate receipt"),
    ),
)
def test_runtime_loader_rejects_every_missing_gate_object(
    published_campaign: tuple[_MemoryS3, dict[str, str | int]],
    key_attribute: str,
    label: str,
) -> None:
    client, publication = published_campaign
    identity = _identity(publication)
    del client.objects[getattr(identity, key_attribute)]

    with pytest.raises(FileNotFoundError, match=rf"{label} not found"):
        load_approved_campaign_gate(_store(client), identity)


@pytest.mark.parametrize(
    ("key_attribute", "error"),
    (
        (
            "runtime_contract_s3_key",
            "runtime contract SHA-256 mismatch",
        ),
        (
            "canonical_manifest_s3_key",
            "manifest SHA-256 mismatch",
        ),
        (
            "manifest_s3_key",
            "manifest shard differs from the canonical partition",
        ),
        (
            "gate_receipt_s3_key",
            "gate receipt SHA-256 mismatch",
        ),
    ),
)
def test_runtime_loader_rejects_modified_gate_object_bytes(
    published_campaign: tuple[_MemoryS3, dict[str, str | int]],
    key_attribute: str,
    error: str,
) -> None:
    client, publication = published_campaign
    identity = _identity(publication)
    key = getattr(identity, key_attribute)
    payload, metadata = client.objects[key]
    client.objects[key] = (payload + b"\n", metadata)

    with pytest.raises(ValueError, match=error):
        load_approved_campaign_gate(_store(client), identity)


@pytest.mark.parametrize("mutation", ("invalid", "extra"))
@pytest.mark.parametrize(
    ("key_attribute", "error"),
    (
        (
            "runtime_contract_s3_key",
            "runtime contract S3 metadata is invalid",
        ),
        (
            "canonical_manifest_s3_key",
            "canonical manifest S3 metadata is invalid",
        ),
        (
            "manifest_s3_key",
            "account manifest S3 metadata is invalid",
        ),
        (
            "gate_receipt_s3_key",
            "gate receipt S3 metadata is invalid",
        ),
    ),
)
def test_runtime_loader_rejects_invalid_or_extra_metadata_for_every_gate_object(
    published_campaign: tuple[_MemoryS3, dict[str, str | int]],
    key_attribute: str,
    error: str,
    mutation: str,
) -> None:
    client, publication = published_campaign
    identity = _identity(publication)
    key = getattr(identity, key_attribute)
    payload, original_metadata = client.objects[key]
    metadata = dict(original_metadata)
    if mutation == "invalid":
        metadata["sha256"] = "0" * 64
    else:
        metadata["unexpected"] = "value"
    client.objects[key] = (payload, metadata)

    with pytest.raises(RuntimeError, match=error):
        load_approved_campaign_gate(_store(client), identity)


def test_runtime_loader_rejects_non_go_gate_receipt_metadata(
    published_campaign: tuple[_MemoryS3, dict[str, str | int]],
) -> None:
    client, publication = published_campaign
    identity = _identity(publication)
    payload, metadata = client.objects[identity.gate_receipt_s3_key]
    client.objects[identity.gate_receipt_s3_key] = (
        payload,
        {**metadata, "status": "NO-GO"},
    )

    with pytest.raises(
        RuntimeError,
        match="gate receipt S3 metadata is invalid",
    ):
        load_approved_campaign_gate(_store(client), identity)


def test_runtime_loader_rejects_changed_canonical_partition(
    published_campaign: tuple[_MemoryS3, dict[str, str | int]],
) -> None:
    client, publication = published_campaign
    identity = _identity(publication)
    canonical_payload, _ = client.objects[identity.canonical_manifest_s3_key]
    canonical = parse_rerun_manifest(canonical_payload)
    cells = canonical.cells[:-1]
    campaign_sha256 = compute_campaign_sha256(
        cells,
        runtime_contract_sha256=identity.runtime_contract_sha256,
    )
    changed_identity = _replace_canonical_manifest(
        client,
        identity,
        cells=cells,
        campaign_sha256=campaign_sha256,
    )

    with pytest.raises(
        ValueError,
        match="manifest shard differs from the canonical partition",
    ):
        load_approved_campaign_gate(_store(client), changed_identity)


def test_runtime_loader_rejects_invalid_canonical_campaign_digest(
    published_campaign: tuple[_MemoryS3, dict[str, str | int]],
) -> None:
    client, publication = published_campaign
    identity = _identity(publication)
    canonical_payload, _ = client.objects[identity.canonical_manifest_s3_key]
    canonical = parse_rerun_manifest(canonical_payload)
    changed_identity = _replace_canonical_manifest(
        client,
        identity,
        cells=canonical.cells,
        campaign_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="canonical campaign SHA-256 mismatch"):
        load_approved_campaign_gate(_store(client), changed_identity)


def test_runtime_loader_rejects_canonical_campaign_scope_mismatch(
    published_campaign: tuple[_MemoryS3, dict[str, str | int]],
) -> None:
    client, publication = published_campaign
    identity = _identity(publication)
    payload, metadata = client.objects[identity.canonical_manifest_s3_key]
    client.objects[identity.canonical_manifest_s3_key] = (
        payload,
        {**metadata, "campaign-sha256": "0" * 64},
    )
    changed_identity = replace(identity, campaign_sha256="0" * 64)

    with pytest.raises(RuntimeError, match="canonical manifest scope is invalid"):
        load_approved_campaign_gate(_store(client), changed_identity)


def test_runtime_loader_rejects_canonical_runtime_scope_mismatch(
    published_campaign: tuple[_MemoryS3, dict[str, str | int]],
) -> None:
    client, publication = published_campaign
    identity = _identity(publication)
    runtime_payload, runtime_metadata = client.objects[identity.runtime_contract_s3_key]
    runtime_contract = parse_runtime_contract(runtime_payload)
    runtime_contract["runtime"]["kernel"] = "6.1.0-scope-mismatch"
    changed_runtime_payload = serialize_runtime_contract(runtime_contract)
    changed_runtime_sha256 = hashlib.sha256(changed_runtime_payload).hexdigest()
    changed_runtime_key = runtime_contract_s3_key(changed_runtime_sha256)
    client.objects[changed_runtime_key] = (
        changed_runtime_payload,
        {**runtime_metadata, "sha256": changed_runtime_sha256},
    )
    canonical_payload, canonical_metadata = client.objects[identity.canonical_manifest_s3_key]
    client.objects[identity.canonical_manifest_s3_key] = (
        canonical_payload,
        {
            **canonical_metadata,
            "runtime-contract-key": changed_runtime_key,
            "runtime-contract-sha256": changed_runtime_sha256,
        },
    )
    changed_identity = replace(
        identity,
        runtime_contract_s3_key=changed_runtime_key,
        runtime_contract_sha256=changed_runtime_sha256,
    )

    with pytest.raises(RuntimeError, match="canonical manifest scope is invalid"):
        load_approved_campaign_gate(_store(client), changed_identity)


def test_runtime_loader_rejects_account_outside_canonical_manifest(
    published_campaign: tuple[_MemoryS3, dict[str, str | int]],
) -> None:
    client, publication = published_campaign
    identity = replace(_identity(publication), account_id="999999999999")

    with pytest.raises(RuntimeError, match="canonical manifest scope is invalid"):
        load_approved_campaign_gate(_store(client), identity)


def test_runtime_loader_rejects_account_shard_that_differs_from_canonical_partition(
    published_campaign: tuple[_MemoryS3, dict[str, str | int]],
) -> None:
    client, publication = published_campaign
    identity = _identity(publication)
    canonical_payload, _ = client.objects[identity.canonical_manifest_s3_key]
    canonical = parse_rerun_manifest(canonical_payload)
    _, metadata = client.objects[identity.manifest_s3_key]
    changed_payload = serialize_rerun_manifest(
        canonical.cells[:-1],
        campaign_sha256=canonical.campaign_sha256,
        runtime_contract_sha256=canonical.runtime_contract_sha256,
    )
    client.objects[identity.manifest_s3_key] = (changed_payload, metadata)

    with pytest.raises(
        ValueError,
        match="manifest shard differs from the canonical partition",
    ):
        load_approved_campaign_gate(_store(client), identity)


def test_runtime_loader_rejects_account_manifest_digest_mismatch(
    published_campaign: tuple[_MemoryS3, dict[str, str | int]],
) -> None:
    client, publication = published_campaign
    identity = replace(_identity(publication), manifest_sha256="0" * 64)

    with pytest.raises(RuntimeError, match="account manifest digest is invalid"):
        load_approved_campaign_gate(_store(client), identity)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("non-go", "gate receipt report differs from the embedded run payloads"),
        (
            "wrong-account-manifest",
            "gate receipt has an invalid account_manifest_sha256",
        ),
    ),
)
def test_runtime_loader_rejects_semantically_invalid_gate_receipt(
    published_campaign: tuple[_MemoryS3, dict[str, str | int]],
    mutation: str,
    error: str,
) -> None:
    client, publication = published_campaign
    changed_identity = _replace_gate_receipt(
        client,
        _identity(publication),
        mutation=mutation,
    )

    with pytest.raises(ValueError, match=error):
        load_approved_campaign_gate(_store(client), changed_identity)


@pytest.mark.parametrize("mutation", ("non-go", "wrong-account-manifest"))
def test_runtime_loader_defensively_rechecks_parsed_gate_receipt_approval(
    published_campaign: tuple[_MemoryS3, dict[str, str | int]],
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    client, publication = published_campaign
    identity = _identity(publication)
    parse_gate_receipt = campaign_gate_module.parse_gate_receipt

    def parse_then_mutate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        receipt = parse_gate_receipt(*args, **kwargs)
        if mutation == "non-go":
            receipt["report"]["status"] = "NO-GO"
        else:
            receipt["account_manifest_sha256"][identity.account_id] = "0" * 64
        return receipt

    monkeypatch.setattr(
        campaign_gate_module,
        "parse_gate_receipt",
        parse_then_mutate,
    )

    with pytest.raises(
        RuntimeError,
        match="gate receipt does not approve the active account manifest",
    ):
        load_approved_campaign_gate(_store(client), identity)
