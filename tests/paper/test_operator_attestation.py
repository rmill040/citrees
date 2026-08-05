"""Tests for cryptographically bound operator-side readbacks."""

from __future__ import annotations

import copy
import stat
from pathlib import Path

import pytest

from paper.benchmark.pipeline.operator_attestation import (
    create_operator_attestation,
    generate_operator_keypair,
    load_operator_private_key,
    load_operator_public_key,
    operator_public_key_from_private_key,
    validate_operator_attestation,
    validate_operator_public_key,
)
from tests.paper.operator_attestation_fixtures import (
    OPERATOR_PRIVATE_KEY_PEM,
    OPERATOR_PUBLIC_KEY,
)

pytestmark = pytest.mark.paper

CAMPAIGN_SHA256 = "a" * 64
MANIFEST_SHA256 = "b" * 64
RUNTIME_CONTRACT_SHA256 = "c" * 64
RUN_PAYLOAD_SHA256S = ("d" * 64, "e" * 64, "f" * 64, "1" * 64)
OBSERVED_AT_UTC = "2026-08-05T12:34:56Z"


def _readback() -> dict[str, object]:
    return {
        "availability_zone_id": "use1-az1",
        "instance_id": "i-0123456789abcdef0",
        "state": "running",
    }


def _attestation() -> dict[str, object]:
    return create_operator_attestation(
        _readback(),
        campaign_sha256=CAMPAIGN_SHA256,
        manifest_sha256=MANIFEST_SHA256,
        observed_at_utc=OBSERVED_AT_UTC,
        private_key_pem=OPERATOR_PRIVATE_KEY_PEM,
        public_key=OPERATOR_PUBLIC_KEY,
        run_payload_sha256s=RUN_PAYLOAD_SHA256S,
        runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
    )


def _validate(record: dict[str, object]) -> dict[str, object]:
    return validate_operator_attestation(
        record,
        campaign_sha256=CAMPAIGN_SHA256,
        manifest_sha256=MANIFEST_SHA256,
        public_key=OPERATOR_PUBLIC_KEY,
        run_payload_sha256s=RUN_PAYLOAD_SHA256S,
        runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
    )


def test_operator_attestation_round_trips_and_binds_exact_runs() -> None:
    attestation = _attestation()

    assert _validate(attestation) == attestation
    assert attestation["run_payload_sha256s"] == sorted(RUN_PAYLOAD_SHA256S)
    assert attestation["readback"] == _readback()

    with pytest.raises(ValueError, match="bindings differ"):
        validate_operator_attestation(
            attestation,
            campaign_sha256=CAMPAIGN_SHA256,
            manifest_sha256=MANIFEST_SHA256,
            public_key=OPERATOR_PUBLIC_KEY,
            run_payload_sha256s=(*RUN_PAYLOAD_SHA256S[:-1], "2" * 64),
            runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(campaign_sha256="0" * 64),
        lambda value: value.update(observed_at_utc="2026-08-05T12:34:57Z"),
        lambda value: value["readback"].update(availability_zone_id="use1-az9"),
        lambda value: value["run_payload_sha256s"].reverse(),
        lambda value: value.update(signature_base64="A" + value["signature_base64"][1:]),
    ],
)
def test_operator_attestation_rejects_binding_or_signature_tampering(mutation) -> None:
    attestation = copy.deepcopy(_attestation())
    mutation(attestation)

    with pytest.raises((TypeError, ValueError)):
        _validate(attestation)


def test_operator_attestation_rejects_wrong_private_or_public_key(tmp_path: Path) -> None:
    private_path = tmp_path / "other-private.pem"
    public_path = tmp_path / "other-public.pem"
    other_public_key = generate_operator_keypair(private_path, public_path)

    with pytest.raises(ValueError, match="does not match"):
        create_operator_attestation(
            _readback(),
            campaign_sha256=CAMPAIGN_SHA256,
            manifest_sha256=MANIFEST_SHA256,
            observed_at_utc=OBSERVED_AT_UTC,
            private_key_pem=load_operator_private_key(private_path),
            public_key=OPERATOR_PUBLIC_KEY,
            run_payload_sha256s=RUN_PAYLOAD_SHA256S,
            runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
        )
    with pytest.raises(ValueError, match="signature verification failed"):
        validate_operator_attestation(
            _attestation(),
            campaign_sha256=CAMPAIGN_SHA256,
            manifest_sha256=MANIFEST_SHA256,
            public_key=other_public_key,
            run_payload_sha256s=RUN_PAYLOAD_SHA256S,
            runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
        )


def test_operator_keypair_is_exclusive_and_private_key_is_mode_0600(tmp_path: Path) -> None:
    private_path = tmp_path / "operator-private.pem"
    public_path = tmp_path / "operator-public.pem"

    public_key = generate_operator_keypair(private_path, public_path)

    assert stat.S_IMODE(private_path.stat().st_mode) == 0o600
    assert load_operator_public_key(public_path) == public_key
    assert (
        operator_public_key_from_private_key(load_operator_private_key(private_path)) == public_key
    )
    with pytest.raises(FileExistsError):
        generate_operator_keypair(private_path, public_path)

    private_path.chmod(0o644)
    with pytest.raises(PermissionError, match="group or others"):
        load_operator_private_key(private_path)


def test_operator_public_key_rejects_digest_tampering() -> None:
    public_key = copy.deepcopy(OPERATOR_PUBLIC_KEY)
    public_key["public_key_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="does not match"):
        validate_operator_public_key(public_key)
