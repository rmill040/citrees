"""Tests for signed EC2 identity and independent operator readback."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from paper.benchmark.pipeline.instance_identity import (
    IMDS_TOKEN_TTL_SECONDS,
    US_EAST_1_RSA2048_CERTIFICATE_PEM,
    AwsCallerIdentity,
    InstanceIdentityEvidence,
    collect_aws_caller_identity,
    collect_instance_identity,
    collect_operator_readback,
    validate_instance_identity,
    validate_instance_identity_record,
    validate_operator_readback,
    verify_pkcs7_signature,
)

pytestmark = pytest.mark.paper

ACCOUNT_ID = "123456789012"
OTHER_ACCOUNT_ID = "210987654321"
INSTANCE_ID = "i-0123456789abcdef0"
OTHER_INSTANCE_ID = "i-11111111111111111"
IMAGE_ID = "ami-0123456789abcdef0"
OTHER_IMAGE_ID = "ami-22222222222222222"
INSTANCE_TYPE = "c6a.8xlarge"
AVAILABILITY_ZONE = "us-east-1a"
AVAILABILITY_ZONE_ID = "use1-az1"
INSTANCE_PROFILE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:instance-profile/citrees-worker"
INSTANCE_ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/citrees-worker"


@dataclass(frozen=True)
class _Signer:
    certificate: bytes
    key_path: Path
    directory: Path


class _StsClient:
    def __init__(self, response: Mapping[str, Any]) -> None:
        self.response = response
        self.calls = 0

    def get_caller_identity(self) -> dict[str, Any]:
        self.calls += 1
        return dict(self.response)


class _Imds:
    def __init__(self, responses: Mapping[str, bytes]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, dict[str, str]]] = []

    def __call__(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str],
    ) -> bytes:
        self.calls.append((method, path, dict(headers)))
        return self.responses[path]


class _Ec2Client:
    def __init__(
        self,
        *,
        instance_response: Mapping[str, Any] | None = None,
        zone_response: Mapping[str, Any] | None = None,
    ) -> None:
        self.instance_response = (
            _describe_instances_response() if instance_response is None else instance_response
        )
        self.zone_response = (
            _describe_availability_zones_response() if zone_response is None else zone_response
        )
        self.instance_calls: list[dict[str, Any]] = []
        self.zone_calls: list[dict[str, Any]] = []

    def describe_instances(self, **kwargs: Any) -> Mapping[str, Any]:
        self.instance_calls.append(kwargs)
        return self.instance_response

    def describe_availability_zones(self, **kwargs: Any) -> Mapping[str, Any]:
        self.zone_calls.append(kwargs)
        return self.zone_response


class _IamClient:
    def __init__(
        self,
        *,
        profile_arn: str = INSTANCE_PROFILE_ARN,
        role_arn: str = INSTANCE_ROLE_ARN,
        roles: list[dict[str, Any]] | None = None,
    ) -> None:
        self.profile_arn = profile_arn
        self.role_arn = role_arn
        self.roles = [{"Arn": role_arn}] if roles is None else roles
        self.calls: list[dict[str, Any]] = []

    def get_instance_profile(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "InstanceProfile": {
                "Arn": self.profile_arn,
                "Roles": self.roles,
            },
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }


def _create_signer(directory: Path) -> _Signer:
    directory.mkdir(exist_ok=True)
    certificate_path = directory / "certificate.pem"
    key_path = directory / "private-key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key_path),
            "-out",
            str(certificate_path),
            "-nodes",
            "-sha256",
            "-subj",
            "/CN=citrees-instance-identity-test",
            "-days",
            "1",
        ],
        capture_output=True,
        check=True,
        timeout=10,
    )
    return _Signer(
        certificate=certificate_path.read_bytes(),
        key_path=key_path,
        directory=directory,
    )


@pytest.fixture(scope="module")
def signer(tmp_path_factory: pytest.TempPathFactory) -> _Signer:
    return _create_signer(tmp_path_factory.mktemp("instance-identity-signer"))


def _sign_document(signer: _Signer, raw_document: bytes, *, name: str) -> bytes:
    document_path = signer.directory / f"{name}.json"
    signature_path = signer.directory / f"{name}.pem"
    document_path.write_bytes(raw_document)
    subprocess.run(
        [
            "openssl",
            "smime",
            "-sign",
            "-binary",
            "-in",
            str(document_path),
            "-signer",
            str(signer.directory / "certificate.pem"),
            "-inkey",
            str(signer.key_path),
            "-outform",
            "PEM",
            "-nodetach",
            "-out",
            str(signature_path),
        ],
        capture_output=True,
        check=True,
        timeout=10,
    )
    lines = signature_path.read_bytes().strip().splitlines()
    if lines[0] != b"-----BEGIN PKCS7-----" or lines[-1] != b"-----END PKCS7-----":
        raise AssertionError("openssl did not generate a PKCS7 PEM fixture")
    return b"".join(lines[1:-1])


def _document(**updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "accountId": ACCOUNT_ID,
        "architecture": "x86_64",
        "availabilityZone": AVAILABILITY_ZONE,
        "billingProducts": None,
        "devpayProductCodes": None,
        "imageId": IMAGE_ID,
        "instanceId": INSTANCE_ID,
        "instanceType": INSTANCE_TYPE,
        "kernelId": None,
        "marketplaceProductCodes": None,
        "pendingTime": "2026-08-05T12:34:56Z",
        "privateIp": "10.0.0.10",
        "ramdiskId": None,
        "region": "us-east-1",
        "version": "2017-09-30",
    }
    value.update(updates)
    return value


def _raw_document(**updates: Any) -> bytes:
    return (
        json.dumps(
            _document(**updates),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _caller(account_id: str = ACCOUNT_ID) -> AwsCallerIdentity:
    return AwsCallerIdentity(
        account_id=account_id,
        arn=(f"arn:aws:sts::{account_id}:assumed-role/citrees-worker/i-0123456789abcdef0"),
        user_id="AROATEST:i-0123456789abcdef0",
    )


def _sts_response(account_id: str = ACCOUNT_ID) -> dict[str, Any]:
    identity = _caller(account_id)
    return {
        "Account": identity.account_id,
        "Arn": identity.arn,
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "UserId": identity.user_id,
    }


def _accept_signature(
    raw_document: bytes,
    rsa2048_signature: bytes,
    certificate_pem: bytes,
) -> None:
    assert raw_document
    assert rsa2048_signature
    assert certificate_pem


def _identity(**document_updates: Any) -> InstanceIdentityEvidence:
    return validate_instance_identity(
        _raw_document(**document_updates),
        b"AA==",
        AVAILABILITY_ZONE_ID,
        _caller(),
        signature_verifier=_accept_signature,
    )


def _signed_identity(signer: _Signer, *, name: str) -> InstanceIdentityEvidence:
    raw_document = _raw_document()
    signature = _sign_document(signer, raw_document, name=name)
    return validate_instance_identity(
        raw_document,
        signature,
        AVAILABILITY_ZONE_ID,
        _caller(),
        certificate_pem=signer.certificate,
    )


def _instance() -> dict[str, Any]:
    return {
        "Architecture": "x86_64",
        "ImageId": IMAGE_ID,
        "IamInstanceProfile": {
            "Arn": INSTANCE_PROFILE_ARN,
            "Id": "AIPATEST",
        },
        "InstanceId": INSTANCE_ID,
        "InstanceLifecycle": "spot",
        "InstanceType": INSTANCE_TYPE,
        "Placement": {
            "AvailabilityZone": AVAILABILITY_ZONE,
            "Tenancy": "default",
        },
        "State": {"Code": 16, "Name": "running"},
        "Tags": [
            {"Key": "Role", "Value": "worker"},
            {"Key": "Name", "Value": "citrees-rerun"},
        ],
    }


def _reservation(instances: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "Groups": [],
        "Instances": [_instance()] if instances is None else instances,
        "OwnerId": ACCOUNT_ID,
        "ReservationId": "r-0123456789abcdef0",
    }


def _describe_instances_response(
    reservations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "Reservations": [_reservation()] if reservations is None else reservations,
        "ResponseMetadata": {"HTTPStatusCode": 200},
    }


def _zone() -> dict[str, Any]:
    return {
        "OptInStatus": "opt-in-not-required",
        "RegionName": "us-east-1",
        "State": "available",
        "ZoneId": AVAILABILITY_ZONE_ID,
        "ZoneName": AVAILABILITY_ZONE,
        "ZoneType": "availability-zone",
    }


def _describe_availability_zones_response(
    zones: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "AvailabilityZones": [_zone()] if zones is None else zones,
        "ResponseMetadata": {"HTTPStatusCode": 200},
    }


def _operator_record(
    evidence: InstanceIdentityEvidence,
) -> dict[str, Any]:
    identity = evidence.identity
    return {
        "architecture": identity.architecture,
        "availability_zone": identity.availability_zone,
        "availability_zone_id": evidence.availability_zone_id,
        "iam_instance_profile_arn": INSTANCE_PROFILE_ARN,
        "iam_role_arn": INSTANCE_ROLE_ARN,
        "image_id": identity.image_id,
        "instance_id": identity.instance_id,
        "instance_lifecycle": "spot",
        "instance_type": identity.instance_type,
        "operator_identity": _caller().to_record(),
        "owner_account_id": identity.account_id,
        "region": identity.region,
        "state": "running",
        "tags": [
            {"key": "Name", "value": "citrees-rerun"},
            {"key": "Role", "value": "worker"},
        ],
    }


def test_pinned_certificate_matches_official_der_fingerprint() -> None:
    lines = US_EAST_1_RSA2048_CERTIFICATE_PEM.strip().splitlines()
    assert lines[0] == b"-----BEGIN CERTIFICATE-----"
    assert lines[-1] == b"-----END CERTIFICATE-----"

    certificate_der = base64.b64decode(b"".join(lines[1:-1]), validate=True)

    assert (
        hashlib.sha256(certificate_der).hexdigest()
        == "70923c22dd074172d0e5d27ea1ca865412525fccd8bc2cd0ff40660e275f0b30"
    )


def test_openssl_verifies_exact_attached_content(
    signer: _Signer,
) -> None:
    raw_document = _raw_document()
    signature = _sign_document(signer, raw_document, name="valid")

    verify_pkcs7_signature(raw_document, signature, signer.certificate)


def test_openssl_rejects_content_or_certificate_mismatch(
    signer: _Signer,
    tmp_path: Path,
) -> None:
    raw_document = _raw_document()
    signature = _sign_document(signer, raw_document, name="mismatch")
    other_signer = _create_signer(tmp_path / "other-signer")

    with pytest.raises(ValueError, match="does not exactly match"):
        verify_pkcs7_signature(
            _raw_document(instanceType="m6i.8xlarge"),
            signature,
            signer.certificate,
        )
    with pytest.raises(ValueError, match="signature verification failed"):
        verify_pkcs7_signature(
            raw_document,
            signature,
            other_signer.certificate,
        )


def test_collect_instance_identity_uses_only_imdsv2_and_live_sts(
    signer: _Signer,
) -> None:
    raw_document = _raw_document()
    signature = _sign_document(signer, raw_document, name="collection")
    imds = _Imds(
        {
            "/api/token": b"opaque-token",
            "/dynamic/instance-identity/document": raw_document,
            "/dynamic/instance-identity/rsa2048": signature,
            "/meta-data/placement/availability-zone-id": (AVAILABILITY_ZONE_ID.encode("ascii")),
        }
    )
    sts = _StsClient(_sts_response())

    evidence = collect_instance_identity(
        sts_client=sts,
        imds_request=imds,
        certificate_pem=signer.certificate,
    )

    assert evidence.raw_document == raw_document
    assert evidence.rsa2048_signature == signature
    assert evidence.identity.account_id == ACCOUNT_ID
    assert evidence.identity.instance_id == INSTANCE_ID
    assert evidence.identity.image_id == IMAGE_ID
    assert evidence.identity.instance_type == INSTANCE_TYPE
    assert evidence.identity.region == "us-east-1"
    assert evidence.identity.availability_zone == AVAILABILITY_ZONE
    assert evidence.availability_zone_id == AVAILABILITY_ZONE_ID
    assert evidence.sts_identity == _caller()
    assert sts.calls == 1
    assert imds.calls == [
        (
            "PUT",
            "/api/token",
            {"X-aws-ec2-metadata-token-ttl-seconds": str(IMDS_TOKEN_TTL_SECONDS)},
        ),
        (
            "GET",
            "/dynamic/instance-identity/document",
            {"X-aws-ec2-metadata-token": "opaque-token"},
        ),
        (
            "GET",
            "/dynamic/instance-identity/rsa2048",
            {"X-aws-ec2-metadata-token": "opaque-token"},
        ),
        (
            "GET",
            "/meta-data/placement/availability-zone-id",
            {"X-aws-ec2-metadata-token": "opaque-token"},
        ),
    ]


def test_instance_identity_record_round_trips_json_and_revalidates_signature(
    signer: _Signer,
) -> None:
    evidence = _signed_identity(signer, name="record-round-trip")

    record = evidence.to_record()
    payload = json.dumps(
        record,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    restored = validate_instance_identity_record(
        json.loads(payload),
        certificate_pem=signer.certificate,
    )

    assert set(record) == {
        "availability_zone_id",
        "raw_document_base64",
        "rsa2048_signature",
        "sts_identity",
    }
    assert base64.b64decode(record["raw_document_base64"], validate=True) == evidence.raw_document
    assert record["rsa2048_signature"] == evidence.rsa2048_signature.decode("ascii")
    assert restored == evidence
    assert restored.to_record() == record
    assert (
        json.dumps(
            restored.to_record(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        == payload
    )


def test_instance_identity_normalizes_wrapped_imds_signature() -> None:
    evidence = validate_instance_identity(
        _raw_document(),
        b"QUJD\nRA==\n",
        AVAILABILITY_ZONE_ID,
        _caller(),
        signature_verifier=_accept_signature,
    )

    assert evidence.rsa2048_signature == b"QUJDRA=="
    assert evidence.to_record()["rsa2048_signature"] == "QUJDRA=="


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_instance_identity_record_rejects_missing_or_extra_fields(
    mutation: str,
) -> None:
    record = _identity().to_record()
    if mutation == "missing":
        del record["availability_zone_id"]
    else:
        record["unexpected"] = "value"

    with pytest.raises(ValueError, match="fields differ"):
        validate_instance_identity_record(
            record,
            signature_verifier=_accept_signature,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_instance_identity_record_rejects_invalid_sts_schema(
    mutation: str,
) -> None:
    record = _identity().to_record()
    sts_record = record["sts_identity"]
    if mutation == "missing":
        del sts_record["user_id"]
    else:
        sts_record["unexpected"] = "value"

    with pytest.raises(ValueError, match="fields differ"):
        validate_instance_identity_record(
            record,
            signature_verifier=_accept_signature,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("raw_document_base64", ""),
        ("raw_document_base64", "not-base64"),
        (
            "raw_document_base64",
            base64.b64encode(_raw_document()).decode("ascii") + "\n",
        ),
        ("raw_document_base64", b"AA=="),
        ("rsa2048_signature", ""),
        ("rsa2048_signature", "AA"),
        ("rsa2048_signature", "AA==\n"),
        ("rsa2048_signature", b"AA=="),
    ],
)
def test_instance_identity_record_rejects_malformed_or_noncanonical_base64(
    field: str,
    value: str | bytes,
) -> None:
    record = _identity().to_record()
    record[field] = value

    with pytest.raises((TypeError, ValueError), match="base64"):
        validate_instance_identity_record(
            record,
            signature_verifier=_accept_signature,
        )


def test_instance_identity_record_rejects_tampered_document_or_signature(
    signer: _Signer,
) -> None:
    evidence = _signed_identity(signer, name="record-tampering")
    record = evidence.to_record()

    tampered_document = copy.deepcopy(record)
    tampered_document["raw_document_base64"] = base64.b64encode(
        _raw_document(instanceType="m6i.8xlarge")
    ).decode("ascii")
    with pytest.raises(ValueError, match="does not exactly match"):
        validate_instance_identity_record(
            tampered_document,
            certificate_pem=signer.certificate,
        )

    tampered_signature = copy.deepcopy(record)
    signature = tampered_signature["rsa2048_signature"]
    replacement = "A" if signature[0] != "A" else "B"
    tampered_signature["rsa2048_signature"] = replacement + signature[1:]
    with pytest.raises(ValueError, match="signature verification failed"):
        validate_instance_identity_record(
            tampered_signature,
            certificate_pem=signer.certificate,
        )


def test_instance_identity_record_rejects_tampered_sts_or_zone_id(
    signer: _Signer,
) -> None:
    evidence = _signed_identity(signer, name="record-scope-tampering")
    record = evidence.to_record()

    tampered_sts = copy.deepcopy(record)
    tampered_sts["sts_identity"] = _caller(OTHER_ACCOUNT_ID).to_record()
    with pytest.raises(ValueError, match="signed accountId"):
        validate_instance_identity_record(
            tampered_sts,
            certificate_pem=signer.certificate,
        )

    malformed_zone = copy.deepcopy(record)
    malformed_zone["availability_zone_id"] = "us-east-1a"
    with pytest.raises(ValueError, match="availability-zone ID"):
        validate_instance_identity_record(
            malformed_zone,
            certificate_pem=signer.certificate,
        )


@pytest.mark.parametrize("token", [b"", b" token", b"token\n", b"\xff"])
def test_collect_instance_identity_rejects_malformed_imdsv2_token(
    token: bytes,
) -> None:
    imds = _Imds({"/api/token": token})

    with pytest.raises(RuntimeError, match="IMDSv2"):
        collect_instance_identity(
            sts_client=_StsClient(_sts_response()),
            imds_request=imds,
            signature_verifier=_accept_signature,
        )


def test_instance_document_rejects_missing_and_extra_fields() -> None:
    missing = _document()
    del missing["imageId"]
    extra = {**_document(), "unexpected": "value"}

    for document in (missing, extra):
        raw_document = json.dumps(document, separators=(",", ":")).encode("ascii")
        with pytest.raises(ValueError, match="fields differ"):
            validate_instance_identity(
                raw_document,
                b"AA==",
                AVAILABILITY_ZONE_ID,
                _caller(),
                signature_verifier=_accept_signature,
            )


def test_instance_document_rejects_duplicate_fields() -> None:
    raw_document = _raw_document().replace(
        b'"accountId":"123456789012"',
        b'"accountId":"123456789012","accountId":"123456789012"',
        1,
    )

    with pytest.raises(ValueError, match="duplicate field 'accountId'"):
        validate_instance_identity(
            raw_document,
            b"AA==",
            AVAILABILITY_ZONE_ID,
            _caller(),
            signature_verifier=_accept_signature,
        )


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"region": "us-west-2"}, "region must equal us-east-1"),
        ({"availabilityZone": "us-west-2a"}, "us-east-1 availability zone"),
        ({"instanceId": "not-an-instance"}, "valid EC2 instance ID"),
        ({"imageId": "not-an-ami"}, "valid EC2 AMI ID"),
        ({"instanceType": "not-an-instance-type"}, "valid EC2 instance type"),
    ],
)
def test_instance_document_rejects_invalid_signed_scope(
    updates: dict[str, str],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_instance_identity(
            _raw_document(**updates),
            b"AA==",
            AVAILABILITY_ZONE_ID,
            _caller(),
            signature_verifier=_accept_signature,
        )


def test_instance_identity_rejects_signed_and_sts_account_mismatch() -> None:
    with pytest.raises(ValueError, match="signed accountId"):
        validate_instance_identity(
            _raw_document(),
            b"AA==",
            AVAILABILITY_ZONE_ID,
            _caller(OTHER_ACCOUNT_ID),
            signature_verifier=_accept_signature,
        )


def test_instance_identity_requires_instance_named_sts_session() -> None:
    caller = AwsCallerIdentity(
        account_id=ACCOUNT_ID,
        arn=f"arn:aws:sts::{ACCOUNT_ID}:assumed-role/citrees-worker/not-the-instance",
        user_id="AROATEST:not-the-instance",
    )

    with pytest.raises(ValueError, match="session named for the signed instance ID"):
        validate_instance_identity(
            _raw_document(),
            b"AA==",
            AVAILABILITY_ZONE_ID,
            caller,
            signature_verifier=_accept_signature,
        )


@pytest.mark.parametrize("signature", [b"", b"not-base64", b"AA= =", b"AA"])
def test_instance_identity_rejects_malformed_signature_before_verifier(
    signature: bytes,
) -> None:
    with pytest.raises((TypeError, ValueError), match="signature"):
        validate_instance_identity(
            _raw_document(),
            signature,
            AVAILABILITY_ZONE_ID,
            _caller(),
            signature_verifier=_accept_signature,
        )


def test_collect_sts_identity_rejects_extra_fields_and_arn_account_mismatch() -> None:
    extra = {**_sts_response(), "Unexpected": "value"}
    with pytest.raises(ValueError, match="extra=\\['Unexpected'\\]"):
        collect_aws_caller_identity(sts_client=_StsClient(extra))

    mismatched_arn = {
        **_sts_response(),
        "Arn": _caller(OTHER_ACCOUNT_ID).arn,
    }
    with pytest.raises(ValueError, match="ARN does not match"):
        collect_aws_caller_identity(sts_client=_StsClient(mismatched_arn))


def test_collect_operator_readback_uses_exact_queries_and_canonical_tags() -> None:
    evidence = _identity()
    ec2 = _Ec2Client()
    iam = _IamClient()
    sts = _StsClient(_sts_response())

    readback = collect_operator_readback(
        evidence,
        ec2_client=ec2,
        iam_client=iam,
        sts_client=sts,
    )

    assert ec2.instance_calls == [{"InstanceIds": [INSTANCE_ID]}]
    assert ec2.zone_calls == [{"ZoneNames": [AVAILABILITY_ZONE]}]
    assert iam.calls == [{"InstanceProfileName": "citrees-worker"}]
    assert sts.calls == 1
    assert readback.owner_account_id == ACCOUNT_ID
    assert readback.operator_identity == _caller()
    assert readback.architecture == "x86_64"
    assert readback.instance_id == INSTANCE_ID
    assert readback.image_id == IMAGE_ID
    assert readback.instance_type == INSTANCE_TYPE
    assert readback.region == "us-east-1"
    assert readback.availability_zone == AVAILABILITY_ZONE
    assert readback.availability_zone_id == AVAILABILITY_ZONE_ID
    assert readback.iam_instance_profile_arn == INSTANCE_PROFILE_ARN
    assert readback.iam_role_arn == INSTANCE_ROLE_ARN
    assert readback.tags == (
        ("Name", "citrees-rerun"),
        ("Role", "worker"),
    )
    assert validate_operator_readback(evidence, readback.to_record()) == readback


@pytest.mark.parametrize("roles", [[], [{"Arn": INSTANCE_ROLE_ARN}, {"Arn": INSTANCE_ROLE_ARN}]])
def test_operator_readback_requires_one_instance_profile_role(
    roles: list[dict[str, Any]],
) -> None:
    with pytest.raises(ValueError, match="exactly one IAM role"):
        collect_operator_readback(
            _identity(),
            ec2_client=_Ec2Client(),
            iam_client=_IamClient(roles=roles),
            sts_client=_StsClient(_sts_response()),
        )


def test_operator_readback_rejects_different_instance_profile_readback() -> None:
    different_profile = f"arn:aws:iam::{ACCOUNT_ID}:instance-profile/different"

    with pytest.raises(ValueError, match="different instance profile"):
        collect_operator_readback(
            _identity(),
            ec2_client=_Ec2Client(),
            iam_client=_IamClient(profile_arn=different_profile),
            sts_client=_StsClient(_sts_response()),
        )


def test_operator_validation_rejects_sts_role_absent_from_instance_profile() -> None:
    evidence = _identity()
    record = _operator_record(evidence)
    record["iam_role_arn"] = f"arn:aws:iam::{ACCOUNT_ID}:role/different-role"

    with pytest.raises(ValueError, match="STS role does not match"):
        validate_operator_readback(evidence, record)


@pytest.mark.parametrize(
    ("instance_response", "zone_response", "match"),
    [
        (
            _describe_instances_response([]),
            _describe_availability_zones_response(),
            "exactly one reservation, observed 0",
        ),
        (
            _describe_instances_response([_reservation(), _reservation()]),
            _describe_availability_zones_response(),
            "exactly one reservation, observed 2",
        ),
        (
            _describe_instances_response([_reservation([])]),
            _describe_availability_zones_response(),
            "exactly one instance, observed 0",
        ),
        (
            _describe_instances_response([_reservation([_instance(), _instance()])]),
            _describe_availability_zones_response(),
            "exactly one instance, observed 2",
        ),
        (
            _describe_instances_response(),
            _describe_availability_zones_response([]),
            "exactly one zone, observed 0",
        ),
        (
            _describe_instances_response(),
            _describe_availability_zones_response([_zone(), _zone()]),
            "exactly one zone, observed 2",
        ),
    ],
)
def test_operator_readback_rejects_aws_cardinality_mismatch(
    instance_response: Mapping[str, Any],
    zone_response: Mapping[str, Any],
    match: str,
) -> None:
    ec2 = _Ec2Client(
        instance_response=instance_response,
        zone_response=zone_response,
    )

    with pytest.raises(ValueError, match=match):
        collect_operator_readback(
            _identity(),
            ec2_client=ec2,
            iam_client=_IamClient(),
            sts_client=_StsClient(_sts_response()),
        )


def test_operator_readback_rejects_duplicate_instance_tags() -> None:
    instance = _instance()
    instance["Tags"] = [
        {"Key": "Name", "Value": "first"},
        {"Key": "Name", "Value": "second"},
    ]
    ec2 = _Ec2Client(instance_response=_describe_instances_response([_reservation([instance])]))

    with pytest.raises(ValueError, match="duplicate tag key 'Name'"):
        collect_operator_readback(
            _identity(),
            ec2_client=ec2,
            iam_client=_IamClient(),
            sts_client=_StsClient(_sts_response()),
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("spot", "spot"),
        (None, "on-demand"),
    ],
)
def test_operator_readback_records_ec2_instance_lifecycle(
    value: str | None,
    expected: str,
) -> None:
    instance = _instance()
    if value is None:
        del instance["InstanceLifecycle"]
    else:
        instance["InstanceLifecycle"] = value
    ec2 = _Ec2Client(instance_response=_describe_instances_response([_reservation([instance])]))

    readback = collect_operator_readback(
        _identity(),
        ec2_client=ec2,
        iam_client=_IamClient(),
        sts_client=_StsClient(_sts_response()),
    )

    assert readback.instance_lifecycle == expected
    assert readback.to_record()["instance_lifecycle"] == expected


def test_operator_readback_rejects_unsupported_instance_lifecycle() -> None:
    instance = _instance()
    instance["InstanceLifecycle"] = "capacity-block"
    ec2 = _Ec2Client(instance_response=_describe_instances_response([_reservation([instance])]))

    with pytest.raises(ValueError, match="InstanceLifecycle"):
        collect_operator_readback(
            _identity(),
            ec2_client=ec2,
            iam_client=_IamClient(),
            sts_client=_StsClient(_sts_response()),
        )


def test_operator_collection_rejects_extra_service_envelope_fields() -> None:
    response = {
        **_describe_instances_response(),
        "Unexpected": "value",
    }

    with pytest.raises(ValueError, match="extra=\\['Unexpected'\\]"):
        collect_operator_readback(
            _identity(),
            ec2_client=_Ec2Client(instance_response=response),
            iam_client=_IamClient(),
            sts_client=_StsClient(_sts_response()),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("owner_account_id", OTHER_ACCOUNT_ID),
        ("architecture", "arm64"),
        ("instance_id", OTHER_INSTANCE_ID),
        ("image_id", OTHER_IMAGE_ID),
        ("instance_type", "m6i.8xlarge"),
        ("region", "us-west-2"),
        ("availability_zone", "us-east-1b"),
        ("availability_zone_id", "use1-az2"),
    ],
)
def test_operator_validation_rejects_every_signed_scope_mismatch(
    field: str,
    value: str,
) -> None:
    evidence = _identity()
    record = _operator_record(evidence)
    record[field] = value

    with pytest.raises(ValueError):
        validate_operator_readback(evidence, record)


def test_operator_validation_rejects_operator_sts_account_mismatch() -> None:
    evidence = _identity()
    record = _operator_record(evidence)
    record["operator_identity"] = _caller(OTHER_ACCOUNT_ID).to_record()

    with pytest.raises(ValueError, match="operator_account_id"):
        validate_operator_readback(evidence, record)


@pytest.mark.parametrize(
    "mutation",
    ["record", "operator_identity", "tag", "unsorted_tags", "duplicate_tags"],
)
def test_operator_validation_rejects_noncanonical_or_extra_schema(
    mutation: str,
) -> None:
    evidence = _identity()
    record = _operator_record(evidence)
    if mutation == "record":
        record["unexpected"] = "value"
    elif mutation == "operator_identity":
        record["operator_identity"]["unexpected"] = "value"
    elif mutation == "tag":
        record["tags"][0]["unexpected"] = "value"
    elif mutation == "unsorted_tags":
        record["tags"] = list(reversed(record["tags"]))
    else:
        record["tags"].append(copy.deepcopy(record["tags"][0]))

    with pytest.raises((TypeError, ValueError)):
        validate_operator_readback(evidence, record)
