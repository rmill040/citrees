"""Signed EC2 instance identity and independent operator readback."""

from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SUPPORTED_REGION = "us-east-1"
IMDS_BASE_URL = "http://169.254.169.254/latest"
IMDS_TOKEN_TTL_SECONDS = 60

# Pinned from the us-east-1 / RSA-2048 section of the official AWS source:
# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/regions-certs.md
# DER SHA-256: 70923c22dd074172d0e5d27ea1ca865412525fccd8bc2cd0ff40660e275f0b30
US_EAST_1_RSA2048_CERTIFICATE_PEM = b"""-----BEGIN CERTIFICATE-----
MIIEEjCCAvqgAwIBAgIJALFpzEAVWaQZMA0GCSqGSIb3DQEBCwUAMFwxCzAJBgNV
BAYTAlVTMRkwFwYDVQQIExBXYXNoaW5ndG9uIFN0YXRlMRAwDgYDVQQHEwdTZWF0
dGxlMSAwHgYDVQQKExdBbWF6b24gV2ViIFNlcnZpY2VzIExMQzAgFw0xNTA4MTQw
ODU5MTJaGA8yMTk1MDExNzA4NTkxMlowXDELMAkGA1UEBhMCVVMxGTAXBgNVBAgT
EFdhc2hpbmd0b24gU3RhdGUxEDAOBgNVBAcTB1NlYXR0bGUxIDAeBgNVBAoTF0Ft
YXpvbiBXZWIgU2VydmljZXMgTExDMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIB
CgKCAQEAjS2vqZu9mEOhOq+0bRpAbCUiapbZMFNQqRg7kTlr7Cf+gDqXKpHPjsng
SfNz+JHQd8WPI+pmNs+q0Z2aTe23klmf2U52KH9/j1k8RlIbap/yFibFTSedmegX
E5r447GbJRsHUmuIIfZTZ/oRlpuIIO5/Vz7SOj22tdkdY2ADp7caZkNxhSP915fk
2jJMTBUOzyXUS2rBU/ulNHbTTeePjcEkvzVYPahD30TeQ+/A+uWUu89bHSQOJR8h
Um4cFApzZgN3aD5j2LrSMu2pctkQwf9CaWyVznqrsGYjYOY66LuFzSCXwqSnFBfv
fFBAFsjCgY24G2DoMyYkF3MyZlu+rwIDAQABo4HUMIHRMAsGA1UdDwQEAwIHgDAd
BgNVHQ4EFgQUrynSPp4uqSECwy+PiO4qyJ8TWSkwgY4GA1UdIwSBhjCBg4AUrynS
Pp4uqSECwy+PiO4qyJ8TWSmhYKReMFwxCzAJBgNVBAYTAlVTMRkwFwYDVQQIExBX
YXNoaW5ndG9uIFN0YXRlMRAwDgYDVQQHEwdTZWF0dGxlMSAwHgYDVQQKExdBbWF6
b24gV2ViIFNlcnZpY2VzIExMQ4IJALFpzEAVWaQZMBIGA1UdEwEB/wQIMAYBAf8C
AQAwDQYJKoZIhvcNAQELBQADggEBADW/s8lXijwdP6NkEoH1m9XLrvK4YTqkNfR6
er/uRRgTx2QjFcMNrx+g87gAml11z+D0crAZ5LbEhDMs+JtZYR3ty0HkDk6SJM85
haoJNAFF7EQ/zCp1EJRIkLLsC7bcDL/Eriv1swt78/BB4RnC9W9kSp/sxd5svJMg
N9a6FAplpNRsWAnbP8JBlAP93oJzblX2LQXgykTghMkQO7NaY5hg/H5o4dMPclTK
lYGqlFUCH6A2vdrxmpKDLmTn5//5pujdD2MN0df6sZWtxwZ0osljV4rDjm9Q3VpA
NWIsDEcp3GUB4proOR+C7PNkY+VGODitBOw09qBGosCBstwyEqY=
-----END CERTIFICATE-----
"""

_DOCUMENT_FIELDS = frozenset(
    {
        "accountId",
        "architecture",
        "availabilityZone",
        "billingProducts",
        "devpayProductCodes",
        "imageId",
        "instanceId",
        "instanceType",
        "kernelId",
        "marketplaceProductCodes",
        "pendingTime",
        "privateIp",
        "ramdiskId",
        "region",
        "version",
    }
)
_CALLER_IDENTITY_FIELDS = frozenset({"account_id", "arn", "user_id"})
_INSTANCE_IDENTITY_RECORD_FIELDS = frozenset(
    {
        "availability_zone_id",
        "raw_document_base64",
        "rsa2048_signature",
        "sts_identity",
    }
)
_OPERATOR_READBACK_FIELDS = frozenset(
    {
        "architecture",
        "availability_zone",
        "availability_zone_id",
        "iam_instance_profile_arn",
        "iam_role_arn",
        "image_id",
        "instance_id",
        "instance_type",
        "operator_identity",
        "owner_account_id",
        "region",
        "state",
        "tags",
    }
)
_AWS_ACCOUNT_ID_PATTERN = re.compile(r"^[0-9]{12}$")
_AWS_ARN_PATTERN = re.compile(r"^arn:aws:(?:iam|sts)::(?P<account_id>[0-9]{12}):[^ \t\r\n]+$")
_IAM_RESOURCE_ARN_PATTERN = re.compile(
    r"^arn:aws:iam::(?P<account_id>[0-9]{12}):"
    r"(?P<resource_kind>instance-profile|role)/"
    r"(?P<resource_name>[A-Za-z0-9+=,.@_-]+(?:/[A-Za-z0-9+=,.@_-]+)*)$"
)
_INSTANCE_ID_PATTERN = re.compile(r"^i-[0-9a-f]{8}(?:[0-9a-f]{9})?$")
_IMAGE_ID_PATTERN = re.compile(r"^ami-[0-9a-f]{8}(?:[0-9a-f]{9})?$")
_INSTANCE_TYPE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*\.[a-z0-9][a-z0-9-]*$")
_AVAILABILITY_ZONE_PATTERN = re.compile(
    rf"^{SUPPORTED_REGION}(?:[a-z]|-[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)$"
)
_AVAILABILITY_ZONE_ID_PATTERN = re.compile(r"^use1-[a-z0-9]+(?:-[a-z0-9]+)*$")
_DOCUMENT_VERSION = "2017-09-30"
_ARCHITECTURES = frozenset({"arm64", "i386", "x86_64"})
_STS_SERVICE_FIELDS = frozenset({"Account", "Arn", "ResponseMetadata", "UserId"})
_DESCRIBE_INSTANCES_FIELDS = frozenset({"Reservations", "ResponseMetadata"})
_DESCRIBE_AVAILABILITY_ZONES_FIELDS = frozenset({"AvailabilityZones", "ResponseMetadata"})
_GET_INSTANCE_PROFILE_FIELDS = frozenset({"InstanceProfile", "ResponseMetadata"})

ImdsRequest = Callable[[str, str, Mapping[str, str]], bytes]
SignatureVerifier = Callable[[bytes, bytes, bytes], None]


@dataclass(frozen=True)
class AwsCallerIdentity:
    """Normalized output from one live STS GetCallerIdentity call."""

    account_id: str
    arn: str
    user_id: str

    def to_record(self) -> dict[str, str]:
        """Return the strict canonical caller-identity record."""
        return {
            "account_id": self.account_id,
            "arn": self.arn,
            "user_id": self.user_id,
        }


@dataclass(frozen=True)
class SignedInstanceIdentity:
    """Strictly parsed fields from one signed EC2 identity document."""

    account_id: str
    architecture: str
    availability_zone: str
    billing_products: tuple[str, ...] | None
    devpay_product_codes: tuple[str, ...] | None
    image_id: str
    instance_id: str
    instance_type: str
    kernel_id: str | None
    marketplace_product_codes: tuple[str, ...] | None
    pending_time: str
    private_ip: str
    ramdisk_id: str | None
    region: str
    version: str


@dataclass(frozen=True)
class InstanceIdentityEvidence:
    """Verified raw EC2 evidence plus its normalized signed identity."""

    raw_document: bytes
    rsa2048_signature: bytes
    availability_zone_id: str
    sts_identity: AwsCallerIdentity
    identity: SignedInstanceIdentity

    def to_record(self) -> dict[str, Any]:
        """Return a strict JSON-safe record without duplicating signed fields."""
        if not isinstance(self.raw_document, bytes) or not self.raw_document:
            raise TypeError("instance identity document must be non-empty bytes")
        if not isinstance(self.sts_identity, AwsCallerIdentity):
            raise TypeError("sts_identity must be an AwsCallerIdentity")
        availability_zone_id = _require_availability_zone_id(
            self.availability_zone_id,
            source="IMDS availability-zone-id",
        )
        signature = _canonical_base64_text(
            self.rsa2048_signature,
            source="RSA-2048 PKCS7 signature",
        )
        sts_identity = _parse_caller_identity_record(self.sts_identity.to_record())
        return {
            "availability_zone_id": availability_zone_id,
            "raw_document_base64": base64.b64encode(self.raw_document).decode("ascii"),
            "rsa2048_signature": signature,
            "sts_identity": sts_identity.to_record(),
        }


@dataclass(frozen=True)
class OperatorInstanceReadback:
    """Canonical independent EC2 and operator-STS readback."""

    owner_account_id: str
    operator_identity: AwsCallerIdentity
    architecture: str
    instance_id: str
    image_id: str
    instance_type: str
    region: str
    availability_zone: str
    availability_zone_id: str
    iam_instance_profile_arn: str
    iam_role_arn: str
    state: str
    tags: tuple[tuple[str, str], ...]

    def to_record(self) -> dict[str, Any]:
        """Return the strict canonical operator-readback record."""
        return {
            "architecture": self.architecture,
            "availability_zone": self.availability_zone,
            "availability_zone_id": self.availability_zone_id,
            "iam_instance_profile_arn": self.iam_instance_profile_arn,
            "iam_role_arn": self.iam_role_arn,
            "image_id": self.image_id,
            "instance_id": self.instance_id,
            "instance_type": self.instance_type,
            "operator_identity": self.operator_identity.to_record(),
            "owner_account_id": self.owner_account_id,
            "region": self.region,
            "state": self.state,
            "tags": [{"key": key, "value": value} for key, value in self.tags],
        }


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    source: str,
) -> None:
    keys = set(value)
    if not all(isinstance(key, str) for key in keys):
        raise ValueError(f"{source} field names must be strings")
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(f"{source} fields differ: missing={missing}, extra={extra}")


def _require_service_fields(
    value: Any,
    *,
    required: frozenset[str],
    allowed: frozenset[str],
    source: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{source} must be an object")
    keys = set(value)
    if not all(isinstance(key, str) for key in keys):
        raise ValueError(f"{source} field names must be strings")
    missing = sorted(required - keys)
    extra = sorted(keys - allowed)
    if missing or extra:
        raise ValueError(f"{source} fields differ: missing={missing}, extra={extra}")
    metadata = value.get("ResponseMetadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError(f"{source}.ResponseMetadata must be an object")
    return value


def _require_concrete_string(value: Any, *, source: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{source} must be a non-empty string without outer whitespace")
    return value


def _require_account_id(value: Any, *, source: str) -> str:
    account_id = _require_concrete_string(value, source=source)
    if not _AWS_ACCOUNT_ID_PATTERN.fullmatch(account_id):
        raise ValueError(f"{source} must be a 12-digit AWS account ID")
    return account_id


def _require_instance_id(value: Any, *, source: str) -> str:
    instance_id = _require_concrete_string(value, source=source)
    if not _INSTANCE_ID_PATTERN.fullmatch(instance_id):
        raise ValueError(f"{source} is not a valid EC2 instance ID")
    return instance_id


def _require_image_id(value: Any, *, source: str) -> str:
    image_id = _require_concrete_string(value, source=source)
    if not _IMAGE_ID_PATTERN.fullmatch(image_id):
        raise ValueError(f"{source} is not a valid EC2 AMI ID")
    return image_id


def _require_instance_type(value: Any, *, source: str) -> str:
    instance_type = _require_concrete_string(value, source=source)
    if not _INSTANCE_TYPE_PATTERN.fullmatch(instance_type):
        raise ValueError(f"{source} is not a valid EC2 instance type")
    return instance_type


def _require_architecture(value: Any, *, source: str) -> str:
    architecture = _require_concrete_string(value, source=source)
    if architecture not in _ARCHITECTURES:
        raise ValueError(f"{source} is not a supported EC2 architecture")
    return architecture


def _require_availability_zone(value: Any, *, source: str) -> str:
    availability_zone = _require_concrete_string(value, source=source)
    if not _AVAILABILITY_ZONE_PATTERN.fullmatch(availability_zone):
        raise ValueError(f"{source} must identify a {SUPPORTED_REGION} availability zone")
    return availability_zone


def _require_availability_zone_id(value: Any, *, source: str) -> str:
    availability_zone_id = _require_concrete_string(value, source=source)
    if not _AVAILABILITY_ZONE_ID_PATTERN.fullmatch(availability_zone_id):
        raise ValueError(f"{source} is not a valid availability-zone ID")
    return availability_zone_id


def _require_iam_resource_arn(
    value: Any,
    *,
    resource_kind: str,
    source: str,
) -> tuple[str, str, str]:
    arn = _require_concrete_string(value, source=source)
    match = _IAM_RESOURCE_ARN_PATTERN.fullmatch(arn)
    if match is None or match.group("resource_kind") != resource_kind:
        raise ValueError(f"{source} is not a valid IAM {resource_kind} ARN")
    return arn, match.group("account_id"), match.group("resource_name")


def _optional_string_tuple(value: Any, *, source: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError(f"{source} must be null or a list")
    return tuple(
        _require_concrete_string(item, source=f"{source}[{index}]")
        for index, item in enumerate(value)
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"instance identity document contains duplicate field {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r} is not allowed")


def _parse_instance_identity_document(raw_document: bytes) -> SignedInstanceIdentity:
    if not isinstance(raw_document, bytes) or not raw_document:
        raise TypeError("instance identity document must be non-empty bytes")
    try:
        text = raw_document.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("instance identity document must be UTF-8 encoded") from exc
    if text.startswith("\ufeff"):
        raise ValueError("instance identity document must not contain a byte-order mark")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid instance identity document JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise TypeError("instance identity document must be an object")
    _require_exact_fields(value, _DOCUMENT_FIELDS, source="instance identity document")

    region = _require_concrete_string(
        value["region"],
        source="instance identity document.region",
    )
    if region != SUPPORTED_REGION:
        raise ValueError(f"instance identity document.region must equal {SUPPORTED_REGION}")
    architecture = _require_concrete_string(
        value["architecture"],
        source="instance identity document.architecture",
    )
    if architecture not in _ARCHITECTURES:
        raise ValueError("instance identity document.architecture is invalid")
    version = _require_concrete_string(
        value["version"],
        source="instance identity document.version",
    )
    if version != _DOCUMENT_VERSION:
        raise ValueError(f"instance identity document.version must equal {_DOCUMENT_VERSION}")

    pending_time = _require_concrete_string(
        value["pendingTime"],
        source="instance identity document.pendingTime",
    )
    if not pending_time.endswith("Z"):
        raise ValueError("instance identity document.pendingTime must be an ISO-8601 UTC time")
    try:
        parsed_pending_time = datetime.fromisoformat(pending_time[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(
            "instance identity document.pendingTime must be an ISO-8601 UTC time"
        ) from exc
    if parsed_pending_time.utcoffset() is None:
        raise ValueError("instance identity document.pendingTime must include a UTC offset")

    private_ip = _require_concrete_string(
        value["privateIp"],
        source="instance identity document.privateIp",
    )
    try:
        ipaddress.ip_address(private_ip)
    except ValueError as exc:
        raise ValueError("instance identity document.privateIp is invalid") from exc

    kernel_id = value["kernelId"]
    if kernel_id is not None:
        kernel_id = _require_concrete_string(
            kernel_id,
            source="instance identity document.kernelId",
        )
    ramdisk_id = value["ramdiskId"]
    if ramdisk_id is not None:
        ramdisk_id = _require_concrete_string(
            ramdisk_id,
            source="instance identity document.ramdiskId",
        )

    return SignedInstanceIdentity(
        account_id=_require_account_id(
            value["accountId"],
            source="instance identity document.accountId",
        ),
        architecture=architecture,
        availability_zone=_require_availability_zone(
            value["availabilityZone"],
            source="instance identity document.availabilityZone",
        ),
        billing_products=_optional_string_tuple(
            value["billingProducts"],
            source="instance identity document.billingProducts",
        ),
        devpay_product_codes=_optional_string_tuple(
            value["devpayProductCodes"],
            source="instance identity document.devpayProductCodes",
        ),
        image_id=_require_image_id(
            value["imageId"],
            source="instance identity document.imageId",
        ),
        instance_id=_require_instance_id(
            value["instanceId"],
            source="instance identity document.instanceId",
        ),
        instance_type=_require_instance_type(
            value["instanceType"],
            source="instance identity document.instanceType",
        ),
        kernel_id=kernel_id,
        marketplace_product_codes=_optional_string_tuple(
            value["marketplaceProductCodes"],
            source="instance identity document.marketplaceProductCodes",
        ),
        pending_time=pending_time,
        private_ip=private_ip,
        ramdisk_id=ramdisk_id,
        region=region,
        version=version,
    )


def _decode_base64_payload(payload: bytes, *, source: str) -> bytes:
    if not isinstance(payload, bytes) or not payload:
        raise TypeError(f"{source} must be non-empty bytes")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source} must be ASCII") from exc
    compact = text.replace("\r", "").replace("\n", "")
    if not compact or any(character.isspace() for character in compact):
        raise ValueError(f"{source} contains invalid whitespace")
    try:
        decoded = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{source} is not valid base64") from exc
    if not decoded or base64.b64encode(decoded).decode("ascii") != compact:
        raise ValueError(f"{source} is not canonical base64")
    return decoded


def _canonical_base64_text(payload: bytes, *, source: str) -> str:
    decoded = _decode_base64_payload(payload, source=source)
    canonical = base64.b64encode(decoded).decode("ascii")
    text = payload.decode("ascii")
    if text != canonical:
        raise ValueError(f"{source} must use canonical base64 without whitespace")
    return text


def _decode_canonical_base64_record_value(value: Any, *, source: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{source} must be a non-empty base64 string")
    try:
        payload = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{source} must be ASCII") from exc
    decoded = _decode_base64_payload(payload, source=source)
    if base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError(f"{source} must use canonical base64 without whitespace")
    return decoded


def _pem_from_base64_payload(payload: bytes, *, label: str, source: str) -> bytes:
    decoded = _decode_base64_payload(payload, source=source)
    canonical = base64.b64encode(decoded).decode("ascii")
    body = "\n".join(canonical[index : index + 64] for index in range(0, len(canonical), 64))
    return f"-----BEGIN {label}-----\n{body}\n-----END {label}-----\n".encode("ascii")


def _normalize_certificate_pem(certificate_pem: bytes) -> bytes:
    if not isinstance(certificate_pem, bytes) or not certificate_pem:
        raise TypeError("RSA-2048 certificate must be non-empty bytes")
    try:
        lines = certificate_pem.decode("ascii").strip().splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("RSA-2048 certificate must be ASCII PEM") from exc
    if (
        len(lines) < 3
        or lines[0] != "-----BEGIN CERTIFICATE-----"
        or lines[-1] != "-----END CERTIFICATE-----"
    ):
        raise ValueError("RSA-2048 certificate must contain exactly one PEM certificate")
    body = "".join(lines[1:-1]).encode("ascii")
    return _pem_from_base64_payload(
        body,
        label="CERTIFICATE",
        source="RSA-2048 certificate body",
    )


def verify_pkcs7_signature(
    raw_document: bytes,
    rsa2048_signature: bytes,
    certificate_pem: bytes = US_EAST_1_RSA2048_CERTIFICATE_PEM,
) -> None:
    """Verify an attached RSA-2048 PKCS7 signature and its exact document bytes."""
    if not isinstance(raw_document, bytes) or not raw_document:
        raise TypeError("instance identity document must be non-empty bytes")
    signature_pem = _pem_from_base64_payload(
        rsa2048_signature,
        label="PKCS7",
        source="RSA-2048 PKCS7 signature",
    )
    normalized_certificate = _normalize_certificate_pem(certificate_pem)

    with tempfile.TemporaryDirectory(prefix="citrees-instance-identity-") as directory:
        root = Path(directory)
        signature_path = root / "signature.pem"
        certificate_path = root / "certificate.pem"
        signature_path.write_bytes(signature_pem)
        certificate_path.write_bytes(normalized_certificate)
        command = [
            "openssl",
            "smime",
            "-verify",
            "-binary",
            "-inform",
            "PEM",
            "-in",
            str(signature_path),
            "-certfile",
            str(certificate_path),
            "-nointern",
            "-noverify",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                check=False,
                timeout=10,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("openssl is required to verify EC2 instance identity") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("openssl timed out while verifying EC2 instance identity") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"EC2 instance identity signature verification failed: {detail}")
    if result.stdout != raw_document:
        raise ValueError("verified PKCS7 content does not exactly match the raw identity document")


def _parse_caller_identity_record(value: Mapping[str, Any]) -> AwsCallerIdentity:
    _require_exact_fields(value, _CALLER_IDENTITY_FIELDS, source="caller identity")
    account_id = _require_account_id(value["account_id"], source="caller identity.account_id")
    arn = _require_concrete_string(value["arn"], source="caller identity.arn")
    match = _AWS_ARN_PATTERN.fullmatch(arn)
    if match is None or match.group("account_id") != account_id:
        raise ValueError("caller identity ARN does not match its AWS account")
    user_id = _require_concrete_string(value["user_id"], source="caller identity.user_id")
    return AwsCallerIdentity(account_id=account_id, arn=arn, user_id=user_id)


def get_assumed_role_identity(arn: str) -> tuple[str, str] | None:
    """Return an STS assumed-role name and session name, if present."""
    normalized = _require_concrete_string(arn, source="AWS ARN")
    match = _AWS_ARN_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError("AWS ARN is invalid")
    resource = normalized.split(":", maxsplit=5)[5]
    prefix = "assumed-role/"
    if not resource.startswith(prefix):
        return None
    role_name, separator, session_name = resource[len(prefix) :].rpartition("/")
    if not separator or not role_name or not session_name:
        raise ValueError("STS assumed-role ARN is invalid")
    return role_name, session_name


def collect_aws_caller_identity(*, sts_client: Any | None = None) -> AwsCallerIdentity:
    """Collect and normalize one live STS caller identity."""
    if sts_client is None:
        import boto3

        sts_client = boto3.client("sts", region_name=SUPPORTED_REGION)
    response = _require_service_fields(
        sts_client.get_caller_identity(),
        required=frozenset({"Account", "Arn", "UserId"}),
        allowed=_STS_SERVICE_FIELDS,
        source="STS GetCallerIdentity response",
    )
    return _parse_caller_identity_record(
        {
            "account_id": response["Account"],
            "arn": response["Arn"],
            "user_id": response["UserId"],
        }
    )


def validate_instance_identity(
    raw_document: bytes,
    rsa2048_signature: bytes,
    availability_zone_id: str,
    sts_identity: AwsCallerIdentity,
    *,
    certificate_pem: bytes = US_EAST_1_RSA2048_CERTIFICATE_PEM,
    signature_verifier: SignatureVerifier = verify_pkcs7_signature,
) -> InstanceIdentityEvidence:
    """Verify and cross-check one complete in-instance identity evidence set."""
    if not isinstance(sts_identity, AwsCallerIdentity):
        raise TypeError("sts_identity must be an AwsCallerIdentity")
    normalized_sts = _parse_caller_identity_record(sts_identity.to_record())
    normalized_zone_id = _require_availability_zone_id(
        availability_zone_id,
        source="IMDS availability-zone-id",
    )
    signature_der = _decode_base64_payload(
        rsa2048_signature,
        source="RSA-2048 PKCS7 signature",
    )
    canonical_signature = base64.b64encode(signature_der)
    signature_verifier(raw_document, canonical_signature, certificate_pem)
    identity = _parse_instance_identity_document(raw_document)
    if identity.account_id != normalized_sts.account_id:
        raise ValueError("signed accountId does not match the live instance STS account")
    assumed_role = get_assumed_role_identity(normalized_sts.arn)
    if assumed_role is None or assumed_role[1] != identity.instance_id:
        raise ValueError(
            "live instance STS identity must be an assumed-role session named "
            "for the signed instance ID"
        )
    return InstanceIdentityEvidence(
        raw_document=raw_document,
        rsa2048_signature=canonical_signature,
        availability_zone_id=normalized_zone_id,
        sts_identity=normalized_sts,
        identity=identity,
    )


def validate_instance_identity_record(
    record: Mapping[str, Any],
    *,
    certificate_pem: bytes = US_EAST_1_RSA2048_CERTIFICATE_PEM,
    signature_verifier: SignatureVerifier = verify_pkcs7_signature,
) -> InstanceIdentityEvidence:
    """Parse and cryptographically revalidate one canonical evidence record."""
    if not isinstance(record, Mapping):
        raise TypeError("instance identity evidence record must be an object")
    _require_exact_fields(
        record,
        _INSTANCE_IDENTITY_RECORD_FIELDS,
        source="instance identity evidence record",
    )
    raw_document = _decode_canonical_base64_record_value(
        record["raw_document_base64"],
        source="instance identity evidence record.raw_document_base64",
    )
    signature_value = record["rsa2048_signature"]
    if not isinstance(signature_value, str):
        raise TypeError(
            "instance identity evidence record.rsa2048_signature must be a non-empty base64 string"
        )
    _decode_canonical_base64_record_value(
        signature_value,
        source="instance identity evidence record.rsa2048_signature",
    )
    sts_record = record["sts_identity"]
    if not isinstance(sts_record, Mapping):
        raise TypeError("instance identity evidence record.sts_identity must be an object")
    return validate_instance_identity(
        raw_document,
        signature_value.encode("ascii"),
        record["availability_zone_id"],
        _parse_caller_identity_record(sts_record),
        certificate_pem=certificate_pem,
        signature_verifier=signature_verifier,
    )


def _default_imds_request(
    method: str,
    path: str,
    headers: Mapping[str, str],
) -> bytes:
    request = urllib.request.Request(
        IMDS_BASE_URL + path,
        headers=dict(headers),
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            if response.status != 200:
                raise RuntimeError(f"IMDS returned unexpected HTTP status {response.status}")
            payload = response.read()
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        raise RuntimeError(f"IMDS request failed for {path}") from exc
    if not isinstance(payload, bytes) or not payload:
        raise RuntimeError(f"IMDS returned an empty response for {path}")
    return payload


def _parse_imds_token(payload: bytes) -> str:
    if not isinstance(payload, bytes) or not payload:
        raise RuntimeError("IMDSv2 returned an empty token")
    try:
        token = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeError("IMDSv2 token must be ASCII") from exc
    if any(ord(character) < 0x21 or ord(character) > 0x7E for character in token):
        raise RuntimeError("IMDSv2 token contains whitespace or control characters")
    return token


def _parse_imds_text(payload: bytes, *, source: str) -> str:
    if not isinstance(payload, bytes) or not payload:
        raise RuntimeError(f"{source} is empty")
    try:
        value = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{source} must be ASCII") from exc
    if not value or value != value.strip():
        raise RuntimeError(f"{source} contains outer whitespace")
    return value


def collect_instance_identity(
    *,
    sts_client: Any | None = None,
    imds_request: ImdsRequest | None = None,
    certificate_pem: bytes = US_EAST_1_RSA2048_CERTIFICATE_PEM,
    signature_verifier: SignatureVerifier = verify_pkcs7_signature,
) -> InstanceIdentityEvidence:
    """Collect and verify signed identity evidence from one EC2 instance."""
    request = _default_imds_request if imds_request is None else imds_request
    token = _parse_imds_token(
        request(
            "PUT",
            "/api/token",
            {"X-aws-ec2-metadata-token-ttl-seconds": str(IMDS_TOKEN_TTL_SECONDS)},
        )
    )
    token_header = {"X-aws-ec2-metadata-token": token}
    raw_document = request(
        "GET",
        "/dynamic/instance-identity/document",
        token_header,
    )
    rsa2048_signature = request(
        "GET",
        "/dynamic/instance-identity/rsa2048",
        token_header,
    )
    availability_zone_id = _parse_imds_text(
        request(
            "GET",
            "/meta-data/placement/availability-zone-id",
            token_header,
        ),
        source="IMDS availability-zone-id",
    )
    return validate_instance_identity(
        raw_document,
        rsa2048_signature,
        availability_zone_id,
        collect_aws_caller_identity(sts_client=sts_client),
        certificate_pem=certificate_pem,
        signature_verifier=signature_verifier,
    )


def _normalize_aws_tags(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError("DescribeInstances instance.Tags must be a list")
    tags: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, tag in enumerate(value):
        source = f"DescribeInstances instance.Tags[{index}]"
        if not isinstance(tag, Mapping):
            raise TypeError(f"{source} must be an object")
        _require_exact_fields(tag, frozenset({"Key", "Value"}), source=source)
        key = _require_concrete_string(tag["Key"], source=f"{source}.Key")
        tag_value = tag["Value"]
        if not isinstance(tag_value, str):
            raise TypeError(f"{source}.Value must be a string")
        if key in seen:
            raise ValueError(f"DescribeInstances returned duplicate tag key {key!r}")
        seen.add(key)
        tags.append((key, tag_value))
    return tuple(sorted(tags))


def _parse_canonical_tags(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise TypeError("operator readback.tags must be a list")
    tags: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, tag in enumerate(value):
        source = f"operator readback.tags[{index}]"
        if not isinstance(tag, Mapping):
            raise TypeError(f"{source} must be an object")
        _require_exact_fields(tag, frozenset({"key", "value"}), source=source)
        key = _require_concrete_string(tag["key"], source=f"{source}.key")
        tag_value = tag["value"]
        if not isinstance(tag_value, str):
            raise TypeError(f"{source}.value must be a string")
        if key in seen:
            raise ValueError(f"operator readback contains duplicate tag key {key!r}")
        seen.add(key)
        tags.append((key, tag_value))
    normalized = tuple(sorted(tags))
    if tuple(tags) != normalized:
        raise ValueError("operator readback.tags must be sorted by key and value")
    return normalized


def validate_operator_readback(
    identity_evidence: InstanceIdentityEvidence,
    record: Mapping[str, Any],
) -> OperatorInstanceReadback:
    """Validate one canonical operator record against signed instance evidence."""
    if not isinstance(identity_evidence, InstanceIdentityEvidence):
        raise TypeError("identity_evidence must be an InstanceIdentityEvidence")
    if not isinstance(record, Mapping):
        raise TypeError("operator readback must be an object")
    _require_exact_fields(record, _OPERATOR_READBACK_FIELDS, source="operator readback")
    operator_record = record["operator_identity"]
    if not isinstance(operator_record, Mapping):
        raise TypeError("operator readback.operator_identity must be an object")
    readback = OperatorInstanceReadback(
        owner_account_id=_require_account_id(
            record["owner_account_id"],
            source="operator readback.owner_account_id",
        ),
        operator_identity=_parse_caller_identity_record(operator_record),
        architecture=_require_architecture(
            record["architecture"],
            source="operator readback.architecture",
        ),
        instance_id=_require_instance_id(
            record["instance_id"],
            source="operator readback.instance_id",
        ),
        image_id=_require_image_id(
            record["image_id"],
            source="operator readback.image_id",
        ),
        instance_type=_require_instance_type(
            record["instance_type"],
            source="operator readback.instance_type",
        ),
        region=_require_concrete_string(
            record["region"],
            source="operator readback.region",
        ),
        availability_zone=_require_availability_zone(
            record["availability_zone"],
            source="operator readback.availability_zone",
        ),
        availability_zone_id=_require_availability_zone_id(
            record["availability_zone_id"],
            source="operator readback.availability_zone_id",
        ),
        iam_instance_profile_arn=_require_iam_resource_arn(
            record["iam_instance_profile_arn"],
            resource_kind="instance-profile",
            source="operator readback.iam_instance_profile_arn",
        )[0],
        iam_role_arn=_require_iam_resource_arn(
            record["iam_role_arn"],
            resource_kind="role",
            source="operator readback.iam_role_arn",
        )[0],
        state=_require_concrete_string(
            record["state"],
            source="operator readback.state",
        ),
        tags=_parse_canonical_tags(record["tags"]),
    )
    if readback.region != SUPPORTED_REGION:
        raise ValueError(f"operator readback.region must equal {SUPPORTED_REGION}")

    signed = identity_evidence.identity
    _, profile_account_id, _ = _require_iam_resource_arn(
        readback.iam_instance_profile_arn,
        resource_kind="instance-profile",
        source="operator readback.iam_instance_profile_arn",
    )
    _, role_account_id, role_resource_name = _require_iam_resource_arn(
        readback.iam_role_arn,
        resource_kind="role",
        source="operator readback.iam_role_arn",
    )
    assumed_role = get_assumed_role_identity(identity_evidence.sts_identity.arn)
    if (
        assumed_role is None
        or assumed_role[0].rsplit("/", maxsplit=1)[-1]
        != role_resource_name.rsplit("/", maxsplit=1)[-1]
    ):
        raise ValueError("in-instance STS role does not match the operator-read IAM role")
    expected = {
        "architecture": signed.architecture,
        "availability_zone": signed.availability_zone,
        "availability_zone_id": identity_evidence.availability_zone_id,
        "image_id": signed.image_id,
        "instance_id": signed.instance_id,
        "instance_type": signed.instance_type,
        "operator_account_id": signed.account_id,
        "owner_account_id": signed.account_id,
        "profile_account_id": signed.account_id,
        "region": signed.region,
        "role_account_id": signed.account_id,
    }
    observed = {
        "architecture": readback.architecture,
        "availability_zone": readback.availability_zone,
        "availability_zone_id": readback.availability_zone_id,
        "image_id": readback.image_id,
        "instance_id": readback.instance_id,
        "instance_type": readback.instance_type,
        "operator_account_id": readback.operator_identity.account_id,
        "owner_account_id": readback.owner_account_id,
        "profile_account_id": profile_account_id,
        "region": readback.region,
        "role_account_id": role_account_id,
    }
    mismatches = [field for field in expected if observed[field] != expected[field]]
    if mismatches:
        raise ValueError(
            "operator readback does not match signed instance identity: " + ", ".join(mismatches)
        )
    return readback


def collect_operator_readback(
    identity_evidence: InstanceIdentityEvidence,
    *,
    ec2_client: Any | None = None,
    iam_client: Any | None = None,
    sts_client: Any | None = None,
) -> OperatorInstanceReadback:
    """Collect and validate one independent operator DescribeInstances readback."""
    if not isinstance(identity_evidence, InstanceIdentityEvidence):
        raise TypeError("identity_evidence must be an InstanceIdentityEvidence")
    if ec2_client is None:
        import boto3

        ec2_client = boto3.client("ec2", region_name=SUPPORTED_REGION)
    if iam_client is None:
        import boto3

        iam_client = boto3.client("iam", region_name=SUPPORTED_REGION)
    instance_id = identity_evidence.identity.instance_id
    instance_response = _require_service_fields(
        ec2_client.describe_instances(InstanceIds=[instance_id]),
        required=frozenset({"Reservations"}),
        allowed=_DESCRIBE_INSTANCES_FIELDS,
        source="EC2 DescribeInstances response",
    )
    reservations = instance_response["Reservations"]
    if not isinstance(reservations, list) or len(reservations) != 1:
        observed = len(reservations) if isinstance(reservations, list) else "non-list"
        raise ValueError(
            f"DescribeInstances must return exactly one reservation, observed {observed}"
        )
    reservation = reservations[0]
    if not isinstance(reservation, Mapping):
        raise TypeError("DescribeInstances reservation must be an object")
    owner_account_id = _require_account_id(
        reservation.get("OwnerId"),
        source="DescribeInstances Reservation.OwnerId",
    )
    instances = reservation.get("Instances")
    if not isinstance(instances, list) or len(instances) != 1:
        observed = len(instances) if isinstance(instances, list) else "non-list"
        raise ValueError(f"DescribeInstances must return exactly one instance, observed {observed}")
    instance = instances[0]
    if not isinstance(instance, Mapping):
        raise TypeError("DescribeInstances instance must be an object")
    observed_instance_id = _require_instance_id(
        instance.get("InstanceId"),
        source="DescribeInstances instance.InstanceId",
    )
    if observed_instance_id != instance_id:
        raise ValueError("DescribeInstances returned a different instance ID")
    placement = instance.get("Placement")
    if not isinstance(placement, Mapping):
        raise TypeError("DescribeInstances instance.Placement must be an object")
    availability_zone = _require_availability_zone(
        placement.get("AvailabilityZone"),
        source="DescribeInstances instance.Placement.AvailabilityZone",
    )
    state = instance.get("State")
    if not isinstance(state, Mapping):
        raise TypeError("DescribeInstances instance.State must be an object")
    state_name = _require_concrete_string(
        state.get("Name"),
        source="DescribeInstances instance.State.Name",
    )
    instance_profile = instance.get("IamInstanceProfile")
    if not isinstance(instance_profile, Mapping):
        raise TypeError("DescribeInstances instance.IamInstanceProfile must be an object")
    _require_exact_fields(
        instance_profile,
        frozenset({"Arn", "Id"}),
        source="DescribeInstances instance.IamInstanceProfile",
    )
    instance_profile_arn, profile_account_id, profile_resource_name = _require_iam_resource_arn(
        instance_profile["Arn"],
        resource_kind="instance-profile",
        source="DescribeInstances instance.IamInstanceProfile.Arn",
    )
    if profile_account_id != owner_account_id:
        raise ValueError("instance profile ARN does not match the reservation owner account")
    profile_name = profile_resource_name.rsplit("/", maxsplit=1)[-1]
    profile_response = _require_service_fields(
        iam_client.get_instance_profile(InstanceProfileName=profile_name),
        required=frozenset({"InstanceProfile"}),
        allowed=_GET_INSTANCE_PROFILE_FIELDS,
        source="IAM GetInstanceProfile response",
    )
    observed_profile = profile_response["InstanceProfile"]
    if not isinstance(observed_profile, Mapping):
        raise TypeError("IAM GetInstanceProfile InstanceProfile must be an object")
    observed_profile_arn, observed_profile_account_id, _ = _require_iam_resource_arn(
        observed_profile.get("Arn"),
        resource_kind="instance-profile",
        source="IAM GetInstanceProfile InstanceProfile.Arn",
    )
    if (
        observed_profile_arn != instance_profile_arn
        or observed_profile_account_id != owner_account_id
    ):
        raise ValueError("IAM GetInstanceProfile returned a different instance profile")
    roles = observed_profile.get("Roles")
    if not isinstance(roles, list) or len(roles) != 1 or not isinstance(roles[0], Mapping):
        observed = len(roles) if isinstance(roles, list) else "non-list"
        raise ValueError(f"instance profile must contain exactly one IAM role, observed {observed}")
    iam_role_arn, role_account_id, _ = _require_iam_resource_arn(
        roles[0].get("Arn"),
        resource_kind="role",
        source="IAM GetInstanceProfile InstanceProfile.Roles[0].Arn",
    )
    if role_account_id != owner_account_id:
        raise ValueError("instance profile role ARN does not match the reservation owner account")
    tags = _normalize_aws_tags(instance.get("Tags"))

    zone_response = _require_service_fields(
        ec2_client.describe_availability_zones(ZoneNames=[availability_zone]),
        required=frozenset({"AvailabilityZones"}),
        allowed=_DESCRIBE_AVAILABILITY_ZONES_FIELDS,
        source="EC2 DescribeAvailabilityZones response",
    )
    zones = zone_response["AvailabilityZones"]
    if not isinstance(zones, list) or len(zones) != 1:
        observed = len(zones) if isinstance(zones, list) else "non-list"
        raise ValueError(
            f"DescribeAvailabilityZones must return exactly one zone, observed {observed}"
        )
    zone = zones[0]
    if not isinstance(zone, Mapping):
        raise TypeError("DescribeAvailabilityZones zone must be an object")
    observed_zone_name = _require_availability_zone(
        zone.get("ZoneName"),
        source="DescribeAvailabilityZones zone.ZoneName",
    )
    if observed_zone_name != availability_zone:
        raise ValueError("DescribeAvailabilityZones returned a different zone name")

    record = {
        "architecture": _require_architecture(
            instance.get("Architecture"),
            source="DescribeInstances instance.Architecture",
        ),
        "availability_zone": availability_zone,
        "availability_zone_id": _require_availability_zone_id(
            zone.get("ZoneId"),
            source="DescribeAvailabilityZones zone.ZoneId",
        ),
        "iam_instance_profile_arn": instance_profile_arn,
        "iam_role_arn": iam_role_arn,
        "image_id": _require_image_id(
            instance.get("ImageId"),
            source="DescribeInstances instance.ImageId",
        ),
        "instance_id": observed_instance_id,
        "instance_type": _require_instance_type(
            instance.get("InstanceType"),
            source="DescribeInstances instance.InstanceType",
        ),
        "operator_identity": collect_aws_caller_identity(sts_client=sts_client).to_record(),
        "owner_account_id": owner_account_id,
        "region": _require_concrete_string(
            zone.get("RegionName"),
            source="DescribeAvailabilityZones zone.RegionName",
        ),
        "state": state_name,
        "tags": [{"key": key, "value": value} for key, value in tags],
    }
    return validate_operator_readback(identity_evidence, record)
