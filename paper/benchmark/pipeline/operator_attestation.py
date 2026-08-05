"""Ed25519 attestations for independent operator-side EC2 readbacks."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from Crypto.PublicKey import ECC
from Crypto.Signature import eddsa

OPERATOR_ATTESTATION_ALGORITHM = "ed25519"
OPERATOR_ATTESTATION_SCHEMA_VERSION = 1
OPERATOR_ATTESTATION_PROFILE = "r_cforest_operator_readback"
OPERATOR_PUBLIC_KEY_FIELDS = frozenset(
    {
        "algorithm",
        "public_key_der_base64",
        "public_key_sha256",
    }
)
OPERATOR_ATTESTATION_FIELDS = frozenset(
    {
        "campaign_sha256",
        "manifest_sha256",
        "observed_at_utc",
        "profile",
        "readback",
        "run_payload_sha256s",
        "runtime_contract_sha256",
        "schema_version",
        "signature_base64",
    }
)

_ATTESTATION_PAYLOAD_FIELDS = OPERATOR_ATTESTATION_FIELDS - {"signature_base64"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_UTC_TIMESTAMP_PATTERN = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")


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


def _require_sha256(value: Any, *, source: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{source} must be 64 lowercase hexadecimal characters")
    return value


def _decode_canonical_base64(value: Any, *, source: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{source} must be a non-empty base64 string")
    try:
        payload = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{source} must be ASCII") from exc
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{source} must be valid base64") from exc
    if not decoded or base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError(f"{source} must use canonical base64")
    return decoded


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def utc_timestamp() -> str:
    """Return the current UTC time in the canonical attestation format."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc_timestamp(value: Any, *, source: str) -> datetime:
    """Parse one canonical whole-second UTC timestamp."""
    if not isinstance(value, str) or not _UTC_TIMESTAMP_PATTERN.fullmatch(value):
        raise ValueError(f"{source} must use YYYY-MM-DDTHH:MM:SSZ")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    if parsed.tzinfo != UTC:
        raise ValueError(f"{source} must use UTC")
    return parsed


def _import_private_key(private_key_pem: bytes) -> Any:
    if not isinstance(private_key_pem, bytes) or not private_key_pem:
        raise TypeError("operator private key must be non-empty PEM bytes")
    try:
        key = ECC.import_key(private_key_pem)
    except (ValueError, IndexError, TypeError) as exc:
        raise ValueError("operator private key is invalid") from exc
    if not key.has_private() or key.curve != "Ed25519":
        raise ValueError("operator private key must be an Ed25519 private key")
    return key


def operator_public_key_from_private_key(private_key_pem: bytes) -> dict[str, str]:
    """Return the canonical public-key record for one Ed25519 private key."""
    key = _import_private_key(private_key_pem)
    public_der = key.public_key().export_key(format="DER")
    return {
        "algorithm": OPERATOR_ATTESTATION_ALGORITHM,
        "public_key_der_base64": base64.b64encode(public_der).decode("ascii"),
        "public_key_sha256": hashlib.sha256(public_der).hexdigest(),
    }


def validate_operator_public_key(value: Mapping[str, Any]) -> dict[str, str]:
    """Validate and normalize one frozen Ed25519 public-key record."""
    if not isinstance(value, Mapping):
        raise TypeError("operator public key must be an object")
    _require_exact_fields(value, OPERATOR_PUBLIC_KEY_FIELDS, source="operator public key")
    if value["algorithm"] != OPERATOR_ATTESTATION_ALGORITHM:
        raise ValueError(
            f"operator public key algorithm must equal {OPERATOR_ATTESTATION_ALGORITHM}"
        )
    public_der = _decode_canonical_base64(
        value["public_key_der_base64"],
        source="operator public key public_key_der_base64",
    )
    try:
        key = ECC.import_key(public_der)
    except (ValueError, IndexError, TypeError) as exc:
        raise ValueError("operator public key DER is invalid") from exc
    if key.has_private() or key.curve != "Ed25519":
        raise ValueError("operator public key must contain only an Ed25519 public key")
    canonical_der = key.public_key().export_key(format="DER")
    if canonical_der != public_der:
        raise ValueError("operator public key DER is not canonical")
    digest = hashlib.sha256(public_der).hexdigest()
    if value["public_key_sha256"] != digest:
        raise ValueError("operator public key SHA-256 does not match its DER bytes")
    return {
        "algorithm": OPERATOR_ATTESTATION_ALGORITHM,
        "public_key_der_base64": base64.b64encode(public_der).decode("ascii"),
        "public_key_sha256": digest,
    }


def load_operator_public_key(path: Path) -> dict[str, str]:
    """Load an Ed25519 public key from PEM or DER and return its frozen record."""
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read operator public key {path}: {exc}") from exc
    try:
        key = ECC.import_key(payload)
    except (ValueError, IndexError, TypeError) as exc:
        raise ValueError(f"operator public key {path} is invalid") from exc
    if key.has_private() or key.curve != "Ed25519":
        raise ValueError("operator public-key file must contain only an Ed25519 public key")
    public_der = key.export_key(format="DER")
    return validate_operator_public_key(
        {
            "algorithm": OPERATOR_ATTESTATION_ALGORITHM,
            "public_key_der_base64": base64.b64encode(public_der).decode("ascii"),
            "public_key_sha256": hashlib.sha256(public_der).hexdigest(),
        }
    )


def load_operator_private_key(path: Path) -> bytes:
    """Load a mode-0600 Ed25519 private key from a local PEM file."""
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        payload = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read operator private key {path}: {exc}") from exc
    if mode & 0o077:
        raise PermissionError("operator private key must not be accessible by group or others")
    _import_private_key(payload)
    return payload


def _exclusive_write(path: Path, payload: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def generate_operator_keypair(private_key_path: Path, public_key_path: Path) -> dict[str, str]:
    """Generate one Ed25519 keypair without overwriting either destination."""
    if private_key_path.resolve() == public_key_path.resolve():
        raise ValueError("operator private and public key paths must differ")
    key = ECC.generate(curve="Ed25519")
    private_pem = key.export_key(format="PEM").encode("ascii")
    public_pem = key.public_key().export_key(format="PEM").encode("ascii")
    _exclusive_write(private_key_path, private_pem, mode=0o600)
    try:
        _exclusive_write(public_key_path, public_pem, mode=0o644)
    except BaseException:
        private_key_path.unlink(missing_ok=True)
        raise
    return load_operator_public_key(public_key_path)


def _normalize_run_payload_sha256s(values: Sequence[str]) -> list[str]:
    normalized = sorted(
        _require_sha256(value, source=f"run payload SHA-256[{index}]")
        for index, value in enumerate(values)
    )
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError("run payload SHA-256 values must be non-empty and unique")
    return normalized


def _attestation_payload(
    *,
    campaign_sha256: str,
    manifest_sha256: str,
    observed_at_utc: str,
    readback: Mapping[str, Any],
    run_payload_sha256s: Sequence[str],
    runtime_contract_sha256: str,
) -> dict[str, Any]:
    if not isinstance(readback, Mapping):
        raise TypeError("operator readback must be an object")
    parse_utc_timestamp(observed_at_utc, source="operator attestation observed_at_utc")
    normalized_readback = json.loads(_canonical_json_bytes(readback).decode("ascii"))
    return {
        "campaign_sha256": _require_sha256(
            campaign_sha256,
            source="operator attestation campaign_sha256",
        ),
        "manifest_sha256": _require_sha256(
            manifest_sha256,
            source="operator attestation manifest_sha256",
        ),
        "observed_at_utc": observed_at_utc,
        "profile": OPERATOR_ATTESTATION_PROFILE,
        "readback": normalized_readback,
        "run_payload_sha256s": _normalize_run_payload_sha256s(run_payload_sha256s),
        "runtime_contract_sha256": _require_sha256(
            runtime_contract_sha256,
            source="operator attestation runtime_contract_sha256",
        ),
        "schema_version": OPERATOR_ATTESTATION_SCHEMA_VERSION,
    }


def create_operator_attestation(
    readback: Mapping[str, Any],
    *,
    campaign_sha256: str,
    manifest_sha256: str,
    observed_at_utc: str,
    private_key_pem: bytes,
    public_key: Mapping[str, Any],
    run_payload_sha256s: Sequence[str],
    runtime_contract_sha256: str,
) -> dict[str, Any]:
    """Sign one operator readback and its exact campaign and run bindings."""
    normalized_public_key = validate_operator_public_key(public_key)
    derived_public_key = operator_public_key_from_private_key(private_key_pem)
    if derived_public_key != normalized_public_key:
        raise ValueError("operator private key does not match the frozen public key")
    payload = _attestation_payload(
        campaign_sha256=campaign_sha256,
        manifest_sha256=manifest_sha256,
        observed_at_utc=observed_at_utc,
        readback=readback,
        run_payload_sha256s=run_payload_sha256s,
        runtime_contract_sha256=runtime_contract_sha256,
    )
    signature = eddsa.new(
        _import_private_key(private_key_pem),
        mode="rfc8032",
    ).sign(_canonical_json_bytes(payload))
    return {
        **payload,
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }


def validate_operator_attestation(
    record: Mapping[str, Any],
    *,
    campaign_sha256: str,
    manifest_sha256: str,
    public_key: Mapping[str, Any],
    run_payload_sha256s: Sequence[str],
    runtime_contract_sha256: str,
) -> dict[str, Any]:
    """Verify one operator attestation and all frozen campaign bindings."""
    if not isinstance(record, Mapping):
        raise TypeError("operator attestation must be an object")
    _require_exact_fields(record, OPERATOR_ATTESTATION_FIELDS, source="operator attestation")
    payload_fields = {field: record[field] for field in sorted(_ATTESTATION_PAYLOAD_FIELDS)}
    normalized = _attestation_payload(
        campaign_sha256=record["campaign_sha256"],
        manifest_sha256=record["manifest_sha256"],
        observed_at_utc=record["observed_at_utc"],
        readback=record["readback"],
        run_payload_sha256s=record["run_payload_sha256s"],
        runtime_contract_sha256=record["runtime_contract_sha256"],
    )
    expected = _attestation_payload(
        campaign_sha256=campaign_sha256,
        manifest_sha256=manifest_sha256,
        observed_at_utc=record["observed_at_utc"],
        readback=record["readback"],
        run_payload_sha256s=run_payload_sha256s,
        runtime_contract_sha256=runtime_contract_sha256,
    )
    if normalized != expected or payload_fields != normalized:
        raise ValueError("operator attestation campaign or run bindings differ")
    signature = _decode_canonical_base64(
        record["signature_base64"],
        source="operator attestation signature_base64",
    )
    normalized_public_key = validate_operator_public_key(public_key)
    public_der = _decode_canonical_base64(
        normalized_public_key["public_key_der_base64"],
        source="operator public key public_key_der_base64",
    )
    verifier = eddsa.new(ECC.import_key(public_der), mode="rfc8032")
    try:
        verifier.verify(_canonical_json_bytes(normalized), signature)
    except ValueError as exc:
        raise ValueError("operator attestation signature verification failed") from exc
    return {
        **normalized,
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }
