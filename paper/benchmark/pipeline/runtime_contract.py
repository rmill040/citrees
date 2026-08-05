"""Canonical schema and content addressing for benchmark runtime contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

from paper.benchmark.pipeline.operator_attestation import validate_operator_public_key

RUNTIME_CONTRACT_SCHEMA_VERSION = 5
RUNTIME_CONTRACT_PROFILE = "r_cforest_runtime"
RUNTIME_CONTRACT_S3_PREFIX = "runtime-contracts"
RUNTIME_CONTRACT_FIELDS = frozenset(
    {
        "operator_attestation_public_key",
        "profile",
        "runtime",
        "schema_version",
    }
)
RUNTIME_PROVENANCE_FIELDS = frozenset(
    {
        "ami_id",
        "container_image_digest",
        "cpu_model",
        "git_sha",
        "instance_type",
        "kernel",
        "logical_cpus",
        "machine",
        "microcode",
        "openssl_version",
        "os_release",
        "python_libraries",
        "r_numerical_libraries",
        "r_runtime",
        "thread_environment",
        "threadpools",
    }
)
PYTHON_LIBRARY_NAMES = frozenset(
    {
        "citrees",
        "numpy",
        "pandas",
        "rpy2",
        "scikit-learn",
        "scipy",
        "threadpoolctl",
    }
)
R_RUNTIME_FIELDS = frozenset({"r", "partykit", "libcoin", "mvtnorm"})
THREAD_ENVIRONMENT = frozenset(
    {
        "BLIS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "R_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    }
)
EXPECTED_THREAD_VALUE = "1"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_IMAGE_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPENSSL_VERSION_PATTERN = re.compile(r"^OpenSSL [^\r\n]+$")
_VERSION_FIELDS = {
    "os_release": frozenset({"ID", "VERSION_ID"}),
    "python_libraries": PYTHON_LIBRARY_NAMES,
    "r_numerical_libraries": frozenset({"blas", "lapack"}),
    "r_runtime": R_RUNTIME_FIELDS,
}
_STRING_RUNTIME_FIELDS = (
    "ami_id",
    "cpu_model",
    "instance_type",
    "kernel",
    "machine",
    "microcode",
)
_THREADPOOL_REQUIRED_FIELDS = frozenset(
    {"filepath", "internal_api", "num_threads", "prefix", "user_api"}
)


def validate_runtime_contract_sha256(value: str) -> str:
    """Validate one exact lowercase hexadecimal runtime-contract digest."""
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError("runtime contract SHA-256 must be 64 lowercase hexadecimal characters")
    return value


def runtime_contract_s3_key(sha256: str) -> str:
    """Return the content-addressed S3 key for a runtime contract."""
    return f"{RUNTIME_CONTRACT_S3_PREFIX}/{validate_runtime_contract_sha256(sha256)}.json"


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


def _require_concrete_string(value: Any, *, source: str) -> str:
    if not isinstance(value, str) or not value.strip() or value == "unknown":
        raise ValueError(f"{source} must be a non-empty concrete string")
    return value


def validate_openssl_version(value: Any, *, source: str = "OpenSSL version") -> str:
    """Validate canonical one-line output from ``openssl version``."""
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not _OPENSSL_VERSION_PATTERN.fullmatch(value)
    ):
        raise ValueError(f"{source} must be canonical one-line output from openssl version")
    return value


def _validate_version_mapping(
    value: Any,
    *,
    expected: frozenset[str],
    source: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{source} must be an object")
    _require_exact_fields(value, expected, source=source)
    return {
        field: _require_concrete_string(value[field], source=f"{source}.{field}")
        for field in sorted(expected)
    }


def _normalize_json_value(value: Any, *, source: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{source} must be finite")
        return value
    if isinstance(value, list):
        return [
            _normalize_json_value(item, source=f"{source}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{source} field names must be strings")
            normalized[key] = _normalize_json_value(item, source=f"{source}.{key}")
        return normalized
    raise TypeError(f"{source} contains unsupported value type {type(value).__name__}")


def _validate_threadpools(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("runtime contract runtime.threadpools must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for index, pool in enumerate(value):
        source = f"runtime contract runtime.threadpools[{index}]"
        if not isinstance(pool, Mapping):
            raise TypeError(f"{source} must be an object")
        missing = sorted(_THREADPOOL_REQUIRED_FIELDS - set(pool))
        if missing:
            raise ValueError(f"{source} fields differ: missing={missing}")
        if type(pool["num_threads"]) is not int or pool["num_threads"] != 1:
            raise ValueError(f"{source}.num_threads must equal 1")
        for field in ("filepath", "internal_api", "prefix", "user_api"):
            _require_concrete_string(pool[field], source=f"{source}.{field}")
        normalized.append(_normalize_json_value(pool, source=source))
    return normalized


def validate_runtime_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one runtime contract."""
    if not isinstance(contract, Mapping):
        raise TypeError("runtime contract must be an object")
    _require_exact_fields(contract, RUNTIME_CONTRACT_FIELDS, source="runtime contract")
    if contract["schema_version"] != RUNTIME_CONTRACT_SCHEMA_VERSION:
        raise ValueError("runtime contract has an unexpected schema version")
    if contract["profile"] != RUNTIME_CONTRACT_PROFILE:
        raise ValueError("runtime contract has an unexpected profile")

    runtime = contract["runtime"]
    if not isinstance(runtime, Mapping):
        raise TypeError("runtime contract runtime must be an object")
    _require_exact_fields(
        runtime,
        RUNTIME_PROVENANCE_FIELDS,
        source="runtime contract runtime",
    )

    normalized_runtime: dict[str, Any] = {}
    for field in _STRING_RUNTIME_FIELDS:
        normalized_runtime[field] = _require_concrete_string(
            runtime[field],
            source=f"runtime contract runtime.{field}",
        )
    normalized_runtime["openssl_version"] = validate_openssl_version(
        runtime["openssl_version"],
        source="runtime contract runtime.openssl_version",
    )
    container_image_digest = _require_concrete_string(
        runtime["container_image_digest"],
        source="runtime contract runtime.container_image_digest",
    )
    if not _IMAGE_DIGEST_PATTERN.fullmatch(container_image_digest):
        raise ValueError(
            "runtime contract runtime.container_image_digest must be an immutable digest"
        )
    normalized_runtime["container_image_digest"] = container_image_digest

    git_sha = _require_concrete_string(
        runtime["git_sha"],
        source="runtime contract runtime.git_sha",
    )
    if not _GIT_SHA_PATTERN.fullmatch(git_sha):
        raise ValueError("runtime contract runtime.git_sha is invalid")
    normalized_runtime["git_sha"] = git_sha

    logical_cpus = runtime["logical_cpus"]
    if type(logical_cpus) is not int or logical_cpus <= 0:
        raise ValueError("runtime contract runtime.logical_cpus must be a positive integer")
    normalized_runtime["logical_cpus"] = logical_cpus

    for field, expected in _VERSION_FIELDS.items():
        normalized_runtime[field] = _validate_version_mapping(
            runtime[field],
            expected=expected,
            source=f"runtime contract runtime.{field}",
        )

    expected_thread_environment = {name: EXPECTED_THREAD_VALUE for name in THREAD_ENVIRONMENT}
    if runtime["thread_environment"] != expected_thread_environment:
        raise ValueError("runtime contract runtime.thread_environment is not frozen")
    normalized_runtime["thread_environment"] = dict(sorted(expected_thread_environment.items()))
    normalized_runtime["threadpools"] = _validate_threadpools(runtime["threadpools"])
    operator_public_key = contract["operator_attestation_public_key"]
    if not isinstance(operator_public_key, Mapping):
        raise TypeError("runtime contract operator_attestation_public_key must be an object")

    return {
        "schema_version": RUNTIME_CONTRACT_SCHEMA_VERSION,
        "profile": RUNTIME_CONTRACT_PROFILE,
        "operator_attestation_public_key": validate_operator_public_key(operator_public_key),
        "runtime": normalized_runtime,
    }


def serialize_runtime_contract(contract: Mapping[str, Any]) -> bytes:
    """Serialize a validated runtime contract to its canonical bytes."""
    normalized = validate_runtime_contract(contract)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"runtime contract contains duplicate field {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r} is not allowed")


def parse_runtime_contract(
    payload: bytes,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Parse canonical runtime-contract bytes and verify their content digest."""
    if not isinstance(payload, bytes):
        raise TypeError("runtime contract payload must be bytes")
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        expected = validate_runtime_contract_sha256(expected_sha256)
        if digest != expected:
            raise ValueError(
                f"runtime contract SHA-256 mismatch: expected {expected}, observed {digest}"
            )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("runtime contract must be UTF-8 encoded") from exc
    if text.startswith("\ufeff"):
        raise ValueError("runtime contract must not contain a UTF-8 byte-order mark")
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid runtime contract JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("runtime contract must be an object")
    canonical = serialize_runtime_contract(parsed)
    if payload != canonical:
        raise ValueError("runtime contract must use canonical JSON bytes")
    return parsed


def runtime_contract_sha256(contract: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of a runtime contract's canonical bytes."""
    return hashlib.sha256(serialize_runtime_contract(contract)).hexdigest()
