"""EC2 instance management for API server and workers.

Launch, list, and terminate EC2 instances that run the experiment
API server and workers via Docker containers pulled from ECR.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import re
import shlex
import textwrap
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import httpx
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)

from paper.benchmark.adapters.store import _normalize_artifact_prefix
from paper.benchmark.cli.console_output import info, step, success, warn
from paper.benchmark.config.constants import (
    API_POLL_INTERVAL_SECONDS,
    API_READINESS_TIMEOUT_SECONDS,
    WORKER_MAX_API_FAILURES,
)
from paper.benchmark.infra.aws import (
    DEFAULT_REGION,
    ensure_campaign_iam_profile,
    ensure_security_group,
    get_aws_account_id,
    get_frozen_git_sha,
    get_resource_name,
    publish_rerun_manifest,
    validate_image_revision,
)

TAG_KEY = "citrees-role"
API_TAG_VALUE = "api"
WORKER_TAG_VALUE = "worker"
MECHANISM_TAG_VALUE = "mechanism"
LOG_GROUP_API = "/citrees/api"
LOG_GROUP_WORKER = "/citrees/worker"
LOG_GROUP_MECHANISM = "/citrees/mechanism"
DEFAULT_MECHANISM_TASKS = ("classification", "regression")
DEFAULT_MECHANISM_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_MECHANISM_FOLDS = (0, 1, 2, 3, 4)
DEFAULT_MECHANISM_MODEL_VARIANTS = ("cif_default",)
DEFAULT_MECHANISM_RANKING_VARIANTS = ("split_importance", "split_count")
_IMAGE_DIGEST_PATTERN = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_LAUNCH_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_QUEUE_STAGES = frozenset({"rankings", "metrics"})
_API_ENDPOINT_ROLE = "api-endpoint"
_API_ENDPOINT_SUFFIX = "citrees.internal."
_API_ENDPOINT_TTL_SECONDS = 10
_ROUTE53_CHANGE_TIMEOUT_SECONDS = 120.0
_ROUTE53_CHANGE_POLL_SECONDS = 2.0
_BENCHMARK_MARKET = "on-demand"
_AMBIGUOUS_EC2_ERROR_CODES = frozenset(
    {
        "InternalError",
        "InternalFailure",
        "RequestLimitExceeded",
        "RequestTimeout",
        "RequestTimeoutException",
        "ServiceUnavailable",
        "ServiceUnavailableException",
        "Unavailable",
    }
)
_CAPACITY_ERROR_CODES = frozenset(
    {
        "InsufficientHostCapacity",
        "InsufficientInstanceCapacity",
        "InsufficientReservedInstanceCapacity",
        "InstanceLimitExceeded",
        "MaxSpotInstanceCountExceeded",
        "VcpuLimitExceeded",
    }
)
_AMBIGUOUS_EC2_RETRY_DELAYS = (1.0, 2.0)
_WORKER_LAUNCH_SCHEMA_VERSION = 8


@dataclass(frozen=True)
class ApiScope:
    """Immutable scope advertised by one running queue server."""

    api_url: str
    public_api_url: str
    artifact_prefix: str
    campaign_sha256: str
    canonical_manifest_s3_key: str
    canonical_manifest_sha256: str
    gate_receipt_s3_key: str
    gate_receipt_sha256: str
    image_uri: str
    manifest_s3_key: str
    manifest_sha256: str
    market: str
    max_cell_attempts: int
    runtime_contract_s3_key: str
    runtime_contract_sha256: str
    stage: str


@dataclass(frozen=True)
class _ApiEndpoint:
    """One campaign-scoped private Route53 API endpoint."""

    hosted_zone_id: str
    hosted_zone_name: str
    hostname: str
    record_name: str

    @property
    def api_url(self) -> str:
        """Return the stable worker-facing API URL."""
        return f"http://{self.hostname}:8000"


@dataclass(frozen=True)
class RuntimeLaunchContract:
    """Launch-critical fields from one canonical runtime contract."""

    ami_id: str
    container_image_digest: str
    git_sha: str
    instance_type: str
    s3_key: str
    sha256: str


@dataclass(frozen=True)
class GateReceiptIdentity:
    """Content identity for one complete reproducibility-gate receipt."""

    s3_key: str
    sha256: str


@dataclass(frozen=True)
class _WorkerSubnetPlacement:
    """One validated benchmark-worker subnet."""

    availability_zone: str
    subnet_id: str
    vpc_id: str


def _csv_arg(values: Sequence[str | int]) -> str:
    """Encode a structured option list for the mechanism runner CLI."""
    return ",".join(str(value) for value in values)


def validate_image_digest_uri(image_uri: str) -> str:
    """Require an immutable container image digest for distributed work."""
    normalized = image_uri.strip()
    if not _IMAGE_DIGEST_PATTERN.fullmatch(normalized):
        raise ValueError("image_uri must be an immutable repository@sha256:<64 hex> URI")
    return normalized


def _validate_launch_id(launch_id: str) -> str:
    """Validate one operator-supplied immutable launch identity."""
    normalized = launch_id.strip()
    if not _LAUNCH_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "launch_id must contain 1-64 lowercase letters, digits, or hyphens, "
            "and must start and end with a letter or digit"
        )
    return normalized


def _validate_worker_launch_id(launch_id: str) -> str:
    """Validate one immutable worker launch identity."""
    return _validate_launch_id(launch_id)


def _validate_api_launch_id(launch_id: str) -> str:
    """Validate one immutable API launch identity."""
    return _validate_launch_id(launch_id)


def _require_on_demand_instance(instance: dict[str, Any], *, source: str) -> None:
    """Require one DescribeInstances row to prove on-demand lifecycle."""
    lifecycle = instance.get("InstanceLifecycle")
    if lifecycle is not None:
        raise RuntimeError(f"{source} must be on-demand, but EC2 reports lifecycle {lifecycle!r}")


def _canonical_json_object(value: dict[str, Any]) -> bytes:
    """Serialize a control record to one deterministic JSON representation."""
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _is_missing_s3_object(exc: ClientError) -> bool:
    """Return whether an S3 client error means that an object is absent."""
    code = exc.response.get("Error", {}).get("Code", "")
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"404", "NoSuchKey", "NotFound"} or status == 404


def _put_immutable_json(
    s3: Any,
    *,
    bucket: str,
    key: str,
    value: dict[str, Any],
    record_kind: str,
) -> str:
    """Write and read back one canonical JSON control record exactly once."""
    payload = _canonical_json_object(value)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    metadata = {
        "record-kind": record_kind,
        "sha256": payload_sha256,
    }
    write_error: Exception | None = None
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType="application/json",
            IfNoneMatch="*",
            Metadata=metadata,
        )
    except Exception as exc:
        write_error = exc

    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if write_error is not None and _is_missing_s3_object(exc):
            raise write_error from None
        raise
    observed_payload = response["Body"].read()
    observed_metadata = response.get("Metadata", {})
    if observed_payload != payload or observed_metadata != metadata:
        message = f"Immutable worker launch record collision at s3://{bucket}/{key}"
        if write_error is not None:
            raise RuntimeError(message) from write_error
        raise RuntimeError(message)
    return payload_sha256


def _read_immutable_json(
    s3: Any,
    *,
    bucket: str,
    key: str,
    record_kind: str,
) -> dict[str, Any] | None:
    """Read and validate one canonical immutable JSON control record."""
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if _is_missing_s3_object(exc):
            return None
        raise
    payload = response["Body"].read()
    try:
        value = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid worker launch record at s3://{bucket}/{key}") from exc
    if not isinstance(value, dict) or _canonical_json_object(value) != payload:
        raise RuntimeError(f"Non-canonical worker launch record at s3://{bucket}/{key}")
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    expected_metadata = {"record-kind": record_kind, "sha256": payload_sha256}
    if response.get("Metadata", {}) != expected_metadata:
        raise RuntimeError(f"Worker launch record metadata mismatch at s3://{bucket}/{key}")
    return value


def _worker_launch_prefix(artifact_prefix: str, launch_id: str) -> str:
    """Return the exact control prefix for one worker launch batch."""
    return f"{artifact_prefix}/_control/worker-launches/{launch_id}"


def _worker_slot_identity(intent_sha256: str, slot: int) -> tuple[str, str]:
    """Derive one stable EC2 client token and request digest per launch slot."""
    payload = _canonical_json_object(
        {
            "intent_sha256": intent_sha256,
            "slot": slot,
        }
    )
    request_sha256 = hashlib.sha256(payload).hexdigest()
    return f"citrees-worker-{request_sha256[:49]}", request_sha256


def _worker_launch_record(
    *,
    artifact_prefix: str,
    availability_zone: str,
    campaign_sha256: str,
    client_token: str,
    instance_family: str,
    instance_id: str,
    instance_type: str,
    intent_sha256: str,
    launch_id: str,
    manifest_sha256: str,
    market: str,
    region: str,
    request_sha256: str,
    slot: int,
    subnet_id: str,
    vpc_id: str,
) -> dict[str, Any]:
    """Build the complete deterministic record for one accepted instance."""
    return {
        "artifact_prefix": artifact_prefix,
        "availability_zone": availability_zone,
        "campaign_sha256": campaign_sha256,
        "client_token": client_token,
        "instance_family": instance_family,
        "instance_id": instance_id,
        "instance_type": instance_type,
        "intent_sha256": intent_sha256,
        "kind": "citrees-benchmark-worker-launch",
        "launch_id": launch_id,
        "manifest_sha256": manifest_sha256,
        "market": market,
        "region": region,
        "request_sha256": request_sha256,
        "schema_version": _WORKER_LAUNCH_SCHEMA_VERSION,
        "slot": slot,
        "subnet_id": subnet_id,
        "vpc_id": vpc_id,
    }


def _load_worker_launch_record(
    s3: Any,
    *,
    artifact_prefix: str,
    availability_zone: str,
    bucket: str,
    campaign_sha256: str,
    client_token: str,
    instance_family: str,
    instance_type: str,
    intent_sha256: str,
    launch_id: str,
    manifest_sha256: str,
    market: str,
    region: str,
    request_sha256: str,
    slot: int,
    subnet_id: str,
    vpc_id: str,
) -> str | None:
    """Return a previously accepted instance after exact record validation."""
    key = f"{_worker_launch_prefix(artifact_prefix, launch_id)}/instances/{slot:06d}.json"
    record = _read_immutable_json(
        s3,
        bucket=bucket,
        key=key,
        record_kind="worker-launch",
    )
    if record is None:
        return None
    instance_id = record.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id:
        raise RuntimeError(f"Worker launch record has no instance ID: s3://{bucket}/{key}")
    expected = _worker_launch_record(
        artifact_prefix=artifact_prefix,
        availability_zone=availability_zone,
        campaign_sha256=campaign_sha256,
        client_token=client_token,
        instance_family=instance_family,
        instance_id=instance_id,
        instance_type=instance_type,
        intent_sha256=intent_sha256,
        launch_id=launch_id,
        manifest_sha256=manifest_sha256,
        market=market,
        region=region,
        request_sha256=request_sha256,
        slot=slot,
        subnet_id=subnet_id,
        vpc_id=vpc_id,
    )
    if record != expected:
        raise RuntimeError(f"Worker launch record scope mismatch at s3://{bucket}/{key}")
    return instance_id


def _publish_worker_launch_record(
    s3: Any,
    *,
    artifact_prefix: str,
    bucket: str,
    record: dict[str, Any],
) -> None:
    """Durably publish one accepted worker instance before returning it."""
    launch_id = str(record["launch_id"])
    slot = int(record["slot"])
    key = f"{_worker_launch_prefix(artifact_prefix, launch_id)}/instances/{slot:06d}.json"
    _put_immutable_json(
        s3,
        bucket=bucket,
        key=key,
        value=record,
        record_kind="worker-launch",
    )


def _validate_worker_instance_placement(
    instance: dict[str, Any],
    *,
    availability_zone: str,
    instance_type: str,
    source: str,
    subnet_id: str,
) -> str:
    """Require one worker instance to match its exact placement contract."""
    instance_id = instance.get("InstanceId")
    if not isinstance(instance_id, str) or not instance_id:
        raise RuntimeError(f"{source} has no instance ID")
    _require_on_demand_instance(instance, source=source)
    if instance.get("InstanceType") != instance_type:
        raise RuntimeError(
            f"{source} {instance_id} has instance type "
            f"{instance.get('InstanceType')!r}, expected {instance_type!r}"
        )
    if instance.get("SubnetId") != subnet_id:
        raise RuntimeError(
            f"{source} {instance_id} is in subnet {instance.get('SubnetId')!r}, "
            f"expected {subnet_id!r}"
        )
    placement = instance.get("Placement")
    observed_zone = placement.get("AvailabilityZone") if isinstance(placement, dict) else None
    if observed_zone != availability_zone:
        raise RuntimeError(
            f"{source} {instance_id} is in availability zone {observed_zone!r}, "
            f"expected {availability_zone!r}"
        )
    return instance_id


def _find_worker_for_exact_tags(
    ec2: Any,
    tags: dict[str, str],
    *,
    availability_zone: str,
    instance_type: str,
    subnet_id: str,
) -> str | None:
    """Recover one accepted launch in any state carrying the exact identity."""
    filters = [{"Name": f"tag:{key}", "Values": [value]} for key, value in sorted(tags.items())]
    instances: dict[str, dict[str, Any]] = {}
    next_token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Filters": filters}
        if next_token is not None:
            kwargs["NextToken"] = next_token
        response = ec2.describe_instances(**kwargs)
        for reservation in response.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                instance_id = instance.get("InstanceId")
                if not isinstance(instance_id, str) or not instance_id:
                    raise RuntimeError("EC2 returned a worker without an instance ID")
                observed_tags = {
                    str(tag["Key"]): str(tag["Value"]) for tag in instance.get("Tags", [])
                }
                if any(observed_tags.get(key) != value for key, value in tags.items()):
                    raise RuntimeError(
                        f"EC2 returned {instance_id} outside the exact worker launch identity"
                    )
                _validate_worker_instance_placement(
                    instance,
                    availability_zone=availability_zone,
                    instance_type=instance_type,
                    source="recovered worker",
                    subnet_id=subnet_id,
                )
                instances[instance_id] = instance
        token = response.get("NextToken")
        next_token = str(token) if token else None
        if next_token is None:
            break
    if len(instances) > 1:
        raise RuntimeError(
            f"Multiple EC2 instances share one worker launch identity: {sorted(instances)}"
        )
    return next(iter(instances), None)


def _single_instance_id(response: dict[str, Any]) -> str:
    """Require one exact instance from a one-instance EC2 request."""
    instances = response.get("Instances")
    if not isinstance(instances, list) or len(instances) != 1:
        count = len(instances) if isinstance(instances, list) else 0
        raise RuntimeError(f"EC2 returned {count} instances for a one-instance worker request")
    instance_id = instances[0].get("InstanceId")
    if not isinstance(instance_id, str) or not instance_id:
        raise RuntimeError("EC2 accepted a worker request without returning an instance ID")
    return instance_id


def _is_ambiguous_ec2_error(exc: Exception) -> bool:
    """Return whether an EC2 failure could follow an accepted request."""
    if isinstance(
        exc,
        (
            ConnectionClosedError,
            ConnectTimeoutError,
            EndpointConnectionError,
            ReadTimeoutError,
        ),
    ):
        return True
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code in _AMBIGUOUS_EC2_ERROR_CODES
    return False


def _launch_or_recover_worker(
    ec2: Any,
    *,
    availability_zone: str,
    instance_type: str,
    recover_before_launch: bool,
    run_kwargs: dict[str, Any],
    subnet_id: str,
    tags: dict[str, str],
) -> str:
    """Recover or idempotently launch one exact worker instance."""
    if recover_before_launch:
        recovered = _find_worker_for_exact_tags(
            ec2,
            tags,
            availability_zone=availability_zone,
            instance_type=instance_type,
            subnet_id=subnet_id,
        )
        if recovered is not None:
            return recovered

    for attempt in range(len(_AMBIGUOUS_EC2_RETRY_DELAYS) + 1):
        try:
            response = ec2.run_instances(**run_kwargs)
        except Exception as exc:
            if not _is_ambiguous_ec2_error(exc):
                raise
            recovered = _find_worker_for_exact_tags(
                ec2,
                tags,
                availability_zone=availability_zone,
                instance_type=instance_type,
                subnet_id=subnet_id,
            )
            if recovered is not None:
                return recovered
            if attempt == len(_AMBIGUOUS_EC2_RETRY_DELAYS):
                raise
            delay = _AMBIGUOUS_EC2_RETRY_DELAYS[attempt]
            warn(
                "EC2 worker launch response was ambiguous; retrying the same "
                f"idempotent request after {delay:g}s"
            )
            time.sleep(delay)
            continue

        try:
            instance_id = _single_instance_id(response)
            instance = response["Instances"][0]
            if not isinstance(instance, dict):
                raise RuntimeError("EC2 returned an invalid worker instance row")
            _validate_worker_instance_placement(
                instance,
                availability_zone=availability_zone,
                instance_type=instance_type,
                source="launched worker",
                subnet_id=subnet_id,
            )
            return instance_id
        except RuntimeError:
            recovered = _find_worker_for_exact_tags(
                ec2,
                tags,
                availability_zone=availability_zone,
                instance_type=instance_type,
                subnet_id=subnet_id,
            )
            if recovered is not None:
                return recovered
            raise

    raise AssertionError("unreachable worker launch retry state")


def _validate_queue_scope(
    artifact_prefix: str,
    stage: str,
) -> tuple[str, str]:
    """Validate and normalize the distributed queue's artifact scope."""
    normalized_prefix = _normalize_artifact_prefix(artifact_prefix)
    if not normalized_prefix.startswith("repairs/"):
        raise ValueError("distributed artifact_prefix must be below repairs/")
    if stage not in _QUEUE_STAGES:
        raise ValueError(f"stage must be one of {sorted(_QUEUE_STAGES)}, got {stage!r}")
    return normalized_prefix, stage


def _api_endpoint_names(campaign_sha256: str, stage: str) -> tuple[str, str, str]:
    """Return deterministic private DNS names for one campaign and stage."""
    from paper.benchmark.pipeline.manifest import validate_manifest_sha256

    campaign_sha256 = validate_manifest_sha256(campaign_sha256)
    if stage not in _QUEUE_STAGES:
        raise ValueError(f"stage must be one of {sorted(_QUEUE_STAGES)}, got {stage!r}")
    campaign_labels = f"{campaign_sha256[:32]}.{campaign_sha256[32:]}"
    hosted_zone_name = f"{stage}.{campaign_labels}.{_API_ENDPOINT_SUFFIX}"
    record_name = f"api.{hosted_zone_name}"
    return hosted_zone_name, record_name, record_name.rstrip(".")


def _api_endpoint_tags(
    *,
    artifact_prefix: str,
    campaign_sha256: str,
    stage: str,
) -> dict[str, str]:
    """Return exact Route53 ownership tags for one API endpoint."""
    artifact_prefix, stage = _validate_queue_scope(artifact_prefix, stage)
    from paper.benchmark.pipeline.manifest import validate_manifest_sha256

    campaign_sha256 = validate_manifest_sha256(campaign_sha256)
    return {
        TAG_KEY: _API_ENDPOINT_ROLE,
        "citrees-artifact-prefix": artifact_prefix,
        "citrees-campaign-sha256": campaign_sha256,
        "citrees-stage": stage,
    }


def _route53_hosted_zone_id(value: object) -> str:
    """Normalize one Route53 hosted-zone identifier."""
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("Route53 hosted zone response is missing Id")
    return value.removeprefix("/hostedzone/")


def _get_default_vpc_id(ec2: Any) -> str:
    """Return the only default VPC visible to the current account and region."""
    response = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])
    vpcs = response.get("Vpcs")
    if not isinstance(vpcs, list) or len(vpcs) != 1:
        count = len(vpcs) if isinstance(vpcs, list) else 0
        raise RuntimeError(f"expected exactly one default VPC, found {count}")
    vpc = vpcs[0]
    vpc_id = vpc.get("VpcId") if isinstance(vpc, dict) else None
    if not isinstance(vpc_id, str) or not vpc_id:
        raise RuntimeError("default VPC response is missing VpcId")
    return vpc_id


def _route53_tags(route53: Any, hosted_zone_id: str) -> dict[str, str]:
    """Read one hosted zone's exact tag mapping."""
    response = route53.list_tags_for_resource(
        ResourceType="hostedzone",
        ResourceId=hosted_zone_id,
    )
    tag_set = response.get("ResourceTagSet")
    tags = tag_set.get("Tags") if isinstance(tag_set, dict) else None
    if not isinstance(tags, list):
        raise RuntimeError(f"Route53 hosted zone {hosted_zone_id} returned invalid tags")
    observed: dict[str, str] = {}
    for tag in tags:
        if not isinstance(tag, dict):
            raise RuntimeError(f"Route53 hosted zone {hosted_zone_id} returned invalid tags")
        key = tag.get("Key")
        value = tag.get("Value")
        if not isinstance(key, str) or not isinstance(value, str) or key in observed:
            raise RuntimeError(f"Route53 hosted zone {hosted_zone_id} returned invalid tags")
        observed[key] = value
    return observed


def _find_api_endpoint(
    ec2: Any,
    route53: Any,
    *,
    artifact_prefix: str,
    campaign_sha256: str,
    stage: str,
    region: str,
) -> _ApiEndpoint | None:
    """Find and validate one campaign-owned private API endpoint."""
    hosted_zone_name, record_name, hostname = _api_endpoint_names(campaign_sha256, stage)
    expected_tags = _api_endpoint_tags(
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        stage=stage,
    )
    response = route53.list_hosted_zones_by_name(
        DNSName=hosted_zone_name,
        MaxItems="100",
    )
    hosted_zones = response.get("HostedZones")
    if not isinstance(hosted_zones, list):
        raise RuntimeError("Route53 hosted zone lookup returned an invalid response")
    exact_zones = [
        zone
        for zone in hosted_zones
        if isinstance(zone, dict) and zone.get("Name") == hosted_zone_name
    ]
    if len(exact_zones) > 1:
        ids = sorted(_route53_hosted_zone_id(zone.get("Id")) for zone in exact_zones)
        raise RuntimeError(f"Multiple Route53 hosted zones exist for the exact API endpoint: {ids}")
    if not exact_zones:
        return None

    zone = exact_zones[0]
    config = zone.get("Config")
    if not isinstance(config, dict) or config.get("PrivateZone") is not True:
        raise RuntimeError(f"API endpoint hosted zone {hosted_zone_name} is not private")
    hosted_zone_id = _route53_hosted_zone_id(zone.get("Id"))
    detail = route53.get_hosted_zone(Id=hosted_zone_id)
    vpcs = detail.get("VPCs")
    default_vpc_id = _get_default_vpc_id(ec2)
    expected_vpc = {"VPCId": default_vpc_id, "VPCRegion": region}
    if not isinstance(vpcs, list) or vpcs != [expected_vpc]:
        raise RuntimeError(
            f"API endpoint hosted zone {hosted_zone_id} is not scoped only to "
            f"default VPC {default_vpc_id} in {region}"
        )
    observed_tags = _route53_tags(route53, hosted_zone_id)
    if observed_tags != expected_tags:
        raise RuntimeError(
            f"API endpoint hosted zone {hosted_zone_id} ownership tags differ: "
            f"expected {expected_tags}, observed {observed_tags}"
        )
    return _ApiEndpoint(
        hosted_zone_id=hosted_zone_id,
        hosted_zone_name=hosted_zone_name,
        hostname=hostname,
        record_name=record_name,
    )


def _ensure_api_endpoint(
    ec2: Any,
    route53: Any,
    *,
    artifact_prefix: str,
    campaign_sha256: str,
    launch_id: str,
    stage: str,
    region: str,
) -> _ApiEndpoint:
    """Create or validate one campaign-scoped private Route53 endpoint."""
    existing = _find_api_endpoint(
        ec2,
        route53,
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        stage=stage,
        region=region,
    )
    if existing is not None:
        return existing

    launch_id = _validate_api_launch_id(launch_id)
    hosted_zone_name, _record_name, _hostname = _api_endpoint_names(campaign_sha256, stage)
    vpc_id = _get_default_vpc_id(ec2)
    caller_reference = hashlib.sha256(
        f"{campaign_sha256}:{stage}:{launch_id}".encode("ascii")
    ).hexdigest()
    response = route53.create_hosted_zone(
        Name=hosted_zone_name,
        VPC={"VPCId": vpc_id, "VPCRegion": region},
        CallerReference=f"citrees-api-{caller_reference}",
        HostedZoneConfig={
            "Comment": f"citrees API endpoint for {stage} campaign {campaign_sha256}",
            "PrivateZone": True,
        },
    )
    zone = response.get("HostedZone")
    hosted_zone_id = _route53_hosted_zone_id(zone.get("Id") if isinstance(zone, dict) else None)
    try:
        tags = _api_endpoint_tags(
            artifact_prefix=artifact_prefix,
            campaign_sha256=campaign_sha256,
            stage=stage,
        )
        route53.change_tags_for_resource(
            ResourceType="hostedzone",
            ResourceId=hosted_zone_id,
            AddTags=[{"Key": key, "Value": value} for key, value in sorted(tags.items())],
        )
        endpoint = _find_api_endpoint(
            ec2,
            route53,
            artifact_prefix=artifact_prefix,
            campaign_sha256=campaign_sha256,
            stage=stage,
            region=region,
        )
    except Exception:
        route53.delete_hosted_zone(Id=hosted_zone_id)
        raise
    if endpoint is None:
        route53.delete_hosted_zone(Id=hosted_zone_id)
        raise RuntimeError("Route53 did not return the newly created API endpoint")
    return endpoint


def _wait_for_route53_change(route53: Any, response: dict[str, Any]) -> None:
    """Wait until one Route53 record change is globally committed."""
    change_info = response.get("ChangeInfo")
    if not isinstance(change_info, dict):
        raise RuntimeError("Route53 record change response is missing ChangeInfo")
    change_id = change_info.get("Id")
    status = change_info.get("Status")
    if not isinstance(change_id, str) or not change_id:
        raise RuntimeError("Route53 record change response is missing Id")
    deadline = time.monotonic() + _ROUTE53_CHANGE_TIMEOUT_SECONDS
    while status != "INSYNC":
        if status != "PENDING":
            raise RuntimeError(f"Route53 record change {change_id} has invalid status {status!r}")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Route53 record change {change_id} did not reach INSYNC")
        time.sleep(_ROUTE53_CHANGE_POLL_SECONDS)
        observed = route53.get_change(Id=change_id).get("ChangeInfo")
        if not isinstance(observed, dict):
            raise RuntimeError(f"Route53 record change {change_id} returned invalid status")
        status = observed.get("Status")


def _get_api_endpoint_record(route53: Any, endpoint: _ApiEndpoint) -> dict[str, Any] | None:
    """Return the campaign endpoint A record when it exists."""
    response = route53.list_resource_record_sets(
        HostedZoneId=endpoint.hosted_zone_id,
        StartRecordName=endpoint.record_name,
        StartRecordType="A",
        MaxItems="1",
    )
    records = response.get("ResourceRecordSets")
    if not isinstance(records, list):
        raise RuntimeError("Route53 record lookup returned an invalid response")
    if not records:
        return None
    record = records[0]
    if not isinstance(record, dict):
        raise RuntimeError("Route53 record lookup returned an invalid response")
    if record.get("Name") != endpoint.record_name or record.get("Type") != "A":
        return None
    return record


def _upsert_api_endpoint(route53: Any, endpoint: _ApiEndpoint, private_ip: str) -> None:
    """Point the stable campaign endpoint at one ready API instance."""
    try:
        address = ipaddress.ip_address(private_ip)
    except ValueError as exc:
        raise RuntimeError(f"API instance private IP is invalid: {private_ip!r}") from exc
    if not isinstance(address, ipaddress.IPv4Address) or not address.is_private:
        raise RuntimeError(f"API instance private IP is invalid: {private_ip!r}")
    record = {
        "Name": endpoint.record_name,
        "Type": "A",
        "TTL": _API_ENDPOINT_TTL_SECONDS,
        "ResourceRecords": [{"Value": str(address)}],
    }
    response = route53.change_resource_record_sets(
        HostedZoneId=endpoint.hosted_zone_id,
        ChangeBatch={
            "Comment": "Point the stable citrees API endpoint at the ready instance",
            "Changes": [
                {
                    "Action": "UPSERT",
                    "ResourceRecordSet": record,
                }
            ],
        },
    )
    _wait_for_route53_change(route53, response)
    observed = _get_api_endpoint_record(route53, endpoint)
    if observed != record:
        raise RuntimeError(
            f"Route53 API endpoint readback differs: expected {record}, observed {observed}"
        )


def _delete_api_endpoint(
    ec2: Any,
    route53: Any,
    *,
    artifact_prefix: str,
    campaign_sha256: str,
    stage: str,
    region: str,
) -> bool:
    """Delete only the private endpoint owned by one exact campaign scope."""
    endpoint = _find_api_endpoint(
        ec2,
        route53,
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        stage=stage,
        region=region,
    )
    if endpoint is None:
        return False
    record = _get_api_endpoint_record(route53, endpoint)
    if record is not None:
        response = route53.change_resource_record_sets(
            HostedZoneId=endpoint.hosted_zone_id,
            ChangeBatch={
                "Comment": "Delete the campaign-scoped citrees API endpoint",
                "Changes": [
                    {
                        "Action": "DELETE",
                        "ResourceRecordSet": record,
                    }
                ],
            },
        )
        _wait_for_route53_change(route53, response)
    inventory = route53.list_resource_record_sets(HostedZoneId=endpoint.hosted_zone_id)
    records = inventory.get("ResourceRecordSets")
    if not isinstance(records, list):
        raise RuntimeError("Route53 endpoint inventory returned an invalid response")
    non_default = [
        item
        for item in records
        if not isinstance(item, dict) or item.get("Type") not in {"NS", "SOA"}
    ]
    if non_default:
        raise RuntimeError(
            f"API endpoint hosted zone {endpoint.hosted_zone_id} contains unexpected records"
        )
    route53.delete_hosted_zone(Id=endpoint.hosted_zone_id)
    return True


def _validate_runtime_contract_scope(
    runtime_contract_sha256: object,
    runtime_contract_s3_key: object,
) -> tuple[str, str]:
    """Validate one immutable runtime contract digest and S3 key."""
    from paper.benchmark.pipeline.runtime_contract import (
        runtime_contract_s3_key as canonical_runtime_contract_s3_key,
    )
    from paper.benchmark.pipeline.runtime_contract import validate_runtime_contract_sha256

    if not isinstance(runtime_contract_sha256, str):
        raise ValueError("runtime_contract_sha256 must be a string")
    digest = validate_runtime_contract_sha256(runtime_contract_sha256)
    if not isinstance(runtime_contract_s3_key, str):
        raise ValueError("runtime_contract_s3_key must be a string")
    expected_key = canonical_runtime_contract_s3_key(digest)
    if runtime_contract_s3_key != expected_key:
        raise ValueError(
            f"runtime_contract_s3_key must be the content-addressed key {expected_key!r}"
        )
    return digest, runtime_contract_s3_key


def _validate_gate_receipt_scope(
    gate_receipt_sha256: object,
    gate_receipt_s3_key: object,
) -> tuple[str, str]:
    """Validate one immutable gate receipt digest and S3 key."""
    from paper.benchmark.experiments.r_cforest_reproducibility import (
        gate_receipt_s3_key as canonical_gate_receipt_s3_key,
    )
    from paper.benchmark.experiments.r_cforest_reproducibility import (
        validate_gate_receipt_sha256,
    )

    if not isinstance(gate_receipt_sha256, str):
        raise ValueError("gate_receipt_sha256 must be a string")
    digest = validate_gate_receipt_sha256(gate_receipt_sha256)
    if not isinstance(gate_receipt_s3_key, str):
        raise ValueError("gate_receipt_s3_key must be a string")
    expected_key = canonical_gate_receipt_s3_key(digest)
    if gate_receipt_s3_key != expected_key:
        raise ValueError(f"gate_receipt_s3_key must be the content-addressed key {expected_key!r}")
    return digest, gate_receipt_s3_key


def _load_runtime_launch_contract(
    path: Path,
) -> RuntimeLaunchContract:
    """Load launch-critical fields from one canonical runtime contract."""
    from paper.benchmark.pipeline.runtime_contract import (
        parse_runtime_contract,
        runtime_contract_s3_key,
        runtime_contract_sha256,
    )

    contract = parse_runtime_contract(path.read_bytes())
    digest = runtime_contract_sha256(contract)
    runtime = contract["runtime"]
    return RuntimeLaunchContract(
        ami_id=str(runtime["ami_id"]),
        container_image_digest=str(runtime["container_image_digest"]),
        git_sha=str(runtime["git_sha"]),
        instance_type=str(runtime["instance_type"]),
        s3_key=runtime_contract_s3_key(digest),
        sha256=digest,
    )


def _load_gate_receipt_identity(path: Path) -> GateReceiptIdentity:
    """Derive the immutable identity of one local gate receipt."""
    from paper.benchmark.experiments.r_cforest_reproducibility import (
        gate_receipt_s3_key,
    )

    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    return GateReceiptIdentity(
        s3_key=gate_receipt_s3_key(digest),
        sha256=digest,
    )


def _verify_published_runtime_contract(
    manifest_info: dict[str, str | int],
    expected: RuntimeLaunchContract,
) -> None:
    """Require manifest publication to attest the supplied runtime contract."""
    try:
        observed = _validate_runtime_contract_scope(
            manifest_info["runtime_contract_sha256"],
            manifest_info["runtime_contract_s3_key"],
        )
    except KeyError as exc:
        raise RuntimeError(
            "Published manifest omits the runtime contract digest or S3 key"
        ) from exc
    except ValueError as exc:
        raise RuntimeError("Published manifest has an invalid runtime contract scope") from exc
    expected_scope = (expected.sha256, expected.s3_key)
    if observed != expected_scope:
        raise RuntimeError(
            "Published manifest runtime contract does not match launch inputs: "
            f"expected sha256={expected.sha256!r}, s3_key={expected.s3_key!r}; "
            f"observed sha256={observed[0]!r}, s3_key={observed[1]!r}"
        )


def _verify_published_gate_receipt(
    manifest_info: dict[str, str | int],
    expected: GateReceiptIdentity,
) -> None:
    """Require manifest publication to attest the supplied gate receipt."""
    try:
        observed = _validate_gate_receipt_scope(
            manifest_info["gate_receipt_sha256"],
            manifest_info["gate_receipt_s3_key"],
        )
    except KeyError as exc:
        raise RuntimeError("Published manifest omits the gate receipt digest or S3 key") from exc
    except ValueError as exc:
        raise RuntimeError("Published manifest has an invalid gate receipt scope") from exc
    expected_scope = (expected.sha256, expected.s3_key)
    if observed != expected_scope:
        raise RuntimeError(
            "Published manifest gate receipt does not match launch inputs: "
            f"expected sha256={expected.sha256!r}, s3_key={expected.s3_key!r}; "
            f"observed sha256={observed[0]!r}, s3_key={observed[1]!r}"
        )


def _require_image_matches_runtime(
    image_uri: str,
    git_sha: str,
    runtime_contract: RuntimeLaunchContract,
) -> None:
    """Require the account-local image to match the frozen runtime identity."""
    image_digest = image_uri.rsplit("@", maxsplit=1)[1]
    if image_digest != runtime_contract.container_image_digest:
        raise ValueError(
            "image_uri digest differs from the runtime contract: "
            f"{image_digest!r} != {runtime_contract.container_image_digest!r}"
        )
    if git_sha != runtime_contract.git_sha:
        raise ValueError(
            "image Git revision differs from the runtime contract: "
            f"{git_sha!r} != {runtime_contract.git_sha!r}"
        )


def _api_client_token(
    *,
    ami_id: str,
    instance_profile_name: str,
    instance_type: str,
    launch_id: str,
    security_group_id: str,
    user_data: str,
) -> str:
    """Derive EC2 idempotency from the complete API instance request."""
    payload = json.dumps(
        {
            "ami_id": ami_id,
            "instance_profile_name": instance_profile_name,
            "instance_type": instance_type,
            "launch_id": launch_id,
            "security_group_id": security_group_id,
            "user_data_sha256": hashlib.sha256(user_data.encode()).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"citrees-api-{hashlib.sha256(payload).hexdigest()[:52]}"


def _wait_for_api_ready(
    api_url: str,
    *,
    artifact_prefix: str,
    campaign_sha256: str,
    canonical_manifest_s3_key: str,
    canonical_manifest_sha256: str,
    gate_receipt_s3_key: str,
    gate_receipt_sha256: str,
    manifest_sha256: str,
    max_cell_attempts: int,
    runtime_contract_s3_key: str,
    runtime_contract_sha256: str,
    stage: str,
    timeout_seconds: float = API_READINESS_TIMEOUT_SECONDS,
    poll_interval: float = API_POLL_INTERVAL_SECONDS,
) -> None:
    """Wait until the API reports the exact immutable queue scope."""
    deadline = time.monotonic() + timeout_seconds
    last_error = "no response"
    with httpx.Client(base_url=api_url, timeout=30.0) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get("/status")
                response.raise_for_status()
                status = response.json()
                observed = {
                    "artifact_prefix": status.get("artifact_prefix"),
                    "campaign_sha256": status.get("campaign_sha256"),
                    "canonical_manifest_s3_key": status.get("canonical_manifest_s3_key"),
                    "canonical_manifest_sha256": status.get("canonical_manifest_sha256"),
                    "gate_receipt_s3_key": status.get("gate_receipt_s3_key"),
                    "gate_receipt_sha256": status.get("gate_receipt_sha256"),
                    "manifest_sha256": status.get("manifest_sha256"),
                    "max_cell_attempts": status.get("max_cell_attempts"),
                    "runtime_contract_s3_key": status.get("runtime_contract_s3_key"),
                    "runtime_contract_sha256": status.get("runtime_contract_sha256"),
                    "stage": status.get("stage"),
                }
                expected = {
                    "artifact_prefix": artifact_prefix,
                    "campaign_sha256": campaign_sha256,
                    "canonical_manifest_s3_key": canonical_manifest_s3_key,
                    "canonical_manifest_sha256": canonical_manifest_sha256,
                    "gate_receipt_s3_key": gate_receipt_s3_key,
                    "gate_receipt_sha256": gate_receipt_sha256,
                    "manifest_sha256": manifest_sha256,
                    "max_cell_attempts": max_cell_attempts,
                    "runtime_contract_s3_key": runtime_contract_s3_key,
                    "runtime_contract_sha256": runtime_contract_sha256,
                    "stage": stage,
                }
                if observed == expected:
                    return
                last_error = f"scope mismatch: expected {expected}, observed {observed}"
            except (httpx.HTTPError, ValueError) as exc:
                last_error = str(exc)
            time.sleep(poll_interval)
    raise RuntimeError(f"API did not become ready at {api_url}: {last_error}")


def get_ami(region: str) -> str:
    """Get the latest Amazon Linux 2023 AMI ID."""
    ssm = boto3.client("ssm", region_name=region)
    ami_param = ssm.get_parameter(
        Name="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
    )
    return ami_param["Parameter"]["Value"]


def get_default_subnet_ids(ec2: Any, *, instance_type: str | None = None) -> list[str]:
    """Return default subnet IDs sorted by AZ for diversified placement."""
    offered_azs: set[str] | None = None
    if instance_type:
        offerings = ec2.describe_instance_type_offerings(
            LocationType="availability-zone",
            Filters=[{"Name": "instance-type", "Values": [instance_type]}],
        )
        offered_azs = {item["Location"] for item in offerings.get("InstanceTypeOfferings", [])}

    response = ec2.describe_subnets(Filters=[{"Name": "default-for-az", "Values": ["true"]}])
    subnets = response.get("Subnets", [])
    if offered_azs is not None:
        subnets = [
            subnet for subnet in subnets if subnet.get("AvailabilityZone", "") in offered_azs
        ]
    subnets = sorted(
        subnets,
        key=lambda subnet: (
            subnet.get("AvailabilityZone", ""),
            subnet.get("SubnetId", ""),
        ),
    )
    return [subnet["SubnetId"] for subnet in subnets]


def _resolve_worker_subnets(
    ec2: Any,
    *,
    excluded_availability_zones: Sequence[str],
    instance_type: str,
    subnet_ids: Sequence[str],
) -> tuple[_WorkerSubnetPlacement, ...]:
    """Validate and order the exact subnets used by benchmark workers."""
    normalized_subnets = tuple(subnet_id.strip() for subnet_id in subnet_ids)
    if any(not subnet_id for subnet_id in normalized_subnets):
        raise ValueError("subnet_ids must not contain empty values")
    if len(set(normalized_subnets)) != len(normalized_subnets):
        raise ValueError("subnet_ids must be unique")

    excluded_zones = tuple(zone.strip() for zone in excluded_availability_zones)
    if any(not zone for zone in excluded_zones):
        raise ValueError("excluded_availability_zones must not contain empty values")
    if len(set(excluded_zones)) != len(excluded_zones):
        raise ValueError("excluded_availability_zones must be unique")

    offerings = ec2.describe_instance_type_offerings(
        LocationType="availability-zone",
        Filters=[{"Name": "instance-type", "Values": [instance_type]}],
    )
    offered_zones = {
        str(item["Location"])
        for item in offerings.get("InstanceTypeOfferings", [])
        if isinstance(item, dict) and item.get("Location")
    }
    if not offered_zones:
        raise RuntimeError(f"EC2 returned no availability zones for {instance_type}")

    if normalized_subnets:
        response = ec2.describe_subnets(SubnetIds=list(normalized_subnets))
    else:
        response = ec2.describe_subnets(Filters=[{"Name": "default-for-az", "Values": ["true"]}])
    rows = response.get("Subnets", [])
    if not isinstance(rows, list):
        raise RuntimeError("EC2 subnet lookup returned an invalid response")

    placements_by_id: dict[str, _WorkerSubnetPlacement] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("EC2 subnet lookup returned an invalid row")
        subnet_id = row.get("SubnetId")
        availability_zone = row.get("AvailabilityZone")
        vpc_id = row.get("VpcId")
        if not isinstance(subnet_id, str) or not subnet_id:
            raise RuntimeError("EC2 subnet lookup omitted SubnetId")
        if not isinstance(availability_zone, str) or not availability_zone:
            raise RuntimeError("EC2 subnet lookup omitted AvailabilityZone")
        if not isinstance(vpc_id, str) or not vpc_id:
            raise RuntimeError("EC2 subnet lookup omitted placement identity")
        if normalized_subnets and subnet_id not in normalized_subnets:
            raise RuntimeError(f"EC2 returned unrequested subnet {subnet_id}")
        if row.get("State") != "available":
            raise RuntimeError(f"subnet {subnet_id} is not available")
        if row.get("DefaultForAz") is not True:
            raise RuntimeError(f"subnet {subnet_id} is outside the default VPC")
        if row.get("MapPublicIpOnLaunch") is not True:
            raise RuntimeError(f"subnet {subnet_id} does not assign public IP addresses")
        available_addresses = row.get("AvailableIpAddressCount")
        if not isinstance(available_addresses, int) or available_addresses < 1:
            raise RuntimeError(f"subnet {subnet_id} has no available IP addresses")
        if availability_zone not in offered_zones:
            if normalized_subnets:
                raise RuntimeError(
                    f"{instance_type} is not offered in subnet {subnet_id} "
                    f"availability zone {availability_zone}"
                )
            continue
        if availability_zone in excluded_zones:
            if normalized_subnets:
                raise RuntimeError(
                    f"subnet {subnet_id} is in excluded availability zone {availability_zone}"
                )
            continue
        placements_by_id[subnet_id] = _WorkerSubnetPlacement(
            availability_zone=availability_zone,
            subnet_id=subnet_id,
            vpc_id=vpc_id,
        )

    if normalized_subnets:
        missing = sorted(set(normalized_subnets) - set(placements_by_id))
        if missing:
            raise RuntimeError(f"EC2 did not return requested subnets: {missing}")
        placements = tuple(placements_by_id[subnet_id] for subnet_id in normalized_subnets)
    else:
        placements = tuple(
            sorted(
                placements_by_id.values(),
                key=lambda placement: (
                    placement.availability_zone,
                    placement.subnet_id,
                ),
            )
        )
    if not placements:
        raise RuntimeError(
            f"No compatible default subnets remain for {instance_type} "
            f"outside {sorted(excluded_zones)}"
        )
    vpc_ids = {placement.vpc_id for placement in placements}
    if len(vpc_ids) != 1:
        raise RuntimeError(f"worker subnets span multiple VPCs: {sorted(vpc_ids)}")
    return placements


def per_boot_container_recovery_hook(
    container_name: str,
    *,
    restart_container: bool,
) -> str:
    """Return a per-boot hook with an explicit control-plane or compute policy."""
    if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", container_name) is None:
        raise ValueError(f"invalid container name: {container_name!r}")
    if not restart_container:
        script = textwrap.dedent(
            f"""\
            mkdir -p /var/lib/cloud/scripts/per-boot
            cat > /var/lib/cloud/scripts/per-boot/{container_name}-recover <<'RECOVERY'
            #!/bin/bash
            set -euo pipefail

            echo "Terminating recovered {container_name} instance"
            shutdown -h now || systemctl poweroff --force --force || poweroff -f || halt -f
            RECOVERY
            chmod 0755 /var/lib/cloud/scripts/per-boot/{container_name}-recover
            """
        )
        return textwrap.indent(script, "        ").rstrip()

    script = textwrap.dedent(
        f"""\
        mkdir -p /var/lib/cloud/scripts/per-boot
        cat > /var/lib/cloud/scripts/per-boot/{container_name}-recover <<'RECOVERY'
        #!/bin/bash
        set -euo pipefail

        shutdown_instance() {{
            trap - EXIT
            echo "Terminating recovered {container_name} instance"
            shutdown -h now || systemctl poweroff --force --force || poweroff -f || halt -f || true
        }}
        trap shutdown_instance EXIT

        systemctl start docker
        if ! docker inspect {container_name} >/dev/null 2>&1; then
            echo "Container {container_name} is missing after host recovery"
            exit 1
        fi
        if [ "$(docker inspect --format '{{{{.State.Running}}}}' {container_name})" = "true" ]; then
            echo "Container {container_name} is already running"
            trap - EXIT
            exit 0
        fi

        echo "Restarting {container_name} after host recovery"
        docker start {container_name}
        docker wait {container_name}
        RECOVERY
        chmod 0755 /var/lib/cloud/scripts/per-boot/{container_name}-recover
        """
    )
    return textwrap.indent(script, "        ").rstrip()


def _make_worker_user_data(
    *,
    region: str,
    ecr_uri: str,
    image_uri: str,
    api_url: str,
    bucket: str,
    git_sha: str,
    instance_type: str,
    artifact_prefix: str,
    campaign_sha256: str,
    canonical_manifest_s3_key: str,
    canonical_manifest_sha256: str,
    gate_receipt_s3_key: str,
    gate_receipt_sha256: str,
    manifest_s3_key: str,
    manifest_sha256: str,
    runtime_contract_s3_key: str,
    runtime_contract_sha256: str,
    stage: str,
) -> str:
    """Generate EC2 user data script that pulls and runs the worker container."""
    recovery_hook = per_boot_container_recovery_hook(
        "citrees-worker",
        restart_container=False,
    )
    return textwrap.dedent(
        f"""\
        #!/bin/bash
        exec > >(tee /var/log/user-data.log) 2>&1
        set -euo pipefail

        shutdown_instance() {{
            trap - EXIT
            echo "Terminating worker instance"
            shutdown -h now || systemctl poweroff --force --force || poweroff -f || halt -f || true
        }}
        trap shutdown_instance EXIT

        # Instance metadata (IMDSv2)
        TOKEN=$(curl --fail --silent --show-error --request PUT \\
            "http://169.254.169.254/latest/api/token" \\
            -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
        test -n "$TOKEN"
        INSTANCE_ID=$(curl --fail --silent --show-error \\
            -H "X-aws-ec2-metadata-token: $TOKEN" \\
            http://169.254.169.254/latest/meta-data/instance-id)
        AVAILABILITY_ZONE=$(curl --fail --silent --show-error \\
            -H "X-aws-ec2-metadata-token: $TOKEN" \\
            http://169.254.169.254/latest/meta-data/placement/availability-zone)
        AMI_ID=$(curl --fail --silent --show-error \\
            -H "X-aws-ec2-metadata-token: $TOKEN" \\
            http://169.254.169.254/latest/meta-data/ami-id)
        test -n "$INSTANCE_ID"
        test -n "$AVAILABILITY_ZONE"
        test -n "$AMI_ID"

        echo "Instance: $INSTANCE_ID ($AVAILABILITY_ZONE, $AMI_ID)"

        # Install Docker + SSM agent
        yum install -y docker amazon-ssm-agent
        systemctl enable --now docker
        systemctl enable --now amazon-ssm-agent

        # Create CloudWatch log group (ignore if exists)
        aws logs create-log-group \\
            --log-group-name {LOG_GROUP_WORKER} \\
            --region {region} 2>/dev/null || true

        # Authenticate to ECR
        aws ecr get-login-password --region {region} | \\
            docker login --username AWS --password-stdin {ecr_uri}

        docker pull {image_uri}
{recovery_hook}

        # Pull and run the worker image
        docker run -d --restart no \\
            --init \\
            --name citrees-worker \\
            --log-driver=awslogs \\
            --log-opt awslogs-region={region} \\
            --log-opt awslogs-group={LOG_GROUP_WORKER} \\
            --log-opt awslogs-stream=$INSTANCE_ID \\
            -e S3_BUCKET={bucket} \\
            -e CITREES_API_URL={api_url} \\
            -e CITREES_ARTIFACT_PREFIX={shlex.quote(artifact_prefix)} \\
            -e CITREES_CAMPAIGN_SHA256={campaign_sha256} \\
            -e CITREES_CANONICAL_MANIFEST_S3_KEY={shlex.quote(canonical_manifest_s3_key)} \\
            -e CITREES_CANONICAL_MANIFEST_SHA256={canonical_manifest_sha256} \\
            -e CITREES_GATE_RECEIPT_S3_KEY={shlex.quote(gate_receipt_s3_key)} \\
            -e CITREES_GATE_RECEIPT_SHA256={gate_receipt_sha256} \\
            -e CITREES_MANIFEST_S3_KEY={shlex.quote(manifest_s3_key)} \\
            -e CITREES_MANIFEST_SHA256={manifest_sha256} \\
            -e CITREES_RUNTIME_CONTRACT_S3_KEY={shlex.quote(runtime_contract_s3_key)} \\
            -e CITREES_RUNTIME_CONTRACT_SHA256={runtime_contract_sha256} \\
            -e CITREES_STAGE={stage} \\
            -e CITREES_IMAGE_URI={image_uri} \\
            -e EC2_AMI_ID=$AMI_ID \\
            -e EC2_AVAILABILITY_ZONE=$AVAILABILITY_ZONE \\
            -e EC2_INSTANCE_ID=$INSTANCE_ID \\
            -e EC2_INSTANCE_TYPE={shlex.quote(instance_type)} \\
            -e AWS_DEFAULT_REGION={region} \\
            -e GIT_SHA={git_sha} \\
            {image_uri} \\
            python -m paper.benchmark.api.worker \\
                --api-url {api_url} \\
                --max-api-failures {WORKER_MAX_API_FAILURES}

        # Keep user data alive until the worker drains or fails. The EXIT trap
        # terminates the instance for both container exit and bootstrap errors.
        docker wait citrees-worker
    """
    )


def _make_api_user_data(
    *,
    region: str,
    ecr_uri: str,
    image_uri: str,
    bucket: str,
    git_sha: str,
    instance_type: str,
    artifact_prefix: str,
    campaign_sha256: str,
    canonical_manifest_s3_key: str,
    canonical_manifest_sha256: str,
    gate_receipt_s3_key: str,
    gate_receipt_sha256: str,
    manifest_s3_key: str,
    manifest_sha256: str,
    runtime_contract_s3_key: str,
    runtime_contract_sha256: str,
    stage: str,
    lease_seconds: int,
    max_cell_attempts: int,
) -> str:
    """Generate EC2 user data script that runs the API server container."""
    recovery_hook = per_boot_container_recovery_hook(
        "citrees-api",
        restart_container=True,
    )
    return textwrap.dedent(
        f"""\
        #!/bin/bash
        exec > >(tee /var/log/user-data.log) 2>&1
        set -euo pipefail

        shutdown_instance() {{
            trap - EXIT
            echo "Terminating API instance"
            shutdown -h now || systemctl poweroff --force --force || poweroff -f || halt -f || true
        }}
        trap shutdown_instance EXIT

        # Instance metadata (IMDSv2)
        TOKEN=$(curl --fail --silent --show-error --request PUT \\
            "http://169.254.169.254/latest/api/token" \\
            -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
        test -n "$TOKEN"
        INSTANCE_ID=$(curl --fail --silent --show-error \\
            -H "X-aws-ec2-metadata-token: $TOKEN" \\
            http://169.254.169.254/latest/meta-data/instance-id)
        AVAILABILITY_ZONE=$(curl --fail --silent --show-error \\
            -H "X-aws-ec2-metadata-token: $TOKEN" \\
            http://169.254.169.254/latest/meta-data/placement/availability-zone)
        AMI_ID=$(curl --fail --silent --show-error \\
            -H "X-aws-ec2-metadata-token: $TOKEN" \\
            http://169.254.169.254/latest/meta-data/ami-id)
        test -n "$INSTANCE_ID"
        test -n "$AVAILABILITY_ZONE"
        test -n "$AMI_ID"

        echo "Instance: $INSTANCE_ID ($AVAILABILITY_ZONE, $AMI_ID)"

        # Install Docker + SSM agent
        yum install -y docker amazon-ssm-agent
        systemctl enable --now docker
        systemctl enable --now amazon-ssm-agent

        # Create CloudWatch log group (ignore if exists)
        aws logs create-log-group \\
            --log-group-name {LOG_GROUP_API} \\
            --region {region} 2>/dev/null || true

        # Authenticate to ECR
        aws ecr get-login-password --region {region} | \\
            docker login --username AWS --password-stdin {ecr_uri}

        # Pull and run the API server
        docker pull {image_uri}
{recovery_hook}
        docker run -d --restart no \\
            --name citrees-api \\
            -p 8000:8000 \\
            --log-driver=awslogs \\
            --log-opt awslogs-region={region} \\
            --log-opt awslogs-group={LOG_GROUP_API} \\
            --log-opt awslogs-stream=$INSTANCE_ID \\
            -e S3_BUCKET={bucket} \\
            -e CITREES_ARTIFACT_PREFIX={shlex.quote(artifact_prefix)} \\
            -e CITREES_CAMPAIGN_SHA256={campaign_sha256} \\
            -e CITREES_CANONICAL_MANIFEST_S3_KEY={shlex.quote(canonical_manifest_s3_key)} \\
            -e CITREES_CANONICAL_MANIFEST_SHA256={canonical_manifest_sha256} \\
            -e CITREES_GATE_RECEIPT_S3_KEY={shlex.quote(gate_receipt_s3_key)} \\
            -e CITREES_GATE_RECEIPT_SHA256={gate_receipt_sha256} \\
            -e CITREES_MANIFEST_S3_KEY={shlex.quote(manifest_s3_key)} \\
            -e CITREES_MANIFEST_SHA256={manifest_sha256} \\
            -e CITREES_RUNTIME_CONTRACT_S3_KEY={shlex.quote(runtime_contract_s3_key)} \\
            -e CITREES_RUNTIME_CONTRACT_SHA256={runtime_contract_sha256} \\
            -e CITREES_STAGE={stage} \\
            -e CITREES_LEASE_SECONDS={lease_seconds} \\
            -e CITREES_MAX_CELL_ATTEMPTS={max_cell_attempts} \\
            -e CITREES_IMAGE_URI={image_uri} \\
            -e EC2_AMI_ID=$AMI_ID \\
            -e EC2_AVAILABILITY_ZONE=$AVAILABILITY_ZONE \\
            -e EC2_INSTANCE_ID=$INSTANCE_ID \\
            -e EC2_INSTANCE_TYPE={shlex.quote(instance_type)} \\
            -e AWS_DEFAULT_REGION={region} \\
            -e GIT_SHA={git_sha} \\
            {image_uri} \\
            uvicorn paper.benchmark.api.server:app --host 0.0.0.0 --port 8000

        # Keep user data alive while the API is healthy. Any container exit or
        # bootstrap failure reaches the EXIT trap and terminates the instance.
        docker wait citrees-api
    """
    )


def _make_mechanism_user_data(
    *,
    region: str,
    ecr_uri: str,
    image_uri: str,
    bucket: str,
    git_sha: str,
    shard_index: int,
    num_shards: int,
    tasks: Sequence[str],
    source: str,
    datasets: Sequence[str],
    seeds: Sequence[int],
    folds: Sequence[int],
    model_variants: Sequence[str],
    ranking_variants: Sequence[str],
    n_jobs: int,
    downstream_n_jobs: int,
) -> str:
    """Generate EC2 user data for one CIF mechanism-ablation shard."""
    from paper.benchmark.experiments.cif_mechanism_ablation import (
        distributed_output_uri,
        mechanism_specification_sha256,
    )

    specification_sha256 = mechanism_specification_sha256(
        tasks=tasks,
        source=source,
        datasets=datasets,
        seeds=seeds,
        folds=folds,
        model_variants=model_variants,
        ranking_variants=ranking_variants,
        n_jobs=n_jobs,
        downstream_n_jobs=downstream_n_jobs,
    )
    output_uri = distributed_output_uri(
        bucket=bucket,
        image_uri=image_uri,
        specification_sha256=specification_sha256,
    )
    command = [
        "python",
        "-m",
        "paper.benchmark.experiments.cif_mechanism_ablation",
        "--distributed",
        "--tasks",
        _csv_arg(tasks),
        "--source",
        source,
        "--seeds",
        _csv_arg(seeds),
        "--folds",
        _csv_arg(folds),
        "--model-variants",
        _csv_arg(model_variants),
        "--ranking-variants",
        _csv_arg(ranking_variants),
        "--num-shards",
        str(num_shards),
        "--shard-index",
        str(shard_index),
        "--n-jobs",
        str(n_jobs),
        "--downstream-n-jobs",
        str(downstream_n_jobs),
    ]
    if datasets:
        command.extend(["--datasets", _csv_arg(datasets)])
    command_text = shlex.join(command)
    recovery_hook = per_boot_container_recovery_hook(
        "citrees-mechanism",
        restart_container=False,
    )

    return textwrap.dedent(
        f"""\
        #!/bin/bash
        exec > >(tee /var/log/user-data.log) 2>&1
        set -euo pipefail
        shutdown_instance() {{
            echo "Mechanism user-data exiting, shutting down instance"
            shutdown -h now || systemctl poweroff --force --force || poweroff -f || halt -f || true
        }}
        trap shutdown_instance EXIT

        # Instance metadata (IMDSv2)
        TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \\
            -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
        INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \\
            http://169.254.169.254/latest/meta-data/instance-id)

        echo "Instance: $INSTANCE_ID"
        echo "Mechanism shard: {shard_index}/{num_shards}"
        echo "Output: {output_uri}"

        # Install Docker + SSM agent
        yum install -y docker amazon-ssm-agent
        systemctl enable --now docker
        systemctl enable --now amazon-ssm-agent

        # Create CloudWatch log group (ignore if exists)
        aws logs create-log-group \\
            --log-group-name {LOG_GROUP_MECHANISM} \\
            --region {region} 2>/dev/null || true

        # Authenticate to ECR
        aws ecr get-login-password --region {region} | \\
            docker login --username AWS --password-stdin {ecr_uri}

        # Pull and run one independent mechanism-ablation shard.
        docker pull {image_uri}
{recovery_hook}
        docker run -d --restart no \\
            --name citrees-mechanism \\
            --log-driver=awslogs \\
            --log-opt awslogs-region={region} \\
            --log-opt awslogs-group={LOG_GROUP_MECHANISM} \\
            --log-opt awslogs-stream=$INSTANCE_ID \\
            -e S3_BUCKET={bucket} \\
            -e CITREES_IMAGE_URI={image_uri} \\
            -e CITREES_MECHANISM_SPEC_SHA256={specification_sha256} \\
            -e AWS_DEFAULT_REGION={region} \\
            -e GIT_SHA={git_sha} \\
            {image_uri} \\
            {command_text}

        # Wait for container to finish and terminate this shard instance.
        EXIT_CODE=$(docker wait citrees-mechanism || echo 1)
        EXIT_CODE=$(echo "$EXIT_CODE" | tail -n 1)
        EXIT_CODE=${{EXIT_CODE:-1}}
        echo "Mechanism container exited with code $EXIT_CODE"
        exit "$EXIT_CODE"
    """
    )


# ---------------------------------------------------------------------------
# API server
# ---------------------------------------------------------------------------


def launch_api(
    instance_type: str,
    image_uri: str,
    *,
    artifact_prefix: str,
    canonical_manifest_path: Path,
    gate_receipt_path: Path,
    launch_id: str,
    manifest_path: Path,
    runtime_contract_path: Path,
    stage: str,
    lease_seconds: int,
    max_cell_attempts: int,
    region: str = DEFAULT_REGION,
) -> dict[str, str]:
    """Launch one on-demand API server behind a stable private DNS endpoint.

    Returns instance identity plus stable worker and public operator URLs.
    """
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be a positive integer")
    if type(max_cell_attempts) is not int or max_cell_attempts <= 0:
        raise ValueError("max_cell_attempts must be a positive integer")
    launch_id = _validate_api_launch_id(launch_id)
    image_uri = validate_image_digest_uri(image_uri)
    artifact_prefix, stage = _validate_queue_scope(artifact_prefix, stage)
    runtime_contract = _load_runtime_launch_contract(runtime_contract_path)
    gate_receipt = _load_gate_receipt_identity(gate_receipt_path)
    manifest_info = publish_rerun_manifest(
        manifest_path,
        canonical_manifest_path,
        runtime_contract_path,
        gate_receipt_path,
        region=region,
    )
    _verify_published_runtime_contract(manifest_info, runtime_contract)
    _verify_published_gate_receipt(manifest_info, gate_receipt)
    runtime_contract_sha256 = runtime_contract.sha256
    runtime_contract_s3_key = runtime_contract.s3_key
    manifest_s3_key = str(manifest_info["key"])
    manifest_sha256 = str(manifest_info["sha256"])
    canonical_manifest_s3_key = str(manifest_info["canonical_manifest_s3_key"])
    canonical_manifest_sha256 = str(manifest_info["canonical_manifest_sha256"])
    campaign_sha256 = str(manifest_info["campaign_sha256"])
    ec2 = boto3.client("ec2", region_name=region)
    route53 = boto3.client("route53")
    active_instances = _api_instances(
        ec2,
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        stage=stage,
        states=("pending", "running", "stopping"),
    )
    if active_instances:
        instance_ids = sorted(str(instance["InstanceId"]) for instance in active_instances)
        raise RuntimeError(
            f"A citrees API server is already active for the exact campaign scope: {instance_ids}"
        )
    git_sha = validate_image_revision(
        image_uri,
        runtime_contract.git_sha,
        region=region,
    )
    _require_image_matches_runtime(image_uri, git_sha, runtime_contract)
    account_id = get_aws_account_id()
    bucket = get_resource_name(account_id)
    ecr_uri = image_uri.split("/")[0]
    sg_id = ensure_security_group(region)
    instance_profile_name = ensure_campaign_iam_profile(
        output_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        read_keys=(
            canonical_manifest_s3_key,
            gate_receipt.s3_key,
            manifest_s3_key,
            runtime_contract_s3_key,
        ),
        write_prefixes=(artifact_prefix,),
        region=region,
    )
    ami_id = runtime_contract.ami_id

    user_data = _make_api_user_data(
        region=region,
        ecr_uri=ecr_uri,
        image_uri=image_uri,
        bucket=bucket,
        git_sha=git_sha,
        instance_type=instance_type,
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        canonical_manifest_s3_key=canonical_manifest_s3_key,
        canonical_manifest_sha256=canonical_manifest_sha256,
        gate_receipt_s3_key=gate_receipt.s3_key,
        gate_receipt_sha256=gate_receipt.sha256,
        manifest_s3_key=manifest_s3_key,
        manifest_sha256=manifest_sha256,
        runtime_contract_s3_key=runtime_contract_s3_key,
        runtime_contract_sha256=runtime_contract_sha256,
        stage=stage,
        lease_seconds=lease_seconds,
        max_cell_attempts=max_cell_attempts,
    )

    market = _BENCHMARK_MARKET
    info(f"Launching {market} API server: {instance_type}, AMI={ami_id}")
    step(f"Launch identity: {launch_id}")
    step(f"Image: {image_uri}")
    step(f"Manifest: s3://{bucket}/{manifest_s3_key} ({manifest_info['cells']} cells)")
    step(f"Security group: {sg_id}")
    step(f"Instance profile: {instance_profile_name}")
    client_token = _api_client_token(
        ami_id=ami_id,
        instance_profile_name=instance_profile_name,
        instance_type=instance_type,
        launch_id=launch_id,
        security_group_id=sg_id,
        user_data=user_data,
    )
    endpoint: _ApiEndpoint | None = None
    instance_id: str | None = None
    try:
        endpoint = _ensure_api_endpoint(
            ec2,
            route53,
            artifact_prefix=artifact_prefix,
            campaign_sha256=campaign_sha256,
            launch_id=launch_id,
            stage=stage,
            region=region,
        )
        step(f"Stable worker endpoint: {endpoint.api_url}")
        run_kwargs: dict[str, Any] = {
            "ImageId": ami_id,
            "InstanceType": instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "IamInstanceProfile": {"Name": instance_profile_name},
            "UserData": base64.b64encode(user_data.encode()).decode(),
            "MetadataOptions": {
                "HttpEndpoint": "enabled",
                "HttpPutResponseHopLimit": 2,
                "HttpTokens": "required",
            },
            "SecurityGroupIds": [sg_id],
            "InstanceInitiatedShutdownBehavior": "terminate",
            "ClientToken": client_token,
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": TAG_KEY, "Value": API_TAG_VALUE},
                        {"Key": "Name", "Value": "citrees-api"},
                        {"Key": "citrees-api-endpoint", "Value": endpoint.hostname},
                        {"Key": "citrees-api-launch-id", "Value": launch_id},
                        {"Key": "citrees-market", "Value": market},
                        {"Key": "citrees-artifact-prefix", "Value": artifact_prefix},
                        {"Key": "citrees-campaign-sha256", "Value": campaign_sha256},
                        {
                            "Key": "citrees-canonical-manifest-key",
                            "Value": canonical_manifest_s3_key,
                        },
                        {
                            "Key": "citrees-canonical-manifest-sha256",
                            "Value": canonical_manifest_sha256,
                        },
                        {
                            "Key": "citrees-gate-receipt-key",
                            "Value": gate_receipt.s3_key,
                        },
                        {
                            "Key": "citrees-gate-receipt-sha256",
                            "Value": gate_receipt.sha256,
                        },
                        {"Key": "citrees-manifest-key", "Value": manifest_s3_key},
                        {"Key": "citrees-manifest-sha256", "Value": manifest_sha256},
                        {"Key": "citrees-image-uri", "Value": image_uri},
                        {
                            "Key": "citrees-runtime-contract-key",
                            "Value": runtime_contract_s3_key,
                        },
                        {
                            "Key": "citrees-runtime-contract-sha256",
                            "Value": runtime_contract_sha256,
                        },
                        {"Key": "citrees-stage", "Value": stage},
                        {"Key": "citrees-lease-seconds", "Value": str(lease_seconds)},
                        {
                            "Key": "citrees-max-cell-attempts",
                            "Value": str(max_cell_attempts),
                        },
                    ],
                }
            ],
        }
        response = ec2.run_instances(**run_kwargs)
        instance_id = _single_instance_id(response)
        step(f"Instance: {instance_id}")

        info("Waiting for API instance addresses...")
        ec2_resource = boto3.resource("ec2", region_name=region)
        instance = ec2_resource.Instance(instance_id)
        instance.wait_until_running()
        instance.reload()
        public_ip = instance.public_ip_address
        private_ip = instance.private_ip_address
        if not public_ip or not private_ip:
            for _ in range(10):
                time.sleep(2)
                instance.reload()
                public_ip = instance.public_ip_address
                private_ip = instance.private_ip_address
                if public_ip and private_ip:
                    break
        if not public_ip:
            raise RuntimeError(f"API instance {instance_id} has no public IP")
        if not private_ip:
            raise RuntimeError(f"API instance {instance_id} has no private IP")

        public_api_url = f"http://{public_ip}:8000"
        info(f"Waiting for API readiness at {public_api_url}...")
        _wait_for_api_ready(
            public_api_url,
            artifact_prefix=artifact_prefix,
            campaign_sha256=campaign_sha256,
            canonical_manifest_s3_key=canonical_manifest_s3_key,
            canonical_manifest_sha256=canonical_manifest_sha256,
            gate_receipt_s3_key=gate_receipt.s3_key,
            gate_receipt_sha256=gate_receipt.sha256,
            manifest_sha256=manifest_sha256,
            max_cell_attempts=max_cell_attempts,
            runtime_contract_s3_key=runtime_contract_s3_key,
            runtime_contract_sha256=runtime_contract_sha256,
            stage=stage,
        )
        observed_instances = _api_instances(
            ec2,
            artifact_prefix=artifact_prefix,
            campaign_sha256=campaign_sha256,
            stage=stage,
            states=("pending", "running", "stopping"),
        )
        observed_ids = sorted(str(item["InstanceId"]) for item in observed_instances)
        if observed_ids != [instance_id]:
            raise RuntimeError(
                "API replacement is not the sole active server for the exact campaign: "
                f"{observed_ids}"
            )
        _upsert_api_endpoint(route53, endpoint, private_ip)
    except Exception as exc:
        cleanup_errors: list[str] = []
        if instance_id is not None:
            try:
                ec2.terminate_instances(InstanceIds=[instance_id])
            except Exception as cleanup_exc:
                cleanup_errors.append(f"instance termination failed: {cleanup_exc}")
        if endpoint is not None:
            try:
                _delete_api_endpoint(
                    ec2,
                    route53,
                    artifact_prefix=artifact_prefix,
                    campaign_sha256=campaign_sha256,
                    stage=stage,
                    region=region,
                )
            except Exception as cleanup_exc:
                cleanup_errors.append(f"endpoint deletion failed: {cleanup_exc}")
        if cleanup_errors:
            raise RuntimeError(
                "API launch failed and scoped cleanup was incomplete: " + "; ".join(cleanup_errors)
            ) from exc
        raise
    success(f"API server ready at {endpoint.api_url}")
    return {
        "instance_id": instance_id,
        "launch_id": launch_id,
        "public_ip": public_ip,
        "api_url": endpoint.api_url,
        "public_api_url": public_api_url,
    }


def _api_instance_filters(
    *,
    artifact_prefix: str,
    campaign_sha256: str,
    stage: str,
    states: Sequence[str],
) -> list[dict[str, Any]]:
    """Return exact EC2 filters for one immutable API campaign scope."""
    artifact_prefix, stage = _validate_queue_scope(artifact_prefix, stage)
    from paper.benchmark.pipeline.manifest import validate_manifest_sha256

    campaign_sha256 = validate_manifest_sha256(campaign_sha256)
    return [
        {"Name": f"tag:{TAG_KEY}", "Values": [API_TAG_VALUE]},
        {
            "Name": "tag:citrees-artifact-prefix",
            "Values": [artifact_prefix],
        },
        {
            "Name": "tag:citrees-campaign-sha256",
            "Values": [campaign_sha256],
        },
        {"Name": "tag:citrees-stage", "Values": [stage]},
        {"Name": "instance-state-name", "Values": list(states)},
    ]


def _api_instances(
    ec2: Any,
    *,
    artifact_prefix: str,
    campaign_sha256: str,
    stage: str,
    states: Sequence[str],
) -> list[dict[str, Any]]:
    """Return API instances in one exact campaign scope and state set."""
    response = ec2.describe_instances(
        Filters=_api_instance_filters(
            artifact_prefix=artifact_prefix,
            campaign_sha256=campaign_sha256,
            stage=stage,
            states=states,
        )
    )
    instances = [
        instance
        for reservation in response.get("Reservations", [])
        for instance in reservation.get("Instances", [])
    ]
    for instance in instances:
        instance_id = instance.get("InstanceId") if isinstance(instance, dict) else None
        if not isinstance(instance_id, str) or not instance_id:
            raise RuntimeError("EC2 returned an API server without an instance ID")
    return instances


def get_api_scope(
    *,
    artifact_prefix: str,
    campaign_sha256: str,
    stage: str,
    region: str = DEFAULT_REGION,
) -> ApiScope | None:
    """Return the running API server for one immutable campaign scope."""
    ec2 = boto3.client("ec2", region_name=region)
    instances = _api_instances(
        ec2,
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        stage=stage,
        states=("pending", "running"),
    )
    if not instances:
        return None
    if len(instances) != 1:
        instance_ids = sorted(instance["InstanceId"] for instance in instances)
        raise RuntimeError(
            f"Expected one running API server for the exact campaign scope, found {instance_ids}"
        )

    instance = instances[0]
    private_ip = instance.get("PrivateIpAddress")
    if not private_ip:
        raise RuntimeError(f"API server {instance['InstanceId']} has no private IP")
    public_ip = instance.get("PublicIpAddress")
    if not public_ip:
        raise RuntimeError(f"API server {instance['InstanceId']} has no public IP")
    tags = {tag["Key"]: tag["Value"] for tag in instance.get("Tags", [])}
    required_tags = {
        "citrees-api-endpoint",
        "citrees-api-launch-id",
        "citrees-artifact-prefix",
        "citrees-campaign-sha256",
        "citrees-canonical-manifest-key",
        "citrees-canonical-manifest-sha256",
        "citrees-gate-receipt-key",
        "citrees-gate-receipt-sha256",
        "citrees-image-uri",
        "citrees-manifest-key",
        "citrees-manifest-sha256",
        "citrees-market",
        "citrees-max-cell-attempts",
        "citrees-runtime-contract-key",
        "citrees-runtime-contract-sha256",
        "citrees-stage",
    }
    missing = sorted(required_tags - set(tags))
    if missing:
        raise RuntimeError(f"API server {instance['InstanceId']} is missing scope tags: {missing}")

    from paper.benchmark.pipeline.manifest import (
        canonical_manifest_s3_key,
        manifest_s3_key,
        validate_manifest_sha256,
    )

    artifact_prefix, stage = _validate_queue_scope(
        tags["citrees-artifact-prefix"],
        tags["citrees-stage"],
    )
    _validate_api_launch_id(tags["citrees-api-launch-id"])
    image_uri = validate_image_digest_uri(tags["citrees-image-uri"])
    market = tags["citrees-market"]
    if market != _BENCHMARK_MARKET:
        raise RuntimeError(f"API server market tag must be {_BENCHMARK_MARKET!r}, got {market!r}")
    _require_on_demand_instance(instance, source="API server")
    campaign_sha256 = validate_manifest_sha256(tags["citrees-campaign-sha256"])
    _hosted_zone_name, _record_name, endpoint_hostname = _api_endpoint_names(
        campaign_sha256,
        stage,
    )
    if tags["citrees-api-endpoint"] != endpoint_hostname:
        raise RuntimeError(
            "API server endpoint tag differs from the deterministic campaign endpoint: "
            f"{tags['citrees-api-endpoint']!r} != {endpoint_hostname!r}"
        )
    canonical_manifest_sha256 = validate_manifest_sha256(tags["citrees-canonical-manifest-sha256"])
    canonical_manifest_key = canonical_manifest_s3_key(canonical_manifest_sha256)
    if tags["citrees-canonical-manifest-key"] != canonical_manifest_key:
        raise RuntimeError(
            "API server canonical manifest key is not content-addressed: "
            f"{tags['citrees-canonical-manifest-key']!r}"
        )
    gate_receipt_sha256, gate_receipt_s3_key = _validate_gate_receipt_scope(
        tags["citrees-gate-receipt-sha256"],
        tags["citrees-gate-receipt-key"],
    )
    manifest_sha256 = validate_manifest_sha256(tags["citrees-manifest-sha256"])
    manifest_key = manifest_s3_key(manifest_sha256)
    if tags["citrees-manifest-key"] != manifest_key:
        raise RuntimeError(
            f"API server manifest key is not content-addressed: {tags['citrees-manifest-key']!r}"
        )
    runtime_contract_sha256, runtime_contract_s3_key = _validate_runtime_contract_scope(
        tags["citrees-runtime-contract-sha256"],
        tags["citrees-runtime-contract-key"],
    )
    try:
        max_cell_attempts = int(tags["citrees-max-cell-attempts"])
    except ValueError as exc:
        raise RuntimeError("API server max-cell-attempts tag is invalid") from exc
    if max_cell_attempts <= 0 or str(max_cell_attempts) != tags["citrees-max-cell-attempts"]:
        raise RuntimeError("API server max-cell-attempts tag is invalid")
    return ApiScope(
        api_url=f"http://{endpoint_hostname}:8000",
        public_api_url=f"http://{public_ip}:8000",
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        canonical_manifest_s3_key=canonical_manifest_key,
        canonical_manifest_sha256=canonical_manifest_sha256,
        gate_receipt_s3_key=gate_receipt_s3_key,
        gate_receipt_sha256=gate_receipt_sha256,
        image_uri=image_uri,
        manifest_s3_key=manifest_key,
        manifest_sha256=manifest_sha256,
        market=market,
        max_cell_attempts=max_cell_attempts,
        runtime_contract_s3_key=runtime_contract_s3_key,
        runtime_contract_sha256=runtime_contract_sha256,
        stage=stage,
    )


def get_api_url(
    *,
    artifact_prefix: str,
    campaign_sha256: str,
    stage: str,
    region: str = DEFAULT_REGION,
) -> str | None:
    """Return the private URL of one campaign-scoped API server."""
    scope = get_api_scope(
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        stage=stage,
        region=region,
    )
    return scope.api_url if scope is not None else None


def terminate_api(
    *,
    artifact_prefix: str,
    campaign_sha256: str,
    stage: str,
    region: str = DEFAULT_REGION,
) -> str | None:
    """Terminate the API and delete its campaign-scoped private endpoint.

    Returns the terminated instance ID, or None if no API instance found.
    """
    ec2 = boto3.client("ec2", region_name=region)
    route53 = boto3.client("route53")
    instances = _api_instances(
        ec2,
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        stage=stage,
        states=("pending", "running", "stopping"),
    )
    instance_ids = [str(instance["InstanceId"]) for instance in instances]
    if len(instance_ids) != 1:
        if instance_ids:
            raise RuntimeError(
                f"Expected one API server for the exact campaign scope, "
                f"found {sorted(instance_ids)}"
            )
        terminated = None
    else:
        ec2.terminate_instances(InstanceIds=instance_ids)
        terminated = instance_ids[0]
        success(f"Terminated API server: {terminated}")
    endpoint_deleted = _delete_api_endpoint(
        ec2,
        route53,
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        stage=stage,
        region=region,
    )
    if endpoint_deleted:
        success("Deleted campaign-scoped API endpoint")
    if terminated is None:
        info("No API server instance found for the exact campaign scope")
    return terminated


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


def launch_workers(
    n: int,
    image_uri: str,
    *,
    artifact_prefix: str,
    canonical_manifest_path: Path,
    excluded_availability_zones: Sequence[str] = (),
    gate_receipt_path: Path,
    launch_id: str,
    manifest_path: Path,
    runtime_contract_path: Path,
    stage: str,
    subnet_ids: Sequence[str] = (),
    region: str = DEFAULT_REGION,
) -> list[str]:
    """Launch N EC2 worker instances.

    The API server's stable private DNS endpoint and immutable scope are
    auto-discovered. Each instance runs a Docker container that pulls configs
    from the API server and executes them.

    Parameters
    ----------
    n : int
        Number of instances to launch.
    image_uri : str
        Full ECR image URI.
    launch_id : str
        Stable operator-supplied identity for this exact launch batch.
    region : str
        AWS region.

    Returns
    -------
    list[str]
        Instance IDs of launched workers.

    Raises
    ------
    RuntimeError
        If no running API server instance can be found.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    launch_id = _validate_worker_launch_id(launch_id)
    image_uri = validate_image_digest_uri(image_uri)
    runtime_contract = _load_runtime_launch_contract(runtime_contract_path)
    gate_receipt = _load_gate_receipt_identity(gate_receipt_path)
    instance_type = runtime_contract.instance_type
    instance_type = instance_type.strip()
    instance_family, separator, instance_size = instance_type.partition(".")
    if not instance_family or not separator or not instance_size:
        raise ValueError("instance_type must include an EC2 family and size")
    artifact_prefix, stage = _validate_queue_scope(artifact_prefix, stage)
    market = _BENCHMARK_MARKET
    manifest_info = publish_rerun_manifest(
        manifest_path,
        canonical_manifest_path,
        runtime_contract_path,
        gate_receipt_path,
        region=region,
    )
    _verify_published_runtime_contract(manifest_info, runtime_contract)
    _verify_published_gate_receipt(manifest_info, gate_receipt)
    runtime_contract_sha256 = runtime_contract.sha256
    runtime_contract_s3_key = runtime_contract.s3_key
    manifest_s3_key = str(manifest_info["key"])
    manifest_sha256 = str(manifest_info["sha256"])
    canonical_manifest_s3_key = str(manifest_info["canonical_manifest_s3_key"])
    canonical_manifest_sha256 = str(manifest_info["canonical_manifest_sha256"])
    campaign_sha256 = str(manifest_info["campaign_sha256"])
    api_scope = get_api_scope(
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        stage=stage,
        region=region,
    )
    if api_scope is None:
        raise RuntimeError(
            "No running API server found. Launch one with: citrees-exp infra launch-api"
        )
    expected_scope = {
        "artifact_prefix": artifact_prefix,
        "campaign_sha256": campaign_sha256,
        "canonical_manifest_s3_key": canonical_manifest_s3_key,
        "canonical_manifest_sha256": canonical_manifest_sha256,
        "gate_receipt_s3_key": gate_receipt.s3_key,
        "gate_receipt_sha256": gate_receipt.sha256,
        "image_uri": image_uri,
        "manifest_s3_key": manifest_s3_key,
        "manifest_sha256": manifest_sha256,
        "market": market,
        "runtime_contract_s3_key": runtime_contract_s3_key,
        "runtime_contract_sha256": runtime_contract_sha256,
        "stage": stage,
    }
    observed_scope = {
        "artifact_prefix": api_scope.artifact_prefix,
        "campaign_sha256": api_scope.campaign_sha256,
        "canonical_manifest_s3_key": api_scope.canonical_manifest_s3_key,
        "canonical_manifest_sha256": api_scope.canonical_manifest_sha256,
        "gate_receipt_s3_key": api_scope.gate_receipt_s3_key,
        "gate_receipt_sha256": api_scope.gate_receipt_sha256,
        "image_uri": api_scope.image_uri,
        "manifest_s3_key": api_scope.manifest_s3_key,
        "manifest_sha256": api_scope.manifest_sha256,
        "market": api_scope.market,
        "runtime_contract_s3_key": api_scope.runtime_contract_s3_key,
        "runtime_contract_sha256": api_scope.runtime_contract_sha256,
        "stage": api_scope.stage,
    }
    if observed_scope != expected_scope:
        raise RuntimeError(
            f"Worker scope does not match running API: expected {expected_scope}, "
            f"observed {observed_scope}"
        )
    git_sha = validate_image_revision(
        image_uri,
        runtime_contract.git_sha,
        region=region,
    )
    _require_image_matches_runtime(image_uri, git_sha, runtime_contract)
    sg_id = ensure_security_group(region)
    instance_profile_name = ensure_campaign_iam_profile(
        output_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        read_keys=(
            canonical_manifest_s3_key,
            gate_receipt.s3_key,
            manifest_s3_key,
            runtime_contract_s3_key,
        ),
        write_prefixes=(artifact_prefix,),
        region=region,
    )
    _wait_for_api_ready(
        api_scope.public_api_url,
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        canonical_manifest_s3_key=canonical_manifest_s3_key,
        canonical_manifest_sha256=canonical_manifest_sha256,
        gate_receipt_s3_key=gate_receipt.s3_key,
        gate_receipt_sha256=gate_receipt.sha256,
        manifest_sha256=manifest_sha256,
        max_cell_attempts=api_scope.max_cell_attempts,
        runtime_contract_s3_key=runtime_contract_s3_key,
        runtime_contract_sha256=runtime_contract_sha256,
        stage=stage,
    )
    api_url = api_scope.api_url

    ec2 = boto3.client("ec2", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    placements = _resolve_worker_subnets(
        ec2,
        excluded_availability_zones=excluded_availability_zones,
        instance_type=instance_type,
        subnet_ids=subnet_ids,
    )
    normalized_excluded_zones = sorted(zone.strip() for zone in excluded_availability_zones)
    slot_placements = tuple(placements[(slot - 1) % len(placements)] for slot in range(1, n + 1))
    account_id = get_aws_account_id()
    bucket = get_resource_name(account_id)
    ecr_uri = image_uri.split("/")[0]
    launch_prefix = _worker_launch_prefix(artifact_prefix, launch_id)
    intent_key = f"{launch_prefix}/intent.json"
    existing_intent = _read_immutable_json(
        s3,
        bucket=bucket,
        key=intent_key,
        record_kind="worker-launch-intent",
    )
    ami_id = runtime_contract.ami_id

    user_data = _make_worker_user_data(
        region=region,
        ecr_uri=ecr_uri,
        image_uri=image_uri,
        api_url=api_url,
        bucket=bucket,
        git_sha=git_sha,
        instance_type=instance_type,
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        canonical_manifest_s3_key=canonical_manifest_s3_key,
        canonical_manifest_sha256=canonical_manifest_sha256,
        gate_receipt_s3_key=gate_receipt.s3_key,
        gate_receipt_sha256=gate_receipt.sha256,
        manifest_s3_key=manifest_s3_key,
        manifest_sha256=manifest_sha256,
        runtime_contract_s3_key=runtime_contract_s3_key,
        runtime_contract_sha256=runtime_contract_sha256,
        stage=stage,
    )

    encoded_user_data = base64.b64encode(user_data.encode()).decode()
    base_run_kwargs: dict[str, Any] = {
        "ImageId": ami_id,
        "InstanceType": instance_type,
        "MinCount": 1,
        "MaxCount": 1,
        "IamInstanceProfile": {"Name": instance_profile_name},
        "UserData": encoded_user_data,
        "MetadataOptions": {
            "HttpEndpoint": "enabled",
            "HttpPutResponseHopLimit": 2,
            "HttpTokens": "required",
        },
        "SecurityGroupIds": [sg_id],
        "InstanceInitiatedShutdownBehavior": "terminate",
    }
    request_contract = dict(base_run_kwargs)
    request_contract.pop("UserData")
    request_contract["UserDataSha256"] = hashlib.sha256(encoded_user_data.encode()).hexdigest()

    intent = {
        "ami_id": ami_id,
        "api_scope": {
            "api_url": api_scope.api_url,
            "artifact_prefix": api_scope.artifact_prefix,
            "campaign_sha256": api_scope.campaign_sha256,
            "canonical_manifest_s3_key": api_scope.canonical_manifest_s3_key,
            "canonical_manifest_sha256": api_scope.canonical_manifest_sha256,
            "gate_receipt_s3_key": api_scope.gate_receipt_s3_key,
            "gate_receipt_sha256": api_scope.gate_receipt_sha256,
            "image_uri": api_scope.image_uri,
            "manifest_s3_key": api_scope.manifest_s3_key,
            "manifest_sha256": api_scope.manifest_sha256,
            "market": api_scope.market,
            "max_cell_attempts": api_scope.max_cell_attempts,
            "runtime_contract_s3_key": api_scope.runtime_contract_s3_key,
            "runtime_contract_sha256": api_scope.runtime_contract_sha256,
            "stage": api_scope.stage,
        },
        "artifact_prefix": artifact_prefix,
        "aws_account_id": account_id,
        "bucket": bucket,
        "campaign_sha256": campaign_sha256,
        "canonical_manifest_s3_key": canonical_manifest_s3_key,
        "canonical_manifest_sha256": canonical_manifest_sha256,
        "gate_receipt_s3_key": gate_receipt.s3_key,
        "gate_receipt_sha256": gate_receipt.sha256,
        "excluded_availability_zones": normalized_excluded_zones,
        "image_git_sha": git_sha,
        "image_uri": image_uri,
        "instance_family": instance_family,
        "instance_profile_name": instance_profile_name,
        "instance_type": instance_type,
        "kind": "citrees-benchmark-worker-launch-intent",
        "launch_id": launch_id,
        "manifest_s3_key": manifest_s3_key,
        "manifest_sha256": manifest_sha256,
        "market": market,
        "region": region,
        "request_contract": request_contract,
        "requested_instances": n,
        "runtime_contract_s3_key": runtime_contract_s3_key,
        "runtime_contract_sha256": runtime_contract_sha256,
        "schema_version": _WORKER_LAUNCH_SCHEMA_VERSION,
        "security_group_id": sg_id,
        "slot_placements": [
            {
                "availability_zone": placement.availability_zone,
                "slot": slot,
                "subnet_id": placement.subnet_id,
                "vpc_id": placement.vpc_id,
            }
            for slot, placement in enumerate(slot_placements, start=1)
        ],
        "stage": stage,
        "subnet_ids": [placement.subnet_id for placement in placements],
        "user_data_sha256": hashlib.sha256(user_data.encode()).hexdigest(),
    }
    if existing_intent is not None and existing_intent != intent:
        existing_count = existing_intent.get("requested_instances")
        if existing_count != n:
            raise RuntimeError(
                f"Worker launch {launch_id!r} is bound to --count {existing_count}; "
                f"recovery requires the same --count, not {n}"
            )
        raise RuntimeError(
            f"Worker launch {launch_id!r} is bound to a different exact launch contract"
        )
    intent_sha256 = _put_immutable_json(
        s3,
        bucket=bucket,
        key=intent_key,
        value=intent,
        record_kind="worker-launch-intent",
    )

    info(f"Launching {n} {market} workers: {instance_type}, AMI={ami_id}")
    step(f"API: {api_url}")
    step(f"Image: {image_uri}")
    step(f"Security group: {sg_id}")
    step(f"Instance profile: {instance_profile_name}")
    step(
        "Subnets: "
        + ", ".join(
            f"{placement.subnet_id} ({placement.availability_zone})" for placement in placements
        )
    )
    step(f"Launch intent: s3://{bucket}/{intent_key}")

    instance_ids: list[str] = []
    for slot in range(1, n + 1):
        placement = slot_placements[slot - 1]
        client_token, request_sha256 = _worker_slot_identity(intent_sha256, slot)
        tags = {
            TAG_KEY: WORKER_TAG_VALUE,
            "Name": "citrees-worker",
            "citrees-artifact-prefix": artifact_prefix,
            "citrees-availability-zone": placement.availability_zone,
            "citrees-campaign-sha256": campaign_sha256,
            "citrees-canonical-manifest-key": canonical_manifest_s3_key,
            "citrees-canonical-manifest-sha256": canonical_manifest_sha256,
            "citrees-gate-receipt-key": gate_receipt.s3_key,
            "citrees-gate-receipt-sha256": gate_receipt.sha256,
            "citrees-image-uri": image_uri,
            "citrees-instance-family": instance_family,
            "citrees-manifest-key": manifest_s3_key,
            "citrees-manifest-sha256": manifest_sha256,
            "citrees-market": market,
            "citrees-runtime-contract-key": runtime_contract_s3_key,
            "citrees-runtime-contract-sha256": runtime_contract_sha256,
            "citrees-stage": stage,
            "citrees-subnet-id": placement.subnet_id,
            "citrees-worker-client-token": client_token,
            "citrees-worker-intent-sha256": intent_sha256,
            "citrees-worker-launch-id": launch_id,
            "citrees-worker-request-sha256": request_sha256,
            "citrees-worker-slot": str(slot),
            "citrees-vpc-id": placement.vpc_id,
        }
        existing_instance_id = _load_worker_launch_record(
            s3,
            artifact_prefix=artifact_prefix,
            availability_zone=placement.availability_zone,
            bucket=bucket,
            campaign_sha256=campaign_sha256,
            client_token=client_token,
            instance_family=instance_family,
            instance_type=instance_type,
            intent_sha256=intent_sha256,
            launch_id=launch_id,
            manifest_sha256=manifest_sha256,
            market=market,
            region=region,
            request_sha256=request_sha256,
            slot=slot,
            subnet_id=placement.subnet_id,
            vpc_id=placement.vpc_id,
        )
        if existing_instance_id is not None:
            instance_ids.append(existing_instance_id)
            step(f"  slot {slot}/{n}: {existing_instance_id} (recorded)")
            continue

        run_kwargs = dict(base_run_kwargs)
        run_kwargs["ClientToken"] = client_token
        run_kwargs["SubnetId"] = placement.subnet_id
        run_kwargs["TagSpecifications"] = [
            {
                "ResourceType": "instance",
                "Tags": [{"Key": key, "Value": value} for key, value in sorted(tags.items())],
            }
        ]
        try:
            instance_id = _launch_or_recover_worker(
                ec2,
                availability_zone=placement.availability_zone,
                instance_type=instance_type,
                recover_before_launch=existing_intent is not None,
                run_kwargs=run_kwargs,
                subnet_id=placement.subnet_id,
                tags=tags,
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in _CAPACITY_ERROR_CODES:
                raise
            warn(
                f"Stopped worker launch at slot {slot}/{n}: {code}. "
                f"Accepted {len(instance_ids)} instance(s)."
            )
            break

        record = _worker_launch_record(
            artifact_prefix=artifact_prefix,
            availability_zone=placement.availability_zone,
            campaign_sha256=campaign_sha256,
            client_token=client_token,
            instance_family=instance_family,
            instance_id=instance_id,
            instance_type=instance_type,
            intent_sha256=intent_sha256,
            launch_id=launch_id,
            manifest_sha256=manifest_sha256,
            market=market,
            region=region,
            request_sha256=request_sha256,
            slot=slot,
            subnet_id=placement.subnet_id,
            vpc_id=placement.vpc_id,
        )
        _publish_worker_launch_record(
            s3,
            artifact_prefix=artifact_prefix,
            bucket=bucket,
            record=record,
        )
        instance_ids.append(instance_id)
        step(f"  slot {slot}/{n}: {instance_id}")

    success(f"Launched {len(instance_ids)} worker instances")

    return instance_ids


def launch_mechanism_workers(
    n: int,
    instance_type: str,
    image_uri: str,
    *,
    region: str = DEFAULT_REGION,
    num_shards: int | None = None,
    shard_start: int = 0,
    subnet_ids: Sequence[str] = (),
    tasks: Sequence[str] = DEFAULT_MECHANISM_TASKS,
    source: str = "real",
    datasets: Sequence[str] = (),
    seeds: Sequence[int] = DEFAULT_MECHANISM_SEEDS,
    folds: Sequence[int] = DEFAULT_MECHANISM_FOLDS,
    model_variants: Sequence[str] = DEFAULT_MECHANISM_MODEL_VARIANTS,
    ranking_variants: Sequence[str] = DEFAULT_MECHANISM_RANKING_VARIANTS,
    n_jobs: int = -1,
    downstream_n_jobs: int = 1,
) -> list[str]:
    """Launch sharded CIF mechanism-ablation workers.

    These workers do not use the FastAPI queue. Each instance runs one stable
    modulo shard of ``paper.benchmark.experiments.cif_mechanism_ablation`` and
    writes rankings, metrics, and fit-timing artifacts to S3.
    """
    total_shards = n if num_shards is None else num_shards
    if n < 1:
        raise ValueError("n must be >= 1")
    image_uri = validate_image_digest_uri(image_uri)
    git_sha = validate_image_revision(
        image_uri,
        get_frozen_git_sha(),
        region=region,
    )
    if total_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_start < 0 or shard_start >= total_shards:
        raise ValueError("shard_start must satisfy 0 <= shard_start < num_shards")
    if shard_start + n > total_shards:
        raise ValueError("shard_start + n must be <= num_shards")

    from paper.benchmark.experiments.cif_mechanism_ablation import (
        distributed_output_uri,
        mechanism_specification_sha256,
    )

    ec2 = boto3.client("ec2", region_name=region)
    account_id = get_aws_account_id()
    bucket = get_resource_name(account_id)
    ecr_uri = image_uri.split("/")[0]
    sg_id = ensure_security_group(region)
    ami_id = get_ami(region)
    placement_subnets = list(subnet_ids) or get_default_subnet_ids(ec2, instance_type=instance_type)
    if not placement_subnets:
        raise RuntimeError(f"No default subnets offer instance type {instance_type}")

    specification_sha256 = mechanism_specification_sha256(
        tasks=tasks,
        source=source,
        datasets=datasets,
        seeds=seeds,
        folds=folds,
        model_variants=model_variants,
        ranking_variants=ranking_variants,
        n_jobs=n_jobs,
        downstream_n_jobs=downstream_n_jobs,
    )
    output_uri = distributed_output_uri(
        bucket=bucket,
        image_uri=image_uri,
        specification_sha256=specification_sha256,
    )
    output_prefix = output_uri.removeprefix(f"s3://{bucket}/")
    if output_prefix == output_uri:
        raise RuntimeError(f"mechanism output URI is outside bucket {bucket}: {output_uri}")
    instance_profile_name = ensure_campaign_iam_profile(
        output_prefix=output_prefix,
        campaign_sha256=specification_sha256,
        read_keys=(),
        write_prefixes=(output_prefix,),
        region=region,
    )

    shard_end = shard_start + n - 1
    info(f"Launching {n} spot mechanism workers: {instance_type}, AMI={ami_id}")
    step(f"Shard range: {shard_start}-{shard_end} of {total_shards}")
    step(f"Image: {image_uri}")
    step(f"Output: {output_uri}")
    step(f"Specification SHA-256: {specification_sha256}")
    step(f"Tasks: {_csv_arg(tasks)}")
    step(f"Seeds: {_csv_arg(seeds)}")
    step(f"Folds: {_csv_arg(folds)}")
    step(f"Model variants: {_csv_arg(model_variants)}")
    step(f"Ranking variants: {_csv_arg(ranking_variants)}")
    if placement_subnets:
        step(f"Subnets: {_csv_arg(placement_subnets)}")
    step(f"Security group: {sg_id}")
    step(f"Instance profile: {instance_profile_name}")

    instance_ids: list[str] = []
    for shard_index in range(shard_start, shard_start + n):
        user_data = _make_mechanism_user_data(
            region=region,
            ecr_uri=ecr_uri,
            image_uri=image_uri,
            bucket=bucket,
            git_sha=git_sha,
            shard_index=shard_index,
            num_shards=total_shards,
            tasks=tasks,
            source=source,
            datasets=datasets,
            seeds=seeds,
            folds=folds,
            model_variants=model_variants,
            ranking_variants=ranking_variants,
            n_jobs=n_jobs,
            downstream_n_jobs=downstream_n_jobs,
        )

        run_kwargs: dict[str, Any] = {
            "ImageId": ami_id,
            "InstanceType": instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "IamInstanceProfile": {"Name": instance_profile_name},
            "UserData": base64.b64encode(user_data.encode()).decode(),
            "MetadataOptions": {
                "HttpEndpoint": "enabled",
                "HttpPutResponseHopLimit": 2,
                "HttpTokens": "required",
            },
            "SecurityGroupIds": [sg_id],
            "InstanceInitiatedShutdownBehavior": "terminate",
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": TAG_KEY, "Value": MECHANISM_TAG_VALUE},
                        {
                            "Key": "Name",
                            "Value": f"citrees-mechanism-{shard_index:03d}",
                        },
                        {"Key": "citrees-shard-index", "Value": str(shard_index)},
                        {"Key": "citrees-num-shards", "Value": str(total_shards)},
                        {
                            "Key": "citrees-mechanism-spec-sha256",
                            "Value": specification_sha256,
                        },
                        {"Key": "citrees-market", "Value": "spot"},
                        {"Key": "citrees-image-uri", "Value": image_uri},
                    ],
                }
            ],
        }
        if placement_subnets:
            run_kwargs["SubnetId"] = placement_subnets[
                (shard_index - shard_start) % len(placement_subnets)
            ]

        run_kwargs["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        }

        try:
            response = ec2.run_instances(**run_kwargs)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {
                "InsufficientInstanceCapacity",
                "MaxSpotInstanceCountExceeded",
                "InstanceLimitExceeded",
                "VcpuLimitExceeded",
            }:
                warn(
                    f"Stopped launch at shard {shard_index}/{total_shards}: {code}. "
                    f"Launched {len(instance_ids)} instance(s)."
                )
                break
            raise
        instance_id = response["Instances"][0]["InstanceId"]
        instance_ids.append(instance_id)
        step(f"  shard {shard_index}/{total_shards}: {instance_id}")

    success(f"Launched {len(instance_ids)} mechanism worker instances")
    return instance_ids


def list_workers(
    launch_id: str,
    *,
    artifact_prefix: str,
    campaign_sha256: str,
    stage: str,
    region: str = DEFAULT_REGION,
) -> list[dict[str, str]]:
    """List running workers from one exact campaign launch.

    Returns a list of dicts with keys: instance_id, state, instance_type,
    launch_time, and launch_id.
    """
    launch_id = _validate_worker_launch_id(launch_id)
    artifact_prefix, stage = _validate_queue_scope(artifact_prefix, stage)
    from paper.benchmark.pipeline.manifest import validate_manifest_sha256

    campaign_sha256 = validate_manifest_sha256(campaign_sha256)
    ec2 = boto3.client("ec2", region_name=region)

    response = ec2.describe_instances(
        Filters=[
            {"Name": f"tag:{TAG_KEY}", "Values": [WORKER_TAG_VALUE]},
            {
                "Name": "tag:citrees-artifact-prefix",
                "Values": [artifact_prefix],
            },
            {
                "Name": "tag:citrees-campaign-sha256",
                "Values": [campaign_sha256],
            },
            {"Name": "tag:citrees-stage", "Values": [stage]},
            {
                "Name": "tag:citrees-worker-launch-id",
                "Values": [launch_id],
            },
            {
                "Name": "instance-state-name",
                "Values": ["pending", "running", "stopping"],
            },
        ]
    )

    workers = []
    for reservation in response.get("Reservations", []):
        for inst in reservation.get("Instances", []):
            tags = {tag["Key"]: tag["Value"] for tag in inst.get("Tags", [])}
            if (
                tags.get(TAG_KEY) != WORKER_TAG_VALUE
                or tags.get("citrees-artifact-prefix") != artifact_prefix
                or tags.get("citrees-campaign-sha256") != campaign_sha256
                or tags.get("citrees-stage") != stage
                or tags.get("citrees-worker-launch-id") != launch_id
                or tags.get("citrees-market") != _BENCHMARK_MARKET
                or inst.get("InstanceLifecycle") is not None
            ):
                raise RuntimeError(
                    "EC2 returned a worker outside the exact campaign launch identity"
                )
            workers.append(
                {
                    "instance_id": inst["InstanceId"],
                    "state": inst["State"]["Name"],
                    "instance_type": inst.get("InstanceType", ""),
                    "launch_time": (
                        inst.get("LaunchTime", "").isoformat() if inst.get("LaunchTime") else ""
                    ),
                    "launch_id": launch_id,
                }
            )

    return workers


def list_mechanism_workers(region: str = DEFAULT_REGION) -> list[dict[str, str]]:
    """List running CIF mechanism-ablation worker instances."""
    ec2 = boto3.client("ec2", region_name=region)

    response = ec2.describe_instances(
        Filters=[
            {"Name": f"tag:{TAG_KEY}", "Values": [MECHANISM_TAG_VALUE]},
            {
                "Name": "instance-state-name",
                "Values": ["pending", "running", "stopping"],
            },
        ]
    )

    workers = []
    for reservation in response.get("Reservations", []):
        for inst in reservation.get("Instances", []):
            tags = {tag["Key"]: tag["Value"] for tag in inst.get("Tags", [])}
            workers.append(
                {
                    "instance_id": inst["InstanceId"],
                    "state": inst["State"]["Name"],
                    "instance_type": inst.get("InstanceType", ""),
                    "launch_time": (
                        inst.get("LaunchTime", "").isoformat() if inst.get("LaunchTime") else ""
                    ),
                    "shard_index": tags.get("citrees-shard-index", ""),
                    "num_shards": tags.get("citrees-num-shards", ""),
                }
            )

    return workers


def terminate_workers(
    launch_id: str,
    *,
    artifact_prefix: str,
    campaign_sha256: str,
    stage: str,
    region: str = DEFAULT_REGION,
) -> list[str]:
    """Terminate running workers from one exact campaign launch.

    Returns list of terminated instance IDs.
    """
    launch_id = _validate_worker_launch_id(launch_id)
    workers = list_workers(
        launch_id,
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        stage=stage,
        region=region,
    )
    if not workers:
        info(f"No worker instances found for launch {launch_id}")
        return []

    ec2 = boto3.client("ec2", region_name=region)
    instance_ids = [w["instance_id"] for w in workers]
    info(f"Terminating {len(instance_ids)} worker instances from launch {launch_id}...")

    ec2.terminate_instances(InstanceIds=instance_ids)

    success(f"Terminated {len(instance_ids)} instances")
    for iid in instance_ids:
        step(f"  {iid}")

    return instance_ids


def terminate_mechanism_workers(region: str = DEFAULT_REGION) -> list[str]:
    """Terminate all running CIF mechanism-ablation worker instances."""
    ec2 = boto3.client("ec2", region_name=region)

    workers = list_mechanism_workers(region)
    if not workers:
        info("No mechanism worker instances found")
        return []

    instance_ids = [w["instance_id"] for w in workers]
    info(f"Terminating {len(instance_ids)} mechanism worker instances...")

    ec2.terminate_instances(InstanceIds=instance_ids)

    success(f"Terminated {len(instance_ids)} mechanism instances")
    for iid in instance_ids:
        step(f"  {iid}")

    return instance_ids


# ---------------------------------------------------------------------------
# CloudWatch logs
# ---------------------------------------------------------------------------


def get_logs(
    role: str,
    instance_id: str | None = None,
    tail: int = 100,
    *,
    region: str = DEFAULT_REGION,
) -> list[dict[str, str]]:
    """Fetch recent CloudWatch logs for an API or worker instance.

    Parameters
    ----------
    role : str
        "api", "worker", or "mechanism".
    instance_id : str, optional
        Instance ID to filter by. If None, returns logs from all streams
        in the log group.
    tail : int
        Number of recent log events to return.
    region : str
        AWS region.

    Returns
    -------
    list[dict[str, str]]
        List of {"timestamp": ..., "message": ...} dicts.
    """
    if role == "api":
        log_group = LOG_GROUP_API
    elif role == "worker":
        log_group = LOG_GROUP_WORKER
    elif role == "mechanism":
        log_group = LOG_GROUP_MECHANISM
    else:
        raise ValueError("role must be 'api', 'worker', or 'mechanism'")
    logs_client = boto3.client("logs", region_name=region)

    kwargs: dict[str, Any] = {
        "logGroupName": log_group,
        "limit": tail,
        "interleaved": True,
    }
    if instance_id:
        kwargs["logStreamNames"] = [instance_id]

    try:
        resp = logs_client.filter_log_events(**kwargs)
    except logs_client.exceptions.ResourceNotFoundException:
        return []

    return [
        {
            "timestamp": str(e.get("timestamp", "")),
            "message": e.get("message", ""),
        }
        for e in resp.get("events", [])
    ]
