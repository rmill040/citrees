"""Launch, store, and materialize immutable JSS replication shards on AWS."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import tarfile
import tempfile
import textwrap
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from functools import cached_property
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import boto3
from botocore.exceptions import ClientError

from paper.benchmark.infra.aws import (
    DEFAULT_REGION,
    ensure_campaign_iam_profile,
    ensure_s3_bucket,
    ensure_security_group,
    get_aws_account_id,
    get_frozen_git_sha,
    get_resource_name,
    validate_image_revision,
)
from paper.benchmark.infra.ec2 import (
    TAG_KEY,
    get_default_subnet_ids,
    per_boot_container_recovery_hook,
    validate_image_digest_uri,
)
from paper.jss.replication import shards

Market = Literal["spot"]
SPOT_MARKET: Market = "spot"
COMPONENTS = (*shards.CALIBRATION_COMPONENTS, "behavior")
CAMPAIGN_SCHEMA_VERSION = 2
ARCHIVE_METADATA_SCHEMA_VERSION = "1"
ROLE_TAG_VALUE = "jss-shard"
LOG_GROUP = "/citrees/jss"
COMPUTE_ACCOUNTING_FILENAME = "cloud_compute_accounting.json"
CLOUD_PROVENANCE_DIRECTORY = "cloud_provenance"
MATERIALIZED_CAMPAIGN_MANIFEST = f"{CLOUD_PROVENANCE_DIRECTORY}/campaign.json"
_AMI_ID_PATTERN = re.compile(r"^ami-[0-9a-f]{8,17}$")
_CAPACITY_ERRORS = frozenset(
    {
        "InsufficientInstanceCapacity",
        "InstanceLimitExceeded",
        "MaxSpotInstanceCountExceeded",
        "VcpuLimitExceeded",
    }
)
_CONDITIONAL_WRITE_CONFLICT_CODES = frozenset(
    {
        "409",
        "412",
        "ConditionalRequestConflict",
        "PreconditionFailed",
    }
)
LAUNCH_RECORD_POLL_SECONDS = 2.0
LAUNCH_RECORD_TIMEOUT_SECONDS = 300.0
CONDITIONAL_WRITE_MAX_ATTEMPTS = 5
CONDITIONAL_WRITE_RETRY_SECONDS = 0.1
EC2_INSTANCE_PROPAGATION_SECONDS = 300.0
EC2_EXACT_LOOKUP_MAX_ATTEMPTS = 3
EC2_EXACT_LOOKUP_RETRY_SECONDS = 1.0


@dataclass(frozen=True)
class CloudCampaign:
    """Immutable scientific and runtime identity for one distributed campaign."""

    profile: shards.Profile
    base_seed: int
    git_sha: str
    image_uri: str
    aws_account_id: str
    bucket: str
    region: str
    market: Market
    instance_type: str
    ami_id: str
    specs: tuple[shards.ShardSpec, ...]
    campaign_sha256: str

    @cached_property
    def spec_inventory(self) -> frozenset[shards.ShardSpec]:
        """Return the constant-time membership index for the shard inventory."""
        return frozenset(self.specs)

    @property
    def output_prefix(self) -> str:
        """Return the content-addressed S3 output prefix."""
        return f"jss/replication/source-{self.git_sha}/campaign-{self.campaign_sha256}"

    @property
    def manifest_key(self) -> str:
        """Return the immutable campaign manifest key."""
        return f"{self.output_prefix}/campaign.json"


@dataclass(frozen=True)
class CloudRuntime:
    """EC2 provenance attached to one completed shard."""

    aws_account_id: str
    instance_id: str
    instance_type: str
    availability_zone: str
    market: Market
    image_uri: str
    ami_id: str
    attempt: int


@dataclass(frozen=True)
class CampaignStatus:
    """Observed immutable outputs for one campaign."""

    total: int
    completed: tuple[shards.ShardSpec, ...]
    missing: tuple[shards.ShardSpec, ...]
    launch_requests: tuple[LaunchRequest, ...]
    request_objects: tuple[StoredLaunchRequest, ...]
    launch_rejections: tuple[LaunchRejection, ...]
    rejection_objects: tuple[StoredLaunchRejection, ...]
    unresolved_requests: tuple[LaunchRequest, ...]
    launches: tuple[LaunchRecord, ...]
    launch_objects: tuple[StoredLaunchRecord, ...]
    instance_outcomes: tuple[InstanceOutcome, ...]
    outcome_objects: tuple[StoredInstanceOutcome, ...]


@dataclass(frozen=True)
class LaunchRequest:
    """One immutable request authorized before EC2 resource creation."""

    spec: shards.ShardSpec
    spec_sha256: str
    request_index: int
    attempt: int
    market: Market
    instance_type: str
    subnet_id: str
    instance_profile_name: str
    security_group_id: str
    client_token: str
    run_instances_sha256: str


@dataclass(frozen=True)
class StoredLaunchRequest:
    """One validated launch request with its immutable S3 object identity."""

    object_key: str
    object_sha256: str
    object_bytes: bytes
    request: LaunchRequest


@dataclass(frozen=True)
class LaunchRejection:
    """One EC2 request rejected before an instance was created."""

    spec_sha256: str
    request_index: int
    client_token: str
    error_code: str


@dataclass(frozen=True)
class StoredLaunchRejection:
    """One validated launch rejection with its immutable S3 object identity."""

    object_key: str
    object_sha256: str
    object_bytes: bytes
    rejection: LaunchRejection


@dataclass(frozen=True)
class LaunchRecord:
    """One accepted EC2 launch request."""

    spec: shards.ShardSpec
    spec_sha256: str
    request_index: int
    attempt: int
    market: Market
    instance_type: str
    client_token: str
    instance_id: str
    availability_zone: str
    launch_time: str
    logical_cpus: int


@dataclass(frozen=True)
class StoredLaunchRecord:
    """One validated launch record with its immutable S3 object identity."""

    object_key: str
    object_sha256: str
    object_bytes: bytes
    record: LaunchRecord


@dataclass(frozen=True)
class InstanceOutcome:
    """First durable observation that one accepted instance terminated."""

    spec_sha256: str
    attempt: int
    instance_id: str
    state: Literal["terminated"]
    state_transition_reason: str
    observed_utc: str


@dataclass(frozen=True)
class StoredInstanceOutcome:
    """One validated terminal-instance outcome and its immutable S3 identity."""

    object_key: str
    object_sha256: str
    object_bytes: bytes
    outcome: InstanceOutcome


@dataclass(frozen=True)
class ObservedInstance:
    """One EC2 instance returned by campaign-scoped discovery."""

    launch: LaunchRecord
    state: str
    state_transition_reason: str


@dataclass(frozen=True)
class MaterializedShard:
    """Validated successful shard inputs used for compute accounting."""

    spec: shards.ShardSpec
    runtime: CloudRuntime
    launch: LaunchRecord
    archive_key: str
    archive_sha256: str
    source_receipt: str
    receipt_sha256: str
    receipt_created_utc: str
    analysis_elapsed_seconds: float
    assignment_count: int
    logical_cpus: int


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_git_sha(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("git_sha must be a lowercase 40-character Git revision")
    return value


def _require_integer(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _require_nonnegative_real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return normalized


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    return normalized


def _require_spot_market(value: object, label: str) -> Market:
    market = _require_string(value, label)
    if market != SPOT_MARKET:
        raise ValueError(f"{label} must be {SPOT_MARKET!r}")
    return SPOT_MARKET


def _is_conditional_write_conflict(error: ClientError) -> bool:
    code = str(error.response.get("Error", {}).get("Code", ""))
    status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in _CONDITIONAL_WRITE_CONFLICT_CODES or status in (409, 412)


def _is_missing_object_error(error: BaseException) -> bool:
    if isinstance(error, KeyError):
        return True
    if not isinstance(error, ClientError):
        return False
    code = str(error.response.get("Error", {}).get("Code", ""))
    status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"404", "NoSuchKey", "NotFound"} or status == 404


def _get_object_if_present(client: Any, *, bucket: str, key: str) -> bytes | None:
    try:
        return client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except (ClientError, KeyError) as error:
        if _is_missing_object_error(error):
            return None
        raise


def _put_immutable_bytes(
    client: Any,
    *,
    bucket: str,
    key: str,
    payload: bytes,
    content_type: str,
    mismatch_message: str,
    metadata: Mapping[str, str] | None = None,
) -> bool:
    for attempt in range(CONDITIONAL_WRITE_MAX_ATTEMPTS):
        arguments: dict[str, object] = {
            "Bucket": bucket,
            "Key": key,
            "Body": payload,
            "ContentType": content_type,
            "IfNoneMatch": "*",
        }
        if metadata is not None:
            arguments["Metadata"] = dict(metadata)
        try:
            client.put_object(**arguments)
            return True
        except ClientError as error:
            if not _is_conditional_write_conflict(error):
                raise
            existing = _get_object_if_present(client, bucket=bucket, key=key)
            if existing is not None:
                if existing != payload:
                    raise RuntimeError(mismatch_message) from error
                return False
            if attempt + 1 == CONDITIONAL_WRITE_MAX_ATTEMPTS:
                raise RuntimeError(
                    f"conditional S3 write did not converge for s3://{bucket}/{key}"
                ) from error
            time.sleep(CONDITIONAL_WRITE_RETRY_SECONDS * (2**attempt))
    raise AssertionError("unreachable conditional S3 write state")


def _parse_canonical_utc(value: object, label: str) -> datetime:
    timestamp = _require_string(value, label)
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise ValueError(f"{label} is invalid") from error
    if parsed.tzinfo is None or parsed.astimezone(UTC).isoformat() != timestamp:
        raise ValueError(f"{label} must be canonical UTC")
    return parsed


def _require_ami_id(value: object) -> str:
    ami_id = _require_string(value, "ami_id")
    if not _AMI_ID_PATTERN.fullmatch(ami_id):
        raise ValueError("ami_id must be an immutable EC2 AMI identifier")
    return ami_id


def shard_spec_sha256(spec: shards.ShardSpec) -> str:
    """Return the canonical identity of one deterministic shard."""
    shards.validate_shard_spec(spec)
    return _sha256_bytes(_canonical_json(asdict(spec)))


def build_shard_specs(
    profile: shards.Profile,
    *,
    base_seed: int,
    shard_counts: Mapping[str, int],
) -> tuple[shards.ShardSpec, ...]:
    """Build the exact canonical shard inventory for one campaign."""
    if set(shard_counts) != set(COMPONENTS):
        raise ValueError(f"shard_counts must contain exactly {list(COMPONENTS)}")
    base_seed = _require_integer(base_seed, "base_seed", minimum=0)
    normalized_counts = {
        component: _require_integer(
            shard_counts[component],
            f"{component} shard count",
            minimum=0,
        )
        for component in COMPONENTS
    }
    if not any(normalized_counts.values()):
        raise ValueError("at least one campaign component must contain shards")
    specs: list[shards.ShardSpec] = []
    for component in shards.CALIBRATION_COMPONENTS:
        count = normalized_counts[component]
        for index in range(count):
            spec = shards.ShardSpec(
                target_analysis="calibration",
                component=component,
                profile=profile,
                shard_index=index,
                num_shards=count,
                base_seed=base_seed,
            )
            shards.validate_shard_spec(spec)
            specs.append(spec)
    behavior_count = normalized_counts["behavior"]
    for index in range(behavior_count):
        spec = shards.ShardSpec(
            target_analysis="behavior",
            component="behavior",
            profile=profile,
            shard_index=index,
            num_shards=behavior_count,
            base_seed=base_seed,
        )
        shards.validate_shard_spec(spec)
        specs.append(spec)
    return tuple(specs)


def _campaign_payload(
    *,
    profile: shards.Profile,
    base_seed: int,
    git_sha: str,
    image_uri: str,
    aws_account_id: str,
    bucket: str,
    region: str,
    market: Market,
    instance_type: str,
    ami_id: str,
    specs: Sequence[shards.ShardSpec],
) -> dict[str, object]:
    return {
        "analysis": "jss_cloud_campaign",
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "profile": profile,
        "base_seed": base_seed,
        "git_sha": git_sha,
        "image_uri": image_uri,
        "aws_account_id": aws_account_id,
        "bucket": bucket,
        "region": region,
        "market": market,
        "instance_type": instance_type,
        "ami_id": ami_id,
        "specs": [asdict(spec) for spec in specs],
    }


def create_campaign(
    profile: shards.Profile,
    *,
    base_seed: int,
    git_sha: str,
    image_uri: str,
    aws_account_id: str,
    bucket: str,
    region: str,
    instance_type: str,
    ami_id: str,
    shard_counts: Mapping[str, int],
) -> CloudCampaign:
    """Create and validate one content-addressed cloud campaign."""
    if profile not in ("smoke", "quick", "full"):
        raise ValueError(f"unknown campaign profile: {profile!r}")
    base_seed = _require_integer(base_seed, "base_seed", minimum=0)
    git_sha = _require_git_sha(git_sha)
    image_uri = validate_image_digest_uri(_require_string(image_uri, "image_uri"))
    aws_account_id = _require_string(aws_account_id, "aws_account_id")
    if not aws_account_id.isdigit() or len(aws_account_id) != 12:
        raise ValueError("aws_account_id must contain 12 digits")
    bucket = _require_string(bucket, "bucket")
    if bucket != get_resource_name(aws_account_id):
        raise ValueError("bucket must be the account-bound citrees bucket")
    region = _require_string(region, "region")
    instance_type = _require_string(instance_type, "instance_type")
    ami_id = _require_ami_id(ami_id)
    specs = build_shard_specs(profile, base_seed=base_seed, shard_counts=shard_counts)
    payload = _campaign_payload(
        profile=profile,
        base_seed=base_seed,
        git_sha=git_sha,
        image_uri=image_uri,
        aws_account_id=aws_account_id,
        bucket=bucket,
        region=region,
        market=SPOT_MARKET,
        instance_type=instance_type,
        ami_id=ami_id,
        specs=specs,
    )
    campaign_sha256 = _sha256_bytes(_canonical_json(payload))
    return CloudCampaign(
        profile=profile,
        base_seed=base_seed,
        git_sha=git_sha,
        image_uri=image_uri,
        aws_account_id=aws_account_id,
        bucket=bucket,
        region=region,
        market=SPOT_MARKET,
        instance_type=instance_type,
        ami_id=ami_id,
        specs=specs,
        campaign_sha256=campaign_sha256,
    )


def campaign_payload(campaign: CloudCampaign) -> dict[str, object]:
    """Return the canonical manifest payload and revalidate its digest."""
    payload = _campaign_payload(
        profile=campaign.profile,
        base_seed=campaign.base_seed,
        git_sha=campaign.git_sha,
        image_uri=campaign.image_uri,
        aws_account_id=campaign.aws_account_id,
        bucket=campaign.bucket,
        region=campaign.region,
        market=campaign.market,
        instance_type=campaign.instance_type,
        ami_id=campaign.ami_id,
        specs=campaign.specs,
    )
    if _sha256_bytes(_canonical_json(payload)) != campaign.campaign_sha256:
        raise ValueError("campaign_sha256 differs from the canonical manifest")
    return payload


def _parse_shard_spec(value: object) -> shards.ShardSpec:
    if not isinstance(value, dict) or set(value) != set(shards.ShardSpec.__dataclass_fields__):
        raise ValueError("campaign shard specification has invalid fields")
    target_analysis = _require_string(value["target_analysis"], "target_analysis")
    component = _require_string(value["component"], "component")
    profile = _require_string(value["profile"], "profile")
    spec = shards.ShardSpec(
        target_analysis=cast(shards.TargetAnalysis, target_analysis),
        component=component,
        profile=cast(shards.Profile, profile),
        shard_index=_require_integer(value["shard_index"], "shard_index"),
        num_shards=_require_integer(value["num_shards"], "num_shards"),
        base_seed=_require_integer(value["base_seed"], "base_seed"),
    )
    shards.validate_shard_spec(spec)
    return spec


def parse_campaign(payload: bytes, expected_sha256: str) -> CloudCampaign:
    """Parse a campaign manifest only when its bytes and structure agree."""
    expected_sha256 = _require_sha256(expected_sha256, "expected campaign digest")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("campaign manifest is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("campaign manifest must be a JSON object")
    expected_fields = {
        "analysis",
        "schema_version",
        "profile",
        "base_seed",
        "git_sha",
        "image_uri",
        "aws_account_id",
        "bucket",
        "region",
        "market",
        "instance_type",
        "ami_id",
        "specs",
    }
    if set(value) != expected_fields:
        raise ValueError("campaign manifest fields differ from the required schema")
    if value["analysis"] != "jss_cloud_campaign":
        raise ValueError("campaign manifest identity or schema version is invalid")
    if _require_integer(value["schema_version"], "schema_version") != CAMPAIGN_SCHEMA_VERSION:
        raise ValueError("campaign manifest identity or schema version is invalid")
    raw_specs = value["specs"]
    if not isinstance(raw_specs, list):
        raise TypeError("campaign specs must be a JSON array")
    specs = tuple(_parse_shard_spec(item) for item in raw_specs)
    shard_counts: dict[str, int] = {component: 0 for component in COMPONENTS}
    for component in COMPONENTS:
        matching = tuple(spec for spec in specs if spec.component == component)
        if not matching:
            continue
        counts = {spec.num_shards for spec in matching}
        if len(counts) != 1:
            raise ValueError(f"campaign specs disagree on {component!r} shard count")
        shard_counts[component] = next(iter(counts))
    profile = cast(
        shards.Profile,
        _require_string(value["profile"], "profile"),
    )
    base_seed = _require_integer(value["base_seed"], "base_seed", minimum=0)
    _require_spot_market(value["market"], "campaign market")
    reconstructed = create_campaign(
        profile,
        base_seed=base_seed,
        git_sha=_require_git_sha(value["git_sha"]),
        image_uri=_require_string(value["image_uri"], "image_uri"),
        aws_account_id=_require_string(value["aws_account_id"], "aws_account_id"),
        bucket=_require_string(value["bucket"], "bucket"),
        region=_require_string(value["region"], "region"),
        instance_type=_require_string(value["instance_type"], "instance_type"),
        ami_id=_require_ami_id(value["ami_id"]),
        shard_counts=shard_counts,
    )
    if specs != reconstructed.specs:
        raise ValueError("campaign specs differ from their canonical inventory")
    if reconstructed.campaign_sha256 != expected_sha256:
        raise ValueError("campaign manifest digest differs from the requested campaign")
    return reconstructed


def archive_key(campaign: CloudCampaign, spec: shards.ShardSpec) -> str:
    """Return the immutable S3 key for one shard archive."""
    if spec not in campaign.spec_inventory:
        raise ValueError("shard specification is outside the campaign")
    return (
        f"{campaign.output_prefix}/shards/{spec.target_analysis}/{spec.component}/"
        f"{spec.shard_index:05d}-of-{spec.num_shards:05d}.tar.gz"
    )


def _manifest_bytes(campaign: CloudCampaign) -> bytes:
    return (
        json.dumps(
            campaign_payload(campaign),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def publish_campaign(
    campaign: CloudCampaign,
    *,
    s3_client: Any | None = None,
) -> None:
    """Publish one immutable campaign manifest or require exact prior bytes."""
    client = s3_client or boto3.client("s3", region_name=campaign.region)
    payload = _manifest_bytes(campaign)
    _put_immutable_bytes(
        client,
        bucket=campaign.bucket,
        key=campaign.manifest_key,
        payload=payload,
        content_type="application/json",
        metadata={"campaign-sha256": campaign.campaign_sha256},
        mismatch_message="existing campaign manifest bytes differ",
    )


def load_campaign(
    *,
    bucket: str,
    manifest_key: str,
    campaign_sha256: str,
    region: str,
    s3_client: Any | None = None,
) -> CloudCampaign:
    """Load and validate one immutable campaign manifest from S3."""
    client = s3_client or boto3.client("s3", region_name=region)
    payload = client.get_object(Bucket=bucket, Key=manifest_key)["Body"].read()
    campaign = parse_campaign(payload, campaign_sha256)
    if campaign.bucket != bucket or campaign.manifest_key != manifest_key:
        raise ValueError("campaign manifest location differs from its content")
    return campaign


def _cloud_receipt(
    output_dir: Path,
    campaign: CloudCampaign,
    spec: shards.ShardSpec,
    runtime: CloudRuntime,
    key: str,
) -> None:
    _validate_runtime(campaign, runtime)
    receipt_path = output_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    if not isinstance(receipt, dict) or receipt.get("analysis") != "jss_shard":
        raise ValueError("cloud shard output has an invalid local receipt")
    if "cloud_execution" in receipt:
        raise ValueError("cloud shard receipt already contains execution metadata")
    receipt["cloud_execution"] = {
        "schema_version": 1,
        "campaign_sha256": campaign.campaign_sha256,
        "archive_key": key,
        "spec_sha256": shard_spec_sha256(spec),
        **asdict(runtime),
    }
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def create_archive(output_dir: Path, archive_path: Path) -> str:
    """Create one archive containing only regular shard output files."""
    output_dir = output_dir.resolve()
    if not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    paths = sorted(output_dir.rglob("*"))
    unsafe = [path for path in paths if path.is_symlink()]
    if unsafe:
        raise ValueError(f"shard output contains symbolic links: {unsafe}")
    files = [path for path in paths if path.is_file()]
    if not files or output_dir / "receipt.json" not in files:
        raise ValueError("shard output must contain receipt.json")
    with tarfile.open(archive_path, mode="w:gz") as archive:
        for path in files:
            archive.add(
                path,
                arcname=path.relative_to(output_dir).as_posix(),
                recursive=False,
            )
    return _sha256_file(archive_path)


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    """Safely extract one shard archive into a new directory."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("shard archive is empty")
        names: set[str] = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != member.name
                or not member.isfile()
                or member.issym()
                or member.islnk()
            ):
                raise ValueError(f"unsafe shard archive member: {member.name!r}")
            if member.name in names:
                raise ValueError(f"duplicate shard archive member: {member.name!r}")
            names.add(member.name)
        output_dir.mkdir()
        archive.extractall(output_dir, members=members, filter="data")


def _archive_metadata(
    campaign: CloudCampaign,
    spec: shards.ShardSpec,
    archive_sha256: str,
) -> dict[str, str]:
    return {
        "schema-version": ARCHIVE_METADATA_SCHEMA_VERSION,
        "archive-sha256": _require_sha256(archive_sha256, "archive digest"),
        "campaign-sha256": campaign.campaign_sha256,
        "spec-sha256": shard_spec_sha256(spec),
        "git-sha": campaign.git_sha,
    }


def _validate_archive_head(
    head: Mapping[str, object],
    *,
    campaign: CloudCampaign,
    spec: shards.ShardSpec,
) -> str:
    metadata = head.get("Metadata")
    if not isinstance(metadata, dict):
        raise ValueError("shard archive has no S3 metadata")
    expected = {
        "schema-version": ARCHIVE_METADATA_SCHEMA_VERSION,
        "campaign-sha256": campaign.campaign_sha256,
        "spec-sha256": shard_spec_sha256(spec),
        "git-sha": campaign.git_sha,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"shard archive metadata differs for {key}")
    content_length = head.get("ContentLength")
    if (
        isinstance(content_length, bool)
        or not isinstance(content_length, int)
        or content_length < 1
    ):
        raise ValueError("shard archive byte count is invalid")
    return _require_sha256(metadata.get("archive-sha256"), "archive metadata digest")


def upload_archive(
    archive_path: Path,
    *,
    campaign: CloudCampaign,
    spec: shards.ShardSpec,
    s3_client: Any | None = None,
) -> bool:
    """Upload one immutable shard archive and independently verify the accepted object."""
    client = s3_client or boto3.client("s3", region_name=campaign.region)
    key = archive_key(campaign, spec)
    archive_sha256 = _sha256_file(archive_path)
    metadata = _archive_metadata(campaign, spec, archive_sha256)
    for attempt in range(CONDITIONAL_WRITE_MAX_ATTEMPTS):
        try:
            with archive_path.open("rb") as stream:
                client.put_object(
                    Bucket=campaign.bucket,
                    Key=key,
                    Body=stream,
                    ContentType="application/gzip",
                    IfNoneMatch="*",
                    Metadata=metadata,
                )
            break
        except ClientError as error:
            if not _is_conditional_write_conflict(error):
                raise
            try:
                _validate_stored_archive(
                    campaign,
                    spec,
                    s3_client=client,
                )
            except (ClientError, KeyError) as lookup_error:
                if not _is_missing_object_error(lookup_error):
                    raise
                if attempt + 1 == CONDITIONAL_WRITE_MAX_ATTEMPTS:
                    raise RuntimeError(
                        f"conditional S3 write did not converge for s3://{campaign.bucket}/{key}"
                    ) from error
                time.sleep(CONDITIONAL_WRITE_RETRY_SECONDS * (2**attempt))
                continue
            return False
    else:
        raise AssertionError("unreachable conditional archive write state")
    accepted_sha256, _ = _validate_stored_archive(
        campaign,
        spec,
        s3_client=client,
    )
    if accepted_sha256 != archive_sha256:
        raise RuntimeError("uploaded shard archive differs from the immutable S3 object")
    return True


def _validate_runtime(campaign: CloudCampaign, runtime: CloudRuntime) -> None:
    if runtime.aws_account_id != campaign.aws_account_id:
        raise ValueError("worker account differs from the campaign")
    if runtime.image_uri != campaign.image_uri:
        raise ValueError("worker image differs from the campaign")
    if runtime.ami_id != campaign.ami_id:
        raise ValueError("worker AMI differs from the campaign")
    if _require_spot_market(runtime.market, "worker market") != campaign.market:
        raise ValueError("worker market differs from the campaign")
    _require_string(runtime.instance_id, "worker instance_id")
    if _require_string(runtime.instance_type, "worker instance_type") != campaign.instance_type:
        raise ValueError("worker instance type differs from the campaign")
    _require_string(runtime.availability_zone, "worker availability_zone")
    _require_integer(runtime.attempt, "worker attempt", minimum=1)


def _validate_cloud_receipt(
    output_dir: Path,
    campaign: CloudCampaign,
    spec: shards.ShardSpec,
) -> CloudRuntime:
    receipt = json.loads((output_dir / "receipt.json").read_text(encoding="ascii"))
    if not isinstance(receipt, dict) or receipt.get("analysis") != "jss_shard":
        raise ValueError("cloud shard output has an invalid local receipt")
    if receipt.get("git_sha") != campaign.git_sha:
        raise ValueError("cloud shard source revision differs from the campaign")
    if receipt.get("git_dirty") is not False:
        raise ValueError("cloud shard source must be clean")
    execution = receipt.get("cloud_execution")
    if not isinstance(execution, dict):
        raise TypeError("cloud_execution must be a JSON object")
    expected_fields = {
        "schema_version",
        "campaign_sha256",
        "archive_key",
        "spec_sha256",
        *CloudRuntime.__dataclass_fields__,
    }
    if set(execution) != expected_fields:
        raise ValueError("cloud_execution fields differ from the required schema")
    if _require_integer(execution["schema_version"], "cloud schema_version") != 1:
        raise ValueError("cloud_execution schema version is invalid")
    if execution["campaign_sha256"] != campaign.campaign_sha256:
        raise ValueError("cloud receipt campaign differs")
    if execution["archive_key"] != archive_key(campaign, spec):
        raise ValueError("cloud receipt archive key differs")
    if execution["spec_sha256"] != shard_spec_sha256(spec):
        raise ValueError("cloud receipt shard specification differs")
    runtime = CloudRuntime(
        aws_account_id=_require_string(
            execution["aws_account_id"],
            "cloud aws_account_id",
        ),
        instance_id=_require_string(execution["instance_id"], "cloud instance_id"),
        instance_type=_require_string(
            execution["instance_type"],
            "cloud instance_type",
        ),
        availability_zone=_require_string(
            execution["availability_zone"],
            "cloud availability_zone",
        ),
        market=_require_spot_market(execution["market"], "cloud market"),
        image_uri=_require_string(execution["image_uri"], "cloud image_uri"),
        ami_id=_require_ami_id(execution["ami_id"]),
        attempt=_require_integer(execution["attempt"], "cloud attempt", minimum=1),
    )
    _validate_runtime(campaign, runtime)
    return runtime


def _download_stored_archive(
    campaign: CloudCampaign,
    spec: shards.ShardSpec,
    archive_path: Path,
    *,
    s3_client: Any,
) -> str:
    key = archive_key(campaign, spec)
    head = s3_client.head_object(Bucket=campaign.bucket, Key=key)
    expected_sha256 = _validate_archive_head(
        head,
        campaign=campaign,
        spec=spec,
    )
    with archive_path.open("wb") as stream:
        body = s3_client.get_object(Bucket=campaign.bucket, Key=key)["Body"]
        shutil.copyfileobj(body, stream)
    if archive_path.stat().st_size != int(head["ContentLength"]):
        raise RuntimeError(f"downloaded archive byte count differs for {key}")
    if _sha256_file(archive_path) != expected_sha256:
        raise RuntimeError(f"downloaded archive digest differs for {key}")
    return expected_sha256


def _validate_stored_archive(
    campaign: CloudCampaign,
    spec: shards.ShardSpec,
    *,
    s3_client: Any,
) -> tuple[str, CloudRuntime]:
    with tempfile.TemporaryDirectory(prefix="citrees-jss-archive-validation-") as temporary:
        root = Path(temporary)
        archive_path = root / "shard.tar.gz"
        archive_sha256 = _download_stored_archive(
            campaign,
            spec,
            archive_path,
            s3_client=s3_client,
        )
        output_dir = root / "shard"
        extract_archive(archive_path, output_dir)
        shards.verify_shard_directory(
            output_dir,
            expected_spec=spec,
            expected_git_sha=campaign.git_sha,
            allow_cloud_execution=True,
            require_cloud_execution=True,
        )
        runtime = _validate_cloud_receipt(output_dir, campaign, spec)
    return archive_sha256, runtime


def _wait_for_matching_launch(
    campaign: CloudCampaign,
    spec: shards.ShardSpec,
    runtime: CloudRuntime,
    *,
    s3_client: Any,
) -> LaunchRecord:
    deadline = time.monotonic() + LAUNCH_RECORD_TIMEOUT_SECONDS
    spec_sha256 = shard_spec_sha256(spec)
    while True:
        launches = list_launch_records(campaign, s3_client=s3_client)
        matching_attempts = [
            launch
            for launch in launches
            if launch.spec_sha256 == spec_sha256 and launch.attempt == runtime.attempt
        ]
        if matching_attempts:
            return _require_matching_launch(launches, spec, runtime)
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out waiting for the immutable EC2 launch record "
                f"for shard {spec_sha256} attempt {runtime.attempt}"
            )
        time.sleep(LAUNCH_RECORD_POLL_SECONDS)


def run_cloud_shard(
    campaign: CloudCampaign,
    spec: shards.ShardSpec,
    runtime: CloudRuntime,
    *,
    s3_client: Any | None = None,
) -> bool:
    """Execute, archive, and conditionally publish one campaign shard."""
    if spec not in campaign.specs:
        raise ValueError("worker shard is outside the campaign")
    _validate_runtime(campaign, runtime)
    client = s3_client or boto3.client("s3", region_name=campaign.region)
    _wait_for_matching_launch(
        campaign,
        spec,
        runtime,
        s3_client=client,
    )
    key = archive_key(campaign, spec)
    with tempfile.TemporaryDirectory(prefix="citrees-jss-cloud-") as temporary:
        root = Path(temporary)
        output_dir = root / "shard"
        written = shards.run_shard_to_directory(spec, output_dir)
        if written.resolve() != output_dir.resolve():
            raise RuntimeError("shard runner returned an unexpected output directory")
        shards.verify_shard_directory(
            output_dir,
            expected_spec=spec,
            expected_git_sha=campaign.git_sha,
        )
        _cloud_receipt(output_dir, campaign, spec, runtime, key)
        shards.verify_shard_directory(
            output_dir,
            expected_spec=spec,
            expected_git_sha=campaign.git_sha,
            allow_cloud_execution=True,
            require_cloud_execution=True,
        )
        _validate_cloud_receipt(output_dir, campaign, spec)
        archive_path = root / "shard.tar.gz"
        create_archive(output_dir, archive_path)
        return upload_archive(
            archive_path,
            campaign=campaign,
            spec=spec,
            s3_client=client,
        )


def _list_archive_keys(campaign: CloudCampaign, s3_client: Any) -> set[str]:
    prefix = f"{campaign.output_prefix}/shards/"
    keys: set[str] = set()
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=campaign.bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item.get("Key")
            if isinstance(key, str):
                keys.add(key)
    return keys


def _launch_request_key(campaign: CloudCampaign, request: LaunchRequest) -> str:
    return (
        f"{campaign.output_prefix}/launch-requests/{request.spec_sha256}/"
        f"{request.request_index:03d}.json"
    )


def publish_launch_request(
    campaign: CloudCampaign,
    request: LaunchRequest,
    *,
    s3_client: Any | None = None,
) -> None:
    """Publish one immutable EC2 request before resource creation."""
    key = _launch_request_key(campaign, request)
    payload = (
        json.dumps(
            {
                "analysis": "jss_cloud_launch_request",
                "schema_version": 1,
                "campaign_sha256": campaign.campaign_sha256,
                **asdict(request),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    _parse_launch_request(campaign, key, payload)
    client = s3_client or boto3.client("s3", region_name=campaign.region)
    _put_immutable_bytes(
        client,
        bucket=campaign.bucket,
        key=key,
        payload=payload,
        content_type="application/json",
        mismatch_message="existing JSS launch request bytes differ",
    )


def _parse_launch_request(
    campaign: CloudCampaign,
    key: str,
    payload: bytes,
) -> LaunchRequest:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"JSS launch request is not valid JSON: {key}") from error
    if not isinstance(value, dict):
        raise TypeError(f"JSS launch request must be a JSON object: {key}")
    expected_fields = {
        "analysis",
        "schema_version",
        "campaign_sha256",
        *LaunchRequest.__dataclass_fields__,
    }
    if set(value) != expected_fields:
        raise ValueError(f"JSS launch request fields differ: {key}")
    if value["analysis"] != "jss_cloud_launch_request":
        raise ValueError(f"JSS launch request identity differs: {key}")
    if _require_integer(value["schema_version"], "launch request schema_version") != 1:
        raise ValueError(f"JSS launch request schema differs: {key}")
    if value["campaign_sha256"] != campaign.campaign_sha256:
        raise ValueError(f"JSS launch request campaign differs: {key}")
    spec = _parse_shard_spec(value["spec"])
    if spec not in campaign.specs:
        raise ValueError(f"JSS launch request shard is outside the campaign: {key}")
    spec_sha256 = _require_sha256(
        value["spec_sha256"],
        "launch request spec_sha256",
    )
    if spec_sha256 != shard_spec_sha256(spec):
        raise ValueError(f"JSS launch request shard digest differs: {key}")
    request_index = _require_integer(
        value["request_index"],
        "launch request index",
        minimum=1,
    )
    attempt = _require_integer(value["attempt"], "launch request attempt", minimum=1)
    market = _require_spot_market(value["market"], "launch request market")
    if market != campaign.market:
        raise ValueError(f"JSS launch request market differs: {key}")
    instance_type = _require_string(
        value["instance_type"],
        "launch request instance type",
    )
    if instance_type != campaign.instance_type:
        raise ValueError(f"JSS launch request instance type differs: {key}")
    subnet_id = _require_string(value["subnet_id"], "launch request subnet")
    instance_profile_name = _require_string(
        value["instance_profile_name"],
        "launch request instance profile",
    )
    security_group_id = _require_string(
        value["security_group_id"],
        "launch request security group",
    )
    client_token = _require_string(value["client_token"], "launch request client token")
    if len(client_token) > 64:
        raise ValueError(f"JSS launch request client token is too long: {key}")
    expected_client_token = _client_token(
        campaign,
        spec,
        request_index=request_index,
        attempt=attempt,
        instance_type=instance_type,
        subnet_id=subnet_id,
        instance_profile_name=instance_profile_name,
        security_group_id=security_group_id,
    )
    if client_token != expected_client_token:
        raise ValueError(f"JSS launch request client token differs: {key}")
    expected_key = (
        f"{campaign.output_prefix}/launch-requests/{spec_sha256}/{request_index:03d}.json"
    )
    if key != expected_key:
        raise ValueError(f"JSS launch request key differs from its content: {key}")
    request = LaunchRequest(
        spec=spec,
        spec_sha256=spec_sha256,
        request_index=request_index,
        attempt=attempt,
        market=market,
        instance_type=instance_type,
        subnet_id=subnet_id,
        instance_profile_name=instance_profile_name,
        security_group_id=security_group_id,
        client_token=client_token,
        run_instances_sha256=_require_sha256(
            value["run_instances_sha256"],
            "launch request RunInstances digest",
        ),
    )
    if request.run_instances_sha256 != _run_instances_sha256(campaign, request):
        raise ValueError(f"JSS launch request RunInstances digest differs: {key}")
    return request


def _load_launch_requests(
    campaign: CloudCampaign,
    *,
    s3_client: Any | None = None,
) -> tuple[StoredLaunchRequest, ...]:
    client = s3_client or boto3.client("s3", region_name=campaign.region)
    prefix = f"{campaign.output_prefix}/launch-requests/"
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=campaign.bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item.get("Key")
            if not isinstance(key, str):
                raise TypeError("JSS launch-request listing contains an invalid key")
            keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError("JSS launch-request listing contains duplicate keys")
    stored_requests: list[StoredLaunchRequest] = []
    for key in sorted(keys):
        payload = client.get_object(Bucket=campaign.bucket, Key=key)["Body"].read()
        stored_requests.append(
            StoredLaunchRequest(
                object_key=key,
                object_sha256=_sha256_bytes(payload),
                object_bytes=payload,
                request=_parse_launch_request(campaign, key, payload),
            )
        )
    grouped: dict[str, list[LaunchRequest]] = {}
    for stored in stored_requests:
        grouped.setdefault(stored.request.spec_sha256, []).append(stored.request)
    for spec_sha256, requests in grouped.items():
        request_indices = sorted(request.request_index for request in requests)
        if request_indices != list(range(1, len(requests) + 1)):
            raise ValueError(
                f"JSS launch request indices are not contiguous for shard {spec_sha256}"
            )
    return tuple(stored_requests)


def _launch_rejection_key(
    campaign: CloudCampaign,
    rejection: LaunchRejection,
) -> str:
    return (
        f"{campaign.output_prefix}/launch-rejections/{rejection.spec_sha256}/"
        f"{rejection.request_index:03d}-{rejection.error_code}.json"
    )


def publish_launch_rejection(
    campaign: CloudCampaign,
    rejection: LaunchRejection,
    *,
    requests: Sequence[LaunchRequest],
    s3_client: Any | None = None,
) -> None:
    """Publish one immutable known-capacity rejection for a launch request."""
    key = _launch_rejection_key(campaign, rejection)
    payload = (
        json.dumps(
            {
                "analysis": "jss_cloud_launch_rejection",
                "schema_version": 1,
                "campaign_sha256": campaign.campaign_sha256,
                **asdict(rejection),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    _parse_launch_rejection(campaign, key, payload, requests=requests)
    client = s3_client or boto3.client("s3", region_name=campaign.region)
    _put_immutable_bytes(
        client,
        bucket=campaign.bucket,
        key=key,
        payload=payload,
        content_type="application/json",
        mismatch_message="existing JSS launch rejection bytes differ",
    )


def _parse_launch_rejection(
    campaign: CloudCampaign,
    key: str,
    payload: bytes,
    *,
    requests: Sequence[LaunchRequest],
) -> LaunchRejection:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"JSS launch rejection is not valid JSON: {key}") from error
    if not isinstance(value, dict):
        raise TypeError(f"JSS launch rejection must be a JSON object: {key}")
    expected_fields = {
        "analysis",
        "schema_version",
        "campaign_sha256",
        *LaunchRejection.__dataclass_fields__,
    }
    if set(value) != expected_fields:
        raise ValueError(f"JSS launch rejection fields differ: {key}")
    if value["analysis"] != "jss_cloud_launch_rejection":
        raise ValueError(f"JSS launch rejection identity differs: {key}")
    if _require_integer(value["schema_version"], "launch rejection schema_version") != 1:
        raise ValueError(f"JSS launch rejection schema differs: {key}")
    if value["campaign_sha256"] != campaign.campaign_sha256:
        raise ValueError(f"JSS launch rejection campaign differs: {key}")
    spec_sha256 = _require_sha256(
        value["spec_sha256"],
        "launch rejection spec_sha256",
    )
    request_index = _require_integer(
        value["request_index"],
        "launch rejection request_index",
        minimum=1,
    )
    client_token = _require_string(
        value["client_token"],
        "launch rejection client_token",
    )
    error_code = _require_string(value["error_code"], "launch rejection error_code")
    if error_code not in _CAPACITY_ERRORS:
        raise ValueError(f"JSS launch rejection error code is unsupported: {key}")
    matches = [
        request
        for request in requests
        if request.spec_sha256 == spec_sha256 and request.request_index == request_index
    ]
    if len(matches) != 1:
        raise ValueError(f"JSS launch rejection has no unique request: {key}")
    request = matches[0]
    if request.client_token != client_token:
        raise ValueError(f"JSS launch rejection differs from its request: {key}")
    expected_key = (
        f"{campaign.output_prefix}/launch-rejections/{spec_sha256}/"
        f"{request_index:03d}-{error_code}.json"
    )
    if key != expected_key:
        raise ValueError(f"JSS launch rejection key differs from its content: {key}")
    return LaunchRejection(
        spec_sha256=spec_sha256,
        request_index=request_index,
        client_token=client_token,
        error_code=error_code,
    )


def _load_launch_rejections(
    campaign: CloudCampaign,
    *,
    requests: Sequence[LaunchRequest],
    s3_client: Any | None = None,
) -> tuple[StoredLaunchRejection, ...]:
    client = s3_client or boto3.client("s3", region_name=campaign.region)
    prefix = f"{campaign.output_prefix}/launch-rejections/"
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=campaign.bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item.get("Key")
            if not isinstance(key, str):
                raise TypeError("JSS launch-rejection listing contains an invalid key")
            keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError("JSS launch-rejection listing contains duplicate keys")
    stored_rejections: list[StoredLaunchRejection] = []
    for key in sorted(keys):
        payload = client.get_object(Bucket=campaign.bucket, Key=key)["Body"].read()
        stored_rejections.append(
            StoredLaunchRejection(
                object_key=key,
                object_sha256=_sha256_bytes(payload),
                object_bytes=payload,
                rejection=_parse_launch_rejection(
                    campaign,
                    key,
                    payload,
                    requests=requests,
                ),
            )
        )
    return tuple(stored_rejections)


def _launch_record_key(campaign: CloudCampaign, record: LaunchRecord) -> str:
    return f"{campaign.output_prefix}/launches/{record.spec_sha256}/{record.attempt:03d}.json"


def publish_launch_record(
    campaign: CloudCampaign,
    record: LaunchRecord,
    *,
    s3_client: Any | None = None,
) -> None:
    """Publish one immutable EC2 launch record or require exact prior bytes."""
    key = _launch_record_key(campaign, record)
    payload = (
        json.dumps(
            {
                "analysis": "jss_cloud_launch",
                "schema_version": 1,
                "campaign_sha256": campaign.campaign_sha256,
                **asdict(record),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    _parse_launch_record(campaign, key, payload)
    client = s3_client or boto3.client("s3", region_name=campaign.region)
    _put_immutable_bytes(
        client,
        bucket=campaign.bucket,
        key=key,
        payload=payload,
        content_type="application/json",
        mismatch_message="existing JSS launch record bytes differ",
    )


def _parse_launch_record(
    campaign: CloudCampaign,
    key: str,
    payload: bytes,
) -> LaunchRecord:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"JSS launch record is not valid JSON: {key}") from error
    if not isinstance(value, dict):
        raise TypeError(f"JSS launch record must be a JSON object: {key}")
    expected_fields = {
        "analysis",
        "schema_version",
        "campaign_sha256",
        *LaunchRecord.__dataclass_fields__,
    }
    if set(value) != expected_fields:
        raise ValueError(f"JSS launch record fields differ: {key}")
    if value["analysis"] != "jss_cloud_launch":
        raise ValueError(f"JSS launch record identity differs: {key}")
    if _require_integer(value["schema_version"], "launch schema_version") != 1:
        raise ValueError(f"JSS launch record schema differs: {key}")
    if value["campaign_sha256"] != campaign.campaign_sha256:
        raise ValueError(f"JSS launch record campaign differs: {key}")
    spec = _parse_shard_spec(value["spec"])
    if spec not in campaign.specs:
        raise ValueError(f"JSS launch record shard is outside the campaign: {key}")
    spec_sha256 = _require_sha256(value["spec_sha256"], "launch spec_sha256")
    if spec_sha256 != shard_spec_sha256(spec):
        raise ValueError(f"JSS launch record shard digest differs: {key}")
    request_index = _require_integer(
        value["request_index"],
        "launch request index",
        minimum=1,
    )
    attempt = _require_integer(value["attempt"], "launch attempt", minimum=1)
    market = _require_spot_market(value["market"], "launch market")
    if market != campaign.market:
        raise ValueError(f"JSS launch record market differs: {key}")
    instance_type = _require_string(value["instance_type"], "launch instance_type")
    if instance_type != campaign.instance_type:
        raise ValueError(f"JSS launch record instance type differs: {key}")
    client_token = _require_string(value["client_token"], "launch client token")
    instance_id = _require_string(value["instance_id"], "launch instance_id")
    availability_zone = _require_string(
        value["availability_zone"],
        "launch availability_zone",
    )
    launch_time = _require_string(value["launch_time"], "launch_time")
    try:
        parsed_launch_time = datetime.fromisoformat(launch_time)
    except ValueError as error:
        raise ValueError(f"JSS launch record time is invalid: {key}") from error
    if (
        parsed_launch_time.tzinfo is None
        or parsed_launch_time.astimezone(UTC).isoformat() != launch_time
    ):
        raise ValueError(f"JSS launch record time is not canonical UTC: {key}")
    expected_key = f"{campaign.output_prefix}/launches/{spec_sha256}/{attempt:03d}.json"
    if key != expected_key:
        raise ValueError(f"JSS launch record key differs from its content: {key}")
    logical_cpus = _require_integer(
        value["logical_cpus"],
        "launch logical_cpus",
        minimum=1,
    )
    return LaunchRecord(
        spec=spec,
        spec_sha256=spec_sha256,
        request_index=request_index,
        attempt=attempt,
        market=market,
        instance_type=instance_type,
        client_token=client_token,
        instance_id=instance_id,
        availability_zone=availability_zone,
        launch_time=launch_time,
        logical_cpus=logical_cpus,
    )


def _validate_launch_history(
    stored_records: Sequence[StoredLaunchRecord],
) -> tuple[StoredLaunchRecord, ...]:
    object_keys = [stored.object_key for stored in stored_records]
    if object_keys != sorted(object_keys) or len(object_keys) != len(set(object_keys)):
        raise ValueError("JSS launch records are not in canonical object-key order")
    grouped: dict[str, list[LaunchRecord]] = {}
    records = tuple(stored.record for stored in stored_records)
    for record in records:
        _require_spot_market(record.market, "launch record market")
        grouped.setdefault(record.spec_sha256, []).append(record)
    for spec_sha256, spec_records in grouped.items():
        attempts = [record.attempt for record in spec_records]
        if sorted(attempts) != list(range(1, len(attempts) + 1)):
            raise ValueError(f"JSS launch attempts are not contiguous for shard {spec_sha256}")
        attempt_order = sorted(spec_records, key=lambda record: record.attempt)
        launch_times = [
            _parse_canonical_utc(record.launch_time, "launch_time") for record in attempt_order
        ]
        if any(
            later <= earlier for earlier, later in zip(launch_times, launch_times[1:], strict=False)
        ):
            raise ValueError(
                f"JSS launch attempt times are not strictly increasing for shard {spec_sha256}"
            )
    instance_ids = [record.instance_id for record in records]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("JSS launch records reuse an EC2 instance identifier")
    return tuple(stored_records)


def _load_launch_records(
    campaign: CloudCampaign,
    *,
    s3_client: Any | None = None,
) -> tuple[StoredLaunchRecord, ...]:
    """Load and validate exact immutable EC2 launch-record objects."""
    client = s3_client or boto3.client("s3", region_name=campaign.region)
    prefix = f"{campaign.output_prefix}/launches/"
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=campaign.bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item.get("Key")
            if not isinstance(key, str):
                raise TypeError("JSS launch listing contains an invalid key")
            keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError("JSS launch listing contains duplicate keys")
    stored_records: list[StoredLaunchRecord] = []
    for key in sorted(keys):
        payload = client.get_object(Bucket=campaign.bucket, Key=key)["Body"].read()
        stored_records.append(
            StoredLaunchRecord(
                object_key=key,
                object_sha256=_sha256_bytes(payload),
                object_bytes=payload,
                record=_parse_launch_record(campaign, key, payload),
            )
        )
    return _validate_launch_history(stored_records)


def list_launch_records(
    campaign: CloudCampaign,
    *,
    s3_client: Any | None = None,
) -> tuple[LaunchRecord, ...]:
    """Return the validated immutable EC2 launch history."""
    _, _, launch_objects, _ = _load_launch_ledger(campaign, s3_client=s3_client)
    return tuple(stored.record for stored in launch_objects)


def _instance_outcome_key(
    campaign: CloudCampaign,
    outcome: InstanceOutcome,
) -> str:
    return (
        f"{campaign.output_prefix}/instance-outcomes/{outcome.spec_sha256}/"
        f"{outcome.attempt:03d}.json"
    )


def _parse_instance_outcome(
    campaign: CloudCampaign,
    key: str,
    payload: bytes,
    *,
    launches: Sequence[LaunchRecord],
) -> InstanceOutcome:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"JSS instance outcome is not valid JSON: {key}") from error
    if not isinstance(value, dict):
        raise TypeError(f"JSS instance outcome must be a JSON object: {key}")
    expected_fields = {
        "analysis",
        "schema_version",
        "campaign_sha256",
        *InstanceOutcome.__dataclass_fields__,
    }
    if set(value) != expected_fields:
        raise ValueError(f"JSS instance outcome fields differ: {key}")
    if value["analysis"] != "jss_cloud_instance_outcome":
        raise ValueError(f"JSS instance outcome identity differs: {key}")
    if _require_integer(value["schema_version"], "instance outcome schema_version") != 1:
        raise ValueError(f"JSS instance outcome schema differs: {key}")
    if value["campaign_sha256"] != campaign.campaign_sha256:
        raise ValueError(f"JSS instance outcome campaign differs: {key}")
    spec_sha256 = _require_sha256(
        value["spec_sha256"],
        "instance outcome spec_sha256",
    )
    attempt = _require_integer(
        value["attempt"],
        "instance outcome attempt",
        minimum=1,
    )
    instance_id = _require_string(
        value["instance_id"],
        "instance outcome instance_id",
    )
    if value["state"] != "terminated":
        raise ValueError(f"JSS instance outcome state differs: {key}")
    state_transition_reason = value["state_transition_reason"]
    if not isinstance(state_transition_reason, str):
        raise TypeError(f"JSS instance outcome transition reason must be a string: {key}")
    observed_utc = _parse_canonical_utc(
        value["observed_utc"],
        "instance outcome observation time",
    ).isoformat()
    matches = [
        launch
        for launch in launches
        if launch.spec_sha256 == spec_sha256 and launch.attempt == attempt
    ]
    if len(matches) != 1 or matches[0].instance_id != instance_id:
        raise ValueError(f"JSS instance outcome has no unique matching launch: {key}")
    if _parse_canonical_utc(
        observed_utc,
        "instance outcome observation time",
    ) < _parse_canonical_utc(matches[0].launch_time, "launch_time"):
        raise ValueError(f"JSS instance outcome predates its launch: {key}")
    expected_key = f"{campaign.output_prefix}/instance-outcomes/{spec_sha256}/{attempt:03d}.json"
    if key != expected_key:
        raise ValueError(f"JSS instance outcome key differs from its content: {key}")
    return InstanceOutcome(
        spec_sha256=spec_sha256,
        attempt=attempt,
        instance_id=instance_id,
        state="terminated",
        state_transition_reason=state_transition_reason,
        observed_utc=observed_utc,
    )


def publish_instance_outcome(
    campaign: CloudCampaign,
    outcome: InstanceOutcome,
    *,
    launches: Sequence[LaunchRecord],
    s3_client: Any | None = None,
) -> InstanceOutcome:
    """Publish the first terminal observation and converge later observers."""
    key = _instance_outcome_key(campaign, outcome)
    payload = (
        json.dumps(
            {
                "analysis": "jss_cloud_instance_outcome",
                "schema_version": 1,
                "campaign_sha256": campaign.campaign_sha256,
                **asdict(outcome),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    parsed = _parse_instance_outcome(
        campaign,
        key,
        payload,
        launches=launches,
    )
    client = s3_client or boto3.client("s3", region_name=campaign.region)
    for attempt in range(CONDITIONAL_WRITE_MAX_ATTEMPTS):
        try:
            client.put_object(
                Bucket=campaign.bucket,
                Key=key,
                Body=payload,
                ContentType="application/json",
                IfNoneMatch="*",
            )
            return parsed
        except ClientError as error:
            if not _is_conditional_write_conflict(error):
                raise
            existing = _get_object_if_present(
                client,
                bucket=campaign.bucket,
                key=key,
            )
            if existing is not None:
                prior = _parse_instance_outcome(
                    campaign,
                    key,
                    existing,
                    launches=launches,
                )
                if (
                    prior.spec_sha256 != parsed.spec_sha256
                    or prior.attempt != parsed.attempt
                    or prior.instance_id != parsed.instance_id
                    or prior.state != parsed.state
                ):
                    raise RuntimeError(
                        "existing JSS instance outcome identifies a different instance"
                    ) from error
                return prior
            if attempt + 1 == CONDITIONAL_WRITE_MAX_ATTEMPTS:
                raise RuntimeError(
                    f"conditional S3 write did not converge for s3://{campaign.bucket}/{key}"
                ) from error
            time.sleep(CONDITIONAL_WRITE_RETRY_SECONDS * (2**attempt))
    raise AssertionError("unreachable conditional S3 write state")


def _load_instance_outcomes(
    campaign: CloudCampaign,
    *,
    launches: Sequence[LaunchRecord],
    s3_client: Any | None = None,
) -> tuple[StoredInstanceOutcome, ...]:
    client = s3_client or boto3.client("s3", region_name=campaign.region)
    prefix = f"{campaign.output_prefix}/instance-outcomes/"
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=campaign.bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item.get("Key")
            if not isinstance(key, str):
                raise TypeError("JSS instance-outcome listing contains an invalid key")
            keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError("JSS instance-outcome listing contains duplicate keys")
    stored_outcomes: list[StoredInstanceOutcome] = []
    for key in sorted(keys):
        payload = client.get_object(Bucket=campaign.bucket, Key=key)["Body"].read()
        stored_outcomes.append(
            StoredInstanceOutcome(
                object_key=key,
                object_sha256=_sha256_bytes(payload),
                object_bytes=payload,
                outcome=_parse_instance_outcome(
                    campaign,
                    key,
                    payload,
                    launches=launches,
                ),
            )
        )
    identities = [
        (stored.outcome.spec_sha256, stored.outcome.attempt) for stored in stored_outcomes
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("JSS instance-outcome ledger contains duplicate attempts")
    return tuple(stored_outcomes)


def _validate_launch_ledger(
    request_objects: Sequence[StoredLaunchRequest],
    rejection_objects: Sequence[StoredLaunchRejection],
    launch_objects: Sequence[StoredLaunchRecord],
) -> tuple[LaunchRequest, ...]:
    request_keys = [stored.object_key for stored in request_objects]
    if request_keys != sorted(request_keys) or len(request_keys) != len(set(request_keys)):
        raise ValueError("JSS launch requests are not in canonical object-key order")
    rejection_keys = [stored.object_key for stored in rejection_objects]
    if rejection_keys != sorted(rejection_keys) or len(rejection_keys) != len(set(rejection_keys)):
        raise ValueError("JSS launch rejections are not in canonical object-key order")
    requests = tuple(stored.request for stored in request_objects)
    rejections = tuple(stored.rejection for stored in rejection_objects)
    launches = tuple(stored.record for stored in launch_objects)
    for request in requests:
        _require_spot_market(request.market, "launch request market")
    for launch in launches:
        _require_spot_market(launch.market, "launch record market")
    request_by_identity = {
        (request.spec_sha256, request.request_index): request for request in requests
    }
    if len(request_by_identity) != len(requests):
        raise ValueError("JSS launch ledger contains duplicate requests")
    rejection_by_identity = {
        (rejection.spec_sha256, rejection.request_index, rejection.error_code): rejection
        for rejection in rejections
    }
    if len(rejection_by_identity) != len(rejections):
        raise ValueError("JSS launch ledger contains duplicate rejections")
    rejections_by_request: dict[tuple[str, int], list[LaunchRejection]] = {}
    for rejection in rejections:
        rejections_by_request.setdefault(
            (rejection.spec_sha256, rejection.request_index),
            [],
        ).append(rejection)
    launch_by_identity = {(launch.spec_sha256, launch.request_index): launch for launch in launches}
    if len(launch_by_identity) != len(launches):
        raise ValueError("JSS launch ledger contains duplicate request outcomes")

    for rejection in rejections:
        matched_request = request_by_identity.get((rejection.spec_sha256, rejection.request_index))
        if matched_request is None:
            raise ValueError("JSS launch rejection has no matching request")
        if (
            rejection.client_token != matched_request.client_token
            or rejection.error_code not in _CAPACITY_ERRORS
        ):
            raise ValueError("JSS launch rejection differs from its request")

    for identity, launch in launch_by_identity.items():
        matched_request = request_by_identity.get(identity)
        if matched_request is None:
            raise ValueError("JSS launch record has no matching request")
        if (
            launch.spec != matched_request.spec
            or launch.attempt != matched_request.attempt
            or launch.market != matched_request.market
            or launch.instance_type != matched_request.instance_type
            or launch.client_token != matched_request.client_token
        ):
            raise ValueError("JSS launch record differs from its request")

    grouped: dict[str, list[LaunchRequest]] = {}
    for request in requests:
        grouped.setdefault(request.spec_sha256, []).append(request)
    unresolved: list[LaunchRequest] = []
    for spec_sha256, spec_requests in grouped.items():
        ordered = sorted(spec_requests, key=lambda request: request.request_index)
        expected_attempt = 1
        for position, request in enumerate(ordered):
            if request.attempt != expected_attempt:
                raise ValueError(
                    f"JSS launch request attempt differs from realized history for "
                    f"shard {spec_sha256}"
                )
            identity = (spec_sha256, request.request_index)
            if identity in launch_by_identity:
                expected_attempt += 1
            elif identity not in rejections_by_request:
                if position + 1 != len(ordered):
                    raise ValueError(
                        f"unresolved JSS launch request is not terminal for shard {spec_sha256}"
                    )
                unresolved.append(request)
    return tuple(unresolved)


def _load_launch_ledger(
    campaign: CloudCampaign,
    *,
    s3_client: Any | None = None,
) -> tuple[
    tuple[StoredLaunchRequest, ...],
    tuple[StoredLaunchRejection, ...],
    tuple[StoredLaunchRecord, ...],
    tuple[LaunchRequest, ...],
]:
    client = s3_client or boto3.client("s3", region_name=campaign.region)
    request_objects = _load_launch_requests(campaign, s3_client=client)
    requests = tuple(stored.request for stored in request_objects)
    rejection_objects = _load_launch_rejections(
        campaign,
        requests=requests,
        s3_client=client,
    )
    launch_objects = _load_launch_records(campaign, s3_client=client)
    unresolved = _validate_launch_ledger(
        request_objects,
        rejection_objects,
        launch_objects,
    )
    return request_objects, rejection_objects, launch_objects, unresolved


def _next_launch_attempt(
    launches: Sequence[LaunchRecord],
    spec: shards.ShardSpec,
) -> int:
    spec_sha256 = shard_spec_sha256(spec)
    return (
        max(
            (record.attempt for record in launches if record.spec_sha256 == spec_sha256),
            default=0,
        )
        + 1
    )


def _next_request_index(
    requests: Sequence[LaunchRequest],
    spec: shards.ShardSpec,
) -> int:
    spec_sha256 = shard_spec_sha256(spec)
    return (
        max(
            (request.request_index for request in requests if request.spec_sha256 == spec_sha256),
            default=0,
        )
        + 1
    )


def _require_matching_launch(
    launches: Sequence[LaunchRecord],
    spec: shards.ShardSpec,
    runtime: CloudRuntime,
) -> LaunchRecord:
    matches = [
        record
        for record in launches
        if record.spec_sha256 == shard_spec_sha256(spec) and record.attempt == runtime.attempt
    ]
    if len(matches) != 1:
        raise ValueError("completed shard has no unique matching launch record")
    launch = matches[0]
    if (
        launch.instance_id != runtime.instance_id
        or launch.instance_type != runtime.instance_type
        or launch.availability_zone != runtime.availability_zone
        or launch.market != runtime.market
    ):
        raise ValueError("completed shard runtime differs from its launch record")
    return launch


def campaign_status(
    campaign: CloudCampaign,
    *,
    s3_client: Any | None = None,
) -> CampaignStatus:
    """Return exact completed and missing shard specifications."""
    client = s3_client or boto3.client("s3", region_name=campaign.region)
    observed = _list_archive_keys(campaign, client)
    expected = {archive_key(campaign, spec): spec for spec in campaign.specs}
    extras = sorted(observed.difference(expected))
    if extras:
        raise ValueError(f"campaign shard prefix contains unexpected objects: {extras}")
    completed = tuple(spec for spec in campaign.specs if archive_key(campaign, spec) in observed)
    request_objects, rejection_objects, launch_objects, unresolved_requests = _load_launch_ledger(
        campaign, s3_client=client
    )
    launch_requests = tuple(stored.request for stored in request_objects)
    launch_rejections = tuple(stored.rejection for stored in rejection_objects)
    launches = tuple(stored.record for stored in launch_objects)
    outcome_objects = _load_instance_outcomes(
        campaign,
        launches=launches,
        s3_client=client,
    )
    instance_outcomes = tuple(stored.outcome for stored in outcome_objects)
    for spec in completed:
        _, runtime = _validate_stored_archive(
            campaign,
            spec,
            s3_client=client,
        )
        _require_matching_launch(launches, spec, runtime)
    missing = tuple(spec for spec in campaign.specs if archive_key(campaign, spec) not in observed)
    return CampaignStatus(
        total=len(campaign.specs),
        completed=completed,
        missing=missing,
        launch_requests=launch_requests,
        request_objects=request_objects,
        launch_rejections=launch_rejections,
        rejection_objects=rejection_objects,
        unresolved_requests=unresolved_requests,
        launches=launches,
        launch_objects=launch_objects,
        instance_outcomes=instance_outcomes,
        outcome_objects=outcome_objects,
    )


def _materialized_path(root: Path, spec: shards.ShardSpec) -> Path:
    if spec.target_analysis == "calibration":
        return root / "calibration" / spec.component / f"{spec.shard_index:05d}"
    return root / "behavior" / f"{spec.shard_index:05d}"


def _require_verified_campaign_inventory(
    campaign: CloudCampaign,
    verified_groups: Sequence[Sequence[shards.VerifiedShard]],
) -> None:
    observed = tuple(shard.spec for group in verified_groups for shard in group)
    if (
        len(observed) != len(campaign.specs)
        or len(set(observed)) != len(observed)
        or set(observed) != set(campaign.specs)
    ):
        raise ValueError("verified shard specification inventory differs from the campaign")


def _materialized_shard(
    root: Path,
    shard_dir: Path,
    *,
    campaign: CloudCampaign,
    spec: shards.ShardSpec,
    runtime: CloudRuntime,
    launch: LaunchRecord,
    archive_sha256: str,
) -> MaterializedShard:
    receipt_path = shard_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    if not isinstance(receipt, dict):
        raise TypeError("materialized shard receipt must be a JSON object")
    hardware = receipt.get("hardware")
    if not isinstance(hardware, dict):
        raise TypeError("materialized shard hardware must be a JSON object")
    logical_cpus = _require_integer(
        hardware.get("logical_cpus"),
        "materialized shard logical_cpus",
        minimum=1,
    )
    if logical_cpus != launch.logical_cpus:
        raise ValueError("materialized shard logical CPU count differs from its launch")
    created_utc = _parse_canonical_utc(
        receipt.get("created_utc"),
        "materialized shard created_utc",
    )
    launch_time = _parse_canonical_utc(
        launch.launch_time,
        "materialized shard launch_time",
    )
    if created_utc < launch_time:
        raise ValueError("materialized shard receipt predates its EC2 launch")
    analysis_elapsed_seconds = _require_nonnegative_real(
        receipt.get("elapsed_seconds"),
        "materialized shard elapsed_seconds",
    )
    if analysis_elapsed_seconds == 0.0:
        raise ValueError("materialized shard elapsed_seconds must be positive")
    return MaterializedShard(
        spec=spec,
        runtime=runtime,
        launch=launch,
        archive_key=archive_key(campaign, spec),
        archive_sha256=_require_sha256(archive_sha256, "materialized archive digest"),
        source_receipt=receipt_path.relative_to(root).as_posix(),
        receipt_sha256=_sha256_file(receipt_path),
        receipt_created_utc=created_utc.isoformat(),
        analysis_elapsed_seconds=analysis_elapsed_seconds,
        assignment_count=_require_integer(
            receipt.get("assignment_count"),
            "materialized shard assignment_count",
            minimum=1,
        ),
        logical_cpus=logical_cpus,
    )


def _launch_to_receipt_seconds(shard: MaterializedShard) -> float:
    launch_time = _parse_canonical_utc(shard.launch.launch_time, "launch_time")
    receipt_time = _parse_canonical_utc(
        shard.receipt_created_utc,
        "receipt_created_utc",
    )
    return (receipt_time - launch_time).total_seconds()


def _launch_to_terminal_observation_seconds(
    launch: LaunchRecord,
    outcome: InstanceOutcome,
) -> float:
    if (
        launch.spec_sha256 != outcome.spec_sha256
        or launch.attempt != outcome.attempt
        or launch.instance_id != outcome.instance_id
    ):
        raise ValueError("instance outcome differs from its launch")
    elapsed = (
        _parse_canonical_utc(outcome.observed_utc, "instance outcome observed_utc")
        - _parse_canonical_utc(launch.launch_time, "launch_time")
    ).total_seconds()
    if elapsed < 0.0:
        raise ValueError("instance outcome predates its launch")
    return elapsed


def _materialized_launch_record_path(stored: StoredLaunchRecord) -> str:
    return (
        f"{CLOUD_PROVENANCE_DIRECTORY}/launches/"
        f"{stored.record.spec_sha256}/{stored.record.attempt:03d}.json"
    )


def _materialized_instance_outcome_path(stored: StoredInstanceOutcome) -> str:
    return (
        f"{CLOUD_PROVENANCE_DIRECTORY}/instance-outcomes/"
        f"{stored.outcome.spec_sha256}/{stored.outcome.attempt:03d}.json"
    )


def _materialized_launch_request_path(stored: StoredLaunchRequest) -> str:
    return (
        f"{CLOUD_PROVENANCE_DIRECTORY}/launch-requests/"
        f"{stored.request.spec_sha256}/{stored.request.request_index:03d}.json"
    )


def _materialized_launch_rejection_path(stored: StoredLaunchRejection) -> str:
    return (
        f"{CLOUD_PROVENANCE_DIRECTORY}/launch-rejections/"
        f"{stored.rejection.spec_sha256}/{stored.rejection.request_index:03d}-"
        f"{stored.rejection.error_code}.json"
    )


def _compute_totals(
    completed: Sequence[MaterializedShard],
    *,
    launches: Sequence[LaunchRecord],
    prior_attempts: Sequence[LaunchRecord],
    outcomes: Sequence[InstanceOutcome],
) -> dict[str, int | float]:
    outcome_by_attempt = {(outcome.spec_sha256, outcome.attempt): outcome for outcome in outcomes}
    if len(outcome_by_attempt) != len(outcomes) or set(outcome_by_attempt) != {
        (launch.spec_sha256, launch.attempt) for launch in launches
    }:
        raise ValueError("terminal instance outcomes differ from launch attempts")
    analysis_elapsed_seconds = math.fsum(shard.analysis_elapsed_seconds for shard in completed)
    successful_launch_to_receipt_seconds = math.fsum(
        _launch_to_receipt_seconds(shard) for shard in completed
    )
    analysis_logical_cpu_capacity_seconds = math.fsum(
        shard.analysis_elapsed_seconds * shard.logical_cpus for shard in completed
    )
    launch_to_receipt_logical_cpu_capacity_seconds = math.fsum(
        _launch_to_receipt_seconds(shard) * shard.logical_cpus for shard in completed
    )
    observed_instance_seconds = math.fsum(
        _launch_to_terminal_observation_seconds(
            launch,
            outcome_by_attempt[(launch.spec_sha256, launch.attempt)],
        )
        for launch in launches
    )
    observed_logical_cpu_capacity_seconds = math.fsum(
        _launch_to_terminal_observation_seconds(
            launch,
            outcome_by_attempt[(launch.spec_sha256, launch.attempt)],
        )
        * launch.logical_cpus
        for launch in launches
    )
    return {
        "launch_attempts": len(launches),
        "terminal_instance_attempts": len(outcomes),
        "accepted_archive_attempts": len(completed),
        "prior_attempts_without_accepted_archive": len(prior_attempts),
        "observed_instance_seconds": observed_instance_seconds,
        "observed_logical_cpu_capacity_seconds": observed_logical_cpu_capacity_seconds,
        "successful_analysis_elapsed_seconds": analysis_elapsed_seconds,
        "successful_analysis_logical_cpu_capacity_seconds": (analysis_logical_cpu_capacity_seconds),
        "successful_launch_to_receipt_seconds": (successful_launch_to_receipt_seconds),
        "successful_launch_to_receipt_logical_cpu_capacity_seconds": (
            launch_to_receipt_logical_cpu_capacity_seconds
        ),
    }


def compute_accounting_payload(
    campaign: CloudCampaign,
    *,
    campaign_manifest_sha256: str,
    campaign_manifest_bytes: int,
    request_objects: Sequence[StoredLaunchRequest],
    rejection_objects: Sequence[StoredLaunchRejection],
    launch_objects: Sequence[StoredLaunchRecord],
    outcome_objects: Sequence[StoredInstanceOutcome],
    completed: Sequence[MaterializedShard],
) -> dict[str, object]:
    """Aggregate exact launch provenance and measured successful-shard timing."""
    if tuple(shard.spec for shard in completed) != campaign.specs:
        raise ValueError("completed shard inventory differs from the campaign")
    if not launch_objects:
        raise ValueError("completed cloud campaign has no launch records")
    unresolved = _validate_launch_ledger(
        request_objects,
        rejection_objects,
        launch_objects,
    )
    if unresolved:
        raise ValueError("completed cloud campaign contains unresolved launch requests")
    requests = tuple(stored.request for stored in request_objects)
    rejections = tuple(stored.rejection for stored in rejection_objects)
    launches = tuple(stored.record for stored in launch_objects)
    outcomes = tuple(stored.outcome for stored in outcome_objects)
    launch_identities = {
        (launch.spec_sha256, launch.attempt, launch.instance_id) for launch in launches
    }
    outcome_identities = {
        (outcome.spec_sha256, outcome.attempt, outcome.instance_id) for outcome in outcomes
    }
    if len(outcomes) != len(launches) or outcome_identities != launch_identities:
        raise ValueError("completed cloud campaign lacks exact terminal instance outcomes")
    completed_by_spec = {shard_spec_sha256(shard.spec): shard for shard in completed}
    if len(completed_by_spec) != len(completed):
        raise ValueError("completed shard inventory contains duplicates")

    prior_attempts: list[LaunchRecord] = []
    for launch in launches:
        accepted = completed_by_spec[launch.spec_sha256].launch
        if launch.attempt == accepted.attempt:
            if launch != accepted:
                raise ValueError("accepted launch record differs from the shard receipt")
            continue
        if launch.attempt > accepted.attempt:
            raise ValueError("launch history continues after an accepted shard completion")
        prior_attempts.append(launch)

    logical_cpu_counts = {shard.logical_cpus for shard in completed}
    if len(logical_cpu_counts) != 1:
        raise ValueError("completed shards disagree on logical CPU count")
    logical_cpus = next(iter(logical_cpu_counts))

    first_launch = min(
        _parse_canonical_utc(record.launch_time, "launch_time") for record in launches
    )
    last_receipt = max(
        _parse_canonical_utc(shard.receipt_created_utc, "receipt_created_utc")
        for shard in completed
    )
    last_terminal_observation = max(
        _parse_canonical_utc(outcome.observed_utc, "instance outcome observed_utc")
        for outcome in outcomes
    )
    campaign_wall_seconds = (last_terminal_observation - first_launch).total_seconds()
    if campaign_wall_seconds <= 0.0:
        raise ValueError("campaign launch-to-receipt duration must be positive")

    spot_totals = _compute_totals(
        completed,
        launches=launches,
        prior_attempts=prior_attempts,
        outcomes=outcomes,
    )
    spot_totals.update(
        {
            "launch_intents": len(requests),
            "distinct_capacity_rejections": len(rejections),
        }
    )

    totals = _compute_totals(
        completed,
        launches=launches,
        prior_attempts=prior_attempts,
        outcomes=outcomes,
    )
    totals.update(
        {
            "launch_intents": len(requests),
            "distinct_capacity_rejections": len(rejections),
            "shards_succeeding_after_prior_attempts": sum(
                shard.runtime.attempt > 1 for shard in completed
            ),
            "logical_cpus_per_instance": logical_cpus,
            "campaign_wall_seconds": campaign_wall_seconds,
        }
    )
    availability_zones = {
        availability_zone: sum(launch.availability_zone == availability_zone for launch in launches)
        for availability_zone in sorted({launch.availability_zone for launch in launches})
    }
    component_summaries: dict[str, dict[str, int | float]] = {}
    for component in COMPONENTS:
        component_completed = tuple(
            shard for shard in completed if shard.spec.component == component
        )
        component_launches = tuple(
            launch for launch in launches if launch.spec.component == component
        )
        component_prior_attempts = tuple(
            launch for launch in prior_attempts if launch.spec.component == component
        )
        component_launch_identities = {
            (launch.spec_sha256, launch.attempt) for launch in component_launches
        }
        component_outcomes = tuple(
            outcome
            for outcome in outcomes
            if (outcome.spec_sha256, outcome.attempt) in component_launch_identities
        )
        component_first_launch = min(
            _parse_canonical_utc(launch.launch_time, "launch_time") for launch in component_launches
        )
        component_last_terminal_observation = max(
            _parse_canonical_utc(
                outcome.observed_utc,
                "instance outcome observed_utc",
            )
            for outcome in component_outcomes
        )
        component_wall_seconds = (
            component_last_terminal_observation - component_first_launch
        ).total_seconds()
        if component_wall_seconds <= 0.0:
            raise ValueError(f"{component} launch-to-receipt duration must be positive")
        component_summary = _compute_totals(
            component_completed,
            launches=component_launches,
            prior_attempts=component_prior_attempts,
            outcomes=component_outcomes,
        )
        assignments = sum(shard.assignment_count for shard in component_completed)
        component_summary.update(
            {
                "launch_intents": sum(request.spec.component == component for request in requests),
                "distinct_capacity_rejections": sum(
                    request.spec.component == component
                    for rejection in rejections
                    for request in requests
                    if (
                        request.spec_sha256 == rejection.spec_sha256
                        and request.request_index == rejection.request_index
                    )
                ),
                "assignments": assignments,
                "component_wall_seconds": component_wall_seconds,
                "assignments_per_component_wall_hour": (
                    assignments * 3600.0 / component_wall_seconds
                ),
            }
        )
        component_summaries[component] = component_summary

    stored_by_attempt = {
        (stored.record.spec_sha256, stored.record.attempt): stored for stored in launch_objects
    }
    return {
        "analysis": "jss_cloud_compute_accounting",
        "schema_version": 1,
        "campaign_sha256": campaign.campaign_sha256,
        "campaign_manifest": {
            "object_key": campaign.manifest_key,
            "materialized_path": MATERIALIZED_CAMPAIGN_MANIFEST,
            "object_sha256": _require_sha256(
                campaign_manifest_sha256,
                "campaign manifest object digest",
            ),
            "bytes": _require_integer(
                campaign_manifest_bytes,
                "campaign manifest byte count",
                minimum=1,
            ),
        },
        "profile": campaign.profile,
        "base_seed": campaign.base_seed,
        "git_sha": campaign.git_sha,
        "image_uri": campaign.image_uri,
        "aws_account_id": campaign.aws_account_id,
        "bucket": campaign.bucket,
        "region": campaign.region,
        "market": campaign.market,
        "instance_type": campaign.instance_type,
        "ami_id": campaign.ami_id,
        "campaign_window_utc": {
            "first_launch": first_launch.isoformat(),
            "last_receipt": last_receipt.isoformat(),
            "last_terminal_observation": last_terminal_observation.isoformat(),
        },
        "totals": totals,
        "markets": {SPOT_MARKET: spot_totals},
        "launch_attempts_by_availability_zone": availability_zones,
        "components": component_summaries,
        "launch_requests": [
            {
                "object_key": stored.object_key,
                "materialized_path": _materialized_launch_request_path(stored),
                "object_sha256": stored.object_sha256,
                **asdict(stored.request),
            }
            for stored in request_objects
        ],
        "launch_rejections": [
            {
                "object_key": stored.object_key,
                "materialized_path": _materialized_launch_rejection_path(stored),
                "object_sha256": stored.object_sha256,
                **asdict(stored.rejection),
            }
            for stored in rejection_objects
        ],
        "launch_records": [
            {
                "object_key": stored.object_key,
                "materialized_path": _materialized_launch_record_path(stored),
                "object_sha256": stored.object_sha256,
                **asdict(stored.record),
            }
            for stored in launch_objects
        ],
        "instance_outcomes": [
            {
                "object_key": stored.object_key,
                "materialized_path": _materialized_instance_outcome_path(stored),
                "object_sha256": stored.object_sha256,
                **asdict(stored.outcome),
            }
            for stored in outcome_objects
        ],
        "completed_shards": [
            {
                "spec": asdict(shard.spec),
                "spec_sha256": shard_spec_sha256(shard.spec),
                "attempt": shard.runtime.attempt,
                "market": shard.runtime.market,
                "instance_id": shard.runtime.instance_id,
                "availability_zone": shard.runtime.availability_zone,
                "archive_key": shard.archive_key,
                "archive_sha256": shard.archive_sha256,
                "source_receipt": shard.source_receipt,
                "receipt_sha256": shard.receipt_sha256,
                "accepted_launch_object_key": stored_by_attempt[
                    (shard.launch.spec_sha256, shard.launch.attempt)
                ].object_key,
                "accepted_launch_object_sha256": stored_by_attempt[
                    (shard.launch.spec_sha256, shard.launch.attempt)
                ].object_sha256,
                "launch_time": shard.launch.launch_time,
                "receipt_created_utc": shard.receipt_created_utc,
                "assignment_count": shard.assignment_count,
                "analysis_elapsed_seconds": shard.analysis_elapsed_seconds,
                "logical_cpus": shard.logical_cpus,
                "analysis_logical_cpu_capacity_seconds": (
                    shard.analysis_elapsed_seconds * shard.logical_cpus
                ),
                "successful_launch_to_receipt_seconds": (_launch_to_receipt_seconds(shard)),
                "successful_launch_to_receipt_logical_cpu_capacity_seconds": (
                    _launch_to_receipt_seconds(shard) * shard.logical_cpus
                ),
            }
            for shard in completed
        ],
    }


def _validated_campaign_manifest(
    campaign: CloudCampaign,
    s3_client: Any,
) -> bytes:
    payload = s3_client.get_object(
        Bucket=campaign.bucket,
        Key=campaign.manifest_key,
    )["Body"].read()
    expected = _manifest_bytes(campaign)
    if payload != expected:
        raise RuntimeError("stored campaign manifest bytes differ from the campaign")
    parse_campaign(payload, campaign.campaign_sha256)
    return payload


def _safe_materialized_file(root: Path, value: object, label: str) -> Path:
    relative = PurePosixPath(_require_string(value, label))
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != str(value):
        raise ValueError(f"{label} is not a canonical relative path")
    path = (root / relative.as_posix()).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{label} does not identify a materialized file")
    return path


def _require_materialized_file_inventory(
    root: Path,
    campaign: CloudCampaign,
    request_objects: Sequence[StoredLaunchRequest],
    rejection_objects: Sequence[StoredLaunchRejection],
    launch_objects: Sequence[StoredLaunchRecord],
    outcome_objects: Sequence[StoredInstanceOutcome],
) -> None:
    expected_files = {
        COMPUTE_ACCOUNTING_FILENAME,
        MATERIALIZED_CAMPAIGN_MANIFEST,
        *(_materialized_launch_request_path(stored) for stored in request_objects),
        *(_materialized_launch_rejection_path(stored) for stored in rejection_objects),
        *(_materialized_launch_record_path(stored) for stored in launch_objects),
        *(_materialized_instance_outcome_path(stored) for stored in outcome_objects),
    }
    for spec in campaign.specs:
        shard_dir = _materialized_path(root, spec)
        expected_files.update(
            path.relative_to(root).as_posix() for path in shard_dir.rglob("*") if path.is_file()
        )
    expected_directories = {
        parent.as_posix()
        for filename in expected_files
        for parent in PurePosixPath(filename).parents
        if parent.as_posix() != "."
    }
    paths = tuple(root.rglob("*"))
    symbolic_links = sorted(
        path.relative_to(root).as_posix() for path in paths if path.is_symlink()
    )
    if symbolic_links:
        raise ValueError(f"materialized cloud output contains symbolic links: {symbolic_links}")
    observed_files = {path.relative_to(root).as_posix() for path in paths if path.is_file()}
    observed_directories = {path.relative_to(root).as_posix() for path in paths if path.is_dir()}
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("materialized cloud file inventory differs from the campaign")


def validate_compute_accounting(output_dir: Path) -> dict[str, object]:
    """Recompute and validate a materialized cloud accounting artifact."""
    root = output_dir.resolve()
    accounting_path = root / COMPUTE_ACCOUNTING_FILENAME
    if not accounting_path.is_file():
        raise FileNotFoundError(accounting_path)
    payload = json.loads(accounting_path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise TypeError("cloud compute accounting must be a JSON object")
    campaign_sha256 = _require_sha256(
        payload.get("campaign_sha256"),
        "accounting campaign digest",
    )

    manifest_metadata = payload.get("campaign_manifest")
    if not isinstance(manifest_metadata, dict):
        raise TypeError("accounting campaign_manifest must be a JSON object")
    manifest_path = _safe_materialized_file(
        root,
        manifest_metadata.get("materialized_path"),
        "campaign manifest materialized_path",
    )
    manifest_bytes = manifest_path.read_bytes()
    campaign = parse_campaign(manifest_bytes, campaign_sha256)
    expected_manifest_metadata = {
        "object_key": campaign.manifest_key,
        "materialized_path": MATERIALIZED_CAMPAIGN_MANIFEST,
        "object_sha256": _sha256_bytes(manifest_bytes),
        "bytes": len(manifest_bytes),
    }
    if manifest_metadata != expected_manifest_metadata:
        raise ValueError("accounting campaign manifest metadata differs")

    raw_requests = payload.get("launch_requests")
    if not isinstance(raw_requests, list):
        raise TypeError("accounting launch_requests must be a JSON array")
    request_objects: list[StoredLaunchRequest] = []
    request_fields = set(LaunchRequest.__dataclass_fields__)
    expected_request_fields = {
        "object_key",
        "materialized_path",
        "object_sha256",
        *request_fields,
    }
    for raw_request in raw_requests:
        if not isinstance(raw_request, dict) or set(raw_request) != expected_request_fields:
            raise ValueError("accounting launch request fields differ")
        request_path = _safe_materialized_file(
            root,
            raw_request["materialized_path"],
            "launch request materialized_path",
        )
        request_bytes = request_path.read_bytes()
        object_key = _require_string(
            raw_request["object_key"],
            "launch request object_key",
        )
        request = _parse_launch_request(
            campaign,
            object_key,
            request_bytes,
        )
        if {field: raw_request[field] for field in request_fields} != asdict(request):
            raise ValueError("accounting launch request content differs")
        stored_request = StoredLaunchRequest(
            object_key=object_key,
            object_sha256=_require_sha256(
                raw_request["object_sha256"],
                "launch request object digest",
            ),
            object_bytes=request_bytes,
            request=request,
        )
        if stored_request.object_sha256 != _sha256_bytes(request_bytes):
            raise ValueError("accounting launch request digest differs")
        if raw_request["materialized_path"] != _materialized_launch_request_path(stored_request):
            raise ValueError("accounting launch request materialized path differs")
        request_objects.append(stored_request)
    requests = tuple(stored.request for stored in request_objects)

    raw_rejections = payload.get("launch_rejections")
    if not isinstance(raw_rejections, list):
        raise TypeError("accounting launch_rejections must be a JSON array")
    rejection_objects: list[StoredLaunchRejection] = []
    rejection_fields = set(LaunchRejection.__dataclass_fields__)
    expected_rejection_fields = {
        "object_key",
        "materialized_path",
        "object_sha256",
        *rejection_fields,
    }
    for raw_rejection in raw_rejections:
        if not isinstance(raw_rejection, dict) or set(raw_rejection) != expected_rejection_fields:
            raise ValueError("accounting launch rejection fields differ")
        rejection_path = _safe_materialized_file(
            root,
            raw_rejection["materialized_path"],
            "launch rejection materialized_path",
        )
        rejection_bytes = rejection_path.read_bytes()
        object_key = _require_string(
            raw_rejection["object_key"],
            "launch rejection object_key",
        )
        rejection = _parse_launch_rejection(
            campaign,
            object_key,
            rejection_bytes,
            requests=requests,
        )
        if {field: raw_rejection[field] for field in rejection_fields} != asdict(rejection):
            raise ValueError("accounting launch rejection content differs")
        stored_rejection = StoredLaunchRejection(
            object_key=object_key,
            object_sha256=_require_sha256(
                raw_rejection["object_sha256"],
                "launch rejection object digest",
            ),
            object_bytes=rejection_bytes,
            rejection=rejection,
        )
        if stored_rejection.object_sha256 != _sha256_bytes(rejection_bytes):
            raise ValueError("accounting launch rejection digest differs")
        if raw_rejection["materialized_path"] != _materialized_launch_rejection_path(
            stored_rejection
        ):
            raise ValueError("accounting launch rejection materialized path differs")
        rejection_objects.append(stored_rejection)

    raw_launches = payload.get("launch_records")
    if not isinstance(raw_launches, list):
        raise TypeError("accounting launch_records must be a JSON array")
    launch_objects: list[StoredLaunchRecord] = []
    launch_fields = set(LaunchRecord.__dataclass_fields__)
    expected_launch_fields = {
        "object_key",
        "materialized_path",
        "object_sha256",
        *launch_fields,
    }
    for raw_launch in raw_launches:
        if not isinstance(raw_launch, dict) or set(raw_launch) != expected_launch_fields:
            raise ValueError("accounting launch record fields differ")
        launch_path = _safe_materialized_file(
            root,
            raw_launch["materialized_path"],
            "launch record materialized_path",
        )
        launch_bytes = launch_path.read_bytes()
        object_key = _require_string(
            raw_launch["object_key"],
            "launch record object_key",
        )
        record = _parse_launch_record(
            campaign,
            object_key,
            launch_bytes,
        )
        if {field: raw_launch[field] for field in launch_fields} != asdict(record):
            raise ValueError("accounting launch record content differs")
        stored = StoredLaunchRecord(
            object_key=object_key,
            object_sha256=_require_sha256(
                raw_launch["object_sha256"],
                "launch record object digest",
            ),
            object_bytes=launch_bytes,
            record=record,
        )
        if stored.object_sha256 != _sha256_bytes(launch_bytes):
            raise ValueError("accounting launch record digest differs")
        if raw_launch["materialized_path"] != _materialized_launch_record_path(stored):
            raise ValueError("accounting launch record materialized path differs")
        launch_objects.append(stored)
    launch_objects_tuple = _validate_launch_history(launch_objects)
    request_objects_tuple = tuple(request_objects)
    rejection_objects_tuple = tuple(rejection_objects)
    unresolved = _validate_launch_ledger(
        request_objects_tuple,
        rejection_objects_tuple,
        launch_objects_tuple,
    )
    if unresolved:
        raise ValueError("materialized campaign contains unresolved launch requests")
    launches = tuple(stored.record for stored in launch_objects_tuple)

    raw_outcomes = payload.get("instance_outcomes")
    if not isinstance(raw_outcomes, list):
        raise TypeError("accounting instance_outcomes must be a JSON array")
    outcome_objects: list[StoredInstanceOutcome] = []
    outcome_fields = set(InstanceOutcome.__dataclass_fields__)
    expected_outcome_fields = {
        "object_key",
        "materialized_path",
        "object_sha256",
        *outcome_fields,
    }
    for raw_outcome in raw_outcomes:
        if not isinstance(raw_outcome, dict) or set(raw_outcome) != expected_outcome_fields:
            raise ValueError("accounting instance outcome fields differ")
        outcome_path = _safe_materialized_file(
            root,
            raw_outcome["materialized_path"],
            "instance outcome materialized_path",
        )
        outcome_bytes = outcome_path.read_bytes()
        object_key = _require_string(
            raw_outcome["object_key"],
            "instance outcome object_key",
        )
        outcome = _parse_instance_outcome(
            campaign,
            object_key,
            outcome_bytes,
            launches=launches,
        )
        if {field: raw_outcome[field] for field in outcome_fields} != asdict(outcome):
            raise ValueError("accounting instance outcome content differs")
        stored_outcome = StoredInstanceOutcome(
            object_key=object_key,
            object_sha256=_require_sha256(
                raw_outcome["object_sha256"],
                "instance outcome object digest",
            ),
            object_bytes=outcome_bytes,
            outcome=outcome,
        )
        if stored_outcome.object_sha256 != _sha256_bytes(outcome_bytes):
            raise ValueError("accounting instance outcome digest differs")
        if raw_outcome["materialized_path"] != _materialized_instance_outcome_path(stored_outcome):
            raise ValueError("accounting instance outcome materialized path differs")
        outcome_objects.append(stored_outcome)
    outcome_objects_tuple = tuple(outcome_objects)
    outcome_identities = [
        (stored.outcome.spec_sha256, stored.outcome.attempt) for stored in outcome_objects_tuple
    ]
    if len(outcome_identities) != len(set(outcome_identities)):
        raise ValueError("accounting instance outcomes contain duplicate attempts")

    raw_completed = payload.get("completed_shards")
    if not isinstance(raw_completed, list) or len(raw_completed) != len(campaign.specs):
        raise ValueError("accounting completed shard inventory differs")
    completed: list[MaterializedShard] = []
    for spec, raw_shard in zip(campaign.specs, raw_completed, strict=True):
        if not isinstance(raw_shard, dict):
            raise TypeError("accounting completed shard must be a JSON object")
        shard_dir = _materialized_path(root, spec)
        runtime = _validate_cloud_receipt(shard_dir, campaign, spec)
        launch = _require_matching_launch(launches, spec, runtime)
        materialized = _materialized_shard(
            root,
            shard_dir,
            campaign=campaign,
            spec=spec,
            runtime=runtime,
            launch=launch,
            archive_sha256=_require_sha256(
                raw_shard.get("archive_sha256"),
                "accounting archive digest",
            ),
        )
        if raw_shard.get("receipt_sha256") != materialized.receipt_sha256:
            raise ValueError("accounting shard receipt digest differs")
        completed.append(materialized)

    calibration_verified = shards.discover_verified_shards(
        root / "calibration",
        target_analysis="calibration",
        profile=campaign.profile,
        base_seed=campaign.base_seed,
    )
    behavior_verified = shards.discover_verified_shards(
        root / "behavior",
        target_analysis="behavior",
        profile=campaign.profile,
        base_seed=campaign.base_seed,
    )
    _require_verified_campaign_inventory(
        campaign,
        (calibration_verified, behavior_verified),
    )

    expected_provenance_files = {
        MATERIALIZED_CAMPAIGN_MANIFEST,
        *(_materialized_launch_request_path(stored) for stored in request_objects_tuple),
        *(_materialized_launch_rejection_path(stored) for stored in rejection_objects_tuple),
        *(_materialized_launch_record_path(stored) for stored in launch_objects_tuple),
        *(_materialized_instance_outcome_path(stored) for stored in outcome_objects_tuple),
    }
    observed_provenance_files = {
        path.relative_to(root).as_posix()
        for path in (root / CLOUD_PROVENANCE_DIRECTORY).rglob("*")
        if path.is_file()
    }
    if observed_provenance_files != expected_provenance_files:
        raise ValueError("materialized cloud provenance file inventory differs")
    _require_materialized_file_inventory(
        root,
        campaign,
        request_objects_tuple,
        rejection_objects_tuple,
        launch_objects_tuple,
        outcome_objects_tuple,
    )

    expected = compute_accounting_payload(
        campaign,
        campaign_manifest_sha256=_sha256_bytes(manifest_bytes),
        campaign_manifest_bytes=len(manifest_bytes),
        request_objects=request_objects_tuple,
        rejection_objects=rejection_objects_tuple,
        launch_objects=launch_objects_tuple,
        outcome_objects=outcome_objects_tuple,
        completed=completed,
    )
    if payload != expected:
        raise ValueError("cloud compute accounting values differ from source artifacts")
    return payload


def materialize_campaign(
    campaign: CloudCampaign,
    output_dir: Path,
    *,
    ec2_client: Any | None = None,
    s3_client: Any | None = None,
) -> Path:
    """Download, verify, and materialize every campaign shard."""
    client = s3_client or boto3.client("s3", region_name=campaign.region)
    campaign_manifest = _validated_campaign_manifest(campaign, client)
    status = campaign_status(campaign, s3_client=client)
    if len(status.instance_outcomes) != len(status.launches) and ec2_client is not None:
        status = refresh_campaign_provenance(
            campaign,
            ec2_client=ec2_client,
            s3_client=client,
        )
    if status.missing:
        raise RuntimeError(f"campaign is incomplete: {len(status.missing)} shards missing")
    if status.unresolved_requests:
        raise RuntimeError(
            f"campaign contains {len(status.unresolved_requests)} unresolved launch requests"
        )
    if len(status.instance_outcomes) != len(status.launches):
        raise RuntimeError("campaign contains accepted instances without terminal outcomes")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-",
        dir=output_dir.parent,
    ) as temporary:
        staging = Path(temporary) / output_dir.name
        staging.mkdir()
        manifest_path = staging / MATERIALIZED_CAMPAIGN_MANIFEST
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_bytes(campaign_manifest)
        for stored_request in status.request_objects:
            request_path = staging / _materialized_launch_request_path(stored_request)
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_bytes(stored_request.object_bytes)
        for stored_rejection in status.rejection_objects:
            rejection_path = staging / _materialized_launch_rejection_path(stored_rejection)
            rejection_path.parent.mkdir(parents=True, exist_ok=True)
            rejection_path.write_bytes(stored_rejection.object_bytes)
        for stored_launch in status.launch_objects:
            launch_path = staging / _materialized_launch_record_path(stored_launch)
            launch_path.parent.mkdir(parents=True, exist_ok=True)
            launch_path.write_bytes(stored_launch.object_bytes)
        for stored_outcome in status.outcome_objects:
            outcome_path = staging / _materialized_instance_outcome_path(stored_outcome)
            outcome_path.parent.mkdir(parents=True, exist_ok=True)
            outcome_path.write_bytes(stored_outcome.object_bytes)
        completed_shards: list[MaterializedShard] = []
        for spec in campaign.specs:
            archive_path = Path(temporary) / f"{shard_spec_sha256(spec)}.tar.gz"
            expected_sha256 = _download_stored_archive(
                campaign,
                spec,
                archive_path,
                s3_client=client,
            )
            shard_dir = _materialized_path(staging, spec)
            extract_archive(archive_path, shard_dir)
            shards.verify_shard_directory(
                shard_dir,
                expected_spec=spec,
                expected_git_sha=campaign.git_sha,
                allow_cloud_execution=True,
                require_cloud_execution=True,
            )
            runtime = _validate_cloud_receipt(shard_dir, campaign, spec)
            launch = _require_matching_launch(status.launches, spec, runtime)
            completed_shards.append(
                _materialized_shard(
                    staging,
                    shard_dir,
                    campaign=campaign,
                    spec=spec,
                    runtime=runtime,
                    launch=launch,
                    archive_sha256=expected_sha256,
                )
            )
        calibration_verified = shards.discover_verified_shards(
            staging / "calibration",
            target_analysis="calibration",
            profile=campaign.profile,
            base_seed=campaign.base_seed,
        )
        behavior_verified = shards.discover_verified_shards(
            staging / "behavior",
            target_analysis="behavior",
            profile=campaign.profile,
            base_seed=campaign.base_seed,
        )
        _require_verified_campaign_inventory(
            campaign,
            (calibration_verified, behavior_verified),
        )
        accounting = compute_accounting_payload(
            campaign,
            campaign_manifest_sha256=_sha256_bytes(campaign_manifest),
            campaign_manifest_bytes=len(campaign_manifest),
            request_objects=status.request_objects,
            rejection_objects=status.rejection_objects,
            launch_objects=status.launch_objects,
            outcome_objects=status.outcome_objects,
            completed=completed_shards,
        )
        (staging / COMPUTE_ACCOUNTING_FILENAME).write_text(
            json.dumps(accounting, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        validate_compute_accounting(staging)
        staging.rename(output_dir)
    return output_dir


def _worker_user_data(
    campaign: CloudCampaign,
    spec: shards.ShardSpec,
    *,
    attempt: int,
) -> str:
    attempt = _require_integer(attempt, "attempt", minimum=1)
    image_registry = campaign.image_uri.split("/", maxsplit=1)[0]
    recovery_hook = per_boot_container_recovery_hook(
        "citrees-jss-shard",
        restart_container=False,
    )
    return textwrap.dedent(
        f"""\
        #!/bin/bash
        exec > >(tee /var/log/user-data.log) 2>&1
        set -euo pipefail

        shutdown_instance() {{
            trap - EXIT
            shutdown -h now || systemctl poweroff --force --force || poweroff -f || halt -f || true
        }}
        trap shutdown_instance EXIT

        TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \\
            -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
        INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \\
            http://169.254.169.254/latest/meta-data/instance-id)
        AVAILABILITY_ZONE=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \\
            http://169.254.169.254/latest/meta-data/placement/availability-zone)

        yum install -y docker amazon-ssm-agent
        systemctl enable --now docker
        systemctl enable --now amazon-ssm-agent
        aws logs create-log-group --log-group-name {LOG_GROUP} \\
            --region {campaign.region} 2>/dev/null || true
        aws ecr get-login-password --region {campaign.region} | \\
            docker login --username AWS --password-stdin {image_registry}
        docker pull {campaign.image_uri}
{recovery_hook}
        docker run -d --restart no \\
            --name citrees-jss-shard \\
            --log-driver=awslogs \\
            --log-opt awslogs-region={campaign.region} \\
            --log-opt awslogs-group={LOG_GROUP} \\
            --log-opt awslogs-stream=$INSTANCE_ID \\
            -e AWS_ACCOUNT_ID={campaign.aws_account_id} \\
            -e AWS_DEFAULT_REGION={campaign.region} \\
            -e CITREES_CAMPAIGN_SHA256={campaign.campaign_sha256} \\
            -e CITREES_IMAGE_URI={campaign.image_uri} \\
            -e CITREES_SOURCE_CLEAN=1 \\
            -e EC2_AMI_ID={campaign.ami_id} \\
            -e EC2_AVAILABILITY_ZONE=$AVAILABILITY_ZONE \\
            -e EC2_INSTANCE_ID=$INSTANCE_ID \\
            -e EC2_INSTANCE_TYPE={shlex.quote(campaign.instance_type)} \\
            -e EC2_LAUNCH_ATTEMPT={attempt} \\
            -e EC2_MARKET={campaign.market} \\
            -e GIT_SHA={campaign.git_sha} \\
            -e S3_BUCKET={campaign.bucket} \\
            {campaign.image_uri} \\
            python -m paper.jss.replication.cloud worker \\
                --bucket {campaign.bucket} \\
                --manifest-key {campaign.manifest_key} \\
                --campaign-sha256 {campaign.campaign_sha256} \\
                --target-analysis {spec.target_analysis} \\
                --component {spec.component} \\
                --profile {spec.profile} \\
                --shard-index {spec.shard_index} \\
                --num-shards {spec.num_shards} \\
                --seed {spec.base_seed}
        EXIT_CODE=$(docker wait citrees-jss-shard)
        if [ "$EXIT_CODE" -ne 0 ]; then
            exit "$EXIT_CODE"
        fi
        """
    )


def _logical_cpus_from_instance(instance: Mapping[str, object]) -> int:
    cpu_options = instance.get("CpuOptions")
    if not isinstance(cpu_options, Mapping):
        raise TypeError("EC2 instance omits CpuOptions")
    cores = _require_integer(
        cpu_options.get("CoreCount"),
        "EC2 instance core count",
        minimum=1,
    )
    threads_per_core = _require_integer(
        cpu_options.get("ThreadsPerCore"),
        "EC2 instance threads per core",
        minimum=1,
    )
    return cores * threads_per_core


def _campaign_instances(
    campaign: CloudCampaign,
    ec2_client: Any,
) -> tuple[ObservedInstance, ...]:
    filters = [
        {"Name": f"tag:{TAG_KEY}", "Values": [ROLE_TAG_VALUE]},
        {
            "Name": "tag:citrees-jss-campaign-sha256",
            "Values": [campaign.campaign_sha256],
        },
    ]
    specs_by_sha256 = {shard_spec_sha256(spec): spec for spec in campaign.specs}
    observed: list[ObservedInstance] = []
    next_token: str | None = None
    seen_tokens: set[str] = set()
    while True:
        arguments: dict[str, object] = {"Filters": filters}
        if next_token is not None:
            arguments["NextToken"] = next_token
        response = ec2_client.describe_instances(**arguments)
        for reservation in response.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                tags = {
                    tag["Key"]: tag["Value"]
                    for tag in instance.get("Tags", [])
                    if "Key" in tag and "Value" in tag
                }
                spec_sha256 = tags.get("citrees-jss-spec-sha256")
                if spec_sha256 is not None:
                    spec = specs_by_sha256.get(spec_sha256)
                    if spec is None:
                        raise ValueError("campaign EC2 instance identifies an unknown JSS shard")
                    attempt_text = _require_string(
                        tags.get("citrees-jss-attempt"),
                        "campaign instance launch attempt",
                    )
                    try:
                        attempt = int(attempt_text)
                    except ValueError as error:
                        raise ValueError("campaign instance launch attempt is invalid") from error
                    if str(attempt) != attempt_text:
                        raise ValueError("campaign instance launch attempt is not canonical")
                    request_index_text = _require_string(
                        tags.get("citrees-jss-request-index"),
                        "campaign instance launch request index",
                    )
                    try:
                        request_index = int(request_index_text)
                    except ValueError as error:
                        raise ValueError(
                            "campaign instance launch request index is invalid"
                        ) from error
                    if str(request_index) != request_index_text:
                        raise ValueError("campaign instance launch request index is not canonical")
                    client_token = _require_string(
                        tags.get("citrees-jss-client-token"),
                        "campaign instance launch client token",
                    )
                    market = _require_spot_market(
                        tags.get("citrees-jss-market"),
                        "campaign instance market",
                    )
                    lifecycle = instance.get("InstanceLifecycle")
                    if lifecycle != SPOT_MARKET:
                        raise ValueError("campaign instance lifecycle differs from its market tag")
                    instance_type = _require_string(
                        instance.get("InstanceType"),
                        "campaign instance type",
                    )
                    if instance_type != campaign.instance_type:
                        raise ValueError("campaign instance type differs from the campaign")
                    placement = instance.get("Placement")
                    if not isinstance(placement, dict):
                        raise TypeError("campaign instance placement must be a mapping")
                    launch_time = instance.get("LaunchTime")
                    if not isinstance(launch_time, datetime) or launch_time.tzinfo is None:
                        raise TypeError("campaign instance launch time must be timezone-aware")
                    state = instance.get("State")
                    if not isinstance(state, Mapping):
                        raise TypeError("campaign instance state must be a mapping")
                    state_name = _require_string(
                        state.get("Name"),
                        "campaign instance state",
                    )
                    if state_name not in {
                        "pending",
                        "running",
                        "shutting-down",
                        "terminated",
                        "stopping",
                        "stopped",
                    }:
                        raise ValueError("campaign instance state is invalid")
                    transition_reason = instance.get("StateTransitionReason", "")
                    if not isinstance(transition_reason, str):
                        raise TypeError("campaign instance transition reason must be a string")
                    observed.append(
                        ObservedInstance(
                            launch=LaunchRecord(
                                spec=spec,
                                spec_sha256=spec_sha256,
                                request_index=_require_integer(
                                    request_index,
                                    "campaign instance launch request index",
                                    minimum=1,
                                ),
                                attempt=_require_integer(
                                    attempt,
                                    "campaign instance launch attempt",
                                    minimum=1,
                                ),
                                market=market,
                                instance_type=instance_type,
                                client_token=client_token,
                                instance_id=_require_string(
                                    instance.get("InstanceId"),
                                    "campaign instance identifier",
                                ),
                                availability_zone=_require_string(
                                    placement.get("AvailabilityZone"),
                                    "campaign instance availability zone",
                                ),
                                launch_time=launch_time.astimezone(UTC).isoformat(),
                                logical_cpus=_logical_cpus_from_instance(instance),
                            ),
                            state=state_name,
                            state_transition_reason=transition_reason,
                        ),
                    )
        raw_next_token = response.get("NextToken")
        if raw_next_token is None:
            break
        next_token = _require_string(raw_next_token, "EC2 pagination token")
        if next_token in seen_tokens:
            raise RuntimeError("EC2 campaign-instance pagination repeated a token")
        seen_tokens.add(next_token)
    identities = [(item.launch.spec_sha256, item.launch.request_index) for item in observed]
    instance_ids = [item.launch.instance_id for item in observed]
    if len(identities) != len(set(identities)) or len(instance_ids) != len(set(instance_ids)):
        raise RuntimeError("EC2 campaign-instance discovery returned duplicate identities")
    spec_positions = {shard_spec_sha256(spec): index for index, spec in enumerate(campaign.specs)}
    return tuple(
        sorted(
            observed,
            key=lambda item: (
                spec_positions[item.launch.spec_sha256],
                item.launch.request_index,
            ),
        )
    )


def _exact_instance_state(
    ec2_client: Any,
    instance_id: str,
) -> tuple[str, str] | None:
    """Return one exact-ID EC2 state, or None when EC2 reports absence."""
    try:
        response = ec2_client.describe_instances(InstanceIds=[instance_id])
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        if code == "InvalidInstanceID.NotFound":
            return None
        raise
    instances = [
        instance
        for reservation in response.get("Reservations", [])
        for instance in reservation.get("Instances", [])
    ]
    if not instances:
        return None
    if len(instances) != 1:
        raise RuntimeError("exact EC2 instance lookup returned multiple instances")
    instance = instances[0]
    observed_id = _require_string(
        instance.get("InstanceId"),
        "exact EC2 instance identifier",
    )
    if observed_id != instance_id:
        raise RuntimeError("exact EC2 instance lookup returned a different instance")
    state = instance.get("State")
    if not isinstance(state, Mapping):
        raise TypeError("exact EC2 instance state must be a mapping")
    state_name = _require_string(state.get("Name"), "exact EC2 instance state")
    if state_name not in {
        "pending",
        "running",
        "shutting-down",
        "terminated",
        "stopping",
        "stopped",
    }:
        raise ValueError("exact EC2 instance state is invalid")
    transition_reason = instance.get("StateTransitionReason", "")
    if not isinstance(transition_reason, str):
        raise TypeError("exact EC2 instance transition reason must be a string")
    return state_name, transition_reason


def _reconcile_exact_instance_state(
    record: LaunchRecord,
    ec2_client: Any,
) -> tuple[str, str] | None:
    """Resolve an exact-ID state while honoring EC2's propagation window."""
    observation = _exact_instance_state(ec2_client, record.instance_id)
    if observation is not None:
        return observation
    launch_age = (
        datetime.now(UTC) - _parse_canonical_utc(record.launch_time, "launch_time")
    ).total_seconds()
    if launch_age < EC2_INSTANCE_PROPAGATION_SECONDS:
        return None
    for retry in range(EC2_EXACT_LOOKUP_MAX_ATTEMPTS - 1):
        time.sleep(EC2_EXACT_LOOKUP_RETRY_SECONDS * (2**retry))
        observation = _exact_instance_state(ec2_client, record.instance_id)
        if observation is not None:
            return observation
    return "terminated", "EC2.InstancePersistentlyAbsentFromExactLookup"


def _reconcile_campaign_instances(
    campaign: CloudCampaign,
    requests: Sequence[LaunchRequest],
    launches: Sequence[LaunchRecord],
    outcomes: Sequence[InstanceOutcome],
    observed: Sequence[ObservedInstance],
    *,
    ec2_client: Any,
    s3_client: Any,
) -> dict[str, LaunchRecord]:
    reconciled = list(launches)
    request_by_identity = {
        (request.spec_sha256, request.request_index): request for request in requests
    }
    for item in observed:
        record = item.launch
        request = request_by_identity.get((record.spec_sha256, record.request_index))
        if request is None:
            raise ValueError("campaign EC2 instance has no immutable launch request")
        if (
            record.attempt != request.attempt
            or record.market != request.market
            or record.instance_type != request.instance_type
            or record.client_token != request.client_token
        ):
            raise ValueError("campaign EC2 instance differs from its launch request")
        matches = [
            launch
            for launch in reconciled
            if launch.spec_sha256 == record.spec_sha256
            and launch.request_index == record.request_index
        ]
        if matches:
            if len(matches) != 1 or matches[0] != record:
                raise ValueError("campaign EC2 instance differs from its immutable launch record")
        else:
            publish_launch_record(campaign, record, s3_client=s3_client)
            reconciled.append(record)
        if item.state == "terminated":
            publish_instance_outcome(
                campaign,
                InstanceOutcome(
                    spec_sha256=record.spec_sha256,
                    attempt=record.attempt,
                    instance_id=record.instance_id,
                    state="terminated",
                    state_transition_reason=item.state_transition_reason,
                    observed_utc=datetime.now(UTC).isoformat(),
                ),
                launches=reconciled,
                s3_client=s3_client,
            )
        elif item.state in {"stopped", "stopping"}:
            raise RuntimeError(
                f"campaign instance {record.instance_id} entered unexpected state {item.state!r}"
            )
    reconciled_records = list_launch_records(campaign, s3_client=s3_client)
    stored_outcomes = _load_instance_outcomes(
        campaign,
        launches=reconciled_records,
        s3_client=s3_client,
    )
    reconciled_outcomes = tuple(stored.outcome for stored in stored_outcomes)
    known_outcomes = {
        (outcome.spec_sha256, outcome.attempt): outcome
        for outcome in (*outcomes, *reconciled_outcomes)
    }
    observed_attempts = {(item.launch.spec_sha256, item.launch.attempt) for item in observed}
    for record in reconciled_records:
        identity = (record.spec_sha256, record.attempt)
        if identity in known_outcomes or identity in observed_attempts:
            continue
        exact_state = _reconcile_exact_instance_state(record, ec2_client)
        if exact_state is None:
            continue
        state, transition_reason = exact_state
        if state == "terminated":
            known_outcomes[identity] = publish_instance_outcome(
                campaign,
                InstanceOutcome(
                    spec_sha256=record.spec_sha256,
                    attempt=record.attempt,
                    instance_id=record.instance_id,
                    state="terminated",
                    state_transition_reason=transition_reason,
                    observed_utc=datetime.now(UTC).isoformat(),
                ),
                launches=reconciled_records,
                s3_client=s3_client,
            )
        elif state in {"stopped", "stopping"}:
            raise RuntimeError(
                f"campaign instance {record.instance_id} entered unexpected state {state!r}"
            )
    active: dict[str, LaunchRecord] = {}
    for record in reconciled_records:
        if (record.spec_sha256, record.attempt) in known_outcomes:
            continue
        if record.spec_sha256 in active:
            raise RuntimeError(
                f"multiple unterminated instances claim JSS shard {record.spec_sha256}"
            )
        active[record.spec_sha256] = record
    return active


def refresh_campaign_provenance(
    campaign: CloudCampaign,
    *,
    ec2_client: Any | None = None,
    s3_client: Any | None = None,
) -> CampaignStatus:
    """Reconcile EC2 launch and terminal state into immutable S3 provenance."""
    s3 = s3_client or boto3.client("s3", region_name=campaign.region)
    ec2 = ec2_client or boto3.client("ec2", region_name=campaign.region)
    status = campaign_status(campaign, s3_client=s3)
    _reconcile_campaign_instances(
        campaign,
        status.launch_requests,
        status.launches,
        status.instance_outcomes,
        _campaign_instances(campaign, ec2),
        ec2_client=ec2,
        s3_client=s3,
    )
    return campaign_status(campaign, s3_client=s3)


def _client_token(
    campaign: CloudCampaign,
    spec: shards.ShardSpec,
    *,
    request_index: int,
    attempt: int,
    instance_type: str,
    subnet_id: str,
    instance_profile_name: str,
    security_group_id: str,
) -> str:
    payload = _canonical_json(
        {
            "campaign_sha256": campaign.campaign_sha256,
            "spec_sha256": shard_spec_sha256(spec),
            "request_index": request_index,
            "attempt": attempt,
            "market": campaign.market,
            "instance_type": instance_type,
            "subnet_id": subnet_id,
            "instance_profile_name": instance_profile_name,
            "security_group_id": security_group_id,
        }
    )
    return f"citrees-jss-{_sha256_bytes(payload)[:51]}"


def _create_launch_request(
    campaign: CloudCampaign,
    spec: shards.ShardSpec,
    *,
    requests: Sequence[LaunchRequest],
    launches: Sequence[LaunchRecord],
    subnet_id: str,
    instance_profile_name: str,
    security_group_id: str,
) -> LaunchRequest:
    request_index = _next_request_index(requests, spec)
    attempt = _next_launch_attempt(launches, spec)
    instance_type = campaign.instance_type
    request = LaunchRequest(
        spec=spec,
        spec_sha256=shard_spec_sha256(spec),
        request_index=request_index,
        attempt=attempt,
        market=campaign.market,
        instance_type=instance_type,
        subnet_id=_require_string(subnet_id, "launch request subnet"),
        instance_profile_name=_require_string(
            instance_profile_name,
            "launch request instance profile",
        ),
        security_group_id=_require_string(
            security_group_id,
            "launch request security group",
        ),
        client_token=_client_token(
            campaign,
            spec,
            request_index=request_index,
            attempt=attempt,
            instance_type=instance_type,
            subnet_id=subnet_id,
            instance_profile_name=instance_profile_name,
            security_group_id=security_group_id,
        ),
        run_instances_sha256="0" * 64,
    )
    return replace(
        request,
        run_instances_sha256=_run_instances_sha256(campaign, request),
    )


def _instance_tags(
    campaign: CloudCampaign,
    request: LaunchRequest,
) -> list[dict[str, str]]:
    if _require_spot_market(request.market, "launch request market") != campaign.market:
        raise ValueError("launch request market differs from the campaign")
    return [
        {"Key": TAG_KEY, "Value": ROLE_TAG_VALUE},
        {"Key": "Name", "Value": "citrees-jss-shard"},
        {
            "Key": "citrees-jss-campaign-sha256",
            "Value": campaign.campaign_sha256,
        },
        {
            "Key": "citrees-jss-spec-sha256",
            "Value": request.spec_sha256,
        },
        {"Key": "citrees-jss-component", "Value": request.spec.component},
        {
            "Key": "citrees-jss-shard-index",
            "Value": str(request.spec.shard_index),
        },
        {"Key": "citrees-jss-request-index", "Value": str(request.request_index)},
        {"Key": "citrees-jss-attempt", "Value": str(request.attempt)},
        {"Key": "citrees-jss-market", "Value": request.market},
        {"Key": "citrees-jss-client-token", "Value": request.client_token},
        {"Key": "citrees-image-uri", "Value": campaign.image_uri},
    ]


def _run_instance_arguments(
    campaign: CloudCampaign,
    request: LaunchRequest,
) -> dict[str, object]:
    user_data = _worker_user_data(
        campaign,
        request.spec,
        attempt=request.attempt,
    )
    tags = _instance_tags(campaign, request)
    arguments: dict[str, object] = {
        "ImageId": campaign.ami_id,
        "InstanceType": request.instance_type,
        "MinCount": 1,
        "MaxCount": 1,
        "IamInstanceProfile": {"Name": request.instance_profile_name},
        "UserData": base64.b64encode(user_data.encode()).decode(),
        "MetadataOptions": {
            "HttpEndpoint": "enabled",
            "HttpTokens": "required",
            "HttpPutResponseHopLimit": 2,
        },
        "SecurityGroupIds": [request.security_group_id],
        "SubnetId": request.subnet_id,
        "InstanceInitiatedShutdownBehavior": "terminate",
        "ClientToken": request.client_token,
        "TagSpecifications": [
            {"ResourceType": "instance", "Tags": tags},
            {"ResourceType": "volume", "Tags": tags},
        ],
    }
    arguments["InstanceMarketOptions"] = {
        "MarketType": SPOT_MARKET,
        "SpotOptions": {
            "SpotInstanceType": "one-time",
            "InstanceInterruptionBehavior": "terminate",
        },
    }
    return arguments


def _run_instances_sha256(
    campaign: CloudCampaign,
    request: LaunchRequest,
) -> str:
    return _sha256_bytes(_canonical_json(_run_instance_arguments(campaign, request)))


def _execute_launch_request(
    campaign: CloudCampaign,
    request: LaunchRequest,
    *,
    requests: Sequence[LaunchRequest],
    ec2_client: Any,
    s3_client: Any,
) -> tuple[LaunchRecord | None, str | None]:
    try:
        response = ec2_client.run_instances(**_run_instance_arguments(campaign, request))
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        if code not in _CAPACITY_ERRORS:
            raise
        publish_launch_rejection(
            campaign,
            LaunchRejection(
                spec_sha256=request.spec_sha256,
                request_index=request.request_index,
                client_token=request.client_token,
                error_code=code,
            ),
            requests=requests,
            s3_client=s3_client,
        )
        return None, code
    instances = response.get("Instances", [])
    if len(instances) != 1:
        raise RuntimeError("EC2 returned an invalid JSS shard launch response")
    instance = instances[0]
    placement = instance.get("Placement")
    if not isinstance(placement, dict):
        raise TypeError("EC2 launch response omits Placement")
    availability_zone = _require_string(
        placement.get("AvailabilityZone"),
        "EC2 availability zone",
    )
    launch_time_value = instance.get("LaunchTime")
    if not isinstance(launch_time_value, datetime) or launch_time_value.tzinfo is None:
        raise TypeError("EC2 launch response omits a timezone-aware LaunchTime")
    record = LaunchRecord(
        spec=request.spec,
        spec_sha256=request.spec_sha256,
        request_index=request.request_index,
        attempt=request.attempt,
        market=request.market,
        instance_type=request.instance_type,
        client_token=request.client_token,
        instance_id=_require_string(instance.get("InstanceId"), "EC2 instance identifier"),
        availability_zone=availability_zone,
        launch_time=launch_time_value.astimezone(UTC).isoformat(),
        logical_cpus=_logical_cpus_from_instance(instance),
    )
    publish_launch_record(campaign, record, s3_client=s3_client)
    return record, None


def launch_missing_shards(
    campaign: CloudCampaign,
    *,
    max_new_instances: int | None = None,
    ec2_client: Any | None = None,
    s3_client: Any | None = None,
) -> tuple[LaunchRecord, ...]:
    """Launch missing, inactive shards as immutable one-time Spot instances."""
    if max_new_instances is not None:
        max_new_instances = _require_integer(
            max_new_instances,
            "max_new_instances",
            minimum=1,
        )
    if get_aws_account_id() != campaign.aws_account_id:
        raise RuntimeError("active AWS account differs from the JSS campaign")
    image_revision = validate_image_revision(
        campaign.image_uri,
        campaign.git_sha,
        region=campaign.region,
    )
    if image_revision != campaign.git_sha:
        raise RuntimeError("launch image revision differs from the JSS campaign")
    s3 = s3_client or boto3.client("s3", region_name=campaign.region)
    publish_campaign(campaign, s3_client=s3)
    status = campaign_status(campaign, s3_client=s3)
    ec2 = ec2_client or boto3.client("ec2", region_name=campaign.region)
    observed = _campaign_instances(campaign, ec2)
    active = _reconcile_campaign_instances(
        campaign,
        status.launch_requests,
        status.launches,
        status.instance_outcomes,
        observed,
        ec2_client=ec2,
        s3_client=s3,
    )
    status = campaign_status(campaign, s3_client=s3)
    inactive_missing = [spec for spec in status.missing if shard_spec_sha256(spec) not in active]
    if status.unresolved_requests:
        raise RuntimeError(
            "JSS launch request has no discoverable EC2 outcome; refusing automatic replay"
        )
    if not inactive_missing and not status.unresolved_requests:
        return ()

    security_group_id = ensure_security_group(campaign.region)
    instance_profile_name = ensure_campaign_iam_profile(
        output_prefix=campaign.output_prefix,
        campaign_sha256=campaign.campaign_sha256,
        read_keys=(campaign.manifest_key,),
        write_prefixes=(f"{campaign.output_prefix}/shards",),
        region=campaign.region,
    )
    placement_subnets = get_default_subnet_ids(
        ec2,
        instance_type=campaign.instance_type,
    )
    if not placement_subnets:
        raise RuntimeError(f"no default subnets offer instance type {campaign.instance_type}")

    requests = list(status.launch_requests)
    launches = list(status.launches)
    observed = _campaign_instances(campaign, ec2)
    active = _reconcile_campaign_instances(
        campaign,
        status.launch_requests,
        status.launches,
        status.instance_outcomes,
        observed,
        ec2_client=ec2,
        s3_client=s3,
    )
    status = campaign_status(campaign, s3_client=s3)
    requests = list(status.launch_requests)
    launches = list(status.launches)
    missing = [spec for spec in status.missing if shard_spec_sha256(spec) not in active]
    if max_new_instances is not None:
        missing = missing[:max_new_instances]
    if not missing:
        return ()

    spec_positions = {spec: index for index, spec in enumerate(campaign.specs)}
    records: list[LaunchRecord] = []
    for spec in missing:
        request_index = _next_request_index(requests, spec)
        subnet_id = placement_subnets[
            (spec_positions[spec] + request_index - 1) % len(placement_subnets)
        ]
        request = _create_launch_request(
            campaign,
            spec,
            requests=requests,
            launches=launches,
            subnet_id=subnet_id,
            instance_profile_name=instance_profile_name,
            security_group_id=security_group_id,
        )
        publish_launch_request(campaign, request, s3_client=s3)
        requests.append(request)
        record, rejection_code = _execute_launch_request(
            campaign,
            request,
            requests=requests,
            ec2_client=ec2,
            s3_client=s3,
        )
        if record is None:
            if rejection_code not in _CAPACITY_ERRORS:
                raise AssertionError("launch rejection must identify a capacity error")
            break
        records.append(record)
        launches.append(record)
    return tuple(records)


def _runtime_from_environment() -> CloudRuntime:
    required = {
        name: os.environ.get(name, "").strip()
        for name in (
            "AWS_ACCOUNT_ID",
            "CITREES_IMAGE_URI",
            "EC2_AMI_ID",
            "EC2_AVAILABILITY_ZONE",
            "EC2_INSTANCE_ID",
            "EC2_INSTANCE_TYPE",
            "EC2_LAUNCH_ATTEMPT",
            "EC2_MARKET",
        )
    }
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        raise RuntimeError(f"cloud worker environment omits {missing}")
    return CloudRuntime(
        aws_account_id=required["AWS_ACCOUNT_ID"],
        instance_id=required["EC2_INSTANCE_ID"],
        instance_type=required["EC2_INSTANCE_TYPE"],
        availability_zone=required["EC2_AVAILABILITY_ZONE"],
        market=_require_spot_market(required["EC2_MARKET"], "EC2_MARKET"),
        image_uri=required["CITREES_IMAGE_URI"],
        ami_id=required["EC2_AMI_ID"],
        attempt=_require_integer(
            int(required["EC2_LAUNCH_ATTEMPT"]),
            "EC2_LAUNCH_ATTEMPT",
            minimum=1,
        ),
    )


def _add_campaign_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=("smoke", "quick", "full"), required=True)
    parser.add_argument("--seed", type=int, default=shards.BASE_SEED)
    for component in COMPONENTS:
        parser.add_argument(
            f"--{component.replace('_', '-')}-shards",
            type=int,
            required=True,
        )
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--instance-type", required=True)
    parser.add_argument("--ami-id", required=True)
    parser.add_argument("--region", default=DEFAULT_REGION)


def _campaign_from_arguments(args: argparse.Namespace) -> CloudCampaign:
    account_id = get_aws_account_id()
    git_sha = get_frozen_git_sha()
    return create_campaign(
        cast(shards.Profile, args.profile),
        base_seed=args.seed,
        git_sha=git_sha,
        image_uri=args.image_uri,
        aws_account_id=account_id,
        bucket=get_resource_name(account_id),
        region=args.region,
        instance_type=args.instance_type,
        ami_id=args.ami_id,
        shard_counts={component: getattr(args, f"{component}_shards") for component in COMPONENTS},
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch = subparsers.add_parser("launch")
    _add_campaign_arguments(launch)
    launch.add_argument("--max-new-instances", type=int)

    status = subparsers.add_parser("status")
    _add_campaign_arguments(status)

    materialize = subparsers.add_parser("materialize")
    _add_campaign_arguments(materialize)
    materialize.add_argument("--output-dir", type=Path, required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--bucket", required=True)
    worker.add_argument("--manifest-key", required=True)
    worker.add_argument("--campaign-sha256", required=True)
    worker.add_argument(
        "--target-analysis",
        choices=("calibration", "behavior"),
        required=True,
    )
    worker.add_argument("--component", required=True)
    worker.add_argument("--profile", choices=("smoke", "quick", "full"), required=True)
    worker.add_argument("--shard-index", type=int, required=True)
    worker.add_argument("--num-shards", type=int, required=True)
    worker.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "worker":
        region = os.environ.get("AWS_DEFAULT_REGION", DEFAULT_REGION)
        campaign = load_campaign(
            bucket=args.bucket,
            manifest_key=args.manifest_key,
            campaign_sha256=args.campaign_sha256,
            region=region,
        )
        spec = shards.ShardSpec(
            target_analysis=cast(shards.TargetAnalysis, args.target_analysis),
            component=args.component,
            profile=cast(shards.Profile, args.profile),
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            base_seed=args.seed,
        )
        created = run_cloud_shard(campaign, spec, _runtime_from_environment())
        print(f"{'Created' if created else 'Reused'} {archive_key(campaign, spec)}.")
        return

    ensure_s3_bucket(args.region)
    campaign = _campaign_from_arguments(args)
    if args.command == "launch":
        records = launch_missing_shards(
            campaign,
            max_new_instances=args.max_new_instances,
        )
        print(f"Launched {len(records)} JSS shard instances.")
    elif args.command == "status":
        status = refresh_campaign_provenance(campaign)
        print(
            json.dumps(
                {
                    "campaign_sha256": campaign.campaign_sha256,
                    "total": status.total,
                    "completed": len(status.completed),
                    "missing": len(status.missing),
                    "launch_attempts": len(status.launches),
                    "terminal_instance_attempts": len(status.instance_outcomes),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        output = materialize_campaign(
            campaign,
            args.output_dir,
            ec2_client=boto3.client("ec2", region_name=campaign.region),
        )
        print(f"Materialized verified cloud shards to {output}.")


if __name__ == "__main__":
    main()
