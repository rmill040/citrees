"""EC2 instance management for API server and workers.

Launch, list, and terminate EC2 instances that run the experiment
API server and workers via Docker containers pulled from ECR.
"""

from __future__ import annotations

import base64
import hashlib
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
from botocore.exceptions import ClientError

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
_QUEUE_STAGES = frozenset({"rankings", "metrics"})


@dataclass(frozen=True)
class ApiScope:
    """Immutable scope advertised by one running queue server."""

    api_url: str
    public_api_url: str
    artifact_prefix: str
    campaign_sha256: str
    image_uri: str
    manifest_s3_key: str
    manifest_sha256: str
    max_cell_attempts: int
    stage: str


def _csv_arg(values: Sequence[str | int]) -> str:
    """Encode a structured option list for the mechanism runner CLI."""
    return ",".join(str(value) for value in values)


def validate_image_digest_uri(image_uri: str) -> str:
    """Require an immutable container image digest for distributed work."""
    normalized = image_uri.strip()
    if not _IMAGE_DIGEST_PATTERN.fullmatch(normalized):
        raise ValueError("image_uri must be an immutable repository@sha256:<64 hex> URI")
    return normalized


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


def _api_client_token(
    *,
    ami_id: str,
    instance_profile_name: str,
    instance_type: str,
    security_group_id: str,
    user_data: str,
) -> str:
    """Derive EC2 idempotency from the complete API instance request."""
    payload = json.dumps(
        {
            "ami_id": ami_id,
            "instance_profile_name": instance_profile_name,
            "instance_type": instance_type,
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
    manifest_sha256: str,
    max_cell_attempts: int,
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
                    "manifest_sha256": status.get("manifest_sha256"),
                    "max_cell_attempts": status.get("max_cell_attempts"),
                    "stage": status.get("stage"),
                }
                expected = {
                    "artifact_prefix": artifact_prefix,
                    "campaign_sha256": campaign_sha256,
                    "manifest_sha256": manifest_sha256,
                    "max_cell_attempts": max_cell_attempts,
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
    manifest_sha256: str,
    stage: str,
) -> str:
    """Generate EC2 user data script that pulls and runs the worker container."""
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
        TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \\
            -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
        INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \\
            http://169.254.169.254/latest/meta-data/instance-id)

        echo "Instance: $INSTANCE_ID"

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

        # Pull and run the worker image
        docker pull {image_uri}
        docker run -d --restart no \\
            --name citrees-worker \\
            --log-driver=awslogs \\
            --log-opt awslogs-region={region} \\
            --log-opt awslogs-group={LOG_GROUP_WORKER} \\
            --log-opt awslogs-stream=$INSTANCE_ID \\
            -e S3_BUCKET={bucket} \\
            -e CITREES_API_URL={api_url} \\
            -e CITREES_ARTIFACT_PREFIX={shlex.quote(artifact_prefix)} \\
            -e CITREES_CAMPAIGN_SHA256={campaign_sha256} \\
            -e CITREES_MANIFEST_SHA256={manifest_sha256} \\
            -e CITREES_STAGE={stage} \\
            -e CITREES_IMAGE_URI={image_uri} \\
            -e EC2_INSTANCE_TYPE={shlex.quote(instance_type)} \\
            -e AWS_ACCOUNT_ID={bucket.removeprefix("citrees-")} \\
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
    manifest_s3_key: str,
    manifest_sha256: str,
    stage: str,
    lease_seconds: int,
    max_cell_attempts: int,
) -> str:
    """Generate EC2 user data script that runs the API server container."""
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
        TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \\
            -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
        INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \\
            http://169.254.169.254/latest/meta-data/instance-id)

        echo "Instance: $INSTANCE_ID"

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
            -e CITREES_MANIFEST_S3_KEY={shlex.quote(manifest_s3_key)} \\
            -e CITREES_MANIFEST_SHA256={manifest_sha256} \\
            -e CITREES_STAGE={stage} \\
            -e CITREES_LEASE_SECONDS={lease_seconds} \\
            -e CITREES_MAX_CELL_ATTEMPTS={max_cell_attempts} \\
            -e CITREES_IMAGE_URI={image_uri} \\
            -e EC2_INSTANCE_TYPE={shlex.quote(instance_type)} \\
            -e AWS_ACCOUNT_ID={bucket.removeprefix("citrees-")} \\
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
    manifest_path: Path,
    stage: str,
    lease_seconds: int,
    max_cell_attempts: int,
    region: str = DEFAULT_REGION,
) -> dict[str, str]:
    """Launch the API server on a single EC2 instance.

    Returns dict with instance_id, public_ip, and api_url.
    """
    if get_api_scope(region) is not None:
        raise RuntimeError("A citrees API server is already pending or running")
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be a positive integer")
    if type(max_cell_attempts) is not int or max_cell_attempts <= 0:
        raise ValueError("max_cell_attempts must be a positive integer")
    image_uri = validate_image_digest_uri(image_uri)
    artifact_prefix, stage = _validate_queue_scope(artifact_prefix, stage)
    git_sha = validate_image_revision(image_uri, region=region)
    manifest_info = publish_rerun_manifest(manifest_path, region=region)
    manifest_s3_key = str(manifest_info["key"])
    manifest_sha256 = str(manifest_info["sha256"])
    campaign_sha256 = str(manifest_info["campaign_sha256"])
    ec2 = boto3.client("ec2", region_name=region)
    account_id = get_aws_account_id()
    bucket = get_resource_name(account_id)
    ecr_uri = image_uri.split("/")[0]
    sg_id = ensure_security_group(region)
    instance_profile_name = ensure_campaign_iam_profile(
        output_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        write_prefixes=(artifact_prefix,),
        region=region,
    )
    ami_id = get_ami(region)

    user_data = _make_api_user_data(
        region=region,
        ecr_uri=ecr_uri,
        image_uri=image_uri,
        bucket=bucket,
        git_sha=git_sha,
        instance_type=instance_type,
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        manifest_s3_key=manifest_s3_key,
        manifest_sha256=manifest_sha256,
        stage=stage,
        lease_seconds=lease_seconds,
        max_cell_attempts=max_cell_attempts,
    )

    info(f"Launching API server: {instance_type}, AMI={ami_id}")
    step(f"Image: {image_uri}")
    step(f"Manifest: s3://{bucket}/{manifest_s3_key} ({manifest_info['cells']} cells)")
    step(f"Security group: {sg_id}")
    step(f"Instance profile: {instance_profile_name}")
    client_token = _api_client_token(
        ami_id=ami_id,
        instance_profile_name=instance_profile_name,
        instance_type=instance_type,
        security_group_id=sg_id,
        user_data=user_data,
    )

    response = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        MinCount=1,
        MaxCount=1,
        IamInstanceProfile={"Name": instance_profile_name},
        UserData=base64.b64encode(user_data.encode()).decode(),
        MetadataOptions={"HttpPutResponseHopLimit": 2},
        SecurityGroupIds=[sg_id],
        InstanceInitiatedShutdownBehavior="terminate",
        ClientToken=client_token,
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": TAG_KEY, "Value": API_TAG_VALUE},
                    {"Key": "Name", "Value": "citrees-api"},
                    {"Key": "citrees-artifact-prefix", "Value": artifact_prefix},
                    {"Key": "citrees-campaign-sha256", "Value": campaign_sha256},
                    {"Key": "citrees-manifest-key", "Value": manifest_s3_key},
                    {"Key": "citrees-manifest-sha256", "Value": manifest_sha256},
                    {"Key": "citrees-image-uri", "Value": image_uri},
                    {"Key": "citrees-stage", "Value": stage},
                    {"Key": "citrees-lease-seconds", "Value": str(lease_seconds)},
                    {
                        "Key": "citrees-max-cell-attempts",
                        "Value": str(max_cell_attempts),
                    },
                ],
            }
        ],
    )

    instance_id = response["Instances"][0]["InstanceId"]
    step(f"Instance: {instance_id}")

    # Wait for public IP assignment
    info("Waiting for public IP...")
    ec2_resource = boto3.resource("ec2", region_name=region)
    instance = ec2_resource.Instance(instance_id)
    instance.wait_until_running()
    instance.reload()

    public_ip = instance.public_ip_address
    if not public_ip:
        # Retry a few times — IP can take a moment after running state
        for _ in range(10):
            time.sleep(2)
            instance.reload()
            public_ip = instance.public_ip_address
            if public_ip:
                break

    if not public_ip:
        warn("Instance running but no public IP assigned")
        ec2.terminate_instances(InstanceIds=[instance_id])
        raise RuntimeError(f"API instance {instance_id} has no public IP")

    api_url = f"http://{public_ip}:8000"
    info(f"Waiting for API readiness at {api_url}...")
    try:
        _wait_for_api_ready(
            api_url,
            artifact_prefix=artifact_prefix,
            campaign_sha256=campaign_sha256,
            manifest_sha256=manifest_sha256,
            max_cell_attempts=max_cell_attempts,
            stage=stage,
        )
    except Exception:
        ec2.terminate_instances(InstanceIds=[instance_id])
        raise
    success(f"API server ready at {api_url}")

    return {"instance_id": instance_id, "public_ip": public_ip, "api_url": api_url}


def get_api_scope(region: str = DEFAULT_REGION) -> ApiScope | None:
    """Return the immutable scope of the single running API server."""
    ec2 = boto3.client("ec2", region_name=region)

    response = ec2.describe_instances(
        Filters=[
            {"Name": f"tag:{TAG_KEY}", "Values": [API_TAG_VALUE]},
            {"Name": "instance-state-name", "Values": ["pending", "running"]},
        ]
    )

    instances = [
        instance
        for reservation in response.get("Reservations", [])
        for instance in reservation.get("Instances", [])
    ]
    if not instances:
        return None
    if len(instances) != 1:
        instance_ids = sorted(instance["InstanceId"] for instance in instances)
        raise RuntimeError(f"Expected one running API server, found {instance_ids}")

    instance = instances[0]
    private_ip = instance.get("PrivateIpAddress")
    if not private_ip:
        raise RuntimeError(f"API server {instance['InstanceId']} has no private IP")
    public_ip = instance.get("PublicIpAddress")
    if not public_ip:
        raise RuntimeError(f"API server {instance['InstanceId']} has no public IP")
    tags = {tag["Key"]: tag["Value"] for tag in instance.get("Tags", [])}
    required_tags = {
        "citrees-artifact-prefix",
        "citrees-campaign-sha256",
        "citrees-image-uri",
        "citrees-manifest-key",
        "citrees-manifest-sha256",
        "citrees-max-cell-attempts",
        "citrees-stage",
    }
    missing = sorted(required_tags - set(tags))
    if missing:
        raise RuntimeError(f"API server {instance['InstanceId']} is missing scope tags: {missing}")

    from paper.benchmark.pipeline.manifest import manifest_s3_key, validate_manifest_sha256

    artifact_prefix, stage = _validate_queue_scope(
        tags["citrees-artifact-prefix"],
        tags["citrees-stage"],
    )
    image_uri = validate_image_digest_uri(tags["citrees-image-uri"])
    campaign_sha256 = validate_manifest_sha256(tags["citrees-campaign-sha256"])
    manifest_sha256 = validate_manifest_sha256(tags["citrees-manifest-sha256"])
    manifest_key = manifest_s3_key(manifest_sha256)
    if tags["citrees-manifest-key"] != manifest_key:
        raise RuntimeError(
            f"API server manifest key is not content-addressed: {tags['citrees-manifest-key']!r}"
        )
    try:
        max_cell_attempts = int(tags["citrees-max-cell-attempts"])
    except ValueError as exc:
        raise RuntimeError("API server max-cell-attempts tag is invalid") from exc
    if max_cell_attempts <= 0 or str(max_cell_attempts) != tags["citrees-max-cell-attempts"]:
        raise RuntimeError("API server max-cell-attempts tag is invalid")
    return ApiScope(
        api_url=f"http://{private_ip}:8000",
        public_api_url=f"http://{public_ip}:8000",
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        image_uri=image_uri,
        manifest_s3_key=manifest_key,
        manifest_sha256=manifest_sha256,
        max_cell_attempts=max_cell_attempts,
        stage=stage,
    )


def get_api_url(region: str = DEFAULT_REGION) -> str | None:
    """Return the private URL of the single running API server."""
    scope = get_api_scope(region)
    return scope.api_url if scope is not None else None


def terminate_api(region: str = DEFAULT_REGION) -> str | None:
    """Terminate the API server instance.

    Returns the terminated instance ID, or None if no API instance found.
    """
    ec2 = boto3.client("ec2", region_name=region)

    response = ec2.describe_instances(
        Filters=[
            {"Name": f"tag:{TAG_KEY}", "Values": [API_TAG_VALUE]},
            {
                "Name": "instance-state-name",
                "Values": ["pending", "running", "stopping"],
            },
        ]
    )

    instance_ids = []
    for reservation in response.get("Reservations", []):
        for inst in reservation.get("Instances", []):
            instance_ids.append(inst["InstanceId"])

    if not instance_ids:
        info("No API server instance found")
        return None

    ec2.terminate_instances(InstanceIds=instance_ids)
    terminated = instance_ids[0]
    success(f"Terminated API server: {terminated}")

    return terminated


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


def launch_workers(
    n: int,
    instance_type: str,
    image_uri: str,
    *,
    artifact_prefix: str,
    manifest_path: Path,
    stage: str,
    spot: bool = False,
    region: str = DEFAULT_REGION,
) -> list[str]:
    """Launch N EC2 worker instances.

    The API server's private IP and immutable scope are auto-discovered.
    Each instance runs a Docker container that pulls configs from the API
    server and executes them. Workers get their stage/task assignment from
    the server via POST /next.

    Parameters
    ----------
    n : int
        Number of instances to launch.
    instance_type : str
        EC2 instance type (e.g., "m5.8xlarge").
    image_uri : str
        Full ECR image URI.
    spot : bool
        Use spot instances instead of on-demand.
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
    image_uri = validate_image_digest_uri(image_uri)
    artifact_prefix, stage = _validate_queue_scope(artifact_prefix, stage)
    manifest_info = publish_rerun_manifest(manifest_path, region=region)
    manifest_sha256 = str(manifest_info["sha256"])
    campaign_sha256 = str(manifest_info["campaign_sha256"])
    api_scope = get_api_scope(region)
    if api_scope is None:
        raise RuntimeError(
            "No running API server found. Launch one with: citrees-exp infra launch-api"
        )
    expected_scope = {
        "artifact_prefix": artifact_prefix,
        "campaign_sha256": campaign_sha256,
        "image_uri": image_uri,
        "manifest_sha256": manifest_sha256,
        "stage": stage,
    }
    observed_scope = {
        "artifact_prefix": api_scope.artifact_prefix,
        "campaign_sha256": api_scope.campaign_sha256,
        "image_uri": api_scope.image_uri,
        "manifest_sha256": api_scope.manifest_sha256,
        "stage": api_scope.stage,
    }
    if observed_scope != expected_scope:
        raise RuntimeError(
            f"Worker scope does not match running API: expected {expected_scope}, "
            f"observed {observed_scope}"
        )
    git_sha = validate_image_revision(image_uri, region=region)
    sg_id = ensure_security_group(region)
    instance_profile_name = ensure_campaign_iam_profile(
        output_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        write_prefixes=(artifact_prefix,),
        region=region,
    )
    _wait_for_api_ready(
        api_scope.public_api_url,
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        manifest_sha256=manifest_sha256,
        max_cell_attempts=api_scope.max_cell_attempts,
        stage=stage,
    )
    api_url = api_scope.api_url

    ec2 = boto3.client("ec2", region_name=region)
    account_id = get_aws_account_id()
    bucket = get_resource_name(account_id)
    ecr_uri = image_uri.split("/")[0]
    ami_id = get_ami(region)

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
        manifest_sha256=manifest_sha256,
        stage=stage,
    )

    pricing = "spot" if spot else "on-demand"
    info(f"Launching {n} {pricing} workers: {instance_type}, AMI={ami_id}")
    step(f"API: {api_url}")
    step(f"Image: {image_uri}")
    step(f"Security group: {sg_id}")
    step(f"Instance profile: {instance_profile_name}")

    run_kwargs: dict[str, Any] = {
        "ImageId": ami_id,
        "InstanceType": instance_type,
        "MinCount": 1,
        "MaxCount": n,
        "IamInstanceProfile": {"Name": instance_profile_name},
        "UserData": base64.b64encode(user_data.encode()).decode(),
        "MetadataOptions": {"HttpPutResponseHopLimit": 2},
        "SecurityGroupIds": [sg_id],
        "InstanceInitiatedShutdownBehavior": "terminate",
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": TAG_KEY, "Value": WORKER_TAG_VALUE},
                    {"Key": "Name", "Value": "citrees-worker"},
                    {"Key": "citrees-artifact-prefix", "Value": artifact_prefix},
                    {"Key": "citrees-campaign-sha256", "Value": campaign_sha256},
                    {"Key": "citrees-manifest-sha256", "Value": manifest_sha256},
                    {"Key": "citrees-image-uri", "Value": image_uri},
                    {"Key": "citrees-stage", "Value": stage},
                ],
            }
        ],
    }

    if spot:
        run_kwargs["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        }

    response = ec2.run_instances(**run_kwargs)

    instance_ids = [inst["InstanceId"] for inst in response["Instances"]]
    success(f"Launched {len(instance_ids)} worker instances")
    for iid in instance_ids:
        step(f"  {iid}")

    return instance_ids


def launch_mechanism_workers(
    n: int,
    instance_type: str,
    image_uri: str,
    *,
    spot: bool = False,
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
    git_sha = validate_image_revision(image_uri, region=region)
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
        write_prefixes=(output_prefix,),
        region=region,
    )

    pricing = "spot" if spot else "on-demand"
    shard_end = shard_start + n - 1
    info(f"Launching {n} {pricing} mechanism workers: {instance_type}, AMI={ami_id}")
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
            "MetadataOptions": {"HttpPutResponseHopLimit": 2},
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
                        {"Key": "citrees-image-uri", "Value": image_uri},
                    ],
                }
            ],
        }
        if placement_subnets:
            run_kwargs["SubnetId"] = placement_subnets[
                (shard_index - shard_start) % len(placement_subnets)
            ]

        if spot:
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


def list_workers(region: str = DEFAULT_REGION) -> list[dict[str, str]]:
    """List running citrees worker instances.

    Returns a list of dicts with keys: instance_id, state, instance_type, launch_time.
    """
    ec2 = boto3.client("ec2", region_name=region)

    response = ec2.describe_instances(
        Filters=[
            {"Name": f"tag:{TAG_KEY}", "Values": [WORKER_TAG_VALUE]},
            {
                "Name": "instance-state-name",
                "Values": ["pending", "running", "stopping"],
            },
        ]
    )

    workers = []
    for reservation in response.get("Reservations", []):
        for inst in reservation.get("Instances", []):
            workers.append(
                {
                    "instance_id": inst["InstanceId"],
                    "state": inst["State"]["Name"],
                    "instance_type": inst.get("InstanceType", ""),
                    "launch_time": (
                        inst.get("LaunchTime", "").isoformat() if inst.get("LaunchTime") else ""
                    ),
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


def terminate_workers(region: str = DEFAULT_REGION) -> list[str]:
    """Terminate all running citrees worker instances.

    Returns list of terminated instance IDs.
    """
    ec2 = boto3.client("ec2", region_name=region)

    workers = list_workers(region)
    if not workers:
        info("No worker instances found")
        return []

    instance_ids = [w["instance_id"] for w in workers]
    info(f"Terminating {len(instance_ids)} worker instances...")

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
