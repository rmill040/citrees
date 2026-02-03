"""EC2 instance management for API server and workers.

Launch, list, and terminate EC2 instances that run the experiment
API server and workers via Docker containers pulled from ECR.
"""

from __future__ import annotations

import base64
import textwrap
import time
from typing import Any

import boto3

from paper.scripts.cli._console import info, step, success, warn
from paper.scripts.infra.aws import (
    DEFAULT_REGION,
    IAM_ROLE_NAME,
    ensure_security_group,
    get_aws_account_id,
    get_git_sha,
    get_resource_name,
)

TAG_KEY = "citrees-role"
API_TAG_VALUE = "api"
WORKER_TAG_VALUE = "worker"
LOG_GROUP_API = "/citrees/api"
LOG_GROUP_WORKER = "/citrees/worker"


def _get_ami(region: str) -> str:
    """Get the latest Amazon Linux 2023 AMI ID."""
    ssm = boto3.client("ssm", region_name=region)
    ami_param = ssm.get_parameter(
        Name="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
    )
    return ami_param["Parameter"]["Value"]


def _make_worker_user_data(
    *,
    region: str,
    ecr_uri: str,
    image_uri: str,
    api_url: str,
    bucket: str,
    git_sha: str,
) -> str:
    """Generate EC2 user data script that pulls and runs the worker container."""
    return textwrap.dedent(f"""\
        #!/bin/bash
        exec > >(tee /var/log/user-data.log) 2>&1
        set -euo pipefail

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
        docker run --rm \\
            --log-driver=awslogs \\
            --log-opt awslogs-region={region} \\
            --log-opt awslogs-group={LOG_GROUP_WORKER} \\
            --log-opt awslogs-stream=$INSTANCE_ID \\
            -e S3_BUCKET={bucket} \\
            -e CITREES_API_URL={api_url} \\
            -e AWS_DEFAULT_REGION={region} \\
            -e GIT_SHA={git_sha} \\
            {image_uri} \\
            python -m paper.scripts.api.worker \\
                --api-url {api_url} \\
                --max-idle-polls 60

        # Container exited (queue drained) — terminate this instance
        echo "Worker container exited, shutting down instance"
        shutdown -h now
    """)


def _make_api_user_data(
    *,
    region: str,
    ecr_uri: str,
    image_uri: str,
    bucket: str,
    git_sha: str,
) -> str:
    """Generate EC2 user data script that runs the API server container."""
    return textwrap.dedent(f"""\
        #!/bin/bash
        exec > >(tee /var/log/user-data.log) 2>&1
        set -euo pipefail

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
        docker run -d --restart unless-stopped \\
            -p 8000:8000 \\
            --log-driver=awslogs \\
            --log-opt awslogs-region={region} \\
            --log-opt awslogs-group={LOG_GROUP_API} \\
            --log-opt awslogs-stream=$INSTANCE_ID \\
            -e S3_BUCKET={bucket} \\
            -e AWS_DEFAULT_REGION={region} \\
            -e GIT_SHA={git_sha} \\
            {image_uri} \\
            uvicorn paper.scripts.api.server:app --host 0.0.0.0 --port 8000
    """)


# ---------------------------------------------------------------------------
# API server
# ---------------------------------------------------------------------------


def launch_api(
    instance_type: str,
    image_uri: str,
    *,
    region: str = DEFAULT_REGION,
) -> dict[str, str]:
    """Launch the API server on a single EC2 instance.

    Returns dict with instance_id, public_ip, and api_url.
    """
    ec2 = boto3.client("ec2", region_name=region)
    account_id = get_aws_account_id()
    bucket = get_resource_name(account_id)
    ecr_uri = image_uri.split("/")[0]
    sg_id = ensure_security_group(region)
    ami_id = _get_ami(region)

    git_sha = get_git_sha()

    user_data = _make_api_user_data(
        region=region,
        ecr_uri=ecr_uri,
        image_uri=image_uri,
        bucket=bucket,
        git_sha=git_sha,
    )

    info(f"Launching API server: {instance_type}, AMI={ami_id}")
    step(f"Image: {image_uri}")
    step(f"Security group: {sg_id}")

    response = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        MinCount=1,
        MaxCount=1,
        IamInstanceProfile={"Name": IAM_ROLE_NAME},
        UserData=base64.b64encode(user_data.encode()).decode(),
        MetadataOptions={"HttpPutResponseHopLimit": 2},
        SecurityGroupIds=[sg_id],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": TAG_KEY, "Value": API_TAG_VALUE},
                    {"Key": "Name", "Value": "citrees-api"},
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
        return {"instance_id": instance_id, "public_ip": "", "api_url": ""}

    api_url = f"http://{public_ip}:8000"
    success(f"API server launching at {api_url}")
    step("Server will be ready once Docker pull + uvicorn startup completes")

    return {"instance_id": instance_id, "public_ip": public_ip, "api_url": api_url}


def get_api_url(region: str = DEFAULT_REGION) -> str | None:
    """Find the running API server instance and return its private-IP URL.

    Workers must use the private IP because security-group self-referencing
    rules do not cover traffic hairpinned through the Internet Gateway.

    Returns http://{private_ip}:8000 or None if no API instance is running.
    """
    ec2 = boto3.client("ec2", region_name=region)

    response = ec2.describe_instances(
        Filters=[
            {"Name": f"tag:{TAG_KEY}", "Values": [API_TAG_VALUE]},
            {"Name": "instance-state-name", "Values": ["pending", "running"]},
        ]
    )

    for reservation in response.get("Reservations", []):
        for inst in reservation.get("Instances", []):
            ip = inst.get("PrivateIpAddress")
            if ip:
                return f"http://{ip}:8000"

    return None


def terminate_api(region: str = DEFAULT_REGION) -> str | None:
    """Terminate the API server instance.

    Returns the terminated instance ID, or None if no API instance found.
    """
    ec2 = boto3.client("ec2", region_name=region)

    response = ec2.describe_instances(
        Filters=[
            {"Name": f"tag:{TAG_KEY}", "Values": [API_TAG_VALUE]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping"]},
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
    spot: bool = False,
    region: str = DEFAULT_REGION,
) -> list[str]:
    """Launch N EC2 worker instances.

    The API server's private IP is auto-discovered via ``get_api_url()``.
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
    api_url = get_api_url(region)
    if not api_url:
        raise RuntimeError(
            "No running API server found. Launch one with: citrees-exp infra launch-api"
        )

    ec2 = boto3.client("ec2", region_name=region)
    account_id = get_aws_account_id()
    bucket = get_resource_name(account_id)
    ecr_uri = image_uri.split("/")[0]
    sg_id = ensure_security_group(region)
    ami_id = _get_ami(region)

    git_sha = get_git_sha()

    user_data = _make_worker_user_data(
        region=region,
        ecr_uri=ecr_uri,
        image_uri=image_uri,
        api_url=api_url,
        bucket=bucket,
        git_sha=git_sha,
    )

    pricing = "spot" if spot else "on-demand"
    info(f"Launching {n} {pricing} workers: {instance_type}, AMI={ami_id}")
    step(f"API: {api_url}")
    step(f"Image: {image_uri}")
    step(f"Security group: {sg_id}")

    run_kwargs: dict[str, Any] = {
        "ImageId": ami_id,
        "InstanceType": instance_type,
        "MinCount": n,
        "MaxCount": n,
        "IamInstanceProfile": {"Name": IAM_ROLE_NAME},
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


def list_workers(region: str = DEFAULT_REGION) -> list[dict[str, str]]:
    """List running citrees worker instances.

    Returns a list of dicts with keys: instance_id, state, instance_type, launch_time.
    """
    ec2 = boto3.client("ec2", region_name=region)

    response = ec2.describe_instances(
        Filters=[
            {"Name": f"tag:{TAG_KEY}", "Values": [WORKER_TAG_VALUE]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping"]},
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
                    "launch_time": inst.get("LaunchTime", "").isoformat()
                    if inst.get("LaunchTime")
                    else "",
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
        "api" or "worker".
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
    log_group = LOG_GROUP_API if role == "api" else LOG_GROUP_WORKER
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
