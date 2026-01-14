"""EC2 compute management for citrees infrastructure."""

from __future__ import annotations

import base64
import textwrap

import boto3
from rich.console import Console

from .config import Config

console = Console()


# =============================================================================
# User Data Scripts
# =============================================================================


def get_server_user_data(config: Config) -> str:
    """Generate user data script for server instance.

    Parameters
    ----------
    config : Config
        Infrastructure configuration.

    Returns
    -------
    str
        Bash script for EC2 user data.
    """
    ecr_registry = f"{config.account_id}.dkr.ecr.{config.region}.amazonaws.com"

    script = textwrap.dedent(f"""\
        #!/bin/bash
        set -ex
        exec > >(tee /var/log/user-data.log) 2>&1

        echo "=== citrees server setup ==="
        echo "Time: $(date)"

        # Install Docker
        yum update -y
        yum install -y docker
        systemctl start docker
        systemctl enable docker

        # ECR login
        echo "=== Logging into ECR ==="
        aws ecr get-login-password --region {config.region} | \\
            docker login --username AWS --password-stdin {ecr_registry}

        # Pull image
        echo "=== Pulling image ==="
        docker pull {config.ecr_image_uri}

        # Run server container
        echo "=== Starting server container ==="
        docker run -d \\
            --name citrees-server \\
            --restart unless-stopped \\
            -p 8000:8000 \\
            -e TABLE_NAME={config.table_name} \\
            -e S3_BUCKET={config.bucket_name} \\
            -e AWS_DEFAULT_REGION={config.region} \\
            {config.ecr_image_uri} server

        echo "=== Server started on port 8000 ==="
        echo "Time: $(date)"
    """)
    return script


def get_worker_user_data(config: Config, server_private_ip: str) -> str:
    """Generate user data script for worker instance.

    Parameters
    ----------
    config : Config
        Infrastructure configuration.
    server_private_ip : str
        Private IP address of the server.

    Returns
    -------
    str
        Bash script for EC2 user data.
    """
    ecr_registry = f"{config.account_id}.dkr.ecr.{config.region}.amazonaws.com"
    server_url = f"http://{server_private_ip}:8000"

    script = textwrap.dedent(f"""\
        #!/bin/bash
        set -ex
        exec > >(tee /var/log/user-data.log) 2>&1

        echo "=== citrees worker setup ==="
        echo "Time: $(date)"

        # Install Docker
        yum update -y
        yum install -y docker
        systemctl start docker
        systemctl enable docker

        # ECR login
        echo "=== Logging into ECR ==="
        aws ecr get-login-password --region {config.region} | \\
            docker login --username AWS --password-stdin {ecr_registry}

        # Pull image
        echo "=== Pulling image ==="
        docker pull {config.ecr_image_uri}

        # Run worker container
        echo "=== Starting worker container ==="
        docker run -d \\
            --name citrees-worker \\
            --restart unless-stopped \\
            -e URL={server_url} \\
            -e TABLE_NAME={config.table_name} \\
            -e S3_BUCKET={config.bucket_name} \\
            -e AWS_DEFAULT_REGION={config.region} \\
            {config.ecr_image_uri} worker

        echo "=== Worker started, connecting to {server_url} ==="
        echo "Time: $(date)"
    """)
    return script


# =============================================================================
# EC2 Instance Management
# =============================================================================


def launch_server(config: Config, ami_id: str) -> tuple[str, str]:
    """Launch server EC2 instance.

    Parameters
    ----------
    config : Config
        Infrastructure configuration.
    ami_id : str
        AMI ID to use.

    Returns
    -------
    tuple[str, str]
        Instance ID and private IP address.
    """
    ec2 = boto3.client("ec2", region_name=config.region)

    user_data = get_server_user_data(config)
    user_data_b64 = base64.b64encode(user_data.encode()).decode()

    params = {
        "ImageId": ami_id,
        "InstanceType": config.server.instance_type,
        "MinCount": 1,
        "MaxCount": 1,
        "SubnetId": config._resolved_private_subnet_id,
        "SecurityGroupIds": [config._resolved_security_group_id],
        "IamInstanceProfile": {"Name": config.iam_instance_profile_name},
        "UserData": user_data_b64,
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": config.server_name},
                    {"Key": "Project", "Value": config.project_name},
                    {"Key": "Role", "Value": "server"},
                ],
            }
        ],
    }

    # Spot instance (not recommended for server, but supported)
    if config.server.spot:
        params["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        }

    response = ec2.run_instances(**params)
    instance = response["Instances"][0]
    instance_id = instance["InstanceId"]

    # Wait for instance to be running
    console.print("  Waiting for server to start...")
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])

    # Get private IP
    response = ec2.describe_instances(InstanceIds=[instance_id])
    private_ip = response["Reservations"][0]["Instances"][0]["PrivateIpAddress"]

    config._server_instance_id = instance_id
    config._server_private_ip = private_ip

    return instance_id, private_ip


def launch_workers(config: Config, ami_id: str) -> list[str]:
    """Launch worker EC2 instances.

    Parameters
    ----------
    config : Config
        Infrastructure configuration.
    ami_id : str
        AMI ID to use.

    Returns
    -------
    list[str]
        List of instance IDs.
    """
    if not config._server_private_ip:
        raise RuntimeError("Server must be launched before workers")

    ec2 = boto3.client("ec2", region_name=config.region)

    user_data = get_worker_user_data(config, config._server_private_ip)
    user_data_b64 = base64.b64encode(user_data.encode()).decode()

    params = {
        "ImageId": ami_id,
        "InstanceType": config.workers.instance_type,
        "MinCount": config.workers.count,
        "MaxCount": config.workers.count,
        "SubnetId": config._resolved_private_subnet_id,
        "SecurityGroupIds": [config._resolved_security_group_id],
        "IamInstanceProfile": {"Name": config.iam_instance_profile_name},
        "UserData": user_data_b64,
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": config.worker_name},
                    {"Key": "Project", "Value": config.project_name},
                    {"Key": "Role", "Value": "worker"},
                ],
            }
        ],
    }

    # Spot instances
    if config.workers.spot:
        params["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        }

    response = ec2.run_instances(**params)
    instance_ids = [inst["InstanceId"] for inst in response["Instances"]]

    # Wait for instances to be running
    console.print(f"  Waiting for {len(instance_ids)} worker(s) to start...")
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=instance_ids)

    config._worker_instance_ids = instance_ids

    return instance_ids


def get_instance_status(config: Config) -> dict:
    """Get status of all instances.

    Parameters
    ----------
    config : Config
        Infrastructure configuration.

    Returns
    -------
    dict
        Status information for server and workers.
    """
    ec2 = boto3.client("ec2", region_name=config.region)

    instance_ids = []
    if config._server_instance_id:
        instance_ids.append(config._server_instance_id)
    instance_ids.extend(config._worker_instance_ids)

    if not instance_ids:
        return {"server": None, "workers": []}

    response = ec2.describe_instances(InstanceIds=instance_ids)

    result = {"server": None, "workers": []}

    for reservation in response["Reservations"]:
        for instance in reservation["Instances"]:
            instance_id = instance["InstanceId"]
            state = instance["State"]["Name"]
            private_ip = instance.get("PrivateIpAddress", "N/A")

            # Get role from tags
            role = "unknown"
            for tag in instance.get("Tags", []):
                if tag["Key"] == "Role":
                    role = tag["Value"]
                    break

            info = {
                "instance_id": instance_id,
                "state": state,
                "private_ip": private_ip,
                "instance_type": instance["InstanceType"],
            }

            if role == "server":
                result["server"] = info
            else:
                result["workers"].append(info)

    return result


def find_instances_by_tag(config: Config) -> dict:
    """Find instances by project tag (for recovery).

    Parameters
    ----------
    config : Config
        Infrastructure configuration.

    Returns
    -------
    dict
        Status information for server and workers.
    """
    ec2 = boto3.client("ec2", region_name=config.region)

    response = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Project", "Values": [config.project_name]},
            {"Name": "instance-state-name", "Values": ["pending", "running"]},
        ]
    )

    result = {"server": None, "workers": []}

    for reservation in response["Reservations"]:
        for instance in reservation["Instances"]:
            instance_id = instance["InstanceId"]
            state = instance["State"]["Name"]
            private_ip = instance.get("PrivateIpAddress", "N/A")

            # Get role from tags
            role = "unknown"
            for tag in instance.get("Tags", []):
                if tag["Key"] == "Role":
                    role = tag["Value"]
                    break

            info = {
                "instance_id": instance_id,
                "state": state,
                "private_ip": private_ip,
                "instance_type": instance["InstanceType"],
            }

            if role == "server":
                result["server"] = info
                config._server_instance_id = instance_id
                config._server_private_ip = private_ip
            else:
                result["workers"].append(info)
                if instance_id not in config._worker_instance_ids:
                    config._worker_instance_ids.append(instance_id)

    return result


def stop_instances(config: Config) -> None:
    """Stop all instances (without terminating).

    Parameters
    ----------
    config : Config
        Infrastructure configuration.
    """
    ec2 = boto3.client("ec2", region_name=config.region)

    instance_ids = []
    if config._server_instance_id:
        instance_ids.append(config._server_instance_id)
    instance_ids.extend(config._worker_instance_ids)

    if instance_ids:
        ec2.stop_instances(InstanceIds=instance_ids)
        console.print(f"  Stopped {len(instance_ids)} instance(s)")


def terminate_instances(config: Config) -> None:
    """Terminate all instances.

    Parameters
    ----------
    config : Config
        Infrastructure configuration.
    """
    ec2 = boto3.client("ec2", region_name=config.region)

    instance_ids = []
    if config._server_instance_id:
        instance_ids.append(config._server_instance_id)
    instance_ids.extend(config._worker_instance_ids)

    if instance_ids:
        ec2.terminate_instances(InstanceIds=instance_ids)
        console.print(f"  Terminating {len(instance_ids)} instance(s)...")

        # Wait for termination
        waiter = ec2.get_waiter("instance_terminated")
        waiter.wait(InstanceIds=instance_ids)

        config._server_instance_id = None
        config._server_private_ip = None
        config._worker_instance_ids = []
