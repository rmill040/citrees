#!/usr/bin/env python3
"""Setup and validate Ray cluster configuration.

Usage:
    # Generate cluster.yaml from template (uses current IP and branch)
    AWS_PROFILE=personal uv run python paper/scripts/infra/ray/setup_cluster.py --generate

    # Generate with specific branch
    AWS_PROFILE=personal uv run python paper/scripts/infra/ray/setup_cluster.py --generate --branch main

    # Update cluster.yaml with latest Ubuntu AMI
    AWS_PROFILE=personal uv run python paper/scripts/infra/ray/setup_cluster.py --update-ami

    # Validate cluster config
    AWS_PROFILE=personal uv run python paper/scripts/infra/ray/setup_cluster.py --validate

    # Show current config
    AWS_PROFILE=personal uv run python paper/scripts/infra/ray/setup_cluster.py --show
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

import boto3
import yaml

CLUSTER_YAML = Path(__file__).parent / "cluster.yaml"
CLUSTER_EXAMPLE_YAML = Path(__file__).parent / "cluster.example.yaml"
UBUNTU_OWNER_ID = "099720109477"  # Canonical
DEFAULT_REGION = "us-east-1"


def get_public_ip() -> str:
    """Get the user's current public IP address."""
    with urllib.request.urlopen("https://ifconfig.me", timeout=10) as response:
        return response.read().decode("utf-8").strip()


def ensure_s3_bucket(region: str = DEFAULT_REGION) -> str:
    """Ensure the S3 bucket for results exists, create if not.

    Returns the bucket name.
    """
    # Import config here to avoid circular imports
    from paper.scripts.infra.config import load_config

    config = load_config()
    bucket_name = config.bucket_name
    s3 = boto3.client("s3", region_name=region)

    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"  S3 bucket exists: {bucket_name}")
    except s3.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code in ("404", "NoSuchBucket"):
            print(f"  Creating S3 bucket: {bucket_name}")
            # us-east-1 doesn't need LocationConstraint
            if region == "us-east-1":
                s3.create_bucket(Bucket=bucket_name)
            else:
                s3.create_bucket(
                    Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": region}
                )
            print(f"  Created S3 bucket: {bucket_name}")
        else:
            raise

    return bucket_name


def get_current_branch() -> str:
    """Get the current git branch name."""
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def generate_config(branch: str | None = None) -> None:
    """Generate cluster.yaml from template with user's current IP and branch."""
    if not CLUSTER_EXAMPLE_YAML.exists():
        raise FileNotFoundError(f"Template not found: {CLUSTER_EXAMPLE_YAML}")

    print("Fetching your public IP...")
    ip = get_public_ip()
    print(f"  Your IP: {ip}")

    if branch is None:
        branch = get_current_branch()
    print(f"  Branch: {branch}")

    print("\nEnsuring S3 bucket for results...")
    bucket_name = ensure_s3_bucket()

    print(f"\nReading template: {CLUSTER_EXAMPLE_YAML}")
    template = CLUSTER_EXAMPLE_YAML.read_text()

    print("Replacing placeholders...")
    config_content = (
        template.replace("__MY_IP__", ip)
        .replace("__BRANCH__", branch)
        .replace("__S3_BUCKET__", bucket_name)
    )

    print(f"Writing: {CLUSTER_YAML}")
    CLUSTER_YAML.write_text(config_content)

    print(f"\nGenerated {CLUSTER_YAML}")
    print("\nNext steps:")
    print(f"  1. Push branch: git push -u origin {branch}")
    print(
        "  2. Deploy: AWS_PROFILE=personal uv run ray up paper/scripts/infra/ray/cluster.yaml --yes"
    )


def get_latest_ubuntu_ami(region: str) -> dict[str, str]:
    """Fetch latest Ubuntu 22.04 AMI for the given region."""
    ec2 = boto3.client("ec2", region_name=region)
    response = ec2.describe_images(
        Owners=[UBUNTU_OWNER_ID],
        Filters=[
            {"Name": "name", "Values": ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]},
            {"Name": "state", "Values": ["available"]},
        ],
    )
    images = sorted(response["Images"], key=lambda x: x["CreationDate"], reverse=True)
    if not images:
        raise RuntimeError(f"No Ubuntu 22.04 AMI found in {region}")
    latest = images[0]
    return {
        "id": latest["ImageId"],
        "name": latest["Name"],
        "created": latest["CreationDate"],
    }


def load_cluster_config() -> dict:
    """Load cluster.yaml."""
    with open(CLUSTER_YAML) as f:
        return yaml.safe_load(f)


def save_cluster_config(config: dict) -> None:
    """Save cluster.yaml."""
    with open(CLUSTER_YAML, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def update_ami(config: dict, ami_id: str) -> dict:
    """Update all node types with the given AMI ID."""
    for node_type, node_config in config.get("available_node_types", {}).items():
        if "node_config" in node_config and "ImageId" in node_config["node_config"]:
            node_config["node_config"]["ImageId"] = ami_id
    return config


def validate_config(config: dict) -> list[str]:
    """Validate cluster configuration. Returns list of errors."""
    errors = []
    region = config.get("provider", {}).get("region")
    if not region:
        errors.append("Missing provider.region")
        return errors

    ec2 = boto3.client("ec2", region_name=region)

    for node_type, node_config in config.get("available_node_types", {}).items():
        nc = node_config.get("node_config", {})
        ami_id = nc.get("ImageId")
        instance_type = nc.get("InstanceType")

        if ami_id:
            try:
                response = ec2.describe_images(ImageIds=[ami_id])
                if not response["Images"]:
                    errors.append(f"{node_type}: AMI {ami_id} not found")
            except Exception as e:
                errors.append(f"{node_type}: AMI validation failed - {e}")

        if instance_type:
            try:
                response = ec2.describe_instance_types(InstanceTypes=[instance_type])
                if not response["InstanceTypes"]:
                    errors.append(f"{node_type}: Instance type {instance_type} not found")
            except Exception as e:
                errors.append(f"{node_type}: Instance type validation failed - {e}")

    return errors


def show_config(config: dict) -> None:
    """Display current cluster configuration."""
    region = config.get("provider", {}).get("region", "unknown")
    print(f"Cluster: {config.get('cluster_name', 'unknown')}")
    print(f"Region: {region}")
    print("\nNode Types:")
    for node_type, node_config in config.get("available_node_types", {}).items():
        nc = node_config.get("node_config", {})
        print(f"  {node_type}:")
        print(f"    Instance: {nc.get('InstanceType', 'N/A')}")
        print(f"    AMI: {nc.get('ImageId', 'N/A')}")
        if "InstanceMarketOptions" in nc:
            print("    Spot: Yes")
        if "resources" in node_config:
            print(f"    Resources: {node_config['resources']}")
        print(
            f"    Min/Max: {node_config.get('min_workers', 'N/A')}/{node_config.get('max_workers', 'N/A')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup and validate Ray cluster config")
    parser.add_argument(
        "--generate", action="store_true", help="Generate cluster.yaml from template with your IP"
    )
    parser.add_argument(
        "--branch", type=str, default=None, help="Git branch to use (default: current branch)"
    )
    parser.add_argument(
        "--update-ami", action="store_true", help="Update AMI to latest Ubuntu 22.04"
    )
    parser.add_argument("--validate", action="store_true", help="Validate cluster configuration")
    parser.add_argument("--show", action="store_true", help="Show current configuration")
    args = parser.parse_args()

    if not any([args.generate, args.update_ami, args.validate, args.show]):
        parser.print_help()
        return 1

    # Generate doesn't need existing cluster.yaml
    if args.generate:
        generate_config(branch=args.branch)
        return 0

    config = load_cluster_config()
    region = config.get("provider", {}).get("region", "us-east-1")

    if args.show:
        show_config(config)

    if args.update_ami:
        print(f"\nFetching latest Ubuntu 22.04 AMI for {region}...")
        ami = get_latest_ubuntu_ami(region)
        print(f"  ID: {ami['id']}")
        print(f"  Name: {ami['name']}")
        print(f"  Created: {ami['created']}")
        config = update_ami(config, ami["id"])
        save_cluster_config(config)
        print(f"\nUpdated {CLUSTER_YAML}")

    if args.validate:
        print("\nValidating cluster configuration...")
        errors = validate_config(config)
        if errors:
            print("Errors found:")
            for err in errors:
                print(f"  - {err}")
            return 1
        print("Configuration valid.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
