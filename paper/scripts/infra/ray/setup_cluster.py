#!/usr/bin/env python3
"""Setup and validate Ray cluster configuration.

Usage (via CLI):
    citrees-exp infra setup           # Full setup: IAM + Docker + cluster.yaml
    citrees-exp infra build           # Build and push Docker image to ECR
    citrees-exp infra generate        # Generate cluster.yaml from template
    citrees-exp infra validate        # Validate cluster config
    citrees-exp infra show            # Show current config
    citrees-exp infra update-ami      # Update AMI in cluster.yaml
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import boto3
import yaml

from paper.scripts.cli._console import error, heading, info, step, success

CLUSTER_YAML = Path(__file__).parent / "cluster.yaml"
CLUSTER_EXAMPLE_YAML = Path(__file__).parent / "cluster.example.yaml"
DOCKERFILE_DIR = Path(__file__).parent.parent / "docker"
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
UBUNTU_OWNER_ID = "099720109477"  # Canonical
DEFAULT_REGION = "us-east-1"
RESOURCE_PREFIX = "citrees"  # All resources: citrees-{account_id}
IAM_ROLE_NAME = "citrees-ray-autoscaler"
DOCKER_PLATFORM = "linux/amd64"  # AWS EC2 instances are amd64


def get_public_ip() -> str:
    """Get the user's current public IP address."""
    with urllib.request.urlopen("https://ifconfig.me", timeout=10) as response:
        return response.read().decode("utf-8").strip()


def ensure_s3_bucket(region: str = DEFAULT_REGION) -> str:
    """Ensure the S3 bucket for results exists, create if not.

    Bucket name follows pattern: citrees-{account_id}

    Returns the bucket name.
    """
    account_id = get_aws_account_id()
    bucket_name = get_resource_name(account_id)
    s3 = boto3.client("s3", region_name=region)

    try:
        s3.head_bucket(Bucket=bucket_name)
        step(f"S3 bucket exists: {bucket_name}")
    except s3.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code in ("404", "NoSuchBucket"):
            step(f"Creating S3 bucket: {bucket_name}")
            # us-east-1 doesn't need LocationConstraint
            if region == "us-east-1":
                s3.create_bucket(Bucket=bucket_name)
            else:
                s3.create_bucket(
                    Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": region}
                )
            success(f"Created S3 bucket: {bucket_name}")
        else:
            raise

    return bucket_name


def get_aws_account_id() -> str:
    """Get the current AWS account ID."""
    sts = boto3.client("sts")
    return sts.get_caller_identity()["Account"]


def get_git_sha() -> str:
    """Get the current git commit SHA."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    return result.stdout.strip()


def get_resource_name(account_id: str) -> str:
    """Get the standardized resource name: citrees-{account_id}.

    Used for both S3 bucket and ECR repository.
    """
    return f"{RESOURCE_PREFIX}-{account_id}"


def ensure_ecr_repo(region: str = DEFAULT_REGION) -> tuple[str, str]:
    """Ensure the ECR repository exists, create if not.

    Returns tuple of (repo_name, repo_uri).
    """
    account_id = get_aws_account_id()
    repo_name = get_resource_name(account_id)
    ecr = boto3.client("ecr", region_name=region)

    try:
        response = ecr.describe_repositories(repositoryNames=[repo_name])
        repo_uri = response["repositories"][0]["repositoryUri"]
        step(f"ECR repository exists: {repo_name}")
    except ecr.exceptions.RepositoryNotFoundException:
        step(f"Creating ECR repository: {repo_name}")
        response = ecr.create_repository(
            repositoryName=repo_name,
            imageScanningConfiguration={"scanOnPush": True},
            imageTagMutability="MUTABLE",
        )
        repo_uri = response["repository"]["repositoryUri"]
        success(f"Created ECR repository: {repo_uri}")

    return repo_name, repo_uri


def ensure_iam_role(region: str = DEFAULT_REGION) -> str:
    """Ensure the IAM role and instance profile exist with necessary permissions.

    Creates:
    - IAM role: citrees-ray-autoscaler
    - Instance profile: citrees-ray-autoscaler
    - Attached policies: EC2, S3, ECR permissions

    Returns the instance profile ARN.
    """
    iam = boto3.client("iam", region_name=region)
    account_id = get_aws_account_id()
    instance_profile_arn = f"arn:aws:iam::{account_id}:instance-profile/{IAM_ROLE_NAME}"

    # Trust policy for EC2
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    # Inline policy for Ray autoscaler (EC2 + S3 + ECR).
    #
    # This is intentionally broader than a minimal policy: Ray's autoscaler can
    # create/delete security groups and keypairs, and spot instances require
    # additional EC2 actions. If you want to lock this down further, start from
    # Ray's AWS policy docs and prune only after a full end-to-end run works.
    ray_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "EC2Autoscaler",
                "Effect": "Allow",
                "Action": [
                    # Discovery
                    "ec2:DescribeAvailabilityZones",
                    "ec2:DescribeInstances",
                    "ec2:DescribeInstanceTypes",
                    "ec2:DescribeRegions",
                    "ec2:DescribeSecurityGroups",
                    "ec2:DescribeSubnets",
                    "ec2:DescribeVpcs",
                    "ec2:DescribeKeyPairs",
                    "ec2:DescribeImages",
                    "ec2:DescribeInstanceStatus",
                    "ec2:DescribeSpotInstanceRequests",
                    "ec2:DescribeSpotPriceHistory",
                    "ec2:DescribeLaunchTemplates",
                    "ec2:DescribeLaunchTemplateVersions",
                    "ec2:DescribeVolumes",
                    # Provisioning / teardown
                    "ec2:RunInstances",
                    "ec2:TerminateInstances",
                    "ec2:CreateFleet",
                    "ec2:RequestSpotInstances",
                    "ec2:CancelSpotInstanceRequests",
                    # Security group + keypair management (common Ray defaults)
                    "ec2:CreateSecurityGroup",
                    "ec2:DeleteSecurityGroup",
                    "ec2:AuthorizeSecurityGroupIngress",
                    "ec2:AuthorizeSecurityGroupEgress",
                    "ec2:RevokeSecurityGroupIngress",
                    "ec2:RevokeSecurityGroupEgress",
                    "ec2:CreateKeyPair",
                    "ec2:DeleteKeyPair",
                    "ec2:ImportKeyPair",
                    # Tagging
                    "ec2:CreateTags",
                    "ec2:DescribeTags",
                    # Volume ops sometimes used by Ray launchers / retries
                    "ec2:AttachVolume",
                    "ec2:DetachVolume",
                    "ec2:ModifyVolume",
                    # Occasionally needed for spot + retries
                    "ec2:ModifyInstanceAttribute",
                ],
                "Resource": "*",
            },
            {
                "Sid": "S3Results",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"],
                "Resource": [
                    f"arn:aws:s3:::{RESOURCE_PREFIX}-{account_id}",
                    f"arn:aws:s3:::{RESOURCE_PREFIX}-{account_id}/*",
                ],
            },
            {
                "Sid": "ECRPull",
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                ],
                "Resource": "*",
            },
            {
                "Sid": "IAMPassRole",
                "Effect": "Allow",
                "Action": "iam:PassRole",
                "Resource": f"arn:aws:iam::{account_id}:role/{IAM_ROLE_NAME}",
            },
        ],
    }

    # Create or update role
    try:
        iam.get_role(RoleName=IAM_ROLE_NAME)
        step(f"IAM role exists: {IAM_ROLE_NAME}")
    except iam.exceptions.NoSuchEntityException:
        step(f"Creating IAM role: {IAM_ROLE_NAME}")
        iam.create_role(
            RoleName=IAM_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Ray autoscaler role for citrees experiments",
        )

    # Put inline policy (idempotent - overwrites if exists)
    iam.put_role_policy(
        RoleName=IAM_ROLE_NAME,
        PolicyName="citrees-ray-policy",
        PolicyDocument=json.dumps(ray_policy),
    )
    step("IAM policy attached: citrees-ray-policy")

    # Create or get instance profile
    try:
        response = iam.get_instance_profile(InstanceProfileName=IAM_ROLE_NAME)
        step(f"Instance profile exists: {IAM_ROLE_NAME}")
        # Check if role is attached
        roles = response["InstanceProfile"].get("Roles", [])
        if not any(r["RoleName"] == IAM_ROLE_NAME for r in roles):
            step("Attaching role to instance profile...")
            iam.add_role_to_instance_profile(
                InstanceProfileName=IAM_ROLE_NAME, RoleName=IAM_ROLE_NAME
            )
            success("Added role to instance profile")
    except iam.exceptions.NoSuchEntityException:
        step(f"Creating instance profile: {IAM_ROLE_NAME}")
        iam.create_instance_profile(InstanceProfileName=IAM_ROLE_NAME)
        # Add role to instance profile
        iam.add_role_to_instance_profile(InstanceProfileName=IAM_ROLE_NAME, RoleName=IAM_ROLE_NAME)
        success("Added role to instance profile")
        # IAM has eventual consistency - wait for propagation
        step("Waiting for IAM propagation (10s)...")
        time.sleep(10)

    return instance_profile_arn


def build_and_push_image(region: str = DEFAULT_REGION) -> str:
    """Build Docker image and push to ECR.

    Returns the full image URI with tag.
    """
    # Check Docker is available
    if not shutil.which("docker"):
        raise RuntimeError("Docker not found. Install Docker and ensure it's running.")

    docker_check = subprocess.run(["docker", "info"], capture_output=True)
    if docker_check.returncode != 0:
        raise RuntimeError("Docker daemon not running. Start Docker and try again.")

    info("Ensuring ECR repository...")
    repo_name, repo_uri = ensure_ecr_repo(region)

    info("Getting ECR login credentials...")
    ecr = boto3.client("ecr", region_name=region)
    token = ecr.get_authorization_token()
    auth_data = token["authorizationData"][0]
    registry = auth_data["proxyEndpoint"]

    # Docker login to ECR
    step(f"Logging into ECR: {registry}")
    password = base64.b64decode(auth_data["authorizationToken"]).decode().split(":")[1]
    login_cmd = subprocess.run(
        ["docker", "login", "--username", "AWS", "--password-stdin", registry],
        input=password,
        capture_output=True,
        text=True,
    )
    if login_cmd.returncode != 0:
        raise RuntimeError(f"Docker login failed: {login_cmd.stderr}")
    success("ECR login successful")

    # Get git SHA for tagging
    git_sha = get_git_sha()[:12]
    image_tag_sha = f"{repo_uri}:{git_sha}"
    image_tag_latest = f"{repo_uri}:latest"

    info("Building Docker image...")
    step(f"Context: {REPO_ROOT}")
    step(f"Dockerfile: {DOCKERFILE_DIR / 'Dockerfile'}")
    step(f"Tags: {git_sha}, latest")
    step(f"Platform: {DOCKER_PLATFORM}")

    # Build the image
    build_cmd = subprocess.run(
        [
            "docker",
            "build",
            "--platform",
            DOCKER_PLATFORM,
            "-t",
            image_tag_sha,
            "-t",
            image_tag_latest,
            "-f",
            str(DOCKERFILE_DIR / "Dockerfile"),
            str(REPO_ROOT),
        ],
        capture_output=False,  # Show build output
    )
    if build_cmd.returncode != 0:
        raise RuntimeError("Docker build failed")
    success("Build successful")

    # Push both tags
    info("Pushing to ECR...")
    for tag in [image_tag_sha, image_tag_latest]:
        step(f"Pushing: {tag}")
        push_cmd = subprocess.run(["docker", "push", tag], capture_output=False)
        if push_cmd.returncode != 0:
            raise RuntimeError(f"Docker push failed for {tag}")

    success("Image pushed successfully")
    step(image_tag_latest)
    step(image_tag_sha)

    return image_tag_latest


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
    """Generate cluster.yaml from template with user's current IP, branch, and Docker image."""
    from paper.scripts.config import load_config

    if not CLUSTER_EXAMPLE_YAML.exists():
        raise FileNotFoundError(f"Template not found: {CLUSTER_EXAMPLE_YAML}")

    info("Loading config.yaml...")
    cfg = load_config()
    step(f"Cluster: max_workers={cfg.cluster.max_workers}, upscaling_speed={cfg.cluster.upscaling_speed}")

    info("Fetching your public IP...")
    ip = get_public_ip()
    step(f"Your IP: {ip}")

    if branch is None:
        branch = get_current_branch()
    step(f"Branch: {branch}")

    info("Getting AWS account ID...")
    account_id = get_aws_account_id()
    step(f"Account ID: {account_id}")

    info("Getting git SHA...")
    git_sha = get_git_sha()
    step(f"Git SHA: {git_sha[:12]}")

    info("Ensuring S3 bucket for results...")
    bucket_name = ensure_s3_bucket()

    info(f"Reading template: {CLUSTER_EXAMPLE_YAML}")
    template = CLUSTER_EXAMPLE_YAML.read_text()

    step("Replacing placeholders...")
    config_content = (
        template.replace("__MY_IP__", ip)
        .replace("__BRANCH__", branch)
        .replace("__S3_BUCKET__", bucket_name)
        .replace("__AWS_ACCOUNT__", account_id)
        .replace("__GIT_SHA__", git_sha)
        .replace("__REGION__", DEFAULT_REGION)
        .replace("__MAX_WORKERS__", str(cfg.cluster.max_workers))
        .replace("__UPSCALING_SPEED__", str(cfg.cluster.upscaling_speed))
        .replace("__IDLE_TIMEOUT_MINUTES__", str(cfg.cluster.idle_timeout_minutes))
    )

    step(f"Writing: {CLUSTER_YAML}")
    CLUSTER_YAML.write_text(config_content)

    success(f"Generated {CLUSTER_YAML}")
    info("Next: citrees-exp cluster up --yes")


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
    for _node_type, node_config in config.get("available_node_types", {}).items():
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
    heading("Cluster Configuration")
    step(f"Name: {config.get('cluster_name', 'unknown')}")
    step(f"Region: {region}")
    info("Node Types:")
    for node_type, node_config in config.get("available_node_types", {}).items():
        nc = node_config.get("node_config", {})
        step(f"{node_type}:")
        step(f"  Instance: {nc.get('InstanceType', 'N/A')}")
        step(f"  AMI: {nc.get('ImageId', 'N/A')}")
        if "InstanceMarketOptions" in nc:
            step("  Spot: Yes")
        if "resources" in node_config:
            step(f"  Resources: {node_config['resources']}")
        step(
            f"  Min/Max: {node_config.get('min_workers', 'N/A')}/{node_config.get('max_workers', 'N/A')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup and validate Ray cluster config")
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Full setup: build Docker image + generate cluster.yaml (one command)",
    )
    parser.add_argument(
        "--build-image", action="store_true", help="Build and push Docker image to ECR"
    )
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

    if not any(
        [args.setup, args.build_image, args.generate, args.update_ami, args.validate, args.show]
    ):
        parser.print_help()
        return 1

    # --setup: Full setup (IAM + ECR + Docker image + cluster config)
    if args.setup:
        heading("Full Setup: IAM + Docker + cluster.yaml")
        info("[1/3] Ensuring IAM role and instance profile...")
        ensure_iam_role()
        info("[2/3] Building and pushing Docker image...")
        build_and_push_image()
        info("[3/3] Generating cluster.yaml...")
        generate_config(branch=args.branch)
        success("Setup complete!")
        info("Next: citrees-exp cluster up --yes")
        return 0

    # --build-image: Just build and push Docker image
    if args.build_image:
        build_and_push_image()
        return 0

    # --generate: Generate cluster.yaml from template
    if args.generate:
        generate_config(branch=args.branch)
        return 0

    config = load_cluster_config()
    region = config.get("provider", {}).get("region", "us-east-1")

    if args.show:
        show_config(config)

    if args.update_ami:
        info(f"Fetching Ubuntu 22.04 AMI for {region}...")
        ami = get_latest_ubuntu_ami(region)
        step(f"ID: {ami['id']}")
        step(f"Name: {ami['name']}")
        step(f"Created: {ami['created']}")
        config = update_ami(config, ami["id"])
        save_cluster_config(config)
        success(f"Updated {CLUSTER_YAML}")

    if args.validate:
        info("Validating cluster configuration...")
        validation_errors = validate_config(config)
        if validation_errors:
            error("Errors found:")
            for err in validation_errors:
                step(f"- {err}")
            return 1
        success("Configuration valid")

    return 0


if __name__ == "__main__":
    sys.exit(main())
