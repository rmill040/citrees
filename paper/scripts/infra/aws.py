"""Reusable AWS infrastructure helpers (S3, ECR, IAM, Docker).

Usage (via CLI):
    citrees-exp infra setup           # Full setup: IAM + Docker
    citrees-exp infra ecr build       # Build and push Docker image to ECR
    citrees-exp infra ecr clean       # Clean ECR images
    citrees-exp infra iam             # Create IAM role
    citrees-exp infra s3              # Create S3 bucket
    citrees-exp infra upload-data     # Upload datasets to S3
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import boto3

from paper.scripts.cli._console import info, step, success

DOCKERFILE_DIR = Path(__file__).parent / "docker"
REPO_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_REGION = "us-east-1"
RESOURCE_PREFIX = "citrees"  # All resources: citrees-{account_id}
IAM_ROLE_NAME = "citrees-worker"
DOCKER_PLATFORM = "linux/amd64"  # AWS EC2 instances are amd64


def get_public_ip() -> str:
    """Get the user's current public IP address."""
    for url in ["https://checkip.amazonaws.com", "https://ifconfig.me", "https://api.ipify.org"]:
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                return response.read().decode("utf-8").strip()
        except Exception:
            continue
    raise RuntimeError("Could not determine public IP from any service")


def get_aws_account_id() -> str:
    """Get the current AWS account ID."""
    sts = boto3.client("sts")
    return sts.get_caller_identity()["Account"]


def get_resource_name(account_id: str) -> str:
    """Get the standardized resource name: citrees-{account_id}.

    Used for both S3 bucket and ECR repository.
    """
    return f"{RESOURCE_PREFIX}-{account_id}"


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
    - IAM role: citrees-worker
    - Instance profile: citrees-worker
    - Attached policies: S3, ECR, IAM:PassRole permissions

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

    # Inline policy for worker instances (S3 + ECR + IAM:PassRole).
    worker_policy = {
        "Version": "2012-10-17",
        "Statement": [
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
            {
                "Sid": "CloudWatchLogs",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                ],
                "Resource": f"arn:aws:logs:*:{account_id}:log-group:/citrees/*",
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
            Description="Worker role for citrees experiments",
        )

    # Put inline policy (idempotent - overwrites if exists)
    iam.put_role_policy(
        RoleName=IAM_ROLE_NAME,
        PolicyName="citrees-worker-policy",
        PolicyDocument=json.dumps(worker_policy),
    )
    step("IAM policy attached: citrees-worker-policy")

    # Attach SSM managed policy for remote access
    ssm_policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
    iam.attach_role_policy(RoleName=IAM_ROLE_NAME, PolicyArn=ssm_policy_arn)
    step("Attached AmazonSSMManagedInstanceCore")

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


def ensure_security_group(region: str = DEFAULT_REGION) -> str:
    """Ensure the citrees security group exists with correct rules.

    Creates a security group ``citrees-sg`` in the default VPC that allows:
    - Inbound TCP 8000 from within the group (worker → API)
    - Inbound TCP 8000 from the caller's public IP (CLI → API)
    - Inbound TCP 22 from the caller's public IP (SSH for debugging)
    - All outbound (VPC default)

    Returns the security group ID.
    """
    ec2 = boto3.client("ec2", region_name=region)
    sg_name = "citrees-sg"

    # Check if it already exists
    resp = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [sg_name]}])
    if resp["SecurityGroups"]:
        sg_id = resp["SecurityGroups"][0]["GroupId"]
        step(f"Security group exists: {sg_name} ({sg_id})")

        # Update IP-based rules to current IP (idempotent)
        my_ip = get_public_ip()
        my_cidr = f"{my_ip}/32"
        existing_rules = resp["SecurityGroups"][0].get("IpPermissions", [])

        def _has_ip_rule(port: int) -> bool:
            return any(
                r.get("FromPort") == port
                and any(ipr.get("CidrIp") == my_cidr for ipr in r.get("IpRanges", []))
                for r in existing_rules
            )

        # Refresh caller-IP rules for SSH (22) and API (8000)
        for port, desc in [(22, "SSH from caller"), (8000, "API from caller")]:
            if _has_ip_rule(port):
                continue
            # Revoke old IP-based rules for this port (keep group-internal rules)
            old_ip_rules = [
                r for r in existing_rules if r.get("FromPort") == port and r.get("IpRanges")
            ]
            if old_ip_rules:
                ec2.revoke_security_group_ingress(GroupId=sg_id, IpPermissions=old_ip_rules)
            ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": port,
                        "ToPort": port,
                        "IpRanges": [{"CidrIp": my_cidr, "Description": desc}],
                    }
                ],
            )
            step(f"Updated port {port} rule to {my_cidr}")

        return sg_id

    # Create new security group in default VPC
    step(f"Creating security group: {sg_name}")
    create_resp = ec2.create_security_group(
        GroupName=sg_name,
        Description="citrees API + worker instances",
    )
    sg_id = create_resp["GroupId"]

    my_ip = get_public_ip()

    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            # API port: allow from within the security group
            {
                "IpProtocol": "tcp",
                "FromPort": 8000,
                "ToPort": 8000,
                "UserIdGroupPairs": [
                    {"GroupId": sg_id, "Description": "API from citrees instances"}
                ],
            },
            # API port: allow from caller's IP (CLI access)
            {
                "IpProtocol": "tcp",
                "FromPort": 8000,
                "ToPort": 8000,
                "IpRanges": [{"CidrIp": f"{my_ip}/32", "Description": "API from caller"}],
            },
            # SSH: allow from caller's IP
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": f"{my_ip}/32", "Description": "SSH from caller"}],
            },
        ],
    )

    success(f"Created security group: {sg_name} ({sg_id})")
    step(f"Inbound 8000 from group, SSH from {my_ip}/32")

    return sg_id


def clean_ecr(region: str = DEFAULT_REGION) -> dict[str, int]:
    """Delete all images from the ECR repository (2-stage).

    Stage 1: Delete tagged images (e.g. :latest, :abc123).
    Stage 2: Delete remaining untagged manifests (orphaned layers).

    Returns dict with counts: {"tagged": N, "untagged": M}.
    """
    account_id = get_aws_account_id()
    repo_name = get_resource_name(account_id)
    ecr = boto3.client("ecr", region_name=region)

    try:
        ecr.describe_repositories(repositoryNames=[repo_name])
    except ecr.exceptions.RepositoryNotFoundException:
        info(f"ECR repository {repo_name} does not exist, nothing to clean")
        return {"tagged": 0, "untagged": 0}

    counts: dict[str, int] = {"tagged": 0, "untagged": 0}

    for stage, tag_status in [("tagged", "TAGGED"), ("untagged", "UNTAGGED")]:
        image_ids: list[dict[str, str]] = []
        paginator = ecr.get_paginator("list_images")
        for page in paginator.paginate(repositoryName=repo_name, filter={"tagStatus": tag_status}):
            image_ids.extend(page.get("imageIds", []))

        if not image_ids:
            step(f"Stage {stage}: no images to delete")
            continue

        # batch_delete_image accepts max 100 per call
        for i in range(0, len(image_ids), 100):
            batch = image_ids[i : i + 100]
            resp = ecr.batch_delete_image(repositoryName=repo_name, imageIds=batch)
            deleted = len(resp.get("imageIds", []))
            counts[stage] += deleted
            failures = resp.get("failures", [])
            if failures:
                for f in failures:
                    step(f"  Failed: {f['imageId']} - {f['failureReason']}")

        step(f"Stage {stage}: deleted {counts[stage]} images")

    return counts


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


def upload_datasets(
    task: str | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, int]:
    """Upload datasets from paper/data/ to S3.

    Parameters
    ----------
    task : str, optional
        Only upload for this task type ("classification" or "regression").
        If None, uploads both.
    dry_run : bool, default False
        If True, show what would be uploaded without actually uploading.
    force : bool, default False
        If True, re-upload even if file already exists in S3.

    Returns
    -------
    dict[str, int]
        Counts: {"uploaded": N, "skipped": N}
    """
    from botocore.exceptions import ClientError

    from paper.scripts.adapters.data import get_data_dir, get_data_s3_prefix

    bucket = ensure_s3_bucket()
    s3 = boto3.client("s3", region_name=DEFAULT_REGION)

    tasks = [task] if task else ["classification", "regression"]
    sources = ["real", "synthetic"]

    uploaded = 0
    skipped = 0

    for tt in tasks:
        for src in sources:
            local_dir = get_data_dir(tt, src)
            if not local_dir.exists():
                continue
            s3_prefix = get_data_s3_prefix(tt, src)

            for parquet in local_dir.glob("*.parquet"):
                s3_key = f"{s3_prefix}/{parquet.name}"

                # Check if already exists in S3 (skip unless --force)
                if not force:
                    try:
                        s3.head_object(Bucket=bucket, Key=s3_key)
                        skipped += 1
                        continue
                    except ClientError as e:
                        if e.response["Error"]["Code"] != "404":
                            raise

                if dry_run:
                    step(f"Would upload: {parquet.name} -> s3://{bucket}/{s3_key}")
                else:
                    step(f"Uploading: {parquet.name}")
                    s3.upload_file(str(parquet), bucket, s3_key)
                uploaded += 1

    return {"uploaded": uploaded, "skipped": skipped}
