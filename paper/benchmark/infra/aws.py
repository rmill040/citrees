"""Reusable AWS infrastructure helpers (S3, ECR, IAM, Docker).

Usage (via CLI):
    citrees-exp infra setup           # Create S3 bucket and build Docker image
    citrees-exp infra ecr build       # Build and push Docker image to ECR
    citrees-exp infra ecr clean       # Clean ECR images
    citrees-exp infra s3              # Create S3 bucket
    citrees-exp infra upload-data     # Upload datasets to S3
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import boto3

from paper.benchmark.adapters.data import TaskType
from paper.benchmark.adapters.store import _normalize_artifact_prefix
from paper.benchmark.cli.console_output import info, step, success
from paper.benchmark.pipeline.manifest import validate_manifest_sha256

DOCKERFILE_RELATIVE_PATH = Path("paper/benchmark/infra/docker/Dockerfile")
DEFAULT_REGION = "us-east-1"
RESOURCE_PREFIX = "citrees"  # All resources: citrees-{account_id}
DOCKER_PLATFORM = "linux/amd64"  # AWS EC2 instances are amd64
_FULL_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_OCI_REVISION_LABEL = "org.opencontainers.image.revision"
_ECR_MANIFEST_MEDIA_TYPES = [
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.index.v1+json",
]


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


def _git_repo_root(candidate: Path) -> Path | None:
    """Resolve a candidate directory to the citrees repository root."""
    if not candidate.is_dir():
        return None
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
        cwd=candidate,
    )
    if result.returncode != 0:
        return None
    root = Path(result.stdout.strip()).resolve()
    required = (root / "pyproject.toml", root / DOCKERFILE_RELATIVE_PATH)
    return root if all(path.is_file() for path in required) else None


def get_source_repo_root() -> Path:
    """Return the explicit or discoverable citrees source checkout."""
    configured = os.environ.get("CITREES_REPO_ROOT")
    if configured:
        root = _git_repo_root(Path(configured).expanduser())
        if root is None:
            raise RuntimeError("CITREES_REPO_ROOT is not a citrees Git checkout")
        return root

    module_checkout = Path(__file__).resolve().parents[3]
    for candidate in (Path.cwd(), module_checkout):
        root = _git_repo_root(candidate)
        if root is not None:
            return root
    raise RuntimeError(
        "distributed image builds require a citrees source checkout; "
        "run from the checkout or set CITREES_REPO_ROOT"
    )


def get_frozen_git_sha(repo_root: Path | None = None) -> str:
    """Return the full commit SHA only when tracked source is clean."""
    root = repo_root.resolve() if repo_root is not None else get_source_repo_root()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    )
    git_sha = revision.stdout.strip()
    if not _FULL_GIT_SHA_PATTERN.fullmatch(git_sha):
        raise RuntimeError("git rev-parse HEAD did not return a full commit SHA")

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    )
    if status.stdout.strip():
        raise RuntimeError(
            "source tree must be clean before building or launching a distributed run"
        )
    return git_sha


@contextmanager
def frozen_source_context(repo_root: Path, git_sha: str) -> Iterator[Path]:
    """Materialize a Docker context containing only bytes tracked by one commit."""
    with tempfile.TemporaryDirectory(prefix="citrees-source-") as temp_dir:
        temp_path = Path(temp_dir)
        archive_path = temp_path / "source.tar"
        context_path = temp_path / "context"
        context_path.mkdir()
        subprocess.run(
            [
                "git",
                "archive",
                "--format=tar",
                "--output",
                str(archive_path),
                git_sha,
            ],
            check=True,
            cwd=repo_root,
        )
        with tarfile.open(archive_path, mode="r") as archive:
            archive.extractall(context_path, filter="data")
        yield context_path


def verify_candidate_image(
    image_tag: str,
    git_sha: str,
    *,
    docker_env: dict[str, str],
) -> None:
    """Verify the exact local image that will be pushed to ECR."""
    revision = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            '{{ index .Config.Labels "org.opencontainers.image.revision" }}',
            image_tag,
        ],
        capture_output=True,
        text=True,
        check=True,
        env=docker_env,
    ).stdout.strip()
    if revision != git_sha:
        raise RuntimeError(f"candidate image revision is {revision!r}, expected {git_sha}")

    checks = [
        [
            "docker",
            "run",
            "--rm",
            image_tag,
            "R",
            "--vanilla",
            "--slave",
            "-e",
            (
                'stopifnot(as.character(getRversion()) == "4.5.2"); '
                'stopifnot(as.character(packageVersion("partykit")) == "1.2.24")'
            ),
        ],
        [
            "docker",
            "run",
            "--rm",
            image_tag,
            "python",
            "-c",
            (
                "import platform; "
                "from importlib.metadata import version; "
                "from paper.benchmark.pipeline.r_methods import _get_partykit; "
                "assert platform.python_version() == '3.12.7'; "
                "assert version('rpy2') == '3.6.7'; "
                "assert version('scikit-learn') == '1.8.0'; "
                "_get_partykit()"
            ),
        ],
        [
            "docker",
            "run",
            "--rm",
            image_tag,
            "pytest",
            "tests/paper/test_stage1.py",
            "tests/paper/test_provenance.py",
            "-q",
            "-rs",
        ],
    ]
    for command in checks:
        result = subprocess.run(command, check=False, env=docker_env)
        if result.returncode != 0:
            raise RuntimeError(f"candidate image verification failed: {' '.join(command)}")


def ensure_s3_bucket(region: str = DEFAULT_REGION) -> str:
    """Ensure the private, versioned S3 bucket for results exists.

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

    s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_versioning(
        Bucket=bucket_name,
        VersioningConfiguration={"Status": "Enabled"},
    )
    s3.put_bucket_encryption(
        Bucket=bucket_name,
        ServerSideEncryptionConfiguration={
            "Rules": [
                {
                    "ApplyServerSideEncryptionByDefault": {
                        "SSEAlgorithm": "AES256",
                    },
                    "BucketKeyEnabled": False,
                }
            ]
        },
    )
    s3.put_bucket_policy(
        Bucket=bucket_name,
        Policy=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "DenyInsecureTransport",
                        "Effect": "Deny",
                        "Principal": "*",
                        "Action": "s3:*",
                        "Resource": [
                            f"arn:aws:s3:::{bucket_name}",
                            f"arn:aws:s3:::{bucket_name}/*",
                        ],
                        "Condition": {"Bool": {"aws:SecureTransport": "false"}},
                    }
                ],
            }
        ),
    )

    return bucket_name


def publish_rerun_manifest(
    manifest_path: Path,
    *,
    region: str = DEFAULT_REGION,
) -> dict[str, str | int]:
    """Validate and publish a private manifest under its content hash."""
    from botocore.exceptions import ClientError

    from paper.benchmark.pipeline.manifest import (
        manifest_s3_key,
        parse_rerun_manifest,
    )

    payload = manifest_path.read_bytes()
    manifest = parse_rerun_manifest(payload)
    account_id = get_aws_account_id()
    if manifest.account_ids != (account_id,):
        raise ValueError(
            f"manifest is bound to accounts {list(manifest.account_ids)}, "
            f"but the active AWS account is {account_id}"
        )
    bucket = ensure_s3_bucket(region)
    key = manifest_s3_key(manifest.sha256)
    s3 = boto3.client("s3", region_name=region)

    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType="text/csv",
            Metadata={
                "sha256": manifest.sha256,
                "campaign-sha256": manifest.campaign_sha256,
                "cell-count": str(len(manifest.cells)),
                "target-aws-account-id": account_id,
            },
            IfNoneMatch="*",
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code not in {"412", "PreconditionFailed"} and status != 412:
            raise

    response = s3.get_object(Bucket=bucket, Key=key)
    observed_payload = response["Body"].read()
    observed_metadata = response.get("Metadata", {})
    if observed_metadata.get("target-aws-account-id") != account_id:
        raise RuntimeError(f"Published manifest account binding is invalid: s3://{bucket}/{key}")
    if observed_metadata.get("campaign-sha256") != manifest.campaign_sha256:
        raise RuntimeError(f"Published manifest campaign binding is invalid: s3://{bucket}/{key}")
    observed = parse_rerun_manifest(
        observed_payload,
        expected_sha256=manifest.sha256,
    )
    if observed_payload != payload:
        raise RuntimeError(
            f"Content-addressed manifest differs from local bytes: s3://{bucket}/{key}"
        )
    if len(observed.cells) != len(manifest.cells):
        raise RuntimeError(f"Manifest cell count changed after upload: s3://{bucket}/{key}")

    return {
        "bucket": bucket,
        "key": key,
        "sha256": manifest.sha256,
        "campaign_sha256": manifest.campaign_sha256,
        "cells": len(manifest.cells),
    }


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
        ecr.put_image_tag_mutability(
            repositoryName=repo_name,
            imageTagMutability="IMMUTABLE",
        )
        ecr.put_image_scanning_configuration(
            repositoryName=repo_name,
            imageScanningConfiguration={"scanOnPush": True},
        )
        step(f"ECR repository exists: {repo_name}")
    except ecr.exceptions.RepositoryNotFoundException:
        step(f"Creating ECR repository: {repo_name}")
        response = ecr.create_repository(
            repositoryName=repo_name,
            imageScanningConfiguration={"scanOnPush": True},
            imageTagMutability="IMMUTABLE",
        )
        repo_uri = response["repository"]["repositoryUri"]
        success(f"Created ECR repository: {repo_uri}")

    return repo_name, repo_uri


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _parse_json_object(payload: str | bytes, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(payload, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{context} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must contain a JSON object")
    return value


def _require_sha256_digest(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not _SHA256_DIGEST_PATTERN.fullmatch(value):
        raise RuntimeError(f"{context} does not contain a valid SHA-256 digest")
    return value


def _load_ecr_manifest(ecr: Any, repository_name: str, image_digest: str) -> dict[str, Any]:
    image_digest = _require_sha256_digest(image_digest, context="ECR image identity")
    response = ecr.batch_get_image(
        repositoryName=repository_name,
        imageIds=[{"imageDigest": image_digest}],
        acceptedMediaTypes=_ECR_MANIFEST_MEDIA_TYPES,
    )
    failures = response.get("failures", [])
    images = response.get("images", [])
    if failures or len(images) != 1:
        raise RuntimeError(f"Could not load remote manifest {image_digest}: failures={failures!r}")
    image = images[0]
    observed_digest = image.get("imageId", {}).get("imageDigest")
    if observed_digest != image_digest:
        raise RuntimeError(f"ECR returned manifest {observed_digest!r}, expected {image_digest}")
    payload = image.get("imageManifest")
    if not isinstance(payload, str):
        raise RuntimeError(f"ECR manifest {image_digest} has no JSON payload")
    payload_bytes = payload.encode()
    actual_digest = f"sha256:{hashlib.sha256(payload_bytes).hexdigest()}"
    if actual_digest != image_digest:
        raise RuntimeError(f"ECR manifest bytes hash to {actual_digest}, expected {image_digest}")
    return _parse_json_object(payload_bytes, context=f"ECR manifest {image_digest}")


def _remote_config_descriptor(
    ecr: Any,
    repository_name: str,
    image_digest: str,
    *,
    depth: int = 0,
) -> tuple[str, int | None]:
    if depth > 1:
        raise RuntimeError("ECR image index nesting exceeds one platform-selection level")
    manifest = _load_ecr_manifest(ecr, repository_name, image_digest)
    config = manifest.get("config")
    manifests = manifest.get("manifests")
    if isinstance(config, dict) and manifests is None:
        config_digest = _require_sha256_digest(
            config.get("digest"),
            context=f"ECR manifest {image_digest} config",
        )
        config_size = config.get("size")
        if config_size is not None and (type(config_size) is not int or config_size < 0):
            raise RuntimeError(f"ECR manifest {image_digest} has an invalid config size")
        return config_digest, config_size
    if isinstance(manifests, list) and config is None:
        platform_digests: list[str] = []
        for descriptor in manifests:
            if not isinstance(descriptor, dict):
                raise RuntimeError(f"ECR image index {image_digest} is malformed")
            platform = descriptor.get("platform")
            if (
                isinstance(platform, dict)
                and platform.get("os") == "linux"
                and platform.get("architecture") == "amd64"
            ):
                platform_digests.append(
                    _require_sha256_digest(
                        descriptor.get("digest"),
                        context=f"ECR image index {image_digest} platform",
                    )
                )
        if len(platform_digests) != 1:
            raise RuntimeError(
                f"ECR image index {image_digest} contains "
                f"{len(platform_digests)} linux/amd64 manifests"
            )
        return _remote_config_descriptor(
            ecr,
            repository_name,
            platform_digests[0],
            depth=depth + 1,
        )
    raise RuntimeError(f"ECR manifest {image_digest} is neither an image nor an image index")


def _remote_image_revision(ecr: Any, repository_name: str, image_digest: str) -> str:
    config_digest, expected_size = _remote_config_descriptor(
        ecr,
        repository_name,
        image_digest,
    )
    response = ecr.get_download_url_for_layer(
        repositoryName=repository_name,
        layerDigest=config_digest,
    )
    download_url = response.get("downloadUrl")
    if not isinstance(download_url, str) or not download_url.startswith("https://"):
        raise RuntimeError(f"ECR config {config_digest} has no HTTPS download URL")
    with urllib.request.urlopen(download_url, timeout=30) as remote:
        payload = remote.read()
    if expected_size is not None and len(payload) != expected_size:
        raise RuntimeError(
            f"ECR config {config_digest} has {len(payload)} bytes, expected {expected_size}"
        )
    actual_digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if actual_digest != config_digest:
        raise RuntimeError(f"ECR config bytes hash to {actual_digest}, expected {config_digest}")
    image_config = _parse_json_object(payload, context=f"ECR config {config_digest}")
    runtime_config = image_config.get("config")
    labels = runtime_config.get("Labels") if isinstance(runtime_config, dict) else None
    revision = labels.get(_OCI_REVISION_LABEL) if isinstance(labels, dict) else None
    if not isinstance(revision, str):
        raise RuntimeError(f"ECR config {config_digest} has no {_OCI_REVISION_LABEL} label")
    return revision


def validate_image_revision(
    image_uri: str,
    *,
    region: str = DEFAULT_REGION,
) -> str:
    """Require a remote image digest built from the active clean revision."""
    git_sha = get_frozen_git_sha()
    account_id = get_aws_account_id()
    repo_name = get_resource_name(account_id)
    ecr = boto3.client("ecr", region_name=region)
    repositories = ecr.describe_repositories(repositoryNames=[repo_name]).get("repositories", [])
    if len(repositories) != 1:
        raise RuntimeError(f"Could not resolve ECR repository {repo_name}")
    repository_uri = str(repositories[0].get("repositoryUri", ""))
    expected_prefix = f"{repository_uri}@"
    if not image_uri.startswith(expected_prefix):
        raise RuntimeError(f"image URI does not belong to active repository {repository_uri}")
    image_digest = image_uri.removeprefix(expected_prefix)
    _require_sha256_digest(image_digest, context="image URI")
    image_details = ecr.describe_images(
        repositoryName=repo_name,
        imageIds=[{"imageDigest": image_digest}],
    ).get("imageDetails", [])
    if len(image_details) != 1:
        raise RuntimeError(f"Could not resolve image digest {image_digest}")
    tags = image_details[0].get("imageTags", [])
    if git_sha not in tags:
        raise RuntimeError(f"image digest is not tagged with active source revision {git_sha}")
    remote_revision = _remote_image_revision(ecr, repo_name, image_digest)
    if remote_revision != git_sha:
        raise RuntimeError(f"remote OCI revision label is {remote_revision!r}, expected {git_sha}")
    return git_sha


def campaign_instance_profile_name(
    *,
    output_prefix: str,
    campaign_sha256: str,
) -> str:
    """Derive one IAM identity from an immutable campaign and output prefix."""
    normalized_prefix = _normalize_artifact_prefix(output_prefix)
    campaign_sha256 = validate_manifest_sha256(campaign_sha256)
    identity = hashlib.sha256(f"{campaign_sha256}\0{normalized_prefix}".encode()).hexdigest()
    return f"citrees-campaign-{identity[:32]}"


def ensure_campaign_iam_profile(
    *,
    output_prefix: str,
    campaign_sha256: str,
    region: str = DEFAULT_REGION,
) -> str:
    """Ensure one campaign-bound role and return its instance-profile name."""
    normalized_prefix = _normalize_artifact_prefix(output_prefix)
    campaign_sha256 = validate_manifest_sha256(campaign_sha256)
    profile_name = campaign_instance_profile_name(
        output_prefix=normalized_prefix,
        campaign_sha256=campaign_sha256,
    )
    role_name = profile_name
    policy_name = f"{profile_name}-runtime"
    iam = boto3.client("iam", region_name=region)
    account_id = get_aws_account_id()
    bucket_name = get_resource_name(account_id)
    bucket_arn = f"arn:aws:s3:::{bucket_name}"
    output_arn = f"{bucket_arn}/{normalized_prefix}/*"

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

    runtime_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "S3ListApprovedPrefixes",
                "Effect": "Allow",
                "Action": "s3:ListBucket",
                "Resource": bucket_arn,
                "Condition": {
                    "StringLike": {
                        "s3:prefix": [
                            "data/*",
                            "rerun-manifests/*",
                            f"{normalized_prefix}/*",
                        ]
                    }
                },
            },
            {
                "Sid": "S3ReadInputsAndArtifacts",
                "Effect": "Allow",
                "Action": "s3:GetObject",
                "Resource": [
                    f"{bucket_arn}/data/*",
                    f"{bucket_arn}/rerun-manifests/*",
                    output_arn,
                ],
            },
            {
                "Sid": "S3WriteArtifacts",
                "Effect": "Allow",
                "Action": "s3:PutObject",
                "Resource": output_arn,
            },
            {
                "Sid": "ECRAuthorization",
                "Effect": "Allow",
                "Action": "ecr:GetAuthorizationToken",
                "Resource": "*",
            },
            {
                "Sid": "ECRPull",
                "Effect": "Allow",
                "Action": [
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                ],
                "Resource": (
                    f"arn:aws:ecr:{region}:{account_id}:repository/{RESOURCE_PREFIX}-{account_id}"
                ),
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

    try:
        iam.get_role(RoleName=role_name)
        step(f"IAM role exists: {role_name}")
    except iam.exceptions.NoSuchEntityException:
        step(f"Creating IAM role: {role_name}")
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description=f"citrees campaign {campaign_sha256[:16]} runtime",
        )
    else:
        iam.update_assume_role_policy(
            RoleName=role_name,
            PolicyDocument=json.dumps(trust_policy),
        )

    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=policy_name,
        PolicyDocument=json.dumps(runtime_policy),
    )
    step(f"IAM policy attached: {policy_name}")

    ssm_policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
    iam.attach_role_policy(RoleName=role_name, PolicyArn=ssm_policy_arn)
    step("Attached AmazonSSMManagedInstanceCore")

    profile_changed = False
    try:
        response = iam.get_instance_profile(InstanceProfileName=profile_name)
        step(f"Instance profile exists: {profile_name}")
        roles = response["InstanceProfile"].get("Roles", [])
        unexpected_roles = sorted(
            role["RoleName"] for role in roles if role.get("RoleName") != role_name
        )
        if unexpected_roles:
            raise RuntimeError(
                f"campaign instance profile {profile_name} contains unexpected roles "
                f"{unexpected_roles}"
            )
        if not roles:
            step("Attaching role to instance profile...")
            iam.add_role_to_instance_profile(
                InstanceProfileName=profile_name,
                RoleName=role_name,
            )
            success("Added role to instance profile")
            profile_changed = True
    except iam.exceptions.NoSuchEntityException:
        step(f"Creating instance profile: {profile_name}")
        iam.create_instance_profile(InstanceProfileName=profile_name)
        iam.add_role_to_instance_profile(
            InstanceProfileName=profile_name,
            RoleName=role_name,
        )
        success("Added role to instance profile")
        profile_changed = True

    if profile_changed:
        step("Waiting for IAM propagation (10s)...")
        time.sleep(10)

    return profile_name


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

        # AWS merges sources with the same protocol and port range into one
        # IpPermission. Revoke only the stale caller ranges so a colocated
        # security-group source remains intact.
        for port, desc in [(22, "SSH from caller"), (8000, "API from caller")]:
            caller_ranges = [
                ip_range
                for rule in existing_rules
                if rule.get("IpProtocol") == "tcp"
                and rule.get("FromPort") == port
                and rule.get("ToPort") == port
                for ip_range in rule.get("IpRanges", [])
                if ip_range.get("Description") == desc
            ]
            stale_ranges = [
                ip_range for ip_range in caller_ranges if ip_range.get("CidrIp") != my_cidr
            ]
            if stale_ranges:
                ec2.revoke_security_group_ingress(
                    GroupId=sg_id,
                    IpPermissions=[
                        {
                            "IpProtocol": "tcp",
                            "FromPort": port,
                            "ToPort": port,
                            "IpRanges": stale_ranges,
                        }
                    ],
                )
            if not any(ip_range.get("CidrIp") == my_cidr for ip_range in caller_ranges):
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

        has_api_self_rule = any(
            rule.get("IpProtocol") == "tcp"
            and rule.get("FromPort") == 8000
            and rule.get("ToPort") == 8000
            and any(pair.get("GroupId") == sg_id for pair in rule.get("UserIdGroupPairs", []))
            for rule in existing_rules
        )
        if not has_api_self_rule:
            ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 8000,
                        "ToPort": 8000,
                        "UserIdGroupPairs": [
                            {
                                "GroupId": sg_id,
                                "Description": "API from citrees instances",
                            }
                        ],
                    }
                ],
            )

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

    Stage 1: Delete full-revision tagged images.
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

    Returns the immutable image URI with digest.
    """
    # Check Docker is available
    if not shutil.which("docker"):
        raise RuntimeError("Docker not found. Install Docker and ensure it's running.")

    docker_check = subprocess.run(["docker", "info"], capture_output=True)
    if docker_check.returncode != 0:
        raise RuntimeError("Docker daemon not running. Start Docker and try again.")

    repo_root = get_source_repo_root()
    git_sha = get_frozen_git_sha(repo_root)
    info("Ensuring ECR repository...")
    repo_name, repo_uri = ensure_ecr_repo(region)

    info("Getting ECR login credentials...")
    ecr = boto3.client("ecr", region_name=region)
    token = ecr.get_authorization_token()
    auth_data = token["authorizationData"][0]
    registry = auth_data["proxyEndpoint"]
    password = base64.b64decode(auth_data["authorizationToken"]).decode().split(":")[1]
    registry_host = registry.removeprefix("https://").removeprefix("http://")
    auth_value = base64.b64encode(f"AWS:{password}".encode()).decode()

    # Use an isolated Docker config to avoid local credential-helper failures.
    with tempfile.TemporaryDirectory(prefix="citrees-docker-") as docker_config_dir:
        config_path = Path(docker_config_dir) / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "auths": {
                        registry_host: {"auth": auth_value},
                        registry: {"auth": auth_value},
                    }
                }
            ),
            encoding="utf-8",
        )
        docker_env = os.environ.copy()
        docker_env["DOCKER_CONFIG"] = docker_config_dir
        buildx_binary = Path.home() / ".docker" / "cli-plugins" / "docker-buildx"
        if buildx_binary.exists():
            plugin_dir = Path(docker_config_dir) / "cli-plugins"
            plugin_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(buildx_binary, plugin_dir / "docker-buildx")

        step(f"Using isolated Docker auth for ECR: {registry_host}")

        # The immutable tag binds the image to one complete source revision.
        image_tag_sha = f"{repo_uri}:{git_sha}"

        with frozen_source_context(repo_root, git_sha) as source_context:
            dockerfile = source_context / DOCKERFILE_RELATIVE_PATH
            info("Building Docker image from frozen source...")
            step(f"Context: {source_context}")
            step(f"Dockerfile: {dockerfile}")
            step(f"Tag: {git_sha}")
            step(f"Platform: {DOCKER_PLATFORM}")

            buildx_check = subprocess.run(
                ["docker", "buildx", "version"],
                capture_output=True,
                text=True,
                env=docker_env,
            )
            common_args = [
                "--platform",
                DOCKER_PLATFORM,
                "--build-arg",
                f"SOURCE_GIT_SHA={git_sha}",
                "-t",
                image_tag_sha,
                "-f",
                str(dockerfile),
                str(source_context),
            ]
            if buildx_check.returncode == 0:
                step("Using docker buildx for cross-platform amd64 build")
                build_cmd = subprocess.run(
                    ["docker", "buildx", "build", "--load", *common_args],
                    check=False,
                    env=docker_env,
                )
            else:
                step("docker buildx unavailable, using docker build")
                build_cmd = subprocess.run(
                    ["docker", "build", *common_args],
                    check=False,
                    env=docker_env,
                )
            if build_cmd.returncode != 0:
                raise RuntimeError("Docker build failed")

        info("Verifying exact candidate image before push...")
        verify_candidate_image(image_tag_sha, git_sha, docker_env=docker_env)

        info("Pushing verified image to ECR...")
        step(f"Pushing: {image_tag_sha}")
        push_cmd = subprocess.run(
            ["docker", "push", image_tag_sha],
            check=False,
            env=docker_env,
        )
        if push_cmd.returncode != 0:
            raise RuntimeError(f"Docker push failed for {image_tag_sha}")

        success("Verified image pushed successfully")
        step(image_tag_sha)

        image_details = ecr.describe_images(
            repositoryName=repo_name,
            imageIds=[{"imageTag": git_sha}],
        ).get("imageDetails", [])
        if len(image_details) != 1 or not image_details[0].get("imageDigest"):
            raise RuntimeError(f"Could not resolve immutable digest for {image_tag_sha}")
        digest_uri = f"{repo_uri}@{image_details[0]['imageDigest']}"
        step(digest_uri)
        return digest_uri


def upload_datasets(
    task: TaskType | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Publish local datasets under immutable content-addressed S3 keys.

    Parameters
    ----------
    task : str, optional
        Only upload for this task type ("classification" or "regression").
        If None, uploads both.
    dry_run : bool, default False
        If True, show what would be uploaded without actually uploading.
    Returns
    -------
    dict[str, int]
        Counts: {"uploaded": N, "skipped": N}
    """
    from botocore.exceptions import ClientError

    from paper.benchmark.adapters.data import (
        DataSource,
        get_data_dir,
        get_dataset_payload_identity,
        get_dataset_prefix,
        get_dataset_s3_key,
        validate_dataset_payload,
    )

    bucket = ensure_s3_bucket()
    s3 = boto3.client("s3", region_name=DEFAULT_REGION)

    tasks: list[TaskType] = [task] if task else ["classification", "regression"]
    sources: list[DataSource] = ["real", "synthetic"]

    uploaded = 0
    skipped = 0

    for tt in tasks:
        for src in sources:
            local_dir = get_data_dir(tt, src)
            if not local_dir.exists():
                continue

            for parquet in local_dir.glob("*.parquet"):
                payload = parquet.read_bytes()
                identity = get_dataset_payload_identity(payload)
                prefix = get_dataset_prefix(tt)
                if not parquet.stem.startswith(prefix):
                    raise ValueError(
                        f"dataset filename {parquet.name!r} must start with {prefix!r}"
                    )
                dataset = parquet.stem.removeprefix(prefix)
                s3_key = get_dataset_s3_key(dataset, tt, src, identity)

                if dry_run:
                    step(f"Would upload: {parquet.name} -> s3://{bucket}/{s3_key}")
                    uploaded += 1
                    continue

                try:
                    existing = s3.get_object(Bucket=bucket, Key=s3_key)
                except ClientError as exc:
                    code = exc.response.get("Error", {}).get("Code", "")
                    if code not in {"404", "NoSuchKey", "NotFound"}:
                        raise
                else:
                    validate_dataset_payload(
                        existing["Body"].read(),
                        identity,
                        location=f"s3://{bucket}/{s3_key}",
                    )
                    skipped += 1
                    continue

                step(f"Uploading: {parquet.name}")
                try:
                    s3.put_object(
                        Bucket=bucket,
                        Key=s3_key,
                        Body=payload,
                        ContentType="application/vnd.apache.parquet",
                        IfNoneMatch="*",
                        Metadata={
                            "sha256": identity.sha256,
                            "dataset": dataset,
                            "task": tt,
                            "source": src,
                            "n-samples": str(identity.n_samples),
                            "n-features": str(identity.n_features),
                        },
                    )
                except ClientError as exc:
                    code = exc.response.get("Error", {}).get("Code", "")
                    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                    if code not in {"412", "PreconditionFailed"} and status != 412:
                        raise

                observed = s3.get_object(Bucket=bucket, Key=s3_key)["Body"].read()
                validate_dataset_payload(
                    observed,
                    identity,
                    location=f"s3://{bucket}/{s3_key}",
                )
                uploaded += 1

    return {"uploaded": uploaded, "skipped": skipped}
