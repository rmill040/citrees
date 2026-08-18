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
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
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
_AWS_ACCOUNT_ID_PATTERN = re.compile(r"^[0-9]{12}$")
_AWS_IDENTITY_ARN_PATTERN = re.compile(
    r"^arn:(?:aws|aws-cn|aws-us-gov):(?:iam|sts)::(?P<account>[0-9]{12}):.+$"
)
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


def get_aws_caller_identity(*, client: Any | None = None) -> dict[str, str]:
    """Return one validated live STS caller identity."""
    sts = boto3.client("sts") if client is None else client
    response = sts.get_caller_identity()
    if not isinstance(response, dict):
        raise RuntimeError("STS caller identity response is invalid")
    account = response.get("Account")
    arn = response.get("Arn")
    user_id = response.get("UserId")
    if (
        not isinstance(account, str)
        or not _AWS_ACCOUNT_ID_PATTERN.fullmatch(account)
        or not isinstance(arn, str)
        or not isinstance(user_id, str)
        or not user_id.strip()
    ):
        raise RuntimeError("STS caller identity response is invalid")
    arn_match = _AWS_IDENTITY_ARN_PATTERN.fullmatch(arn)
    if arn_match is None or arn_match.group("account") != account:
        raise RuntimeError("STS caller identity response is invalid")
    return {
        "Account": account,
        "Arn": arn,
        "UserId": user_id,
    }


def get_aws_account_id() -> str:
    """Return the current account from a validated live STS identity."""
    return get_aws_caller_identity()["Account"]


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
            "--init",
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
            "--init",
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
            "--init",
            image_tag,
            "python",
            "-m",
            "paper.benchmark.experiments.r_cforest_reproducibility",
            "--help",
        ],
        [
            "docker",
            "run",
            "--rm",
            "--init",
            image_tag,
            "python",
            "-m",
            "paper.jss.replication.cloud",
            "--help",
        ],
        [
            "docker",
            "run",
            "--rm",
            "--init",
            "-e",
            f"GIT_SHA={git_sha}",
            "-e",
            "CITREES_SOURCE_CLEAN=1",
            image_tag,
            "python",
            "-c",
            (
                "from paper.jss.replication.shards import capture_execution_context; "
                "context = capture_execution_context('calibration'); "
                f"assert context.git_sha == {git_sha!r}; "
                "assert context.git_dirty is False"
            ),
        ],
        [
            "docker",
            "run",
            "--rm",
            "--init",
            "-e",
            f"GIT_SHA={git_sha}",
            "-e",
            "CITREES_SOURCE_CLEAN=1",
            image_tag,
            "python",
            "-m",
            "paper.jss.replication.shards",
            "calibration-shard",
            "--profile",
            "smoke",
            "--component",
            "selector",
            "--shard-index",
            "0",
            "--num-shards",
            "2",
            "--output-dir",
            "/tmp/jss-selector-smoke",
        ],
        [
            "docker",
            "run",
            "--rm",
            "--init",
            "-e",
            f"GIT_SHA={git_sha}",
            "-e",
            "CITREES_SOURCE_CLEAN=1",
            image_tag,
            "python",
            "-m",
            "paper.jss.replication.shards",
            "behavior-shard",
            "--profile",
            "smoke",
            "--shard-index",
            "0",
            "--num-shards",
            "8",
            "--output-dir",
            "/tmp/jss-behavior-smoke",
        ],
        [
            "docker",
            "run",
            "--rm",
            "--init",
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
                    },
                    {
                        "Sid": "DenyUnconditionalJSSWrites",
                        "Effect": "Deny",
                        "Principal": "*",
                        "Action": "s3:PutObject",
                        "Resource": f"arn:aws:s3:::{bucket_name}/jss/replication/*",
                        "Condition": {"Null": {"s3:if-none-match": "true"}},
                    },
                ],
            }
        ),
    )

    return bucket_name


def _publish_immutable_bytes(
    client: Any,
    *,
    bucket: str,
    key: str,
    payload: bytes,
    content_type: str,
    metadata: dict[str, str],
) -> None:
    """Publish exact bytes once and require an identical readback."""
    from botocore.exceptions import ClientError

    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType=content_type,
            Metadata=metadata,
            IfNoneMatch="*",
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code not in {"412", "PreconditionFailed"} and status != 412:
            raise

    response = client.get_object(Bucket=bucket, Key=key)
    if response["Body"].read() != payload:
        raise RuntimeError(
            f"Content-addressed object differs from local bytes: s3://{bucket}/{key}"
        )
    if response.get("Metadata", {}) != metadata:
        raise RuntimeError(f"Published object metadata is invalid: s3://{bucket}/{key}")


def publish_rerun_manifest(
    manifest_path: Path,
    canonical_manifest_path: Path,
    runtime_contract_path: Path,
    gate_receipt_path: Path,
    *,
    region: str = DEFAULT_REGION,
) -> dict[str, str | int]:
    """Publish one canonical campaign and its exact active-account shard."""
    from paper.benchmark.experiments.r_cforest_reproducibility import (
        GATE_RECEIPT_PROFILE,
        GATE_RECEIPT_SCHEMA_VERSION,
        gate_receipt_s3_key,
        parse_gate_receipt,
    )
    from paper.benchmark.pipeline.manifest import (
        canonical_manifest_s3_key,
        manifest_s3_key,
        parse_rerun_manifest,
        validate_canonical_campaign,
        verify_account_manifest_shard,
    )
    from paper.benchmark.pipeline.runtime_contract import (
        parse_runtime_contract,
        runtime_contract_s3_key,
        runtime_contract_sha256,
    )

    canonical_payload = canonical_manifest_path.read_bytes()
    canonical = parse_rerun_manifest(canonical_payload)
    validate_canonical_campaign(canonical)
    account_id = get_aws_account_id()
    manifest_payload = manifest_path.read_bytes()
    manifest = verify_account_manifest_shard(
        canonical,
        account_id=account_id,
        payload=manifest_payload,
    )

    runtime_payload = runtime_contract_path.read_bytes()
    runtime_contract = parse_runtime_contract(runtime_payload)
    runtime_sha256 = runtime_contract_sha256(runtime_contract)
    for name, candidate in (
        ("canonical manifest", canonical),
        ("account manifest", manifest),
    ):
        if candidate.runtime_contract_sha256 != runtime_sha256:
            raise ValueError(
                f"{name} runtime contract digest does not match the supplied "
                f"contract: {candidate.runtime_contract_sha256} != {runtime_sha256}"
            )

    gate_receipt_payload = gate_receipt_path.read_bytes()
    gate_receipt = parse_gate_receipt(
        gate_receipt_payload,
        manifest=canonical,
        runtime_contract=runtime_contract,
    )
    if gate_receipt["account_manifest_sha256"].get(account_id) != manifest.sha256:
        raise ValueError(
            "gate receipt does not bind the active account manifest: "
            f"account={account_id}, manifest={manifest.sha256}"
        )

    gate_receipt_sha256 = hashlib.sha256(gate_receipt_payload).hexdigest()
    gate_receipt_key = gate_receipt_s3_key(gate_receipt_sha256)
    runtime_key = runtime_contract_s3_key(runtime_sha256)
    canonical_key = canonical_manifest_s3_key(canonical.sha256)
    manifest_key = manifest_s3_key(manifest.sha256)
    bucket = ensure_s3_bucket(region)
    s3 = boto3.client("s3", region_name=region)

    runtime_metadata = {
        "profile": str(runtime_contract["profile"]),
        "schema-version": str(runtime_contract["schema_version"]),
        "sha256": runtime_sha256,
    }
    _publish_immutable_bytes(
        s3,
        bucket=bucket,
        key=runtime_key,
        payload=runtime_payload,
        content_type="application/json",
        metadata=runtime_metadata,
    )
    parse_runtime_contract(runtime_payload, expected_sha256=runtime_sha256)

    canonical_metadata = {
        "campaign-sha256": canonical.campaign_sha256,
        "cell-count": str(len(canonical.cells)),
        "profile": "canonical-campaign",
        "runtime-contract-key": runtime_key,
        "runtime-contract-sha256": runtime_sha256,
        "sha256": canonical.sha256,
        "target-aws-account-ids": ",".join(canonical.account_ids),
    }
    _publish_immutable_bytes(
        s3,
        bucket=bucket,
        key=canonical_key,
        payload=canonical_payload,
        content_type="text/csv",
        metadata=canonical_metadata,
    )
    validate_canonical_campaign(
        parse_rerun_manifest(canonical_payload, expected_sha256=canonical.sha256)
    )

    gate_receipt_metadata = {
        "campaign-sha256": canonical.campaign_sha256,
        "manifest-sha256": canonical.sha256,
        "profile": GATE_RECEIPT_PROFILE,
        "runtime-contract-sha256": runtime_sha256,
        "schema-version": str(GATE_RECEIPT_SCHEMA_VERSION),
        "sha256": gate_receipt_sha256,
        "status": str(gate_receipt["report"]["status"]),
    }
    _publish_immutable_bytes(
        s3,
        bucket=bucket,
        key=gate_receipt_key,
        payload=gate_receipt_payload,
        content_type="application/json",
        metadata=gate_receipt_metadata,
    )
    parse_gate_receipt(
        gate_receipt_payload,
        manifest=canonical,
        runtime_contract=runtime_contract,
        expected_sha256=gate_receipt_sha256,
    )

    manifest_metadata = {
        "campaign-sha256": manifest.campaign_sha256,
        "canonical-manifest-key": canonical_key,
        "canonical-manifest-sha256": canonical.sha256,
        "cell-count": str(len(manifest.cells)),
        "gate-receipt-key": gate_receipt_key,
        "gate-receipt-sha256": gate_receipt_sha256,
        "runtime-contract-key": runtime_key,
        "runtime-contract-sha256": runtime_sha256,
        "sha256": manifest.sha256,
        "target-aws-account-id": account_id,
    }
    _publish_immutable_bytes(
        s3,
        bucket=bucket,
        key=manifest_key,
        payload=manifest_payload,
        content_type="text/csv",
        metadata=manifest_metadata,
    )
    verify_account_manifest_shard(
        canonical,
        account_id=account_id,
        payload=manifest_payload,
    )

    return {
        "bucket": bucket,
        "key": manifest_key,
        "sha256": manifest.sha256,
        "campaign_sha256": canonical.campaign_sha256,
        "canonical_manifest_s3_key": canonical_key,
        "canonical_manifest_sha256": canonical.sha256,
        "gate_receipt_s3_key": gate_receipt_key,
        "gate_receipt_sha256": gate_receipt_sha256,
        "runtime_contract_s3_key": runtime_key,
        "runtime_contract_sha256": runtime_sha256,
        "cells": len(manifest.cells),
        "canonical_cells": len(canonical.cells),
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


def _load_ecr_manifest(
    ecr: Any,
    repository_name: str,
    image_digest: str,
    *,
    image_tag: str | None = None,
) -> dict[str, Any]:
    image_digest = _require_sha256_digest(image_digest, context="ECR image identity")
    if image_tag is not None and not _FULL_GIT_SHA_PATTERN.fullmatch(image_tag):
        raise RuntimeError("ECR image tag must be one full Git revision")
    image_id = {"imageDigest": image_digest} if image_tag is None else {"imageTag": image_tag}
    response = ecr.batch_get_image(
        repositoryName=repository_name,
        imageIds=[image_id],
        acceptedMediaTypes=_ECR_MANIFEST_MEDIA_TYPES,
    )
    failures = response.get("failures", [])
    images = response.get("images", [])
    if failures or len(images) != 1:
        raise RuntimeError(
            f"Could not load remote manifest {image_digest}"
            f"{f' through tag {image_tag}' if image_tag is not None else ''}: "
            f"failures={failures!r}"
        )
    image = images[0]
    returned_image_id = image.get("imageId", {})
    observed_digest = returned_image_id.get("imageDigest")
    if observed_digest != image_digest:
        raise RuntimeError(f"ECR returned manifest {observed_digest!r}, expected {image_digest}")
    if image_tag is not None and returned_image_id.get("imageTag") != image_tag:
        raise RuntimeError(
            f"ECR returned tag {returned_image_id.get('imageTag')!r}, expected {image_tag}"
        )
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
    image_tag: str | None = None,
) -> tuple[str, int | None]:
    if depth > 1:
        raise RuntimeError("ECR image index nesting exceeds one platform-selection level")
    manifest = _load_ecr_manifest(
        ecr,
        repository_name,
        image_digest,
        image_tag=image_tag,
    )
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


def _remote_image_revision(
    ecr: Any,
    repository_name: str,
    image_digest: str,
    *,
    image_tag: str | None = None,
) -> str:
    config_digest, expected_size = _remote_config_descriptor(
        ecr,
        repository_name,
        image_digest,
        image_tag=image_tag,
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
    expected_git_sha: str,
    *,
    region: str = DEFAULT_REGION,
) -> str:
    """Require a remote image digest built from one expected clean revision."""
    get_frozen_git_sha()
    if not _FULL_GIT_SHA_PATTERN.fullmatch(expected_git_sha):
        raise ValueError("expected_git_sha must be a full hexadecimal commit ID")
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
    if expected_git_sha not in tags:
        raise RuntimeError(
            f"image digest is not tagged with expected source revision {expected_git_sha}"
        )
    remote_revision = _remote_image_revision(
        ecr,
        repo_name,
        image_digest,
        image_tag=expected_git_sha,
    )
    if remote_revision != expected_git_sha:
        raise RuntimeError(
            f"remote OCI revision label is {remote_revision!r}, expected {expected_git_sha}"
        )
    return expected_git_sha


def campaign_instance_profile_name(
    *,
    output_prefix: str,
    campaign_sha256: str,
    read_keys: Sequence[str],
    write_prefixes: Sequence[str],
) -> str:
    """Derive one IAM identity from immutable campaign access."""
    normalized_prefix = _normalize_artifact_prefix(output_prefix)
    campaign_sha256 = validate_manifest_sha256(campaign_sha256)
    normalized_read_keys = _normalize_campaign_read_keys(
        read_keys,
        output_prefix=normalized_prefix,
    )
    normalized_write_prefixes = tuple(
        sorted({_normalize_artifact_prefix(prefix) for prefix in write_prefixes})
    )
    if not normalized_write_prefixes:
        raise ValueError("campaign write prefixes must not be empty")
    if any(
        prefix != normalized_prefix and not prefix.startswith(f"{normalized_prefix}/")
        for prefix in normalized_write_prefixes
    ):
        raise ValueError("campaign write prefixes must lie within the output prefix")
    identity = hashlib.sha256(
        "\0".join(
            (
                campaign_sha256,
                normalized_prefix,
                "read-keys",
                *normalized_read_keys,
                "write-prefixes",
                *normalized_write_prefixes,
            )
        ).encode()
    ).hexdigest()
    return f"citrees-campaign-{identity[:32]}"


def _normalize_campaign_read_keys(
    read_keys: Sequence[str],
    *,
    output_prefix: str,
) -> tuple[str, ...]:
    """Validate exact immutable S3 object keys for one campaign."""
    from paper.benchmark.experiments.r_cforest_reproducibility import (
        GATE_RECEIPT_S3_PREFIX,
    )
    from paper.benchmark.pipeline.manifest import (
        CANONICAL_MANIFEST_S3_PREFIX,
        MANIFEST_S3_PREFIX,
    )
    from paper.benchmark.pipeline.runtime_contract import (
        RUNTIME_CONTRACT_S3_PREFIX,
    )

    if isinstance(read_keys, str):
        raise TypeError("campaign read keys must be a sequence of strings")
    control_prefixes = {
        f"{CANONICAL_MANIFEST_S3_PREFIX}/": (".csv", validate_manifest_sha256),
        f"{MANIFEST_S3_PREFIX}/": (".csv", validate_manifest_sha256),
        f"{RUNTIME_CONTRACT_S3_PREFIX}/": (
            ".json",
            validate_manifest_sha256,
        ),
        f"{GATE_RECEIPT_S3_PREFIX}/": (
            ".json",
            validate_manifest_sha256,
        ),
    }
    normalized: set[str] = set()
    for key in read_keys:
        exact_key = _normalize_artifact_prefix(key)
        if exact_key != key:
            raise ValueError("campaign read keys must already be normalized")
        if "*" in exact_key or "?" in exact_key or "${" in exact_key:
            raise ValueError(
                "campaign read keys must not contain IAM wildcards or policy variables"
            )
        matching_control_prefix = next(
            (
                (prefix, suffix, validator)
                for prefix, (suffix, validator) in control_prefixes.items()
                if exact_key.startswith(prefix)
            ),
            None,
        )
        if matching_control_prefix is not None:
            prefix, suffix, validator = matching_control_prefix
            if not exact_key.endswith(suffix):
                raise ValueError("campaign control read keys must be content-addressed objects")
            digest = exact_key.removeprefix(prefix).removesuffix(suffix)
            validator(digest)
        elif not exact_key.startswith(f"{output_prefix}/"):
            raise ValueError(
                "campaign read keys must belong to an approved control namespace "
                "or the campaign output prefix"
            )
        normalized.add(exact_key)
    return tuple(sorted(normalized))


def _iam_policy_document(value: object, label: str) -> dict[str, Any]:
    """Parse one IAM policy document returned by AWS."""
    if isinstance(value, str):
        try:
            value = json.loads(urllib.parse.unquote(value))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def _iam_role(response: object, role_name: str) -> dict[str, Any]:
    """Return one exact role object from an IAM response."""
    if not isinstance(response, dict) or not isinstance(response.get("Role"), dict):
        raise RuntimeError(f"IAM role readback is invalid for {role_name}")
    role = response["Role"]
    if role.get("RoleName") != role_name:
        raise RuntimeError(f"IAM role readback differs for {role_name}")
    return role


def _list_role_policy_names(iam: Any, role_name: str) -> tuple[str, ...]:
    """List every inline policy attached to one role."""
    names: list[str] = []
    marker: str | None = None
    seen_markers: set[str] = set()
    while True:
        arguments: dict[str, object] = {"RoleName": role_name}
        if marker is not None:
            arguments["Marker"] = marker
        response = iam.list_role_policies(**arguments)
        page_names = response.get("PolicyNames")
        if not isinstance(page_names, list) or any(
            not isinstance(name, str) for name in page_names
        ):
            raise RuntimeError(f"IAM inline policy inventory is invalid for {role_name}")
        names.extend(page_names)
        if not response.get("IsTruncated", False):
            break
        next_marker = response.get("Marker")
        if not isinstance(next_marker, str) or not next_marker or next_marker in seen_markers:
            raise RuntimeError(f"IAM inline policy pagination is invalid for {role_name}")
        seen_markers.add(next_marker)
        marker = next_marker
    if len(names) != len(set(names)):
        raise RuntimeError(f"IAM inline policy inventory contains duplicates for {role_name}")
    return tuple(sorted(names))


def _list_attached_role_policy_arns(iam: Any, role_name: str) -> tuple[str, ...]:
    """List every managed policy attached to one role."""
    arns: list[str] = []
    marker: str | None = None
    seen_markers: set[str] = set()
    while True:
        arguments: dict[str, object] = {"RoleName": role_name}
        if marker is not None:
            arguments["Marker"] = marker
        response = iam.list_attached_role_policies(**arguments)
        policies = response.get("AttachedPolicies")
        if not isinstance(policies, list):
            raise RuntimeError(f"IAM managed policy inventory is invalid for {role_name}")
        for policy in policies:
            if not isinstance(policy, dict) or not isinstance(policy.get("PolicyArn"), str):
                raise RuntimeError(f"IAM managed policy inventory is invalid for {role_name}")
            arns.append(policy["PolicyArn"])
        if not response.get("IsTruncated", False):
            break
        next_marker = response.get("Marker")
        if not isinstance(next_marker, str) or not next_marker or next_marker in seen_markers:
            raise RuntimeError(f"IAM managed policy pagination is invalid for {role_name}")
        seen_markers.add(next_marker)
        marker = next_marker
    if len(arns) != len(set(arns)):
        raise RuntimeError(f"IAM managed policy inventory contains duplicates for {role_name}")
    return tuple(sorted(arns))


def _instance_profile_role_names(response: object, profile_name: str) -> tuple[str, ...]:
    """Return the exact role inventory from one instance-profile response."""
    if not isinstance(response, dict) or not isinstance(response.get("InstanceProfile"), dict):
        raise RuntimeError(f"IAM instance profile readback is invalid for {profile_name}")
    roles = response["InstanceProfile"].get("Roles")
    if not isinstance(roles, list):
        raise RuntimeError(f"IAM instance profile readback is invalid for {profile_name}")
    role_names: list[str] = []
    for role in roles:
        if not isinstance(role, dict) or not isinstance(role.get("RoleName"), str):
            raise RuntimeError(f"IAM instance profile readback is invalid for {profile_name}")
        role_names.append(role["RoleName"])
    if len(role_names) != len(set(role_names)):
        raise RuntimeError(f"IAM instance profile contains duplicate roles for {profile_name}")
    return tuple(sorted(role_names))


def ensure_campaign_iam_profile(
    *,
    output_prefix: str,
    campaign_sha256: str,
    read_keys: Sequence[str],
    write_prefixes: Sequence[str],
    region: str = DEFAULT_REGION,
) -> str:
    """Ensure one campaign-bound role and return its instance-profile name."""
    normalized_prefix = _normalize_artifact_prefix(output_prefix)
    campaign_sha256 = validate_manifest_sha256(campaign_sha256)
    normalized_read_keys = _normalize_campaign_read_keys(
        read_keys,
        output_prefix=normalized_prefix,
    )
    normalized_write_prefixes = tuple(
        sorted({_normalize_artifact_prefix(prefix) for prefix in write_prefixes})
    )
    if not normalized_write_prefixes:
        raise ValueError("campaign write prefixes must not be empty")
    if any(
        prefix != normalized_prefix and not prefix.startswith(f"{normalized_prefix}/")
        for prefix in normalized_write_prefixes
    ):
        raise ValueError("campaign write prefixes must lie within the output prefix")
    profile_name = campaign_instance_profile_name(
        output_prefix=normalized_prefix,
        campaign_sha256=campaign_sha256,
        read_keys=normalized_read_keys,
        write_prefixes=normalized_write_prefixes,
    )
    role_name = profile_name
    policy_name = f"{profile_name}-runtime"
    iam = boto3.client("iam", region_name=region)
    account_id = get_aws_account_id()
    bucket_name = get_resource_name(account_id)
    bucket_arn = f"arn:aws:s3:::{bucket_name}"
    output_arn = f"{bucket_arn}/{normalized_prefix}/*"
    read_arns = [f"{bucket_arn}/{key}" for key in normalized_read_keys]
    write_arns = [f"{bucket_arn}/{prefix}/*" for prefix in normalized_write_prefixes]

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
                    *read_arns,
                    output_arn,
                ],
            },
            {
                "Sid": "S3WriteArtifacts",
                "Effect": "Allow",
                "Action": "s3:PutObject",
                "Resource": write_arns,
                "Condition": {"StringEquals": {"s3:if-none-match": "*"}},
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
        existing_role = _iam_role(iam.get_role(RoleName=role_name), role_name)
        step(f"IAM role exists: {role_name}")
    except iam.exceptions.NoSuchEntityException:
        step(f"Creating IAM role: {role_name}")
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description=f"citrees campaign {campaign_sha256[:16]} runtime",
        )
    else:
        if existing_role.get("PermissionsBoundary") is not None:
            raise RuntimeError(f"campaign IAM role {role_name} has a permissions boundary")
        iam.update_assume_role_policy(
            RoleName=role_name,
            PolicyDocument=json.dumps(trust_policy),
        )

    for unexpected_policy_name in _list_role_policy_names(iam, role_name):
        if unexpected_policy_name == policy_name:
            continue
        iam.delete_role_policy(
            RoleName=role_name,
            PolicyName=unexpected_policy_name,
        )
        step(f"Deleted unexpected IAM inline policy: {unexpected_policy_name}")
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=policy_name,
        PolicyDocument=json.dumps(runtime_policy),
    )
    step(f"IAM policy attached: {policy_name}")

    ssm_policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
    attached_policy_arns = _list_attached_role_policy_arns(iam, role_name)
    for unexpected_policy_arn in attached_policy_arns:
        if unexpected_policy_arn == ssm_policy_arn:
            continue
        iam.detach_role_policy(
            RoleName=role_name,
            PolicyArn=unexpected_policy_arn,
        )
        step(f"Detached unexpected IAM managed policy: {unexpected_policy_arn}")
    if ssm_policy_arn not in attached_policy_arns:
        iam.attach_role_policy(RoleName=role_name, PolicyArn=ssm_policy_arn)
        step("Attached AmazonSSMManagedInstanceCore")

    profile_changed = False
    try:
        response = iam.get_instance_profile(InstanceProfileName=profile_name)
        step(f"Instance profile exists: {profile_name}")
        profile_role_names = _instance_profile_role_names(response, profile_name)
        for unexpected_role_name in profile_role_names:
            if unexpected_role_name == role_name:
                continue
            iam.remove_role_from_instance_profile(
                InstanceProfileName=profile_name,
                RoleName=unexpected_role_name,
            )
            step(f"Removed unexpected instance-profile role: {unexpected_role_name}")
            profile_changed = True
        if role_name not in profile_role_names:
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

    verified_role = _iam_role(iam.get_role(RoleName=role_name), role_name)
    if verified_role.get("PermissionsBoundary") is not None:
        raise RuntimeError(f"campaign IAM role {role_name} has a permissions boundary")
    observed_trust_policy = _iam_policy_document(
        verified_role.get("AssumeRolePolicyDocument"),
        f"campaign IAM role {role_name} trust policy",
    )
    if observed_trust_policy != trust_policy:
        raise RuntimeError(f"campaign IAM role {role_name} trust policy differs")

    inline_policy_names = _list_role_policy_names(iam, role_name)
    if inline_policy_names != (policy_name,):
        raise RuntimeError(f"campaign IAM role {role_name} inline policy inventory differs")
    inline_policy = iam.get_role_policy(
        RoleName=role_name,
        PolicyName=policy_name,
    )
    if inline_policy.get("RoleName") != role_name or inline_policy.get("PolicyName") != policy_name:
        raise RuntimeError(f"campaign IAM role {role_name} runtime policy identity differs")
    observed_runtime_policy = _iam_policy_document(
        inline_policy.get("PolicyDocument"),
        f"campaign IAM role {role_name} runtime policy",
    )
    if observed_runtime_policy != runtime_policy:
        raise RuntimeError(f"campaign IAM role {role_name} runtime policy differs")

    attached_policy_arns = _list_attached_role_policy_arns(iam, role_name)
    if attached_policy_arns != (ssm_policy_arn,):
        raise RuntimeError(f"campaign IAM role {role_name} managed policy inventory differs")

    verified_profile = iam.get_instance_profile(InstanceProfileName=profile_name)
    if _instance_profile_role_names(verified_profile, profile_name) != (role_name,):
        raise RuntimeError(f"campaign instance profile {profile_name} role inventory differs")

    return profile_name


def ensure_security_group(region: str = DEFAULT_REGION) -> str:
    """Ensure the citrees security group exists with correct rules.

    Creates a security group ``citrees-sg`` in the default VPC that allows:
    - Inbound TCP 8000 from within the group (worker → API)
    - Inbound TCP 8000 from the caller's public IP (CLI → API)
    - All outbound (VPC default)

    Returns the security group ID.
    """
    ec2 = boto3.client("ec2", region_name=region)
    sg_name = "citrees-sg"

    vpc_response = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])
    default_vpcs = vpc_response.get("Vpcs")
    if not isinstance(default_vpcs, list) or len(default_vpcs) != 1:
        count = len(default_vpcs) if isinstance(default_vpcs, list) else 0
        raise RuntimeError(f"expected exactly one default VPC, found {count}")
    default_vpc = default_vpcs[0]
    vpc_id = default_vpc.get("VpcId") if isinstance(default_vpc, dict) else None
    if not isinstance(vpc_id, str) or not vpc_id:
        raise RuntimeError("default VPC response is missing VpcId")

    response = ec2.describe_security_groups(
        Filters=[
            {"Name": "group-name", "Values": [sg_name]},
            {"Name": "vpc-id", "Values": [vpc_id]},
        ]
    )
    security_groups = response.get("SecurityGroups")
    if not isinstance(security_groups, list):
        raise RuntimeError("security group lookup returned an invalid response")
    if len(security_groups) > 1:
        raise RuntimeError(f"multiple citrees security groups found in default VPC {vpc_id}")

    if security_groups:
        security_group = security_groups[0]
        if not isinstance(security_group, dict):
            raise RuntimeError("security group lookup returned an invalid response")
        if security_group.get("VpcId") != vpc_id:
            raise RuntimeError(f"citrees security group is outside default VPC {vpc_id}")
        sg_id = security_group.get("GroupId")
        if not isinstance(sg_id, str) or not sg_id:
            raise RuntimeError("citrees security group response is missing GroupId")
        step(f"Security group exists: {sg_name} ({sg_id})")

        my_ip = get_public_ip()
        my_cidr = f"{my_ip}/32"
        existing_rules = security_group.get("IpPermissions", [])
        if not isinstance(existing_rules, list):
            raise RuntimeError("citrees security group ingress response is invalid")

        def allows_ssh(rule: object) -> bool:
            if not isinstance(rule, dict):
                raise RuntimeError("citrees security group ingress response is invalid")
            protocol = rule.get("IpProtocol")
            if protocol == "-1":
                return True
            if protocol != "tcp":
                return False
            from_port = rule.get("FromPort")
            to_port = rule.get("ToPort")
            return (
                isinstance(from_port, int)
                and isinstance(to_port, int)
                and from_port <= 22 <= to_port
            )

        revocations = [rule for rule in existing_rules if allows_ssh(rule)]
        caller_ranges = [
            ip_range
            for rule in existing_rules
            if isinstance(rule, dict)
            and rule.get("IpProtocol") == "tcp"
            and rule.get("FromPort") == 8000
            and rule.get("ToPort") == 8000
            for ip_range in rule.get("IpRanges", [])
            if ip_range.get("Description") == "API from caller"
        ]
        stale_ranges = [ip_range for ip_range in caller_ranges if ip_range.get("CidrIp") != my_cidr]
        if stale_ranges:
            revocations.append(
                {
                    "IpProtocol": "tcp",
                    "FromPort": 8000,
                    "ToPort": 8000,
                    "IpRanges": stale_ranges,
                }
            )
        if revocations:
            ec2.revoke_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=revocations,
            )

        if not any(ip_range.get("CidrIp") == my_cidr for ip_range in caller_ranges):
            ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 8000,
                        "ToPort": 8000,
                        "IpRanges": [
                            {
                                "CidrIp": my_cidr,
                                "Description": "API from caller",
                            }
                        ],
                    }
                ],
            )
            step(f"Updated port 8000 rule to {my_cidr}")

        has_api_self_rule = any(
            isinstance(rule, dict)
            and rule.get("IpProtocol") == "tcp"
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

    step(f"Creating security group: {sg_name}")
    create_resp = ec2.create_security_group(
        GroupName=sg_name,
        Description="citrees API + worker instances",
        VpcId=vpc_id,
    )
    sg_id = create_resp.get("GroupId")
    if not isinstance(sg_id, str) or not sg_id:
        raise RuntimeError("created citrees security group response is missing GroupId")

    my_ip = get_public_ip()

    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 8000,
                "ToPort": 8000,
                "UserIdGroupPairs": [
                    {"GroupId": sg_id, "Description": "API from citrees instances"}
                ],
            },
            {
                "IpProtocol": "tcp",
                "FromPort": 8000,
                "ToPort": 8000,
                "IpRanges": [{"CidrIp": f"{my_ip}/32", "Description": "API from caller"}],
            },
        ],
    )

    success(f"Created security group: {sg_name} ({sg_id})")
    step(f"Inbound 8000 from group and {my_ip}/32")

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
