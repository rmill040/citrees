"""Tests for safe distributed benchmark launch configuration."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError, ReadTimeoutError

from paper.benchmark.cli.infra import (
    launch_mechanism_workers_cmd,
    launch_workers_cmd,
)
from paper.benchmark.experiments.cif_mechanism_ablation import (
    mechanism_specification_sha256,
)
from paper.benchmark.infra import aws as aws_infra
from paper.benchmark.infra import ec2 as ec2_infra
from paper.benchmark.infra.ec2 import (
    ApiScope,
    _api_client_token,
    _make_api_user_data,
    _make_mechanism_user_data,
    _make_worker_user_data,
    _validate_queue_scope,
    get_api_scope,
    launch_api,
    launch_mechanism_workers,
    launch_workers,
    validate_image_digest_uri,
)

pytestmark = pytest.mark.paper

DIGEST_URI = (
    "123456789012.dkr.ecr.us-east-1.amazonaws.com/citrees"
    "@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)
MANIFEST_SHA256 = "b" * 64
CAMPAIGN_SHA256 = "e" * 64
MANIFEST_KEY = f"rerun-manifests/{MANIFEST_SHA256}.csv"
SSM_POLICY_ARN = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
CAMPAIGN_TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}


class _WorkerLaunchKwargs(TypedDict):
    n: int
    instance_type: str
    image_uri: str
    artifact_prefix: str
    launch_id: str
    manifest_path: Path
    stage: str
    spot: bool


class _MemoryS3:
    """In-memory S3 double with conditional immutable writes."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.events = events

    def put_object(self, **kwargs: Any) -> dict[str, object]:
        key = str(kwargs["Key"])
        if self.events is not None:
            self.events.append(f"s3:{key}")
        if kwargs.get("IfNoneMatch") == "*" and key in self.objects:
            raise ClientError(
                {
                    "Error": {"Code": "PreconditionFailed"},
                    "ResponseMetadata": {"HTTPStatusCode": 412},
                },
                "PutObject",
            )
        body = kwargs["Body"]
        payload = body if isinstance(body, bytes) else str(body).encode("utf-8")
        self.objects[key] = (payload, dict(kwargs.get("Metadata", {})))
        return {}

    def get_object(self, **kwargs: Any) -> dict[str, object]:
        key = str(kwargs["Key"])
        if key not in self.objects:
            raise ClientError(
                {
                    "Error": {"Code": "NoSuchKey"},
                    "ResponseMetadata": {"HTTPStatusCode": 404},
                },
                "GetObject",
            )
        payload, metadata = self.objects[key]
        return {"Body": io.BytesIO(payload), "Metadata": metadata}


class _NoSuchEntityError(Exception):
    pass


def _campaign_iam_client(
    profile_name: str,
    *,
    initial_inline_policy_names: tuple[str, ...] = (),
    initial_attached_policy_arns: tuple[str, ...] = (),
    initial_profile_roles: tuple[str, ...] | None = None,
    readback_trust_policy: object | None = None,
    readback_inline_policy_names: tuple[str, ...] | None = None,
    readback_runtime_policy: object | None = None,
    readback_attached_policy_arns: tuple[str, ...] | None = None,
    readback_profile_roles: tuple[str, ...] | None = None,
) -> MagicMock:
    policy_name = f"{profile_name}-runtime"
    initial_profile_roles = initial_profile_roles or (profile_name,)
    client = MagicMock()
    client.exceptions.NoSuchEntityException = _NoSuchEntityError
    client.get_role.side_effect = [
        {
            "Role": {
                "RoleName": profile_name,
                "AssumeRolePolicyDocument": {"Version": "stale"},
            }
        },
        {
            "Role": {
                "RoleName": profile_name,
                "AssumeRolePolicyDocument": (
                    CAMPAIGN_TRUST_POLICY
                    if readback_trust_policy is None
                    else readback_trust_policy
                ),
            }
        },
    ]
    client.list_role_policies.side_effect = [
        {"PolicyNames": list(initial_inline_policy_names)},
        {
            "PolicyNames": list(
                (policy_name,)
                if readback_inline_policy_names is None
                else readback_inline_policy_names
            )
        },
    ]
    client.list_attached_role_policies.side_effect = [
        {
            "AttachedPolicies": [
                {"PolicyName": arn.rsplit("/", maxsplit=1)[-1], "PolicyArn": arn}
                for arn in initial_attached_policy_arns
            ]
        },
        {
            "AttachedPolicies": [
                {"PolicyName": arn.rsplit("/", maxsplit=1)[-1], "PolicyArn": arn}
                for arn in (
                    (SSM_POLICY_ARN,)
                    if readback_attached_policy_arns is None
                    else readback_attached_policy_arns
                )
            ]
        },
    ]
    client.get_instance_profile.side_effect = [
        {
            "InstanceProfile": {
                "Roles": [{"RoleName": role_name} for role_name in initial_profile_roles]
            }
        },
        {
            "InstanceProfile": {
                "Roles": [
                    {"RoleName": role_name}
                    for role_name in (
                        (profile_name,)
                        if readback_profile_roles is None
                        else readback_profile_roles
                    )
                ]
            }
        },
    ]

    def get_role_policy(**_: object) -> dict[str, object]:
        policy = readback_runtime_policy
        if policy is None:
            policy = json.loads(client.put_role_policy.call_args.kwargs["PolicyDocument"])
        return {
            "RoleName": profile_name,
            "PolicyName": policy_name,
            "PolicyDocument": policy,
        }

    client.get_role_policy.side_effect = get_role_policy
    return client


def _patch_campaign_iam(
    monkeypatch: pytest.MonkeyPatch,
    client: MagicMock,
) -> None:
    monkeypatch.setattr(aws_infra, "get_aws_account_id", lambda: "123456789012")
    monkeypatch.setattr(aws_infra.boto3, "client", lambda *args, **kwargs: client)
    monkeypatch.setattr(aws_infra.time, "sleep", lambda seconds: None)


@pytest.mark.parametrize(
    "command",
    [launch_workers_cmd, launch_mechanism_workers_cmd],
)
def test_distributed_workers_default_to_on_demand(command: object) -> None:
    assert inspect.signature(command).parameters["spot"].default is False


def test_mechanism_launch_has_no_mutable_output_or_overwrite_controls() -> None:
    for command in (launch_mechanism_workers_cmd, launch_mechanism_workers):
        parameters = inspect.signature(command).parameters
        assert "output_uri" not in parameters
        assert "force" not in parameters


def test_distributed_launch_requires_immutable_image_digest() -> None:
    assert validate_image_digest_uri(f" {DIGEST_URI} ") == DIGEST_URI

    for invalid in (
        "",
        "repository:latest",
        "repository:abc123",
        "repository@sha256:abc",
        "repository@sha256:" + "g" * 64,
    ):
        with pytest.raises(ValueError, match="immutable"):
            validate_image_digest_uri(invalid)


def test_candidate_image_pins_complete_statistical_runtime() -> None:
    dockerfile = Path("paper/benchmark/infra/docker/Dockerfile").read_text()
    dockerignore = Path(".dockerignore").read_text().splitlines()

    assert "FROM rocker/r-ver:4.5.2@sha256:" in dockerfile
    assert "SOURCE_GIT_SHA" in dockerfile
    assert 'org.opencontainers.image.revision="$SOURCE_GIT_SHA"' in dockerfile
    assert "snapshot.ubuntu.com/ubuntu/20260801T000000Z" in dockerfile
    assert "build-essential=12.10ubuntu1" in dockerfile
    assert "/partykit_1.2-24.tar.gz" in dockerfile
    assert "packagemanager.posit.co/cran/2026-08-01/src/contrib/inum_1.0-5.tar.gz" in dockerfile
    assert (
        "9d1b4365f8e03f4e1e4989b7f91ea9e65c25dc171d807db9b206addbc0eb65fe /tmp/inum_1.0-5.tar.gz"
        in dockerfile
    )
    assert 'partykit="1.2.24"' in dockerfile
    assert "uv python install 3.12.7" in dockerfile
    assert "UV_PYTHON=3.12.7" in dockerfile
    assert 'version("rpy2") == "3.6.7"' in dockerfile
    assert 'version("scikit-learn") == "1.8.0"' in dockerfile
    assert "COPY paper/jss/replication ./paper/jss/replication" in dockerfile
    assert "paper/jss/" not in dockerignore
    assert "!paper/jss/replication/" in dockerignore
    assert "!paper/jss/replication/**" in dockerignore
    for variable in (
        "BLIS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "R_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        assert f"ENV {variable}=1" in dockerfile
    assert "ENV NUMBA_DISABLE_JIT=0" in dockerfile
    assert "ENV PYTHONHASHSEED=0" in dockerfile
    assert "COPY . ." not in dockerfile


def test_queue_scope_normalizes_prefix_and_stage() -> None:
    scope = _validate_queue_scope(
        " repairs/r-baselines/run-001/ ",
        "rankings",
    )

    assert scope == (
        "repairs/r-baselines/run-001",
        "rankings",
    )


def test_api_client_token_is_deterministic_over_complete_request() -> None:
    request = {
        "ami_id": "ami-test",
        "instance_profile_name": "citrees-campaign-test",
        "instance_type": "m5.large",
        "security_group_id": "sg-test",
        "user_data": "#!/bin/bash\ntrue\n",
    }
    token = _api_client_token(**request)

    assert token == _api_client_token(**request)
    assert len(token) == 64
    assert token.startswith("citrees-api-")
    for field in request:
        changed = dict(request)
        changed[field] += "-different"
        assert _api_client_token(**changed) != token


@pytest.mark.parametrize(
    ("artifact_prefix", "stage"),
    [
        ("", "rankings"),
        ("repairs/run", "both"),
    ],
)
def test_queue_scope_rejects_unsafe_or_ambiguous_values(
    artifact_prefix: str,
    stage: str,
) -> None:
    with pytest.raises(ValueError):
        _validate_queue_scope(artifact_prefix, stage)

    with pytest.raises(ValueError, match="below repairs"):
        _validate_queue_scope("experiments/run-001", "rankings")


def test_mechanism_user_data_uses_derived_distributed_scope() -> None:
    specification_sha256 = mechanism_specification_sha256(
        tasks=("classification", "regression"),
        source="real",
        datasets=(),
        seeds=(0, 1),
        folds=(0, 1),
        model_variants=("cif_default",),
        ranking_variants=("split_importance", "split_count"),
        n_jobs=-1,
        downstream_n_jobs=1,
    )
    script = _make_mechanism_user_data(
        region="us-east-1",
        ecr_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com",
        image_uri=DIGEST_URI,
        bucket="citrees-123456789012",
        git_sha="abc123",
        shard_index=2,
        num_shards=8,
        tasks=("classification", "regression"),
        source="real",
        datasets=(),
        seeds=(0, 1),
        folds=(0, 1),
        model_variants=("cif_default",),
        ranking_variants=("split_importance", "split_count"),
        n_jobs=-1,
        downstream_n_jobs=1,
    )

    assert "--output-uri" not in script
    assert "paper.benchmark.experiments.cif_mechanism_ablation --distributed" in script
    assert f"-e CITREES_IMAGE_URI={DIGEST_URI}" in script
    assert f"-e CITREES_MECHANISM_SPEC_SHA256={specification_sha256}" in script


def test_mechanism_launch_scopes_profile_to_derived_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.run_instances.return_value = {"Instances": [{"InstanceId": "i-mechanism"}]}
    ensure_profile = MagicMock(return_value="citrees-campaign-test")
    monkeypatch.setattr(
        ec2_infra,
        "validate_image_revision",
        lambda image_uri, region: "a" * 40,
    )
    monkeypatch.setattr(ec2_infra, "get_aws_account_id", lambda: "123456789012")
    monkeypatch.setattr(ec2_infra, "ensure_security_group", lambda region: "sg-test")
    monkeypatch.setattr(ec2_infra, "ensure_campaign_iam_profile", ensure_profile)
    monkeypatch.setattr(ec2_infra, "get_ami", lambda region: "ami-test")
    monkeypatch.setattr(ec2_infra.boto3, "client", lambda *args, **kwargs: client)
    specification_sha256 = mechanism_specification_sha256(
        tasks=("classification",),
        source="real",
        datasets=("madelon",),
        seeds=(0,),
        folds=(0,),
        model_variants=("cif_default",),
        ranking_variants=("split_importance",),
        n_jobs=1,
        downstream_n_jobs=1,
    )

    assert launch_mechanism_workers(
        n=1,
        instance_type="c6a.8xlarge",
        image_uri=DIGEST_URI,
        num_shards=1,
        subnet_ids=("subnet-test",),
        tasks=("classification",),
        datasets=("madelon",),
        seeds=(0,),
        folds=(0,),
        model_variants=("cif_default",),
        ranking_variants=("split_importance",),
        n_jobs=1,
        downstream_n_jobs=1,
    ) == ["i-mechanism"]
    expected_prefix = (
        "experiments/cif_mechanism_ablation/image-sha256/"
        f"{'a' * 64}/spec-sha256/{specification_sha256}"
    )
    ensure_profile.assert_called_once_with(
        output_prefix=expected_prefix,
        campaign_sha256=specification_sha256,
        write_prefixes=(expected_prefix,),
        region="us-east-1",
    )
    assert client.run_instances.call_args.kwargs["IamInstanceProfile"] == {
        "Name": "citrees-campaign-test"
    }


def test_api_user_data_carries_complete_queue_and_provenance_scope() -> None:
    script = _make_api_user_data(
        region="us-east-1",
        ecr_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com",
        image_uri=DIGEST_URI,
        bucket="citrees-123456789012",
        git_sha="abc123",
        instance_type="m5.large",
        artifact_prefix="repairs/r-baselines/run-001",
        campaign_sha256=CAMPAIGN_SHA256,
        manifest_s3_key=MANIFEST_KEY,
        manifest_sha256=MANIFEST_SHA256,
        stage="rankings",
        lease_seconds=900,
        max_cell_attempts=3,
    )

    assert "-e CITREES_ARTIFACT_PREFIX=repairs/r-baselines/run-001" in script
    assert f"-e CITREES_CAMPAIGN_SHA256={CAMPAIGN_SHA256}" in script
    assert f"-e CITREES_MANIFEST_S3_KEY={MANIFEST_KEY}" in script
    assert f"-e CITREES_MANIFEST_SHA256={MANIFEST_SHA256}" in script
    assert "-e CITREES_STAGE=rankings" in script
    assert "-e CITREES_LEASE_SECONDS=900" in script
    assert "-e CITREES_MAX_CELL_ATTEMPTS=3" in script
    assert f"-e CITREES_IMAGE_URI={DIGEST_URI}" in script
    assert "-e EC2_INSTANCE_TYPE=m5.large" in script
    assert "-e AWS_ACCOUNT_ID=123456789012" in script
    assert f"docker pull {DIGEST_URI}" in script
    assert "trap shutdown_instance EXIT" in script
    assert script.index("trap shutdown_instance EXIT") < script.index("# Instance metadata")
    assert "shutdown -h now || systemctl poweroff --force --force" in script
    assert "docker run -d --restart no" in script
    assert "--name citrees-api" in script
    assert "docker wait citrees-api" in script


def test_worker_user_data_matches_api_scope() -> None:
    script = _make_worker_user_data(
        region="us-east-1",
        ecr_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com",
        image_uri=DIGEST_URI,
        api_url="http://10.0.0.10:8000",
        bucket="citrees-123456789012",
        git_sha="abc123",
        instance_type="c6a.8xlarge",
        artifact_prefix="repairs/r-baselines/run-001",
        campaign_sha256=CAMPAIGN_SHA256,
        manifest_sha256=MANIFEST_SHA256,
        stage="rankings",
    )

    assert "-e CITREES_ARTIFACT_PREFIX=repairs/r-baselines/run-001" in script
    assert f"-e CITREES_CAMPAIGN_SHA256={CAMPAIGN_SHA256}" in script
    assert f"-e CITREES_MANIFEST_SHA256={MANIFEST_SHA256}" in script
    assert "-e CITREES_STAGE=rankings" in script
    assert f"-e CITREES_IMAGE_URI={DIGEST_URI}" in script
    assert "-e EC2_INSTANCE_TYPE=c6a.8xlarge" in script
    assert "-e AWS_ACCOUNT_ID=123456789012" in script
    assert "docker run -d --restart no" in script
    assert "--restart on-failure" not in script
    assert "--api-url http://10.0.0.10:8000" in script
    assert "trap shutdown_instance EXIT" in script
    assert script.index("trap shutdown_instance EXIT") < script.index("# Instance metadata")
    assert "shutdown -h now || systemctl poweroff --force --force" in script


def _mock_api_launch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    public_ip: str | None,
) -> tuple[MagicMock, MagicMock]:
    ec2_client = MagicMock()
    ec2_client.run_instances.return_value = {"Instances": [{"InstanceId": "i-api"}]}
    instance = MagicMock()
    instance.public_ip_address = public_ip
    ec2_resource = MagicMock()
    ec2_resource.Instance.return_value = instance

    monkeypatch.setattr(ec2_infra, "get_api_scope", lambda region: None)
    monkeypatch.setattr(ec2_infra, "validate_image_revision", lambda image_uri, region: "a" * 40)
    monkeypatch.setattr(
        ec2_infra,
        "publish_rerun_manifest",
        lambda path, region: {
            "key": MANIFEST_KEY,
            "sha256": MANIFEST_SHA256,
            "campaign_sha256": CAMPAIGN_SHA256,
            "cells": 1,
        },
    )
    monkeypatch.setattr(ec2_infra, "get_aws_account_id", lambda: "123456789012")
    monkeypatch.setattr(ec2_infra, "ensure_security_group", lambda region: "sg-test")
    monkeypatch.setattr(
        ec2_infra,
        "ensure_campaign_iam_profile",
        lambda **kwargs: "citrees-campaign-test",
    )
    monkeypatch.setattr(ec2_infra, "get_ami", lambda region: "ami-test")
    monkeypatch.setattr(ec2_infra.boto3, "client", lambda *args, **kwargs: ec2_client)
    monkeypatch.setattr(ec2_infra.boto3, "resource", lambda *args, **kwargs: ec2_resource)
    monkeypatch.setattr(ec2_infra.time, "sleep", lambda seconds: None)
    return ec2_client, instance


def _launch_test_api(tmp_path: Path) -> dict[str, str]:
    return launch_api(
        instance_type="m5.large",
        image_uri=DIGEST_URI,
        artifact_prefix="repairs/run-001",
        manifest_path=tmp_path / "manifest.csv",
        stage="rankings",
        lease_seconds=900,
        max_cell_attempts=3,
    )


def test_api_launch_terminates_instance_when_public_ip_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ec2_client, _instance = _mock_api_launch(monkeypatch, public_ip=None)

    with pytest.raises(RuntimeError, match="public IP"):
        _launch_test_api(tmp_path)

    ec2_client.terminate_instances.assert_called_once_with(InstanceIds=["i-api"])


def test_api_launch_terminates_instance_when_readiness_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ec2_client, _instance = _mock_api_launch(monkeypatch, public_ip="203.0.113.10")
    readiness_error = RuntimeError("readiness failed")
    monkeypatch.setattr(
        ec2_infra,
        "_wait_for_api_ready",
        MagicMock(side_effect=readiness_error),
    )

    with pytest.raises(RuntimeError, match="readiness failed"):
        _launch_test_api(tmp_path)

    ec2_client.terminate_instances.assert_called_once_with(InstanceIds=["i-api"])
    assert (
        ec2_client.run_instances.call_args.kwargs["InstanceInitiatedShutdownBehavior"]
        == "terminate"
    )
    assert ec2_client.run_instances.call_args.kwargs["IamInstanceProfile"] == {
        "Name": "citrees-campaign-test"
    }
    client_token = ec2_client.run_instances.call_args.kwargs["ClientToken"]
    assert len(client_token) == 64
    assert client_token.startswith("citrees-api-")


def test_running_api_scope_requires_complete_immutable_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.describe_instances.return_value = {
        "Reservations": [
            {
                "Instances": [
                    {
                        "InstanceId": "i-api",
                        "PrivateIpAddress": "10.0.0.10",
                        "PublicIpAddress": "203.0.113.10",
                        "Tags": [
                            {"Key": "citrees-artifact-prefix", "Value": "repairs/run-001"},
                            {
                                "Key": "citrees-campaign-sha256",
                                "Value": CAMPAIGN_SHA256,
                            },
                            {"Key": "citrees-image-uri", "Value": DIGEST_URI},
                            {"Key": "citrees-manifest-key", "Value": MANIFEST_KEY},
                            {
                                "Key": "citrees-manifest-sha256",
                                "Value": MANIFEST_SHA256,
                            },
                            {"Key": "citrees-max-cell-attempts", "Value": "3"},
                            {"Key": "citrees-stage", "Value": "rankings"},
                        ],
                    }
                ]
            }
        ]
    }
    monkeypatch.setattr(ec2_infra.boto3, "client", lambda *args, **kwargs: client)

    scope = get_api_scope()

    assert scope == ApiScope(
        api_url="http://10.0.0.10:8000",
        public_api_url="http://203.0.113.10:8000",
        artifact_prefix="repairs/run-001",
        campaign_sha256=CAMPAIGN_SHA256,
        image_uri=DIGEST_URI,
        manifest_s3_key=MANIFEST_KEY,
        manifest_sha256=MANIFEST_SHA256,
        max_cell_attempts=3,
        stage="rankings",
    )

    client.describe_instances.return_value["Reservations"][0]["Instances"][0]["Tags"].pop()
    with pytest.raises(RuntimeError, match="missing scope tags"):
        get_api_scope()


def test_worker_launch_rejects_scope_mismatch_before_ec2_launch(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("fixture")
    monkeypatch.setattr(
        ec2_infra,
        "publish_rerun_manifest",
        lambda path, region: {
            "key": MANIFEST_KEY,
            "sha256": MANIFEST_SHA256,
            "campaign_sha256": CAMPAIGN_SHA256,
            "cells": 1,
        },
    )
    monkeypatch.setattr(
        ec2_infra,
        "get_api_scope",
        lambda region: ApiScope(
            api_url="http://10.0.0.10:8000",
            public_api_url="http://203.0.113.10:8000",
            artifact_prefix="repairs/other-run",
            campaign_sha256=CAMPAIGN_SHA256,
            image_uri=DIGEST_URI,
            manifest_s3_key=MANIFEST_KEY,
            manifest_sha256=MANIFEST_SHA256,
            max_cell_attempts=3,
            stage="rankings",
        ),
    )
    monkeypatch.setattr(ec2_infra, "_wait_for_api_ready", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="does not match running API"):
        launch_workers(
            n=1,
            instance_type="c6a.8xlarge",
            image_uri=DIGEST_URI,
            artifact_prefix="repairs/run-001",
            launch_id="scope-mismatch",
            manifest_path=manifest_path,
            stage="rankings",
        )


def test_worker_launch_refreshes_ingress_before_api_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("fixture", encoding="utf-8")
    events: list[str] = []
    client = MagicMock()
    s3 = _MemoryS3()
    client.run_instances.return_value = {"Instances": [{"InstanceId": "i-worker"}]}
    client.describe_instances.return_value = {"Reservations": []}
    monkeypatch.setattr(
        ec2_infra,
        "publish_rerun_manifest",
        lambda path, region: {
            "key": MANIFEST_KEY,
            "sha256": MANIFEST_SHA256,
            "campaign_sha256": CAMPAIGN_SHA256,
            "cells": 1,
        },
    )
    monkeypatch.setattr(
        ec2_infra,
        "get_api_scope",
        lambda region: ApiScope(
            api_url="http://10.0.0.10:8000",
            public_api_url="http://203.0.113.10:8000",
            artifact_prefix="repairs/run-001",
            campaign_sha256=CAMPAIGN_SHA256,
            image_uri=DIGEST_URI,
            manifest_s3_key=MANIFEST_KEY,
            manifest_sha256=MANIFEST_SHA256,
            max_cell_attempts=3,
            stage="rankings",
        ),
    )
    monkeypatch.setattr(
        ec2_infra,
        "ensure_security_group",
        lambda region: events.append("security") or "sg-test",
    )
    monkeypatch.setattr(
        ec2_infra,
        "ensure_campaign_iam_profile",
        lambda **kwargs: "citrees-campaign-test",
    )
    monkeypatch.setattr(
        ec2_infra,
        "_wait_for_api_ready",
        lambda *args, **kwargs: events.append("readiness"),
    )
    monkeypatch.setattr(
        ec2_infra,
        "validate_image_revision",
        lambda image_uri, region: "a" * 40,
    )
    monkeypatch.setattr(ec2_infra, "get_aws_account_id", lambda: "123456789012")
    monkeypatch.setattr(ec2_infra, "get_ami", lambda region: "ami-test")
    monkeypatch.setattr(
        ec2_infra.boto3,
        "client",
        lambda service, **kwargs: {"ec2": client, "s3": s3}[service],
    )

    assert launch_workers(
        n=1,
        instance_type="c6a.8xlarge",
        image_uri=DIGEST_URI,
        artifact_prefix="repairs/run-001",
        launch_id="ingress-refresh",
        manifest_path=manifest_path,
        stage="rankings",
    ) == ["i-worker"]
    assert events == ["security", "readiness"]
    assert client.run_instances.call_args.kwargs["IamInstanceProfile"] == {
        "Name": "citrees-campaign-test"
    }


def test_worker_launch_is_durable_idempotent_and_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("fixture", encoding="utf-8")
    events: list[str] = []
    s3 = _MemoryS3(events)
    ec2 = MagicMock()
    ec2.describe_instances.return_value = {"Reservations": []}

    def run_instances(**kwargs: Any) -> dict[str, object]:
        slot = len(ec2.run_instances.call_args_list)
        events.append(f"ec2:{slot}")
        return {"Instances": [{"InstanceId": f"i-worker-{slot}"}]}

    ec2.run_instances.side_effect = run_instances
    monkeypatch.setattr(
        ec2_infra,
        "publish_rerun_manifest",
        lambda path, region: {
            "bucket": "citrees-123456789012",
            "key": MANIFEST_KEY,
            "sha256": MANIFEST_SHA256,
            "campaign_sha256": CAMPAIGN_SHA256,
            "cells": 1,
        },
    )
    monkeypatch.setattr(
        ec2_infra,
        "get_api_scope",
        lambda region: ApiScope(
            api_url="http://10.0.0.10:8000",
            public_api_url="http://203.0.113.10:8000",
            artifact_prefix="repairs/run-001",
            campaign_sha256=CAMPAIGN_SHA256,
            image_uri=DIGEST_URI,
            manifest_s3_key=MANIFEST_KEY,
            manifest_sha256=MANIFEST_SHA256,
            max_cell_attempts=3,
            stage="rankings",
        ),
    )
    monkeypatch.setattr(ec2_infra, "ensure_security_group", lambda region: "sg-test")
    monkeypatch.setattr(
        ec2_infra,
        "ensure_campaign_iam_profile",
        lambda **kwargs: "citrees-campaign-test",
    )
    monkeypatch.setattr(ec2_infra, "_wait_for_api_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(ec2_infra, "validate_image_revision", lambda image_uri, region: "a" * 40)
    monkeypatch.setattr(ec2_infra, "get_aws_account_id", lambda: "123456789012")
    monkeypatch.setattr(ec2_infra, "get_ami", lambda region: "ami-test")
    monkeypatch.setattr(
        ec2_infra.boto3,
        "client",
        lambda service, **kwargs: {"ec2": ec2, "s3": s3}[service],
    )

    launch_kwargs: _WorkerLaunchKwargs = {
        "n": 2,
        "instance_type": "c6a.8xlarge",
        "image_uri": DIGEST_URI,
        "artifact_prefix": "repairs/run-001",
        "manifest_path": manifest_path,
        "stage": "rankings",
        "spot": True,
        "launch_id": "scale-001",
    }
    assert launch_workers(**launch_kwargs) == ["i-worker-1", "i-worker-2"]

    assert events[0].endswith("/worker-launches/scale-001/intent.json")
    assert events.index("ec2:1") > 0
    assert len(ec2.run_instances.call_args_list) == 2
    client_tokens: set[str] = set()
    for call in ec2.run_instances.call_args_list:
        assert call.kwargs["MinCount"] == 1
        assert call.kwargs["MaxCount"] == 1
        assert call.kwargs["ClientToken"].startswith("citrees-worker-")
        client_tokens.add(call.kwargs["ClientToken"])
        tags = {tag["Key"]: tag["Value"] for tag in call.kwargs["TagSpecifications"][0]["Tags"]}
        assert tags["citrees-instance-family"] == "c6a"
        assert tags["citrees-market"] == "spot"
        assert tags["citrees-worker-launch-id"] == "scale-001"
    assert len(client_tokens) == 2

    intent_key = "repairs/run-001/_control/worker-launches/scale-001/intent.json"
    intent = json.loads(s3.objects[intent_key][0])
    assert intent["instance_family"] == "c6a"
    assert intent["market"] == "spot"
    assert intent["requested_instances"] == 2
    first_request = ec2.run_instances.call_args_list[0].kwargs
    request_contract = intent["request_contract"]
    for key, value in first_request.items():
        if key not in {"ClientToken", "TagSpecifications", "UserData"}:
            assert request_contract[key] == value
    assert (
        request_contract["UserDataSha256"]
        == hashlib.sha256(first_request["UserData"].encode()).hexdigest()
    )
    outcome_keys = sorted(key for key in s3.objects if "/instances/" in key)
    assert len(outcome_keys) == 2
    assert {json.loads(s3.objects[key][0])["instance_id"] for key in outcome_keys} == {
        "i-worker-1",
        "i-worker-2",
    }
    assert {json.loads(s3.objects[key][0])["market"] for key in outcome_keys} == {"spot"}
    assert {json.loads(s3.objects[key][0])["instance_family"] for key in outcome_keys} == {"c6a"}

    ec2.run_instances.reset_mock()
    events.clear()
    assert launch_workers(**launch_kwargs) == ["i-worker-1", "i-worker-2"]
    ec2.run_instances.assert_not_called()


def _mock_worker_launch_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, MagicMock, _MemoryS3]:
    """Install deterministic benchmark worker launch dependencies."""
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("fixture", encoding="utf-8")
    ec2 = MagicMock()
    ec2.describe_instances.return_value = {"Reservations": []}
    s3 = _MemoryS3()
    monkeypatch.setattr(
        ec2_infra,
        "publish_rerun_manifest",
        lambda path, region: {
            "bucket": "citrees-123456789012",
            "key": MANIFEST_KEY,
            "sha256": MANIFEST_SHA256,
            "campaign_sha256": CAMPAIGN_SHA256,
            "cells": 1,
        },
    )
    monkeypatch.setattr(
        ec2_infra,
        "get_api_scope",
        lambda region: ApiScope(
            api_url="http://10.0.0.10:8000",
            public_api_url="http://203.0.113.10:8000",
            artifact_prefix="repairs/run-001",
            campaign_sha256=CAMPAIGN_SHA256,
            image_uri=DIGEST_URI,
            manifest_s3_key=MANIFEST_KEY,
            manifest_sha256=MANIFEST_SHA256,
            max_cell_attempts=3,
            stage="rankings",
        ),
    )
    monkeypatch.setattr(ec2_infra, "ensure_security_group", lambda region: "sg-test")
    monkeypatch.setattr(
        ec2_infra,
        "ensure_campaign_iam_profile",
        lambda **kwargs: "citrees-campaign-test",
    )
    monkeypatch.setattr(ec2_infra, "_wait_for_api_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(ec2_infra, "validate_image_revision", lambda image_uri, region: "a" * 40)
    monkeypatch.setattr(ec2_infra, "get_aws_account_id", lambda: "123456789012")
    monkeypatch.setattr(ec2_infra, "get_ami", lambda region: "ami-test")
    monkeypatch.setattr(ec2_infra.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        ec2_infra.boto3,
        "client",
        lambda service, **kwargs: {"ec2": ec2, "s3": s3}[service],
    )
    return manifest_path, ec2, s3


def _worker_launch_kwargs(
    manifest_path: Path,
    *,
    n: int = 1,
    spot: bool = False,
) -> _WorkerLaunchKwargs:
    """Build one complete direct worker launch invocation."""
    return {
        "n": n,
        "instance_type": "c6a.8xlarge",
        "image_uri": DIGEST_URI,
        "artifact_prefix": "repairs/run-001",
        "launch_id": "scale-001",
        "manifest_path": manifest_path,
        "stage": "rankings",
        "spot": spot,
    }


def test_worker_launch_recovers_ambiguous_timeout_with_same_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, ec2, _s3 = _mock_worker_launch_dependencies(tmp_path, monkeypatch)
    ec2.run_instances.side_effect = [
        ReadTimeoutError(endpoint_url="https://ec2.us-east-1.amazonaws.com"),
        {"Instances": [{"InstanceId": "i-recovered"}]},
    ]

    assert launch_workers(**_worker_launch_kwargs(manifest_path)) == ["i-recovered"]

    assert len(ec2.run_instances.call_args_list) == 2
    first = ec2.run_instances.call_args_list[0].kwargs
    second = ec2.run_instances.call_args_list[1].kwargs
    assert first["ClientToken"] == second["ClientToken"]
    assert first["TagSpecifications"] == second["TagSpecifications"]


@pytest.mark.parametrize(
    "error_code",
    [
        "InternalError",
        "InternalFailure",
        "RequestLimitExceeded",
        "ServiceUnavailable",
        "Unavailable",
    ],
)
def test_worker_launch_retries_ambiguous_ec2_server_error_with_same_token(
    error_code: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, ec2, _s3 = _mock_worker_launch_dependencies(tmp_path, monkeypatch)
    ec2.run_instances.side_effect = [
        ClientError(
            {"Error": {"Code": error_code, "Message": "ambiguous"}},
            "RunInstances",
        ),
        {"Instances": [{"InstanceId": "i-recovered"}]},
    ]

    assert launch_workers(**_worker_launch_kwargs(manifest_path)) == ["i-recovered"]
    assert ec2.run_instances.call_count == 2
    assert {call.kwargs["ClientToken"] for call in ec2.run_instances.call_args_list} == {
        ec2.run_instances.call_args_list[0].kwargs["ClientToken"]
    }


def test_worker_launch_exhausts_ambiguous_retries_without_changing_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, ec2, s3 = _mock_worker_launch_dependencies(tmp_path, monkeypatch)
    ec2.run_instances.side_effect = [
        ReadTimeoutError(endpoint_url="https://ec2.us-east-1.amazonaws.com") for _ in range(3)
    ]

    with pytest.raises(ReadTimeoutError):
        launch_workers(**_worker_launch_kwargs(manifest_path))

    assert ec2.run_instances.call_count == 3
    assert len({call.kwargs["ClientToken"] for call in ec2.run_instances.call_args_list}) == 1
    assert not any("/instances/" in key for key in s3.objects)


def test_worker_launch_recovers_exact_tags_after_ambiguous_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, ec2, _s3 = _mock_worker_launch_dependencies(tmp_path, monkeypatch)
    describe_calls = 0

    def describe_instances(**kwargs: Any) -> dict[str, object]:
        nonlocal describe_calls
        describe_calls += 1
        tags = [
            {
                "Key": item["Name"].removeprefix("tag:"),
                "Value": item["Values"][0],
            }
            for item in kwargs["Filters"]
        ]
        return {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-tag-recovered",
                            "Tags": tags,
                        }
                    ]
                }
            ]
        }

    ec2.describe_instances.side_effect = describe_instances
    ec2.run_instances.side_effect = ReadTimeoutError(
        endpoint_url="https://ec2.us-east-1.amazonaws.com"
    )

    assert launch_workers(**_worker_launch_kwargs(manifest_path)) == ["i-tag-recovered"]
    assert ec2.run_instances.call_count == 1


def test_worker_launch_reconciles_partial_capacity_on_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, ec2, _s3 = _mock_worker_launch_dependencies(tmp_path, monkeypatch)
    capacity_error = ClientError(
        {"Error": {"Code": "InsufficientInstanceCapacity", "Message": "full"}},
        "RunInstances",
    )
    ec2.run_instances.side_effect = [
        {"Instances": [{"InstanceId": "i-worker-1"}]},
        capacity_error,
    ]
    kwargs = _worker_launch_kwargs(manifest_path, n=3)

    assert launch_workers(**kwargs) == ["i-worker-1"]
    failed_slot_token = ec2.run_instances.call_args_list[1].kwargs["ClientToken"]

    ec2.run_instances.reset_mock(side_effect=True)
    ec2.run_instances.side_effect = [
        {"Instances": [{"InstanceId": "i-worker-2"}]},
        {"Instances": [{"InstanceId": "i-worker-3"}]},
    ]
    assert launch_workers(**kwargs) == [
        "i-worker-1",
        "i-worker-2",
        "i-worker-3",
    ]
    assert ec2.run_instances.call_count == 2
    assert ec2.run_instances.call_args_list[0].kwargs["ClientToken"] == failed_slot_token


def test_worker_launch_accounts_terminated_instance_with_exact_tags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, ec2, s3 = _mock_worker_launch_dependencies(tmp_path, monkeypatch)
    ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-worker"}]}
    kwargs = _worker_launch_kwargs(manifest_path)
    assert launch_workers(**kwargs) == ["i-worker"]
    tags = ec2.run_instances.call_args.kwargs["TagSpecifications"][0]["Tags"]
    record_key = next(key for key in s3.objects if "/instances/" in key)
    del s3.objects[record_key]
    ec2.run_instances.reset_mock()
    ec2.describe_instances.return_value = {
        "Reservations": [
            {
                "Instances": [
                    {
                        "InstanceId": "i-worker",
                        "State": {"Name": "terminated"},
                        "Tags": tags,
                    }
                ]
            }
        ]
    }

    assert launch_workers(**kwargs) == ["i-worker"]
    ec2.run_instances.assert_not_called()
    assert record_key in s3.objects


def test_worker_launch_rejects_reused_identity_with_changed_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, ec2, _s3 = _mock_worker_launch_dependencies(tmp_path, monkeypatch)
    ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-worker"}]}
    kwargs = _worker_launch_kwargs(manifest_path)
    assert launch_workers(**kwargs) == ["i-worker"]
    ec2.run_instances.reset_mock()

    with pytest.raises(RuntimeError, match="different exact launch contract"):
        launch_workers(**_worker_launch_kwargs(manifest_path, spot=True))

    ec2.run_instances.assert_not_called()


def test_worker_launch_replay_requires_same_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, ec2, _s3 = _mock_worker_launch_dependencies(tmp_path, monkeypatch)
    ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-worker"}]}
    assert launch_workers(**_worker_launch_kwargs(manifest_path)) == ["i-worker"]
    ec2.run_instances.reset_mock()

    with pytest.raises(RuntimeError, match="recovery requires the same --count"):
        launch_workers(**_worker_launch_kwargs(manifest_path, n=2))

    ec2.run_instances.assert_not_called()


def test_worker_launch_preserves_prior_records_across_operational_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, ec2, s3 = _mock_worker_launch_dependencies(tmp_path, monkeypatch)
    operational_error = ClientError(
        {"Error": {"Code": "UnauthorizedOperation", "Message": "denied"}},
        "RunInstances",
    )
    ec2.run_instances.side_effect = [
        {"Instances": [{"InstanceId": "i-worker-1"}]},
        operational_error,
    ]
    kwargs = _worker_launch_kwargs(manifest_path, n=2)

    with pytest.raises(ClientError, match="UnauthorizedOperation"):
        launch_workers(**kwargs)

    records = [
        json.loads(payload)
        for key, (payload, _metadata) in s3.objects.items()
        if "/instances/" in key
    ]
    assert [record["instance_id"] for record in records] == ["i-worker-1"]
    failed_slot_token = ec2.run_instances.call_args_list[1].kwargs["ClientToken"]

    ec2.run_instances.reset_mock(side_effect=True)
    ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-worker-2"}]}
    assert launch_workers(**kwargs) == ["i-worker-1", "i-worker-2"]
    assert ec2.run_instances.call_count == 1
    assert ec2.run_instances.call_args.kwargs["ClientToken"] == failed_slot_token


def test_worker_launch_rejects_nonexact_ec2_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, ec2, s3 = _mock_worker_launch_dependencies(tmp_path, monkeypatch)
    ec2.run_instances.return_value = {
        "Instances": [
            {"InstanceId": "i-worker-1"},
            {"InstanceId": "i-worker-2"},
        ]
    }

    with pytest.raises(RuntimeError, match="2 instances"):
        launch_workers(**_worker_launch_kwargs(manifest_path))

    assert ec2.run_instances.call_count == 1
    assert not any("/instances/" in key for key in s3.objects)


def test_worker_launch_propagates_operational_ec2_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, ec2, s3 = _mock_worker_launch_dependencies(tmp_path, monkeypatch)
    ec2.run_instances.side_effect = ClientError(
        {"Error": {"Code": "UnauthorizedOperation", "Message": "denied"}},
        "RunInstances",
    )

    with pytest.raises(ClientError, match="UnauthorizedOperation"):
        launch_workers(**_worker_launch_kwargs(manifest_path))

    assert ec2.run_instances.call_count == 1
    assert not any("/instances/" in key for key in s3.objects)


def test_worker_launch_requires_durable_intent_before_ec2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, ec2, s3 = _mock_worker_launch_dependencies(tmp_path, monkeypatch)
    write_error = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "denied"}},
        "PutObject",
    )
    monkeypatch.setattr(s3, "put_object", MagicMock(side_effect=write_error))

    with pytest.raises(ClientError, match="AccessDenied"):
        launch_workers(**_worker_launch_kwargs(manifest_path))

    ec2.run_instances.assert_not_called()


@pytest.mark.parametrize(
    "launch_id",
    ["", "UPPER", "contains spaces", "../escape", "trailing-", "a" * 65],
)
def test_worker_launch_rejects_invalid_identity_before_aws(
    launch_id: str,
) -> None:
    with pytest.raises(ValueError, match="launch_id"):
        launch_workers(
            n=1,
            instance_type="c6a.8xlarge",
            image_uri=DIGEST_URI,
            artifact_prefix="repairs/run-001",
            launch_id=launch_id,
            manifest_path=Path("unused.csv"),
            stage="rankings",
        )


def test_s3_bucket_is_private_and_versioned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    monkeypatch.setattr(aws_infra, "get_aws_account_id", lambda: "123456789012")
    monkeypatch.setattr(aws_infra.boto3, "client", lambda *args, **kwargs: client)

    bucket = aws_infra.ensure_s3_bucket()

    assert bucket == "citrees-123456789012"
    client.put_public_access_block.assert_called_once_with(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    client.put_bucket_versioning.assert_called_once_with(
        Bucket=bucket,
        VersioningConfiguration={"Status": "Enabled"},
    )
    client.put_bucket_encryption.assert_called_once()
    policy = json.loads(client.put_bucket_policy.call_args.kwargs["Policy"])
    statements = {statement["Sid"]: statement for statement in policy["Statement"]}
    assert statements["DenyInsecureTransport"]["Condition"] == {
        "Bool": {"aws:SecureTransport": "false"}
    }
    assert statements["DenyUnconditionalJSSWrites"] == {
        "Sid": "DenyUnconditionalJSSWrites",
        "Effect": "Deny",
        "Principal": "*",
        "Action": "s3:PutObject",
        "Resource": "arn:aws:s3:::citrees-123456789012/jss/replication/*",
        "Condition": {"Null": {"s3:if-none-match": "true"}},
    }


def test_campaign_role_can_write_only_its_exact_output_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_prefix = "repairs/run-001"
    profile_name = aws_infra.campaign_instance_profile_name(
        output_prefix=output_prefix,
        campaign_sha256=CAMPAIGN_SHA256,
        write_prefixes=(output_prefix,),
    )
    client = _campaign_iam_client(profile_name)
    _patch_campaign_iam(monkeypatch, client)

    assert (
        aws_infra.ensure_campaign_iam_profile(
            output_prefix=output_prefix,
            campaign_sha256=CAMPAIGN_SHA256,
            write_prefixes=(output_prefix,),
        )
        == profile_name
    )
    assert profile_name != aws_infra.campaign_instance_profile_name(
        output_prefix="repairs/run-002",
        campaign_sha256=CAMPAIGN_SHA256,
        write_prefixes=("repairs/run-002",),
    )

    policy = client.put_role_policy.call_args.kwargs["PolicyDocument"]
    assert "s3:DeleteObject" not in policy
    assert "iam:PassRole" not in policy

    statements = {statement["Sid"]: statement for statement in json.loads(policy)["Statement"]}
    assert statements["S3WriteArtifacts"]["Resource"] == [
        "arn:aws:s3:::citrees-123456789012/repairs/run-001/*"
    ]
    assert statements["S3WriteArtifacts"]["Condition"] == {
        "StringEquals": {"s3:if-none-match": "*"}
    }
    assert statements["S3ListApprovedPrefixes"]["Condition"]["StringLike"]["s3:prefix"] == [
        "data/*",
        "rerun-manifests/*",
        "repairs/run-001/*",
    ]
    assert statements["S3ReadInputsAndArtifacts"]["Resource"] == [
        "arn:aws:s3:::citrees-123456789012/data/*",
        "arn:aws:s3:::citrees-123456789012/rerun-manifests/*",
        "arn:aws:s3:::citrees-123456789012/repairs/run-001/*",
    ]
    assert statements["ECRPull"]["Resource"] == (
        "arn:aws:ecr:us-east-1:123456789012:repository/citrees-123456789012"
    )


def test_campaign_role_can_limit_writes_to_shard_subprefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_prefix = "jss/replication/source-a/campaign-b"
    write_prefix = f"{output_prefix}/shards"
    profile_name = aws_infra.campaign_instance_profile_name(
        output_prefix=output_prefix,
        campaign_sha256=CAMPAIGN_SHA256,
        write_prefixes=(write_prefix,),
    )
    client = _campaign_iam_client(profile_name)
    _patch_campaign_iam(monkeypatch, client)

    assert (
        aws_infra.ensure_campaign_iam_profile(
            output_prefix=output_prefix,
            campaign_sha256=CAMPAIGN_SHA256,
            write_prefixes=(write_prefix,),
        )
        == profile_name
    )
    statements = {
        statement["Sid"]: statement
        for statement in json.loads(client.put_role_policy.call_args.kwargs["PolicyDocument"])[
            "Statement"
        ]
    }
    assert statements["S3WriteArtifacts"]["Resource"] == [
        f"arn:aws:s3:::citrees-123456789012/{write_prefix}/*"
    ]
    assert profile_name != aws_infra.campaign_instance_profile_name(
        output_prefix=output_prefix,
        campaign_sha256=CAMPAIGN_SHA256,
        write_prefixes=(output_prefix,),
    )


def test_campaign_role_converges_dirty_existing_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_prefix = "jss/replication/source-a/campaign-b"
    write_prefix = f"{output_prefix}/shards"
    profile_name = aws_infra.campaign_instance_profile_name(
        output_prefix=output_prefix,
        campaign_sha256=CAMPAIGN_SHA256,
        write_prefixes=(write_prefix,),
    )
    runtime_policy_name = f"{profile_name}-runtime"
    unexpected_managed_policy = "arn:aws:iam::123456789012:policy/unexpected"
    client = _campaign_iam_client(
        profile_name,
        initial_inline_policy_names=(runtime_policy_name, "unexpected-inline"),
        initial_attached_policy_arns=(SSM_POLICY_ARN, unexpected_managed_policy),
        initial_profile_roles=("unexpected-role",),
    )
    _patch_campaign_iam(monkeypatch, client)

    assert (
        aws_infra.ensure_campaign_iam_profile(
            output_prefix=output_prefix,
            campaign_sha256=CAMPAIGN_SHA256,
            write_prefixes=(write_prefix,),
        )
        == profile_name
    )

    client.delete_role_policy.assert_called_once_with(
        RoleName=profile_name,
        PolicyName="unexpected-inline",
    )
    client.detach_role_policy.assert_called_once_with(
        RoleName=profile_name,
        PolicyArn=unexpected_managed_policy,
    )
    client.remove_role_from_instance_profile.assert_called_once_with(
        InstanceProfileName=profile_name,
        RoleName="unexpected-role",
    )
    client.add_role_to_instance_profile.assert_called_once_with(
        InstanceProfileName=profile_name,
        RoleName=profile_name,
    )


def test_campaign_role_rejects_permissions_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_prefix = "jss/replication/source-a/campaign-b"
    write_prefix = f"{output_prefix}/shards"
    profile_name = aws_infra.campaign_instance_profile_name(
        output_prefix=output_prefix,
        campaign_sha256=CAMPAIGN_SHA256,
        write_prefixes=(write_prefix,),
    )
    client = MagicMock()
    client.exceptions.NoSuchEntityException = _NoSuchEntityError
    client.get_role.return_value = {
        "Role": {
            "RoleName": profile_name,
            "AssumeRolePolicyDocument": CAMPAIGN_TRUST_POLICY,
            "PermissionsBoundary": {
                "PermissionsBoundaryType": "Policy",
                "PermissionsBoundaryArn": "arn:aws:iam::123456789012:policy/boundary",
            },
        }
    }
    _patch_campaign_iam(monkeypatch, client)

    with pytest.raises(RuntimeError, match="permissions boundary"):
        aws_infra.ensure_campaign_iam_profile(
            output_prefix=output_prefix,
            campaign_sha256=CAMPAIGN_SHA256,
            write_prefixes=(write_prefix,),
        )

    client.update_assume_role_policy.assert_not_called()
    client.put_role_policy.assert_not_called()
    client.attach_role_policy.assert_not_called()


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("trust", "trust policy"),
        ("inline_inventory", "inline policy inventory"),
        ("runtime_policy", "runtime policy"),
        ("managed_inventory", "managed policy inventory"),
        ("profile_role", "instance profile"),
    ],
)
def test_campaign_role_rejects_postmutation_readback_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
    message: str,
) -> None:
    output_prefix = "jss/replication/source-a/campaign-b"
    write_prefix = f"{output_prefix}/shards"
    profile_name = aws_infra.campaign_instance_profile_name(
        output_prefix=output_prefix,
        campaign_sha256=CAMPAIGN_SHA256,
        write_prefixes=(write_prefix,),
    )
    policy_name = f"{profile_name}-runtime"
    options: dict[str, object] = {}
    if mismatch == "trust":
        options["readback_trust_policy"] = {"Version": "2012-10-17", "Statement": []}
    elif mismatch == "inline_inventory":
        options["readback_inline_policy_names"] = (policy_name, "unexpected-inline")
    elif mismatch == "runtime_policy":
        options["readback_runtime_policy"] = {
            "Version": "2012-10-17",
            "Statement": [],
        }
    elif mismatch == "managed_inventory":
        options["readback_attached_policy_arns"] = (
            SSM_POLICY_ARN,
            "arn:aws:iam::123456789012:policy/unexpected",
        )
    else:
        options["readback_profile_roles"] = ()
    client = _campaign_iam_client(profile_name, **options)
    _patch_campaign_iam(monkeypatch, client)

    with pytest.raises(RuntimeError, match=message):
        aws_infra.ensure_campaign_iam_profile(
            output_prefix=output_prefix,
            campaign_sha256=CAMPAIGN_SHA256,
            write_prefixes=(write_prefix,),
        )


def test_security_group_refresh_preserves_self_ingress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.describe_security_groups.return_value = {
        "SecurityGroups": [
            {
                "GroupId": "sg-test",
                "IpPermissions": [
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 8000,
                        "ToPort": 8000,
                        "IpRanges": [
                            {
                                "CidrIp": "198.51.100.10/32",
                                "Description": "API from caller",
                            }
                        ],
                        "UserIdGroupPairs": [
                            {
                                "GroupId": "sg-test",
                                "Description": "API from citrees instances",
                            }
                        ],
                    },
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 22,
                        "ToPort": 22,
                        "IpRanges": [
                            {
                                "CidrIp": "203.0.113.20/32",
                                "Description": "SSH from caller",
                            }
                        ],
                    },
                ],
            }
        ]
    }
    monkeypatch.setattr(aws_infra, "get_public_ip", lambda: "203.0.113.20")
    monkeypatch.setattr(aws_infra.boto3, "client", lambda *args, **kwargs: client)

    assert aws_infra.ensure_security_group() == "sg-test"

    revoked = client.revoke_security_group_ingress.call_args.kwargs["IpPermissions"]
    assert revoked == [
        {
            "IpProtocol": "tcp",
            "FromPort": 8000,
            "ToPort": 8000,
            "IpRanges": [
                {
                    "CidrIp": "198.51.100.10/32",
                    "Description": "API from caller",
                }
            ],
        }
    ]
    authorized = [
        permission
        for call in client.authorize_security_group_ingress.call_args_list
        for permission in call.kwargs["IpPermissions"]
    ]
    assert all(not permission.get("UserIdGroupPairs") for permission in authorized)


def test_security_group_refresh_repairs_missing_self_ingress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.describe_security_groups.return_value = {
        "SecurityGroups": [
            {
                "GroupId": "sg-test",
                "IpPermissions": [
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 8000,
                        "ToPort": 8000,
                        "IpRanges": [
                            {
                                "CidrIp": "203.0.113.20/32",
                                "Description": "API from caller",
                            }
                        ],
                    },
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 22,
                        "ToPort": 22,
                        "IpRanges": [
                            {
                                "CidrIp": "203.0.113.20/32",
                                "Description": "SSH from caller",
                            }
                        ],
                    },
                ],
            }
        ]
    }
    monkeypatch.setattr(aws_infra, "get_public_ip", lambda: "203.0.113.20")
    monkeypatch.setattr(aws_infra.boto3, "client", lambda *args, **kwargs: client)

    assert aws_infra.ensure_security_group() == "sg-test"

    client.revoke_security_group_ingress.assert_not_called()
    client.authorize_security_group_ingress.assert_called_once_with(
        GroupId="sg-test",
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 8000,
                "ToPort": 8000,
                "UserIdGroupPairs": [
                    {
                        "GroupId": "sg-test",
                        "Description": "API from citrees instances",
                    }
                ],
            }
        ],
    )


def test_ecr_repository_is_always_immutable_and_scanned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.describe_repositories.return_value = {
        "repositories": [
            {"repositoryUri": ("123456789012.dkr.ecr.us-east-1.amazonaws.com/citrees-123456789012")}
        ]
    }
    monkeypatch.setattr(aws_infra, "get_aws_account_id", lambda: "123456789012")
    monkeypatch.setattr(aws_infra.boto3, "client", lambda *args, **kwargs: client)

    repo_name, _repo_uri = aws_infra.ensure_ecr_repo()

    client.put_image_tag_mutability.assert_called_once_with(
        repositoryName=repo_name,
        imageTagMutability="IMMUTABLE",
    )
    client.put_image_scanning_configuration.assert_called_once_with(
        repositoryName=repo_name,
        imageScanningConfiguration={"scanOnPush": True},
    )


def test_frozen_revision_requires_a_clean_full_git_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean = [
        SimpleNamespace(stdout="a" * 40 + "\n"),
        SimpleNamespace(stdout=""),
    ]
    monkeypatch.setattr(aws_infra.subprocess, "run", MagicMock(side_effect=clean))

    assert aws_infra.get_frozen_git_sha(Path("/repo")) == "a" * 40

    dirty = [
        SimpleNamespace(stdout="a" * 40 + "\n"),
        SimpleNamespace(stdout=" M paper/benchmark/infra/aws.py\n"),
    ]
    monkeypatch.setattr(aws_infra.subprocess, "run", MagicMock(side_effect=dirty))
    with pytest.raises(RuntimeError, match="source tree must be clean"):
        aws_infra.get_frozen_git_sha(Path("/repo"))


def test_frozen_source_context_excludes_ignored_and_untracked_files(tmp_path: Path) -> None:
    """The Docker source context must contain only bytes committed to Git."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
    (repo / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    (repo / "ignored.txt").write_text("ignored\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    with aws_infra.frozen_source_context(repo, git_sha) as context:
        assert (context / "tracked.txt").read_text(encoding="utf-8") == "tracked\n"
        assert not (context / "ignored.txt").exists()
        assert not (context / "untracked.txt").exists()
        assert not (context / ".git").exists()


def test_remote_image_config_exposes_verified_revision_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40
    config_payload = json.dumps(
        {"config": {"Labels": {"org.opencontainers.image.revision": revision}}},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    config_digest = f"sha256:{hashlib.sha256(config_payload).hexdigest()}"
    manifest_payload = json.dumps(
        {
            "schemaVersion": 2,
            "config": {
                "digest": config_digest,
                "size": len(config_payload),
            },
            "layers": [],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest_digest = f"sha256:{hashlib.sha256(manifest_payload.encode()).hexdigest()}"
    client = MagicMock()
    client.batch_get_image.return_value = {
        "images": [
            {
                "imageId": {"imageDigest": manifest_digest},
                "imageManifest": manifest_payload,
            }
        ],
        "failures": [],
    }
    client.get_download_url_for_layer.return_value = {"downloadUrl": "https://ecr.example/config"}
    monkeypatch.setattr(
        aws_infra.urllib.request,
        "urlopen",
        lambda url, timeout: io.BytesIO(config_payload),
    )

    assert aws_infra._remote_image_revision(client, "citrees-test", manifest_digest) == revision
    client.get_download_url_for_layer.assert_called_once_with(
        repositoryName="citrees-test",
        layerDigest=config_digest,
    )


def test_image_digest_must_match_tag_and_remote_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/citrees-123456789012"
    client = MagicMock()
    client.describe_repositories.return_value = {
        "repositories": [{"repositoryUri": repository_uri}]
    }
    client.describe_images.return_value = {"imageDetails": [{"imageTags": ["a" * 40]}]}
    monkeypatch.setattr(aws_infra, "get_frozen_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(aws_infra, "get_aws_account_id", lambda: "123456789012")
    monkeypatch.setattr(aws_infra.boto3, "client", lambda *args, **kwargs: client)
    remote_revision = MagicMock(return_value="a" * 40)
    monkeypatch.setattr(aws_infra, "_remote_image_revision", remote_revision)

    assert aws_infra.validate_image_revision(repository_uri + "@sha256:" + "b" * 64) == "a" * 40
    remote_revision.assert_called_once_with(
        client,
        "citrees-123456789012",
        "sha256:" + "b" * 64,
    )

    client.describe_images.return_value = {"imageDetails": [{"imageTags": ["c" * 40]}]}
    with pytest.raises(RuntimeError, match="active source revision"):
        aws_infra.validate_image_revision(repository_uri + "@sha256:" + "b" * 64)

    client.describe_images.return_value = {"imageDetails": [{"imageTags": ["a" * 40]}]}
    remote_revision.return_value = "d" * 40
    with pytest.raises(RuntimeError, match="remote OCI revision label"):
        aws_infra.validate_image_revision(repository_uri + "@sha256:" + "b" * 64)


def test_manifest_publish_is_content_addressed_and_round_trip_verified(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"private manifest"
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_bytes(payload)
    manifest = SimpleNamespace(
        sha256=MANIFEST_SHA256,
        campaign_sha256=CAMPAIGN_SHA256,
        cells=(object(),),
        account_ids=("123456789012",),
    )
    client = MagicMock()
    client.get_object.return_value = {
        "Body": io.BytesIO(payload),
        "Metadata": {
            "campaign-sha256": CAMPAIGN_SHA256,
            "target-aws-account-id": "123456789012",
        },
    }
    monkeypatch.setattr(
        "paper.benchmark.pipeline.manifest.parse_rerun_manifest",
        lambda body, expected_sha256=None: manifest,
    )
    monkeypatch.setattr(aws_infra, "ensure_s3_bucket", lambda region: "citrees-test")
    monkeypatch.setattr(aws_infra, "get_aws_account_id", lambda: "123456789012")
    monkeypatch.setattr(aws_infra.boto3, "client", lambda *args, **kwargs: client)

    result = aws_infra.publish_rerun_manifest(manifest_path)

    assert result == {
        "bucket": "citrees-test",
        "key": MANIFEST_KEY,
        "sha256": MANIFEST_SHA256,
        "campaign_sha256": CAMPAIGN_SHA256,
        "cells": 1,
    }
    call = client.put_object.call_args
    assert call.kwargs["Bucket"] == "citrees-test"
    assert call.kwargs["Key"] == MANIFEST_KEY
    assert call.kwargs["Body"] == payload
    assert call.kwargs["IfNoneMatch"] == "*"
    assert call.kwargs["Metadata"]["target-aws-account-id"] == "123456789012"
    assert call.kwargs["Metadata"]["campaign-sha256"] == CAMPAIGN_SHA256
