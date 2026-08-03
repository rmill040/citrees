"""Tests for safe distributed benchmark launch configuration."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

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
    _validate_image_digest_uri,
    _validate_queue_scope,
    get_api_scope,
    launch_api,
    launch_mechanism_workers,
    launch_workers,
)

pytestmark = pytest.mark.paper

DIGEST_URI = (
    "123456789012.dkr.ecr.us-east-1.amazonaws.com/citrees"
    "@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)
MANIFEST_SHA256 = "b" * 64
CAMPAIGN_SHA256 = "e" * 64
MANIFEST_KEY = f"rerun-manifests/{MANIFEST_SHA256}.csv"


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
    assert _validate_image_digest_uri(f" {DIGEST_URI} ") == DIGEST_URI

    for invalid in (
        "",
        "repository:latest",
        "repository:abc123",
        "repository@sha256:abc",
        "repository@sha256:" + "g" * 64,
    ):
        with pytest.raises(ValueError, match="immutable"):
            _validate_image_digest_uri(invalid)


def test_candidate_image_pins_complete_statistical_runtime() -> None:
    dockerfile = Path("paper/benchmark/infra/docker/Dockerfile").read_text()

    assert "FROM rocker/r-ver:4.5.2@sha256:" in dockerfile
    assert "SOURCE_GIT_SHA" in dockerfile
    assert 'org.opencontainers.image.revision="$SOURCE_GIT_SHA"' in dockerfile
    assert "snapshot.ubuntu.com/ubuntu/20260801T000000Z" in dockerfile
    assert "build-essential=12.10ubuntu1" in dockerfile
    assert "/partykit_1.2-24.tar.gz" in dockerfile
    assert "packagemanager.posit.co/cran/2026-08-01/src/contrib/inum_1.0-5.tar.gz" in dockerfile
    assert 'partykit="1.2.24"' in dockerfile
    assert "uv python install 3.12.7" in dockerfile
    assert "UV_PYTHON=3.12.7" in dockerfile
    assert 'version("rpy2") == "3.6.7"' in dockerfile
    assert 'version("scikit-learn") == "1.8.0"' in dockerfile
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
    monkeypatch.setattr(ec2_infra, "_get_ami", lambda region: "ami-test")
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
    ensure_profile.assert_called_once_with(
        output_prefix=(
            "experiments/cif_mechanism_ablation/image-sha256/"
            f"{'a' * 64}/spec-sha256/{specification_sha256}"
        ),
        campaign_sha256=specification_sha256,
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
    monkeypatch.setattr(ec2_infra, "_get_ami", lambda region: "ami-test")
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
    client.run_instances.return_value = {"Instances": [{"InstanceId": "i-worker"}]}
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
    monkeypatch.setattr(ec2_infra, "_get_ami", lambda region: "ami-test")
    monkeypatch.setattr(ec2_infra.boto3, "client", lambda *args, **kwargs: client)

    assert launch_workers(
        n=1,
        instance_type="c6a.8xlarge",
        image_uri=DIGEST_URI,
        artifact_prefix="repairs/run-001",
        manifest_path=manifest_path,
        stage="rankings",
    ) == ["i-worker"]
    assert events == ["security", "readiness"]
    assert client.run_instances.call_args.kwargs["IamInstanceProfile"] == {
        "Name": "citrees-campaign-test"
    }


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
    assert policy["Statement"][0]["Condition"] == {"Bool": {"aws:SecureTransport": "false"}}


def test_campaign_role_can_write_only_its_exact_output_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_prefix = "repairs/run-001"
    profile_name = aws_infra.campaign_instance_profile_name(
        output_prefix=output_prefix,
        campaign_sha256=CAMPAIGN_SHA256,
    )
    client = MagicMock()
    client.get_instance_profile.return_value = {
        "InstanceProfile": {"Roles": [{"RoleName": profile_name}]}
    }
    monkeypatch.setattr(aws_infra, "get_aws_account_id", lambda: "123456789012")
    monkeypatch.setattr(aws_infra.boto3, "client", lambda *args, **kwargs: client)

    assert (
        aws_infra.ensure_campaign_iam_profile(
            output_prefix=output_prefix,
            campaign_sha256=CAMPAIGN_SHA256,
        )
        == profile_name
    )
    assert profile_name != aws_infra.campaign_instance_profile_name(
        output_prefix="repairs/run-002",
        campaign_sha256=CAMPAIGN_SHA256,
    )

    policy = client.put_role_policy.call_args.kwargs["PolicyDocument"]
    assert "s3:DeleteObject" not in policy
    assert "iam:PassRole" not in policy

    statements = {statement["Sid"]: statement for statement in json.loads(policy)["Statement"]}
    assert statements["S3WriteArtifacts"]["Resource"] == (
        "arn:aws:s3:::citrees-123456789012/repairs/run-001/*"
    )
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
