"""Tests for safe distributed benchmark launch configuration."""

from __future__ import annotations

import base64
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
from typer.testing import CliRunner

from paper.benchmark.cli.infra import (
    app as infra_app,
)
from paper.benchmark.cli.infra import (
    launch_api_cmd,
    launch_mechanism_workers_cmd,
    launch_workers_cmd,
    list_workers_cmd,
    terminate_workers_cmd,
)
from paper.benchmark.config.constants import STAGE1_SELECTION_TIMEOUT_SECONDS
from paper.benchmark.experiments.cif_mechanism_ablation import (
    mechanism_specification_sha256,
)
from paper.benchmark.experiments.r_cforest_reproducibility import (
    GATE_RECEIPT_PROFILE,
    GATE_RECEIPT_SCHEMA_VERSION,
    gate_receipt_s3_key,
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
    terminate_api,
    validate_image_digest_uri,
)
from paper.benchmark.pipeline.manifest import (
    canonical_manifest_s3_key,
    manifest_s3_key,
)
from paper.benchmark.pipeline.runtime_contract import (
    EXPECTED_THREAD_VALUE,
    PYTHON_LIBRARY_NAMES,
    R_RUNTIME_FIELDS,
    RUNTIME_CONTRACT_PROFILE,
    RUNTIME_CONTRACT_SCHEMA_VERSION,
    THREAD_ENVIRONMENT,
    runtime_contract_s3_key,
    runtime_contract_sha256,
    serialize_runtime_contract,
)
from tests.paper.operator_attestation_fixtures import OPERATOR_PUBLIC_KEY

pytestmark = pytest.mark.paper

DIGEST_URI = (
    "123456789012.dkr.ecr.us-east-1.amazonaws.com/citrees"
    "@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)
MANIFEST_SHA256 = "b" * 64
CANONICAL_MANIFEST_SHA256 = "c" * 64
CAMPAIGN_SHA256 = "e" * 64
API_HOSTED_ZONE_NAME = f"rankings.{CAMPAIGN_SHA256[:32]}.{CAMPAIGN_SHA256[32:]}.citrees.internal."
API_HOSTNAME = f"api.{API_HOSTED_ZONE_NAME}".rstrip(".")
API_URL = f"http://{API_HOSTNAME}:8000"
WORKER_AVAILABILITY_ZONE = "us-east-1a"
WORKER_SUBNET_ID = "subnet-0aaaaaaaaaaaaaaa1"
WORKER_VPC_ID = "vpc-test"
MANIFEST_KEY = f"rerun-manifests/{MANIFEST_SHA256}.csv"
CANONICAL_MANIFEST_KEY = f"canonical-rerun-manifests/{CANONICAL_MANIFEST_SHA256}.csv"
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


def _worker_instance(
    instance_id: str,
    *,
    availability_zone: str = WORKER_AVAILABILITY_ZONE,
    instance_type: str = "c6a.8xlarge",
    subnet_id: str = WORKER_SUBNET_ID,
) -> dict[str, object]:
    """Build one exact on-demand worker instance row."""
    return {
        "InstanceId": instance_id,
        "InstanceType": instance_type,
        "Placement": {"AvailabilityZone": availability_zone},
        "SubnetId": subnet_id,
    }


def _configure_worker_subnets(ec2: MagicMock) -> None:
    """Configure one compatible default subnet for benchmark worker tests."""
    ec2.describe_instance_type_offerings.return_value = {
        "InstanceTypeOfferings": [{"Location": WORKER_AVAILABILITY_ZONE}]
    }
    ec2.describe_subnets.return_value = {
        "Subnets": [
            {
                "AvailabilityZone": WORKER_AVAILABILITY_ZONE,
                "AvailableIpAddressCount": 100,
                "DefaultForAz": True,
                "MapPublicIpOnLaunch": True,
                "State": "available",
                "SubnetId": WORKER_SUBNET_ID,
                "VpcId": WORKER_VPC_ID,
            }
        ]
    }


def _runtime_contract() -> dict[str, object]:
    """Return one canonical launch contract matching the image fixture."""
    return {
        "schema_version": RUNTIME_CONTRACT_SCHEMA_VERSION,
        "profile": RUNTIME_CONTRACT_PROFILE,
        "operator_attestation_public_key": OPERATOR_PUBLIC_KEY,
        "runtime": {
            "ami_id": "ami-test",
            "container_image_digest": "sha256:" + "a" * 64,
            "cpu_affinity": list(range(32)),
            "cpu_model": "AMD EPYC 9R14",
            "git_sha": "a" * 40,
            "instance_type": "c6a.8xlarge",
            "kernel": "6.1.0-fixture",
            "logical_cpus": 32,
            "machine": "x86_64",
            "microcode": "0x1000065",
            "openssl_version": "OpenSSL 3.0.13 30 Jan 2024",
            "os_release": {"ID": "amzn", "VERSION_ID": "2023"},
            "python_libraries": {name: "1.0" for name in PYTHON_LIBRARY_NAMES},
            "r_numerical_libraries": {
                "blas": "/usr/local/lib/R/lib/libRblas.so",
                "lapack": "/usr/local/lib/R/lib/libRlapack.so",
            },
            "r_selection_timeout_seconds": STAGE1_SELECTION_TIMEOUT_SECONDS,
            "r_runtime": {name: "1.0" for name in R_RUNTIME_FIELDS},
            "thread_environment": {name: EXPECTED_THREAD_VALUE for name in THREAD_ENVIRONMENT},
            "threadpools": [
                {
                    "filepath": "/app/.venv/lib/libopenblas.so",
                    "internal_api": "openblas",
                    "num_threads": 1,
                    "prefix": "libopenblas",
                    "user_api": "blas",
                }
            ],
        },
    }


RUNTIME_CONTRACT_SHA256 = runtime_contract_sha256(_runtime_contract())
RUNTIME_CONTRACT_KEY = runtime_contract_s3_key(RUNTIME_CONTRACT_SHA256)
GATE_RECEIPT_PAYLOAD = b'{"fixture":"complete-r-cforest-gate"}'
GATE_RECEIPT_SHA256 = hashlib.sha256(GATE_RECEIPT_PAYLOAD).hexdigest()
GATE_RECEIPT_KEY = gate_receipt_s3_key(GATE_RECEIPT_SHA256)


def _assert_per_boot_recovery(
    script: str,
    container_name: str,
    *,
    restart_container: bool,
) -> None:
    recovery_path = f"/var/lib/cloud/scripts/per-boot/{container_name}-recover"
    assert recovery_path in script
    if restart_container:
        assert f"docker inspect {container_name}" in script
        assert f"docker inspect --format '{{{{.State.Running}}}}' {container_name}" in script
        assert f"docker start {container_name}" in script
        assert f"docker wait {container_name}" in script
    else:
        assert f"Terminating recovered {container_name} instance" in script
        assert f"docker inspect {container_name}" not in script
        assert f"docker start {container_name}" not in script
    assert "chmod 0755" in script
    assert script.index(recovery_path) < script.index("docker run -d --restart no")
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def _write_runtime_contract(directory: Path) -> Path:
    path = directory / "runtime-contract.json"
    path.write_bytes(serialize_runtime_contract(_runtime_contract()))
    return path


def _write_gate_receipt(directory: Path) -> Path:
    path = directory / "gate-receipt.json"
    path.write_bytes(GATE_RECEIPT_PAYLOAD)
    return path


def _write_canonical_manifest(directory: Path) -> Path:
    path = directory / "canonical-manifest.csv"
    path.write_text("fixture", encoding="utf-8")
    return path


def _published_manifest() -> dict[str, str | int]:
    return {
        "bucket": "citrees-123456789012",
        "key": MANIFEST_KEY,
        "sha256": MANIFEST_SHA256,
        "campaign_sha256": CAMPAIGN_SHA256,
        "canonical_manifest_s3_key": CANONICAL_MANIFEST_KEY,
        "canonical_manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "gate_receipt_sha256": GATE_RECEIPT_SHA256,
        "gate_receipt_s3_key": GATE_RECEIPT_KEY,
        "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
        "runtime_contract_s3_key": RUNTIME_CONTRACT_KEY,
        "cells": 1,
    }


def _api_scope(
    *,
    api_url: str = API_URL,
    artifact_prefix: str = "repairs/run-001",
    market: str = "on-demand",
    public_api_url: str = "http://203.0.113.10:8000",
    runtime_contract_s3_key: str = RUNTIME_CONTRACT_KEY,
    runtime_contract_sha256: str = RUNTIME_CONTRACT_SHA256,
) -> ApiScope:
    """Return one complete running API scope for worker-launch tests."""
    return ApiScope(
        api_url=api_url,
        public_api_url=public_api_url,
        artifact_prefix=artifact_prefix,
        campaign_sha256=CAMPAIGN_SHA256,
        canonical_manifest_s3_key=CANONICAL_MANIFEST_KEY,
        canonical_manifest_sha256=CANONICAL_MANIFEST_SHA256,
        gate_receipt_s3_key=GATE_RECEIPT_KEY,
        gate_receipt_sha256=GATE_RECEIPT_SHA256,
        image_uri=DIGEST_URI,
        manifest_s3_key=MANIFEST_KEY,
        manifest_sha256=MANIFEST_SHA256,
        market=market,
        max_cell_attempts=3,
        runtime_contract_s3_key=runtime_contract_s3_key,
        runtime_contract_sha256=runtime_contract_sha256,
        stage="rankings",
    )


def _api_instance(
    *,
    api_endpoint: str = API_HOSTNAME,
    launch_id: str = "api-initial",
    market: str = "on-demand",
    instance_lifecycle: str | None = None,
) -> dict[str, Any]:
    """Return one API DescribeInstances row with immutable scope tags."""
    instance: dict[str, Any] = {
        "InstanceId": "i-api",
        "PrivateIpAddress": "10.0.0.10",
        "PublicIpAddress": "203.0.113.10",
        "Tags": [
            {"Key": "citrees-api-endpoint", "Value": api_endpoint},
            {"Key": "citrees-api-launch-id", "Value": launch_id},
            {"Key": "citrees-artifact-prefix", "Value": "repairs/run-001"},
            {"Key": "citrees-campaign-sha256", "Value": CAMPAIGN_SHA256},
            {
                "Key": "citrees-canonical-manifest-key",
                "Value": CANONICAL_MANIFEST_KEY,
            },
            {
                "Key": "citrees-canonical-manifest-sha256",
                "Value": CANONICAL_MANIFEST_SHA256,
            },
            {"Key": "citrees-gate-receipt-key", "Value": GATE_RECEIPT_KEY},
            {
                "Key": "citrees-gate-receipt-sha256",
                "Value": GATE_RECEIPT_SHA256,
            },
            {"Key": "citrees-image-uri", "Value": DIGEST_URI},
            {"Key": "citrees-manifest-key", "Value": MANIFEST_KEY},
            {"Key": "citrees-manifest-sha256", "Value": MANIFEST_SHA256},
            {"Key": "citrees-market", "Value": market},
            {"Key": "citrees-max-cell-attempts", "Value": "3"},
            {
                "Key": "citrees-runtime-contract-key",
                "Value": RUNTIME_CONTRACT_KEY,
            },
            {
                "Key": "citrees-runtime-contract-sha256",
                "Value": RUNTIME_CONTRACT_SHA256,
            },
            {"Key": "citrees-stage", "Value": "rankings"},
        ],
    }
    if instance_lifecycle is not None:
        instance["InstanceLifecycle"] = instance_lifecycle
    return instance


class _WorkerLaunchKwargs(TypedDict):
    n: int
    image_uri: str
    artifact_prefix: str
    canonical_manifest_path: Path
    gate_receipt_path: Path
    launch_id: str
    manifest_path: Path
    runtime_contract_path: Path
    stage: str


class _ApiLaunchKwargs(TypedDict):
    instance_type: str
    image_uri: str
    artifact_prefix: str
    canonical_manifest_path: Path
    gate_receipt_path: Path
    manifest_path: Path
    runtime_contract_path: Path
    stage: str
    lease_seconds: int
    max_cell_attempts: int


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


class _MemoryRoute53:
    """In-memory private hosted-zone and record-set double."""

    def __init__(self) -> None:
        self.zones: dict[str, dict[str, Any]] = {}
        self.change_count = 0

    def list_hosted_zones_by_name(self, **kwargs: Any) -> dict[str, object]:
        dns_name = str(kwargs["DNSName"])
        zones = [
            {
                "Id": f"/hostedzone/{zone_id}",
                "Name": zone["name"],
                "Config": {"PrivateZone": True},
            }
            for zone_id, zone in sorted(self.zones.items())
            if str(zone["name"]) >= dns_name
        ]
        return {"HostedZones": zones, "IsTruncated": False}

    def create_hosted_zone(self, **kwargs: Any) -> dict[str, object]:
        zone_id = f"Z{len(self.zones) + 1:08d}"
        self.zones[zone_id] = {
            "name": str(kwargs["Name"]),
            "vpcs": [dict(kwargs["VPC"])],
            "tags": {},
            "records": {},
        }
        return {
            "HostedZone": {
                "Id": f"/hostedzone/{zone_id}",
                "Name": kwargs["Name"],
                "Config": {"PrivateZone": True},
            }
        }

    def change_tags_for_resource(self, **kwargs: Any) -> dict[str, object]:
        zone = self.zones[str(kwargs["ResourceId"])]
        tags = zone["tags"]
        assert isinstance(tags, dict)
        for tag in kwargs.get("AddTags", []):
            tags[str(tag["Key"])] = str(tag["Value"])
        return {}

    def list_tags_for_resource(self, **kwargs: Any) -> dict[str, object]:
        zone_id = str(kwargs["ResourceId"])
        tags = self.zones[zone_id]["tags"]
        assert isinstance(tags, dict)
        return {
            "ResourceTagSet": {
                "ResourceType": "hostedzone",
                "ResourceId": zone_id,
                "Tags": [{"Key": key, "Value": value} for key, value in sorted(tags.items())],
            }
        }

    def get_hosted_zone(self, **kwargs: Any) -> dict[str, object]:
        zone = self.zones[str(kwargs["Id"])]
        return {"VPCs": list(zone["vpcs"])}

    def change_resource_record_sets(self, **kwargs: Any) -> dict[str, object]:
        zone = self.zones[str(kwargs["HostedZoneId"])]
        records = zone["records"]
        assert isinstance(records, dict)
        for change in kwargs["ChangeBatch"]["Changes"]:
            record = dict(change["ResourceRecordSet"])
            key = (str(record["Name"]), str(record["Type"]))
            if change["Action"] == "UPSERT":
                records[key] = record
            elif change["Action"] == "DELETE":
                assert records[key] == record
                del records[key]
            else:
                raise AssertionError(f"unsupported Route53 action: {change['Action']}")
        self.change_count += 1
        return {
            "ChangeInfo": {
                "Id": f"/change/C{self.change_count:08d}",
                "Status": "INSYNC",
            }
        }

    def get_change(self, **kwargs: Any) -> dict[str, object]:
        return {"ChangeInfo": {"Id": kwargs["Id"], "Status": "INSYNC"}}

    def list_resource_record_sets(self, **kwargs: Any) -> dict[str, object]:
        zone = self.zones[str(kwargs["HostedZoneId"])]
        zone_name = str(zone["name"])
        records = [
            {
                "Name": zone_name,
                "Type": "NS",
                "TTL": 172800,
                "ResourceRecords": [{"Value": "ns.example.internal."}],
            },
            {
                "Name": zone_name,
                "Type": "SOA",
                "TTL": 900,
                "ResourceRecords": [
                    {
                        "Value": "ns.example.internal. hostmaster.example.internal. 1 7200 900 1209600 86400"
                    }
                ],
            },
        ]
        custom_records = zone["records"]
        assert isinstance(custom_records, dict)
        records.extend(dict(record) for record in custom_records.values())
        records.sort(key=lambda record: (str(record["Name"]), str(record["Type"])))
        start_name = kwargs.get("StartRecordName")
        start_type = kwargs.get("StartRecordType")
        if isinstance(start_name, str):
            records = [
                record
                for record in records
                if (str(record["Name"]), str(record["Type"])) >= (start_name, str(start_type or ""))
            ]
        max_items = kwargs.get("MaxItems")
        if max_items is not None:
            records = records[: int(max_items)]
        return {"ResourceRecordSets": records, "IsTruncated": False}

    def delete_hosted_zone(self, **kwargs: Any) -> dict[str, object]:
        zone_id = str(kwargs["Id"])
        records = self.zones[zone_id]["records"]
        assert records == {}
        del self.zones[zone_id]
        return {}

    def target(self) -> str | None:
        """Return the only custom A-record target."""
        targets = [
            str(record["ResourceRecords"][0]["Value"])
            for zone in self.zones.values()
            for record in zone["records"].values()
        ]
        assert len(targets) <= 1
        return targets[0] if targets else None


class _MemoryEc2:
    """In-memory EC2 API-instance lifecycle double."""

    def __init__(self) -> None:
        self.instances: dict[str, dict[str, Any]] = {}
        self.run_requests: list[dict[str, Any]] = []

    def describe_vpcs(self, **kwargs: Any) -> dict[str, object]:
        assert kwargs["Filters"] == [{"Name": "is-default", "Values": ["true"]}]
        return {"Vpcs": [{"VpcId": "vpc-default"}]}

    def run_instances(self, **kwargs: Any) -> dict[str, object]:
        self.run_requests.append(kwargs)
        index = len(self.run_requests)
        instance_id = f"i-api-{index}"
        tags = [dict(tag) for tag in kwargs["TagSpecifications"][0]["Tags"]]
        self.instances[instance_id] = {
            "InstanceId": instance_id,
            "PrivateIpAddress": f"10.0.0.{index * 10}",
            "PublicIpAddress": f"203.0.113.{index * 10}",
            "State": {"Name": "running"},
            "Tags": tags,
        }
        return {"Instances": [{"InstanceId": instance_id}]}

    def describe_instances(self, **kwargs: Any) -> dict[str, object]:
        filters = kwargs["Filters"]
        selected: list[dict[str, Any]] = []
        for instance in self.instances.values():
            tags = {tag["Key"]: tag["Value"] for tag in instance["Tags"]}
            state = str(instance["State"]["Name"])
            matches = True
            for item in filters:
                name = str(item["Name"])
                values = [str(value) for value in item["Values"]]
                if name == "instance-state-name":
                    matches = matches and state in values
                elif name.startswith("tag:"):
                    matches = matches and tags.get(name.removeprefix("tag:")) in values
                else:
                    raise AssertionError(f"unsupported EC2 filter: {name}")
            if matches:
                selected.append(instance)
        return {"Reservations": ([{"Instances": selected}] if selected else [])}

    def terminate_instances(self, **kwargs: Any) -> dict[str, object]:
        for instance_id in kwargs["InstanceIds"]:
            self.instances[str(instance_id)]["State"] = {"Name": "terminated"}
        return {}

    def interrupt(self, instance_id: str) -> None:
        """Simulate one completed instance loss without endpoint cleanup."""
        self.instances[instance_id]["State"] = {"Name": "terminated"}


class _MemoryEc2Instance:
    """Boto3 resource view over one in-memory EC2 instance."""

    def __init__(self, ec2: _MemoryEc2, instance_id: str) -> None:
        self.ec2 = ec2
        self.instance_id = instance_id

    @property
    def public_ip_address(self) -> str:
        return str(self.ec2.instances[self.instance_id]["PublicIpAddress"])

    @property
    def private_ip_address(self) -> str:
        return str(self.ec2.instances[self.instance_id]["PrivateIpAddress"])

    def wait_until_running(self) -> None:
        assert self.ec2.instances[self.instance_id]["State"] == {"Name": "running"}

    def reload(self) -> None:
        return None


class _MemoryEc2Resource:
    """Boto3 EC2 resource double."""

    def __init__(self, ec2: _MemoryEc2) -> None:
        self.ec2 = ec2

    def Instance(self, instance_id: str) -> _MemoryEc2Instance:
        return _MemoryEc2Instance(self.ec2, instance_id)


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
    [launch_api_cmd, launch_workers_cmd, launch_mechanism_workers_cmd],
)
def test_distributed_launch_market_is_not_operator_selectable(command: object) -> None:
    assert "market" not in inspect.signature(command).parameters
    assert "spot" not in inspect.signature(command).parameters


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "launch-api",
            "--image-uri",
            DIGEST_URI,
            "--artifact-prefix",
            "repairs/run-001",
            "--launch-id",
            "api-initial",
            "--manifest",
            "{manifest}",
            "--max-cell-attempts",
            "3",
        ],
        [
            "launch-workers",
            "--image-uri",
            DIGEST_URI,
            "--artifact-prefix",
            "repairs/run-001",
            "--launch-id",
            "scale-001",
            "--manifest",
            "{manifest}",
        ],
    ],
)
def test_runtime_contract_cli_inputs_are_required(
    arguments: list[str],
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("fixture", encoding="utf-8")
    resolved = [str(manifest_path) if value == "{manifest}" else value for value in arguments]

    result = CliRunner().invoke(
        infra_app,
        resolved,
        env={
            "CITREES_RUNTIME_CONTRACT_SHA256": "",
            "CITREES_RUNTIME_CONTRACT_S3_KEY": "",
        },
    )

    assert result.exit_code == 2
    assert "--runtime-contract" in result.output
    assert "--gate-receipt" in result.output


def test_runtime_contract_cli_file_is_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("fixture", encoding="utf-8")
    canonical_manifest_path = _write_canonical_manifest(tmp_path)
    runtime_contract_path = _write_runtime_contract(tmp_path)
    gate_receipt_path = _write_gate_receipt(tmp_path)
    launch_api_mock = MagicMock(
        return_value={
            "instance_id": "i-api",
            "launch_id": "api-initial",
            "public_ip": "203.0.113.10",
            "api_url": API_URL,
            "public_api_url": "http://203.0.113.10:8000",
        }
    )
    launch_workers_mock = MagicMock(return_value=["i-worker"])
    monkeypatch.setattr(ec2_infra, "launch_api", launch_api_mock)
    monkeypatch.setattr(ec2_infra, "launch_workers", launch_workers_mock)
    api_result = CliRunner().invoke(
        infra_app,
        [
            "launch-api",
            "--image-uri",
            DIGEST_URI,
            "--artifact-prefix",
            "repairs/run-001",
            "--launch-id",
            "api-initial",
            "--canonical-manifest",
            str(canonical_manifest_path),
            "--manifest",
            str(manifest_path),
            "--runtime-contract",
            str(runtime_contract_path),
            "--gate-receipt",
            str(gate_receipt_path),
            "--max-cell-attempts",
            "3",
        ],
    )
    worker_result = CliRunner().invoke(
        infra_app,
        [
            "launch-workers",
            "--image-uri",
            DIGEST_URI,
            "--artifact-prefix",
            "repairs/run-001",
            "--launch-id",
            "scale-001",
            "--canonical-manifest",
            str(canonical_manifest_path),
            "--manifest",
            str(manifest_path),
            "--runtime-contract",
            str(runtime_contract_path),
            "--gate-receipt",
            str(gate_receipt_path),
            "--subnets",
            "subnet-0fff,subnet-0aaa",
            "--exclude-availability-zones",
            "us-east-1d",
        ],
    )

    assert api_result.exit_code == 0, api_result.output
    assert worker_result.exit_code == 0, worker_result.output
    assert launch_api_mock.call_args.kwargs["launch_id"] == "api-initial"
    assert launch_api_mock.call_args.kwargs["canonical_manifest_path"] == canonical_manifest_path
    assert launch_api_mock.call_args.kwargs["gate_receipt_path"] == gate_receipt_path
    assert launch_api_mock.call_args.kwargs["runtime_contract_path"] == runtime_contract_path
    assert launch_workers_mock.call_args.kwargs["gate_receipt_path"] == gate_receipt_path
    assert launch_workers_mock.call_args.kwargs["subnet_ids"] == (
        "subnet-0fff",
        "subnet-0aaa",
    )
    assert launch_workers_mock.call_args.kwargs["excluded_availability_zones"] == ("us-east-1d",)
    assert (
        launch_workers_mock.call_args.kwargs["canonical_manifest_path"] == canonical_manifest_path
    )
    assert launch_workers_mock.call_args.kwargs["runtime_contract_path"] == runtime_contract_path


def test_runtime_contract_cli_parameters_are_explicit() -> None:
    for command in (launch_api_cmd, launch_workers_cmd):
        parameters = inspect.signature(command).parameters
        assert parameters["canonical_manifest_path"].default is None
        assert parameters["gate_receipt_path"].default is None
        assert parameters["runtime_contract_path"].default is None
        assert "runtime_contract_sha256" not in parameters
        assert "runtime_contract_s3_key" not in parameters


@pytest.mark.parametrize("command", ["api-url", "terminate-api"])
def test_api_lifecycle_cli_requires_exact_campaign_scope(command: str) -> None:
    result = CliRunner().invoke(infra_app, [command])

    assert result.exit_code == 2
    assert "--artifact-prefix" in result.output
    assert "--campaign-sha256" in result.output


def test_api_lifecycle_cli_forwards_exact_campaign_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_scope = MagicMock(return_value=SimpleNamespace(public_api_url="http://203.0.113.10:8000"))
    terminate = MagicMock(return_value="i-api")
    monkeypatch.setattr(ec2_infra, "get_api_scope", get_scope)
    monkeypatch.setattr(ec2_infra, "terminate_api", terminate)
    arguments = [
        "--artifact-prefix",
        "repairs/run-001",
        "--campaign-sha256",
        CAMPAIGN_SHA256,
        "--stage",
        "rankings",
    ]

    url_result = CliRunner().invoke(infra_app, ["api-url", *arguments])
    terminate_result = CliRunner().invoke(infra_app, ["terminate-api", *arguments])

    assert url_result.exit_code == 0, url_result.output
    assert "http://203.0.113.10:8000" in url_result.output
    assert terminate_result.exit_code == 0, terminate_result.output
    expected = {
        "artifact_prefix": "repairs/run-001",
        "campaign_sha256": CAMPAIGN_SHA256,
        "stage": "rankings",
    }
    get_scope.assert_called_once_with(**expected)
    terminate.assert_called_once_with(**expected)


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
    assert "COPY paper/benchmark ./paper/benchmark" in dockerfile
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


def test_candidate_image_verification_invokes_r_cforest_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    git_sha = "a" * 40
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        del kwargs
        commands.append(command)
        stdout = git_sha if command[:2] == ["docker", "inspect"] else ""
        return SimpleNamespace(returncode=0, stdout=stdout)

    monkeypatch.setattr(aws_infra.subprocess, "run", run)

    aws_infra.verify_candidate_image(
        "citrees:test",
        git_sha,
        docker_env={},
    )

    docker_runs = [command for command in commands if command[:2] == ["docker", "run"]]
    assert docker_runs
    assert all(command[2:4] == ["--rm", "--init"] for command in docker_runs)
    assert [
        "docker",
        "run",
        "--rm",
        "--init",
        "citrees:test",
        "python",
        "-m",
        "paper.benchmark.experiments.r_cforest_reproducibility",
        "--help",
    ] in commands


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
        "launch_id": "api-initial",
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


def test_container_recovery_hook_rejects_shell_metacharacters() -> None:
    with pytest.raises(ValueError, match="invalid container name"):
        ec2_infra.per_boot_container_recovery_hook(
            "worker; reboot",
            restart_container=False,
        )


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
    _assert_per_boot_recovery(
        script,
        "citrees-mechanism",
        restart_container=False,
    )


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
        read_keys=(),
        write_prefixes=(expected_prefix,),
        region="us-east-1",
    )
    assert client.run_instances.call_args.kwargs["IamInstanceProfile"] == {
        "Name": "citrees-campaign-test"
    }
    assert client.run_instances.call_args.kwargs["MetadataOptions"] == {
        "HttpEndpoint": "enabled",
        "HttpPutResponseHopLimit": 2,
        "HttpTokens": "required",
    }
    assert client.run_instances.call_args.kwargs["InstanceMarketOptions"] == {
        "MarketType": "spot",
        "SpotOptions": {
            "SpotInstanceType": "one-time",
            "InstanceInterruptionBehavior": "terminate",
        },
    }
    tags = {
        tag["Key"]: tag["Value"]
        for tag in client.run_instances.call_args.kwargs["TagSpecifications"][0]["Tags"]
    }
    assert tags["citrees-market"] == "spot"


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
        canonical_manifest_s3_key=CANONICAL_MANIFEST_KEY,
        canonical_manifest_sha256=CANONICAL_MANIFEST_SHA256,
        gate_receipt_s3_key=GATE_RECEIPT_KEY,
        gate_receipt_sha256=GATE_RECEIPT_SHA256,
        manifest_s3_key=MANIFEST_KEY,
        manifest_sha256=MANIFEST_SHA256,
        runtime_contract_s3_key=RUNTIME_CONTRACT_KEY,
        runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
        stage="rankings",
        lease_seconds=900,
        max_cell_attempts=3,
    )

    assert "-e CITREES_ARTIFACT_PREFIX=repairs/r-baselines/run-001" in script
    assert f"-e CITREES_CAMPAIGN_SHA256={CAMPAIGN_SHA256}" in script
    assert f"-e CITREES_CANONICAL_MANIFEST_S3_KEY={CANONICAL_MANIFEST_KEY}" in script
    assert f"-e CITREES_CANONICAL_MANIFEST_SHA256={CANONICAL_MANIFEST_SHA256}" in script
    assert f"-e CITREES_GATE_RECEIPT_S3_KEY={GATE_RECEIPT_KEY}" in script
    assert f"-e CITREES_GATE_RECEIPT_SHA256={GATE_RECEIPT_SHA256}" in script
    assert f"-e CITREES_MANIFEST_S3_KEY={MANIFEST_KEY}" in script
    assert f"-e CITREES_MANIFEST_SHA256={MANIFEST_SHA256}" in script
    assert f"-e CITREES_RUNTIME_CONTRACT_S3_KEY={RUNTIME_CONTRACT_KEY}" in script
    assert f"-e CITREES_RUNTIME_CONTRACT_SHA256={RUNTIME_CONTRACT_SHA256}" in script
    assert "-e CITREES_STAGE=rankings" in script
    assert "-e CITREES_LEASE_SECONDS=900" in script
    assert "-e CITREES_MAX_CELL_ATTEMPTS=3" in script
    assert f"-e CITREES_IMAGE_URI={DIGEST_URI}" in script
    assert "-e EC2_AMI_ID=$AMI_ID" in script
    assert "-e EC2_AVAILABILITY_ZONE=$AVAILABILITY_ZONE" in script
    assert "-e EC2_INSTANCE_ID=$INSTANCE_ID" in script
    assert "-e EC2_INSTANCE_TYPE=m5.large" in script
    assert "AWS_ACCOUNT_ID" not in script
    assert f"docker pull {DIGEST_URI}" in script
    assert "trap shutdown_instance EXIT" in script
    assert script.index("trap shutdown_instance EXIT") < script.index("# Instance metadata")
    assert "shutdown -h now || systemctl poweroff --force --force" in script
    assert "docker run -d --restart no" in script
    assert "--name citrees-api" in script
    assert "docker wait citrees-api" in script
    _assert_per_boot_recovery(
        script,
        "citrees-api",
        restart_container=True,
    )
    assert "curl --fail --silent --show-error --request PUT" in script
    assert "latest/meta-data/instance-id" in script
    assert "latest/meta-data/placement/availability-zone" in script
    assert "latest/meta-data/ami-id" in script
    assert "${INSTANCE_ID:-" not in script
    assert "${AVAILABILITY_ZONE:-" not in script
    assert "${AMI_ID:-" not in script


def test_worker_user_data_matches_api_scope() -> None:
    script = _make_worker_user_data(
        region="us-east-1",
        ecr_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com",
        image_uri=DIGEST_URI,
        api_url=API_URL,
        bucket="citrees-123456789012",
        git_sha="abc123",
        instance_type="c6a.8xlarge",
        artifact_prefix="repairs/r-baselines/run-001",
        campaign_sha256=CAMPAIGN_SHA256,
        canonical_manifest_s3_key=CANONICAL_MANIFEST_KEY,
        canonical_manifest_sha256=CANONICAL_MANIFEST_SHA256,
        gate_receipt_s3_key=GATE_RECEIPT_KEY,
        gate_receipt_sha256=GATE_RECEIPT_SHA256,
        manifest_s3_key=MANIFEST_KEY,
        manifest_sha256=MANIFEST_SHA256,
        runtime_contract_s3_key=RUNTIME_CONTRACT_KEY,
        runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
        stage="rankings",
    )

    assert "-e CITREES_ARTIFACT_PREFIX=repairs/r-baselines/run-001" in script
    assert f"-e CITREES_CAMPAIGN_SHA256={CAMPAIGN_SHA256}" in script
    assert f"-e CITREES_CANONICAL_MANIFEST_S3_KEY={CANONICAL_MANIFEST_KEY}" in script
    assert f"-e CITREES_CANONICAL_MANIFEST_SHA256={CANONICAL_MANIFEST_SHA256}" in script
    assert f"-e CITREES_GATE_RECEIPT_S3_KEY={GATE_RECEIPT_KEY}" in script
    assert f"-e CITREES_GATE_RECEIPT_SHA256={GATE_RECEIPT_SHA256}" in script
    assert f"-e CITREES_MANIFEST_S3_KEY={MANIFEST_KEY}" in script
    assert f"-e CITREES_MANIFEST_SHA256={MANIFEST_SHA256}" in script
    assert f"-e CITREES_RUNTIME_CONTRACT_S3_KEY={RUNTIME_CONTRACT_KEY}" in script
    assert f"-e CITREES_RUNTIME_CONTRACT_SHA256={RUNTIME_CONTRACT_SHA256}" in script
    assert "-e CITREES_STAGE=rankings" in script
    assert f"-e CITREES_IMAGE_URI={DIGEST_URI}" in script
    assert "-e EC2_AMI_ID=$AMI_ID" in script
    assert "-e EC2_AVAILABILITY_ZONE=$AVAILABILITY_ZONE" in script
    assert "-e EC2_INSTANCE_ID=$INSTANCE_ID" in script
    assert "-e EC2_INSTANCE_TYPE=c6a.8xlarge" in script
    assert "AWS_ACCOUNT_ID" not in script
    assert "docker run -d --restart no" in script
    assert "--init" in script
    assert "--restart on-failure" not in script
    _assert_per_boot_recovery(
        script,
        "citrees-worker",
        restart_container=False,
    )
    assert f"-e CITREES_API_URL={API_URL}" in script
    assert f"--api-url {API_URL}" in script
    assert "10.0.0.10" not in script
    assert "trap shutdown_instance EXIT" in script
    assert script.index("trap shutdown_instance EXIT") < script.index("# Instance metadata")
    assert "shutdown -h now || systemctl poweroff --force --force" in script
    assert "curl --fail --silent --show-error --request PUT" in script
    assert "latest/meta-data/instance-id" in script
    assert "latest/meta-data/placement/availability-zone" in script
    assert "latest/meta-data/ami-id" in script
    assert "${INSTANCE_ID:-" not in script
    assert "${AVAILABILITY_ZONE:-" not in script
    assert "${AMI_ID:-" not in script


def _mock_api_launch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    public_ip: str | None,
) -> tuple[MagicMock, MagicMock]:
    ec2_client = MagicMock()
    ec2_client.run_instances.return_value = {"Instances": [{"InstanceId": "i-api"}]}
    instance = MagicMock()
    instance.public_ip_address = public_ip
    instance.private_ip_address = "10.0.0.10"
    ec2_resource = MagicMock()
    ec2_resource.Instance.return_value = instance
    endpoint = ec2_infra._ApiEndpoint(
        hosted_zone_id="ZTEST",
        hosted_zone_name=API_HOSTED_ZONE_NAME,
        hostname=API_HOSTNAME,
        record_name=f"{API_HOSTNAME}.",
    )

    monkeypatch.setattr(
        ec2_infra,
        "_api_instances",
        MagicMock(side_effect=[[], [{"InstanceId": "i-api"}]]),
    )
    monkeypatch.setattr(ec2_infra, "_ensure_api_endpoint", MagicMock(return_value=endpoint))
    monkeypatch.setattr(ec2_infra, "_upsert_api_endpoint", MagicMock())
    monkeypatch.setattr(ec2_infra, "_delete_api_endpoint", MagicMock(return_value=True))
    monkeypatch.setattr(ec2_infra, "validate_image_revision", lambda image_uri, region: "a" * 40)
    monkeypatch.setattr(
        ec2_infra,
        "publish_rerun_manifest",
        lambda manifest_path,
        canonical_manifest_path,
        runtime_contract_path,
        gate_receipt_path,
        *,
        region: {
            "key": MANIFEST_KEY,
            "sha256": MANIFEST_SHA256,
            "campaign_sha256": CAMPAIGN_SHA256,
            "canonical_manifest_s3_key": CANONICAL_MANIFEST_KEY,
            "canonical_manifest_sha256": CANONICAL_MANIFEST_SHA256,
            "gate_receipt_sha256": GATE_RECEIPT_SHA256,
            "gate_receipt_s3_key": GATE_RECEIPT_KEY,
            "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
            "runtime_contract_s3_key": RUNTIME_CONTRACT_KEY,
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
        launch_id="api-initial",
        canonical_manifest_path=_write_canonical_manifest(tmp_path),
        gate_receipt_path=_write_gate_receipt(tmp_path),
        manifest_path=tmp_path / "manifest.csv",
        runtime_contract_path=_write_runtime_contract(tmp_path),
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
    assert ec2_client.run_instances.call_args.kwargs["MetadataOptions"] == {
        "HttpEndpoint": "enabled",
        "HttpPutResponseHopLimit": 2,
        "HttpTokens": "required",
    }
    tags = {
        tag["Key"]: tag["Value"]
        for tag in ec2_client.run_instances.call_args.kwargs["TagSpecifications"][0]["Tags"]
    }
    assert tags["citrees-api-endpoint"] == API_HOSTNAME
    assert tags["citrees-api-launch-id"] == "api-initial"
    assert tags["citrees-gate-receipt-key"] == GATE_RECEIPT_KEY
    assert tags["citrees-gate-receipt-sha256"] == GATE_RECEIPT_SHA256
    assert tags["citrees-runtime-contract-key"] == RUNTIME_CONTRACT_KEY
    assert tags["citrees-runtime-contract-sha256"] == RUNTIME_CONTRACT_SHA256
    readiness_call = ec2_infra._wait_for_api_ready.call_args
    assert readiness_call.kwargs["gate_receipt_s3_key"] == GATE_RECEIPT_KEY
    assert readiness_call.kwargs["gate_receipt_sha256"] == GATE_RECEIPT_SHA256
    assert readiness_call.kwargs["runtime_contract_s3_key"] == RUNTIME_CONTRACT_KEY
    assert readiness_call.kwargs["runtime_contract_sha256"] == RUNTIME_CONTRACT_SHA256


def test_api_launch_requests_on_demand_without_market_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ec2_client, _instance = _mock_api_launch(monkeypatch, public_ip="203.0.113.10")
    monkeypatch.setattr(ec2_infra, "_wait_for_api_ready", MagicMock())

    _launch_test_api(tmp_path)

    kwargs = ec2_client.run_instances.call_args.kwargs
    assert "InstanceMarketOptions" not in kwargs
    tags = {tag["Key"]: tag["Value"] for tag in kwargs["TagSpecifications"][0]["Tags"]}
    assert tags["citrees-market"] == "on-demand"


def test_campaign_endpoint_supports_api_replacement_and_scoped_teardown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ec2 = _MemoryEc2()
    route53 = _MemoryRoute53()
    monkeypatch.setattr(
        ec2_infra,
        "publish_rerun_manifest",
        lambda manifest_path,
        canonical_manifest_path,
        runtime_contract_path,
        gate_receipt_path,
        *,
        region: (_published_manifest()),
    )
    monkeypatch.setattr(
        ec2_infra,
        "validate_image_revision",
        lambda image_uri, region: "a" * 40,
    )
    monkeypatch.setattr(ec2_infra, "get_aws_account_id", lambda: "123456789012")
    monkeypatch.setattr(ec2_infra, "ensure_security_group", lambda region: "sg-test")
    monkeypatch.setattr(
        ec2_infra,
        "ensure_campaign_iam_profile",
        lambda **kwargs: "citrees-campaign-test",
    )
    monkeypatch.setattr(ec2_infra, "_wait_for_api_ready", MagicMock())
    monkeypatch.setattr(
        ec2_infra.boto3,
        "client",
        lambda service, **kwargs: {"ec2": ec2, "route53": route53}[service],
    )
    monkeypatch.setattr(
        ec2_infra.boto3,
        "resource",
        lambda service, **kwargs: _MemoryEc2Resource(ec2),
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("fixture", encoding="utf-8")
    launch_kwargs: _ApiLaunchKwargs = {
        "instance_type": "m5.large",
        "image_uri": DIGEST_URI,
        "artifact_prefix": "repairs/run-001",
        "canonical_manifest_path": _write_canonical_manifest(tmp_path),
        "gate_receipt_path": _write_gate_receipt(tmp_path),
        "manifest_path": manifest_path,
        "runtime_contract_path": _write_runtime_contract(tmp_path),
        "stage": "rankings",
        "lease_seconds": 900,
        "max_cell_attempts": 1,
    }

    first = launch_api(launch_id="api-initial", **launch_kwargs)
    assert first["api_url"] == API_URL
    assert first["public_api_url"] == "http://203.0.113.10:8000"
    assert route53.target() == "10.0.0.10"
    assert len(route53.zones) == 1
    first_zone = next(iter(route53.zones.values()))
    assert first_zone["name"] == API_HOSTED_ZONE_NAME
    assert first_zone["vpcs"] == [{"VPCId": "vpc-default", "VPCRegion": "us-east-1"}]
    first_record = next(iter(first_zone["records"].values()))
    assert first_record["TTL"] == 10

    ec2.interrupt(first["instance_id"])
    replacement = launch_api(launch_id="api-replacement", **launch_kwargs)

    assert replacement["api_url"] == first["api_url"] == API_URL
    assert replacement["public_api_url"] == "http://203.0.113.20:8000"
    assert route53.target() == "10.0.0.20"
    assert len(route53.zones) == 1
    assert len(ec2.run_requests) == 2
    assert ec2.run_requests[0]["UserData"] == ec2.run_requests[1]["UserData"]
    assert ec2.run_requests[0]["ClientToken"] != ec2.run_requests[1]["ClientToken"]
    assert all("InstanceMarketOptions" not in request for request in ec2.run_requests)
    launch_ids = [
        {tag["Key"]: tag["Value"] for tag in request["TagSpecifications"][0]["Tags"]}[
            "citrees-api-launch-id"
        ]
        for request in ec2.run_requests
    ]
    assert launch_ids == ["api-initial", "api-replacement"]
    active = ec2.describe_instances(
        Filters=ec2_infra._api_instance_filters(
            artifact_prefix="repairs/run-001",
            campaign_sha256=CAMPAIGN_SHA256,
            stage="rankings",
            states=("pending", "running", "stopping"),
        )
    )
    assert [
        instance["InstanceId"]
        for reservation in active["Reservations"]
        for instance in reservation["Instances"]
    ] == [replacement["instance_id"]]
    scope = get_api_scope(
        artifact_prefix="repairs/run-001",
        campaign_sha256=CAMPAIGN_SHA256,
        stage="rankings",
    )
    assert scope is not None
    assert scope.api_url == API_URL
    assert scope.public_api_url == replacement["public_api_url"]

    with pytest.raises(RuntimeError, match="already active"):
        launch_api(launch_id="api-concurrent", **launch_kwargs)
    assert len(ec2.run_requests) == 2

    unrelated_campaign = "f" * 64
    unrelated_endpoint = ec2_infra._ensure_api_endpoint(
        ec2,
        route53,
        artifact_prefix="repairs/run-002",
        campaign_sha256=unrelated_campaign,
        launch_id="api-unrelated",
        stage="rankings",
        region="us-east-1",
    )
    ec2_infra._upsert_api_endpoint(route53, unrelated_endpoint, "10.0.0.99")
    ec2.interrupt(replacement["instance_id"])

    assert (
        terminate_api(
            artifact_prefix="repairs/run-001",
            campaign_sha256=CAMPAIGN_SHA256,
            stage="rankings",
        )
        is None
    )
    assert len(route53.zones) == 1
    assert route53.target() == "10.0.0.99"


def test_running_api_scope_requires_complete_immutable_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    instance = _api_instance()
    client.describe_instances.return_value = {"Reservations": [{"Instances": [instance]}]}
    monkeypatch.setattr(ec2_infra.boto3, "client", lambda *args, **kwargs: client)

    scope = get_api_scope(
        artifact_prefix="repairs/run-001",
        campaign_sha256=CAMPAIGN_SHA256,
        stage="rankings",
    )

    assert scope == ApiScope(
        api_url=API_URL,
        public_api_url="http://203.0.113.10:8000",
        artifact_prefix="repairs/run-001",
        campaign_sha256=CAMPAIGN_SHA256,
        canonical_manifest_s3_key=CANONICAL_MANIFEST_KEY,
        canonical_manifest_sha256=CANONICAL_MANIFEST_SHA256,
        gate_receipt_s3_key=GATE_RECEIPT_KEY,
        gate_receipt_sha256=GATE_RECEIPT_SHA256,
        image_uri=DIGEST_URI,
        manifest_s3_key=MANIFEST_KEY,
        manifest_sha256=MANIFEST_SHA256,
        market="on-demand",
        max_cell_attempts=3,
        runtime_contract_s3_key=RUNTIME_CONTRACT_KEY,
        runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
        stage="rankings",
    )

    complete_tags = list(instance["Tags"])
    for missing_key in (
        "citrees-api-endpoint",
        "citrees-api-launch-id",
        "citrees-canonical-manifest-key",
        "citrees-canonical-manifest-sha256",
        "citrees-gate-receipt-key",
        "citrees-gate-receipt-sha256",
        "citrees-runtime-contract-key",
        "citrees-runtime-contract-sha256",
    ):
        instance["Tags"] = [tag for tag in complete_tags if tag["Key"] != missing_key]
        with pytest.raises(RuntimeError, match="missing scope tags"):
            get_api_scope(
                artifact_prefix="repairs/run-001",
                campaign_sha256=CAMPAIGN_SHA256,
                stage="rankings",
            )
    instance["Tags"] = complete_tags


@pytest.mark.parametrize(
    ("market", "instance_lifecycle", "message"),
    [
        ("on-demand", "spot", "must be on-demand"),
        ("spot", None, "market tag must be 'on-demand'"),
        ("invalid", None, "market tag must be 'on-demand'"),
    ],
)
def test_running_api_scope_rejects_unproven_market(
    market: str,
    instance_lifecycle: str | None,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.describe_instances.return_value = {
        "Reservations": [
            {
                "Instances": [
                    _api_instance(
                        market=market,
                        instance_lifecycle=instance_lifecycle,
                    )
                ]
            }
        ]
    }
    monkeypatch.setattr(ec2_infra.boto3, "client", lambda *args, **kwargs: client)

    with pytest.raises(RuntimeError, match=message):
        get_api_scope(
            artifact_prefix="repairs/run-001",
            campaign_sha256=CAMPAIGN_SHA256,
            stage="rankings",
        )


def test_api_discovery_and_termination_are_isolated_by_campaign_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    second_campaign_sha256 = "f" * 64

    def instance(
        *,
        instance_id: str,
        artifact_prefix: str,
        campaign_sha256: str,
        stage: str,
        private_ip: str,
        public_ip: str,
    ) -> dict[str, object]:
        endpoint = f"api.{stage}.{campaign_sha256[:32]}.{campaign_sha256[32:]}.citrees.internal"
        return {
            "InstanceId": instance_id,
            "PrivateIpAddress": private_ip,
            "PublicIpAddress": public_ip,
            "Tags": [
                {"Key": "citrees-api-endpoint", "Value": endpoint},
                {"Key": "citrees-api-launch-id", "Value": f"api-{instance_id}"},
                {"Key": "citrees-artifact-prefix", "Value": artifact_prefix},
                {"Key": "citrees-campaign-sha256", "Value": campaign_sha256},
                {
                    "Key": "citrees-canonical-manifest-key",
                    "Value": CANONICAL_MANIFEST_KEY,
                },
                {
                    "Key": "citrees-canonical-manifest-sha256",
                    "Value": CANONICAL_MANIFEST_SHA256,
                },
                {"Key": "citrees-gate-receipt-key", "Value": GATE_RECEIPT_KEY},
                {
                    "Key": "citrees-gate-receipt-sha256",
                    "Value": GATE_RECEIPT_SHA256,
                },
                {"Key": "citrees-image-uri", "Value": DIGEST_URI},
                {"Key": "citrees-manifest-key", "Value": MANIFEST_KEY},
                {"Key": "citrees-manifest-sha256", "Value": MANIFEST_SHA256},
                {"Key": "citrees-market", "Value": "on-demand"},
                {"Key": "citrees-max-cell-attempts", "Value": "3"},
                {
                    "Key": "citrees-runtime-contract-key",
                    "Value": RUNTIME_CONTRACT_KEY,
                },
                {
                    "Key": "citrees-runtime-contract-sha256",
                    "Value": RUNTIME_CONTRACT_SHA256,
                },
                {"Key": "citrees-stage", "Value": stage},
            ],
        }

    instances = {
        ("repairs/run-001", CAMPAIGN_SHA256, "rankings"): instance(
            instance_id="i-first",
            artifact_prefix="repairs/run-001",
            campaign_sha256=CAMPAIGN_SHA256,
            stage="rankings",
            private_ip="10.0.0.10",
            public_ip="203.0.113.10",
        ),
        ("repairs/run-002", second_campaign_sha256, "rankings"): instance(
            instance_id="i-second",
            artifact_prefix="repairs/run-002",
            campaign_sha256=second_campaign_sha256,
            stage="rankings",
            private_ip="10.0.0.20",
            public_ip="203.0.113.20",
        ),
    }
    client = MagicMock()

    def describe_instances(*, Filters: list[dict[str, object]]) -> dict[str, object]:
        filters = {item["Name"]: item["Values"] for item in Filters}
        key = (
            filters["tag:citrees-artifact-prefix"][0],
            filters["tag:citrees-campaign-sha256"][0],
            filters["tag:citrees-stage"][0],
        )
        selected = instances.get(key)
        return {"Reservations": ([{"Instances": [selected]}] if selected is not None else [])}

    client.describe_instances.side_effect = describe_instances
    delete_endpoint = MagicMock(return_value=True)
    monkeypatch.setattr(ec2_infra, "_delete_api_endpoint", delete_endpoint)
    monkeypatch.setattr(ec2_infra.boto3, "client", lambda *args, **kwargs: client)

    first = get_api_scope(
        artifact_prefix="repairs/run-001",
        campaign_sha256=CAMPAIGN_SHA256,
        stage="rankings",
    )
    second = get_api_scope(
        artifact_prefix="repairs/run-002",
        campaign_sha256=second_campaign_sha256,
        stage="rankings",
    )

    assert first is not None
    assert first.api_url == API_URL
    assert first.public_api_url == "http://203.0.113.10:8000"
    assert second is not None
    assert second.api_url == (
        f"http://api.rankings.{second_campaign_sha256[:32]}."
        f"{second_campaign_sha256[32:]}.citrees.internal:8000"
    )
    assert second.public_api_url == "http://203.0.113.20:8000"
    assert (
        terminate_api(
            artifact_prefix="repairs/run-001",
            campaign_sha256=CAMPAIGN_SHA256,
            stage="rankings",
        )
        == "i-first"
    )
    client.terminate_instances.assert_called_once_with(InstanceIds=["i-first"])
    delete_endpoint.assert_called_once_with(
        client,
        client,
        artifact_prefix="repairs/run-001",
        campaign_sha256=CAMPAIGN_SHA256,
        stage="rankings",
        region="us-east-1",
    )


def _mock_api_status_client(
    monkeypatch: pytest.MonkeyPatch,
    status: dict[str, object],
) -> MagicMock:
    """Install one deterministic API status response."""
    response = MagicMock()
    response.json.return_value = status
    client = MagicMock()
    client.__enter__.return_value = client
    client.get.return_value = response
    monkeypatch.setattr(ec2_infra.httpx, "Client", MagicMock(return_value=client))
    return client


def test_api_readiness_accepts_exact_runtime_contract_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _mock_api_status_client(
        monkeypatch,
        {
            "artifact_prefix": "repairs/run-001",
            "campaign_sha256": CAMPAIGN_SHA256,
            "canonical_manifest_s3_key": CANONICAL_MANIFEST_KEY,
            "canonical_manifest_sha256": CANONICAL_MANIFEST_SHA256,
            "gate_receipt_s3_key": GATE_RECEIPT_KEY,
            "gate_receipt_sha256": GATE_RECEIPT_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "max_cell_attempts": 3,
            "runtime_contract_s3_key": RUNTIME_CONTRACT_KEY,
            "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
            "stage": "rankings",
        },
    )

    ec2_infra._wait_for_api_ready(
        "http://203.0.113.10:8000",
        artifact_prefix="repairs/run-001",
        campaign_sha256=CAMPAIGN_SHA256,
        canonical_manifest_s3_key=CANONICAL_MANIFEST_KEY,
        canonical_manifest_sha256=CANONICAL_MANIFEST_SHA256,
        gate_receipt_s3_key=GATE_RECEIPT_KEY,
        gate_receipt_sha256=GATE_RECEIPT_SHA256,
        manifest_sha256=MANIFEST_SHA256,
        max_cell_attempts=3,
        runtime_contract_s3_key=RUNTIME_CONTRACT_KEY,
        runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
        stage="rankings",
        timeout_seconds=1.0,
        poll_interval=0.0,
    )

    client.get.assert_called_once_with("/status")


@pytest.mark.parametrize(
    "field",
    [
        "canonical_manifest_s3_key",
        "canonical_manifest_sha256",
        "gate_receipt_s3_key",
        "gate_receipt_sha256",
        "runtime_contract_s3_key",
        "runtime_contract_sha256",
    ],
)
def test_api_readiness_rejects_runtime_contract_scope_mismatch(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status: dict[str, object] = {
        "artifact_prefix": "repairs/run-001",
        "campaign_sha256": CAMPAIGN_SHA256,
        "canonical_manifest_s3_key": CANONICAL_MANIFEST_KEY,
        "canonical_manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "gate_receipt_s3_key": GATE_RECEIPT_KEY,
        "gate_receipt_sha256": GATE_RECEIPT_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "max_cell_attempts": 3,
        "runtime_contract_s3_key": RUNTIME_CONTRACT_KEY,
        "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
        "stage": "rankings",
    }
    status[field] = "different"
    _mock_api_status_client(monkeypatch, status)
    monotonic_values = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(ec2_infra.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(ec2_infra.time, "sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="scope mismatch"):
        ec2_infra._wait_for_api_ready(
            "http://203.0.113.10:8000",
            artifact_prefix="repairs/run-001",
            campaign_sha256=CAMPAIGN_SHA256,
            canonical_manifest_s3_key=CANONICAL_MANIFEST_KEY,
            canonical_manifest_sha256=CANONICAL_MANIFEST_SHA256,
            gate_receipt_s3_key=GATE_RECEIPT_KEY,
            gate_receipt_sha256=GATE_RECEIPT_SHA256,
            manifest_sha256=MANIFEST_SHA256,
            max_cell_attempts=3,
            runtime_contract_s3_key=RUNTIME_CONTRACT_KEY,
            runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
            stage="rankings",
            timeout_seconds=1.0,
            poll_interval=0.0,
        )


@pytest.mark.parametrize(
    "issue",
    ["missing-sha256", "missing-key", "mismatched-sha256", "mismatched-key"],
)
def test_api_launch_rejects_unattested_runtime_contract_before_ec2(
    issue: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication: dict[str, str | int] = {
        "key": MANIFEST_KEY,
        "sha256": MANIFEST_SHA256,
        "campaign_sha256": CAMPAIGN_SHA256,
        "canonical_manifest_s3_key": CANONICAL_MANIFEST_KEY,
        "canonical_manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "gate_receipt_sha256": GATE_RECEIPT_SHA256,
        "gate_receipt_s3_key": GATE_RECEIPT_KEY,
        "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
        "runtime_contract_s3_key": RUNTIME_CONTRACT_KEY,
        "cells": 1,
    }
    if issue == "missing-sha256":
        publication.pop("runtime_contract_sha256")
    elif issue == "missing-key":
        publication.pop("runtime_contract_s3_key")
    elif issue == "mismatched-sha256":
        publication["runtime_contract_sha256"] = "d" * 64
    else:
        publication["runtime_contract_s3_key"] = "runtime-contracts/different.json"

    ec2_client = MagicMock()
    validate_image_revision = MagicMock(return_value="a" * 40)
    monkeypatch.setattr(ec2_infra, "get_api_scope", lambda **kwargs: None)
    monkeypatch.setattr(
        ec2_infra,
        "publish_rerun_manifest",
        lambda manifest_path,
        canonical_manifest_path,
        runtime_contract_path,
        gate_receipt_path,
        *,
        region: (publication),
    )
    monkeypatch.setattr(ec2_infra, "validate_image_revision", validate_image_revision)
    monkeypatch.setattr(ec2_infra.boto3, "client", MagicMock(return_value=ec2_client))

    with pytest.raises(RuntimeError, match="runtime contract"):
        launch_api(
            instance_type="m5.large",
            image_uri=DIGEST_URI,
            artifact_prefix="repairs/run-001",
            launch_id="api-invalid-runtime",
            canonical_manifest_path=_write_canonical_manifest(tmp_path),
            gate_receipt_path=_write_gate_receipt(tmp_path),
            manifest_path=tmp_path / "manifest.csv",
            runtime_contract_path=_write_runtime_contract(tmp_path),
            stage="rankings",
            lease_seconds=900,
            max_cell_attempts=3,
        )

    validate_image_revision.assert_not_called()
    ec2_client.run_instances.assert_not_called()


@pytest.mark.parametrize(
    "mismatch",
    ["runtime_contract_sha256", "runtime_contract_s3_key"],
)
def test_worker_launch_rejects_runtime_contract_scope_mismatch_before_ec2(
    mismatch: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("fixture", encoding="utf-8")
    canonical_manifest_path = _write_canonical_manifest(tmp_path)
    runtime_contract_path = _write_runtime_contract(tmp_path)
    gate_receipt_path = _write_gate_receipt(tmp_path)
    monkeypatch.setattr(
        ec2_infra,
        "publish_rerun_manifest",
        lambda manifest_path,
        canonical_manifest_path,
        runtime_contract_path,
        gate_receipt_path,
        *,
        region: (_published_manifest()),
    )
    monkeypatch.setattr(
        ec2_infra,
        "get_api_scope",
        lambda **kwargs: _api_scope(
            runtime_contract_s3_key=(
                "runtime-contracts/different.json"
                if mismatch == "runtime_contract_s3_key"
                else RUNTIME_CONTRACT_KEY
            ),
            runtime_contract_sha256=(
                "d" * 64 if mismatch == "runtime_contract_sha256" else RUNTIME_CONTRACT_SHA256
            ),
        ),
    )
    validate_image_revision = MagicMock(return_value="a" * 40)
    ec2_client = MagicMock()
    monkeypatch.setattr(ec2_infra, "validate_image_revision", validate_image_revision)
    monkeypatch.setattr(ec2_infra.boto3, "client", MagicMock(return_value=ec2_client))

    with pytest.raises(RuntimeError, match="does not match running API"):
        launch_workers(
            n=1,
            image_uri=DIGEST_URI,
            artifact_prefix="repairs/run-001",
            canonical_manifest_path=canonical_manifest_path,
            gate_receipt_path=gate_receipt_path,
            launch_id="runtime-mismatch",
            manifest_path=manifest_path,
            runtime_contract_path=runtime_contract_path,
            stage="rankings",
        )

    validate_image_revision.assert_not_called()
    ec2_client.run_instances.assert_not_called()


def test_worker_launch_rejects_scope_mismatch_before_ec2_launch(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("fixture")
    canonical_manifest_path = _write_canonical_manifest(tmp_path)
    runtime_contract_path = _write_runtime_contract(tmp_path)
    gate_receipt_path = _write_gate_receipt(tmp_path)
    monkeypatch.setattr(
        ec2_infra,
        "publish_rerun_manifest",
        lambda manifest_path,
        canonical_manifest_path,
        runtime_contract_path,
        gate_receipt_path,
        *,
        region: (_published_manifest()),
    )
    monkeypatch.setattr(
        ec2_infra,
        "get_api_scope",
        lambda **kwargs: _api_scope(
            artifact_prefix="repairs/other-run",
        ),
    )
    monkeypatch.setattr(ec2_infra, "_wait_for_api_ready", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="does not match running API"):
        launch_workers(
            n=1,
            image_uri=DIGEST_URI,
            artifact_prefix="repairs/run-001",
            canonical_manifest_path=canonical_manifest_path,
            gate_receipt_path=gate_receipt_path,
            launch_id="scope-mismatch",
            manifest_path=manifest_path,
            runtime_contract_path=runtime_contract_path,
            stage="rankings",
        )


def test_worker_launch_refreshes_ingress_before_api_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("fixture", encoding="utf-8")
    canonical_manifest_path = _write_canonical_manifest(tmp_path)
    runtime_contract_path = _write_runtime_contract(tmp_path)
    gate_receipt_path = _write_gate_receipt(tmp_path)
    events: list[str] = []
    client = MagicMock()
    s3 = _MemoryS3()
    _configure_worker_subnets(client)
    client.run_instances.return_value = {"Instances": [_worker_instance("i-worker")]}
    client.describe_instances.return_value = {"Reservations": []}
    monkeypatch.setattr(
        ec2_infra,
        "publish_rerun_manifest",
        lambda manifest_path,
        canonical_manifest_path,
        runtime_contract_path,
        gate_receipt_path,
        *,
        region: (_published_manifest()),
    )
    monkeypatch.setattr(
        ec2_infra,
        "get_api_scope",
        lambda **kwargs: _api_scope(),
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
        image_uri=DIGEST_URI,
        artifact_prefix="repairs/run-001",
        canonical_manifest_path=canonical_manifest_path,
        gate_receipt_path=gate_receipt_path,
        launch_id="ingress-refresh",
        manifest_path=manifest_path,
        runtime_contract_path=runtime_contract_path,
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
    canonical_manifest_path = _write_canonical_manifest(tmp_path)
    runtime_contract_path = _write_runtime_contract(tmp_path)
    gate_receipt_path = _write_gate_receipt(tmp_path)
    events: list[str] = []
    s3 = _MemoryS3(events)
    ec2 = MagicMock()
    _configure_worker_subnets(ec2)
    ec2.describe_instances.return_value = {"Reservations": []}

    def run_instances(**kwargs: Any) -> dict[str, object]:
        slot = len(ec2.run_instances.call_args_list)
        events.append(f"ec2:{slot}")
        return {"Instances": [_worker_instance(f"i-worker-{slot}")]}

    ec2.run_instances.side_effect = run_instances
    monkeypatch.setattr(
        ec2_infra,
        "publish_rerun_manifest",
        lambda manifest_path,
        canonical_manifest_path,
        runtime_contract_path,
        gate_receipt_path,
        *,
        region: (_published_manifest()),
    )
    api_scopes = MagicMock(
        side_effect=[
            _api_scope(public_api_url="http://203.0.113.10:8000"),
            _api_scope(public_api_url="http://203.0.113.20:8000"),
        ]
    )
    monkeypatch.setattr(ec2_infra, "get_api_scope", api_scopes)
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
        "image_uri": DIGEST_URI,
        "artifact_prefix": "repairs/run-001",
        "canonical_manifest_path": canonical_manifest_path,
        "gate_receipt_path": gate_receipt_path,
        "manifest_path": manifest_path,
        "runtime_contract_path": runtime_contract_path,
        "stage": "rankings",
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
        assert tags["citrees-market"] == "on-demand"
        assert tags["citrees-gate-receipt-key"] == GATE_RECEIPT_KEY
        assert tags["citrees-gate-receipt-sha256"] == GATE_RECEIPT_SHA256
        assert tags["citrees-runtime-contract-key"] == RUNTIME_CONTRACT_KEY
        assert tags["citrees-runtime-contract-sha256"] == RUNTIME_CONTRACT_SHA256
        assert tags["citrees-worker-launch-id"] == "scale-001"
    assert len(client_tokens) == 2

    intent_key = "repairs/run-001/_control/worker-launches/scale-001/intent.json"
    intent = json.loads(s3.objects[intent_key][0])
    assert intent["instance_family"] == "c6a"
    assert intent["market"] == "on-demand"
    assert intent["schema_version"] == 8
    assert intent["excluded_availability_zones"] == []
    assert intent["subnet_ids"] == [WORKER_SUBNET_ID]
    assert intent["slot_placements"] == [
        {
            "availability_zone": WORKER_AVAILABILITY_ZONE,
            "slot": 1,
            "subnet_id": WORKER_SUBNET_ID,
            "vpc_id": WORKER_VPC_ID,
        },
        {
            "availability_zone": WORKER_AVAILABILITY_ZONE,
            "slot": 2,
            "subnet_id": WORKER_SUBNET_ID,
            "vpc_id": WORKER_VPC_ID,
        },
    ]
    assert intent["api_scope"]["api_url"] == API_URL
    assert intent["api_scope"]["market"] == "on-demand"
    assert "public_api_url" not in intent["api_scope"]
    assert intent["requested_instances"] == 2
    assert intent["gate_receipt_s3_key"] == GATE_RECEIPT_KEY
    assert intent["gate_receipt_sha256"] == GATE_RECEIPT_SHA256
    assert intent["api_scope"]["gate_receipt_s3_key"] == GATE_RECEIPT_KEY
    assert intent["api_scope"]["gate_receipt_sha256"] == GATE_RECEIPT_SHA256
    assert intent["runtime_contract_s3_key"] == RUNTIME_CONTRACT_KEY
    assert intent["runtime_contract_sha256"] == RUNTIME_CONTRACT_SHA256
    assert intent["api_scope"]["runtime_contract_s3_key"] == RUNTIME_CONTRACT_KEY
    assert intent["api_scope"]["runtime_contract_sha256"] == RUNTIME_CONTRACT_SHA256
    first_request = ec2.run_instances.call_args_list[0].kwargs
    assert "InstanceMarketOptions" not in first_request
    assert first_request["SubnetId"] == WORKER_SUBNET_ID
    user_data = base64.b64decode(first_request["UserData"]).decode("utf-8")
    assert f"-e CITREES_API_URL={API_URL}" in user_data
    assert f"--api-url {API_URL}" in user_data
    assert "10.0.0.10" not in user_data
    assert "10.0.0.20" not in user_data
    request_contract = intent["request_contract"]
    for key, value in first_request.items():
        if key not in {"ClientToken", "SubnetId", "TagSpecifications", "UserData"}:
            assert request_contract[key] == value
    assert (
        request_contract["UserDataSha256"]
        == hashlib.sha256(first_request["UserData"].encode()).hexdigest()
    )
    assert request_contract["MetadataOptions"] == {
        "HttpEndpoint": "enabled",
        "HttpPutResponseHopLimit": 2,
        "HttpTokens": "required",
    }
    outcome_keys = sorted(key for key in s3.objects if "/instances/" in key)
    assert len(outcome_keys) == 2
    assert {json.loads(s3.objects[key][0])["instance_id"] for key in outcome_keys} == {
        "i-worker-1",
        "i-worker-2",
    }
    assert {json.loads(s3.objects[key][0])["market"] for key in outcome_keys} == {"on-demand"}
    assert {json.loads(s3.objects[key][0])["instance_family"] for key in outcome_keys} == {"c6a"}
    assert {json.loads(s3.objects[key][0])["subnet_id"] for key in outcome_keys} == {
        WORKER_SUBNET_ID
    }
    assert {json.loads(s3.objects[key][0])["availability_zone"] for key in outcome_keys} == {
        WORKER_AVAILABILITY_ZONE
    }

    ec2.run_instances.reset_mock()
    events.clear()
    assert launch_workers(**launch_kwargs) == ["i-worker-1", "i-worker-2"]
    ec2.run_instances.assert_not_called()
    assert api_scopes.call_count == 2


def test_worker_launch_rejects_explicit_subnet_in_excluded_zone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, ec2, s3 = _mock_worker_launch_dependencies(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="excluded availability zone us-east-1a"):
        launch_workers(
            **_worker_launch_kwargs(manifest_path),
            excluded_availability_zones=(WORKER_AVAILABILITY_ZONE,),
            subnet_ids=(WORKER_SUBNET_ID,),
        )

    ec2.run_instances.assert_not_called()
    assert not any("/worker-launches/" in key for key in s3.objects)


def test_worker_launch_replay_rejects_changed_subnet_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, ec2, _s3 = _mock_worker_launch_dependencies(tmp_path, monkeypatch)
    second_subnet_id = "subnet-0fffffffffffffff1"
    rows = {
        WORKER_SUBNET_ID: {
            "AvailabilityZone": WORKER_AVAILABILITY_ZONE,
            "AvailableIpAddressCount": 100,
            "DefaultForAz": True,
            "MapPublicIpOnLaunch": True,
            "State": "available",
            "SubnetId": WORKER_SUBNET_ID,
            "VpcId": WORKER_VPC_ID,
        },
        second_subnet_id: {
            "AvailabilityZone": "us-east-1f",
            "AvailableIpAddressCount": 100,
            "DefaultForAz": True,
            "MapPublicIpOnLaunch": True,
            "State": "available",
            "SubnetId": second_subnet_id,
            "VpcId": WORKER_VPC_ID,
        },
    }
    ec2.describe_instance_type_offerings.return_value = {
        "InstanceTypeOfferings": [
            {"Location": WORKER_AVAILABILITY_ZONE},
            {"Location": "us-east-1f"},
        ]
    }
    ec2.describe_subnets.side_effect = lambda **kwargs: {
        "Subnets": [rows[subnet_id] for subnet_id in kwargs["SubnetIds"]]
    }
    ec2.run_instances.return_value = {"Instances": [_worker_instance("i-worker")]}
    kwargs = _worker_launch_kwargs(manifest_path)

    assert launch_workers(**kwargs, subnet_ids=(WORKER_SUBNET_ID,)) == ["i-worker"]
    ec2.run_instances.reset_mock()

    with pytest.raises(RuntimeError, match="different exact launch contract"):
        launch_workers(**kwargs, subnet_ids=(second_subnet_id,))

    ec2.run_instances.assert_not_called()


def test_worker_launch_rejects_wrong_ec2_placement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, ec2, s3 = _mock_worker_launch_dependencies(tmp_path, monkeypatch)
    ec2.run_instances.return_value = {
        "Instances": [
            _worker_instance(
                "i-worker",
                availability_zone="us-east-1f",
                subnet_id="subnet-0fffffffffffffff1",
            )
        ]
    }

    with pytest.raises(RuntimeError, match="expected 'subnet-0aaaaaaaaaaaaaaa1'"):
        launch_workers(
            **_worker_launch_kwargs(manifest_path),
            subnet_ids=(WORKER_SUBNET_ID,),
        )

    assert not any("/instances/" in key for key in s3.objects)


def test_worker_launch_rejects_unknown_explicit_subnet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, ec2, s3 = _mock_worker_launch_dependencies(tmp_path, monkeypatch)
    ec2.describe_subnets.return_value = {"Subnets": []}

    with pytest.raises(RuntimeError, match="did not return requested subnets"):
        launch_workers(
            **_worker_launch_kwargs(manifest_path),
            subnet_ids=("subnet-0fffffffffffffff1",),
        )

    ec2.run_instances.assert_not_called()
    assert not any("/worker-launches/" in key for key in s3.objects)


def _mock_worker_launch_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, MagicMock, _MemoryS3]:
    """Install deterministic benchmark worker launch dependencies."""
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("fixture", encoding="utf-8")
    _write_canonical_manifest(tmp_path)
    _write_runtime_contract(tmp_path)
    _write_gate_receipt(tmp_path)
    ec2 = MagicMock()
    _configure_worker_subnets(ec2)
    ec2.describe_instances.return_value = {"Reservations": []}
    s3 = _MemoryS3()
    monkeypatch.setattr(
        ec2_infra,
        "publish_rerun_manifest",
        lambda manifest_path,
        canonical_manifest_path,
        runtime_contract_path,
        gate_receipt_path,
        *,
        region: (_published_manifest()),
    )
    monkeypatch.setattr(
        ec2_infra,
        "get_api_scope",
        lambda **kwargs: _api_scope(),
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
) -> _WorkerLaunchKwargs:
    """Build one complete direct worker launch invocation."""
    return {
        "n": n,
        "image_uri": DIGEST_URI,
        "artifact_prefix": "repairs/run-001",
        "canonical_manifest_path": manifest_path.parent / "canonical-manifest.csv",
        "gate_receipt_path": manifest_path.parent / "gate-receipt.json",
        "launch_id": "scale-001",
        "manifest_path": manifest_path,
        "runtime_contract_path": manifest_path.parent / "runtime-contract.json",
        "stage": "rankings",
    }


def test_worker_launch_recovers_ambiguous_timeout_with_same_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, ec2, _s3 = _mock_worker_launch_dependencies(tmp_path, monkeypatch)
    ec2.run_instances.side_effect = [
        ReadTimeoutError(endpoint_url="https://ec2.us-east-1.amazonaws.com"),
        {"Instances": [_worker_instance("i-recovered")]},
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
        {"Instances": [_worker_instance("i-recovered")]},
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
                            **_worker_instance("i-tag-recovered"),
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
        {"Instances": [_worker_instance("i-worker-1")]},
        capacity_error,
    ]
    kwargs = _worker_launch_kwargs(manifest_path, n=3)

    assert launch_workers(**kwargs) == ["i-worker-1"]
    failed_slot_token = ec2.run_instances.call_args_list[1].kwargs["ClientToken"]

    ec2.run_instances.reset_mock(side_effect=True)
    ec2.run_instances.side_effect = [
        {"Instances": [_worker_instance("i-worker-2")]},
        {"Instances": [_worker_instance("i-worker-3")]},
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
    ec2.run_instances.return_value = {"Instances": [_worker_instance("i-worker")]}
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
                        **_worker_instance("i-worker"),
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


def test_worker_launch_rejects_market_mismatch_before_ec2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, ec2, _s3 = _mock_worker_launch_dependencies(tmp_path, monkeypatch)
    ec2.run_instances.return_value = {"Instances": [_worker_instance("i-worker")]}
    monkeypatch.setattr(
        ec2_infra,
        "get_api_scope",
        lambda **kwargs: _api_scope(market="spot"),
    )

    with pytest.raises(RuntimeError, match="does not match running API"):
        launch_workers(**_worker_launch_kwargs(manifest_path))

    ec2.run_instances.assert_not_called()


def test_worker_launch_replay_requires_same_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, ec2, _s3 = _mock_worker_launch_dependencies(tmp_path, monkeypatch)
    ec2.run_instances.return_value = {"Instances": [_worker_instance("i-worker")]}
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
        {"Instances": [_worker_instance("i-worker-1")]},
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
    ec2.run_instances.return_value = {"Instances": [_worker_instance("i-worker-2")]}
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
            image_uri=DIGEST_URI,
            artifact_prefix="repairs/run-001",
            canonical_manifest_path=Path("unused-canonical-manifest.csv"),
            gate_receipt_path=Path("unused-gate-receipt.json"),
            launch_id=launch_id,
            manifest_path=Path("unused.csv"),
            runtime_contract_path=Path("unused-runtime-contract.json"),
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


@pytest.mark.parametrize(
    "response",
    [
        {},
        {
            "Account": "not-an-account",
            "Arn": "arn:aws:iam::123456789012:root",
            "UserId": "fixture",
        },
        {
            "Account": "123456789012",
            "Arn": "arn:aws:iam::210987654321:root",
            "UserId": "fixture",
        },
        {
            "Account": "123456789012",
            "Arn": "arn:aws:s3:::fixture",
            "UserId": "fixture",
        },
        {
            "Account": "123456789012",
            "Arn": "arn:aws:sts::123456789012:assumed-role/role/session",
            "UserId": "",
        },
    ],
)
def test_live_sts_identity_rejects_malformed_or_cross_account_response(
    response: dict[str, str],
) -> None:
    client = MagicMock()
    client.get_caller_identity.return_value = response

    with pytest.raises(RuntimeError, match="STS caller identity"):
        aws_infra.get_aws_caller_identity(client=client)


@pytest.mark.parametrize(
    "arn",
    [
        "arn:aws:iam::123456789012:root",
        "arn:aws:sts::123456789012:assumed-role/citrees/worker",
    ],
)
def test_live_sts_identity_accepts_exact_account_binding(arn: str) -> None:
    client = MagicMock()
    client.get_caller_identity.return_value = {
        "Account": "123456789012",
        "Arn": arn,
        "UserId": "AROAEXAMPLE:worker",
        "ResponseMetadata": {"RequestId": "ignored"},
    }

    assert aws_infra.get_aws_caller_identity(client=client) == {
        "Account": "123456789012",
        "Arn": arn,
        "UserId": "AROAEXAMPLE:worker",
    }


def test_campaign_role_can_write_only_its_exact_output_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_prefix = "repairs/run-001"
    profile_name = aws_infra.campaign_instance_profile_name(
        output_prefix=output_prefix,
        campaign_sha256=CAMPAIGN_SHA256,
        read_keys=(MANIFEST_KEY, RUNTIME_CONTRACT_KEY),
        write_prefixes=(output_prefix,),
    )
    client = _campaign_iam_client(profile_name)
    _patch_campaign_iam(monkeypatch, client)

    assert (
        aws_infra.ensure_campaign_iam_profile(
            output_prefix=output_prefix,
            campaign_sha256=CAMPAIGN_SHA256,
            read_keys=(MANIFEST_KEY, RUNTIME_CONTRACT_KEY),
            write_prefixes=(output_prefix,),
        )
        == profile_name
    )
    assert profile_name != aws_infra.campaign_instance_profile_name(
        output_prefix="repairs/run-002",
        campaign_sha256=CAMPAIGN_SHA256,
        read_keys=(MANIFEST_KEY, RUNTIME_CONTRACT_KEY),
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
        "repairs/run-001/*",
    ]
    assert statements["S3ReadInputsAndArtifacts"]["Resource"] == [
        "arn:aws:s3:::citrees-123456789012/data/*",
        f"arn:aws:s3:::citrees-123456789012/{MANIFEST_KEY}",
        f"arn:aws:s3:::citrees-123456789012/{RUNTIME_CONTRACT_KEY}",
        "arn:aws:s3:::citrees-123456789012/repairs/run-001/*",
    ]
    assert profile_name != aws_infra.campaign_instance_profile_name(
        output_prefix=output_prefix,
        campaign_sha256=CAMPAIGN_SHA256,
        read_keys=(MANIFEST_KEY,),
        write_prefixes=(output_prefix,),
    )
    assert statements["ECRPull"]["Resource"] == (
        "arn:aws:ecr:us-east-1:123456789012:repository/citrees-123456789012"
    )


def test_campaign_role_requires_canonical_exact_read_keys() -> None:
    arguments = {
        "output_prefix": "repairs/run-001",
        "campaign_sha256": CAMPAIGN_SHA256,
        "write_prefixes": ("repairs/run-001",),
    }
    expected = aws_infra.campaign_instance_profile_name(
        **arguments,
        read_keys=(MANIFEST_KEY, RUNTIME_CONTRACT_KEY),
    )
    assert expected == aws_infra.campaign_instance_profile_name(
        **arguments,
        read_keys=(RUNTIME_CONTRACT_KEY, MANIFEST_KEY),
    )

    with pytest.raises(TypeError, match="sequence of strings"):
        aws_infra.campaign_instance_profile_name(
            **arguments,
            read_keys=MANIFEST_KEY,
        )
    for invalid_keys in (
        ("rerun-manifests/*",),
        ("rerun-manifests/not-a-digest.csv",),
        (f"rerun-manifests/{MANIFEST_SHA256}.json",),
        ("runtime-contracts/contract?.json",),
        (f"runtime-contracts/{RUNTIME_CONTRACT_SHA256}.csv",),
        (f"runtime-gate-receipts/{GATE_RECEIPT_SHA256}.csv",),
        (f" {MANIFEST_KEY}",),
        (f"{MANIFEST_KEY}/",),
    ):
        with pytest.raises(ValueError):
            aws_infra.campaign_instance_profile_name(
                **arguments,
                read_keys=invalid_keys,
            )


def test_campaign_role_can_limit_writes_to_shard_subprefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_prefix = "jss/replication/source-a/campaign-b"
    write_prefix = f"{output_prefix}/shards"
    profile_name = aws_infra.campaign_instance_profile_name(
        output_prefix=output_prefix,
        campaign_sha256=CAMPAIGN_SHA256,
        read_keys=(f"{output_prefix}/campaign.json",),
        write_prefixes=(write_prefix,),
    )
    client = _campaign_iam_client(profile_name)
    _patch_campaign_iam(monkeypatch, client)

    assert (
        aws_infra.ensure_campaign_iam_profile(
            output_prefix=output_prefix,
            campaign_sha256=CAMPAIGN_SHA256,
            read_keys=(f"{output_prefix}/campaign.json",),
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
        read_keys=(f"{output_prefix}/campaign.json",),
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
        read_keys=(f"{output_prefix}/campaign.json",),
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
            read_keys=(f"{output_prefix}/campaign.json",),
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
        read_keys=(f"{output_prefix}/campaign.json",),
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
            read_keys=(f"{output_prefix}/campaign.json",),
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
        read_keys=(f"{output_prefix}/campaign.json",),
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
            read_keys=(f"{output_prefix}/campaign.json",),
            write_prefixes=(write_prefix,),
        )


def test_security_group_refreshes_existing_group_in_exact_default_vpc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.describe_vpcs.return_value = {"Vpcs": [{"VpcId": "vpc-default"}]}
    client.describe_security_groups.return_value = {
        "SecurityGroups": [
            {
                "GroupId": "sg-test",
                "VpcId": "vpc-default",
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
                    }
                ],
            }
        ]
    }
    monkeypatch.setattr(aws_infra, "get_public_ip", lambda: "203.0.113.20")
    monkeypatch.setattr(aws_infra.boto3, "client", lambda *args, **kwargs: client)

    assert aws_infra.ensure_security_group() == "sg-test"

    client.describe_vpcs.assert_called_once_with(
        Filters=[{"Name": "is-default", "Values": ["true"]}]
    )
    client.describe_security_groups.assert_called_once_with(
        Filters=[
            {"Name": "group-name", "Values": ["citrees-sg"]},
            {"Name": "vpc-id", "Values": ["vpc-default"]},
        ]
    )
    client.create_security_group.assert_not_called()
    client.revoke_security_group_ingress.assert_called_once_with(
        GroupId="sg-test",
        IpPermissions=[
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
        ],
    )
    client.authorize_security_group_ingress.assert_called_once_with(
        GroupId="sg-test",
        IpPermissions=[
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
            }
        ],
    )


def test_security_group_creation_is_bound_to_default_vpc_without_ssh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.describe_vpcs.return_value = {"Vpcs": [{"VpcId": "vpc-default"}]}
    client.describe_security_groups.return_value = {"SecurityGroups": []}
    client.create_security_group.return_value = {"GroupId": "sg-created"}
    monkeypatch.setattr(aws_infra, "get_public_ip", lambda: "203.0.113.20")
    monkeypatch.setattr(aws_infra.boto3, "client", lambda *args, **kwargs: client)

    assert aws_infra.ensure_security_group() == "sg-created"

    client.create_security_group.assert_called_once_with(
        GroupName="citrees-sg",
        Description="citrees API + worker instances",
        VpcId="vpc-default",
    )
    client.authorize_security_group_ingress.assert_called_once_with(
        GroupId="sg-created",
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 8000,
                "ToPort": 8000,
                "UserIdGroupPairs": [
                    {
                        "GroupId": "sg-created",
                        "Description": "API from citrees instances",
                    }
                ],
            },
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
        ],
    )
    assert all(
        permission.get("FromPort") != 22 and permission.get("ToPort") != 22
        for permission in client.authorize_security_group_ingress.call_args.kwargs["IpPermissions"]
    )


def test_security_group_refresh_repairs_missing_self_ingress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.describe_vpcs.return_value = {"Vpcs": [{"VpcId": "vpc-default"}]}
    client.describe_security_groups.return_value = {
        "SecurityGroups": [
            {
                "GroupId": "sg-test",
                "VpcId": "vpc-default",
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
                    }
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


def test_security_group_rejects_wrong_vpc_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.describe_vpcs.return_value = {"Vpcs": [{"VpcId": "vpc-default"}]}
    client.describe_security_groups.return_value = {
        "SecurityGroups": [
            {
                "GroupId": "sg-wrong-vpc",
                "VpcId": "vpc-other",
                "IpPermissions": [],
            }
        ]
    }
    monkeypatch.setattr(aws_infra, "get_public_ip", lambda: "203.0.113.20")
    monkeypatch.setattr(aws_infra.boto3, "client", lambda *args, **kwargs: client)

    with pytest.raises(RuntimeError, match="outside default VPC vpc-default"):
        aws_infra.ensure_security_group()

    client.create_security_group.assert_not_called()
    client.authorize_security_group_ingress.assert_not_called()
    client.revoke_security_group_ingress.assert_not_called()


@pytest.mark.parametrize(
    "vpcs",
    [
        [],
        [{"VpcId": "vpc-default-a"}, {"VpcId": "vpc-default-b"}],
    ],
)
def test_security_group_requires_exactly_one_default_vpc(
    monkeypatch: pytest.MonkeyPatch,
    vpcs: list[dict[str, str]],
) -> None:
    client = MagicMock()
    client.describe_vpcs.return_value = {"Vpcs": vpcs}
    client.create_security_group.return_value = {"GroupId": "sg-created"}
    monkeypatch.setattr(aws_infra, "get_public_ip", lambda: "203.0.113.20")
    monkeypatch.setattr(aws_infra.boto3, "client", lambda *args, **kwargs: client)

    with pytest.raises(RuntimeError, match="exactly one default VPC"):
        aws_infra.ensure_security_group()

    client.describe_security_groups.assert_not_called()
    client.create_security_group.assert_not_called()


def test_security_group_rejects_ambiguous_exact_vpc_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.describe_vpcs.return_value = {"Vpcs": [{"VpcId": "vpc-default"}]}
    client.describe_security_groups.return_value = {
        "SecurityGroups": [
            {"GroupId": "sg-a", "VpcId": "vpc-default", "IpPermissions": []},
            {"GroupId": "sg-b", "VpcId": "vpc-default", "IpPermissions": []},
        ]
    }
    monkeypatch.setattr(aws_infra, "get_public_ip", lambda: "203.0.113.20")
    monkeypatch.setattr(aws_infra.boto3, "client", lambda *args, **kwargs: client)

    with pytest.raises(RuntimeError, match="multiple citrees security groups"):
        aws_infra.ensure_security_group()

    client.create_security_group.assert_not_called()
    client.authorize_security_group_ingress.assert_not_called()


def test_security_group_revokes_every_stale_ssh_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ssh_permissions = [
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
        {
            "IpProtocol": "tcp",
            "FromPort": 20,
            "ToPort": 25,
            "Ipv6Ranges": [{"CidrIpv6": "2001:db8::/64"}],
        },
        {
            "IpProtocol": "-1",
            "UserIdGroupPairs": [{"GroupId": "sg-peer"}],
        },
    ]
    client = MagicMock()
    client.describe_vpcs.return_value = {"Vpcs": [{"VpcId": "vpc-default"}]}
    client.describe_security_groups.return_value = {
        "SecurityGroups": [
            {
                "GroupId": "sg-test",
                "VpcId": "vpc-default",
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
                        "UserIdGroupPairs": [
                            {
                                "GroupId": "sg-test",
                                "Description": "API from citrees instances",
                            }
                        ],
                    },
                    *ssh_permissions,
                ],
            }
        ]
    }
    monkeypatch.setattr(aws_infra, "get_public_ip", lambda: "203.0.113.20")
    monkeypatch.setattr(aws_infra.boto3, "client", lambda *args, **kwargs: client)

    assert aws_infra.ensure_security_group() == "sg-test"

    client.revoke_security_group_ingress.assert_called_once_with(
        GroupId="sg-test",
        IpPermissions=ssh_permissions,
    )
    client.authorize_security_group_ingress.assert_not_called()


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
    payload = b"account manifest"
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_bytes(payload)
    canonical_payload = b"canonical manifest"
    canonical_manifest_path = tmp_path / "canonical-manifest.csv"
    canonical_manifest_path.write_bytes(canonical_payload)
    runtime_contract_path = _write_runtime_contract(tmp_path)
    gate_receipt_path = _write_gate_receipt(tmp_path)
    manifest = SimpleNamespace(
        sha256=MANIFEST_SHA256,
        campaign_sha256=CAMPAIGN_SHA256,
        runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
        cells=(object(),),
        account_ids=("123456789012",),
    )
    canonical = SimpleNamespace(
        sha256=CANONICAL_MANIFEST_SHA256,
        campaign_sha256=CAMPAIGN_SHA256,
        runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
        cells=(object(), object()),
        account_ids=("123456789012", "210987654321"),
    )
    client = _MemoryS3()
    parse_manifest = MagicMock(
        side_effect=lambda body, expected_sha256=None: (
            canonical if body == canonical_payload else manifest
        )
    )
    monkeypatch.setattr(
        "paper.benchmark.pipeline.manifest.parse_rerun_manifest",
        parse_manifest,
    )
    validate_canonical = MagicMock()
    monkeypatch.setattr(
        "paper.benchmark.pipeline.manifest.validate_canonical_campaign",
        validate_canonical,
    )
    verify_shard = MagicMock(return_value=manifest)
    monkeypatch.setattr(
        "paper.benchmark.pipeline.manifest.verify_account_manifest_shard",
        verify_shard,
    )
    parse_gate_receipt = MagicMock(
        return_value={
            "account_manifest_sha256": {
                "123456789012": MANIFEST_SHA256,
                "210987654321": "d" * 64,
            },
            "report": {"status": "GO"},
        }
    )
    monkeypatch.setattr(
        "paper.benchmark.experiments.r_cforest_reproducibility.parse_gate_receipt",
        parse_gate_receipt,
    )
    monkeypatch.setattr(aws_infra, "ensure_s3_bucket", lambda region: "citrees-test")
    monkeypatch.setattr(aws_infra, "get_aws_account_id", lambda: "123456789012")
    monkeypatch.setattr(aws_infra.boto3, "client", lambda *args, **kwargs: client)

    result = aws_infra.publish_rerun_manifest(
        manifest_path,
        canonical_manifest_path,
        runtime_contract_path,
        gate_receipt_path,
    )

    assert result == {
        "bucket": "citrees-test",
        "key": MANIFEST_KEY,
        "sha256": MANIFEST_SHA256,
        "campaign_sha256": CAMPAIGN_SHA256,
        "canonical_manifest_s3_key": CANONICAL_MANIFEST_KEY,
        "canonical_manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "gate_receipt_s3_key": GATE_RECEIPT_KEY,
        "gate_receipt_sha256": GATE_RECEIPT_SHA256,
        "runtime_contract_s3_key": RUNTIME_CONTRACT_KEY,
        "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
        "cells": 1,
        "canonical_cells": 2,
    }
    manifest_payload, manifest_metadata = client.objects[MANIFEST_KEY]
    assert manifest_payload == payload
    assert manifest_metadata["target-aws-account-id"] == "123456789012"
    assert manifest_metadata["campaign-sha256"] == CAMPAIGN_SHA256
    assert manifest_metadata["canonical-manifest-sha256"] == CANONICAL_MANIFEST_SHA256
    published_canonical, canonical_metadata = client.objects[CANONICAL_MANIFEST_KEY]
    assert published_canonical == canonical_payload
    assert canonical_metadata["target-aws-account-ids"] == ("123456789012,210987654321")
    runtime_payload, runtime_metadata = client.objects[RUNTIME_CONTRACT_KEY]
    assert runtime_payload == runtime_contract_path.read_bytes()
    assert runtime_metadata["sha256"] == RUNTIME_CONTRACT_SHA256
    gate_payload, gate_metadata = client.objects[GATE_RECEIPT_KEY]
    assert gate_payload == GATE_RECEIPT_PAYLOAD
    assert gate_metadata == {
        "campaign-sha256": CAMPAIGN_SHA256,
        "manifest-sha256": CANONICAL_MANIFEST_SHA256,
        "profile": GATE_RECEIPT_PROFILE,
        "runtime-contract-sha256": RUNTIME_CONTRACT_SHA256,
        "schema-version": str(GATE_RECEIPT_SCHEMA_VERSION),
        "sha256": GATE_RECEIPT_SHA256,
        "status": "GO",
    }
    assert parse_gate_receipt.call_count == 2
    verify_shard.assert_called()


def test_one_account_canonical_and_shard_metadata_do_not_collide() -> None:
    payload = b"one-account manifest"
    digest = hashlib.sha256(payload).hexdigest()
    canonical_key = canonical_manifest_s3_key(digest)
    shard_key = manifest_s3_key(digest)
    client = _MemoryS3()

    aws_infra._publish_immutable_bytes(
        client,
        bucket="citrees-test",
        key=canonical_key,
        payload=payload,
        content_type="text/csv",
        metadata={"profile": "canonical-campaign", "sha256": digest},
    )
    aws_infra._publish_immutable_bytes(
        client,
        bucket="citrees-test",
        key=shard_key,
        payload=payload,
        content_type="text/csv",
        metadata={"target-aws-account-id": "123456789012", "sha256": digest},
    )

    assert canonical_key != shard_key
    assert client.objects[canonical_key][0] == client.objects[shard_key][0] == payload
    assert client.objects[canonical_key][1]["profile"] == "canonical-campaign"
    assert client.objects[shard_key][1]["target-aws-account-id"] == "123456789012"


def test_worker_listing_and_termination_require_one_exact_campaign_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch_id = "campaign-rankings-001"
    artifact_prefix = "repairs/run-001"
    ec2 = MagicMock()
    ec2.describe_instances.return_value = {
        "Reservations": [
            {
                "Instances": [
                    {
                        "InstanceId": "i-0123456789abcdef0",
                        "InstanceType": "c6a.8xlarge",
                        "LaunchTime": None,
                        "State": {"Name": "running"},
                        "Tags": [
                            {"Key": ec2_infra.TAG_KEY, "Value": ec2_infra.WORKER_TAG_VALUE},
                            {
                                "Key": "citrees-artifact-prefix",
                                "Value": artifact_prefix,
                            },
                            {
                                "Key": "citrees-campaign-sha256",
                                "Value": CAMPAIGN_SHA256,
                            },
                            {"Key": "citrees-stage", "Value": "rankings"},
                            {"Key": "citrees-worker-launch-id", "Value": launch_id},
                            {"Key": "citrees-market", "Value": "on-demand"},
                        ],
                    }
                ]
            }
        ]
    }
    monkeypatch.setattr(ec2_infra.boto3, "client", lambda *args, **kwargs: ec2)

    scope = {
        "artifact_prefix": artifact_prefix,
        "campaign_sha256": CAMPAIGN_SHA256,
        "stage": "rankings",
    }
    workers = ec2_infra.list_workers(launch_id, **scope)
    terminated = ec2_infra.terminate_workers(launch_id, **scope)

    expected_filters = [
        {"Name": f"tag:{ec2_infra.TAG_KEY}", "Values": [ec2_infra.WORKER_TAG_VALUE]},
        {"Name": "tag:citrees-artifact-prefix", "Values": [artifact_prefix]},
        {
            "Name": "tag:citrees-campaign-sha256",
            "Values": [CAMPAIGN_SHA256],
        },
        {"Name": "tag:citrees-stage", "Values": ["rankings"]},
        {"Name": "tag:citrees-worker-launch-id", "Values": [launch_id]},
        {
            "Name": "instance-state-name",
            "Values": ["pending", "running", "stopping"],
        },
    ]
    assert workers == [
        {
            "instance_id": "i-0123456789abcdef0",
            "state": "running",
            "instance_type": "c6a.8xlarge",
            "launch_time": "",
            "launch_id": launch_id,
        }
    ]
    assert terminated == ["i-0123456789abcdef0"]
    assert ec2.describe_instances.call_count == 2
    assert all(
        call.kwargs == {"Filters": expected_filters}
        for call in ec2.describe_instances.call_args_list
    )
    ec2.terminate_instances.assert_called_once_with(InstanceIds=["i-0123456789abcdef0"])


@pytest.mark.parametrize(
    ("tag_name", "tag_value"),
    [
        ("citrees-artifact-prefix", "repairs/other-run"),
        ("citrees-campaign-sha256", "f" * 64),
        ("citrees-stage", "metrics"),
        ("citrees-worker-launch-id", "different-launch"),
    ],
)
def test_worker_listing_rejects_ec2_rows_outside_campaign_launch_identity(
    tag_name: str,
    tag_value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tags = {
        ec2_infra.TAG_KEY: ec2_infra.WORKER_TAG_VALUE,
        "citrees-artifact-prefix": "repairs/run-001",
        "citrees-campaign-sha256": CAMPAIGN_SHA256,
        "citrees-stage": "rankings",
        "citrees-worker-launch-id": "campaign-rankings-001",
        "citrees-market": "on-demand",
    }
    tags[tag_name] = tag_value
    ec2 = MagicMock()
    ec2.describe_instances.return_value = {
        "Reservations": [
            {
                "Instances": [
                    {
                        "InstanceId": "i-0123456789abcdef0",
                        "InstanceType": "c6a.8xlarge",
                        "LaunchTime": None,
                        "State": {"Name": "running"},
                        "Tags": [{"Key": key, "Value": value} for key, value in tags.items()],
                    }
                ]
            }
        ]
    }
    monkeypatch.setattr(ec2_infra.boto3, "client", lambda *args, **kwargs: ec2)

    with pytest.raises(RuntimeError, match="outside the exact campaign launch identity"):
        ec2_infra.list_workers(
            "campaign-rankings-001",
            artifact_prefix="repairs/run-001",
            campaign_sha256=CAMPAIGN_SHA256,
            stage="rankings",
        )


@pytest.mark.parametrize(
    ("market", "instance_lifecycle"),
    [
        ("spot", None),
        ("on-demand", "spot"),
    ],
)
def test_worker_listing_rejects_spot_instances(
    market: str,
    instance_lifecycle: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = {
        "InstanceId": "i-0123456789abcdef0",
        "InstanceType": "c6a.8xlarge",
        "LaunchTime": None,
        "State": {"Name": "running"},
        "Tags": [
            {"Key": ec2_infra.TAG_KEY, "Value": ec2_infra.WORKER_TAG_VALUE},
            {"Key": "citrees-artifact-prefix", "Value": "repairs/run-001"},
            {"Key": "citrees-campaign-sha256", "Value": CAMPAIGN_SHA256},
            {"Key": "citrees-stage", "Value": "rankings"},
            {
                "Key": "citrees-worker-launch-id",
                "Value": "campaign-rankings-001",
            },
            {"Key": "citrees-market", "Value": market},
        ],
    }
    if instance_lifecycle is not None:
        instance["InstanceLifecycle"] = instance_lifecycle
    ec2 = MagicMock()
    ec2.describe_instances.return_value = {"Reservations": [{"Instances": [instance]}]}
    monkeypatch.setattr(ec2_infra.boto3, "client", lambda *args, **kwargs: ec2)

    with pytest.raises(
        RuntimeError,
        match="outside the exact campaign launch identity",
    ):
        ec2_infra.list_workers(
            "campaign-rankings-001",
            artifact_prefix="repairs/run-001",
            campaign_sha256=CAMPAIGN_SHA256,
            stage="rankings",
        )


@pytest.mark.parametrize("command", ["list-workers", "terminate-workers"])
@pytest.mark.parametrize(
    "missing_option",
    ["--launch-id", "--artifact-prefix", "--campaign-sha256", "--stage"],
)
def test_worker_lifecycle_cli_requires_exact_campaign_launch_scope(
    command: str,
    missing_option: str,
) -> None:
    arguments = [
        "--launch-id",
        "campaign-rankings-001",
        "--artifact-prefix",
        "repairs/run-001",
        "--campaign-sha256",
        CAMPAIGN_SHA256,
        "--stage",
        "rankings",
    ]
    missing_index = arguments.index(missing_option)
    del arguments[missing_index : missing_index + 2]
    result = CliRunner().invoke(
        infra_app,
        [command, *arguments],
        env={
            "CITREES_ARTIFACT_PREFIX": "",
            "CITREES_CAMPAIGN_SHA256": "",
            "CITREES_STAGE": "",
            "CITREES_WORKER_LAUNCH_ID": "",
        },
    )

    assert result.exit_code == 2
    assert missing_option in result.output


def test_worker_lifecycle_cli_forwards_exact_campaign_launch_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    list_workers = MagicMock(return_value=[])
    terminate_workers = MagicMock(return_value=[])
    monkeypatch.setattr(ec2_infra, "list_workers", list_workers)
    monkeypatch.setattr(ec2_infra, "terminate_workers", terminate_workers)
    arguments = [
        "--launch-id",
        "campaign-rankings-001",
        "--artifact-prefix",
        "repairs/run-001",
        "--campaign-sha256",
        CAMPAIGN_SHA256,
        "--stage",
        "rankings",
    ]

    list_result = CliRunner().invoke(infra_app, ["list-workers", *arguments])
    terminate_result = CliRunner().invoke(infra_app, ["terminate-workers", *arguments])

    assert list_result.exit_code == 0, list_result.output
    assert terminate_result.exit_code == 0, terminate_result.output
    expected = {
        "artifact_prefix": "repairs/run-001",
        "campaign_sha256": CAMPAIGN_SHA256,
        "stage": "rankings",
    }
    list_workers.assert_called_once_with("campaign-rankings-001", **expected)
    terminate_workers.assert_called_once_with("campaign-rankings-001", **expected)


def test_worker_lifecycle_python_contract_requires_complete_scope() -> None:
    for command in (list_workers_cmd, terminate_workers_cmd):
        parameters = inspect.signature(command).parameters
        for name in ("launch_id", "artifact_prefix", "campaign_sha256", "stage"):
            assert parameters[name].default is inspect.Parameter.empty
