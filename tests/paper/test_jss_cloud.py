"""Tests for immutable JSS shard transport and EC2 launch."""

from __future__ import annotations

import base64
import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import tarfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from botocore.exceptions import ClientError, ReadTimeoutError

from paper.jss.replication import behavior, cloud, shards

pytestmark = pytest.mark.paper

ACCOUNT_ID = "123456789012"
BUCKET = f"citrees-{ACCOUNT_ID}"
GIT_SHA = "a" * 40
AMI_ID = "ami-" + "1" * 17
IMAGE_URI = (
    f"{ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/citrees"
    "@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
)
SPOT_OPTIONS = {
    "MarketType": "spot",
    "SpotOptions": {
        "SpotInstanceType": "one-time",
        "InstanceInterruptionBehavior": "terminate",
    },
}


def _precondition_failed(operation: str) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": "PreconditionFailed", "Message": "exists"},
            "ResponseMetadata": {"HTTPStatusCode": 412},
        },
        operation,
    )


def _ec2_error(code: str) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": "test EC2 error"},
            "ResponseMetadata": {"HTTPStatusCode": 400},
        },
        "DescribeInstances",
    )


class _Paginator:
    def __init__(self, client: _MemoryS3) -> None:
        self.client = client

    def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, object]]:
        assert Bucket == BUCKET
        return [
            {
                "Contents": [
                    {"Key": key} for key in sorted(self.client.objects) if key.startswith(Prefix)
                ]
            }
        ]


class _MemoryS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.put_calls: list[dict[str, Any]] = []

    def put_object(self, **kwargs: Any) -> dict[str, object]:
        self.put_calls.append(kwargs)
        key = kwargs["Key"]
        if key in self.objects and kwargs.get("IfNoneMatch") == "*":
            raise _precondition_failed("PutObject")
        body = kwargs["Body"]
        payload = body.read() if hasattr(body, "read") else body
        if not isinstance(payload, bytes):
            raise TypeError("test S3 payload must be bytes")
        self.objects[key] = payload
        self.metadata[key] = dict(kwargs.get("Metadata", {}))
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        assert Bucket == BUCKET
        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        assert Bucket == BUCKET
        return {
            "ContentLength": len(self.objects[Key]),
            "Metadata": self.metadata[Key],
        }

    def get_paginator(self, operation: str) -> _Paginator:
        assert operation == "list_objects_v2"
        return _Paginator(self)


class _ConditionalConflictS3(_MemoryS3):
    def put_object(self, **kwargs: Any) -> dict[str, object]:
        super().put_object(**kwargs)
        raise ClientError(
            {
                "Error": {
                    "Code": "ConditionalRequestConflict",
                    "Message": "concurrent conditional write",
                },
                "ResponseMetadata": {"HTTPStatusCode": 409},
            },
            "PutObject",
        )


class _WinnerlessConditionalConflictS3(_MemoryS3):
    def __init__(self) -> None:
        super().__init__()
        self.conflicted_keys: set[str] = set()

    def put_object(self, **kwargs: Any) -> dict[str, object]:
        key = str(kwargs["Key"])
        if key not in self.conflicted_keys:
            self.conflicted_keys.add(key)
            self.put_calls.append(kwargs)
            raise ClientError(
                {
                    "Error": {
                        "Code": "ConditionalRequestConflict",
                        "Message": "retry required",
                    },
                    "ResponseMetadata": {"HTTPStatusCode": 409},
                },
                "PutObject",
            )
        return super().put_object(**kwargs)


class _FailFirstLaunchRecordS3(_MemoryS3):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def put_object(self, **kwargs: Any) -> dict[str, object]:
        if "/launches/" in str(kwargs["Key"]) and not self.failed:
            self.failed = True
            raise RuntimeError("simulated crash before launch-record publication")
        return super().put_object(**kwargs)


class _DelayedLaunchPaginator(_Paginator):
    def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, object]]:
        pages = super().paginate(Bucket=Bucket, Prefix=Prefix)
        if "/launches/" in Prefix:
            assert isinstance(self.client, _DelayedLaunchS3)
            self.client.launch_listing_calls += 1
            if self.client.launch_listing_calls == 1:
                return [{"Contents": []}]
        return pages


class _DelayedLaunchS3(_MemoryS3):
    def __init__(self) -> None:
        super().__init__()
        self.launch_listing_calls = 0

    def get_paginator(self, operation: str) -> _Paginator:
        assert operation == "list_objects_v2"
        return _DelayedLaunchPaginator(self)


class _MemoryEC2:
    def __init__(self, active_spec_sha256: str | None = None) -> None:
        self.active_spec_sha256 = active_spec_sha256
        self.describe_calls = 0
        self.run_calls: list[dict[str, object]] = []

    def describe_instances(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        self.describe_calls += 1
        if self.active_spec_sha256 is not None:
            campaign = _campaign()
            spec = next(
                spec
                for spec in campaign.specs
                if cloud.shard_spec_sha256(spec) == self.active_spec_sha256
            )
            request = _launch_request(campaign, spec, attempt=1)
            return {
                "Reservations": [
                    {
                        "Instances": [
                            {
                                "InstanceId": "i-active",
                                "InstanceType": campaign.instance_type,
                                "InstanceLifecycle": "spot",
                                "LaunchTime": datetime(2026, 8, 4, 0, tzinfo=UTC),
                                "CpuOptions": {"CoreCount": 1, "ThreadsPerCore": 2},
                                "Placement": {"AvailabilityZone": "us-east-1a"},
                                "State": {"Name": "running"},
                                "Tags": [
                                    {
                                        "Key": "citrees-jss-spec-sha256",
                                        "Value": self.active_spec_sha256,
                                    },
                                    {
                                        "Key": "citrees-jss-request-index",
                                        "Value": str(request.request_index),
                                    },
                                    {"Key": "citrees-jss-attempt", "Value": "1"},
                                    {"Key": "citrees-jss-market", "Value": "spot"},
                                    {
                                        "Key": "citrees-jss-client-token",
                                        "Value": request.client_token,
                                    },
                                    {"Key": "citrees-jss-component", "Value": spec.component},
                                ],
                            }
                        ]
                    }
                ]
            }
        return {"Reservations": []}

    def run_instances(self, **kwargs: object) -> dict[str, object]:
        self.run_calls.append(kwargs)
        index = len(self.run_calls)
        return {
            "Instances": [
                {
                    "InstanceId": f"i-launched-{index}",
                    "LaunchTime": datetime(2026, 8, 4, index, tzinfo=UTC),
                    "CpuOptions": {"CoreCount": 1, "ThreadsPerCore": 2},
                    "Placement": {"AvailabilityZone": "us-east-1a"},
                }
            ]
        }


class _PaginatedEC2(_MemoryEC2):
    def describe_instances(self, **kwargs: object) -> dict[str, object]:
        token = kwargs.get("NextToken")
        spec = _campaign().specs[0 if token is None else 1]
        request = _launch_request(_campaign(), spec, attempt=1)
        response: dict[str, object] = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": f"i-page-{1 if token is None else 2}",
                            "InstanceType": _campaign().instance_type,
                            "InstanceLifecycle": "spot",
                            "LaunchTime": datetime(
                                2026,
                                8,
                                4,
                                0,
                                0,
                                1 if token is None else 2,
                                tzinfo=UTC,
                            ),
                            "CpuOptions": {"CoreCount": 1, "ThreadsPerCore": 2},
                            "Placement": {"AvailabilityZone": "us-east-1a"},
                            "State": {"Name": "running"},
                            "Tags": [
                                {
                                    "Key": "citrees-jss-spec-sha256",
                                    "Value": cloud.shard_spec_sha256(spec),
                                },
                                {
                                    "Key": "citrees-jss-request-index",
                                    "Value": str(request.request_index),
                                },
                                {"Key": "citrees-jss-attempt", "Value": "1"},
                                {"Key": "citrees-jss-market", "Value": "spot"},
                                {
                                    "Key": "citrees-jss-client-token",
                                    "Value": request.client_token,
                                },
                            ],
                        }
                    ]
                }
            ]
        }
        if token is None:
            response["NextToken"] = "second"
        return response


class _AllActiveEC2(_MemoryEC2):
    def describe_instances(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        campaign = _campaign()
        instances: list[dict[str, object]] = []
        for index, spec in enumerate(campaign.specs):
            request = _launch_request(campaign, spec, attempt=1)
            instances.append(
                {
                    "InstanceId": f"i-active-{index}",
                    "InstanceType": campaign.instance_type,
                    "InstanceLifecycle": "spot",
                    "LaunchTime": datetime(2026, 8, 4, 0, 0, index, tzinfo=UTC),
                    "CpuOptions": {"CoreCount": 1, "ThreadsPerCore": 2},
                    "Placement": {"AvailabilityZone": "us-east-1a"},
                    "State": {"Name": "running"},
                    "Tags": [
                        {
                            "Key": "citrees-jss-spec-sha256",
                            "Value": cloud.shard_spec_sha256(spec),
                        },
                        {
                            "Key": "citrees-jss-request-index",
                            "Value": str(request.request_index),
                        },
                        {"Key": "citrees-jss-attempt", "Value": "1"},
                        {"Key": "citrees-jss-market", "Value": "spot"},
                        {
                            "Key": "citrees-jss-client-token",
                            "Value": request.client_token,
                        },
                    ],
                }
            )
        return {"Reservations": [{"Instances": instances}]}


class _SpotCapacityEC2(_MemoryEC2):
    def __init__(self) -> None:
        super().__init__()
        self.attempt_calls: list[dict[str, object]] = []

    def run_instances(self, **kwargs: object) -> dict[str, object]:
        self.attempt_calls.append(kwargs)
        if len(self.attempt_calls) == 1:
            raise ClientError(
                {
                    "Error": {
                        "Code": "MaxSpotInstanceCountExceeded",
                        "Message": "spot cap",
                    }
                },
                "RunInstances",
            )
        return super().run_instances(**kwargs)


class _ZonalSpotCapacityEC2(_MemoryEC2):
    def __init__(self) -> None:
        super().__init__()
        self.spot_attempts = 0
        self.attempt_calls: list[dict[str, object]] = []

    def run_instances(self, **kwargs: object) -> dict[str, object]:
        self.attempt_calls.append(kwargs)
        self.spot_attempts += 1
        if self.spot_attempts == 1:
            raise ClientError(
                {
                    "Error": {
                        "Code": "InsufficientInstanceCapacity",
                        "Message": "zonal spot capacity",
                    }
                },
                "RunInstances",
            )
        return super().run_instances(**kwargs)


class _OnDemandObservedEC2(_MemoryEC2):
    def __init__(self, active_spec_sha256: str) -> None:
        super().__init__(active_spec_sha256)

    def describe_instances(self, **kwargs: object) -> dict[str, object]:
        response = super().describe_instances(**kwargs)
        reservations = response["Reservations"]
        assert isinstance(reservations, list)
        instance = reservations[0]["Instances"][0]  # type: ignore[index]
        assert isinstance(instance, dict)
        instance["InstanceLifecycle"] = None
        tags = instance["Tags"]
        assert isinstance(tags, list)
        market_tag = next(
            tag for tag in tags if isinstance(tag, dict) and tag.get("Key") == "citrees-jss-market"
        )
        market_tag["Value"] = "on-demand"
        return response


class _IdempotentTerminatedEC2(_MemoryEC2):
    def __init__(self) -> None:
        super().__init__()
        self.responses: dict[str, dict[str, object]] = {}

    def describe_instances(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        instances: list[dict[str, object]] = []
        calls_by_token = {str(call["ClientToken"]): call for call in self.run_calls}
        for token, response in self.responses.items():
            call = calls_by_token[token]
            raw_instance = response["Instances"][0]  # type: ignore[index]
            assert isinstance(raw_instance, dict)
            tags = call["TagSpecifications"][0]["Tags"]  # type: ignore[index]
            instance = {
                **raw_instance,
                "InstanceType": call["InstanceType"],
                "InstanceLifecycle": ("spot" if "InstanceMarketOptions" in call else None),
                "State": {"Name": "terminated"},
                "StateTransitionReason": "Client.UserInitiatedShutdown",
                "Tags": tags,
            }
            instances.append(instance)
        return {"Reservations": [{"Instances": instances}] if instances else []}

    def run_instances(self, **kwargs: object) -> dict[str, object]:
        self.run_calls.append(kwargs)
        token = str(kwargs["ClientToken"])
        if token in self.responses:
            return self.responses[token]
        index = len(self.responses) + 1
        response: dict[str, object] = {
            "Instances": [
                {
                    "InstanceId": f"i-launched-{index}",
                    "LaunchTime": datetime(2026, 8, 4, index, tzinfo=UTC),
                    "CpuOptions": {"CoreCount": 1, "ThreadsPerCore": 2},
                    "Placement": {"AvailabilityZone": "us-east-1a"},
                }
            ]
        }
        self.responses[token] = response
        return response


class _IdempotentRunningEC2(_IdempotentTerminatedEC2):
    def describe_instances(self, **kwargs: object) -> dict[str, object]:
        response = super().describe_instances(**kwargs)
        reservations = response["Reservations"]
        assert isinstance(reservations, list)
        for reservation in reservations:
            assert isinstance(reservation, dict)
            instances = reservation["Instances"]
            assert isinstance(instances, list)
            for instance in instances:
                assert isinstance(instance, dict)
                instance["State"] = {"Name": "running"}
                instance["StateTransitionReason"] = ""
        return response


def _exact_instance_response(
    instance_id: str,
    state: str,
    *,
    reason: str = "",
) -> dict[str, object]:
    return {
        "Reservations": [
            {
                "Instances": [
                    {
                        "InstanceId": instance_id,
                        "State": {"Name": state},
                        "StateTransitionReason": reason,
                    }
                ]
            }
        ]
    }


class _ScriptedExactLookupEC2(_MemoryEC2):
    def __init__(
        self,
        instance_id: str,
        exact_results: list[dict[str, object] | BaseException],
    ) -> None:
        super().__init__()
        self.instance_id = instance_id
        self.exact_results = exact_results
        self.exact_calls = 0

    def describe_instances(self, **kwargs: object) -> dict[str, object]:
        self.describe_calls += 1
        if "Filters" in kwargs:
            return {"Reservations": []}
        assert kwargs == {"InstanceIds": [self.instance_id]}
        result = self.exact_results[min(self.exact_calls, len(self.exact_results) - 1)]
        self.exact_calls += 1
        if isinstance(result, BaseException):
            raise result
        return result

    def run_instances(self, **kwargs: object) -> dict[str, object]:
        response = super().run_instances(**kwargs)
        instance = response["Instances"][0]  # type: ignore[index]
        assert isinstance(instance, dict)
        instance["LaunchTime"] = datetime.now(UTC) + timedelta(seconds=len(self.run_calls))
        return response


class _TimeoutAfterAcceptedEC2(_IdempotentTerminatedEC2):
    def __init__(self) -> None:
        super().__init__()
        self.timed_out = False

    def run_instances(self, **kwargs: object) -> dict[str, object]:
        if self.timed_out:
            return super().run_instances(**kwargs)
        self.timed_out = True
        response = super().run_instances(**kwargs)
        assert response["Instances"]
        raise ReadTimeoutError(endpoint_url="https://ec2.us-east-1.amazonaws.com")


class _UndiscoverableAcceptedEC2(_TimeoutAfterAcceptedEC2):
    def describe_instances(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        return {"Reservations": []}


def _campaign() -> cloud.CloudCampaign:
    return cloud.create_campaign(
        "smoke",
        base_seed=7,
        git_sha=GIT_SHA,
        image_uri=IMAGE_URI,
        aws_account_id=ACCOUNT_ID,
        bucket=BUCKET,
        region="us-east-1",
        instance_type="c6a.large",
        ami_id=AMI_ID,
        shard_counts={component: 1 for component in cloud.COMPONENTS},
    )


def _runtime(
    *,
    instance_id: str = "i-worker",
    attempt: int = 1,
) -> cloud.CloudRuntime:
    return cloud.CloudRuntime(
        aws_account_id=ACCOUNT_ID,
        instance_id=instance_id,
        instance_type="c6a.large",
        availability_zone="us-east-1a",
        market="spot",
        image_uri=IMAGE_URI,
        ami_id=AMI_ID,
        attempt=attempt,
    )


def _launch_request(
    campaign: cloud.CloudCampaign,
    spec: shards.ShardSpec,
    *,
    attempt: int,
    request_index: int | None = None,
    subnet_id: str = "subnet-a",
) -> cloud.LaunchRequest:
    request_index = request_index or attempt
    instance_profile_name = "citrees-campaign-test"
    security_group_id = "sg-test"
    request = cloud.LaunchRequest(
        spec=spec,
        spec_sha256=cloud.shard_spec_sha256(spec),
        request_index=request_index,
        attempt=attempt,
        market="spot",
        instance_type=campaign.instance_type,
        subnet_id=subnet_id,
        instance_profile_name=instance_profile_name,
        security_group_id=security_group_id,
        client_token=cloud._client_token(
            campaign,
            spec,
            request_index=request_index,
            attempt=attempt,
            instance_type=campaign.instance_type,
            subnet_id=subnet_id,
            instance_profile_name=instance_profile_name,
            security_group_id=security_group_id,
        ),
        run_instances_sha256="0" * 64,
    )
    return replace(
        request,
        run_instances_sha256=cloud._run_instances_sha256(campaign, request),
    )


def _publish_request(
    client: _MemoryS3,
    campaign: cloud.CloudCampaign,
    spec: shards.ShardSpec,
    *,
    attempt: int,
    request_index: int | None = None,
    subnet_id: str = "subnet-a",
) -> cloud.LaunchRequest:
    request = _launch_request(
        campaign,
        spec,
        attempt=attempt,
        request_index=request_index,
        subnet_id=subnet_id,
    )
    cloud.publish_launch_request(campaign, request, s3_client=client)
    return request


def _patch_launch_prerequisites(monkeypatch: pytest.MonkeyPatch) -> None:
    campaign = _campaign()

    def ensure_profile(**kwargs: object) -> str:
        assert kwargs == {
            "output_prefix": campaign.output_prefix,
            "campaign_sha256": campaign.campaign_sha256,
            "read_keys": (campaign.manifest_key,),
            "write_prefixes": (f"{campaign.output_prefix}/shards",),
            "region": campaign.region,
        }
        return "citrees-campaign-test"

    monkeypatch.setattr(cloud, "get_aws_account_id", lambda: ACCOUNT_ID)
    monkeypatch.setattr(
        cloud,
        "validate_image_revision",
        lambda image_uri, expected_git_sha, region: GIT_SHA,
    )
    monkeypatch.setattr(cloud, "ensure_security_group", lambda region: "sg-test")
    monkeypatch.setattr(
        cloud,
        "get_default_subnet_ids",
        lambda ec2, instance_type: ["subnet-a", "subnet-b"],
    )
    monkeypatch.setattr(
        cloud,
        "ensure_campaign_iam_profile",
        ensure_profile,
    )


def _local_shard(
    root: Path,
    campaign: cloud.CloudCampaign,
    spec: shards.ShardSpec,
    runtime: cloud.CloudRuntime | None = None,
) -> Path:
    output = root / f"{spec.component}-{spec.shard_index}"
    _write_base_local_shard(output, campaign, spec)
    cloud._cloud_receipt(
        output,
        campaign,
        spec,
        runtime or _runtime(),
        cloud.archive_key(campaign, spec),
    )
    return output


def _write_base_local_shard(
    output: Path,
    campaign: cloud.CloudCampaign,
    spec: shards.ShardSpec,
) -> Path:
    output.mkdir(parents=True)
    artifact_metadata: dict[str, dict[str, object]] = {}
    table_metadata: dict[str, dict[str, object]] = {}
    for name in sorted(shards._expected_artifact_names(spec)):
        path = output / name
        frame = pd.DataFrame({"value": [1]})
        frame.to_parquet(path, index=False)
        artifact_metadata[name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        table_metadata[name.removesuffix(".parquet")] = {
            "rows": 1,
            "columns": ["value"],
        }
    assignments = shards._assignment_payload(spec)
    source_sha256 = {
        str(path.relative_to(shards.REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in shards._source_files(spec.target_analysis)
    }
    input_sha256 = (
        {dataset.name: dataset.sha256 for dataset in behavior.load_behavior_datasets()}
        if spec.target_analysis == "behavior"
        else {}
    )
    context = {
        "git_sha": campaign.git_sha,
        "git_dirty": False,
        "python": "3.12.7",
        "platform": "test-platform",
        "hardware": {
            "logical_cpus": 2,
            "machine": "x86_64",
            "processor": "test",
        },
        "blas_configuration": {"test": "fixture"},
        "thread_environment": {name: "" for name in shards.THREAD_ENVIRONMENT_KEYS},
        "r_environment": {
            "r": "fixture",
            "partykit": "fixture",
            "libcoin": "fixture",
            "mvtnorm": "fixture",
        },
        "source_sha256": source_sha256,
        "input_sha256": input_sha256,
        "versions": {package: "fixture" for package in shards.RUNTIME_PACKAGES},
    }
    (output / "receipt.json").write_text(
        json.dumps(
            {
                "analysis": "jss_shard",
                "schema_version": 1,
                "spec": vars(spec),
                "spec_sha256": shards._json_sha256(vars(spec)),
                "assignments": assignments,
                "assignment_count": len(assignments),
                "assignments_sha256": shards._json_sha256(assignments),
                "created_utc": "2026-08-04T00:00:10+00:00",
                "elapsed_seconds": 2.5,
                "execution_context_sha256": shards._json_sha256(context),
                "scientific_context_sha256": shards._json_sha256(
                    shards._scientific_context_payload(context)
                ),
                **context,
                "tables": table_metadata,
                "artifacts": artifact_metadata,
            }
        )
        + "\n",
        encoding="ascii",
    )
    return output


def _publish_runtime_launch(
    client: _MemoryS3,
    campaign: cloud.CloudCampaign,
    spec: shards.ShardSpec,
    runtime: cloud.CloudRuntime,
    *,
    launch_time: datetime | None = None,
) -> None:
    request = _publish_request(
        client,
        campaign,
        spec,
        attempt=runtime.attempt,
    )
    cloud.publish_launch_record(
        campaign,
        cloud.LaunchRecord(
            spec=spec,
            spec_sha256=cloud.shard_spec_sha256(spec),
            request_index=request.request_index,
            attempt=runtime.attempt,
            market=runtime.market,
            instance_type=runtime.instance_type,
            client_token=request.client_token,
            instance_id=runtime.instance_id,
            availability_zone=runtime.availability_zone,
            launch_time=(
                launch_time
                or datetime(
                    2026,
                    8,
                    4,
                    0,
                    0,
                    runtime.attempt - 1,
                    tzinfo=UTC,
                )
            ).isoformat(),
            logical_cpus=2,
        ),
        s3_client=client,
    )


def _publish_terminal_outcome(
    client: _MemoryS3,
    campaign: cloud.CloudCampaign,
    spec: shards.ShardSpec,
    runtime: cloud.CloudRuntime,
    *,
    observed_utc: str = "2026-08-04T00:00:10+00:00",
    reason: str = "Client.UserInitiatedShutdown",
) -> None:
    launches = cloud.list_launch_records(campaign, s3_client=client)
    cloud.publish_instance_outcome(
        campaign,
        cloud.InstanceOutcome(
            spec_sha256=cloud.shard_spec_sha256(spec),
            attempt=runtime.attempt,
            instance_id=runtime.instance_id,
            state="terminated",
            state_transition_reason=reason,
            observed_utc=observed_utc,
        ),
        launches=launches,
        s3_client=client,
    )


def _upload_local_shard(
    root: Path,
    client: _MemoryS3,
    campaign: cloud.CloudCampaign,
    spec: shards.ShardSpec,
) -> Path:
    cloud.publish_campaign(campaign, s3_client=client)
    runtime = _runtime(instance_id=f"i-{cloud.shard_spec_sha256(spec)[:12]}")
    output = _local_shard(root, campaign, spec, runtime)
    archive = root / f"{spec.component}-{spec.shard_index}.tar.gz"
    cloud.create_archive(output, archive)
    assert cloud.upload_archive(
        archive,
        campaign=campaign,
        spec=spec,
        s3_client=client,
    )
    _publish_runtime_launch(client, campaign, spec, runtime)
    _publish_terminal_outcome(client, campaign, spec, runtime)
    return archive


def test_campaign_manifest_round_trip_and_tampering_rejection() -> None:
    campaign = _campaign()
    payload = cloud._manifest_bytes(campaign)
    manifest = json.loads(payload)

    assert cloud.parse_campaign(payload, campaign.campaign_sha256) == campaign
    assert campaign.market == "spot"
    assert manifest["schema_version"] == 2
    assert manifest["market"] == "spot"

    value = dict(manifest)
    value["base_seed"] = True
    with pytest.raises(TypeError, match="base_seed must be an integer"):
        cloud.parse_campaign(json.dumps(value).encode(), campaign.campaign_sha256)

    value = json.loads(payload)
    value["specs"] = [spec for spec in value["specs"] if spec["component"] != "behavior"]
    with pytest.raises(ValueError, match="omit component 'behavior'"):
        cloud.parse_campaign(json.dumps(value).encode(), campaign.campaign_sha256)

    value = json.loads(payload)
    value["specs"].reverse()
    with pytest.raises(ValueError, match="canonical inventory"):
        cloud.parse_campaign(json.dumps(value).encode(), campaign.campaign_sha256)

    value = dict(manifest)
    value["market"] = "on-demand"
    with pytest.raises(ValueError, match="campaign market must be 'spot'"):
        cloud.parse_campaign(json.dumps(value).encode(), campaign.campaign_sha256)

    with pytest.raises(ValueError, match="requested campaign"):
        cloud.parse_campaign(payload, "c" * 64)


@pytest.mark.parametrize("invalid", [True, 1.5, "1", None])
def test_campaign_rejects_noninteger_shard_counts(invalid: object) -> None:
    counts: dict[str, object] = {component: 1 for component in cloud.COMPONENTS}
    counts["selector"] = invalid
    with pytest.raises(TypeError, match="selector shard count must be an integer"):
        cloud.create_campaign(
            "smoke",
            base_seed=7,
            git_sha=GIT_SHA,
            image_uri=IMAGE_URI,
            aws_account_id=ACCOUNT_ID,
            bucket=BUCKET,
            region="us-east-1",
            instance_type="c6a.large",
            ami_id=AMI_ID,
            shard_counts=counts,  # type: ignore[arg-type]
        )


def test_campaign_rejects_zero_shard_count() -> None:
    counts = {component: 1 for component in cloud.COMPONENTS}
    counts["selector"] = 0
    with pytest.raises(ValueError, match="at least 1"):
        cloud.create_campaign(
            "smoke",
            base_seed=7,
            git_sha=GIT_SHA,
            image_uri=IMAGE_URI,
            aws_account_id=ACCOUNT_ID,
            bucket=BUCKET,
            region="us-east-1",
            instance_type="c6a.large",
            ami_id=AMI_ID,
            shard_counts=counts,
        )


def test_campaign_manifest_publication_is_immutable() -> None:
    campaign = _campaign()
    client = _MemoryS3()

    cloud.publish_campaign(campaign, s3_client=client)
    cloud.publish_campaign(campaign, s3_client=client)
    assert client.put_calls[0]["IfNoneMatch"] == "*"

    client.objects[campaign.manifest_key] = b"tampered"
    with pytest.raises(RuntimeError, match="manifest bytes differ"):
        cloud.publish_campaign(campaign, s3_client=client)


def test_archive_key_membership_scales_linearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = cloud.create_campaign(
        "full",
        base_seed=7,
        git_sha=GIT_SHA,
        image_uri=IMAGE_URI,
        aws_account_id=ACCOUNT_ID,
        bucket=BUCKET,
        region="us-east-1",
        instance_type="c6a.large",
        ami_id=AMI_ID,
        shard_counts={component: 100 for component in cloud.COMPONENTS},
    )
    original_eq = shards.ShardSpec.__eq__
    equality_calls = 0

    def counted_eq(left: object, right: object) -> bool:
        nonlocal equality_calls
        equality_calls += 1
        return original_eq(left, right)

    monkeypatch.setattr(shards.ShardSpec, "__eq__", counted_eq)
    for spec in campaign.specs:
        cloud.archive_key(campaign, spec)

    assert equality_calls <= 2 * len(campaign.specs)


def test_archive_rejects_traversal_and_symbolic_links(tmp_path: Path) -> None:
    malicious = tmp_path / "malicious.tar.gz"
    with tarfile.open(malicious, "w:gz") as archive:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))

    with pytest.raises(ValueError, match="unsafe shard archive member"):
        cloud.extract_archive(malicious, tmp_path / "extracted")
    assert not (tmp_path / "escape").exists()

    alias = tmp_path / "alias.tar.gz"
    with tarfile.open(alias, "w:gz") as archive:
        info = tarfile.TarInfo("./receipt.json")
        info.size = 2
        archive.addfile(info, io.BytesIO(b"{}"))
    with pytest.raises(ValueError, match="unsafe shard archive member"):
        cloud.extract_archive(alias, tmp_path / "alias-extracted")

    output = tmp_path / "output"
    output.mkdir()
    (output / "receipt.json").write_text("{}\n", encoding="ascii")
    (output / "link").symlink_to(output / "receipt.json")
    with pytest.raises(ValueError, match="symbolic links"):
        cloud.create_archive(output, tmp_path / "linked.tar.gz")


def test_archive_conflict_validates_the_accepted_object_independently(tmp_path: Path) -> None:
    campaign = _campaign()
    spec = campaign.specs[0]
    client = _MemoryS3()
    output = _local_shard(tmp_path, campaign, spec)
    archive = tmp_path / "shard.tar.gz"
    cloud.create_archive(output, archive)

    assert cloud.upload_archive(
        archive,
        campaign=campaign,
        spec=spec,
        s3_client=client,
    )
    assert not cloud.upload_archive(
        archive,
        campaign=campaign,
        spec=spec,
        s3_client=client,
    )

    retry_archive = tmp_path / "retry.tar.gz"
    receipt = output / "receipt.json"
    mtime = receipt.stat().st_mtime
    os.utime(receipt, (mtime + 10, mtime + 10))
    cloud.create_archive(output, retry_archive)
    assert retry_archive.read_bytes() != archive.read_bytes()
    assert not cloud.upload_archive(
        retry_archive,
        campaign=campaign,
        spec=spec,
        s3_client=client,
    )

    key = cloud.archive_key(campaign, spec)
    existing = bytearray(client.objects[key])
    existing[-1] ^= 1
    client.objects[key] = bytes(existing)
    with pytest.raises(RuntimeError, match="downloaded archive digest differs"):
        cloud.upload_archive(
            archive,
            campaign=campaign,
            spec=spec,
            s3_client=client,
        )


def test_cloud_worker_embeds_and_validates_runtime_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    spec = campaign.specs[0]
    runtime = _runtime()
    client = _MemoryS3()

    def run_shard(
        observed_spec: shards.ShardSpec,
        output_dir: Path,
    ) -> Path:
        assert observed_spec == spec
        return _write_base_local_shard(output_dir, campaign, spec)

    monkeypatch.setattr(shards, "run_shard_to_directory", run_shard)
    _publish_runtime_launch(client, campaign, spec, runtime)
    assert cloud.run_cloud_shard(
        campaign,
        spec,
        runtime,
        s3_client=client,
    )

    archive_path = tmp_path / "download.tar.gz"
    archive_path.write_bytes(client.objects[cloud.archive_key(campaign, spec)])
    extracted = tmp_path / "extracted"
    cloud.extract_archive(archive_path, extracted)
    cloud._validate_cloud_receipt(extracted, campaign, spec)
    receipt = json.loads((extracted / "receipt.json").read_text(encoding="ascii"))
    assert receipt["cloud_execution"]["instance_id"] == runtime.instance_id
    assert receipt["cloud_execution"]["market"] == "spot"

    receipt["git_sha"] = "0" * 40
    (extracted / "receipt.json").write_text(
        json.dumps(receipt) + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="source revision"):
        cloud._validate_cloud_receipt(extracted, campaign, spec)

    receipt["git_sha"] = campaign.git_sha
    receipt["cloud_execution"]["market"] = "on-demand"
    (extracted / "receipt.json").write_text(
        json.dumps(receipt) + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="cloud market must be 'spot'"):
        cloud._validate_cloud_receipt(extracted, campaign, spec)

    on_demand_runtime = replace(
        runtime,
        market=cast(cloud.Market, "on-demand"),
    )
    with pytest.raises(ValueError, match="worker market must be 'spot'"):
        cloud._validate_runtime(campaign, on_demand_runtime)


def test_cloud_worker_rejects_unverified_output_before_upload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del tmp_path
    campaign = _campaign()
    spec = campaign.specs[0]
    runtime = _runtime()
    client = _MemoryS3()
    _publish_runtime_launch(client, campaign, spec, runtime)

    def run_invalid(_: shards.ShardSpec, output_dir: Path) -> Path:
        output_dir.mkdir()
        (output_dir / "receipt.json").write_text(
            json.dumps(
                {
                    "analysis": "jss_shard",
                    "git_dirty": False,
                    "git_sha": campaign.git_sha,
                }
            )
            + "\n",
            encoding="ascii",
        )
        return output_dir

    monkeypatch.setattr(shards, "run_shard_to_directory", run_invalid)
    with pytest.raises(ValueError, match="receipt fields differ"):
        cloud.run_cloud_shard(
            campaign,
            spec,
            runtime,
            s3_client=client,
        )
    assert cloud.archive_key(campaign, spec) not in client.objects


def test_cloud_worker_waits_for_immutable_launch_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    spec = campaign.specs[0]
    runtime = _runtime()
    client = _DelayedLaunchS3()
    _publish_runtime_launch(client, campaign, spec, runtime)

    monkeypatch.setattr(
        shards,
        "run_shard_to_directory",
        lambda observed, output: _write_base_local_shard(output, campaign, observed),
    )
    monkeypatch.setattr(cloud, "LAUNCH_RECORD_POLL_SECONDS", 0.0)
    assert cloud.run_cloud_shard(
        campaign,
        spec,
        runtime,
        s3_client=client,
    )
    assert client.launch_listing_calls == 2


def test_campaign_status_and_materialization_validate_every_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    for spec in campaign.specs:
        _upload_local_shard(tmp_path, client, campaign, spec)

    status = cloud.campaign_status(campaign, s3_client=client)
    assert status.total == len(campaign.specs)
    assert status.completed == campaign.specs
    assert status.missing == ()

    monkeypatch.setattr(shards, "_git_sha", lambda: GIT_SHA)
    output = cloud.materialize_campaign(
        campaign,
        tmp_path / "materialized",
        s3_client=client,
    )
    assert (output / "calibration" / "selector" / "00000" / "receipt.json").is_file()
    assert (output / "behavior" / "00000" / "receipt.json").is_file()
    accounting = json.loads(
        (output / cloud.COMPUTE_ACCOUNTING_FILENAME).read_text(encoding="ascii")
    )
    assert accounting["analysis"] == "jss_cloud_compute_accounting"
    assert accounting["campaign_sha256"] == campaign.campaign_sha256
    assert accounting["totals"] == {
        "accepted_archive_attempts": 5,
        "campaign_wall_seconds": 10.0,
        "distinct_capacity_rejections": 0,
        "launch_attempts": 5,
        "launch_intents": 5,
        "logical_cpus_per_instance": 2,
        "observed_instance_seconds": 50.0,
        "observed_logical_cpu_capacity_seconds": 100.0,
        "prior_attempts_without_accepted_archive": 0,
        "shards_succeeding_after_prior_attempts": 0,
        "successful_analysis_elapsed_seconds": 12.5,
        "successful_analysis_logical_cpu_capacity_seconds": 25.0,
        "successful_launch_to_receipt_logical_cpu_capacity_seconds": 100.0,
        "successful_launch_to_receipt_seconds": 50.0,
        "terminal_instance_attempts": 5,
    }
    assert accounting["markets"]["spot"]["accepted_archive_attempts"] == 5
    assert set(accounting["markets"]) == {"spot"}
    assert accounting["market"] == "spot"
    assert accounting["components"]["behavior"]["assignments"] == len(
        behavior.behavior_cell_inventory("smoke")
    )
    assert accounting["campaign_manifest"]["object_key"] == campaign.manifest_key
    assert len(accounting["launch_records"][0]["object_sha256"]) == 64
    assert len(accounting["launch_requests"]) == 5
    assert accounting["launch_rejections"] == []
    assert len(accounting["completed_shards"]) == 5
    assert len(accounting["instance_outcomes"]) == 5
    assert cloud.validate_compute_accounting(output) == accounting
    assert (output / cloud.MATERIALIZED_CAMPAIGN_MANIFEST).is_file()
    for launch in accounting["launch_records"]:
        assert (output / launch["materialized_path"]).is_file()
    for request in accounting["launch_requests"]:
        assert (output / request["materialized_path"]).is_file()
    for outcome in accounting["instance_outcomes"]:
        assert (output / outcome["materialized_path"]).is_file()

    accounting["totals"]["launch_attempts"] += 1
    (output / cloud.COMPUTE_ACCOUNTING_FILENAME).write_text(
        json.dumps(accounting, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="values differ"):
        cloud.validate_compute_accounting(output)


def test_materialization_requires_exact_verified_spec_inventory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    for spec in campaign.specs:
        _upload_local_shard(tmp_path, client, campaign, spec)

    def wrong_inventory(
        root: Path,
        *,
        target_analysis: shards.TargetAnalysis,
        **_: object,
    ) -> tuple[SimpleNamespace, ...]:
        del root
        count = 4 if target_analysis == "calibration" else 1
        return tuple(SimpleNamespace(spec=campaign.specs[0]) for _ in range(count))

    monkeypatch.setattr(shards, "discover_verified_shards", wrong_inventory)
    with pytest.raises(ValueError, match="specification inventory differs"):
        cloud.materialize_campaign(
            campaign,
            tmp_path / "wrong-inventory",
            s3_client=client,
        )


def test_compute_accounting_rejects_untracked_materialized_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    for spec in campaign.specs:
        _upload_local_shard(tmp_path, client, campaign, spec)
    monkeypatch.setattr(shards, "_git_sha", lambda: GIT_SHA)
    output = cloud.materialize_campaign(
        campaign,
        tmp_path / "materialized-with-extra",
        s3_client=client,
    )
    (output / "calibration" / "untracked.txt").write_text(
        "untracked\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="file inventory differs"):
        cloud.validate_compute_accounting(output)


def test_compute_accounting_records_success_after_prior_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    cloud.publish_campaign(campaign, s3_client=client)
    first_spec = campaign.specs[0]
    interrupted = _runtime(instance_id="i-interrupted", attempt=1)
    _publish_runtime_launch(client, campaign, first_spec, interrupted)
    _publish_terminal_outcome(
        client,
        campaign,
        first_spec,
        interrupted,
        observed_utc="2026-08-04T00:00:04+00:00",
        reason="Server.SpotInstanceTermination",
    )
    completed_runtime = _runtime(
        instance_id="i-recovered",
        attempt=2,
    )
    first_output = _local_shard(tmp_path, campaign, first_spec, completed_runtime)
    first_archive = tmp_path / "recovered.tar.gz"
    cloud.create_archive(first_output, first_archive)
    assert cloud.upload_archive(
        first_archive,
        campaign=campaign,
        spec=first_spec,
        s3_client=client,
    )
    _publish_runtime_launch(client, campaign, first_spec, completed_runtime)
    _publish_terminal_outcome(
        client,
        campaign,
        first_spec,
        completed_runtime,
    )
    for spec in campaign.specs[1:]:
        _upload_local_shard(tmp_path, client, campaign, spec)

    monkeypatch.setattr(shards, "_git_sha", lambda: GIT_SHA)
    output = cloud.materialize_campaign(
        campaign,
        tmp_path / "recovered-materialized",
        s3_client=client,
    )
    accounting = json.loads(
        (output / cloud.COMPUTE_ACCOUNTING_FILENAME).read_text(encoding="ascii")
    )

    assert accounting["totals"]["launch_attempts"] == 6
    assert accounting["totals"]["terminal_instance_attempts"] == 6
    assert accounting["totals"]["accepted_archive_attempts"] == 5
    assert accounting["totals"]["prior_attempts_without_accepted_archive"] == 1
    assert accounting["totals"]["observed_instance_seconds"] == 53.0
    assert accounting["totals"]["observed_logical_cpu_capacity_seconds"] == 106.0
    assert accounting["totals"]["shards_succeeding_after_prior_attempts"] == 1
    assert accounting["markets"]["spot"]["prior_attempts_without_accepted_archive"] == 1
    assert accounting["markets"]["spot"]["accepted_archive_attempts"] == 5
    assert set(accounting["markets"]) == {"spot"}
    assert accounting["completed_shards"][0]["attempt"] == 2


def test_compute_accounting_rejects_launch_after_accepted_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    for spec in campaign.specs:
        _upload_local_shard(tmp_path, client, campaign, spec)
    invalid_runtime = _runtime(
        instance_id="i-invalid-later",
        attempt=2,
    )
    _publish_runtime_launch(
        client,
        campaign,
        campaign.specs[0],
        invalid_runtime,
    )
    _publish_terminal_outcome(
        client,
        campaign,
        campaign.specs[0],
        invalid_runtime,
    )
    monkeypatch.setattr(shards, "_git_sha", lambda: GIT_SHA)

    with pytest.raises(ValueError, match="continues after an accepted"):
        cloud.materialize_campaign(
            campaign,
            tmp_path / "invalid-materialized",
            s3_client=client,
        )


def test_campaign_status_rejects_unexpected_archive_key() -> None:
    campaign = _campaign()
    client = _MemoryS3()
    key = f"{campaign.output_prefix}/shards/unexpected.tar.gz"
    client.objects[key] = b"x"
    client.metadata[key] = {}

    with pytest.raises(ValueError, match="unexpected objects"):
        cloud.campaign_status(campaign, s3_client=client)


def test_campaign_status_hashes_archive_bodies(tmp_path: Path) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    spec = campaign.specs[0]
    _upload_local_shard(tmp_path, client, campaign, spec)
    key = cloud.archive_key(campaign, spec)
    corrupted = bytearray(client.objects[key])
    corrupted[-1] ^= 1
    client.objects[key] = bytes(corrupted)

    with pytest.raises(RuntimeError, match="downloaded archive digest differs"):
        cloud.campaign_status(campaign, s3_client=client)


def test_launch_intents_and_capacity_observations_converge_across_controllers() -> None:
    campaign = _campaign()
    spec = campaign.specs[0]
    client = _MemoryS3()
    kwargs: dict[str, Any] = {
        "requests": (),
        "launches": (),
        "subnet_id": "subnet-a",
        "instance_profile_name": "citrees-campaign-test",
        "security_group_id": "sg-test",
    }
    first = cloud._create_launch_request(campaign, spec, **kwargs)
    second = cloud._create_launch_request(campaign, spec, **kwargs)

    assert first == second
    cloud.publish_launch_request(campaign, first, s3_client=client)
    cloud.publish_launch_request(campaign, second, s3_client=client)
    for error_code in ("InsufficientInstanceCapacity", "MaxSpotInstanceCountExceeded"):
        cloud.publish_launch_rejection(
            campaign,
            cloud.LaunchRejection(
                spec_sha256=first.spec_sha256,
                request_index=first.request_index,
                client_token=first.client_token,
                error_code=error_code,
            ),
            requests=(first,),
            s3_client=client,
        )
    runtime = _runtime(instance_id="i-concurrent")
    _publish_runtime_launch(client, campaign, spec, runtime)

    status = cloud.campaign_status(campaign, s3_client=client)
    assert status.unresolved_requests == ()
    assert len(status.launch_rejections) == 2
    assert len(status.launches) == 1


def test_launch_request_and_record_reject_on_demand_market() -> None:
    campaign = _campaign()
    spec = campaign.specs[0]
    client = _MemoryS3()
    request = _launch_request(campaign, spec, attempt=1)
    on_demand_request = replace(
        request,
        market=cast(cloud.Market, "on-demand"),
    )

    with pytest.raises(ValueError, match="launch request market must be 'spot'"):
        cloud.publish_launch_request(campaign, on_demand_request, s3_client=client)
    with pytest.raises(ValueError, match="launch request market must be 'spot'"):
        cloud._run_instance_arguments(campaign, on_demand_request)

    valid_request = _publish_request(client, campaign, spec, attempt=1)
    on_demand_record = cloud.LaunchRecord(
        spec=spec,
        spec_sha256=cloud.shard_spec_sha256(spec),
        request_index=valid_request.request_index,
        attempt=1,
        market=cast(cloud.Market, "on-demand"),
        instance_type=campaign.instance_type,
        client_token=valid_request.client_token,
        instance_id="i-on-demand",
        availability_zone="us-east-1a",
        launch_time="2026-08-04T00:00:00+00:00",
        logical_cpus=2,
    )
    with pytest.raises(ValueError, match="launch market must be 'spot'"):
        cloud.publish_launch_record(campaign, on_demand_record, s3_client=client)


def test_distinct_capacity_rejections_materialize_without_path_collision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    for spec in campaign.specs:
        _upload_local_shard(tmp_path, client, campaign, spec)
    status = cloud.campaign_status(campaign, s3_client=client)
    request = next(
        request for request in status.launch_requests if request.spec == campaign.specs[0]
    )
    for error_code in ("InsufficientInstanceCapacity", "MaxSpotInstanceCountExceeded"):
        cloud.publish_launch_rejection(
            campaign,
            cloud.LaunchRejection(
                spec_sha256=request.spec_sha256,
                request_index=request.request_index,
                client_token=request.client_token,
                error_code=error_code,
            ),
            requests=status.launch_requests,
            s3_client=client,
        )
    monkeypatch.setattr(shards, "_git_sha", lambda: GIT_SHA)

    output = cloud.materialize_campaign(
        campaign,
        tmp_path / "multi-rejection-materialized",
        s3_client=client,
    )
    accounting = cloud.validate_compute_accounting(output)
    raw_rejections = accounting["launch_rejections"]
    assert isinstance(raw_rejections, list)
    rejection_paths = {
        rejection["materialized_path"]
        for rejection in raw_rejections
        if isinstance(rejection, dict)
    }
    assert len(rejection_paths) == 2
    assert all((output / path).is_file() for path in rejection_paths)
    totals = accounting["totals"]
    assert isinstance(totals, dict)
    assert totals["launch_intents"] == 5
    assert totals["distinct_capacity_rejections"] == 2


def test_terminal_outcome_keeps_first_observation() -> None:
    campaign = _campaign()
    spec = campaign.specs[0]
    runtime = _runtime(instance_id="i-observed")
    client = _MemoryS3()
    _publish_runtime_launch(client, campaign, spec, runtime)
    launches = cloud.list_launch_records(campaign, s3_client=client)
    first = cloud.InstanceOutcome(
        spec_sha256=cloud.shard_spec_sha256(spec),
        attempt=1,
        instance_id=runtime.instance_id,
        state="terminated",
        state_transition_reason="Server.SpotInstanceTermination",
        observed_utc="2026-08-04T00:00:05+00:00",
    )
    later = replace(first, observed_utc="2026-08-04T00:00:09+00:00")

    assert (
        cloud.publish_instance_outcome(
            campaign,
            first,
            launches=launches,
            s3_client=client,
        )
        == first
    )
    assert (
        cloud.publish_instance_outcome(
            campaign,
            later,
            launches=launches,
            s3_client=client,
        )
        == first
    )


def test_materialization_requires_terminal_outcome_for_every_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    for spec in campaign.specs:
        _upload_local_shard(tmp_path, client, campaign, spec)
    missing = campaign.specs[0]
    outcome_key = (
        f"{campaign.output_prefix}/instance-outcomes/{cloud.shard_spec_sha256(missing)}/001.json"
    )
    del client.objects[outcome_key]
    del client.metadata[outcome_key]
    monkeypatch.setattr(shards, "_git_sha", lambda: GIT_SHA)

    with pytest.raises(RuntimeError, match="without terminal outcomes"):
        cloud.materialize_campaign(
            campaign,
            tmp_path / "missing-outcome",
            s3_client=client,
        )


def test_conditional_request_conflicts_reconcile_existing_objects(
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    spec = campaign.specs[0]
    runtime = _runtime()
    client = _ConditionalConflictS3()

    cloud.publish_campaign(campaign, s3_client=client)
    _publish_runtime_launch(client, campaign, spec, runtime)
    output = _local_shard(tmp_path, campaign, spec, runtime)
    archive = tmp_path / "conditional-conflict.tar.gz"
    cloud.create_archive(output, archive)
    assert not cloud.upload_archive(
        archive,
        campaign=campaign,
        spec=spec,
        s3_client=client,
    )
    assert campaign.manifest_key in client.objects
    assert cloud.archive_key(campaign, spec) in client.objects
    assert len(cloud.list_launch_records(campaign, s3_client=client)) == 1


def test_winnerless_conditional_request_conflicts_are_retried(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    spec = campaign.specs[0]
    runtime = _runtime()
    client = _WinnerlessConditionalConflictS3()
    monkeypatch.setattr(cloud, "CONDITIONAL_WRITE_RETRY_SECONDS", 0.0)

    cloud.publish_campaign(campaign, s3_client=client)
    _publish_runtime_launch(client, campaign, spec, runtime)
    output = _local_shard(tmp_path, campaign, spec, runtime)
    archive = tmp_path / "winnerless-conflict.tar.gz"
    cloud.create_archive(output, archive)
    assert cloud.upload_archive(
        archive,
        campaign=campaign,
        spec=spec,
        s3_client=client,
    )

    writes_by_key: dict[str, int] = {}
    for call in client.put_calls:
        key = str(call["Key"])
        writes_by_key[key] = writes_by_key.get(key, 0) + 1
    assert writes_by_key[campaign.manifest_key] == 2
    assert writes_by_key[cloud.archive_key(campaign, spec)] == 2
    launch_key = next(key for key in writes_by_key if "/launches/" in key)
    assert writes_by_key[launch_key] == 2


def test_worker_user_data_preserves_shell_continuations() -> None:
    campaign = _campaign()
    script = cloud._worker_user_data(
        campaign,
        campaign.specs[0],
        attempt=1,
    )
    lines = script.splitlines()

    assert next(line for line in lines if line.startswith("TOKEN=")).endswith("\\")
    assert next(line for line in lines if line.startswith("INSTANCE_ID=")).endswith("\\")
    assert next(line for line in lines if line.startswith("AVAILABILITY_ZONE=")).endswith("\\")
    assert next(line for line in lines if line.startswith("docker run ")).endswith("\\")
    assert "EXIT_CODE=$(docker wait citrees-jss-shard)" in script
    recovery_path = "/var/lib/cloud/scripts/per-boot/citrees-jss-shard-recover"
    assert recovery_path in script
    assert "Terminating recovered citrees-jss-shard instance" in script
    assert "docker inspect citrees-jss-shard" not in script
    assert "docker start citrees-jss-shard" not in script
    assert script.index(recovery_path) < script.index("docker run -d --restart no")
    subprocess.run(
        ["bash", "-n"],
        input=script,
        text=True,
        check=True,
    )


def test_launch_skips_completed_and_active_shards_and_uses_spot_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    _upload_local_shard(tmp_path, client, campaign, campaign.specs[0])
    _publish_request(
        client,
        campaign,
        campaign.specs[1],
        attempt=1,
    )
    ec2 = _MemoryEC2(cloud.shard_spec_sha256(campaign.specs[1]))

    _patch_launch_prerequisites(monkeypatch)
    records = cloud.launch_missing_shards(
        campaign,
        max_new_instances=2,
        ec2_client=ec2,
        s3_client=client,
    )

    assert [record.spec for record in records] == list(campaign.specs[2:4])
    assert [record.market for record in records] == ["spot", "spot"]
    assert all(call["InstanceMarketOptions"] == SPOT_OPTIONS for call in ec2.run_calls)
    assert ec2.run_calls[0]["ImageId"] == AMI_ID
    assert ec2.run_calls[0]["MetadataOptions"] == {
        "HttpEndpoint": "enabled",
        "HttpTokens": "required",
        "HttpPutResponseHopLimit": 2,
    }
    assert {call["SubnetId"] for call in ec2.run_calls} == {"subnet-a", "subnet-b"}
    user_data = base64.b64decode(str(ec2.run_calls[0]["UserData"])).decode()
    assert "--shard-index 0" in user_data
    launch_keys = [key for key in client.objects if f"{campaign.output_prefix}/launches/" in key]
    assert len(launch_keys) == 4
    for key in launch_keys:
        assert json.loads(client.objects[key])["campaign_sha256"] == campaign.campaign_sha256
    status = cloud.campaign_status(campaign, s3_client=client)
    assert len(status.launches) == 4
    assert {record.availability_zone for record in status.launches} == {"us-east-1a"}


def test_unterminated_launch_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    ec2 = _IdempotentRunningEC2()
    _patch_launch_prerequisites(monkeypatch)

    first = cloud.launch_missing_shards(
        campaign,
        max_new_instances=1,
        ec2_client=ec2,
        s3_client=client,
    )
    second = cloud.launch_missing_shards(
        campaign,
        max_new_instances=1,
        ec2_client=ec2,
        s3_client=client,
    )

    assert len(first) == 1
    assert len(second) == 1
    assert second[0].spec != first[0].spec
    assert all(record.attempt == 1 for record in (*first, *second))
    assert all(record.market == "spot" for record in (*first, *second))
    assert all("InstanceMarketOptions" in call for call in ec2.run_calls)
    assert len(ec2.run_calls) == 2


@pytest.mark.parametrize("state", ["pending", "running", "shutting-down"])
def test_tag_discovery_miss_exact_lookup_active_state_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    runtime = _runtime(instance_id=f"i-{state}", attempt=1)
    _publish_runtime_launch(client, campaign, campaign.specs[0], runtime)
    ec2 = _ScriptedExactLookupEC2(
        runtime.instance_id,
        [_exact_instance_response(runtime.instance_id, state)],
    )
    _patch_launch_prerequisites(monkeypatch)

    records = cloud.launch_missing_shards(
        campaign,
        max_new_instances=1,
        ec2_client=ec2,
        s3_client=client,
    )

    assert [(record.spec, record.attempt) for record in records] == [(campaign.specs[1], 1)]
    assert cloud.campaign_status(campaign, s3_client=client).instance_outcomes == ()
    assert ec2.exact_calls == 2


def test_tag_discovery_miss_exact_lookup_terminated_records_actual_reason_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    terminated = _runtime(instance_id="i-terminated", attempt=1)
    _publish_runtime_launch(client, campaign, campaign.specs[0], terminated)
    ec2 = _ScriptedExactLookupEC2(
        terminated.instance_id,
        [
            _exact_instance_response(
                terminated.instance_id,
                "terminated",
                reason="Server.SpotInstanceTermination",
            )
        ],
    )
    _patch_launch_prerequisites(monkeypatch)

    records = cloud.launch_missing_shards(
        campaign,
        max_new_instances=1,
        ec2_client=ec2,
        s3_client=client,
    )

    assert [(record.spec, record.attempt, record.market) for record in records] == [
        (campaign.specs[0], 2, "spot")
    ]
    outcomes = cloud.campaign_status(campaign, s3_client=client).instance_outcomes
    assert len(outcomes) == 1
    assert outcomes[0].instance_id == terminated.instance_id
    assert outcomes[0].state_transition_reason == "Server.SpotInstanceTermination"
    assert ec2.exact_calls == 1


@pytest.mark.parametrize(
    "exact_result",
    [
        {"Reservations": []},
        _ec2_error("InvalidInstanceID.NotFound"),
    ],
    ids=["empty-response", "not-found"],
)
def test_recent_exact_lookup_absence_is_indeterminate_and_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    exact_result: dict[str, object] | BaseException,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    runtime = _runtime(instance_id="i-recent", attempt=1)
    _publish_runtime_launch(
        client,
        campaign,
        campaign.specs[0],
        runtime,
        launch_time=datetime.now(UTC),
    )
    ec2 = _ScriptedExactLookupEC2(runtime.instance_id, [exact_result])
    _patch_launch_prerequisites(monkeypatch)

    records = cloud.launch_missing_shards(
        campaign,
        max_new_instances=1,
        ec2_client=ec2,
        s3_client=client,
    )

    assert [(record.spec, record.attempt) for record in records] == [(campaign.specs[1], 1)]
    assert cloud.campaign_status(campaign, s3_client=client).instance_outcomes == ()
    assert ec2.exact_calls == 2


def test_old_exact_lookup_not_found_then_running_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    runtime = _runtime(instance_id="i-delayed", attempt=1)
    _publish_runtime_launch(
        client,
        campaign,
        campaign.specs[0],
        runtime,
        launch_time=datetime.now(UTC)
        - timedelta(seconds=cloud.EC2_INSTANCE_PROPAGATION_SECONDS + 1),
    )
    ec2 = _ScriptedExactLookupEC2(
        runtime.instance_id,
        [
            _ec2_error("InvalidInstanceID.NotFound"),
            _exact_instance_response(runtime.instance_id, "running"),
        ],
    )
    sleep_delays: list[float] = []
    _patch_launch_prerequisites(monkeypatch)
    monkeypatch.setattr(cloud.time, "sleep", sleep_delays.append)

    records = cloud.launch_missing_shards(
        campaign,
        max_new_instances=1,
        ec2_client=ec2,
        s3_client=client,
    )

    assert [(record.spec, record.attempt) for record in records] == [(campaign.specs[1], 1)]
    assert cloud.campaign_status(campaign, s3_client=client).instance_outcomes == ()
    assert ec2.exact_calls == 3
    assert sleep_delays == [cloud.EC2_EXACT_LOOKUP_RETRY_SECONDS]


@pytest.mark.parametrize(
    "exact_result",
    [
        {"Reservations": []},
        _ec2_error("InvalidInstanceID.NotFound"),
    ],
    ids=["empty-response", "not-found"],
)
def test_old_persistent_exact_lookup_absence_is_recorded_and_retried(
    monkeypatch: pytest.MonkeyPatch,
    exact_result: dict[str, object] | BaseException,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    expired = _runtime(instance_id="i-expired", attempt=1)
    _publish_runtime_launch(
        client,
        campaign,
        campaign.specs[0],
        expired,
        launch_time=datetime.now(UTC)
        - timedelta(seconds=cloud.EC2_INSTANCE_PROPAGATION_SECONDS + 1),
    )
    ec2 = _ScriptedExactLookupEC2(expired.instance_id, [exact_result])
    sleep_delays: list[float] = []
    _patch_launch_prerequisites(monkeypatch)
    monkeypatch.setattr(cloud.time, "sleep", sleep_delays.append)

    records = cloud.launch_missing_shards(
        campaign,
        max_new_instances=1,
        ec2_client=ec2,
        s3_client=client,
    )

    assert [(record.spec, record.attempt, record.market) for record in records] == [
        (campaign.specs[0], 2, "spot")
    ]
    outcomes = cloud.campaign_status(campaign, s3_client=client).instance_outcomes
    assert len(outcomes) == 1
    assert outcomes[0].instance_id == expired.instance_id
    assert outcomes[0].state_transition_reason == "EC2.InstancePersistentlyAbsentFromExactLookup"
    assert ec2.exact_calls == cloud.EC2_EXACT_LOOKUP_MAX_ATTEMPTS
    assert sleep_delays == [
        cloud.EC2_EXACT_LOOKUP_RETRY_SECONDS,
        cloud.EC2_EXACT_LOOKUP_RETRY_SECONDS * 2,
    ]


@pytest.mark.parametrize(
    "error_code",
    ["UnauthorizedOperation", "RequestLimitExceeded", "ServiceUnavailable"],
)
def test_exact_lookup_operational_errors_propagate_without_outcome_or_launch(
    monkeypatch: pytest.MonkeyPatch,
    error_code: str,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    runtime = _runtime(instance_id="i-error", attempt=1)
    _publish_runtime_launch(client, campaign, campaign.specs[0], runtime)
    ec2 = _ScriptedExactLookupEC2(runtime.instance_id, [_ec2_error(error_code)])
    _patch_launch_prerequisites(monkeypatch)

    with pytest.raises(ClientError) as raised:
        cloud.launch_missing_shards(
            campaign,
            max_new_instances=1,
            ec2_client=ec2,
            s3_client=client,
        )

    assert raised.value.response["Error"]["Code"] == error_code
    assert cloud.campaign_status(campaign, s3_client=client).instance_outcomes == ()
    assert ec2.run_calls == []
    assert ec2.exact_calls == 1


def test_active_instance_discovery_reads_every_page() -> None:
    campaign = _campaign()
    observed = cloud._campaign_instances(campaign, _PaginatedEC2())

    assert {item.launch.spec_sha256 for item in observed} == {
        cloud.shard_spec_sha256(campaign.specs[0]),
        cloud.shard_spec_sha256(campaign.specs[1]),
    }
    assert [item.launch.instance_id for item in observed] == ["i-page-1", "i-page-2"]


def test_active_instance_discovery_rejects_on_demand_instance() -> None:
    campaign = _campaign()
    spec_sha256 = cloud.shard_spec_sha256(campaign.specs[0])

    with pytest.raises(ValueError, match="campaign instance market must be 'spot'"):
        cloud._campaign_instances(campaign, _OnDemandObservedEC2(spec_sha256))


def test_active_instances_recover_missing_launch_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    for spec in campaign.specs:
        _publish_request(
            client,
            campaign,
            spec,
            attempt=1,
        )
    _patch_launch_prerequisites(monkeypatch)

    records = cloud.launch_missing_shards(
        campaign,
        ec2_client=_AllActiveEC2(),
        s3_client=client,
    )

    assert records == ()
    launches = cloud.list_launch_records(campaign, s3_client=client)
    assert len(launches) == len(campaign.specs)
    assert {record.spec_sha256: record.instance_id for record in launches} == {
        cloud.shard_spec_sha256(spec): f"i-active-{index}"
        for index, spec in enumerate(campaign.specs)
    }


def test_terminated_launch_is_recovered_before_spot_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    client = _FailFirstLaunchRecordS3()
    ec2 = _IdempotentTerminatedEC2()
    _patch_launch_prerequisites(monkeypatch)
    monkeypatch.setattr(
        cloud,
        "get_default_subnet_ids",
        lambda ec2_client, instance_type: ["subnet-a"],
    )

    with pytest.raises(RuntimeError, match="simulated crash"):
        cloud.launch_missing_shards(
            campaign,
            max_new_instances=1,
            ec2_client=ec2,
            s3_client=client,
        )
    cloud.launch_missing_shards(
        campaign,
        max_new_instances=1,
        ec2_client=ec2,
        s3_client=client,
    )

    launches = sorted(
        cloud.list_launch_records(campaign, s3_client=client),
        key=lambda record: record.attempt,
    )
    assert [(record.attempt, record.market, record.instance_id) for record in launches] == [
        (1, "spot", "i-launched-1"),
        (2, "spot", "i-launched-2"),
    ]
    assert all("InstanceMarketOptions" in call for call in ec2.run_calls)
    assert len(ec2.run_calls) == 2
    assert ec2.run_calls[1]["ClientToken"] != ec2.run_calls[0]["ClientToken"]


def test_timeout_after_acceptance_recovers_without_replaying_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    ec2 = _TimeoutAfterAcceptedEC2()
    _patch_launch_prerequisites(monkeypatch)
    monkeypatch.setattr(
        cloud,
        "get_default_subnet_ids",
        lambda ec2_client, instance_type: ["subnet-a"],
    )

    with pytest.raises(ReadTimeoutError):
        cloud.launch_missing_shards(
            campaign,
            max_new_instances=1,
            ec2_client=ec2,
            s3_client=client,
        )
    unresolved = cloud.campaign_status(campaign, s3_client=client)
    assert len(unresolved.unresolved_requests) == 1
    assert unresolved.launches == ()

    cloud.launch_missing_shards(
        campaign,
        max_new_instances=1,
        ec2_client=ec2,
        s3_client=client,
    )

    launches = sorted(
        cloud.list_launch_records(campaign, s3_client=client),
        key=lambda record: record.attempt,
    )
    assert [(record.attempt, record.market, record.instance_id) for record in launches] == [
        (1, "spot", "i-launched-1"),
        (2, "spot", "i-launched-2"),
    ]
    assert all("InstanceMarketOptions" in call for call in ec2.run_calls)
    assert len(ec2.run_calls) == 2
    assert ec2.run_calls[1]["ClientToken"] != ec2.run_calls[0]["ClientToken"]


def test_undiscoverable_ambiguous_launch_is_not_replayed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    ec2 = _UndiscoverableAcceptedEC2()
    _patch_launch_prerequisites(monkeypatch)

    with pytest.raises(ReadTimeoutError):
        cloud.launch_missing_shards(
            campaign,
            max_new_instances=1,
            ec2_client=ec2,
            s3_client=client,
        )
    with pytest.raises(RuntimeError, match="refusing automatic replay"):
        cloud.launch_missing_shards(
            campaign,
            max_new_instances=1,
            ec2_client=ec2,
            s3_client=client,
        )

    status = cloud.campaign_status(campaign, s3_client=client)
    assert len(status.unresolved_requests) == 1
    assert status.launches == ()
    assert len(ec2.run_calls) == 1


def test_retry_attempts_come_from_immutable_launch_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    ec2 = _IdempotentTerminatedEC2()
    _patch_launch_prerequisites(monkeypatch)

    first = cloud.launch_missing_shards(
        campaign,
        max_new_instances=1,
        ec2_client=ec2,
        s3_client=client,
    )
    second = cloud.launch_missing_shards(
        campaign,
        max_new_instances=1,
        ec2_client=ec2,
        s3_client=client,
    )

    assert [first[0].attempt, second[0].attempt] == [1, 2]
    assert [first[0].market, second[0].market] == ["spot", "spot"]
    assert all("InstanceMarketOptions" in call for call in ec2.run_calls)
    launches = cloud.list_launch_records(campaign, s3_client=client)
    assert [record.attempt for record in launches] == [1, 2]


def test_launch_history_rejects_time_reversed_attempts() -> None:
    campaign = _campaign()
    client = _MemoryS3()
    spec = campaign.specs[0]
    _publish_runtime_launch(
        client,
        campaign,
        spec,
        _runtime(instance_id="i-first", attempt=1),
    )
    _publish_runtime_launch(
        client,
        campaign,
        spec,
        _runtime(instance_id="i-second", attempt=2),
    )
    second_key = next(
        key for key in client.objects if "/launches/" in key and key.endswith("/002.json")
    )
    second = json.loads(client.objects[second_key])
    second["launch_time"] = "2026-08-03T23:59:59+00:00"
    client.objects[second_key] = (
        json.dumps(second, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")

    with pytest.raises(ValueError, match="not strictly increasing"):
        cloud.list_launch_records(campaign, s3_client=client)


def test_spot_capacity_limit_stops_launch_pass_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    ec2 = _SpotCapacityEC2()
    _patch_launch_prerequisites(monkeypatch)

    records = cloud.launch_missing_shards(
        campaign,
        max_new_instances=3,
        ec2_client=ec2,
        s3_client=client,
    )

    assert len(ec2.attempt_calls) == 1
    assert ec2.attempt_calls[0]["InstanceMarketOptions"] == SPOT_OPTIONS
    assert records == ()
    status = cloud.campaign_status(campaign, s3_client=client)
    assert len(status.launch_requests) == 1
    assert len(status.launch_rejections) == 1
    assert status.launches == ()


def test_zonal_spot_capacity_error_stops_launch_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    client = _MemoryS3()
    ec2 = _ZonalSpotCapacityEC2()
    _patch_launch_prerequisites(monkeypatch)

    records = cloud.launch_missing_shards(
        campaign,
        max_new_instances=4,
        ec2_client=ec2,
        s3_client=client,
    )

    assert ec2.spot_attempts == 1
    assert ec2.attempt_calls[0]["InstanceMarketOptions"] == SPOT_OPTIONS
    assert records == ()


@pytest.mark.parametrize(
    ("argument", "value", "match"),
    [
        ("max_new_instances", True, "max_new_instances must be an integer"),
        ("max_new_instances", 1.5, "max_new_instances must be an integer"),
    ],
)
def test_launch_rejects_noninteger_counts(
    argument: str,
    value: object,
    match: str,
) -> None:
    campaign = _campaign()
    kwargs: dict[str, object] = {
        "max_new_instances": 1,
    }
    kwargs[argument] = value

    with pytest.raises(TypeError, match=match):
        cloud.launch_missing_shards(campaign, **kwargs)  # type: ignore[arg-type]


def test_launch_api_and_cli_have_no_spot_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "spot_count" not in inspect.signature(cloud.launch_missing_shards).parameters
    argv = [
        "cloud",
        "launch",
        "--profile",
        "smoke",
        "--seed",
        "7",
        *[
            argument
            for component in cloud.COMPONENTS
            for argument in (f"--{component.replace('_', '-')}-shards", "1")
        ],
        "--image-uri",
        IMAGE_URI,
        "--instance-type",
        "c6a.large",
        "--ami-id",
        AMI_ID,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = cloud._parse_args()
    assert args.command == "launch"
    assert not hasattr(args, "spot_count")

    monkeypatch.setattr(sys, "argv", [*argv, "--spot-count", "1"])
    with pytest.raises(SystemExit):
        cloud._parse_args()


def test_cloud_driver_is_part_of_shard_source_identity() -> None:
    cloud_path = Path(cloud.__file__).resolve()
    package_files = set((shards.REPO_ROOT / "citrees").glob("*.py"))
    for target in ("calibration", "behavior"):
        source_files = set(shards._source_files(target))  # type: ignore[arg-type]
        assert cloud_path in source_files
        assert package_files.issubset(source_files)
