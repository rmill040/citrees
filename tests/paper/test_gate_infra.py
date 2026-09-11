"""Tests for reproducibility-gate host tooling and worker droplet targeting."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

from paper.benchmark.experiments.r_cforest_reproducibility import gate_output_prefix
from paper.benchmark.infra import ec2 as ec2_infra
from paper.benchmark.infra import gate
from paper.benchmark.pipeline.operator_attestation import generate_operator_keypair
from paper.benchmark.pipeline.runtime_contract import (
    runtime_contract_sha256,
    serialize_runtime_contract,
)
from tests.paper.test_manifest import _runtime_contract

pytestmark = pytest.mark.paper

GIT_SHA = "a" * 40
DIGEST = "sha256:" + "b" * 64
IMAGE_URI = f"123456789012.dkr.ecr.us-east-1.amazonaws.com/citrees-123456789012@{DIGEST}"
NONCE = "c" * 32
SUBNETS = {"subnet-a": "us-east-1a", "subnet-f": "us-east-1f", "subnet-a2": "us-east-1a"}


class _Ec2:
    def __init__(self) -> None:
        self.run_requests: list[dict[str, Any]] = []
        self.instances: dict[str, dict[str, Any]] = {}
        self.terminated: list[str] = []

    def run_instances(self, **kwargs: Any) -> dict[str, Any]:
        self.run_requests.append(kwargs)
        instance_id = f"i-gate-{len(self.run_requests)}"
        self.instances[instance_id] = {
            "InstanceId": instance_id,
            "State": {"Name": "running"},
            "Tags": kwargs["TagSpecifications"][0]["Tags"],
        }
        return {"Instances": [{"InstanceId": instance_id}]}

    def describe_subnets(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "Subnets": [
                {"SubnetId": sid, "AvailabilityZone": SUBNETS[sid], "State": "available"}
                for sid in kwargs["SubnetIds"]
            ]
        }

    def describe_instances(self, **kwargs: Any) -> dict[str, Any]:
        wanted = {f["Name"]: set(f["Values"]) for f in kwargs["Filters"]}
        rows = []
        for instance in self.instances.values():
            tags = {t["Key"]: t["Value"] for t in instance["Tags"]}
            if instance["State"]["Name"] not in wanted["instance-state-name"]:
                continue
            if tags.get("citrees-gate-identity") not in wanted["tag:citrees-gate-identity"]:
                continue
            rows.append(instance)
        return {"Reservations": [{"Instances": rows}] if rows else []}

    def terminate_instances(self, **kwargs: Any) -> dict[str, Any]:
        for instance_id in kwargs["InstanceIds"]:
            self.terminated.append(instance_id)
            self.instances[instance_id]["State"] = {"Name": "terminated"}
        return {}


class _S3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        keys = sorted(k for k in self.objects if k.startswith(kwargs["Prefix"]))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        class _Body:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self) -> bytes:
                return self._data

        return {"Body": _Body(self.objects[kwargs["Key"]])}

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["IfNoneMatch"] == "*"
        if kwargs["Key"] in self.objects:
            raise ClientError(
                {"Error": {"Code": "PreconditionFailed", "Message": "exists"}}, "PutObject"
            )
        self.objects[kwargs["Key"]] = kwargs["Body"]
        return {}


@pytest.fixture
def attempt(monkeypatch: pytest.MonkeyPatch) -> gate.GateAttempt:
    monkeypatch.setattr(gate, "get_frozen_git_sha", lambda: GIT_SHA)
    monkeypatch.setattr(gate, "validate_image_revision", lambda uri, sha, region: sha)
    monkeypatch.setattr(gate, "get_ami", lambda region: "ami-0123456789abcdef0")
    monkeypatch.setattr(gate, "get_aws_account_id", lambda: "123456789012")
    monkeypatch.setattr(
        gate, "ensure_campaign_iam_profile", lambda **kwargs: "citrees-campaign-gate"
    )
    monkeypatch.setattr(gate, "ensure_security_group", lambda region: "sg-gate")
    return gate.create_gate_attempt(IMAGE_URI, launch_nonce=NONCE)


def _matching_contract(attempt: gate.GateAttempt) -> dict[str, Any]:
    contract = _runtime_contract()
    contract["runtime"].update(
        container_image_digest=attempt.image_digest,
        git_sha=attempt.source_git_sha,
        ami_id=attempt.ami_id,
        instance_type=gate.GATE_INSTANCE_TYPE,
    )
    return contract


def _user_data(request: dict[str, Any]) -> str:
    return base64.b64decode(request["UserData"]).decode()


def test_attempt_binds_revision_image_and_nonce_and_round_trips(
    attempt: gate.GateAttempt, tmp_path: Path
) -> None:
    assert attempt.prefix == gate_output_prefix(GIT_SHA, DIGEST, NONCE)
    assert attempt.bucket == "citrees-123456789012"
    assert attempt.image_digest == DIGEST
    attempt.save(tmp_path / "attempt.json")
    assert gate.GateAttempt.load(tmp_path / "attempt.json") == attempt
    assert attempt.runtime_contract_key("d" * 64).endswith(
        "/control/runtime-contract-" + "d" * 64 + ".json"
    )
    assert attempt.run_payload_key("e" * 64, "arc-a-repeat-1").endswith(
        "/runs/manifest-" + "e" * 64 + "/arc-a-repeat-1.json"
    )


def test_freeze_host_launch_request_and_user_data(
    attempt: gate.GateAttempt, tmp_path: Path
) -> None:
    public_key = generate_operator_keypair(tmp_path / "op.pem", tmp_path / "op.pub")
    ec2 = _Ec2()
    instance_id = gate.launch_freeze_host(
        attempt, operator_public_key_path=tmp_path / "op.pub", subnet_id="subnet-f", ec2=ec2
    )
    assert instance_id == "i-gate-1"
    request = ec2.run_requests[0]
    assert request["ImageId"] == attempt.ami_id
    assert request["InstanceType"] == gate.GATE_INSTANCE_TYPE
    assert request["IamInstanceProfile"] == {"Name": "citrees-campaign-gate"}
    assert request["SecurityGroupIds"] == ["sg-gate"]
    assert request["SubnetId"] == "subnet-f"
    assert request["MetadataOptions"]["HttpPutResponseHopLimit"] == 2
    assert request["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert "AdditionalInfo" not in request
    tags = {t["Key"]: t["Value"] for t in request["TagSpecifications"][0]["Tags"]}
    assert tags[ec2_infra.TAG_KEY] == "r-cforest-reproducibility-gate"
    assert tags["citrees-market"] == "on-demand" and tags["citrees-image-digest"] == DIGEST
    assert tags["citrees-gate-launch-nonce"] == NONCE and tags["citrees-source-git-sha"] == GIT_SHA
    assert tags["citrees-artifact-prefix"] == attempt.prefix
    assert tags["citrees-gate-role"] == "freeze"
    assert tags["citrees-gate-identity"] == attempt.identity
    text = _user_data(request)
    assert "freeze-runtime --operator-public-key /gate/operator-public-key.pem" in text
    pem = (tmp_path / "op.pub").read_text().strip()
    assert f"<<'KEY'\n{pem}\nKEY\n" in text
    assert public_key["algorithm"] == "ed25519"
    assert f"-e GIT_SHA={GIT_SHA}" in text and f"-e CITREES_IMAGE_URI={IMAGE_URI}" in text
    assert "--if-none-match '*'" in text
    assert f"{attempt.control_prefix}/runtime-contract-$SHA.json" in text
    assert "ROLE=freeze" in text
    assert "aws sts get-caller-identity" in text
    assert f"put_once {attempt.logs_prefix}/$ROLE-$INSTANCE_ID.log /var/log/user-data.log" in text


def test_freeze_host_targets_a_droplet_when_asked(
    attempt: gate.GateAttempt, tmp_path: Path
) -> None:
    generate_operator_keypair(tmp_path / "op.pem", tmp_path / "op.pub")
    ec2 = _Ec2()
    gate.launch_freeze_host(
        attempt,
        operator_public_key_path=tmp_path / "op.pub",
        subnet_id="subnet-f",
        target_droplet="30.99.51.144",
        ec2=ec2,
    )
    request = ec2.run_requests[0]
    assert request["AdditionalInfo"] == "target-droplet=30.99.51.144"
    tags = {t["Key"]: t["Value"] for t in request["TagSpecifications"][0]["Tags"]}
    assert tags["citrees-target-droplet"] == "30.99.51.144"


def test_gate_hosts_need_two_zones_and_run_every_repeat(attempt: gate.GateAttempt) -> None:
    ec2 = _Ec2()
    with pytest.raises(ValueError, match="distinct availability zones"):
        gate.launch_gate_hosts(
            attempt,
            manifest_sha256="e" * 64,
            runtime_contract_sha256="d" * 64,
            subnet_ids=["subnet-a", "subnet-a2"],
            ec2=ec2,
        )
    with pytest.raises(ValueError, match="one droplet per gate host"):
        gate.launch_gate_hosts(
            attempt,
            manifest_sha256="e" * 64,
            runtime_contract_sha256="d" * 64,
            subnet_ids=["subnet-a", "subnet-f"],
            target_droplets=["30.99.51.144"],
            ec2=ec2,
        )
    launched = gate.launch_gate_hosts(
        attempt,
        manifest_sha256="e" * 64,
        runtime_contract_sha256="d" * 64,
        subnet_ids=["subnet-f", "subnet-a"],
        target_droplets=["30.99.51.144", "29.81.7.184"],
        ec2=ec2,
    )
    assert launched == {"arc-a": "i-gate-1", "arc-b": "i-gate-2"}
    first, second = ec2.run_requests
    assert first["SubnetId"] == "subnet-f" and second["SubnetId"] == "subnet-a"
    assert first["AdditionalInfo"] == "target-droplet=30.99.51.144"
    assert second["AdditionalInfo"] == "target-droplet=29.81.7.184"
    text = _user_data(first)
    assert "RUN_ID=arc-a-repeat-$REPEAT" in text and "for REPEAT in 1 2; do" in text
    assert attempt.manifest_key("e" * 64) in text and attempt.runtime_contract_key("d" * 64) in text
    assert f"{'e' * 64}  /root/gate/manifest.csv" in text
    assert attempt.runs_prefix("e" * 64) in text
    assert f"sleep {gate.GATE_KEEPALIVE_SECONDS}" in text
    assert "RUN_ID=arc-b-repeat-$REPEAT" in _user_data(second)
    assert "ROLE=arc-a" in text and "ROLE=arc-b" in _user_data(second)
    assert "tee /root/gate/$RUN_ID.json.stderr.log" in text
    assert gate.list_gate_instances(attempt, ec2=ec2) == [
        {"instance_id": "i-gate-1", "role": "run", "slot": "arc-a", "state": "running"},
        {"instance_id": "i-gate-2", "role": "run", "slot": "arc-b", "state": "running"},
    ]
    assert gate.terminate_gate_hosts(attempt, ec2=ec2) == ["i-gate-1", "i-gate-2"]
    assert gate.list_gate_instances(attempt, ec2=ec2) == []


def test_wait_for_runtime_contract_requires_binding_to_the_attempt(
    attempt: gate.GateAttempt, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate.time, "sleep", lambda seconds: None)
    s3 = _S3()
    with pytest.raises(TimeoutError):
        gate.wait_for_runtime_contract(attempt, timeout_seconds=0, poll_seconds=0, s3=s3)

    contract = _matching_contract(attempt)
    payload = serialize_runtime_contract(contract)
    digest = runtime_contract_sha256(contract)
    s3.objects[attempt.runtime_contract_key(digest)] = payload
    assert gate.wait_for_runtime_contract(attempt, timeout_seconds=0, s3=s3) == (digest, payload)

    # A freeze that printed JSON with a trailing newline is normalized and republished.
    s3 = _S3()
    noisy = payload + b"\n"
    s3.objects[attempt.runtime_contract_key(hashlib.sha256(noisy).hexdigest())] = noisy
    assert gate.wait_for_runtime_contract(attempt, timeout_seconds=0, s3=s3) == (digest, payload)
    assert s3.objects[attempt.runtime_contract_key(digest)] == payload

    drifted = _matching_contract(attempt)
    drifted["runtime"]["git_sha"] = "f" * 40
    s3 = _S3()
    s3.objects[attempt.runtime_contract_key(runtime_contract_sha256(drifted))] = (
        serialize_runtime_contract(drifted)
    )
    with pytest.raises(RuntimeError, match="does not bind this attempt"):
        gate.wait_for_runtime_contract(attempt, timeout_seconds=0, s3=s3)


def test_manifest_upload_is_create_only_but_idempotent(attempt: gate.GateAttempt) -> None:
    s3 = _S3()
    payload = b"task,...\n"
    key = gate.upload_gate_manifest(attempt, manifest_payload=payload, s3=s3)
    assert key == attempt.manifest_key(hashlib.sha256(payload).hexdigest())
    assert gate.upload_gate_manifest(attempt, manifest_payload=payload, s3=s3) == key
    s3.objects[key] = b"tampered"
    with pytest.raises(RuntimeError, match="different bytes"):
        gate.upload_gate_manifest(attempt, manifest_payload=payload, s3=s3)


def test_wait_for_gate_runs_returns_all_four_payloads(
    attempt: gate.GateAttempt, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate.time, "sleep", lambda seconds: None)
    s3 = _S3()
    manifest_sha256 = "e" * 64
    for run_id in gate.EXPECTED_RUN_IDS[:-1]:
        s3.objects[attempt.run_payload_key(manifest_sha256, run_id)] = run_id.encode()
    with pytest.raises(TimeoutError, match="arc-b-repeat-2"):
        gate.wait_for_gate_runs(attempt, manifest_sha256=manifest_sha256, timeout_seconds=0, s3=s3)
    s3.objects[attempt.run_payload_key(manifest_sha256, "arc-b-repeat-2")] = b"last"
    payloads = gate.wait_for_gate_runs(
        attempt, manifest_sha256=manifest_sha256, timeout_seconds=0, s3=s3
    )
    assert set(payloads) == set(gate.EXPECTED_RUN_IDS)
    assert payloads["arc-b-repeat-2"] == b"last"


def test_complete_gate_wires_readbacks_receipt_and_nonce(
    attempt: gate.GateAttempt, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    generate_operator_keypair(tmp_path / "op.pem", tmp_path / "op.pub")
    calls: dict[str, Any] = {}

    def readbacks(payloads: Any, **kwargs: Any) -> list[dict[str, str]]:
        calls["readbacks"] = kwargs
        return [{"readback": "one"}]

    def receipt(payloads: Any, readbacks_: Any, **kwargs: Any) -> dict[str, Any]:
        calls["receipt"] = kwargs
        return {"receipt": True, "readbacks": readbacks_}

    def serialize(receipt_: Any, **kwargs: Any) -> bytes:
        return json.dumps(receipt_, sort_keys=True).encode()

    monkeypatch.setattr(gate, "collect_live_operator_readbacks", readbacks)
    monkeypatch.setattr(gate, "create_gate_receipt", receipt)
    monkeypatch.setattr(gate, "serialize_gate_receipt", serialize)
    manifest = object()
    contract = {"contract": True}
    digest = gate.complete_gate(
        attempt,
        manifest=manifest,  # type: ignore[arg-type]
        runtime_contract=contract,
        payloads=[{"p": 1}],
        operator_private_key_path=tmp_path / "op.pem",
        operator_profile="operator-profile",
        output_path=tmp_path / "receipt.json",
    )
    written = (tmp_path / "receipt.json").read_bytes()
    assert digest == hashlib.sha256(written).hexdigest()
    assert calls["readbacks"]["operator_profiles"] == ["operator-profile"]
    assert calls["readbacks"]["manifest"] is manifest
    pem = calls["readbacks"]["operator_private_key_pem"]
    assert isinstance(pem, bytes) and pem == (tmp_path / "op.pem").read_bytes()
    assert calls["receipt"]["gate_launch_nonce"] == NONCE
    assert json.loads(written)["readbacks"] == [{"readback": "one"}]


def test_worker_slots_rotate_across_droplets_within_the_cap() -> None:
    assert ec2_infra._slot_target_droplets(3, ()) == (None, None, None)
    assert ec2_infra._slot_target_droplets(3, ["10.0.0.1", "10.0.0.2"]) == (
        "10.0.0.1",
        "10.0.0.2",
        "10.0.0.1",
    )
    with pytest.raises(ValueError, match="exceed 6 per droplet"):
        ec2_infra._slot_target_droplets(13, ["10.0.0.1", "10.0.0.2"])
    with pytest.raises(ValueError, match="IPv4"):
        ec2_infra._slot_target_droplets(1, ["droplet-a"])
    with pytest.raises(ValueError, match="unique"):
        ec2_infra._slot_target_droplets(2, ["10.0.0.1", "10.0.0.1"])


def test_fetch_gate_logs_downloads_shipped_logs(attempt: gate.GateAttempt, tmp_path: Path) -> None:
    s3 = _S3()
    assert gate.fetch_gate_logs(attempt, output_dir=tmp_path / "logs", s3=s3) == []
    s3.objects[f"{attempt.logs_prefix}/freeze-i-1.log"] = b"boom"
    written = gate.fetch_gate_logs(attempt, output_dir=tmp_path / "logs", s3=s3)
    assert [p.name for p in written] == ["freeze-i-1.log"]
    assert written[0].read_bytes() == b"boom"
