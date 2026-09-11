"""Reproducibility-gate hosts for one immutable image.

The distributed benchmark refuses to run unless a runtime contract, a manifest
bound to it, and a GO receipt from the R-cforest reproducibility gate all agree
with the running image. This module launches the hosts that produce that
evidence in three repeatable steps: freeze the runtime contract on one host,
run the four gate repeats on two hosts in distinct availability zones, and
assemble the receipt locally while those hosts are still running.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import shlex
import textwrap
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from paper.benchmark.cli.console_output import info, step
from paper.benchmark.experiments.r_cforest_reproducibility import (
    GATE_HOST_SLOTS,
    GATE_MARKET,
    GATE_REPEATS,
    collect_live_operator_readbacks,
    create_gate_receipt,
    gate_launch_identity,
    gate_output_prefix,
    serialize_gate_receipt,
)
from paper.benchmark.infra.aws import (
    DEFAULT_REGION,
    ensure_campaign_iam_profile,
    ensure_security_group,
    get_aws_account_id,
    get_frozen_git_sha,
    get_resource_name,
    validate_image_revision,
)
from paper.benchmark.infra.ec2 import TAG_KEY, get_ami
from paper.benchmark.pipeline.manifest import RerunManifest
from paper.benchmark.pipeline.operator_attestation import (
    load_operator_private_key,
    load_operator_public_key,
)
from paper.benchmark.pipeline.runtime_contract import (
    runtime_contract_sha256,
    serialize_runtime_contract,
    validate_runtime_contract,
)

GATE_INSTANCE_TYPE = "c6a.8xlarge"
GATE_TAG_VALUE = "gate"
GATE_ROLE_TAG_VALUE = "r-cforest-reproducibility-gate"
GATE_KEEPALIVE_SECONDS = 6 * 3600
GATE_MODULE = "paper.benchmark.experiments.r_cforest_reproducibility"
ATTEMPT_FILE_NAME = "attempt.json"
EXPECTED_RUN_IDS = tuple(
    f"{slot}-repeat-{repeat}" for slot in GATE_HOST_SLOTS for repeat in GATE_REPEATS
)


@dataclass(frozen=True)
class GateAttempt:
    """Immutable identity of one gate attempt for one image."""

    ami_id: str
    bucket: str
    identity: str
    image_digest: str
    image_uri: str
    launch_nonce: str
    prefix: str
    region: str
    source_git_sha: str

    @property
    def control_prefix(self) -> str:
        return f"{self.prefix}/control"

    def runtime_contract_key(self, sha256: str) -> str:
        return f"{self.control_prefix}/runtime-contract-{sha256}.json"

    def manifest_key(self, sha256: str) -> str:
        return f"{self.control_prefix}/manifest-{sha256}.csv"

    @property
    def logs_prefix(self) -> str:
        return f"{self.prefix}/logs"

    def runs_prefix(self, manifest_sha256: str) -> str:
        return f"{self.prefix}/runs/manifest-{manifest_sha256}"

    def run_payload_key(self, manifest_sha256: str, run_id: str) -> str:
        return f"{self.runs_prefix(manifest_sha256)}/{run_id}.json"

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> GateAttempt:
        return cls(**json.loads(path.read_text()))


def create_gate_attempt(
    image_uri: str,
    *,
    region: str = DEFAULT_REGION,
    launch_nonce: str | None = None,
) -> GateAttempt:
    """Bind a fresh gate attempt to the clean source revision and its image."""
    source_git_sha = get_frozen_git_sha()
    validate_image_revision(image_uri, source_git_sha, region=region)
    image_digest = image_uri.rsplit("@", maxsplit=1)[1]
    nonce = launch_nonce if launch_nonce is not None else secrets.token_hex(16)
    return GateAttempt(
        ami_id=get_ami(region),
        bucket=get_resource_name(get_aws_account_id()),
        identity=gate_launch_identity(source_git_sha, image_digest, nonce),
        image_digest=image_digest,
        image_uri=image_uri,
        launch_nonce=nonce,
        prefix=gate_output_prefix(source_git_sha, image_digest, nonce),
        region=region,
        source_git_sha=source_git_sha,
    )


def gate_instance_profile(attempt: GateAttempt) -> str:
    """Ensure the create-only IAM profile for one gate attempt."""
    return ensure_campaign_iam_profile(
        output_prefix=attempt.prefix,
        campaign_sha256=attempt.identity,
        read_keys=(),
        write_prefixes=(attempt.prefix,),
        region=attempt.region,
    )


def _user_data_head(attempt: GateAttempt, *, role_label: str) -> str:
    ecr_uri = attempt.image_uri.split("/")[0]
    return textwrap.dedent(
        f"""\
        #!/bin/bash
        exec > >(tee /var/log/user-data.log) 2>&1
        set -euo pipefail
        ROLE={shlex.quote(role_label)}

        put_once() {{
            # Create-only upload; the instance role forbids overwrites.
            aws s3api put-object --bucket {attempt.bucket} --key "$1" --body "$2" \\
                --if-none-match '*' --region {attempt.region} >/dev/null
        }}

        TOKEN=$(curl --fail --silent --show-error --request PUT \\
            "http://169.254.169.254/latest/api/token" \\
            -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
        INSTANCE_ID=$(curl --fail --silent --show-error \\
            -H "X-aws-ec2-metadata-token: $TOKEN" \\
            http://169.254.169.254/latest/meta-data/instance-id)
        echo "Gate role $ROLE on $INSTANCE_ID"

        shutdown_instance() {{
            set +e
            trap - EXIT
            echo "Shipping user-data log and terminating gate instance"
            put_once {shlex.quote(attempt.logs_prefix)}/$ROLE-$INSTANCE_ID.log /var/log/user-data.log || true
            shutdown -h now || systemctl poweroff --force --force || poweroff -f || halt -f || true
        }}
        trap shutdown_instance EXIT

        yum install -y docker amazon-ssm-agent
        systemctl enable --now docker
        systemctl enable --now amazon-ssm-agent

        # The instance profile was created moments before launch; wait for credentials.
        for attempt in $(seq 1 18); do
            if aws sts get-caller-identity --region {attempt.region} >/dev/null 2>&1; then break; fi
            echo "Waiting for instance credentials ($attempt)"; sleep 10
        done
        aws sts get-caller-identity --region {attempt.region}

        aws ecr get-login-password --region {attempt.region} | \\
            docker login --username AWS --password-stdin {ecr_uri}
        docker pull {attempt.image_uri}
        mkdir -p /root/gate

        run_gate_module() {{
            docker run --rm --init \\
                -e CITREES_IMAGE_URI={attempt.image_uri} \\
                -e GIT_SHA={attempt.source_git_sha} \\
                -e AWS_DEFAULT_REGION={attempt.region} \\
                -e S3_BUCKET={attempt.bucket} \\
                -v /root/gate:/gate \\
                {attempt.image_uri} \\
                python -m {GATE_MODULE} "$@"
        }}
        """
    )


def make_freeze_user_data(attempt: GateAttempt, *, operator_public_key_pem: bytes) -> str:
    """Freeze the runtime contract on one host and publish it create-only."""
    pem = operator_public_key_pem.decode("ascii").strip()
    if "BEGIN PUBLIC KEY" not in pem:
        raise ValueError("operator public key must be a PEM public key")
    body = textwrap.dedent(
        f"""\
        cat > /root/gate/operator-public-key.pem <<'KEY'
        __OPERATOR_PUBLIC_KEY_PEM__
        KEY
        run_gate_module freeze-runtime --operator-public-key /gate/operator-public-key.pem \\
            | tr -d '\\n' > /root/gate/runtime-contract.json
        SHA=$(sha256sum /root/gate/runtime-contract.json | cut -d' ' -f1)
        put_once {shlex.quote(attempt.control_prefix)}/runtime-contract-$SHA.json /root/gate/runtime-contract.json
        echo "Published runtime contract $SHA"
        """
    )
    # Substitute after dedent so the multi-line PEM keeps every line at column zero.
    return _user_data_head(attempt, role_label="freeze") + body.replace(
        "__OPERATOR_PUBLIC_KEY_PEM__", pem
    )


def make_gate_run_user_data(
    attempt: GateAttempt,
    *,
    host_slot: str,
    manifest_sha256: str,
    runtime_contract_sha256: str,
) -> str:
    """Run every repeat for one host slot, publish payloads, then stay up for readback."""
    if host_slot not in GATE_HOST_SLOTS:
        raise ValueError(f"host_slot must be one of {GATE_HOST_SLOTS}")
    manifest_key = attempt.manifest_key(manifest_sha256)
    contract_key = attempt.runtime_contract_key(runtime_contract_sha256)
    runs_prefix = attempt.runs_prefix(manifest_sha256)
    repeats = " ".join(str(repeat) for repeat in GATE_REPEATS)
    return _user_data_head(attempt, role_label=host_slot) + textwrap.dedent(
        f"""\
        aws s3 cp s3://{attempt.bucket}/{manifest_key} /root/gate/manifest.csv --region {attempt.region}
        aws s3 cp s3://{attempt.bucket}/{contract_key} /root/gate/runtime-contract.json --region {attempt.region}
        echo "{manifest_sha256}  /root/gate/manifest.csv" | sha256sum -c -
        echo "{runtime_contract_sha256}  /root/gate/runtime-contract.json" | sha256sum -c -

        for REPEAT in {repeats}; do
            RUN_ID={host_slot}-repeat-$REPEAT
            run_gate_module run --run-id $RUN_ID --manifest /gate/manifest.csv \\
                --runtime-contract /gate/runtime-contract.json \\
                > /root/gate/$RUN_ID.json 2> >(tee /root/gate/$RUN_ID.json.stderr.log >&2)
            put_once {shlex.quote(runs_prefix)}/$RUN_ID.json /root/gate/$RUN_ID.json
            put_once {shlex.quote(runs_prefix)}/$RUN_ID.json.stderr.log /root/gate/$RUN_ID.json.stderr.log
            echo "Published $RUN_ID"
        done

        # Stay up so the operator readback sees a running host, then terminate.
        sleep {GATE_KEEPALIVE_SECONDS}
        """
    )


def _run_instance(
    ec2: Any,
    attempt: GateAttempt,
    *,
    user_data: str,
    subnet_id: str,
    instance_profile_name: str,
    security_group_id: str,
    role: str,
    host_slot: str | None,
    target_droplet: str | None,
) -> str:
    encoded = base64.b64encode(user_data.encode()).decode()
    # The receipt's live readback requires this exact tag set on every gate host.
    tags = {
        "Name": f"citrees-gate-{role}" if host_slot is None else f"citrees-gate-{host_slot}",
        "citrees-artifact-prefix": attempt.prefix,
        "citrees-gate-identity": attempt.identity,
        "citrees-gate-launch-nonce": attempt.launch_nonce,
        "citrees-gate-role": role,
        "citrees-image-digest": attempt.image_digest,
        "citrees-image-uri": attempt.image_uri,
        "citrees-market": GATE_MARKET,
        TAG_KEY: GATE_ROLE_TAG_VALUE,
        "citrees-source-git-sha": attempt.source_git_sha,
        "citrees-subnet-id": subnet_id,
    }
    if host_slot is not None:
        tags["citrees-host-slot"] = host_slot
    if target_droplet is not None:
        tags["citrees-target-droplet"] = target_droplet
    token_payload = json.dumps(
        {
            "identity": attempt.identity,
            "role": role,
            "slot": host_slot,
            "subnet_id": subnet_id,
            "user_data_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    run_kwargs: dict[str, Any] = {
        "ImageId": attempt.ami_id,
        "InstanceType": GATE_INSTANCE_TYPE,
        "MinCount": 1,
        "MaxCount": 1,
        "IamInstanceProfile": {"Name": instance_profile_name},
        "UserData": encoded,
        "MetadataOptions": {
            "HttpEndpoint": "enabled",
            "HttpPutResponseHopLimit": 2,
            "HttpTokens": "required",
        },
        "SecurityGroupIds": [security_group_id],
        "SubnetId": subnet_id,
        "InstanceInitiatedShutdownBehavior": "terminate",
        "ClientToken": hashlib.sha256(token_payload.encode()).hexdigest()[:64],
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [{"Key": key, "Value": value} for key, value in sorted(tags.items())],
            }
        ],
    }
    if target_droplet is not None:
        run_kwargs["AdditionalInfo"] = f"target-droplet={target_droplet}"
    response = ec2.run_instances(**run_kwargs)
    instances = response.get("Instances", [])
    if len(instances) != 1:
        raise RuntimeError("EC2 did not return exactly one gate instance")
    return str(instances[0]["InstanceId"])


def launch_freeze_host(
    attempt: GateAttempt,
    *,
    operator_public_key_path: Path,
    subnet_id: str,
    target_droplet: str | None = None,
    ec2: Any | None = None,
) -> str:
    """Launch the single host that freezes the runtime contract."""
    load_operator_public_key(operator_public_key_path)  # validate before launching
    client = boto3.client("ec2", region_name=attempt.region) if ec2 is None else ec2
    instance_id = _run_instance(
        client,
        attempt,
        user_data=make_freeze_user_data(
            attempt, operator_public_key_pem=operator_public_key_path.read_bytes()
        ),
        subnet_id=subnet_id,
        instance_profile_name=gate_instance_profile(attempt),
        security_group_id=ensure_security_group(attempt.region),
        role="freeze",
        host_slot=None,
        target_droplet=target_droplet,
    )
    info(f"Launched freeze host {instance_id} in {subnet_id}")
    return instance_id


def _distinct_zone_subnets(ec2: Any, subnet_ids: Sequence[str]) -> dict[str, str]:
    """Require exactly two available subnets in two different availability zones."""
    normalized = tuple(subnet_id.strip() for subnet_id in subnet_ids)
    if len(normalized) != len(GATE_HOST_SLOTS) or len(set(normalized)) != len(normalized):
        raise ValueError(f"gate hosts need exactly {len(GATE_HOST_SLOTS)} distinct subnets")
    response = ec2.describe_subnets(SubnetIds=list(normalized))
    zones: dict[str, str] = {}
    for row in response.get("Subnets", []):
        if row.get("State") != "available":
            raise RuntimeError(f"subnet {row.get('SubnetId')} is not available")
        zones[str(row["SubnetId"])] = str(row["AvailabilityZone"])
    if set(zones) != set(normalized):
        raise RuntimeError("EC2 did not describe every requested gate subnet")
    if len(set(zones.values())) != len(GATE_HOST_SLOTS):
        raise ValueError("gate hosts must be placed in distinct availability zones")
    return {subnet_id: zones[subnet_id] for subnet_id in normalized}


def launch_gate_hosts(
    attempt: GateAttempt,
    *,
    manifest_sha256: str,
    runtime_contract_sha256: str,
    subnet_ids: Sequence[str],
    target_droplets: Sequence[str] = (),
    ec2: Any | None = None,
) -> dict[str, str]:
    """Launch one host per gate slot; hosts must sit in distinct availability zones."""
    if target_droplets and len(target_droplets) != len(GATE_HOST_SLOTS):
        raise ValueError("target_droplets must name one droplet per gate host or be empty")
    client = boto3.client("ec2", region_name=attempt.region) if ec2 is None else ec2
    zones = _distinct_zone_subnets(client, subnet_ids)
    instance_profile_name = gate_instance_profile(attempt)
    security_group_id = ensure_security_group(attempt.region)
    launched: dict[str, str] = {}
    for index, (host_slot, subnet_id) in enumerate(zip(GATE_HOST_SLOTS, zones, strict=True)):
        droplet = target_droplets[index] if target_droplets else None
        instance_id = _run_instance(
            client,
            attempt,
            user_data=make_gate_run_user_data(
                attempt,
                host_slot=host_slot,
                manifest_sha256=manifest_sha256,
                runtime_contract_sha256=runtime_contract_sha256,
            ),
            subnet_id=subnet_id,
            instance_profile_name=instance_profile_name,
            security_group_id=security_group_id,
            role="run",
            host_slot=host_slot,
            target_droplet=droplet,
        )
        step(f"{host_slot}: {instance_id} in {subnet_id} ({zones[subnet_id]})")
        launched[host_slot] = instance_id
    return launched


def _list_keys(s3: Any, *, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        response = s3.list_objects_v2(**kwargs)
        keys.extend(str(item["Key"]) for item in response.get("Contents", []))
        if not response.get("IsTruncated"):
            return keys
        token = str(response.get("NextContinuationToken"))


def _get_bytes(s3: Any, *, bucket: str, key: str) -> bytes:
    return bytes(s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def wait_for_runtime_contract(
    attempt: GateAttempt,
    *,
    timeout_seconds: int = 1800,
    poll_seconds: int = 30,
    s3: Any | None = None,
) -> tuple[str, bytes]:
    """Wait for the frozen contract and verify it binds this attempt's image and revision."""
    client = boto3.client("s3", region_name=attempt.region) if s3 is None else s3
    prefix = f"{attempt.control_prefix}/runtime-contract-"
    deadline = time.monotonic() + timeout_seconds
    while True:
        keys = _list_keys(client, bucket=attempt.bucket, prefix=prefix)
        if keys:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"no runtime contract under s3://{attempt.bucket}/{prefix}")
        time.sleep(poll_seconds)
    if len(keys) != 1:
        raise RuntimeError(f"expected one runtime contract, found {len(keys)}: {keys}")
    raw = _get_bytes(client, bucket=attempt.bucket, key=keys[0])
    # The freeze prints JSON with a trailing newline; the contract format is the
    # canonical serialization, so normalize and republish under the canonical key.
    contract = validate_runtime_contract(json.loads(raw.decode("utf-8")))
    payload = serialize_runtime_contract(contract)
    digest = runtime_contract_sha256(contract)
    canonical_key = attempt.runtime_contract_key(digest)
    if keys[0] != canonical_key:
        try:
            client.put_object(
                Bucket=attempt.bucket,
                Key=canonical_key,
                Body=payload,
                ContentType="application/json",
                IfNoneMatch="*",
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in {"PreconditionFailed", "412"}:
                raise
        if _get_bytes(client, bucket=attempt.bucket, key=canonical_key) != payload:
            raise RuntimeError(f"s3://{attempt.bucket}/{canonical_key} holds different bytes")
    runtime = contract["runtime"]
    mismatches = {
        name: (observed, expected)
        for name, observed, expected in (
            ("container_image_digest", runtime["container_image_digest"], attempt.image_digest),
            ("git_sha", runtime["git_sha"], attempt.source_git_sha),
            ("ami_id", runtime["ami_id"], attempt.ami_id),
            ("instance_type", runtime["instance_type"], GATE_INSTANCE_TYPE),
        )
        if observed != expected
    }
    if mismatches:
        raise RuntimeError(f"runtime contract does not bind this attempt: {mismatches}")
    return digest, payload


def upload_gate_manifest(
    attempt: GateAttempt,
    *,
    manifest_payload: bytes,
    s3: Any | None = None,
) -> str:
    """Publish the canonical manifest to the attempt's control prefix exactly once."""
    client = boto3.client("s3", region_name=attempt.region) if s3 is None else s3
    digest = hashlib.sha256(manifest_payload).hexdigest()
    key = attempt.manifest_key(digest)
    try:
        client.put_object(
            Bucket=attempt.bucket,
            Key=key,
            Body=manifest_payload,
            ContentType="text/csv",
            IfNoneMatch="*",
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {"PreconditionFailed", "412"}:
            raise
        existing = _get_bytes(client, bucket=attempt.bucket, key=key)
        if existing != manifest_payload:
            raise RuntimeError(f"s3://{attempt.bucket}/{key} holds different bytes") from exc
    return key


def wait_for_gate_runs(
    attempt: GateAttempt,
    *,
    manifest_sha256: str,
    timeout_seconds: int = 3 * 3600,
    poll_seconds: int = 60,
    s3: Any | None = None,
) -> dict[str, bytes]:
    """Wait for all four run payloads and return them keyed by run identifier."""
    client = boto3.client("s3", region_name=attempt.region) if s3 is None else s3
    expected = {
        run_id: attempt.run_payload_key(manifest_sha256, run_id) for run_id in EXPECTED_RUN_IDS
    }
    deadline = time.monotonic() + timeout_seconds
    while True:
        present = set(
            _list_keys(client, bucket=attempt.bucket, prefix=attempt.runs_prefix(manifest_sha256))
        )
        missing = [run_id for run_id, key in expected.items() if key not in present]
        if not missing:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"gate runs still missing after {timeout_seconds}s: {missing}")
        time.sleep(poll_seconds)
    return {
        run_id: _get_bytes(client, bucket=attempt.bucket, key=key)
        for run_id, key in expected.items()
    }


def complete_gate(
    attempt: GateAttempt,
    *,
    manifest: RerunManifest,
    runtime_contract: dict[str, Any],
    payloads: Sequence[dict[str, Any]],
    operator_private_key_path: Path,
    operator_profile: str,
    output_path: Path,
) -> str:
    """Sign live readbacks, create the GO receipt, write it, and return its digest."""
    readbacks = collect_live_operator_readbacks(
        payloads,
        manifest=manifest,
        operator_private_key_pem=load_operator_private_key(operator_private_key_path),
        operator_profiles=[operator_profile],
        runtime_contract=runtime_contract,
    )
    receipt = create_gate_receipt(
        payloads,
        readbacks,
        gate_launch_nonce=attempt.launch_nonce,
        manifest=manifest,
        runtime_contract=runtime_contract,
    )
    serialized = serialize_gate_receipt(
        receipt, manifest=manifest, runtime_contract=runtime_contract
    )
    output_path.write_bytes(serialized)
    return hashlib.sha256(serialized).hexdigest()


def list_gate_instances(attempt: GateAttempt, *, ec2: Any | None = None) -> list[dict[str, str]]:
    """Return live instances tagged with this attempt's identity."""
    client = boto3.client("ec2", region_name=attempt.region) if ec2 is None else ec2
    response = client.describe_instances(
        Filters=[
            {"Name": f"tag:{TAG_KEY}", "Values": [GATE_ROLE_TAG_VALUE]},
            {"Name": "tag:citrees-gate-identity", "Values": [attempt.identity]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping"]},
        ]
    )
    rows: list[dict[str, str]] = []
    for reservation in response.get("Reservations", []):
        for instance in reservation.get("Instances", []):
            tags = {tag["Key"]: tag["Value"] for tag in instance.get("Tags", [])}
            rows.append(
                {
                    "instance_id": str(instance["InstanceId"]),
                    "role": tags.get("citrees-gate-role", ""),
                    "slot": tags.get("citrees-host-slot", ""),
                    "state": str(instance.get("State", {}).get("Name", "")),
                }
            )
    return sorted(rows, key=lambda row: (row["role"], row["slot"], row["instance_id"]))


def terminate_gate_hosts(attempt: GateAttempt, *, ec2: Any | None = None) -> list[str]:
    """Terminate every live instance of this attempt."""
    client = boto3.client("ec2", region_name=attempt.region) if ec2 is None else ec2
    instance_ids = [row["instance_id"] for row in list_gate_instances(attempt, ec2=client)]
    if instance_ids:
        client.terminate_instances(InstanceIds=instance_ids)
    return instance_ids


def fetch_gate_logs(attempt: GateAttempt, *, output_dir: Path, s3: Any | None = None) -> list[Path]:
    """Download every shipped user-data log of this attempt."""
    client = boto3.client("s3", region_name=attempt.region) if s3 is None else s3
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for key in _list_keys(client, bucket=attempt.bucket, prefix=f"{attempt.logs_prefix}/"):
        path = output_dir / key.rsplit("/", maxsplit=1)[1]
        path.write_bytes(_get_bytes(client, bucket=attempt.bucket, key=key))
        written.append(path)
    return written
