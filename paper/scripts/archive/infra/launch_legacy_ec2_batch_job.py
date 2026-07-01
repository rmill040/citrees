#!/usr/bin/env python3
"""Launch a one-off EC2 batch host and start detached experiment commands.

Archived infrastructure helper; not part of the current paper-facing workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import stat
import subprocess
import tarfile
import tempfile
import textwrap
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PROFILE = os.environ.get("AWS_PROFILE", "mm")
DEFAULT_REGION = "us-east-1"
DEFAULT_INSTANCE_TYPE = "c7i.24xlarge"
DEFAULT_VOLUME_SIZE_GB = 256
DEFAULT_KEY_DIR = Path.home() / "aws-creds"
DEFAULT_KEY_NAME = "citrees-mm-batch"
DEFAULT_SECURITY_GROUP = "citrees-batch-ssh"
DEFAULT_REMOTE_USER = "ec2-user"
DEFAULT_INSTANCE_NAME = "citrees-batch"
DEFAULT_INSTANCE_PROFILE = "citrees-worker"
DEFAULT_S3_PREFIX = "adhoc-batch"
LOCAL_DATA_DIR = REPO_ROOT.parent / "data"
SCRATCH_RUN_DIR = REPO_ROOT / "scratch" / "ec2-batch"
SSH_OPTIONS = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=10",
]
REPO_EXCLUDES = {
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "scratch/.uv_cache",
    "scratch/ec2-batch",
}
REPO_INCLUDE_FILES = {
    "README.md",
    "mkdocs.yml",
    "pyproject.toml",
    "uv.lock",
    ".pre-commit-config.yaml",
}
REPO_INCLUDE_PREFIXES = (
    "citrees",
    "docs",
    "paper/arxiv",
    "paper/docs",
    "paper/scripts",
    "tests",
    "tools",
)


@dataclass(frozen=True)
class LaunchConfig:
    """Describe one ad hoc EC2 batch launch."""

    profile: str
    region: str
    instance_type: str
    volume_size_gb: int
    key_name: str
    key_dir: Path
    iam_instance_profile: str
    security_group_name: str
    instance_name: str
    remote_user: str
    sync_local_data: bool
    s3_bucket: str | None
    s3_prefix: str
    collect_paths: tuple[str, ...]
    commands: tuple[str, ...]
    spot: bool
    shutdown_when_done: bool


@dataclass(frozen=True)
class InstanceDetails:
    """Track the launched instance and how to reach it."""

    instance_id: str
    public_ip: str
    public_dns: str
    key_path: Path
    remote_user: str


def _run(
    cmd: Sequence[str],
    *,
    capture_output: bool = True,
    check: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a local subprocess and return the completed process."""
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        capture_output=capture_output,
        check=check,
        text=True,
    )


def _aws(
    config: LaunchConfig, *args: str, capture_output: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run one AWS CLI command with the configured profile and region."""
    return _run(
        ["aws", "--profile", config.profile, "--region", config.region, *args],
        capture_output=capture_output,
        check=True,
    )


def _require_local_tool(name: str) -> None:
    """Raise a clear error if a required local binary is missing."""
    if shutil.which(name) is None:
        raise RuntimeError(f"Required local tool is not installed: {name}")


def _current_public_cidr() -> str:
    """Return the caller's current public IP as a /32 CIDR block."""
    for url in (
        "https://checkip.amazonaws.com",
        "https://api.ipify.org",
        "https://ifconfig.me/ip",
    ):
        try:
            with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
                ip = response.read().decode("utf-8").strip()
            if ip:
                return f"{ip}/32"
        except Exception:
            continue
    raise RuntimeError("Could not determine current public IP for SSH ingress")


def _ensure_key_pair(config: LaunchConfig) -> Path:
    """Create or reuse an EC2 key pair and save the private key locally."""
    config.key_dir.mkdir(parents=True, exist_ok=True)
    key_path = config.key_dir / f"{config.key_name}.pem"

    exists_remote = True
    try:
        _aws(config, "ec2", "describe-key-pairs", "--key-names", config.key_name)
    except subprocess.CalledProcessError:
        exists_remote = False

    if exists_remote and key_path.exists():
        key_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        return key_path

    if exists_remote and not key_path.exists():
        raise RuntimeError(
            f"EC2 key pair '{config.key_name}' exists in AWS, but {key_path} is missing locally. "
            "Pick a new key name or delete the remote key pair first."
        )

    if key_path.exists() and not exists_remote:
        raise RuntimeError(
            f"Local key file {key_path} exists, but EC2 key pair '{config.key_name}' does not. "
            "Pick a new key name or remove the stale local file."
        )

    result = _aws(
        config,
        "ec2",
        "create-key-pair",
        "--key-name",
        config.key_name,
        "--query",
        "KeyMaterial",
        "--output",
        "text",
    )
    key_path.write_text(result.stdout)
    key_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return key_path


def _default_vpc_id(config: LaunchConfig) -> str:
    """Return the default VPC id in the configured region."""
    result = _aws(
        config,
        "ec2",
        "describe-vpcs",
        "--filters",
        "Name=isDefault,Values=true",
        "--query",
        "Vpcs[0].VpcId",
        "--output",
        "text",
    )
    vpc_id = result.stdout.strip()
    if not vpc_id or vpc_id == "None":
        raise RuntimeError("No default VPC found in the selected region")
    return vpc_id


def _default_s3_bucket(config: LaunchConfig) -> str:
    """Return the default results bucket for the selected AWS account."""
    result = _aws(
        config,
        "sts",
        "get-caller-identity",
        "--query",
        "Account",
        "--output",
        "text",
    )
    account_id = result.stdout.strip()
    if not account_id:
        raise RuntimeError("Could not determine AWS account id for default S3 bucket")
    return f"citrees-{account_id}"


def _default_subnet_id(config: LaunchConfig, vpc_id: str) -> str:
    """Return one default subnet id from the selected VPC."""
    result = _aws(
        config,
        "ec2",
        "describe-subnets",
        "--filters",
        f"Name=vpc-id,Values={vpc_id}",
        "Name=default-for-az,Values=true",
        "--query",
        "Subnets[0].SubnetId",
        "--output",
        "text",
    )
    subnet_id = result.stdout.strip()
    if not subnet_id or subnet_id == "None":
        raise RuntimeError(f"No default subnet found in VPC {vpc_id}")
    return subnet_id


def _ensure_security_group(config: LaunchConfig, vpc_id: str, ssh_cidr: str) -> str:
    """Create or reuse a security group that allows SSH from the caller IP."""
    result = _aws(
        config,
        "ec2",
        "describe-security-groups",
        "--filters",
        f"Name=group-name,Values={config.security_group_name}",
        f"Name=vpc-id,Values={vpc_id}",
        "--query",
        "SecurityGroups[0].GroupId",
        "--output",
        "text",
    )
    group_id = result.stdout.strip()

    if not group_id or group_id == "None":
        create = _aws(
            config,
            "ec2",
            "create-security-group",
            "--group-name",
            config.security_group_name,
            "--description",
            "SSH access for citrees ad hoc batch hosts",
            "--vpc-id",
            vpc_id,
            "--query",
            "GroupId",
            "--output",
            "text",
        )
        group_id = create.stdout.strip()

    ingress_rule = json.dumps(
        [
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": ssh_cidr, "Description": "SSH from caller"}],
            }
        ]
    )
    try:
        _aws(
            config,
            "ec2",
            "authorize-security-group-ingress",
            "--group-id",
            group_id,
            "--ip-permissions",
            ingress_rule,
        )
    except subprocess.CalledProcessError as exc:
        duplicate = "InvalidPermission.Duplicate" in (exc.stderr or "")
        if not duplicate:
            raise

    return group_id


def _latest_al2023_ami(config: LaunchConfig) -> str:
    """Return the latest Amazon Linux 2023 x86_64 AMI id."""
    result = _aws(
        config,
        "ssm",
        "get-parameter",
        "--name",
        "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64",
        "--query",
        "Parameter.Value",
        "--output",
        "text",
    )
    return result.stdout.strip()


def _launch_instance(
    config: LaunchConfig,
    *,
    subnet_id: str,
    security_group_id: str,
    key_name: str,
    ami_id: str,
) -> str:
    """Launch one EC2 instance and return the new instance id."""
    network_interfaces = json.dumps(
        [
            {
                "AssociatePublicIpAddress": True,
                "DeviceIndex": 0,
                "SubnetId": subnet_id,
                "Groups": [security_group_id],
            }
        ]
    )
    block_device_mappings = json.dumps(
        [
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "DeleteOnTermination": True,
                    "VolumeSize": config.volume_size_gb,
                    "VolumeType": "gp3",
                },
            }
        ]
    )
    tag_specifications = json.dumps(
        [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": config.instance_name},
                    {"Key": "citrees-role", "Value": "adhoc-batch"},
                ],
            }
        ]
    )

    cmd = [
        "ec2",
        "run-instances",
        "--image-id",
        ami_id,
        "--instance-type",
        config.instance_type,
        "--count",
        "1",
        "--key-name",
        key_name,
        "--iam-instance-profile",
        f"Name={config.iam_instance_profile}",
        "--network-interfaces",
        network_interfaces,
        "--block-device-mappings",
        block_device_mappings,
        "--instance-initiated-shutdown-behavior",
        "terminate" if config.shutdown_when_done else "stop",
        "--tag-specifications",
        tag_specifications,
        "--query",
        "Instances[0].InstanceId",
        "--output",
        "text",
    ]
    if config.spot:
        market_options = json.dumps(
            {
                "MarketType": "spot",
                "SpotOptions": {
                    "SpotInstanceType": "one-time",
                    "InstanceInterruptionBehavior": "terminate",
                },
            }
        )
        cmd.extend(["--instance-market-options", market_options])

    result = _aws(config, *cmd)
    instance_id = result.stdout.strip()
    if not instance_id:
        raise RuntimeError("AWS CLI did not return an instance id")
    return instance_id


def _wait_for_instance(config: LaunchConfig, instance_id: str) -> InstanceDetails:
    """Wait until the instance is SSH-ready and return its connection details.

    AL2023 instances may perform a one-time first-boot reboot (e.g., for
    chronicled agent install or SELinux relabeling).  We wait for the instance
    to be running, but _wait_for_ssh handles transient SSH failures caused by
    this reboot transparently.
    """
    _aws(
        config,
        "ec2",
        "wait",
        "instance-running",
        "--instance-ids",
        instance_id,
        capture_output=False,
    )
    describe = _aws(
        config,
        "ec2",
        "describe-instances",
        "--instance-ids",
        instance_id,
        "--query",
        "Reservations[0].Instances[0].{PublicIp:PublicIpAddress,PublicDns:PublicDnsName}",
        "--output",
        "json",
    )
    payload = json.loads(describe.stdout)
    public_ip = str(payload["PublicIp"])
    public_dns = str(payload["PublicDns"])
    return InstanceDetails(
        instance_id=instance_id,
        public_ip=public_ip,
        public_dns=public_dns,
        key_path=config.key_dir / f"{config.key_name}.pem",
        remote_user=config.remote_user,
    )


def _should_include_repo_member(path: Path) -> bool:
    """Return whether one repo path should be packed into the sync archive."""
    rel = path.relative_to(REPO_ROOT)
    rel_str = rel.as_posix()
    if rel_str in REPO_EXCLUDES:
        return False
    if any(part in REPO_EXCLUDES for part in rel.parts):
        return False
    if rel_str in REPO_INCLUDE_FILES:
        return True
    for prefix in REPO_INCLUDE_PREFIXES:
        if rel_str == prefix or rel_str.startswith(f"{prefix}/"):
            return True
    return False


def _build_archive(
    source_dir: Path, *, arcname: str, include_filter: Callable[[Path], bool]
) -> Path:
    """Create a gzip tar archive for one directory tree."""
    fd, tmp_name = tempfile.mkstemp(prefix=f"{arcname}-", suffix=".tar.gz")
    os.close(fd)
    archive_path = Path(tmp_name)
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in sorted(source_dir.rglob("*")):
            if not include_filter(path):
                continue
            rel = path.relative_to(source_dir)
            tar.add(path, arcname=str(Path(arcname) / rel), recursive=False)
        tar.add(source_dir, arcname=arcname, recursive=False)
    return archive_path


def _build_repo_archive() -> Path:
    """Create the repo sync archive with a conservative exclude set."""
    return _build_archive(
        REPO_ROOT, arcname=REPO_ROOT.name, include_filter=_should_include_repo_member
    )


def _build_data_archive() -> Path:
    """Create the optional local data sync archive."""
    if not LOCAL_DATA_DIR.exists():
        raise RuntimeError(f"Local data directory does not exist: {LOCAL_DATA_DIR}")
    return _build_archive(LOCAL_DATA_DIR, arcname="data", include_filter=lambda _: True)


def _ssh_base_args(details: InstanceDetails) -> list[str]:
    """Return the shared SSH arguments for this host."""
    return [
        "ssh",
        *SSH_OPTIONS,
        "-i",
        str(details.key_path),
        f"{details.remote_user}@{details.public_ip}",
    ]


def _scp_base_args(details: InstanceDetails) -> list[str]:
    """Return the shared SCP arguments for this host."""
    return ["scp", *SSH_OPTIONS, "-i", str(details.key_path)]


def _ssh(details: InstanceDetails, command: str) -> None:
    """Run one remote shell command over SSH."""
    _run([*_ssh_base_args(details), command], capture_output=False, check=True)


def _scp(details: InstanceDetails, local_path: Path, remote_path: str) -> None:
    """Copy one local file to the instance with SCP."""
    _run(
        [
            *_scp_base_args(details),
            str(local_path),
            f"{details.remote_user}@{details.public_ip}:{remote_path}",
        ],
        capture_output=False,
        check=True,
    )


def _wait_for_ssh(details: InstanceDetails, *, timeout_seconds: int = 480) -> None:
    """Poll SSH until the instance accepts remote commands.

    Uses a generous default timeout (480s) because AL2023 instances may
    reboot once shortly after first launch (chronicled agent install).
    """
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            _run([*_ssh_base_args(details), "echo ok"], capture_output=True, check=True)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(5)
    raise RuntimeError(f"SSH did not become ready within {timeout_seconds}s") from last_error


def _remote_launcher_script(config: LaunchConfig) -> str:
    """Build the remote detached launcher script body."""
    quoted_commands = "\n".join(
        [
            f"printf '%s\\n' {shlex.quote(f'[command {idx}/{len(config.commands)}] {command}')}\n{command}"
            for idx, command in enumerate(config.commands, start=1)
        ]
    )
    collect_lines = "\n".join(
        [
            textwrap.dedent(
                f"""\
                if [ -e "$HOME/citrees/{path}" ]; then
                    if [ -d "$HOME/citrees/{path}" ]; then
                        aws s3 sync "$HOME/citrees/{path}" "$S3_DEST_ROOT/{path}/" --only-show-errors || true
                    else
                        aws s3 cp "$HOME/citrees/{path}" "$S3_DEST_ROOT/{path}" --only-show-errors || true
                    fi
                fi
                """
            ).strip()
            for path in config.collect_paths
        ]
    )
    shutdown_line = "sudo shutdown -h now" if config.shutdown_when_done else ""
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
        INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)

        # Cancel any pending system reboots (AL2023 SSM/patching)
        sudo shutdown -c 2>/dev/null || true
        # Disable unattended updates that can trigger reboots
        sudo systemctl stop amazon-ssm-agent 2>/dev/null || true
        sudo systemctl disable amazon-ssm-agent 2>/dev/null || true

        sudo dnf install -y git rsync tar tmux unzip >/dev/null
        if ! command -v curl >/dev/null 2>&1; then
            sudo dnf install -y curl-minimal >/dev/null
        fi
        if ! command -v aws >/dev/null 2>&1; then
            sudo dnf install -y awscli2 >/dev/null || sudo dnf install -y awscli >/dev/null
        fi
        if ! command -v uv >/dev/null 2>&1; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
        fi

        export PATH="$HOME/.local/bin:$PATH"
        export AWS_DEFAULT_REGION={shlex.quote(config.region)}
        export EC2_INSTANCE_ID="$INSTANCE_ID"
        export EC2_INSTANCE_TYPE={shlex.quote(config.instance_type)}
        export S3_BUCKET={shlex.quote(config.s3_bucket or "")}
        export RPY2_CFFI_MODE=ABI
        export UV_PYTHON=3.13
        mkdir -p "$HOME/logs" "$HOME/data" "$HOME/citrees/scratch/.uv_cache"
        export S3_DEST_ROOT="s3://$S3_BUCKET/{shlex.quote(config.s3_prefix)}/$INSTANCE_ID"

        sync_artifacts() {{
            printf '%s\\n' "$1" > "$HOME/logs/exit_code.txt"
            if [ -n "$S3_BUCKET" ]; then
                aws s3 sync "$HOME/logs" "$S3_DEST_ROOT/logs/" --only-show-errors || true
                {collect_lines}
            fi
        }}

        finish() {{
            exit_code=$?
            sync_artifacts "$exit_code"
            {shutdown_line}
            exit "$exit_code"
        }}
        trap finish EXIT

        cd "$HOME/citrees"
        uv sync --group paper

        {quoted_commands}
        """
    )


def _sync_project(details: InstanceDetails, config: LaunchConfig) -> None:
    """Copy the repo and optional local data archives, then extract them remotely."""
    repo_archive = _build_repo_archive()
    data_archive = _build_data_archive() if config.sync_local_data else None
    try:
        _scp(details, repo_archive, "~/citrees-repo.tar.gz")
        if data_archive is not None:
            _scp(details, data_archive, "~/citrees-data.tar.gz")
        extract_lines = [
            "rm -rf ~/citrees",
            "mkdir -p ~/citrees ~/data ~/logs",
            "tar -xzf ~/citrees-repo.tar.gz -C ~",
            "rm -f ~/citrees-repo.tar.gz",
        ]
        if data_archive is not None:
            extract_lines.extend(
                [
                    "rm -rf ~/data",
                    "tar -xzf ~/citrees-data.tar.gz -C ~",
                    "rm -f ~/citrees-data.tar.gz",
                ]
            )
        _ssh(details, " && ".join(extract_lines))
    finally:
        repo_archive.unlink(missing_ok=True)
        if data_archive is not None:
            data_archive.unlink(missing_ok=True)


def _start_remote_job(details: InstanceDetails, config: LaunchConfig) -> str:
    """Upload the launcher script and start it in the background."""
    fd, tmp_name = tempfile.mkstemp(prefix="citrees-remote-run-", suffix=".sh")
    os.close(fd)
    launcher_path = Path(tmp_name)
    launcher_path.write_text(_remote_launcher_script(config))
    try:
        _scp(details, launcher_path, "~/citrees-run.sh")
        _ssh(details, "chmod +x ~/citrees-run.sh")
        result = _run(
            [
                *_ssh_base_args(details),
                "nohup bash ~/citrees-run.sh > ~/logs/citrees-run.log 2>&1 < /dev/null & echo $!",
            ],
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
    finally:
        launcher_path.unlink(missing_ok=True)


def _write_manifest(config: LaunchConfig, details: InstanceDetails, pid: str) -> Path:
    """Persist one local manifest for the launched batch run."""
    SCRATCH_RUN_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    manifest_path = SCRATCH_RUN_DIR / f"{timestamp}-{details.instance_id}.json"
    payload: dict[str, Any] = {
        "launch_config": {
            **asdict(config),
            "key_dir": str(config.key_dir),
        },
        "instance": {
            "instance_id": details.instance_id,
            "public_ip": details.public_ip,
            "public_dns": details.public_dns,
            "remote_user": details.remote_user,
            "key_path": str(details.key_path),
            "remote_pid": pid,
        },
        "repo_root": str(REPO_ROOT),
        "local_data_dir": str(LOCAL_DATA_DIR),
        "s3_collection": {
            "bucket": config.s3_bucket,
            "prefix": config.s3_prefix,
            "collect_paths": list(config.collect_paths),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def parse_args() -> LaunchConfig:
    """Parse CLI arguments into one immutable launch config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="AWS CLI profile to use")
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS region")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE, help="EC2 instance type")
    parser.add_argument(
        "--volume-size-gb", type=int, default=DEFAULT_VOLUME_SIZE_GB, help="Root EBS volume size"
    )
    parser.add_argument("--key-name", default=DEFAULT_KEY_NAME, help="EC2 key pair name")
    parser.add_argument(
        "--key-dir", type=Path, default=DEFAULT_KEY_DIR, help="Local directory for PEM files"
    )
    parser.add_argument(
        "--iam-instance-profile",
        default=DEFAULT_INSTANCE_PROFILE,
        help="EC2 instance profile name used for S3 access",
    )
    parser.add_argument(
        "--security-group-name", default=DEFAULT_SECURITY_GROUP, help="SSH security group name"
    )
    parser.add_argument("--instance-name", default=DEFAULT_INSTANCE_NAME, help="EC2 Name tag")
    parser.add_argument("--remote-user", default=DEFAULT_REMOTE_USER, help="Remote SSH username")
    parser.add_argument(
        "--sync-local-data",
        action="store_true",
        help="Also sync ../data to ~/data on the instance",
    )
    parser.add_argument(
        "--spot",
        action="store_true",
        help="Launch the instance as a one-time spot request",
    )
    parser.add_argument(
        "--s3-bucket",
        default=None,
        help="S3 bucket for collected artifacts (default: citrees-{account_id})",
    )
    parser.add_argument(
        "--s3-prefix",
        default=DEFAULT_S3_PREFIX,
        help="S3 prefix under the bucket for collected artifacts",
    )
    parser.add_argument(
        "--collect-path",
        action="append",
        default=["paper/results"],
        help="Repo-relative path to sync to S3 on exit. Pass multiple times for multiple paths.",
    )
    parser.add_argument(
        "--shutdown-when-done",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shutdown the host after the remote commands finish",
    )
    parser.add_argument(
        "--command",
        action="append",
        required=True,
        help="Remote shell command to run after setup. Pass multiple times for multiple commands.",
    )
    args = parser.parse_args()
    return LaunchConfig(
        profile=args.profile,
        region=args.region,
        instance_type=args.instance_type,
        volume_size_gb=args.volume_size_gb,
        key_name=args.key_name,
        key_dir=args.key_dir.expanduser(),
        iam_instance_profile=args.iam_instance_profile,
        security_group_name=args.security_group_name,
        instance_name=args.instance_name,
        remote_user=args.remote_user,
        sync_local_data=bool(args.sync_local_data),
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix.rstrip("/"),
        collect_paths=tuple(dict.fromkeys(args.collect_path)),
        commands=tuple(args.command),
        spot=bool(args.spot),
        shutdown_when_done=bool(args.shutdown_when_done),
    )


def main() -> None:
    """Launch the EC2 host, sync the workspace, and start the remote job."""
    for tool in ("aws", "ssh", "scp"):
        _require_local_tool(tool)

    config = parse_args()
    if config.s3_bucket is None:
        config = replace(config, s3_bucket=_default_s3_bucket(config))
    ssh_cidr = _current_public_cidr()
    key_path = _ensure_key_pair(config)
    vpc_id = _default_vpc_id(config)
    subnet_id = _default_subnet_id(config, vpc_id)
    security_group_id = _ensure_security_group(config, vpc_id, ssh_cidr)
    ami_id = _latest_al2023_ami(config)
    instance_id = _launch_instance(
        config,
        subnet_id=subnet_id,
        security_group_id=security_group_id,
        key_name=config.key_name,
        ami_id=ami_id,
    )
    details = _wait_for_instance(config, instance_id)
    details = InstanceDetails(
        instance_id=details.instance_id,
        public_ip=details.public_ip,
        public_dns=details.public_dns,
        key_path=key_path,
        remote_user=details.remote_user,
    )
    _wait_for_ssh(details)
    _sync_project(details, config)
    remote_pid = _start_remote_job(details, config)
    manifest_path = _write_manifest(config, details, remote_pid)

    print(f"instance_id={details.instance_id}")
    print(f"public_ip={details.public_ip}")
    print(f"public_dns={details.public_dns}")
    print(f"remote_pid={remote_pid}")
    print(f"key_path={details.key_path}")
    print(f"manifest={manifest_path}")
    print(f"s3_bucket={config.s3_bucket}")
    print(f"s3_prefix={config.s3_prefix}/{details.instance_id}")
    print()
    print("Tail the remote log with:")
    print(
        " ".join(
            [
                "ssh",
                *SSH_OPTIONS,
                "-i",
                str(details.key_path),
                f"{details.remote_user}@{details.public_ip}",
                shlex.quote("tail -f ~/logs/citrees-run.log"),
            ]
        )
    )


if __name__ == "__main__":
    main()
