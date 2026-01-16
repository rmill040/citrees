"""Configuration loading, validation, and resource naming."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import boto3
import yaml


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class VPCConfig:
    """VPC configuration."""

    vpc_id: str = "create"
    cidr_block: str = "10.0.0.0/16"
    private_subnet_cidr: str = "10.0.1.0/24"
    public_subnet_cidr: str = "10.0.2.0/24"
    availability_zone: str = "us-east-1a"


@dataclass
class DynamoDBConfig:
    """DynamoDB configuration."""

    table_name: str | None = None


@dataclass
class S3Config:
    """S3 configuration."""

    bucket_name: str | None = None


@dataclass
class ECRConfig:
    """ECR configuration."""

    repository_name: str | None = None
    image_tag: str = "latest"


@dataclass
class ServerConfig:
    """Server instance configuration."""

    instance_type: str = "c7i.2xlarge"
    spot: bool = False


@dataclass
class WorkersConfig:
    """Worker instances configuration."""

    instance_type: str = "c7i.8xlarge"
    spot: bool = True
    count: int = 5


@dataclass
class IAMConfig:
    """IAM configuration."""

    role_name: str | None = None


@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    type: Literal["classification", "regression"] = "classification"
    n_seeds: int = 10
    stale_timeout_minutes: int = 30
    # Ray CPU scheduling for experiment tasks
    selection_cpus_default: int = 1
    selection_cpus_threaded: int = 8
    selection_cpus_cif: int = 16
    selection_cpus_cif_large: int = 32
    selection_cif_large_threshold: int = 10_000_000
    selection_cpus_overrides: dict[str, int] = field(default_factory=dict)


@dataclass
class Config:
    """Complete infrastructure configuration."""

    project_name: str = "citrees"
    region: str = "us-east-1"
    ami_id: str | None = None

    vpc: VPCConfig = field(default_factory=VPCConfig)
    dynamodb: DynamoDBConfig = field(default_factory=DynamoDBConfig)
    s3: S3Config = field(default_factory=S3Config)
    ecr: ECRConfig = field(default_factory=ECRConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    workers: WorkersConfig = field(default_factory=WorkersConfig)
    iam: IAMConfig = field(default_factory=IAMConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # Resolved values (populated after setup)
    _account_id: str | None = field(default=None, repr=False)
    _resolved_vpc_id: str | None = field(default=None, repr=False)
    _resolved_private_subnet_id: str | None = field(default=None, repr=False)
    _resolved_public_subnet_id: str | None = field(default=None, repr=False)
    _resolved_security_group_id: str | None = field(default=None, repr=False)
    _resolved_nat_gateway_id: str | None = field(default=None, repr=False)
    _resolved_internet_gateway_id: str | None = field(default=None, repr=False)
    _resolved_route_table_id: str | None = field(default=None, repr=False)
    _server_instance_id: str | None = field(default=None, repr=False)
    _server_private_ip: str | None = field(default=None, repr=False)
    _worker_instance_ids: list[str] = field(default_factory=list, repr=False)

    # ==========================================================================
    # Account ID
    # ==========================================================================

    @property
    def account_id(self) -> str:
        """Get AWS account ID, fetching from STS if not cached."""
        if self._account_id is None:
            sts = boto3.client("sts", region_name=self.region)
            self._account_id = sts.get_caller_identity()["Account"]
        return self._account_id

    # ==========================================================================
    # Resource Names (auto-generated)
    # ==========================================================================

    @property
    def vpc_name(self) -> str:
        """VPC name tag."""
        return f"{self.project_name}-vpc-{self.account_id}"

    @property
    def private_subnet_name(self) -> str:
        """Private subnet name tag."""
        return f"{self.project_name}-private-subnet-{self.account_id}"

    @property
    def public_subnet_name(self) -> str:
        """Public subnet name tag."""
        return f"{self.project_name}-public-subnet-{self.account_id}"

    @property
    def internet_gateway_name(self) -> str:
        """Internet gateway name tag."""
        return f"{self.project_name}-igw-{self.account_id}"

    @property
    def nat_gateway_name(self) -> str:
        """NAT gateway name tag."""
        return f"{self.project_name}-nat-{self.account_id}"

    @property
    def security_group_name(self) -> str:
        """Security group name."""
        return f"{self.project_name}-sg-{self.account_id}"

    @property
    def route_table_name(self) -> str:
        """Route table name tag."""
        return f"{self.project_name}-rt-{self.account_id}"

    @property
    def table_name(self) -> str:
        """DynamoDB table name."""
        return self.dynamodb.table_name or f"{self.project_name}-results-{self.account_id}"

    @property
    def bucket_name(self) -> str:
        """S3 bucket name (may call STS if not configured)."""
        return self.s3.bucket_name or f"{self.project_name}-results-{self.account_id}"

    @property
    def s3_bucket(self) -> str:
        """S3 bucket - reads from S3_BUCKET env var first (set by cluster setup), avoids STS calls on workers."""
        return os.environ.get("S3_BUCKET") or self.bucket_name

    @property
    def ecr_repository_name(self) -> str:
        """ECR repository name."""
        return self.ecr.repository_name or f"{self.project_name}-{self.account_id}"

    @property
    def ecr_image_uri(self) -> str:
        """Full ECR image URI."""
        return f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/{self.ecr_repository_name}:{self.ecr.image_tag}"

    @property
    def iam_role_name(self) -> str:
        """IAM role name."""
        return self.iam.role_name or f"{self.project_name}-worker-role-{self.account_id}"

    @property
    def iam_instance_profile_name(self) -> str:
        """IAM instance profile name."""
        return f"{self.project_name}-instance-profile-{self.account_id}"

    @property
    def server_name(self) -> str:
        """Server instance name tag."""
        return f"{self.project_name}-server-{self.account_id}"

    @property
    def worker_name(self) -> str:
        """Worker instance name tag."""
        return f"{self.project_name}-worker-{self.account_id}"

    # ==========================================================================
    # IAM Policy Document
    # ==========================================================================

    @property
    def iam_policy_document(self) -> dict:
        """Generate IAM policy scoped to citrees resources."""
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "DynamoDB",
                    "Effect": "Allow",
                    "Action": [
                        "dynamodb:GetItem",
                        "dynamodb:PutItem",
                        "dynamodb:UpdateItem",
                        "dynamodb:DeleteItem",
                        "dynamodb:Query",
                        "dynamodb:Scan",
                    ],
                    "Resource": f"arn:aws:dynamodb:{self.region}:{self.account_id}:table/{self.project_name}-*",
                },
                {
                    "Sid": "S3",
                    "Effect": "Allow",
                    "Action": [
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:DeleteObject",
                        "s3:ListBucket",
                    ],
                    "Resource": [
                        f"arn:aws:s3:::{self.project_name}-*",
                        f"arn:aws:s3:::{self.project_name}-*/*",
                    ],
                },
                {
                    "Sid": "ECR",
                    "Effect": "Allow",
                    "Action": [
                        "ecr:GetAuthorizationToken",
                        "ecr:BatchGetImage",
                        "ecr:GetDownloadUrlForLayer",
                    ],
                    "Resource": "*",
                },
                {
                    "Sid": "SSM",
                    "Effect": "Allow",
                    "Action": [
                        "ssm:UpdateInstanceInformation",
                        "ssmmessages:CreateControlChannel",
                        "ssmmessages:CreateDataChannel",
                        "ssmmessages:OpenControlChannel",
                        "ssmmessages:OpenDataChannel",
                        "ec2messages:AcknowledgeMessage",
                        "ec2messages:DeleteMessage",
                        "ec2messages:FailMessage",
                        "ec2messages:GetEndpoint",
                        "ec2messages:GetMessages",
                        "ec2messages:SendReply",
                    ],
                    "Resource": "*",
                },
            ],
        }

    @property
    def iam_trust_policy(self) -> dict:
        """IAM trust policy for EC2."""
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "ec2.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }

    # ==========================================================================
    # State File
    # ==========================================================================

    def save_state(self, path: Path) -> None:
        """Save resolved state to file."""
        state = {
            "account_id": self._account_id,
            "vpc_id": self._resolved_vpc_id,
            "private_subnet_id": self._resolved_private_subnet_id,
            "public_subnet_id": self._resolved_public_subnet_id,
            "security_group_id": self._resolved_security_group_id,
            "nat_gateway_id": self._resolved_nat_gateway_id,
            "internet_gateway_id": self._resolved_internet_gateway_id,
            "route_table_id": self._resolved_route_table_id,
            "server_instance_id": self._server_instance_id,
            "server_private_ip": self._server_private_ip,
            "worker_instance_ids": self._worker_instance_ids,
        }
        path.write_text(yaml.dump(state, default_flow_style=False))

    def load_state(self, path: Path) -> None:
        """Load resolved state from file."""
        if not path.exists():
            return
        state = yaml.safe_load(path.read_text())
        if state:
            self._account_id = state.get("account_id")
            self._resolved_vpc_id = state.get("vpc_id")
            self._resolved_private_subnet_id = state.get("private_subnet_id")
            self._resolved_public_subnet_id = state.get("public_subnet_id")
            self._resolved_security_group_id = state.get("security_group_id")
            self._resolved_nat_gateway_id = state.get("nat_gateway_id")
            self._resolved_internet_gateway_id = state.get("internet_gateway_id")
            self._resolved_route_table_id = state.get("route_table_id")
            self._server_instance_id = state.get("server_instance_id")
            self._server_private_ip = state.get("server_private_ip")
            self._worker_instance_ids = state.get("worker_instance_ids", [])


# =============================================================================
# Loading
# =============================================================================


def load_config(path: Path | None = None) -> Config:
    """Load configuration from YAML file.

    Parameters
    ----------
    path : Path | None
        Path to config file. If None, uses default config.yaml in same directory.

    Returns
    -------
    Config
        Loaded and validated configuration.
    """
    if path is None:
        path = Path(__file__).parent / "config.yaml"

    if not path.exists():
        return Config()

    data = yaml.safe_load(path.read_text())
    if not data:
        return Config()

    # Build config from YAML
    config = Config(
        project_name=data.get("project", {}).get("name", "citrees"),
        region=data.get("region", "us-east-1"),
        ami_id=data.get("ami_id"),
    )

    # VPC
    if vpc_data := data.get("vpc"):
        config.vpc = VPCConfig(
            vpc_id=vpc_data.get("vpc_id", "create"),
            cidr_block=vpc_data.get("cidr_block", "10.0.0.0/16"),
            private_subnet_cidr=vpc_data.get("private_subnet_cidr", "10.0.1.0/24"),
            public_subnet_cidr=vpc_data.get("public_subnet_cidr", "10.0.2.0/24"),
            availability_zone=vpc_data.get("availability_zone", "us-east-1a"),
        )

    # DynamoDB
    if ddb_data := data.get("dynamodb"):
        config.dynamodb = DynamoDBConfig(table_name=ddb_data.get("table_name"))

    # S3
    if s3_data := data.get("s3"):
        config.s3 = S3Config(bucket_name=s3_data.get("bucket_name"))

    # ECR
    if ecr_data := data.get("ecr"):
        config.ecr = ECRConfig(
            repository_name=ecr_data.get("repository_name"),
            image_tag=ecr_data.get("image_tag", "latest"),
        )

    # Server
    if server_data := data.get("server"):
        config.server = ServerConfig(
            instance_type=server_data.get("instance_type", "c7i.2xlarge"),
            spot=server_data.get("spot", False),
        )

    # Workers
    if workers_data := data.get("workers"):
        config.workers = WorkersConfig(
            instance_type=workers_data.get("instance_type", "c7i.8xlarge"),
            spot=workers_data.get("spot", True),
            count=workers_data.get("count", 5),
        )

    # IAM
    if iam_data := data.get("iam"):
        config.iam = IAMConfig(role_name=iam_data.get("role_name"))

    # Experiment
    if exp_data := data.get("experiment"):
        config.experiment = ExperimentConfig(
            type=exp_data.get("type", "classification"),
            n_seeds=exp_data.get("n_seeds", 10),
            stale_timeout_minutes=exp_data.get("stale_timeout_minutes", 30),
            selection_cpus_default=exp_data.get("selection_cpus_default", 1),
            selection_cpus_threaded=exp_data.get("selection_cpus_threaded", 8),
            selection_cpus_cif=exp_data.get("selection_cpus_cif", 16),
            selection_cpus_cif_large=exp_data.get("selection_cpus_cif_large", 32),
            selection_cif_large_threshold=exp_data.get("selection_cif_large_threshold", 10_000_000),
        )

    # Load state if exists
    state_path = path.parent / ".citrees-state.yaml"
    config.load_state(state_path)

    return config
