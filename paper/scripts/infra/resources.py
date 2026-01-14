"""AWS resource management for citrees infrastructure."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from rich.console import Console

from .config import Config

console = Console()


# =============================================================================
# VPC Resources
# =============================================================================


def create_vpc(config: Config) -> str:
    """Create VPC with tags.

    Returns
    -------
    str
        VPC ID.
    """
    ec2 = boto3.client("ec2", region_name=config.region)

    response = ec2.create_vpc(
        CidrBlock=config.vpc.cidr_block,
        TagSpecifications=[
            {
                "ResourceType": "vpc",
                "Tags": [
                    {"Key": "Name", "Value": config.vpc_name},
                    {"Key": "Project", "Value": config.project_name},
                ],
            }
        ],
    )
    vpc_id = response["Vpc"]["VpcId"]

    # Enable DNS hostnames
    ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsHostnames={"Value": True})

    # Wait for VPC to be available
    waiter = ec2.get_waiter("vpc_available")
    waiter.wait(VpcIds=[vpc_id])

    config._resolved_vpc_id = vpc_id
    return vpc_id


def create_internet_gateway(config: Config) -> str:
    """Create and attach internet gateway.

    Returns
    -------
    str
        Internet gateway ID.
    """
    ec2 = boto3.client("ec2", region_name=config.region)

    response = ec2.create_internet_gateway(
        TagSpecifications=[
            {
                "ResourceType": "internet-gateway",
                "Tags": [
                    {"Key": "Name", "Value": config.internet_gateway_name},
                    {"Key": "Project", "Value": config.project_name},
                ],
            }
        ],
    )
    igw_id = response["InternetGateway"]["InternetGatewayId"]

    # Attach to VPC
    ec2.attach_internet_gateway(InternetGatewayId=igw_id, VpcId=config._resolved_vpc_id)

    config._resolved_internet_gateway_id = igw_id
    return igw_id


def create_public_subnet(config: Config) -> str:
    """Create public subnet for NAT gateway.

    Returns
    -------
    str
        Subnet ID.
    """
    ec2 = boto3.client("ec2", region_name=config.region)

    response = ec2.create_subnet(
        VpcId=config._resolved_vpc_id,
        CidrBlock=config.vpc.public_subnet_cidr,
        AvailabilityZone=config.vpc.availability_zone,
        TagSpecifications=[
            {
                "ResourceType": "subnet",
                "Tags": [
                    {"Key": "Name", "Value": config.public_subnet_name},
                    {"Key": "Project", "Value": config.project_name},
                ],
            }
        ],
    )
    subnet_id = response["Subnet"]["SubnetId"]

    # Create route table for public subnet
    rt_response = ec2.create_route_table(
        VpcId=config._resolved_vpc_id,
        TagSpecifications=[
            {
                "ResourceType": "route-table",
                "Tags": [
                    {"Key": "Name", "Value": f"{config.project_name}-public-rt-{config.account_id}"},
                    {"Key": "Project", "Value": config.project_name},
                ],
            }
        ],
    )
    public_rt_id = rt_response["RouteTable"]["RouteTableId"]

    # Add route to internet gateway
    ec2.create_route(
        RouteTableId=public_rt_id,
        DestinationCidrBlock="0.0.0.0/0",
        GatewayId=config._resolved_internet_gateway_id,
    )

    # Associate route table with subnet
    ec2.associate_route_table(RouteTableId=public_rt_id, SubnetId=subnet_id)

    config._resolved_public_subnet_id = subnet_id
    return subnet_id


def create_nat_gateway(config: Config) -> str:
    """Create NAT gateway in public subnet.

    Returns
    -------
    str
        NAT gateway ID.
    """
    ec2 = boto3.client("ec2", region_name=config.region)

    # Allocate Elastic IP
    eip_response = ec2.allocate_address(
        Domain="vpc",
        TagSpecifications=[
            {
                "ResourceType": "elastic-ip",
                "Tags": [
                    {"Key": "Name", "Value": f"{config.project_name}-nat-eip-{config.account_id}"},
                    {"Key": "Project", "Value": config.project_name},
                ],
            }
        ],
    )
    allocation_id = eip_response["AllocationId"]

    # Create NAT gateway
    response = ec2.create_nat_gateway(
        SubnetId=config._resolved_public_subnet_id,
        AllocationId=allocation_id,
        TagSpecifications=[
            {
                "ResourceType": "natgateway",
                "Tags": [
                    {"Key": "Name", "Value": config.nat_gateway_name},
                    {"Key": "Project", "Value": config.project_name},
                ],
            }
        ],
    )
    nat_id = response["NatGateway"]["NatGatewayId"]

    # Wait for NAT gateway to be available
    console.print("  Waiting for NAT gateway (this takes ~2 minutes)...", style="dim")
    waiter = ec2.get_waiter("nat_gateway_available")
    waiter.wait(NatGatewayIds=[nat_id])

    config._resolved_nat_gateway_id = nat_id
    return nat_id


def create_private_subnet(config: Config) -> str:
    """Create private subnet for EC2 instances.

    Returns
    -------
    str
        Subnet ID.
    """
    ec2 = boto3.client("ec2", region_name=config.region)

    response = ec2.create_subnet(
        VpcId=config._resolved_vpc_id,
        CidrBlock=config.vpc.private_subnet_cidr,
        AvailabilityZone=config.vpc.availability_zone,
        TagSpecifications=[
            {
                "ResourceType": "subnet",
                "Tags": [
                    {"Key": "Name", "Value": config.private_subnet_name},
                    {"Key": "Project", "Value": config.project_name},
                ],
            }
        ],
    )
    subnet_id = response["Subnet"]["SubnetId"]

    # Create route table for private subnet
    rt_response = ec2.create_route_table(
        VpcId=config._resolved_vpc_id,
        TagSpecifications=[
            {
                "ResourceType": "route-table",
                "Tags": [
                    {"Key": "Name", "Value": config.route_table_name},
                    {"Key": "Project", "Value": config.project_name},
                ],
            }
        ],
    )
    rt_id = rt_response["RouteTable"]["RouteTableId"]

    # Add route to NAT gateway
    ec2.create_route(
        RouteTableId=rt_id,
        DestinationCidrBlock="0.0.0.0/0",
        NatGatewayId=config._resolved_nat_gateway_id,
    )

    # Associate route table with subnet
    ec2.associate_route_table(RouteTableId=rt_id, SubnetId=subnet_id)

    config._resolved_private_subnet_id = subnet_id
    config._resolved_route_table_id = rt_id
    return subnet_id


def create_security_group(config: Config) -> str:
    """Create security group for EC2 instances.

    Returns
    -------
    str
        Security group ID.
    """
    ec2 = boto3.client("ec2", region_name=config.region)

    response = ec2.create_security_group(
        GroupName=config.security_group_name,
        Description=f"{config.project_name} distributed experiments",
        VpcId=config._resolved_vpc_id,
        TagSpecifications=[
            {
                "ResourceType": "security-group",
                "Tags": [
                    {"Key": "Name", "Value": config.security_group_name},
                    {"Key": "Project", "Value": config.project_name},
                ],
            }
        ],
    )
    sg_id = response["GroupId"]

    # Allow port 8000 within VPC (server-worker communication)
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 8000,
                "ToPort": 8000,
                "IpRanges": [{"CidrIp": config.vpc.cidr_block, "Description": "Server API"}],
            }
        ],
    )

    # Allow all outbound (default, but explicit)
    # Already allowed by default

    config._resolved_security_group_id = sg_id
    return sg_id


def setup_vpc(config: Config) -> None:
    """Set up complete VPC infrastructure.

    Creates VPC, subnets, internet gateway, NAT gateway, and security group.
    """
    if config.vpc.vpc_id != "create":
        # Using existing VPC - just need security group
        config._resolved_vpc_id = config.vpc.vpc_id
        console.print(f"  Using existing VPC: {config.vpc.vpc_id}", style="dim")

        # Still need to create security group in existing VPC
        sg_id = create_security_group(config)
        console.print(f"  Created security group: {sg_id}")
        return

    console.print("  Creating VPC...")
    vpc_id = create_vpc(config)
    console.print(f"  VPC: {vpc_id}")

    console.print("  Creating internet gateway...")
    igw_id = create_internet_gateway(config)
    console.print(f"  Internet gateway: {igw_id}")

    console.print("  Creating public subnet...")
    public_subnet_id = create_public_subnet(config)
    console.print(f"  Public subnet: {public_subnet_id}")

    console.print("  Creating NAT gateway...")
    nat_id = create_nat_gateway(config)
    console.print(f"  NAT gateway: {nat_id}")

    console.print("  Creating private subnet...")
    private_subnet_id = create_private_subnet(config)
    console.print(f"  Private subnet: {private_subnet_id}")

    console.print("  Creating security group...")
    sg_id = create_security_group(config)
    console.print(f"  Security group: {sg_id}")


# =============================================================================
# DynamoDB
# =============================================================================


def create_dynamodb_table(config: Config) -> str:
    """Create DynamoDB table for experiment tracking.

    Returns
    -------
    str
        Table name.
    """
    dynamodb = boto3.client("dynamodb", region_name=config.region)

    try:
        dynamodb.create_table(
            TableName=config.table_name,
            KeySchema=[{"AttributeName": "config_idx", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "config_idx", "AttributeType": "N"}],
            BillingMode="PAY_PER_REQUEST",
            Tags=[
                {"Key": "Name", "Value": config.table_name},
                {"Key": "Project", "Value": config.project_name},
            ],
        )

        # Wait for table to be active
        waiter = dynamodb.get_waiter("table_exists")
        waiter.wait(TableName=config.table_name)

    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            console.print(f"  Table already exists: {config.table_name}", style="dim")
        else:
            raise

    return config.table_name


# =============================================================================
# S3
# =============================================================================


def create_s3_bucket(config: Config) -> str:
    """Create S3 bucket for results.

    Returns
    -------
    str
        Bucket name.
    """
    s3 = boto3.client("s3", region_name=config.region)

    try:
        # us-east-1 doesn't need LocationConstraint
        if config.region == "us-east-1":
            s3.create_bucket(Bucket=config.bucket_name)
        else:
            s3.create_bucket(
                Bucket=config.bucket_name,
                CreateBucketConfiguration={"LocationConstraint": config.region},
            )

        # Add tags
        s3.put_bucket_tagging(
            Bucket=config.bucket_name,
            Tagging={
                "TagSet": [
                    {"Key": "Name", "Value": config.bucket_name},
                    {"Key": "Project", "Value": config.project_name},
                ]
            },
        )

    except ClientError as e:
        if e.response["Error"]["Code"] in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            console.print(f"  Bucket already exists: {config.bucket_name}", style="dim")
        else:
            raise

    return config.bucket_name


# =============================================================================
# ECR
# =============================================================================


def create_ecr_repository(config: Config) -> str:
    """Create ECR repository.

    Returns
    -------
    str
        Repository URI.
    """
    ecr = boto3.client("ecr", region_name=config.region)

    try:
        response = ecr.create_repository(
            repositoryName=config.ecr_repository_name,
            imageScanningConfiguration={"scanOnPush": False},
            tags=[
                {"Key": "Name", "Value": config.ecr_repository_name},
                {"Key": "Project", "Value": config.project_name},
            ],
        )
        repo_uri = response["repository"]["repositoryUri"]

    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryAlreadyExistsException":
            console.print(f"  Repository already exists: {config.ecr_repository_name}", style="dim")
            # Get existing repo URI
            response = ecr.describe_repositories(repositoryNames=[config.ecr_repository_name])
            repo_uri = response["repositories"][0]["repositoryUri"]
        else:
            raise

    return repo_uri


def build_and_push_image(config: Config) -> None:
    """Build Docker image and push to ECR."""
    # Find project root (where Dockerfile is)
    project_root = Path(__file__).resolve().parents[3]
    dockerfile_path = project_root / "Dockerfile"

    if not dockerfile_path.exists():
        raise FileNotFoundError(f"Dockerfile not found at {dockerfile_path}")

    ecr_uri = f"{config.account_id}.dkr.ecr.{config.region}.amazonaws.com"
    image_uri = config.ecr_image_uri

    # ECR login
    console.print("  Logging into ECR...")
    login_cmd = f"aws ecr get-login-password --region {config.region} | docker login --username AWS --password-stdin {ecr_uri}"
    subprocess.run(login_cmd, shell=True, check=True, capture_output=True)

    # Build image
    console.print("  Building Docker image...")
    build_cmd = f"docker build -t {image_uri} {project_root}"
    subprocess.run(build_cmd, shell=True, check=True)

    # Push image
    console.print("  Pushing image to ECR...")
    push_cmd = f"docker push {image_uri}"
    subprocess.run(push_cmd, shell=True, check=True)

    console.print(f"  Image: {image_uri}")


# =============================================================================
# IAM
# =============================================================================


def create_iam_role(config: Config) -> str:
    """Create IAM role and instance profile.

    Returns
    -------
    str
        Role name.
    """
    iam = boto3.client("iam", region_name=config.region)

    # Create role
    try:
        iam.create_role(
            RoleName=config.iam_role_name,
            AssumeRolePolicyDocument=json.dumps(config.iam_trust_policy),
            Description=f"{config.project_name} EC2 worker role",
            Tags=[
                {"Key": "Name", "Value": config.iam_role_name},
                {"Key": "Project", "Value": config.project_name},
            ],
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            console.print(f"  Role already exists: {config.iam_role_name}", style="dim")
        else:
            raise

    # Attach inline policy
    policy_name = f"{config.project_name}-policy-{config.account_id}"
    try:
        iam.put_role_policy(
            RoleName=config.iam_role_name,
            PolicyName=policy_name,
            PolicyDocument=json.dumps(config.iam_policy_document),
        )
    except ClientError:
        pass  # Policy update is fine

    # Create instance profile
    try:
        iam.create_instance_profile(
            InstanceProfileName=config.iam_instance_profile_name,
            Tags=[
                {"Key": "Name", "Value": config.iam_instance_profile_name},
                {"Key": "Project", "Value": config.project_name},
            ],
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            console.print(f"  Instance profile already exists: {config.iam_instance_profile_name}", style="dim")
        else:
            raise

    # Add role to instance profile
    try:
        iam.add_role_to_instance_profile(
            InstanceProfileName=config.iam_instance_profile_name,
            RoleName=config.iam_role_name,
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "LimitExceeded":
            pass  # Role already attached
        else:
            raise

    # Wait for instance profile to propagate
    time.sleep(10)

    return config.iam_role_name


# =============================================================================
# AMI
# =============================================================================


def get_latest_ami(config: Config) -> str:
    """Get latest Amazon Linux 2023 AMI ID.

    Returns
    -------
    str
        AMI ID.
    """
    if config.ami_id:
        return config.ami_id

    ec2 = boto3.client("ec2", region_name=config.region)

    response = ec2.describe_images(
        Owners=["amazon"],
        Filters=[
            {"Name": "name", "Values": ["al2023-ami-*-x86_64"]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": ["x86_64"]},
        ],
    )

    # Sort by creation date and get latest
    images = sorted(response["Images"], key=lambda x: x["CreationDate"], reverse=True)
    if not images:
        raise RuntimeError("No Amazon Linux 2023 AMI found")

    return images[0]["ImageId"]


# =============================================================================
# Teardown
# =============================================================================


def teardown_instances(config: Config) -> None:
    """Terminate all EC2 instances."""
    ec2 = boto3.client("ec2", region_name=config.region)

    instance_ids = []
    if config._server_instance_id:
        instance_ids.append(config._server_instance_id)
    instance_ids.extend(config._worker_instance_ids)

    if instance_ids:
        console.print(f"  Terminating {len(instance_ids)} instance(s)...")
        ec2.terminate_instances(InstanceIds=instance_ids)

        # Wait for termination
        waiter = ec2.get_waiter("instance_terminated")
        waiter.wait(InstanceIds=instance_ids)

        config._server_instance_id = None
        config._server_private_ip = None
        config._worker_instance_ids = []


def teardown_vpc(config: Config) -> None:
    """Delete VPC and all associated resources."""
    if not config._resolved_vpc_id:
        return

    if config.vpc.vpc_id != "create":
        # Don't delete existing VPC, just security group
        if config._resolved_security_group_id:
            ec2 = boto3.client("ec2", region_name=config.region)
            try:
                ec2.delete_security_group(GroupId=config._resolved_security_group_id)
                console.print(f"  Deleted security group: {config._resolved_security_group_id}")
            except ClientError:
                pass
        return

    ec2 = boto3.client("ec2", region_name=config.region)

    # Delete NAT gateway
    if config._resolved_nat_gateway_id:
        console.print(f"  Deleting NAT gateway: {config._resolved_nat_gateway_id}")
        ec2.delete_nat_gateway(NatGatewayId=config._resolved_nat_gateway_id)

        # Wait for NAT gateway deletion
        console.print("  Waiting for NAT gateway deletion...")
        while True:
            try:
                response = ec2.describe_nat_gateways(NatGatewayIds=[config._resolved_nat_gateway_id])
                state = response["NatGateways"][0]["State"]
                if state == "deleted":
                    break
                time.sleep(10)
            except ClientError:
                break

    # Release Elastic IPs
    console.print("  Releasing Elastic IPs...")
    eips = ec2.describe_addresses(
        Filters=[{"Name": "tag:Project", "Values": [config.project_name]}]
    )
    for eip in eips.get("Addresses", []):
        try:
            ec2.release_address(AllocationId=eip["AllocationId"])
        except ClientError:
            pass

    # Delete subnets
    for subnet_id in [config._resolved_private_subnet_id, config._resolved_public_subnet_id]:
        if subnet_id:
            try:
                ec2.delete_subnet(SubnetId=subnet_id)
                console.print(f"  Deleted subnet: {subnet_id}")
            except ClientError:
                pass

    # Delete route tables (except main)
    route_tables = ec2.describe_route_tables(
        Filters=[{"Name": "vpc-id", "Values": [config._resolved_vpc_id]}]
    )
    for rt in route_tables.get("RouteTables", []):
        # Skip main route table
        if any(assoc.get("Main") for assoc in rt.get("Associations", [])):
            continue
        try:
            # Disassociate first
            for assoc in rt.get("Associations", []):
                if assoc.get("RouteTableAssociationId"):
                    ec2.disassociate_route_table(AssociationId=assoc["RouteTableAssociationId"])
            ec2.delete_route_table(RouteTableId=rt["RouteTableId"])
            console.print(f"  Deleted route table: {rt['RouteTableId']}")
        except ClientError:
            pass

    # Delete security group
    if config._resolved_security_group_id:
        try:
            ec2.delete_security_group(GroupId=config._resolved_security_group_id)
            console.print(f"  Deleted security group: {config._resolved_security_group_id}")
        except ClientError:
            pass

    # Detach and delete internet gateway
    if config._resolved_internet_gateway_id:
        try:
            ec2.detach_internet_gateway(
                InternetGatewayId=config._resolved_internet_gateway_id,
                VpcId=config._resolved_vpc_id,
            )
            ec2.delete_internet_gateway(InternetGatewayId=config._resolved_internet_gateway_id)
            console.print(f"  Deleted internet gateway: {config._resolved_internet_gateway_id}")
        except ClientError:
            pass

    # Delete VPC
    try:
        ec2.delete_vpc(VpcId=config._resolved_vpc_id)
        console.print(f"  Deleted VPC: {config._resolved_vpc_id}")
    except ClientError as e:
        console.print(f"  [yellow]Could not delete VPC: {e}[/yellow]")


def teardown_storage(config: Config, delete_data: bool = False) -> None:
    """Delete DynamoDB table and optionally S3 bucket."""
    # DynamoDB
    dynamodb = boto3.client("dynamodb", region_name=config.region)
    try:
        dynamodb.delete_table(TableName=config.table_name)
        console.print(f"  Deleted DynamoDB table: {config.table_name}")
    except ClientError:
        pass

    # S3 (only if delete_data is True)
    if delete_data:
        s3 = boto3.resource("s3", region_name=config.region)
        try:
            bucket = s3.Bucket(config.bucket_name)
            bucket.objects.all().delete()
            bucket.delete()
            console.print(f"  Deleted S3 bucket: {config.bucket_name}")
        except ClientError:
            pass


def teardown_ecr(config: Config) -> None:
    """Delete ECR repository."""
    ecr = boto3.client("ecr", region_name=config.region)
    try:
        ecr.delete_repository(repositoryName=config.ecr_repository_name, force=True)
        console.print(f"  Deleted ECR repository: {config.ecr_repository_name}")
    except ClientError:
        pass


def teardown_iam(config: Config) -> None:
    """Delete IAM role and instance profile."""
    iam = boto3.client("iam", region_name=config.region)

    # Remove role from instance profile
    try:
        iam.remove_role_from_instance_profile(
            InstanceProfileName=config.iam_instance_profile_name,
            RoleName=config.iam_role_name,
        )
    except ClientError:
        pass

    # Delete instance profile
    try:
        iam.delete_instance_profile(InstanceProfileName=config.iam_instance_profile_name)
        console.print(f"  Deleted instance profile: {config.iam_instance_profile_name}")
    except ClientError:
        pass

    # Delete inline policy
    policy_name = f"{config.project_name}-policy-{config.account_id}"
    try:
        iam.delete_role_policy(RoleName=config.iam_role_name, PolicyName=policy_name)
    except ClientError:
        pass

    # Delete role
    try:
        iam.delete_role(RoleName=config.iam_role_name)
        console.print(f"  Deleted IAM role: {config.iam_role_name}")
    except ClientError:
        pass
