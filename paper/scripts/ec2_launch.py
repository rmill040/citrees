#!/usr/bin/env python3
"""EC2 instance launcher for citrees distributed experiments.

Launches EC2 instances configured to run citrees experiment servers or workers.
Instances are bootstrapped via user data scripts that:
1. Install Python and dependencies
2. Clone the citrees repository
3. Install the package with benchmarking dependencies
4. Start the appropriate server or worker process

Prerequisites:
- AWS credentials configured (aws configure or environment variables)
- IAM role with DynamoDB access for instances
- Security group allowing:
  - SSH (port 22) for debugging
  - Port 8000 (server only, from worker security group)
- Key pair for SSH access

Usage:
    # Launch a server
    python ec2_launch.py server \\
        --table-name ClfFeatureSelection \\
        --instance-type c5.xlarge \\
        --key-name my-keypair \\
        --iam-role citrees-worker-role \\
        --security-group sg-xxx

    # Launch workers
    python ec2_launch.py worker \\
        --server-url http://10.0.0.5:8000 \\
        --table-name ClfFeatureSelection \\
        --instance-type c5.xlarge \\
        --count 5 \\
        --key-name my-keypair \\
        --iam-role citrees-worker-role \\
        --security-group sg-xxx
"""
import argparse
import base64
import sys
import textwrap
from typing import Optional

try:
    import boto3
except ImportError:
    print("boto3 is required. Install with: pip install boto3")
    sys.exit(1)


# Default AMI: Amazon Linux 2023 (update for your region)
# Find latest AMI: aws ec2 describe-images --owners amazon --filters "Name=name,Values=al2023-ami-*-x86_64" --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId'
DEFAULT_AMI = "ami-0c55b159cbfafe1f0"  # Placeholder - override with --ami


def get_server_user_data(
    *,
    table_name: str,
    experiment_type: str,
    region: str,
    repo_url: str,
    branch: str,
) -> str:
    """Generate user data script for server instance."""
    # Determine which server script to run
    if experiment_type == "feature_selection":
        server_script = f"{experiment_type[:3]}_feature_selection_server"
    else:
        server_script = f"{experiment_type[:3]}_cv_server"

    script = textwrap.dedent(f"""\
        #!/bin/bash
        set -ex

        # Log all output
        exec > >(tee /var/log/user-data.log) 2>&1

        echo "=== Starting server setup ==="

        # Install system dependencies
        yum update -y
        yum install -y python3.11 python3.11-pip git

        # Install uv for fast package management
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"

        # Clone repository
        cd /home/ec2-user
        git clone --depth 1 --branch {branch} {repo_url} citrees
        cd citrees

        # Install package with benchmarking dependencies
        uv sync --group paper
        uv pip install ".[bench]"

        # Set environment variables
        export TABLE_NAME={table_name}
        export AWS_DEFAULT_REGION={region}

        # Change to scripts directory and run server
        cd paper/scripts

        echo "=== Starting FastAPI server ==="
        # Run with nohup so it survives SSH disconnection
        nohup uv run uvicorn {server_script}:app --host 0.0.0.0 --port 8000 > /var/log/citrees-server.log 2>&1 &

        echo "=== Server started on port 8000 ==="
    """)
    return script


def get_worker_user_data(
    *,
    server_url: str,
    table_name: str,
    experiment_type: str,
    region: str,
    n_jobs_outer: int,
    n_jobs_inner: int,
    skip: str,
    repo_url: str,
    branch: str,
) -> str:
    """Generate user data script for worker instance."""
    # Determine which worker script to run
    if experiment_type == "feature_selection":
        worker_script = f"{experiment_type[:3]}_feature_selection_worker"
    else:
        worker_script = f"{experiment_type[:3]}_cv_worker"

    script = textwrap.dedent(f"""\
        #!/bin/bash
        set -ex

        # Log all output
        exec > >(tee /var/log/user-data.log) 2>&1

        echo "=== Starting worker setup ==="

        # Install system dependencies
        yum update -y
        yum install -y python3.11 python3.11-pip git

        # Install uv for fast package management
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"

        # Clone repository
        cd /home/ec2-user
        git clone --depth 1 --branch {branch} {repo_url} citrees
        cd citrees

        # Install package with benchmarking dependencies
        uv sync --group paper
        uv pip install ".[bench]"

        # Set environment variables
        export URL={server_url}
        export TABLE_NAME={table_name}
        export AWS_DEFAULT_REGION={region}
        export N_JOBS_OUTER={n_jobs_outer}
        export N_JOBS_INNER={n_jobs_inner}
        export SKIP="{skip}"

        # Change to scripts directory and run worker
        cd paper/scripts

        echo "=== Starting worker ==="
        # Run worker (will exit when no more configs)
        uv run python {worker_script}.py > /var/log/citrees-worker.log 2>&1

        echo "=== Worker completed ==="

        # Optional: terminate instance when done (uncomment if desired)
        # shutdown -h now
    """)
    return script


def launch_instances(
    *,
    ec2_client,
    role: str,
    count: int,
    instance_type: str,
    ami_id: str,
    key_name: str,
    security_group: str,
    iam_role: str,
    user_data: str,
    name_tag: str,
    subnet_id: Optional[str] = None,
) -> list:
    """Launch EC2 instances with the given configuration."""
    # Encode user data as base64
    user_data_b64 = base64.b64encode(user_data.encode()).decode()

    # Build launch parameters
    params = {
        "ImageId": ami_id,
        "InstanceType": instance_type,
        "MinCount": count,
        "MaxCount": count,
        "KeyName": key_name,
        "SecurityGroupIds": [security_group],
        "UserData": user_data_b64,
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": name_tag},
                    {"Key": "Project", "Value": "citrees"},
                    {"Key": "Role", "Value": role},
                ],
            }
        ],
    }

    # Add IAM instance profile if specified
    if iam_role:
        params["IamInstanceProfile"] = {"Name": iam_role}

    # Add subnet if specified
    if subnet_id:
        params["SubnetId"] = subnet_id

    # Launch instances
    response = ec2_client.run_instances(**params)

    return response["Instances"]


def main():
    parser = argparse.ArgumentParser(
        description="Launch EC2 instances for citrees experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Instance type to launch")

    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--table-name",
        required=True,
        help="DynamoDB table name (e.g., ClfFeatureSelection)",
    )
    common.add_argument(
        "--experiment-type",
        choices=["clf_feature_selection", "reg_feature_selection", "clf_cv", "reg_cv"],
        default="clf_feature_selection",
        help="Type of experiment to run",
    )
    common.add_argument(
        "--instance-type",
        default="c5.xlarge",
        help="EC2 instance type (default: c5.xlarge)",
    )
    common.add_argument(
        "--ami",
        default=DEFAULT_AMI,
        help="AMI ID (default: Amazon Linux 2023)",
    )
    common.add_argument(
        "--key-name",
        required=True,
        help="SSH key pair name",
    )
    common.add_argument(
        "--security-group",
        required=True,
        help="Security group ID",
    )
    common.add_argument(
        "--iam-role",
        required=True,
        help="IAM instance profile name (must have DynamoDB access)",
    )
    common.add_argument(
        "--subnet-id",
        help="VPC subnet ID (optional)",
    )
    common.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    common.add_argument(
        "--repo-url",
        default="https://github.com/rmill040/citrees.git",
        help="Git repository URL",
    )
    common.add_argument(
        "--branch",
        default="main",
        help="Git branch to checkout (default: main)",
    )
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="Print user data script without launching",
    )

    # Server subcommand
    server_parser = subparsers.add_parser(
        "server",
        parents=[common],
        help="Launch a server instance",
    )

    # Worker subcommand
    worker_parser = subparsers.add_parser(
        "worker",
        parents=[common],
        help="Launch worker instances",
    )
    worker_parser.add_argument(
        "--server-url",
        required=True,
        help="Server URL (e.g., http://10.0.0.5:8000)",
    )
    worker_parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of worker instances to launch (default: 1)",
    )
    worker_parser.add_argument(
        "--n-jobs-outer",
        type=int,
        default=1,
        help="Parallel requests to server (default: 1)",
    )
    worker_parser.add_argument(
        "--n-jobs-inner",
        type=int,
        default=-1,
        help="Parallelism within each experiment (default: -1, all cores)",
    )
    worker_parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated list of methods to skip",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Determine experiment type prefix
    exp_prefix = args.experiment_type.split("_")[0]  # 'clf' or 'reg'

    if args.command == "server":
        user_data = get_server_user_data(
            table_name=args.table_name,
            experiment_type=args.experiment_type,
            region=args.region,
            repo_url=args.repo_url,
            branch=args.branch,
        )
        name_tag = f"citrees-{args.experiment_type}-server"
        count = 1
    else:  # worker
        user_data = get_worker_user_data(
            server_url=args.server_url,
            table_name=args.table_name,
            experiment_type=args.experiment_type,
            region=args.region,
            n_jobs_outer=args.n_jobs_outer,
            n_jobs_inner=args.n_jobs_inner,
            skip=args.skip,
            repo_url=args.repo_url,
            branch=args.branch,
        )
        name_tag = f"citrees-{args.experiment_type}-worker"
        count = args.count

    if args.dry_run:
        print("=== User Data Script ===")
        print(user_data)
        print("\n=== Launch Configuration ===")
        print(f"Instance type: {args.instance_type}")
        print(f"AMI: {args.ami}")
        print(f"Count: {count}")
        print(f"Key pair: {args.key_name}")
        print(f"Security group: {args.security_group}")
        print(f"IAM role: {args.iam_role}")
        print(f"Name tag: {name_tag}")
        return

    # Launch instances
    ec2 = boto3.client("ec2", region_name=args.region)

    print(f"Launching {count} {args.command} instance(s)...")

    instances = launch_instances(
        ec2_client=ec2,
        role=args.command,
        count=count,
        instance_type=args.instance_type,
        ami_id=args.ami,
        key_name=args.key_name,
        security_group=args.security_group,
        iam_role=args.iam_role,
        user_data=user_data,
        name_tag=name_tag,
        subnet_id=args.subnet_id,
    )

    print(f"\nLaunched {len(instances)} instance(s):")
    for inst in instances:
        print(f"  - {inst['InstanceId']}")

    print("\nInstances are starting. Check AWS Console for status.")
    print("View bootstrap logs: ssh ec2-user@<ip> 'tail -f /var/log/user-data.log'")

    if args.command == "server":
        print("\nAfter startup, get the server's private IP:")
        print(f"  aws ec2 describe-instances --instance-ids {instances[0]['InstanceId']} --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text")


if __name__ == "__main__":
    main()
