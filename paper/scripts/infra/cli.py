#!/usr/bin/env python3
"""citrees distributed experiment infrastructure CLI.

Usage:
    uv run python -m paper.scripts.infra.cli setup
    uv run python -m paper.scripts.infra.cli launch
    uv run python -m paper.scripts.infra.cli status
    uv run python -m paper.scripts.infra.cli logs server
    uv run python -m paper.scripts.infra.cli teardown
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .compute import (
    find_instances_by_tag,
    get_instance_status,
    launch_server,
    launch_workers,
    terminate_instances,
)
from .config import Config, load_config
from .resources import (
    build_and_push_image,
    create_dynamodb_table,
    create_ecr_repository,
    create_iam_role,
    create_s3_bucket,
    get_latest_ami,
    setup_vpc,
    teardown_ecr,
    teardown_iam,
    teardown_instances,
    teardown_storage,
    teardown_vpc,
)

app = typer.Typer(
    name="citrees-infra",
    help="Manage citrees distributed experiment infrastructure.",
    no_args_is_help=True,
)
console = Console()


def get_state_path(config_path: Path) -> Path:
    """Get state file path for a config file."""
    return config_path.parent / ".citrees-state.yaml"


# =============================================================================
# Setup Command
# =============================================================================


@app.command()
def setup(
    config_path: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to config.yaml"),
    ] = None,
    skip_docker: Annotated[
        bool,
        typer.Option("--skip-docker", help="Skip Docker build and push"),
    ] = False,
) -> None:
    """Provision all AWS resources (VPC, DynamoDB, S3, ECR, IAM).

    Creates everything needed to run distributed experiments:
    - VPC with private subnet and NAT gateway
    - DynamoDB table for experiment tracking
    - S3 bucket for results
    - ECR repository for Docker image
    - IAM role with scoped permissions
    - Builds and pushes Docker image to ECR
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    config = load_config(config_path)
    state_path = get_state_path(config_path)

    console.print(Panel.fit("[bold blue]citrees Infrastructure Setup[/bold blue]"))
    console.print()

    # Show configuration
    table = Table(title="Configuration", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    table.add_row("Project", config.project_name)
    table.add_row("Region", config.region)
    table.add_row("Account ID", config.account_id)
    table.add_row("VPC", config.vpc.vpc_id)
    table.add_row("Server", f"{config.server.instance_type} ({'spot' if config.server.spot else 'on-demand'})")
    table.add_row("Workers", f"{config.workers.count}x {config.workers.instance_type} ({'spot' if config.workers.spot else 'on-demand'})")
    console.print(table)
    console.print()

    # VPC
    console.print("[bold]1. VPC & Networking[/bold]")
    setup_vpc(config)
    console.print()

    # DynamoDB
    console.print("[bold]2. DynamoDB[/bold]")
    table_name = create_dynamodb_table(config)
    console.print(f"  Table: {table_name}")
    console.print()

    # S3
    console.print("[bold]3. S3[/bold]")
    bucket_name = create_s3_bucket(config)
    console.print(f"  Bucket: {bucket_name}")
    console.print()

    # ECR
    console.print("[bold]4. ECR[/bold]")
    repo_uri = create_ecr_repository(config)
    console.print(f"  Repository: {repo_uri}")
    console.print()

    # IAM
    console.print("[bold]5. IAM[/bold]")
    role_name = create_iam_role(config)
    console.print(f"  Role: {role_name}")
    console.print(f"  Instance profile: {config.iam_instance_profile_name}")
    console.print()

    # Docker
    if not skip_docker:
        console.print("[bold]6. Docker Image[/bold]")
        build_and_push_image(config)
        console.print()
    else:
        console.print("[bold]6. Docker Image[/bold] [dim](skipped)[/dim]")
        console.print()

    # Save state
    config.save_state(state_path)

    console.print(Panel.fit("[bold green]Setup complete![/bold green]"))
    console.print()
    console.print("Next steps:")
    console.print("  1. Launch experiment:  [cyan]uv run python -m paper.scripts.infra.cli launch[/cyan]")
    console.print("  2. Check status:       [cyan]uv run python -m paper.scripts.infra.cli status[/cyan]")


# =============================================================================
# Launch Command
# =============================================================================


@app.command()
def launch(
    config_path: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to config.yaml"),
    ] = None,
    workers_only: Annotated[
        bool,
        typer.Option("--workers-only", help="Only launch workers (server must exist)"),
    ] = False,
) -> None:
    """Launch server and worker EC2 instances.

    Starts the distributed experiment:
    - 1 server instance (on-demand by default)
    - N worker instances (spot by default)
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    config = load_config(config_path)
    state_path = get_state_path(config_path)

    # Verify setup was run
    if not config._resolved_private_subnet_id:
        console.print("[red]Error: Run 'setup' first to provision infrastructure.[/red]")
        raise typer.Exit(1)

    console.print(Panel.fit("[bold blue]Launching Experiment[/bold blue]"))
    console.print()

    # Get AMI
    ami_id = get_latest_ami(config)
    console.print(f"Using AMI: {ami_id}")
    console.print()

    # Launch server
    if not workers_only:
        console.print("[bold]1. Launching Server[/bold]")
        instance_id, private_ip = launch_server(config, ami_id)
        console.print(f"  Instance: {instance_id}")
        console.print(f"  Private IP: {private_ip}")
        console.print()
    else:
        if not config._server_private_ip:
            console.print("[red]Error: No server found. Run without --workers-only.[/red]")
            raise typer.Exit(1)
        console.print(f"[bold]1. Using existing server[/bold]: {config._server_instance_id}")
        console.print()

    # Launch workers
    console.print(f"[bold]2. Launching {config.workers.count} Worker(s)[/bold]")
    instance_ids = launch_workers(config, ami_id)
    for i, iid in enumerate(instance_ids):
        console.print(f"  Worker {i}: {iid}")
    console.print()

    # Save state
    config.save_state(state_path)

    console.print(Panel.fit("[bold green]Launch complete![/bold green]"))
    console.print()
    console.print("Monitor progress:")
    console.print("  [cyan]uv run python -m paper.scripts.infra.cli status[/cyan]")
    console.print()
    console.print("View logs:")
    console.print("  [cyan]uv run python -m paper.scripts.infra.cli logs server[/cyan]")
    console.print("  [cyan]uv run python -m paper.scripts.infra.cli logs worker-0[/cyan]")


# =============================================================================
# Scale Command
# =============================================================================


@app.command()
def scale(
    count: Annotated[
        int,
        typer.Argument(help="Number of workers to add"),
    ],
    config_path: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to config.yaml"),
    ] = None,
) -> None:
    """Add more worker instances to the running experiment.

    Examples:
        scale 5     # Add 5 more workers
        scale 10    # Add 10 more workers
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    config = load_config(config_path)
    state_path = get_state_path(config_path)

    # Verify server is running
    if not config._server_private_ip:
        # Try to find it
        find_instances_by_tag(config)
        if not config._server_private_ip:
            console.print("[red]Error: No server found. Run 'launch' first.[/red]")
            raise typer.Exit(1)

    console.print(Panel.fit(f"[bold blue]Adding {count} Worker(s)[/bold blue]"))
    console.print()

    # Get AMI
    ami_id = get_latest_ami(config)

    # Temporarily override worker count
    original_count = config.workers.count
    config.workers.count = count

    # Launch additional workers
    console.print(f"[bold]Launching {count} additional worker(s)[/bold]")
    console.print(f"  Server: {config._server_private_ip}")

    instance_ids = launch_workers(config, ami_id)

    # Restore count and save state
    config.workers.count = original_count
    config.save_state(state_path)

    console.print()
    for i, iid in enumerate(instance_ids):
        console.print(f"  New worker: {iid}")

    console.print()
    console.print(Panel.fit(f"[bold green]Added {count} worker(s)![/bold green]"))
    console.print()
    console.print(f"Total workers: {len(config._worker_instance_ids)}")


# =============================================================================
# Status Command
# =============================================================================


@app.command()
def status(
    config_path: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to config.yaml"),
    ] = None,
) -> None:
    """Show status of infrastructure and running instances."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    config = load_config(config_path)
    state_path = get_state_path(config_path)

    console.print(Panel.fit("[bold blue]Infrastructure Status[/bold blue]"))
    console.print()

    # Resources table
    resources_table = Table(title="AWS Resources")
    resources_table.add_column("Resource", style="cyan")
    resources_table.add_column("Name/ID")
    resources_table.add_column("Status", style="green")

    resources_table.add_row("VPC", config._resolved_vpc_id or "Not created", "OK" if config._resolved_vpc_id else "Missing")
    resources_table.add_row("Private Subnet", config._resolved_private_subnet_id or "Not created", "OK" if config._resolved_private_subnet_id else "Missing")
    resources_table.add_row("Security Group", config._resolved_security_group_id or "Not created", "OK" if config._resolved_security_group_id else "Missing")
    resources_table.add_row("DynamoDB", config.table_name, "OK")
    resources_table.add_row("S3", config.bucket_name, "OK")
    resources_table.add_row("ECR", config.ecr_repository_name, "OK")
    resources_table.add_row("IAM Role", config.iam_role_name, "OK")

    console.print(resources_table)
    console.print()

    # Try to find instances by tag if state is missing
    if not config._server_instance_id and not config._worker_instance_ids:
        find_instances_by_tag(config)
        config.save_state(state_path)

    # Instance status
    if config._server_instance_id or config._worker_instance_ids:
        instance_status = get_instance_status(config)

        instances_table = Table(title="EC2 Instances")
        instances_table.add_column("Role", style="cyan")
        instances_table.add_column("Instance ID")
        instances_table.add_column("Type")
        instances_table.add_column("Private IP")
        instances_table.add_column("State", style="green")

        if instance_status["server"]:
            s = instance_status["server"]
            instances_table.add_row(
                "server",
                s["instance_id"],
                s["instance_type"],
                s["private_ip"],
                s["state"],
            )

        for i, w in enumerate(instance_status["workers"]):
            instances_table.add_row(
                f"worker-{i}",
                w["instance_id"],
                w["instance_type"],
                w["private_ip"],
                w["state"],
            )

        console.print(instances_table)
    else:
        console.print("[dim]No instances running.[/dim]")

    console.print()

    # Debug commands
    console.print("[bold]Debug Commands:[/bold]")
    if config._server_instance_id:
        console.print(f"  Connect to server: [cyan]aws ssm start-session --target {config._server_instance_id}[/cyan]")
    if config._worker_instance_ids:
        console.print(f"  Connect to worker: [cyan]aws ssm start-session --target {config._worker_instance_ids[0]}[/cyan]")


# =============================================================================
# Logs Command
# =============================================================================


@app.command()
def logs(
    target: Annotated[
        str,
        typer.Argument(help="Target: 'server', 'worker-N', or instance ID"),
    ],
    config_path: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to config.yaml"),
    ] = None,
    follow: Annotated[
        bool,
        typer.Option("--follow", "-f", help="Follow logs (stream)"),
    ] = False,
    bootstrap: Annotated[
        bool,
        typer.Option("--bootstrap", "-b", help="Show bootstrap logs instead of container logs"),
    ] = False,
) -> None:
    """View logs from server or worker instances via SSM.

    Examples:
        logs server           # Server container logs
        logs worker-0         # First worker container logs
        logs server -b        # Server bootstrap logs
        logs server -f        # Follow server logs
        logs i-abc123         # Logs by instance ID
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    config = load_config(config_path)

    # Resolve target to instance ID
    if target == "server":
        if not config._server_instance_id:
            console.print("[red]Error: No server instance found.[/red]")
            raise typer.Exit(1)
        instance_id = config._server_instance_id
        container_name = "citrees-server"
    elif target.startswith("worker-"):
        idx = int(target.split("-")[1])
        if idx >= len(config._worker_instance_ids):
            console.print(f"[red]Error: Worker {idx} not found. Have {len(config._worker_instance_ids)} workers.[/red]")
            raise typer.Exit(1)
        instance_id = config._worker_instance_ids[idx]
        container_name = "citrees-worker"
    elif target.startswith("i-"):
        instance_id = target
        container_name = "citrees-server"  # Guess
    else:
        console.print(f"[red]Error: Unknown target '{target}'. Use 'server', 'worker-N', or instance ID.[/red]")
        raise typer.Exit(1)

    # Build SSM command
    if bootstrap:
        remote_cmd = "cat /var/log/user-data.log"
    elif follow:
        remote_cmd = f"docker logs -f {container_name}"
    else:
        remote_cmd = f"docker logs --tail 100 {container_name}"

    console.print(f"[dim]Connecting to {instance_id}...[/dim]")
    console.print()

    # Run SSM command
    cmd = [
        "aws", "ssm", "start-session",
        "--target", instance_id,
        "--document-name", "AWS-StartInteractiveCommand",
        "--parameters", f"command=[\"{remote_cmd}\"]",
        "--region", config.region,
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass


# =============================================================================
# Teardown Command
# =============================================================================


@app.command()
def teardown(
    config_path: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to config.yaml"),
    ] = None,
    keep_data: Annotated[
        bool,
        typer.Option("--keep-data", help="Keep S3 bucket and data"),
    ] = True,
    keep_ecr: Annotated[
        bool,
        typer.Option("--keep-ecr", help="Keep ECR repository and images"),
    ] = True,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Skip confirmation"),
    ] = False,
) -> None:
    """Tear down infrastructure and terminate instances.

    By default, keeps S3 data and ECR images. Use flags to delete:
        --keep-data=false   Delete S3 bucket and all data
        --keep-ecr=false    Delete ECR repository and images
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    config = load_config(config_path)
    state_path = get_state_path(config_path)

    console.print(Panel.fit("[bold red]Teardown Infrastructure[/bold red]"))
    console.print()

    # Show what will be deleted
    console.print("[bold]Will delete:[/bold]")
    console.print("  - All EC2 instances (server + workers)")
    console.print("  - VPC and networking (if created by setup)")
    console.print("  - DynamoDB table")
    console.print("  - IAM role and instance profile")

    if not keep_data:
        console.print("  - [red]S3 bucket and ALL data[/red]")
    else:
        console.print("  - [dim]S3 bucket (keeping)[/dim]")

    if not keep_ecr:
        console.print("  - [red]ECR repository and images[/red]")
    else:
        console.print("  - [dim]ECR repository (keeping)[/dim]")

    console.print()

    if not force:
        confirm = typer.confirm("Are you sure you want to continue?")
        if not confirm:
            raise typer.Abort()

    console.print()

    # Find any running instances
    find_instances_by_tag(config)

    # Terminate instances
    console.print("[bold]1. Terminating instances[/bold]")
    terminate_instances(config)
    console.print()

    # Delete IAM
    console.print("[bold]2. Deleting IAM[/bold]")
    teardown_iam(config)
    console.print()

    # Delete ECR
    if not keep_ecr:
        console.print("[bold]3. Deleting ECR[/bold]")
        teardown_ecr(config)
    else:
        console.print("[bold]3. Keeping ECR[/bold]")
    console.print()

    # Delete storage
    console.print("[bold]4. Deleting storage[/bold]")
    teardown_storage(config, delete_data=not keep_data)
    console.print()

    # Delete VPC
    console.print("[bold]5. Deleting VPC[/bold]")
    teardown_vpc(config)
    console.print()

    # Remove state file
    if state_path.exists():
        state_path.unlink()

    console.print(Panel.fit("[bold green]Teardown complete![/bold green]"))


# =============================================================================
# SSM Connect Command
# =============================================================================


@app.command()
def connect(
    target: Annotated[
        str,
        typer.Argument(help="Target: 'server', 'worker-N', or instance ID"),
    ],
    config_path: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to config.yaml"),
    ] = None,
) -> None:
    """Open interactive SSM session to an instance.

    Examples:
        connect server        # Connect to server
        connect worker-0      # Connect to first worker
        connect i-abc123      # Connect by instance ID
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    config = load_config(config_path)

    # Resolve target to instance ID
    if target == "server":
        if not config._server_instance_id:
            console.print("[red]Error: No server instance found.[/red]")
            raise typer.Exit(1)
        instance_id = config._server_instance_id
    elif target.startswith("worker-"):
        idx = int(target.split("-")[1])
        if idx >= len(config._worker_instance_ids):
            console.print(f"[red]Error: Worker {idx} not found.[/red]")
            raise typer.Exit(1)
        instance_id = config._worker_instance_ids[idx]
    elif target.startswith("i-"):
        instance_id = target
    else:
        console.print(f"[red]Error: Unknown target '{target}'.[/red]")
        raise typer.Exit(1)

    console.print(f"[dim]Connecting to {instance_id}...[/dim]")
    console.print("[dim]Tip: Run 'docker logs citrees-server' or 'docker logs citrees-worker'[/dim]")
    console.print()

    cmd = ["aws", "ssm", "start-session", "--target", instance_id, "--region", config.region]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
