"""Infrastructure commands for AWS setup and management.

Commands for setting up IAM, ECR, Docker images, and cluster configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.syntax import Syntax

from paper.scripts.cli._console import console, error, heading, info, success

app = typer.Typer(
    name="infra",
    help="AWS infrastructure setup",
    no_args_is_help=True,
)

# Path constants
RAY_DIR = Path(__file__).parent.parent / "infra" / "ray"
CLUSTER_YAML = RAY_DIR / "cluster.yaml"
CLUSTER_EXAMPLE_YAML = RAY_DIR / "cluster.example.yaml"


@app.command()
def setup(
    branch: Annotated[
        str | None,
        typer.Option(
            "--branch",
            "-b",
            help="Git branch to use (default: current branch)",
        ),
    ] = None,
) -> None:
    """Full infrastructure setup: IAM + Docker + cluster.yaml.

    This performs all setup steps in sequence:
    1. Create IAM role and instance profile
    2. Build and push Docker image to ECR
    3. Generate cluster.yaml with current settings

    \b
    Example:
        citrees-exp infra setup
        citrees-exp infra setup --branch feat/paper
    """
    from paper.scripts.infra.ray.setup_cluster import (
        build_and_push_image,
        ensure_iam_role,
        generate_config,
    )

    heading("Full Setup: IAM + Docker + cluster.yaml")

    console.print("\n[1/3] Ensuring IAM role and instance profile...")
    with console.status("Creating IAM resources..."):
        ensure_iam_role()
    success("IAM role ready")

    console.print("\n[2/3] Building and pushing Docker image...")
    image_uri = build_and_push_image()
    success(f"Docker image pushed: {image_uri}")

    console.print("\n[3/3] Generating cluster.yaml...")
    generate_config(branch=branch)
    success(f"Created {CLUSTER_YAML}")

    heading("Setup Complete")
    console.print("\nNext step:")
    console.print("  [cyan]citrees-exp cluster up --yes[/]")


@app.command()
def build() -> None:
    """Build and push Docker image to ECR.

    Builds the Docker image from paper/scripts/infra/docker/Dockerfile
    and pushes it to ECR with both :latest and :{git_sha} tags.
    """
    from paper.scripts.infra.ray.setup_cluster import build_and_push_image

    heading("Building Docker Image")

    with console.status("Building and pushing..."):
        image_uri = build_and_push_image()

    success(f"Image pushed: {image_uri}")


@app.command()
def generate(
    branch: Annotated[
        str | None,
        typer.Option(
            "--branch",
            "-b",
            help="Git branch to use (default: current branch)",
        ),
    ] = None,
) -> None:
    """Generate cluster.yaml from template.

    Creates cluster.yaml with:
    - Your current public IP for SSH access
    - Current git branch for code sync
    - Auto-derived S3 bucket name
    """
    from paper.scripts.infra.ray.setup_cluster import generate_config

    heading("Generating cluster.yaml")

    generate_config(branch=branch)
    success(f"Created {CLUSTER_YAML}")


@app.command()
def validate() -> None:
    """Validate cluster configuration.

    Checks that:
    - AMI IDs exist and are valid
    - Instance types are available
    - Region settings are correct
    """
    if not CLUSTER_YAML.exists():
        error(f"Cluster config not found: {CLUSTER_YAML}")
        error("Run 'citrees-exp infra generate' first.")
        raise typer.Exit(1)

    from paper.scripts.infra.ray.setup_cluster import load_cluster_config, validate_config

    heading("Validating cluster configuration")

    config = load_cluster_config()
    region = config.get("provider", {}).get("region", "us-east-1")

    with console.status(f"Validating against AWS ({region})..."):
        errors = validate_config(config)

    if errors:
        error("Validation failed:")
        for err in errors:
            console.print(f"  [error]\u2717[/] {err}")
        raise typer.Exit(1)

    success("Configuration is valid")


@app.command()
def show(
    raw: Annotated[
        bool,
        typer.Option(
            "--raw",
            help="Print raw YAML without syntax highlighting",
        ),
    ] = False,
) -> None:
    """Display cluster configuration.

    Shows current cluster.yaml with syntax highlighting.
    """
    if not CLUSTER_YAML.exists():
        error(f"Cluster config not found: {CLUSTER_YAML}")
        error("Run 'citrees-exp infra generate' first.")
        raise typer.Exit(1)

    content = CLUSTER_YAML.read_text()
    if raw:
        console.print(content, markup=False)
        return

    syntax = Syntax(content, "yaml", theme="monokai", line_numbers=True)
    console.print(syntax)


@app.command(name="find-ami")
def find_ami(
    region: Annotated[
        str,
        typer.Option(
            "--region",
            "-r",
            help="AWS region",
        ),
    ] = "us-east-1",
) -> None:
    """Find latest Ubuntu 22.04 AMI for a region.

    This is a read-only command that queries AWS for the latest
    Ubuntu 22.04 AMI without modifying any files.
    """
    from paper.scripts.infra.ray.setup_cluster import get_latest_ubuntu_ami

    heading(f"Finding latest Ubuntu 22.04 AMI ({region})")

    with console.status("Querying AWS..."):
        ami = get_latest_ubuntu_ami(region)

    console.print(f"  [muted]AMI ID:[/] [highlight]{ami['id']}[/]")
    console.print(f"  [muted]Name:[/] {ami['name']}")
    console.print(f"  [muted]Created:[/] {ami['created']}")


@app.command(name="update-ami")
def update_ami() -> None:
    """Update AMI IDs in cluster.yaml to latest Ubuntu 22.04.

    Fetches the latest Ubuntu 22.04 AMI for the configured region
    and updates all node types in cluster.yaml.
    """
    if not CLUSTER_YAML.exists():
        error(f"Cluster config not found: {CLUSTER_YAML}")
        error("Run 'citrees-exp infra generate' first.")
        raise typer.Exit(1)

    from paper.scripts.infra.ray.setup_cluster import (
        get_latest_ubuntu_ami,
        load_cluster_config,
        save_cluster_config,
    )
    from paper.scripts.infra.ray.setup_cluster import (
        update_ami as do_update_ami,
    )

    config = load_cluster_config()
    region = config.get("provider", {}).get("region", "us-east-1")

    heading(f"Updating AMI ({region})")

    with console.status("Fetching latest AMI..."):
        ami = get_latest_ubuntu_ami(region)

    info(f"Latest AMI: {ami['id']} ({ami['name']})")

    config = do_update_ami(config, ami["id"])
    save_cluster_config(config)

    success(f"Updated {CLUSTER_YAML}")


@app.command()
def iam() -> None:
    """Create IAM role and instance profile for Ray.

    Creates:
    - IAM role: citrees-ray-autoscaler
    - Instance profile: citrees-ray-autoscaler
    - Attached policies for EC2, S3, and ECR access
    """
    from paper.scripts.infra.ray.setup_cluster import ensure_iam_role

    heading("Creating IAM resources")

    with console.status("Creating IAM role and instance profile..."):
        arn = ensure_iam_role()

    success(f"IAM resources ready: {arn}")


@app.command()
def ecr() -> None:
    """Create ECR repository for Docker images.

    Creates ECR repository: citrees-{account_id}
    """
    from paper.scripts.infra.ray.setup_cluster import ensure_ecr_repo

    heading("Creating ECR repository")

    with console.status("Creating ECR repository..."):
        repo_name, repo_uri = ensure_ecr_repo()

    success(f"ECR repository ready: {repo_uri}")


@app.command()
def s3() -> None:
    """Create S3 bucket for experiment results.

    Creates S3 bucket: citrees-{account_id}
    """
    from paper.scripts.infra.ray.setup_cluster import ensure_s3_bucket

    heading("Creating S3 bucket")

    with console.status("Creating S3 bucket..."):
        bucket_name = ensure_s3_bucket()

    success(f"S3 bucket ready: {bucket_name}")
