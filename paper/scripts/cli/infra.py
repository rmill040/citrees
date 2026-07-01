"""Infrastructure commands for AWS setup and management.

Commands for setting up IAM, ECR, Docker images, S3, and EC2 workers.
"""

from __future__ import annotations

from typing import Annotated

import typer

from paper.scripts.cli._console import (
    console,
    create_table,
    error,
    heading,
    info,
    success,
    warn,
)

app = typer.Typer(
    name="infra",
    help="AWS infrastructure setup",
    no_args_is_help=True,
)


def _split_csv(value: str) -> tuple[str, ...]:
    """Parse comma-separated CLI options into a typed tuple."""
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _split_int_csv(value: str) -> tuple[int, ...]:
    """Parse comma-separated integer CLI options."""
    return tuple(int(part) for part in _split_csv(value))


# ---------------------------------------------------------------------------
# ECR subcommand group
# ---------------------------------------------------------------------------
ecr_app = typer.Typer(
    name="ecr",
    help="ECR repository and Docker image management",
    no_args_is_help=True,
)
app.add_typer(ecr_app, name="ecr")


@ecr_app.command()
def create() -> None:
    """Create ECR repository for Docker images.

    Creates ECR repository: citrees-{account_id}
    """
    from paper.scripts.infra.aws import ensure_ecr_repo

    heading("Creating ECR repository")

    with console.status("Creating ECR repository..."):
        _repo_name, repo_uri = ensure_ecr_repo()

    success(f"ECR repository ready: {repo_uri}")


@ecr_app.command()
def build() -> None:
    """Build and push Docker image to ECR.

    Builds the Docker image from paper/scripts/infra/docker/Dockerfile
    and pushes it to ECR with both :latest and :{git_sha} tags.

    Example:
        citrees-exp infra ecr build
    """
    from paper.scripts.infra.aws import build_and_push_image

    heading("Building Docker Image")

    image_uri = build_and_push_image()

    success(f"Image pushed: {image_uri}")


@ecr_app.command()
def clean() -> None:
    """Clear all images from the ECR repository.

    Two-stage cleanup:
    1. Delete tagged images (:latest, :{sha}, etc.)
    2. Delete remaining untagged manifests (orphaned layers)

    The repository itself is preserved.

    Example:
        citrees-exp infra ecr clean
    """
    from paper.scripts.infra.aws import clean_ecr

    heading("Cleaning ECR Repository")

    with console.status("Deleting images..."):
        counts = clean_ecr()

    total = counts["tagged"] + counts["untagged"]
    if total == 0:
        info("Repository already empty")
    else:
        success(
            f"Deleted {counts['tagged']} tagged images, {counts['untagged']} untagged manifests"
        )


# ---------------------------------------------------------------------------
# Top-level infra commands
# ---------------------------------------------------------------------------


@app.command()
def setup() -> None:
    """Full infrastructure setup: IAM + Docker image.

    This performs all setup steps in sequence:
    1. Create IAM role and instance profile
    2. Build and push Docker image to ECR

    Example:
        citrees-exp infra setup
    """
    from paper.scripts.infra.aws import build_and_push_image, ensure_iam_role

    heading("Full Setup: IAM + Docker")

    console.print("\n[1/2] Ensuring IAM role and instance profile...")
    with console.status("Creating IAM resources..."):
        ensure_iam_role()
    success("IAM role ready")

    console.print("\n[2/2] Building and pushing Docker image...")
    image_uri = build_and_push_image()
    success(f"Docker image pushed: {image_uri}")

    heading("Setup Complete")


@app.command()
def iam() -> None:
    """Create IAM role and instance profile for workers.

    Creates:
    - IAM role: citrees-worker
    - Instance profile: citrees-worker
    - Attached policies for S3 and ECR access
    """
    from paper.scripts.infra.aws import ensure_iam_role

    heading("Creating IAM resources")

    with console.status("Creating IAM role and instance profile..."):
        arn = ensure_iam_role()

    success(f"IAM resources ready: {arn}")


@app.command()
def s3() -> None:
    """Create S3 bucket for experiment results.

    Creates S3 bucket: citrees-{account_id}
    """
    from paper.scripts.infra.aws import ensure_s3_bucket

    heading("Creating S3 bucket")

    with console.status("Creating S3 bucket..."):
        bucket_name = ensure_s3_bucket()

    success(f"S3 bucket ready: {bucket_name}")


@app.command(name="upload-data")
def upload_data(
    task: Annotated[
        str | None,
        typer.Option(
            "--task",
            "-t",
            help="Only upload for this task type (classification/regression)",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            "-n",
            help="Show what would be uploaded without uploading",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Re-upload even if file already exists in S3",
        ),
    ] = False,
) -> None:
    """Upload datasets to S3 for workers.

    Uploads parquet files from paper/data/ to s3://{bucket}/data/
    Skips files that already exist in S3 (use --force to re-upload).
    """
    from paper.scripts.infra.aws import upload_datasets

    heading("Uploading Datasets to S3")

    if dry_run:
        info("Dry run - no files will be uploaded")

    with console.status("Scanning and uploading..."):
        result = upload_datasets(task=task, dry_run=dry_run, force=force)

    if dry_run:
        info(f"Would upload {result['uploaded']} files, skip {result['skipped']} existing")
    else:
        success(f"Uploaded {result['uploaded']} files, skipped {result['skipped']} existing")


# ---------------------------------------------------------------------------
# EC2 API server commands
# ---------------------------------------------------------------------------


@app.command(name="launch-api")
def launch_api_cmd(
    instance_type: Annotated[
        str,
        typer.Option(
            "--instance-type",
            "-i",
            help="EC2 instance type",
        ),
    ] = "m5.large",
    image_uri: Annotated[
        str,
        typer.Option(
            "--image-uri",
            help="ECR image URI",
        ),
    ] = "",
) -> None:
    """Launch the API server on an EC2 instance.

    Starts a single instance running the FastAPI queue server on port 8000.
    Workers connect to this server to pull and report experiment configs.
    """
    from paper.scripts.infra.ec2 import launch_api

    if not image_uri:
        from paper.scripts.infra.aws import ensure_ecr_repo

        _name, repo_uri = ensure_ecr_repo()
        image_uri = f"{repo_uri}:latest"
        info(f"Using image: {image_uri}")

    heading("Launching API Server")

    result = launch_api(instance_type=instance_type, image_uri=image_uri)

    if result["api_url"]:
        console.print(f"\n  API URL: [bold cyan]{result['api_url']}[/]")
        console.print("  Instance: " + result["instance_id"])


@app.command(name="api-url")
def api_url_cmd() -> None:
    """Print the running API server URL (public IP for external access).

    Finds the EC2 instance tagged as the API server and prints its URL.
    Useful for scripting: $(citrees-exp infra api-url)
    """
    import boto3

    from paper.scripts.infra.aws import DEFAULT_REGION
    from paper.scripts.infra.ec2 import API_TAG_VALUE, TAG_KEY

    ec2 = boto3.client("ec2", region_name=DEFAULT_REGION)
    response = ec2.describe_instances(
        Filters=[
            {"Name": f"tag:{TAG_KEY}", "Values": [API_TAG_VALUE]},
            {"Name": "instance-state-name", "Values": ["pending", "running"]},
        ]
    )
    for reservation in response.get("Reservations", []):
        for inst in reservation.get("Instances", []):
            ip = inst.get("PublicIpAddress")
            if ip:
                console.print(f"http://{ip}:8000")
                return

    error("No running API server found")
    raise typer.Exit(1)


@app.command(name="terminate-api")
def terminate_api_cmd() -> None:
    """Terminate the API server instance."""
    from paper.scripts.infra.ec2 import terminate_api

    heading("Terminating API Server")

    result = terminate_api()
    if result:
        success(f"Terminated: {result}")
    else:
        info("No API server to terminate")


# ---------------------------------------------------------------------------
# EC2 worker commands
# ---------------------------------------------------------------------------


@app.command(name="launch-workers")
def launch_workers_cmd(
    n: Annotated[
        int,
        typer.Option(
            "--count",
            "-n",
            help="Number of worker instances to launch",
        ),
    ] = 1,
    instance_type: Annotated[
        str,
        typer.Option(
            "--instance-type",
            "-i",
            help="EC2 instance type",
        ),
    ] = "c6a.8xlarge",
    spot: Annotated[
        bool,
        typer.Option(
            "--spot/--no-spot",
            help="Use spot instances (default: spot)",
        ),
    ] = True,
    image_uri: Annotated[
        str,
        typer.Option(
            "--image-uri",
            help="ECR image URI",
        ),
    ] = "",
) -> None:
    """Launch EC2 worker instances.

    Each instance pulls a Docker image from ECR and runs a worker process
    that gets work assignments from the API server. The API server's private
    IP is auto-discovered from EC2.
    """
    from paper.scripts.infra.ec2 import launch_workers

    if not image_uri:
        from paper.scripts.infra.aws import ensure_ecr_repo

        _name, repo_uri = ensure_ecr_repo()
        image_uri = f"{repo_uri}:latest"
        info(f"Using image: {image_uri}")

    heading(f"Launching {n} Workers")

    launch_workers(
        n=n,
        instance_type=instance_type,
        image_uri=image_uri,
        spot=spot,
    )


@app.command(name="launch-mechanism-workers")
def launch_mechanism_workers_cmd(
    n: Annotated[
        int,
        typer.Option(
            "--count",
            "-n",
            help="Number of sharded mechanism workers to launch",
        ),
    ] = 1,
    instance_type: Annotated[
        str,
        typer.Option(
            "--instance-type",
            "-i",
            help="EC2 instance type",
        ),
    ] = "c6a.8xlarge",
    spot: Annotated[
        bool,
        typer.Option(
            "--spot/--no-spot",
            help="Use spot instances (default: spot)",
        ),
    ] = True,
    image_uri: Annotated[
        str,
        typer.Option(
            "--image-uri",
            help="ECR image URI",
        ),
    ] = "",
    output_uri: Annotated[
        str,
        typer.Option(
            "--output-uri",
            help="S3 output prefix; defaults to s3://{bucket}/experiments/cif_mechanism_ablation",
        ),
    ] = "",
    num_shards: Annotated[
        int,
        typer.Option(
            "--num-shards",
            help="Global shard modulus; defaults to --count",
        ),
    ] = 0,
    shard_start: Annotated[
        int,
        typer.Option(
            "--shard-start",
            help="First shard index to launch",
        ),
    ] = 0,
    subnets: Annotated[
        str,
        typer.Option(
            "--subnets",
            help="Optional comma-separated subnet IDs; defaults to all default subnets",
        ),
    ] = "",
    tasks: Annotated[
        str,
        typer.Option(
            "--tasks",
            help="Comma-separated tasks: classification,regression",
        ),
    ] = "classification,regression",
    source: Annotated[
        str,
        typer.Option(
            "--source",
            help="Dataset source: real, synthetic, or all",
        ),
    ] = "real",
    datasets: Annotated[
        str,
        typer.Option(
            "--datasets",
            help="Optional comma-separated dataset filter",
        ),
    ] = "",
    seeds: Annotated[
        str,
        typer.Option(
            "--seeds",
            help="Comma-separated seed indices",
        ),
    ] = "0,1,2,3,4",
    folds: Annotated[
        str,
        typer.Option(
            "--folds",
            help="Comma-separated fold indices",
        ),
    ] = "0,1,2,3,4",
    model_variants: Annotated[
        str,
        typer.Option(
            "--model-variants",
            help="Comma-separated CIF model variants",
        ),
    ] = "cif_default",
    ranking_variants: Annotated[
        str,
        typer.Option(
            "--ranking-variants",
            help="Ranking readouts",
        ),
    ] = "split_importance,split_count",
    n_jobs: Annotated[
        int,
        typer.Option(
            "--n-jobs",
            help="CIF n_jobs inside each worker",
        ),
    ] = -1,
    downstream_n_jobs: Annotated[
        int,
        typer.Option(
            "--downstream-n-jobs",
            help="Downstream learner n_jobs inside each worker",
        ),
    ] = 1,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite existing mechanism-ablation artifacts",
        ),
    ] = False,
) -> None:
    """Launch sharded EC2 workers for CIF mechanism ablations.

    This command runs the paper-side mechanism-ablation runner directly on each
    instance, using stable modulo sharding. It does not require the API server.
    """
    from paper.scripts.infra.ec2 import launch_mechanism_workers

    if source not in {"real", "synthetic", "all"}:
        error("source must be one of: real, synthetic, all")
        raise typer.Exit(1)

    if not image_uri:
        from paper.scripts.infra.aws import ensure_ecr_repo

        _name, repo_uri = ensure_ecr_repo()
        image_uri = f"{repo_uri}:latest"
        info(f"Using image: {image_uri}")

    heading(f"Launching {n} Mechanism Workers")

    launch_mechanism_workers(
        n=n,
        instance_type=instance_type,
        image_uri=image_uri,
        spot=spot,
        num_shards=num_shards or None,
        shard_start=shard_start,
        subnet_ids=_split_csv(subnets),
        output_uri=output_uri or None,
        tasks=_split_csv(tasks),
        source=source,
        datasets=_split_csv(datasets),
        seeds=_split_int_csv(seeds),
        folds=_split_int_csv(folds),
        model_variants=_split_csv(model_variants),
        ranking_variants=_split_csv(ranking_variants),
        n_jobs=n_jobs,
        downstream_n_jobs=downstream_n_jobs,
        force=force,
    )


@app.command(name="list-workers")
def list_workers_cmd() -> None:
    """List running worker instances."""
    from paper.scripts.infra.ec2 import list_workers

    heading("Worker Instances")

    workers = list_workers()
    if not workers:
        info("No worker instances found")
        return

    table = create_table(
        title=f"Workers ({len(workers)})",
        columns=[
            ("Instance ID", ""),
            ("State", ""),
            ("Type", ""),
            ("Launched", ""),
        ],
    )
    for w in workers:
        table.add_row(
            w["instance_id"],
            w["state"],
            w["instance_type"],
            w["launch_time"],
        )
    console.print(table)


@app.command(name="list-mechanism-workers")
def list_mechanism_workers_cmd() -> None:
    """List running CIF mechanism-ablation worker instances."""
    from paper.scripts.infra.ec2 import list_mechanism_workers

    heading("Mechanism Worker Instances")

    workers = list_mechanism_workers()
    if not workers:
        info("No mechanism worker instances found")
        return

    table = create_table(
        title=f"Mechanism workers ({len(workers)})",
        columns=[
            ("Instance ID", ""),
            ("State", ""),
            ("Type", ""),
            ("Shard", ""),
            ("Launched", ""),
        ],
    )
    for w in workers:
        shard = f"{w['shard_index']}/{w['num_shards']}" if w["shard_index"] else ""
        table.add_row(
            w["instance_id"],
            w["state"],
            w["instance_type"],
            shard,
            w["launch_time"],
        )
    console.print(table)


@app.command(name="terminate-workers")
def terminate_workers_cmd() -> None:
    """Terminate all running worker instances."""
    from paper.scripts.infra.ec2 import terminate_workers

    heading("Terminating Workers")

    terminated = terminate_workers()
    if terminated:
        success(f"Terminated {len(terminated)} instances")
    else:
        info("No workers to terminate")


@app.command(name="terminate-mechanism-workers")
def terminate_mechanism_workers_cmd() -> None:
    """Terminate all running CIF mechanism-ablation workers."""
    from paper.scripts.infra.ec2 import terminate_mechanism_workers

    heading("Terminating Mechanism Workers")

    terminated = terminate_mechanism_workers()
    if terminated:
        success(f"Terminated {len(terminated)} mechanism instances")
    else:
        info("No mechanism workers to terminate")


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


@app.command()
def logs(
    role: Annotated[
        str,
        typer.Argument(
            help="Role to fetch logs for: api or worker",
            metavar="ROLE",
        ),
    ] = "api",
    instance_id: Annotated[
        str | None,
        typer.Option(
            "--instance",
            "-i",
            help="Instance ID to filter by (default: all instances)",
        ),
    ] = None,
    tail: Annotated[
        int,
        typer.Option(
            "--tail",
            "-n",
            help="Number of log events to show",
        ),
    ] = 100,
) -> None:
    """Fetch recent CloudWatch logs for API or worker instances.

    Container stdout/stderr is streamed to CloudWatch via the awslogs
    Docker log driver. Log groups: /citrees/api and /citrees/worker.

    Examples:
        citrees-exp infra logs api
        citrees-exp infra logs worker --instance i-0abc123
        citrees-exp infra logs api --tail 50
    """
    from paper.scripts.infra.ec2 import get_logs

    if role not in ("api", "worker", "mechanism"):
        error("Role must be 'api', 'worker', or 'mechanism'")
        raise typer.Exit(1)

    heading(f"CloudWatch Logs: /citrees/{role}")
    if instance_id:
        info(f"Instance: {instance_id}")

    events = get_logs(role, instance_id=instance_id, tail=tail)

    if not events:
        warn("No log events found")
        return

    for event in events:
        console.print(event["message"], highlight=False)
