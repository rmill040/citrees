"""Infrastructure commands for AWS setup and management."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from paper.benchmark.cli.console_output import (
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
    from paper.benchmark.infra.aws import ensure_ecr_repo

    heading("Creating ECR repository")

    with console.status("Creating ECR repository..."):
        _repo_name, repo_uri = ensure_ecr_repo()

    success(f"ECR repository ready: {repo_uri}")


@ecr_app.command()
def build() -> None:
    """Build and push Docker image to ECR.

    Builds the Docker image from paper/benchmark/infra/docker/Dockerfile
    and pushes it to ECR with one immutable full-commit tag.

    Example:
        citrees-exp infra ecr build
    """
    from paper.benchmark.infra.aws import build_and_push_image

    heading("Building Docker Image")

    image_uri = build_and_push_image()

    success(f"Image pushed: {image_uri}")


@ecr_app.command()
def clean() -> None:
    """Clear all images from the ECR repository.

    Two-stage cleanup:
    1. Delete full-revision tagged images
    2. Delete remaining untagged manifests (orphaned layers)

    The repository itself is preserved.

    Example:
        citrees-exp infra ecr clean
    """
    from paper.benchmark.infra.aws import clean_ecr

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
    """Create the S3 bucket and build the immutable Docker image.

    This performs all setup steps in sequence:
    1. Create the private, versioned S3 bucket
    2. Build and push Docker image to ECR

    Example:
        citrees-exp infra setup
    """
    from paper.benchmark.infra.aws import build_and_push_image, ensure_s3_bucket

    heading("Full Setup: S3 + Docker")

    console.print("\n[1/2] Ensuring S3 bucket...")
    with console.status("Creating S3 bucket..."):
        bucket_name = ensure_s3_bucket()
    success(f"S3 bucket ready: {bucket_name}")

    console.print("\n[2/2] Building and pushing Docker image...")
    image_uri = build_and_push_image()
    success(f"Docker image pushed: {image_uri}")

    heading("Setup Complete")


@app.command()
def s3() -> None:
    """Create S3 bucket for experiment results.

    Creates S3 bucket: citrees-{account_id}
    """
    from paper.benchmark.infra.aws import ensure_s3_bucket

    heading("Creating S3 bucket")

    with console.status("Creating S3 bucket..."):
        bucket_name = ensure_s3_bucket()

    success(f"S3 bucket ready: {bucket_name}")


@app.command(name="upload-data")
def upload_data(
    task: Annotated[
        Literal["classification", "regression"] | None,
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
) -> None:
    """Publish datasets to immutable content-addressed S3 keys.

    Existing keys are accepted only when their bytes match the local dataset.
    """
    from paper.benchmark.infra.aws import upload_datasets

    heading("Uploading Datasets to S3")

    if dry_run:
        info("Dry run - no files will be uploaded")

    with console.status("Scanning and uploading..."):
        result = upload_datasets(task=task, dry_run=dry_run)

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
            help="Immutable ECR image URI in repository@sha256:digest form",
        ),
    ] = "",
    artifact_prefix: Annotated[
        str,
        typer.Option(
            "--artifact-prefix",
            envvar="CITREES_ARTIFACT_PREFIX",
            help="Isolated S3 prefix for corrected artifacts",
        ),
    ] = "",
    launch_id: Annotated[
        str,
        typer.Option(
            "--launch-id",
            help="Unique identity for this API instance; replacements require a new value",
        ),
    ] = "",
    canonical_manifest_path: Annotated[
        Path | None,
        typer.Option(
            "--canonical-manifest",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Complete canonical manifest bound to the GO receipt",
        ),
    ] = None,
    manifest_path: Annotated[
        Path | None,
        typer.Option(
            "--manifest",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Private validated rerun manifest for this account shard",
        ),
    ] = None,
    runtime_contract_path: Annotated[
        Path | None,
        typer.Option(
            "--runtime-contract",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Canonical gate-approved runtime contract bound to the manifest",
        ),
    ] = None,
    gate_receipt_path: Annotated[
        Path | None,
        typer.Option(
            "--gate-receipt",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Complete immutable GO receipt for the reproducibility gate",
        ),
    ] = None,
    stage: Annotated[
        Literal["rankings", "metrics"],
        typer.Option(
            "--stage",
            envvar="CITREES_STAGE",
            help="Run rankings or metrics as a separate phase",
        ),
    ] = "rankings",
    lease_seconds: Annotated[
        int,
        typer.Option(
            "--lease-seconds",
            min=1,
            help="Assignment lease duration; workers heartbeat while a cell runs",
        ),
    ] = 900,
    max_cell_attempts: Annotated[
        int | None,
        typer.Option(
            "--max-cell-attempts",
            min=1,
            envvar="CITREES_MAX_CELL_ATTEMPTS",
            help="Explicit fixed attempt budget for each manifest cell",
        ),
    ] = None,
) -> None:
    """Launch an on-demand API server behind a stable private DNS endpoint.

    Workers use one campaign-scoped endpoint across API replacements.
    """
    from paper.benchmark.infra.ec2 import launch_api

    if (
        not image_uri
        or not artifact_prefix
        or not launch_id
        or canonical_manifest_path is None
        or manifest_path is None
        or runtime_contract_path is None
        or gate_receipt_path is None
        or max_cell_attempts is None
    ):
        error(
            "--image-uri, --artifact-prefix, --launch-id, --canonical-manifest, "
            "--manifest, --runtime-contract, --gate-receipt, and "
            "--max-cell-attempts are required"
        )
        raise typer.Exit(2)

    heading("Launching API Server")

    result = launch_api(
        instance_type=instance_type,
        image_uri=image_uri,
        artifact_prefix=artifact_prefix,
        launch_id=launch_id,
        canonical_manifest_path=canonical_manifest_path,
        gate_receipt_path=gate_receipt_path,
        manifest_path=manifest_path,
        runtime_contract_path=runtime_contract_path,
        stage=stage,
        lease_seconds=lease_seconds,
        max_cell_attempts=max_cell_attempts,
    )

    if result["api_url"]:
        console.print(f"\n  Worker API URL: [bold cyan]{result['api_url']}[/]")
        console.print(f"  Public API URL: [bold cyan]{result['public_api_url']}[/]")
        console.print("  Instance: " + result["instance_id"])


@app.command(name="api-url")
def api_url_cmd(
    artifact_prefix: Annotated[
        str,
        typer.Option(
            "--artifact-prefix",
            envvar="CITREES_ARTIFACT_PREFIX",
            help="Exact API server artifact prefix",
        ),
    ] = "",
    campaign_sha256: Annotated[
        str,
        typer.Option(
            "--campaign-sha256",
            envvar="CITREES_CAMPAIGN_SHA256",
            help="Exact API server campaign digest",
        ),
    ] = "",
    stage: Annotated[
        Literal["rankings", "metrics"],
        typer.Option(
            "--stage",
            envvar="CITREES_STAGE",
            help="Exact API server phase",
        ),
    ] = "rankings",
) -> None:
    """Print the public URL for one campaign-scoped API server."""
    from paper.benchmark.infra.ec2 import get_api_scope

    if not artifact_prefix or not campaign_sha256:
        error("--artifact-prefix and --campaign-sha256 are required")
        raise typer.Exit(2)

    scope = get_api_scope(
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        stage=stage,
    )
    if scope is not None:
        console.print(scope.public_api_url)
        return

    error("No running API server found for the exact campaign scope")
    raise typer.Exit(1)


@app.command(name="terminate-api")
def terminate_api_cmd(
    artifact_prefix: Annotated[
        str,
        typer.Option(
            "--artifact-prefix",
            envvar="CITREES_ARTIFACT_PREFIX",
            help="Exact API server artifact prefix",
        ),
    ] = "",
    campaign_sha256: Annotated[
        str,
        typer.Option(
            "--campaign-sha256",
            envvar="CITREES_CAMPAIGN_SHA256",
            help="Exact API server campaign digest",
        ),
    ] = "",
    stage: Annotated[
        Literal["rankings", "metrics"],
        typer.Option(
            "--stage",
            envvar="CITREES_STAGE",
            help="Exact API server phase",
        ),
    ] = "rankings",
) -> None:
    """Terminate one campaign-scoped API server."""
    from paper.benchmark.infra.ec2 import terminate_api

    if not artifact_prefix or not campaign_sha256:
        error("--artifact-prefix and --campaign-sha256 are required")
        raise typer.Exit(2)

    heading("Terminating API Server")

    result = terminate_api(
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        stage=stage,
    )
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
    image_uri: Annotated[
        str,
        typer.Option(
            "--image-uri",
            help="Immutable ECR image URI in repository@sha256:digest form",
        ),
    ] = "",
    artifact_prefix: Annotated[
        str,
        typer.Option(
            "--artifact-prefix",
            envvar="CITREES_ARTIFACT_PREFIX",
            help="Must match the API server artifact prefix",
        ),
    ] = "",
    canonical_manifest_path: Annotated[
        Path | None,
        typer.Option(
            "--canonical-manifest",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Complete canonical manifest bound to the GO receipt",
        ),
    ] = None,
    subnets: Annotated[
        str,
        typer.Option(
            "--subnets",
            help=(
                "Comma-separated default-VPC subnet IDs; workers rotate through "
                "them in the supplied order"
            ),
        ),
    ] = "",
    excluded_availability_zones: Annotated[
        str,
        typer.Option(
            "--exclude-availability-zones",
            help="Comma-separated availability zones that must not receive workers",
        ),
    ] = "",
    launch_id: Annotated[
        str,
        typer.Option(
            "--launch-id",
            help="Stable identity for this exact launch batch; reuse it only to recover",
        ),
    ] = "",
    manifest_path: Annotated[
        Path | None,
        typer.Option(
            "--manifest",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Private rerun manifest; must match the running API",
        ),
    ] = None,
    runtime_contract_path: Annotated[
        Path | None,
        typer.Option(
            "--runtime-contract",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Canonical gate-approved runtime contract; must match the running API",
        ),
    ] = None,
    gate_receipt_path: Annotated[
        Path | None,
        typer.Option(
            "--gate-receipt",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Complete immutable GO receipt; must match the running API campaign",
        ),
    ] = None,
    stage: Annotated[
        Literal["rankings", "metrics"],
        typer.Option(
            "--stage",
            envvar="CITREES_STAGE",
            help="Must match the API server phase",
        ),
    ] = "rankings",
) -> None:
    """Launch EC2 worker instances.

    Each instance pulls a Docker image from ECR and runs a worker process
    that gets work assignments from the API server's stable private endpoint.
    """
    from paper.benchmark.infra.ec2 import launch_workers

    if (
        not image_uri
        or not artifact_prefix
        or not launch_id
        or canonical_manifest_path is None
        or manifest_path is None
        or runtime_contract_path is None
        or gate_receipt_path is None
    ):
        error(
            "--image-uri, --artifact-prefix, --launch-id, --canonical-manifest, "
            "--manifest, --runtime-contract, and --gate-receipt are required"
        )
        raise typer.Exit(2)

    heading(f"Launching {n} Workers")

    launch_workers(
        n=n,
        image_uri=image_uri,
        artifact_prefix=artifact_prefix,
        canonical_manifest_path=canonical_manifest_path,
        excluded_availability_zones=_split_csv(excluded_availability_zones),
        gate_receipt_path=gate_receipt_path,
        launch_id=launch_id,
        manifest_path=manifest_path,
        runtime_contract_path=runtime_contract_path,
        stage=stage,
        subnet_ids=_split_csv(subnets),
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
    image_uri: Annotated[
        str,
        typer.Option(
            "--image-uri",
            help="Immutable ECR image URI in repository@sha256:digest form",
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
) -> None:
    """Launch sharded EC2 workers for CIF mechanism ablations.

    This command runs the paper-side mechanism-ablation runner directly on each
    instance, using stable modulo sharding. It does not require the API server.
    """
    from paper.benchmark.infra.ec2 import launch_mechanism_workers

    if source not in {"real", "synthetic", "all"}:
        error("source must be one of: real, synthetic, all")
        raise typer.Exit(1)

    if not image_uri:
        error("--image-uri is required")
        raise typer.Exit(2)

    heading(f"Launching {n} Mechanism Workers")

    launch_mechanism_workers(
        n=n,
        instance_type=instance_type,
        image_uri=image_uri,
        num_shards=num_shards or None,
        shard_start=shard_start,
        subnet_ids=_split_csv(subnets),
        tasks=_split_csv(tasks),
        source=source,
        datasets=_split_csv(datasets),
        seeds=_split_int_csv(seeds),
        folds=_split_int_csv(folds),
        model_variants=_split_csv(model_variants),
        ranking_variants=_split_csv(ranking_variants),
        n_jobs=n_jobs,
        downstream_n_jobs=downstream_n_jobs,
    )


@app.command(name="list-workers")
def list_workers_cmd(
    launch_id: Annotated[
        str,
        typer.Option(
            "--launch-id",
            help="Exact worker launch batch to list",
        ),
    ],
    artifact_prefix: Annotated[
        str,
        typer.Option(
            "--artifact-prefix",
            envvar="CITREES_ARTIFACT_PREFIX",
            help="Exact worker artifact prefix",
        ),
    ],
    campaign_sha256: Annotated[
        str,
        typer.Option(
            "--campaign-sha256",
            envvar="CITREES_CAMPAIGN_SHA256",
            help="Exact worker campaign digest",
        ),
    ],
    stage: Annotated[
        Literal["rankings", "metrics"],
        typer.Option(
            "--stage",
            envvar="CITREES_STAGE",
            help="Exact worker pipeline stage",
        ),
    ],
) -> None:
    """List running worker instances from one exact campaign launch."""
    from paper.benchmark.infra.ec2 import list_workers

    heading("Worker Instances")

    workers = list_workers(
        launch_id,
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        stage=stage,
    )
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
    from paper.benchmark.infra.ec2 import list_mechanism_workers

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
def terminate_workers_cmd(
    launch_id: Annotated[
        str,
        typer.Option(
            "--launch-id",
            help="Exact worker launch batch to terminate",
        ),
    ],
    artifact_prefix: Annotated[
        str,
        typer.Option(
            "--artifact-prefix",
            envvar="CITREES_ARTIFACT_PREFIX",
            help="Exact worker artifact prefix",
        ),
    ],
    campaign_sha256: Annotated[
        str,
        typer.Option(
            "--campaign-sha256",
            envvar="CITREES_CAMPAIGN_SHA256",
            help="Exact worker campaign digest",
        ),
    ],
    stage: Annotated[
        Literal["rankings", "metrics"],
        typer.Option(
            "--stage",
            envvar="CITREES_STAGE",
            help="Exact worker pipeline stage",
        ),
    ],
) -> None:
    """Terminate worker instances from one exact campaign launch."""
    from paper.benchmark.infra.ec2 import terminate_workers

    heading("Terminating Workers")

    terminated = terminate_workers(
        launch_id,
        artifact_prefix=artifact_prefix,
        campaign_sha256=campaign_sha256,
        stage=stage,
    )
    if terminated:
        success(f"Terminated {len(terminated)} instances")
    else:
        info("No workers to terminate")


@app.command(name="terminate-mechanism-workers")
def terminate_mechanism_workers_cmd() -> None:
    """Terminate all running CIF mechanism-ablation workers."""
    from paper.benchmark.infra.ec2 import terminate_mechanism_workers

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
    from paper.benchmark.infra.ec2 import get_logs

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
