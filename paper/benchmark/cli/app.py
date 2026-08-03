"""citrees-exp CLI - Experiment infrastructure for citrees.

This is the main entry point for the CLI. It provides commands for running
experiments, checking progress, and managing infrastructure.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import httpx
import typer

from paper.benchmark.cli.console_output import (
    console,
    create_table,
    error,
    format_number,
    heading,
    info,
    progress_bar,
    success,
    warn,
)

# Create the main app
app = typer.Typer(
    name="citrees-exp",
    help="[bold cyan]citrees[/] experiment infrastructure CLI",
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _get_list_app() -> typer.Typer:
    from paper.benchmark.cli.list import app as list_app

    return list_app


def _get_infra_app() -> typer.Typer:
    from paper.benchmark.cli.infra import app as infra_app

    return infra_app


def _get_cluster_app() -> typer.Typer:
    from paper.benchmark.cli.cluster import app as cluster_app

    return cluster_app


def _get_manifest_app() -> typer.Typer:
    from paper.benchmark.cli.manifest import app as manifest_app

    return manifest_app


app.add_typer(_get_list_app(), name="list", help="List datasets and methods")
app.add_typer(_get_infra_app(), name="infra", help="AWS infrastructure setup")
app.add_typer(_get_cluster_app(), name="cluster", help="API server and worker operations")
app.add_typer(_get_manifest_app(), name="manifest", help="Account-bound rerun manifests")


def _poll_status(api_url: str) -> dict[str, Any]:
    """Get full queue status from the API."""
    client = httpx.Client(base_url=api_url, timeout=10.0)
    try:
        resp = client.get("/status")
        resp.raise_for_status()
        return resp.json()
    finally:
        client.close()


@app.command()
def run(
    api_url: Annotated[
        str,
        typer.Option(
            "--api-url",
            help="API server URL (default: $CITREES_API_URL or http://localhost:8000)",
            envvar="CITREES_API_URL",
        ),
    ] = "http://localhost:8000",
    poll_interval: Annotated[
        float,
        typer.Option(
            "--poll-interval",
            help="Seconds between status polls",
        ),
    ] = 10.0,
) -> None:
    """Poll the API server and display queue progress until all queues are empty.

    The server auto-populates its queues on startup from S3. This command
    just connects and shows live progress. Press Ctrl+C to stop polling.

    Examples:
        citrees-exp run
        citrees-exp run --api-url http://api-host:8000
        citrees-exp run --poll-interval 5
    """
    heading("Experiment Pipeline — Status Poller")
    info(f"API: {api_url}")
    console.print()

    while True:
        try:
            data = _poll_status(api_url)
        except httpx.ConnectError:
            warn(f"Cannot reach API at {api_url}, retrying...")
            time.sleep(poll_interval)
            continue

        queues = data.get("queues", {})
        total_pending = sum(q.get("pending", 0) for q in queues.values())
        total_failed = sum(q.get("failed", 0) for q in queues.values())

        table = create_table(
            title="Queue Status",
            columns=[
                ("Queue", ""),
                ("Available", "number"),
                ("In flight", "number"),
                ("Pending", "number"),
                ("Completed", "number"),
                ("Failed", "number"),
                ("Initial", "number"),
                ("Progress", ""),
            ],
        )
        for name, counts in queues.items():
            available = counts.get("available", 0)
            in_flight = counts.get("in_flight", 0)
            pending = counts.get("pending", 0)
            completed = counts.get("completed", 0)
            failed = counts.get("failed", 0)
            initial = counts.get("initial", 0)
            done = completed + failed
            bar = progress_bar(done, initial) if initial > 0 else ""
            table.add_row(
                name,
                str(available),
                str(in_flight),
                str(pending),
                str(completed),
                str(failed),
                str(initial),
                bar,
            )
        console.print(table)

        if total_pending == 0:
            if total_failed:
                error(f"Queues terminated with {format_number(total_failed)} failed cells")
                raise typer.Exit(1)
            success("All queues empty")
            break

        info(f"Total pending: {format_number(total_pending)}")
        console.print()
        time.sleep(poll_interval)


@app.command()
def smoke(
    task: Annotated[
        Literal["classification", "regression"],
        typer.Argument(
            help="Task type: classification or regression",
            metavar="TASK",
        ),
    ] = "classification",
    methods: Annotated[
        str | None,
        typer.Option(
            "--methods",
            "-m",
            help="Comma-separated list of methods (default: mc,rf for clf; pc,rf for reg)",
        ),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Dataset name (default: smallest available)",
        ),
    ] = None,
    source: Annotated[
        Literal["all", "real", "synthetic"],
        typer.Option(
            "--source",
            help="Dataset source filter: real, synthetic, or all",
        ),
    ] = "real",
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Seed index",
        ),
    ] = 0,
    max_configs_per_method: Annotated[
        int | None,
        typer.Option(
            "--max-configs-per-method",
            help="Limit configs per method (useful for testing)",
        ),
    ] = None,
) -> None:
    """Run a minimal smoke test of the experiment pipeline.

    Uses LocalRunner for direct sequential execution (no API server needed).
    Validates:
    - Pipeline execution works
    - S3 read/write paths are correct
    - Artifact schemas are valid

    Examples:
        citrees-exp smoke classification
        citrees-exp smoke regression --methods pc,rf --dataset cpu_act
    """
    from paper.benchmark.adapters import LocalRunner, S3Store, get_dataset_shape, get_datasets
    from paper.benchmark.pipeline import ExperimentGrid

    heading("Smoke Test")

    # Pick dataset (smallest one)
    all_datasets = get_datasets(task, source=cast(Literal["real", "synthetic", "all"], source))
    if not all_datasets:
        error(f"No datasets for task={task}, source={source}")
        raise SystemExit(1)

    if dataset:
        chosen = dataset
    else:
        shapes = {d: get_dataset_shape(d, task) for d in all_datasets}
        chosen = min(all_datasets, key=lambda d: shapes[d][0] * shapes[d][1])

    # Pick methods
    method_str = methods or ("mc,rf" if task == "classification" else "pc,rf")

    info(f"Task: {task}")
    info(f"Dataset: {chosen}")
    info(f"Methods: {method_str}")
    info(f"Seed: {seed}")
    console.print()

    # Build minimal grid
    grid = ExperimentGrid.from_cli(
        task=task,
        methods=method_str,
        datasets=chosen,
        seeds=str(seed),
        n_seeds=5,
        max_configs_per_method=max_configs_per_method,
    )

    store = S3Store.from_env(validate_uploads=True)
    info(f"S3: {store.bucket}")
    console.print()

    runner = LocalRunner()

    # Stage 1
    heading("Stage 1: Rankings")
    configs = grid.as_list()

    with console.status(f"Running {len(configs)} tasks..."):
        results = runner.run("rankings", configs, store)

    failures = [r for r in results if r.is_failure]
    if failures:
        error(f"Stage 1 failures: {[str(r.config) for r in failures]}")
        raise SystemExit(1)
    success(f"Stage 1: {len(results)} done")

    # Stage 2
    heading("Stage 2: Metrics")

    with console.status(f"Running {len(configs)} tasks..."):
        results = runner.run("metrics", configs, store)

    failures = [r for r in results if r.is_failure]
    if failures:
        error(f"Stage 2 failures: {[str(r.config) for r in failures]}")
        raise SystemExit(1)
    success(f"Stage 2: {len(results)} done")

    # Validate
    heading("Validating Artifacts")
    for cfg_item in configs:
        with console.status(f"Checking {cfg_item.method.label}..."):
            rdf = store.load("rankings", cfg_item)
            mdf = store.load("metrics", cfg_item)
        success(f"{cfg_item.method.label}: {len(rdf)} rankings, {len(mdf)} metrics")

    success("Smoke test passed")


@app.command()
def check(
    manifest_path: Annotated[
        Path,
        typer.Option(
            "--manifest",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Account-bound manifest for this artifact namespace",
        ),
    ],
    stage: Annotated[
        Literal["all", "rankings", "metrics"],
        typer.Option(
            "--stage",
            help="Artifact stage to reconcile",
        ),
    ] = "all",
) -> None:
    """Fail unless manifest-required artifacts are exact and valid."""
    from paper.benchmark.cli.manifest import reconcile_manifest

    reconcile_manifest(manifest_path, stage)


def _version_callback(value: bool) -> None:
    """Print version and exit if --version flag is passed."""
    if value:
        import citrees

        ver = getattr(citrees, "__version__", "unknown")
        console.print(f"citrees-exp version: citrees {ver}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """citrees experiment infrastructure CLI."""
    if ctx.invoked_subcommand is None and not version:
        console.print(ctx.get_help())
        raise typer.Exit()


if __name__ == "__main__":
    app()
