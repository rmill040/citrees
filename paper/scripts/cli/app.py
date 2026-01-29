"""citrees-exp CLI - Experiment infrastructure for citrees.

This is the main entry point for the CLI. It provides commands for running
experiments, checking progress, and managing infrastructure.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Annotated, Literal, cast

import typer
from rich.progress import Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from paper.scripts.cli._console import (
    console,
    create_table,
    error,
    format_number,
    format_percent,
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


# Import and register subcommand groups lazily for faster startup
def _get_config_app() -> typer.Typer:
    from paper.scripts.cli.config import app as config_app

    return config_app


def _get_list_app() -> typer.Typer:
    from paper.scripts.cli.list import app as list_app

    return list_app


def _get_infra_app() -> typer.Typer:
    from paper.scripts.cli.infra import app as infra_app

    return infra_app


def _get_cluster_app() -> typer.Typer:
    from paper.scripts.cli.cluster import app as cluster_app

    return cluster_app


# Register subcommand groups
app.add_typer(_get_config_app(), name="config", help="Configuration management")
app.add_typer(_get_list_app(), name="list", help="List datasets and methods")
app.add_typer(_get_infra_app(), name="infra", help="AWS infrastructure setup")
app.add_typer(_get_cluster_app(), name="cluster", help="Ray cluster operations")


def _resolve_ray_address(address: str) -> str:
    """Resolve Ray address, auto-detecting cluster head if needed.

    - "local": returns as-is (local Ray mode)
    - "auto": gets head IP from cluster.yaml, returns ray://<ip>:10001
    - other: returns as-is (user-specified address)
    """
    if address == "local":
        return address

    if address == "auto":
        # Try to get head IP from cluster config
        from paper.scripts.cli.cluster import _get_head_ip

        head_ip = _get_head_ip()
        if head_ip:
            resolved = f"ray://{head_ip}:10001"
            info(f"Auto-detected Ray address: {resolved}")
            return resolved
        else:
            # Fall back to "auto" which lets Ray try to find a local cluster
            warn("Could not get head IP from cluster config, using local auto-detection")
            return "auto"

    return address


# Type aliases for common parameters
Task = Annotated[
    Literal["classification", "regression"],
    typer.Argument(
        help="Task type: classification or regression",
        metavar="TASK",
    ),
]

StageOption = Annotated[
    Literal["all", "stage1", "stage2"],
    typer.Option(
        "--stage",
        "-s",
        help="Stage to run: all, stage1, or stage2",
    ),
]


@app.command()
def run(
    task: Task,
    stage: StageOption = "all",
    methods: Annotated[
        str | None,
        typer.Option(
            "--methods",
            "-m",
            help="Comma-separated list of methods to run",
        ),
    ] = None,
    datasets: Annotated[
        str | None,
        typer.Option(
            "--datasets",
            "-d",
            help="Comma-separated list of datasets to run",
        ),
    ] = None,
    seeds: Annotated[
        str | None,
        typer.Option(
            "--seeds",
            help="Comma-separated list of seed indices to run",
        ),
    ] = None,
    source: Annotated[
        Literal["all", "real", "synthetic"],
        typer.Option(
            "--source",
            help="Dataset source filter: real, synthetic, or all",
        ),
    ] = "all",
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Re-run all configs, even if results already exist in S3",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show what would be run without executing",
        ),
    ] = False,
    dry_run_limit: Annotated[
        int,
        typer.Option(
            "--dry-run-limit",
            help="Max items to show in dry run",
        ),
    ] = 20,
    ray_address: Annotated[
        str,
        typer.Option(
            "--ray-address",
            help="Ray cluster address (default: auto)",
        ),
    ] = "auto",
    max_configs_per_method: Annotated[
        int | None,
        typer.Option(
            "--max-configs-per-method",
            help="Limit configs per method (useful for testing)",
        ),
    ] = None,
) -> None:
    """Run experiment pipeline (Stage 1 rankings + Stage 2 metrics).

    By default, skips configs that already have results in S3.
    Use --force to re-run all configs.

    \b
    Examples:
        citrees-exp run classification
        citrees-exp run classification --stage stage1
        citrees-exp run regression -m cit,boruta --dry-run
        citrees-exp run classification --force  # re-run everything
    """
    from paper.scripts.adapters import RayRunner, S3Store, get_dataset_shape
    from paper.scripts.config import load_config
    from paper.scripts.pipeline import ExperimentGrid
    from paper.scripts.pipeline.stage1 import selection_num_cpus
    from paper.scripts.pipeline.stage2 import evaluation_num_cpus

    heading("Experiment Pipeline")

    cfg = load_config()

    info(f"Task: {task}")
    info(f"Stage: {stage}")
    if dry_run:
        info("[dim](dry run mode)[/]")
    console.print()

    # Build the experiment grid
    try:
        grid = ExperimentGrid.from_cli(
            task=task,
            methods=methods,
            datasets=datasets,
            seeds=seeds,
            source=cast(Literal["all", "real", "synthetic"], source),
            n_seeds=cfg.experiment.n_seeds,
            max_configs_per_method=max_configs_per_method,
        )
    except ValueError as e:
        error(str(e))
        raise SystemExit(1) from e

    # Show grid summary
    summary = grid.summary()
    table = create_table(
        title="Experiment Grid",
        columns=[("Parameter", ""), ("Value", "number")],
    )
    table.add_row("Methods", format_number(summary["methods"]))
    table.add_row("Datasets", format_number(summary["datasets"]))
    table.add_row("Seeds", format_number(summary["seeds"]))
    table.add_row("Total configs", format_number(summary["total"]))
    console.print(table)
    console.print()

    # Check S3 for completed items (skip if --force)
    store: S3Store | None = None
    completed_rankings: set[tuple[str, str, int]] = set()
    completed_metrics: set[tuple[str, str, int]] = set()

    if not force:
        store = S3Store.from_config()
        with console.status("Checking S3 for completed items..."):
            if stage in {"all", "stage1", "stage2"}:
                completed_rankings = store.list_completed("rankings", task)
            if stage in {"all", "stage2"}:
                completed_metrics = store.list_completed("metrics", task)
        info(f"Found {len(completed_rankings)} completed rankings")
        info(f"Found {len(completed_metrics)} completed metrics")
        console.print()

    # Get pending items
    stage1_configs = grid.filter_pending(completed_rankings) if not force else grid.as_list()

    # Dry run output
    if dry_run:
        heading("Dry Run")

        # Precompute dataset shapes for resource estimation
        dataset_shapes = {d: get_dataset_shape(d, task) for d in grid.datasets}

        if stage in {"all", "stage1"}:
            console.print(f"[bold]Stage 1 (Rankings):[/] {len(stage1_configs)} configs")
            for i, cfg in enumerate(stage1_configs[:dry_run_limit]):
                n_samples, n_features = dataset_shapes[cfg.dataset]
                cpus = selection_num_cpus(
                    cfg.method.name, n_samples=n_samples, n_features=n_features
                )
                console.print(f"  {i + 1}. {cfg} (cpus={cpus})")
            if len(stage1_configs) > dry_run_limit:
                console.print(f"  [dim]... and {len(stage1_configs) - dry_run_limit} more[/]")
            console.print()

        if stage in {"all", "stage2"}:
            if not force:
                available = completed_rankings | {cfg.key for cfg in stage1_configs}
                stage2_configs = [
                    cfg for cfg in grid if cfg.key not in completed_metrics and cfg.key in available
                ]
            else:
                stage2_configs = grid.as_list()

            cpus = evaluation_num_cpus(task)
            console.print(
                f"[bold]Stage 2 (Metrics):[/] {len(stage2_configs)} configs (cpus={cpus})"
            )
            for i, cfg in enumerate(stage2_configs[:dry_run_limit]):
                console.print(f"  {i + 1}. {cfg}")
            if len(stage2_configs) > dry_run_limit:
                console.print(f"  [dim]... and {len(stage2_configs) - dry_run_limit} more[/]")

        return

    # Actual execution
    if store is None:
        store = S3Store.from_config()

    runner = RayRunner(address=_resolve_ray_address(ray_address))

    with console.status("Initializing Ray..."):
        runner.init()
    success("Ray initialized")

    # Stage 1: Feature Selection
    if stage in {"all", "stage1"} and stage1_configs:
        heading("Stage 1: Feature Selection")
        info(f"Running {len(stage1_configs)} tasks")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Selecting features...", total=len(stage1_configs))

            def on_complete_s1(cfg, result):
                progress.advance(task_id)
                if result.is_failure:
                    console.print(f"[red]  x {cfg}[/]")

            results = runner.run("rankings", stage1_configs, store, on_complete_s1)

        done_count = sum(1 for r in results if r.status == "done")
        skipped_count = sum(1 for r in results if r.status == "skipped")
        failed_count = sum(1 for r in results if r.status == "failed")
        if failed_count:
            warn(f"Stage 1: {failed_count} failures")
        success(f"Stage 1 complete: {done_count} done, {skipped_count} skipped")

        # Track newly completed for stage2
        for r in results:
            if r.status in {"done", "skipped"}:
                completed_rankings.add(r.config.key)

    # Stage 2: Evaluation
    if stage in {"all", "stage2"}:
        heading("Stage 2: Evaluation")

        if not force:
            stage2_configs = [
                cfg
                for cfg in grid
                if cfg.key not in completed_metrics and cfg.key in completed_rankings
            ]
        elif stage == "stage2":
            stage2_configs = grid.as_list()
        else:
            # After stage1, run what we just completed
            stage2_configs = [cfg for cfg in grid if cfg.key in completed_rankings]

        if not stage2_configs:
            warn("No Stage 2 items to run")
        else:
            info(f"Running {len(stage2_configs)} tasks")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task("Evaluating...", total=len(stage2_configs))

                def on_complete_s2(cfg, result):
                    progress.advance(task_id)
                    if result.is_failure:
                        console.print(f"[red]  x {cfg}[/]")

                results = runner.run("metrics", stage2_configs, store, on_complete_s2)

            done_count = sum(1 for r in results if r.status == "done")
            skipped_count = sum(1 for r in results if r.status == "skipped")
            no_rankings_count = sum(1 for r in results if r.status == "no_rankings")
            failed_count = sum(1 for r in results if r.status == "failed")
            if failed_count:
                warn(f"Stage 2: {failed_count} failures")
            success(
                f"Stage 2 complete: {done_count} done, {skipped_count} skipped, {no_rankings_count} no-rankings"
            )

    success("Pipeline complete")


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
    ray_address: Annotated[
        str,
        typer.Option(
            "--ray-address",
            help="Ray cluster address (use 'local' for local mode)",
        ),
    ] = "auto",
    max_configs_per_method: Annotated[
        int | None,
        typer.Option(
            "--max-configs-per-method",
            help="Limit configs per method (useful for testing)",
        ),
    ] = None,
) -> None:
    """Run a minimal smoke test of the experiment pipeline.

    Validates:
    - Ray scheduling works
    - S3 read/write paths are correct
    - Artifact schemas are valid

    \b
    Examples:
        citrees-exp smoke classification
        citrees-exp smoke regression --methods pc,rf --dataset cpu_act
    """
    from paper.scripts.adapters import RayRunner, S3Store, get_dataset_shape, get_datasets
    from paper.scripts.pipeline import ExperimentGrid

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

    store = S3Store.from_config()
    info(f"S3: {store.bucket}")
    console.print()

    runner = RayRunner(address=_resolve_ray_address(ray_address))

    with console.status("Initializing Ray..."):
        runner.init()
    success("Ray initialized")

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
    for cfg in configs:
        with console.status(f"Checking {cfg.method.label}..."):
            rdf = store.load("rankings", cfg)
            mdf = store.load("metrics", cfg)
        success(f"{cfg.method.label}: {len(rdf)} rankings, {len(mdf)} metrics")

    success("Smoke test passed")


@app.command()
def check(
    stage: Annotated[
        Literal["rankings", "metrics"],
        typer.Option(
            "--stage",
            "-s",
            help="Stage to check: rankings or metrics",
        ),
    ] = "rankings",
    task: Annotated[
        Literal["classification", "regression"],
        typer.Option(
            "--task",
            "-t",
            help="Task type: classification or regression",
        ),
    ] = "classification",
    by_method: Annotated[
        bool,
        typer.Option(
            "--by-method",
            help="Show progress grouped by method",
        ),
    ] = False,
    by_dataset: Annotated[
        bool,
        typer.Option(
            "--by-dataset",
            help="Show progress grouped by dataset",
        ),
    ] = False,
    synthetic_only: Annotated[
        bool,
        typer.Option(
            "--synthetic-only",
            help="Only show synthetic datasets",
        ),
    ] = False,
) -> None:
    """Check experiment progress from S3.

    \b
    Examples:
        citrees-exp check
        citrees-exp check --stage metrics --by-method
        citrees-exp check --by-dataset --synthetic-only
    """
    from paper.scripts.adapters import S3Store, get_datasets
    from paper.scripts.config import load_config
    from paper.scripts.pipeline import ExperimentGrid

    cfg = load_config()
    n_seeds = cfg.experiment.n_seeds

    # Build full grid to know expected counts
    all_datasets = get_datasets(task)
    grid = ExperimentGrid.from_cli(task=task, n_seeds=n_seeds)

    if synthetic_only:
        all_datasets = [d for d in all_datasets if "synthetic" in d]

    method_labels = [m.label for m in grid.methods]
    total_expected = len(all_datasets) * len(method_labels) * n_seeds

    heading(f"Progress: {stage.upper()} ({task})")

    store = S3Store.from_config()
    info(f"S3: {store.bucket}")
    console.print(
        f"  Expected: {len(all_datasets)} datasets x {len(method_labels)} methods x {n_seeds} seeds = {format_number(total_expected)}"
    )
    console.print()

    # Fetch completed from S3
    with console.status(f"Listing s3://{store.bucket}/{stage}/{task}/..."):
        completed = store.list_completed(stage, task)

    # Organize by dataset
    completed_by_dataset: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for method_label, dataset, seed in completed:
        completed_by_dataset[dataset].add((method_label, seed))

    total_done = len(completed)
    pct = format_percent(total_done, total_expected)
    bar = progress_bar(total_done, total_expected, width=30)

    console.print(f"  {bar} {format_number(total_done)} / {format_number(total_expected)} ({pct})")
    console.print(f"  Remaining: {format_number(total_expected - total_done)}")
    console.print()

    if by_method:
        heading("By Method")
        method_counts: dict[str, int] = defaultdict(int)
        for method_label, _ds, _seed in completed:
            method_counts[method_label] += 1

        expected_per = len(all_datasets) * n_seeds
        table = create_table(columns=[("Method", ""), ("Progress", ""), ("Done", "number")])
        for m in method_labels:
            cnt = method_counts.get(m, 0)
            table.add_row(m[:30], progress_bar(cnt, expected_per), f"{cnt}/{expected_per}")
        console.print(table)
        console.print()

    if by_dataset:
        heading("By Dataset")
        expected_per = len(method_labels) * n_seeds
        incomplete = [
            (d, len(completed_by_dataset.get(d, set())))
            for d in all_datasets
            if len(completed_by_dataset.get(d, set())) < expected_per
        ]
        complete = [
            d for d in all_datasets if len(completed_by_dataset.get(d, set())) >= expected_per
        ]

        if incomplete:
            table = create_table(
                title=f"Incomplete ({len(incomplete)})",
                columns=[("Dataset", ""), ("Progress", ""), ("Done", "number")],
            )
            for d, cnt in sorted(incomplete, key=lambda x: x[1]):
                table.add_row(d[:40], progress_bar(cnt, expected_per), f"{cnt}/{expected_per}")
            console.print(table)

        console.print(f"\n  [success]Complete:[/] {len(complete)} datasets")


@app.command()
def watch() -> None:
    """Live dashboard showing experiment progress.

    Updates every second with progress bars for each method and dataset.
    Press Ctrl+C to exit.
    """
    from paper.scripts.cli.watch import run_watch

    run_watch()


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
