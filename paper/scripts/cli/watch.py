"""Live progress dashboard for experiments.

Uses Rich Live display to show real-time experiment progress.
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from paper.scripts.cli._console import console, format_number, progress_bar


def run_watch() -> None:
    """Run the live progress dashboard.

    Updates every 5 seconds with progress bars for each stage.
    Press Ctrl+C to exit.
    """
    from paper.scripts.adapters import S3Store, get_datasets
    from paper.scripts.config import load_config
    from paper.scripts.pipeline import expand_method_configs, get_methods

    config = load_config()
    n_seeds = config.experiment.n_seeds
    task_type = config.experiment.type
    store = S3Store.from_config()
    bucket = store.bucket

    # Get expected counts
    dataset_list = get_datasets(task_type)  # type: ignore
    method_list = get_methods(task_type)
    method_configs = expand_method_configs(method_list)
    method_labels = [cfg.label for cfg in method_configs]

    def make_layout() -> Layout:
        """Create the dashboard layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )
        layout["main"].split_row(
            Layout(name="rankings", ratio=1),
            Layout(name="metrics", ratio=1),
        )
        return layout

    def make_header() -> Panel:
        """Create the header panel."""
        from rich.text import Text

        header = Text()
        header.append("citrees-exp", style="bold cyan")
        header.append(" | ", style="dim")
        header.append(f"task={task_type}", style="green")
        header.append(" | ", style="dim")
        header.append(f"bucket={bucket}", style="yellow")
        return Panel(header, style="blue")

    def make_footer() -> Panel:
        """Create the footer panel."""
        from rich.text import Text

        footer = Text()
        footer.append(f"Last updated: {datetime.now().strftime('%H:%M:%S')}", style="dim")
        footer.append(" | ", style="dim")
        footer.append("Press Ctrl+C to exit", style="dim italic")
        return Panel(footer, style="dim")

    def make_progress_table(
        title: str,
        completed: dict[str, set[tuple[str, int]]],
        method_labels: list[str],
        datasets: list[str],
        n_seeds: int,
    ) -> Table:
        """Create a progress table for a stage."""
        from rich import box

        total_completed = sum(len(items) for items in completed.values())
        total_expected = len(datasets) * len(method_labels) * n_seeds

        pct = 100 * total_completed / total_expected if total_expected > 0 else 0
        bar = progress_bar(total_completed, total_expected, width=20)

        table = Table(
            title=f"{title} [{bar}] {pct:.1f}%",
            box=box.ROUNDED,
            expand=True,
            show_header=True,
            header_style="bold",
        )
        table.add_column("Category", style="cyan")
        table.add_column("Progress", justify="right")

        # Summary row
        table.add_row(
            "Total",
            f"{format_number(total_completed)} / {format_number(total_expected)}",
        )

        # Top 5 methods by count
        method_counts: dict[str, int] = defaultdict(int)
        for dataset_items in completed.values():
            for method, _ in dataset_items:
                method_counts[method] += 1

        top_methods = sorted(method_counts.items(), key=lambda x: -x[1])[:5]
        expected_per_method = len(datasets) * n_seeds

        for method, count in top_methods:
            bar = progress_bar(count, expected_per_method, width=10)
            table.add_row(
                method[:20],
                f"{bar} {count}/{expected_per_method}",
            )

        return table

    def fetch_progress(stage: str, task_type: str) -> dict[str, set[tuple[str, int]]]:
        """Fetch progress from S3 (silently)."""
        import boto3

        from paper.scripts.config.constants import AWS_REGION

        s3 = boto3.client("s3", region_name=AWS_REGION)
        completed: dict[str, set[tuple[str, int]]] = defaultdict(set)
        prefix = f"{stage}/{task_type}/"

        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    parts = key.split("/")
                    if len(parts) >= 4:
                        dataset = parts[2]
                        filename = parts[3]
                        method_seed = filename.replace(".parquet", "")
                        if "_seed" in method_seed:
                            method, seed_str = method_seed.rsplit("_seed", 1)
                            seed = int(seed_str)
                            completed[dataset].add((method, seed))
        except Exception:
            pass  # Silently ignore errors in live update

        return completed

    layout = make_layout()

    console.print("\n[bold cyan]Starting live dashboard...[/]")
    console.print("[dim]Fetching initial data from S3...[/]\n")

    try:
        with Live(layout, console=console, refresh_per_second=1):
            while True:
                # Update header
                layout["header"].update(make_header())

                # Fetch progress
                rankings = fetch_progress("rankings", task_type)
                metrics = fetch_progress("metrics", task_type)

                # Update progress panels
                rankings_table = make_progress_table(
                    "Stage 1: Rankings",
                    rankings,
                    method_labels,
                    dataset_list,
                    n_seeds,
                )
                metrics_table = make_progress_table(
                    "Stage 2: Metrics",
                    metrics,
                    method_labels,
                    dataset_list,
                    n_seeds,
                )

                layout["rankings"].update(
                    Panel(rankings_table, title="Rankings", border_style="blue")
                )
                layout["metrics"].update(
                    Panel(metrics_table, title="Metrics", border_style="green")
                )

                # Update footer
                layout["footer"].update(make_footer())

                # Wait before next update
                time.sleep(5)

    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard stopped.[/]")
