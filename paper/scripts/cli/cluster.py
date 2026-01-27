"""Ray cluster operations.

Commands for managing Ray clusters on AWS.
"""

from __future__ import annotations

import subprocess
import webbrowser
from pathlib import Path
from typing import Annotated

import typer

from paper.scripts.cli._console import console, create_table, error, heading, info, success, warn

app = typer.Typer(
    name="cluster",
    help="Ray cluster operations",
    no_args_is_help=True,
)

# Path to cluster config
RAY_DIR = Path(__file__).parent.parent / "infra" / "ray"
CLUSTER_YAML = RAY_DIR / "cluster.yaml"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _check_cluster_config() -> None:
    """Check that cluster.yaml exists."""
    if not CLUSTER_YAML.exists():
        error(f"Cluster config not found: {CLUSTER_YAML}")
        error("Run 'citrees-exp infra generate' first.")
        raise typer.Exit(1)


def _run_ray_command(
    args: list[str],
    description: str,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """Run a ray command with the cluster config."""
    cmd = ["ray"] + args
    if "--" in cmd:
        cmd.insert(cmd.index("--"), str(CLUSTER_YAML))
    else:
        cmd.append(str(CLUSTER_YAML))
    info(f"Running: {' '.join(cmd)}")

    if capture:
        return subprocess.run(cmd, capture_output=True, text=True)
    else:
        return subprocess.run(cmd)


@app.command()
def up(
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip confirmation prompts",
        ),
    ] = False,
    no_restart: Annotated[
        bool,
        typer.Option(
            "--no-restart",
            help="Do not restart Ray on the head node",
        ),
    ] = False,
) -> None:
    """Start the Ray cluster.

    Provisions head node and worker nodes according to cluster.yaml.

    \b
    Examples:
        citrees-exp cluster up
        citrees-exp cluster up --yes
    """
    _check_cluster_config()

    heading("Starting Ray cluster")

    args = ["up"]
    if yes:
        args.append("--yes")
    if no_restart:
        args.append("--no-restart")

    result = _run_ray_command(args, "Starting cluster")

    if result.returncode == 0:
        success("Cluster started")
        console.print("\nNext steps:")
        console.print("  [cyan]citrees-exp cluster status[/] - Check cluster status")
        console.print("  [cyan]citrees-exp cluster dashboard[/] - Open Ray dashboard")
    else:
        error("Failed to start cluster")
        raise typer.Exit(1)


@app.command()
def down(
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip confirmation prompts",
        ),
    ] = False,
    workers_only: Annotated[
        bool,
        typer.Option(
            "--workers-only",
            help="Only teardown workers, keep head node",
        ),
    ] = False,
) -> None:
    """Teardown the Ray cluster.

    Terminates all cluster nodes and cleans up resources.

    \b
    Examples:
        citrees-exp cluster down
        citrees-exp cluster down --yes
        citrees-exp cluster down --workers-only
    """
    _check_cluster_config()

    heading("Tearing down Ray cluster")

    args = ["down"]
    if yes:
        args.append("--yes")
    if workers_only:
        args.append("--workers-only")

    result = _run_ray_command(args, "Tearing down cluster")

    if result.returncode == 0:
        success("Cluster terminated")
    else:
        error("Failed to teardown cluster")
        raise typer.Exit(1)


@app.command()
def status() -> None:
    """Show cluster status.

    Displays information about running nodes and resources.
    """
    _check_cluster_config()

    heading("Cluster Status")

    # Run ray status
    result = subprocess.run(
        ["ray", "status", "--address", "auto"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        console.print(result.stdout)
    else:
        # Try to get status from cluster directly
        result = _run_ray_command(
            ["exec", "--", "ray", "status"],
            "Getting cluster status",
            capture=True,
        )
        if result.returncode == 0:
            console.print(result.stdout)
        else:
            warn("Could not get cluster status")
            console.print("[dim]Cluster may not be running or not accessible.[/]")


@app.command()
def dashboard(
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Local port for port forwarding",
        ),
    ] = 8265,
) -> None:
    """Open Ray dashboard in browser.

    Sets up port forwarding to the head node and opens the dashboard.
    """
    _check_cluster_config()

    heading("Opening Ray Dashboard")

    url = f"http://localhost:{port}"
    info(f"Dashboard URL: {url}")

    # Start port forwarding in background
    console.print("\n[dim]Starting port forwarding (Ctrl+C to stop)...[/]")

    try:
        # Open browser after a short delay
        import threading
        import time

        def open_browser():
            time.sleep(2)
            webbrowser.open(url)

        threading.Thread(target=open_browser, daemon=True).start()

        # Run port forward
        subprocess.run(
            [
                "ray",
                "dashboard",
                str(CLUSTER_YAML),
                "--port",
                str(port),
            ]
        )
    except KeyboardInterrupt:
        console.print("\n[dim]Port forwarding stopped.[/]")


@app.command()
def ssh() -> None:
    """SSH to the head node.

    Opens an interactive SSH session to the cluster head node.
    """
    _check_cluster_config()

    heading("Connecting to head node")

    subprocess.run(
        [
            "ray",
            "attach",
            str(CLUSTER_YAML),
            "--no-config-cache",
        ]
    )


@app.command()
def logs(
    follow: Annotated[
        bool,
        typer.Option(
            "--follow",
            "-f",
            help="Stream logs continuously",
        ),
    ] = False,
    lines: Annotated[
        int,
        typer.Option(
            "--lines",
            "-n",
            help="Number of lines to show",
        ),
    ] = 100,
) -> None:
    """View Ray cluster logs.

    Shows logs from the head node. Use --follow to stream continuously.
    """
    _check_cluster_config()

    heading("Cluster Logs")

    if follow:
        # Tail logs continuously
        subprocess.run(
            [
                "ray",
                "exec",
                str(CLUSTER_YAML),
                "--",
                "tail",
                "-f",
                "/tmp/ray/session_latest/logs/raylet.out",
            ]
        )
    else:
        # Show last N lines
        result = subprocess.run(
            [
                "ray",
                "exec",
                str(CLUSTER_YAML),
                "--",
                "tail",
                "-n",
                str(lines),
                "/tmp/ray/session_latest/logs/raylet.out",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            console.print(result.stdout)
        else:
            warn("Could not fetch logs")
            if result.stderr:
                console.print(f"[dim]{result.stderr}[/]")


@app.command()
def attach() -> None:
    """Attach to running cluster session.

    Attaches to the head node with screen session support.
    """
    _check_cluster_config()

    heading("Attaching to cluster")

    subprocess.run(
        [
            "ray",
            "attach",
            str(CLUSTER_YAML),
            "--start",
        ]
    )


@app.command()
def estimate() -> None:
    """Estimate AWS costs for the cluster.

    Parses cluster.yaml and estimates hourly/daily costs based on
    instance types and spot pricing.
    """
    _check_cluster_config()

    heading("Cost Estimate")

    import yaml

    with open(CLUSTER_YAML) as f:
        config = yaml.safe_load(f)

    region = config.get("provider", {}).get("region", "us-east-1")

    # Instance type pricing (approximate on-demand prices)
    # In production, you'd fetch these from AWS Pricing API
    INSTANCE_PRICES = {
        "m5.large": 0.096,
        "m5.xlarge": 0.192,
        "m5.2xlarge": 0.384,
        "m5.4xlarge": 0.768,
        "m5.8xlarge": 1.536,
        "m5.12xlarge": 2.304,
        "m5.16xlarge": 3.072,
        "m5.24xlarge": 4.608,
        "c5.large": 0.085,
        "c5.xlarge": 0.170,
        "c5.2xlarge": 0.340,
        "c5.4xlarge": 0.680,
        "c5.9xlarge": 1.530,
        "c5.18xlarge": 3.060,
        "r5.large": 0.126,
        "r5.xlarge": 0.252,
        "r5.2xlarge": 0.504,
        "r5.4xlarge": 1.008,
        "r5.8xlarge": 2.016,
        "r5.12xlarge": 3.024,
        "r5.16xlarge": 4.032,
        "r5.24xlarge": 6.048,
    }

    table = create_table(
        title=f"Cost Estimate ({region})",
        columns=[
            ("Node Type", ""),
            ("Instance", ""),
            ("Count", "number"),
            ("Spot", ""),
            ("$/hr (each)", "number"),
            ("$/hr (total)", "number"),
        ],
    )

    total_hourly = 0.0

    for node_type, node_config in config.get("available_node_types", {}).items():
        nc = node_config.get("node_config", {})
        instance_type = nc.get("InstanceType", "unknown")
        is_spot = "InstanceMarketOptions" in nc

        # Get min/max workers
        if node_type == config.get("head_node_type"):
            count = 1
        else:
            count = node_config.get("max_workers", 0)

        # Get price
        base_price = INSTANCE_PRICES.get(instance_type, 0.0)
        # Spot is typically 60-70% cheaper
        price = base_price * 0.3 if is_spot else base_price
        total = price * count

        total_hourly += total

        table.add_row(
            node_type,
            instance_type,
            str(count),
            "\u2713" if is_spot else "",
            f"${price:.3f}",
            f"${total:.2f}",
        )

    console.print(table)

    console.print(f"\n  [muted]Estimated hourly cost:[/] [highlight]${total_hourly:.2f}/hr[/]")
    console.print(f"  [muted]Estimated daily cost:[/] [highlight]${total_hourly * 24:.2f}/day[/]")
    console.print("\n  [dim]Note: Actual costs may vary. Spot prices fluctuate.[/]")


@app.command()
def submit(
    script: Annotated[
        str,
        typer.Argument(
            help="Script or command to submit",
        ),
    ],
    args: Annotated[
        list[str] | None,
        typer.Argument(
            help="Additional arguments for the script",
        ),
    ] = None,
) -> None:
    """Submit a job to the cluster.

    Runs a script on the head node. The script should be a path relative
    to the repository root or an absolute path.

    \b
    Examples:
        citrees-exp cluster submit paper/scripts/pipeline/stage1.py --help
    """
    _check_cluster_config()

    heading("Submitting job")

    cmd = ["ray", "submit", str(CLUSTER_YAML), script]
    if args:
        cmd.extend(args)

    info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        success("Job submitted")
    else:
        error("Job submission failed")
        raise typer.Exit(1)


@app.command()
def sync(
    direction: Annotated[
        str,
        typer.Argument(
            help="Sync direction: up (local->cluster) or down (cluster->local)",
        ),
    ] = "up",
) -> None:
    """Sync files with the cluster.

    \b
    Examples:
        citrees-exp cluster sync up      # Push local changes to cluster
        citrees-exp cluster sync down    # Pull changes from cluster
    """
    _check_cluster_config()

    if direction == "up":
        heading("Syncing local -> cluster")
        src = str(REPO_ROOT)
        result = subprocess.run(
            [
                "ray",
                "rsync-up",
                str(CLUSTER_YAML),
                src,
                "/home/ubuntu/citrees",
            ]
        )
    elif direction == "down":
        heading("Syncing cluster -> local")
        dest = REPO_ROOT / "paper" / "results"
        dest.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "ray",
                "rsync-down",
                str(CLUSTER_YAML),
                "/home/ubuntu/citrees/paper/results",
                str(dest),
            ]
        )
    else:
        error(f"Invalid direction: {direction}")
        error("Use 'up' or 'down'")
        raise typer.Exit(1)

    if result.returncode == 0:
        success("Sync complete")
    else:
        error("Sync failed")
        raise typer.Exit(1)
