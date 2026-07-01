"""API server and worker operations.

Commands for managing the experiment API server and workers.
"""

from __future__ import annotations

from typing import Annotated

import typer

from paper.scripts.cli._console import console, create_table, error, heading, info

app = typer.Typer(
    name="cluster",
    help="API server and worker operations",
    no_args_is_help=True,
)


@app.command(name="api-start")
def api_start(
    host: Annotated[
        str,
        typer.Option(
            "--host",
            help="Bind host",
        ),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Bind port",
        ),
    ] = 8000,
) -> None:
    """Start the API queue server.

    The server auto-discovers the configured experiment set and checks S3 for
    completed work on startup. No configuration needed.

    Examples:
        citrees-exp cluster api-start
        citrees-exp cluster api-start --port 9000
    """
    import uvicorn

    heading("Starting API Server")
    info(f"Host: {host}:{port}")

    uvicorn.run(
        "paper.scripts.api.server:app",
        host=host,
        port=port,
        log_level="info",
    )


@app.command(name="api-status")
def api_status(
    api_url: Annotated[
        str,
        typer.Option(
            "--api-url",
            help="API server URL",
            envvar="CITREES_API_URL",
        ),
    ] = "http://localhost:8000",
) -> None:
    """Show API queue status.

    Examples:
        citrees-exp cluster api-status
        citrees-exp cluster api-status --api-url http://api-host:8000
    """
    import httpx

    heading("API Queue Status")
    info(f"API: {api_url}")

    try:
        resp = httpx.get(f"{api_url}/status", timeout=10.0)
        resp.raise_for_status()
    except httpx.ConnectError:
        error(f"Cannot reach API at {api_url}")
        raise typer.Exit(1) from None

    data = resp.json()
    queues = data.get("queues", {})

    if not queues:
        info("No queues")
        return

    table = create_table(
        title="Queue Status",
        columns=[
            ("Queue", ""),
            ("Pending", "number"),
            ("Initial", "number"),
        ],
    )
    for name, counts in queues.items():
        table.add_row(
            name,
            str(counts.get("pending", 0)),
            str(counts.get("initial", 0)),
        )
    console.print(table)


@app.command(name="worker-start")
def worker_start(
    api_url: Annotated[
        str,
        typer.Option(
            "--api-url",
            help="API server URL",
            envvar="CITREES_API_URL",
        ),
    ] = "http://localhost:8000",
    poll_interval: Annotated[
        float,
        typer.Option(
            "--poll-interval",
            help="Seconds between polls when queue is empty",
        ),
    ] = 5.0,
) -> None:
    """Start a worker process.

    The worker gets its stage and task assignment from the server via
    POST /next. No --stage or --task flags needed.

    Examples:
        citrees-exp cluster worker-start
        citrees-exp cluster worker-start --api-url http://api:8000
    """
    from paper.scripts.api.worker import run_worker

    heading("Starting Worker")
    info(f"API: {api_url}")

    run_worker(api_url, poll_interval=poll_interval)
