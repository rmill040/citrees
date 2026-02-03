"""Pull-based worker that pops configs from the API server and executes them.

The worker takes no stage or task arguments — it reads the assigned stage
and task from the /next response. The server decides what to hand out.

Run as:
    python -m paper.scripts.api.worker --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Any

import httpx
from loguru import logger

from paper.scripts.adapters.store import S3Store
from paper.scripts.pipeline.types import ExperimentConfig, MethodConfig

_shutdown = False


def _handle_signal(signum: int, frame: Any) -> None:
    global _shutdown
    name = signal.Signals(signum).name
    if _shutdown:
        logger.warning("Received {} again, forcing exit", name)
        sys.exit(1)
    logger.warning("Received {}, finishing current task then exiting...", name)
    _shutdown = True


def _deserialize_config(d: dict[str, Any]) -> ExperimentConfig:
    """Reconstruct ExperimentConfig from JSON dict."""
    method_dict = d["method"]
    params = tuple(sorted(method_dict.get("params", {}).items()))
    method = MethodConfig(name=method_dict["name"], params=params)
    return ExperimentConfig(
        method=method,
        dataset=d["dataset"],
        seed=d["seed"],
        task=d["task"],
    )


def run_worker(
    api_url: str,
    poll_interval: float = 5.0,
    max_idle_polls: int | None = None,
) -> None:
    """Main worker loop: pop config -> execute -> repeat.

    Parameters
    ----------
    api_url : str
        URL of the API server.
    poll_interval : float
        Seconds between polls when queue is empty.
    max_idle_polls : int or None
        Exit after this many consecutive empty-queue polls.
        None means poll forever.
    """
    from paper.scripts.pipeline.stage1 import _run_selection
    from paper.scripts.pipeline.stage2 import _run_evaluation

    store = S3Store.from_config()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("Worker started: api_url={}, max_idle_polls={}", api_url, max_idle_polls)

    client = httpx.Client(base_url=api_url, timeout=30.0)
    idle_count = 0

    while not _shutdown:
        # Pop next config
        try:
            resp = client.post("/next")
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
            logger.warning("Cannot reach API at {}, retrying...", api_url)
            time.sleep(poll_interval)
            continue

        if resp.status_code == 204:
            idle_count += 1
            logger.debug("Queue empty ({}/{}), sleeping {}s", idle_count, max_idle_polls or "∞", poll_interval)
            if max_idle_polls is not None and idle_count >= max_idle_polls:
                logger.info("Reached max idle polls ({}), exiting", max_idle_polls)
                break
            time.sleep(poll_interval)
            continue

        if resp.status_code != 200:
            logger.warning("Unexpected status {} from /next", resp.status_code)
            time.sleep(poll_interval)
            continue

        # Got work — reset idle counter
        idle_count = 0

        data = resp.json()
        stage = data["stage"]
        cfg = _deserialize_config(data)
        logger.info("Running: stage={} {}", stage, cfg)

        run_fn = _run_selection if stage == "rankings" else _run_evaluation
        result = run_fn(cfg, store)

        if result.is_failure:
            logger.error("Failed: {} — {}: {}", cfg, result.error_type, result.error)
        else:
            logger.info("Done: {} (status={})", cfg, result.status)

    client.close()
    logger.info("Worker shutting down")


def main() -> None:
    parser = argparse.ArgumentParser(description="citrees experiment worker")
    parser.add_argument("--api-url", required=True, help="API server URL")
    parser.add_argument(
        "--poll-interval", type=float, default=5.0, help="Seconds between polls when queue is empty"
    )
    parser.add_argument(
        "--max-idle-polls", type=int, default=None, help="Exit after N consecutive empty polls (default: run forever)"
    )
    args = parser.parse_args()
    run_worker(args.api_url, poll_interval=args.poll_interval, max_idle_polls=args.max_idle_polls)


if __name__ == "__main__":
    main()
