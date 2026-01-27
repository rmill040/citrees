"""Runner protocol and implementations for distributed execution.

Provides a clean abstraction for running experiments, with Ray as the
primary backend for distributed execution.
"""

from __future__ import annotations

import signal
import time
from collections.abc import Callable, Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from paper.scripts.adapters.store import Store
from paper.scripts.pipeline.types import ExperimentConfig, Result, StageType

# Shutdown flag for graceful termination.
# Note: This is intentionally global because signal handlers are process-global.
# Multiple RayRunner instances share this state, which is the correct behavior
# for coordinated shutdown on SIGINT/SIGTERM.
_shutdown_requested = False


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    if _shutdown_requested:
        logger.warning("Received {} again, forcing exit", sig_name)
        raise SystemExit(1)
    logger.warning("Received {}, initiating graceful shutdown...", sig_name)
    _shutdown_requested = True


def register_signal_handlers() -> None:
    """Register signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


class Runner(Protocol):
    """Protocol for experiment execution.

    Defines the interface for running experiments across configurations.
    Implementations can use Ray, local execution, or other backends.
    """

    def run(
        self,
        stage: StageType,
        configs: Iterable[ExperimentConfig],
        store: Store,
        on_complete: Callable[[ExperimentConfig, Result], None] | None = None,
    ) -> list[Result]:
        """Run experiments for the given configurations.

        Parameters
        ----------
        stage : StageType
            Either "rankings" or "metrics".
        configs : Iterable[ExperimentConfig]
            Experiment configurations to run.
        store : Store
            Storage backend for saving results.
        on_complete : callable, optional
            Callback invoked for each completed task.

        Returns
        -------
        list[Result]
            Results for all configurations.
        """
        ...


class RayRunner:
    """Ray-based distributed runner.

    Executes experiments in parallel using Ray with configurable
    resource allocation per task.

    Parameters
    ----------
    address : str, default "auto"
        Ray cluster address. Use "auto" for existing cluster,
        "local" for local mode.
    batch_size : int, default 32
        Number of tasks to wait for at once.
    progress_every_s : float, default 30.0
        Interval for progress logging.

    Examples
    --------
    >>> runner = RayRunner(address="local")
    >>> results = runner.run("rankings", configs, store)
    >>> len([r for r in results if r.is_success])
    42
    """

    def __init__(
        self,
        address: str = "auto",
        batch_size: int = 32,
        progress_every_s: float = 30.0,
    ):
        self.address = address
        self.batch_size = batch_size
        self.progress_every_s = progress_every_s
        self._initialized = False

    def init(self) -> None:
        """Initialize Ray connection."""
        if self._initialized:
            return

        import ray

        register_signal_handlers()

        # Exclude large files/directories to avoid exceeding Ray's package limit
        excludes = [
            ".git/",
            ".venv/",
            ".uv-cache/",
            ".ruff_cache/",
            ".pytest_cache/",
            ".mypy_cache/",
            "__pycache__/",
            "paper/data/",
            "paper/results/",
            "scratch/",
            "pyproject.toml",
            "uv.lock",
            "setup.py",
            "setup.cfg",
        ]

        if self.address == "local":
            # Local mode: workers share the current environment
            repo_root = Path(__file__).resolve().parents[3]
            env_vars = {"CITREES_REPO_ROOT": str(repo_root)}
            ray.init(
                resources={"selection": 4, "evaluation": 4},
                runtime_env={"excludes": excludes, "env_vars": env_vars},
                ignore_reinit_error=True,
            )
        else:
            ray.init(
                address=self.address,
                runtime_env={"excludes": excludes},
                ignore_reinit_error=True,
            )

        self._initialized = True

    def run(
        self,
        stage: StageType,
        configs: Iterable[ExperimentConfig],
        store: Store,
        on_complete: Callable[[ExperimentConfig, Result], None] | None = None,
    ) -> list[Result]:
        """Run experiments using Ray.

        Submits tasks to Ray and collects results with progress logging.
        Supports graceful shutdown on SIGINT/SIGTERM.
        """

        self.init()

        # Import task functions based on stage
        if stage == "rankings":
            from paper.scripts.pipeline.stage1 import run_selection_task

            task_fn = run_selection_task
        else:
            from paper.scripts.pipeline.stage2 import run_evaluation_task

            task_fn = run_evaluation_task

        # Submit all tasks
        config_list = list(configs)
        futures = [task_fn.remote(cfg, store) for cfg in config_list]

        # Collect results with progress logging
        return self._drain_futures(futures, stage, on_complete)

    def _drain_futures(
        self,
        futures: list[Any],
        stage: StageType,
        on_complete: Callable[[ExperimentConfig, Result], None] | None,
    ) -> list[Result]:
        """Drain Ray futures with progress logging."""
        global _shutdown_requested
        import ray

        pending = list(futures)
        results: list[Result] = []
        status_counts: dict[str, int] = {}

        start = time.perf_counter()
        last_log = start

        while pending:
            # Check for shutdown request
            if _shutdown_requested:
                logger.warning(
                    "Shutdown requested [{}]: cancelling {} pending tasks",
                    stage,
                    len(pending),
                )
                for future in pending:
                    with suppress(Exception):
                        ray.cancel(future, force=False)
                break

            ready, pending = ray.wait(
                pending,
                num_returns=min(self.batch_size, len(pending)),
                timeout=10,
            )

            if ready:
                batch: list[Result] = ray.get(ready)
                for result in batch:
                    results.append(result)
                    status_counts[result.status] = status_counts.get(result.status, 0) + 1

                    if on_complete:
                        on_complete(result.config, result)

            now = time.perf_counter()
            if now - last_log >= self.progress_every_s:
                elapsed = now - start
                done = status_counts.get("done", 0)
                failed = status_counts.get("failed", 0)
                rate = done / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "Progress [{}]: done={}, failed={}, pending={}, rate={:.3f} configs/s",
                    stage,
                    done,
                    failed,
                    len(pending),
                    rate,
                )
                last_log = now

        elapsed_total = time.perf_counter() - start
        logger.info(
            "Completed [{}]: counts={}, elapsed_seconds={:.1f}",
            stage,
            status_counts,
            elapsed_total,
        )

        return results


class LocalRunner:
    """Local sequential runner for testing and debugging.

    Executes experiments one at a time without Ray.
    Useful for debugging and small-scale runs.
    """

    def run(
        self,
        stage: StageType,
        configs: Iterable[ExperimentConfig],
        store: Store,
        on_complete: Callable[[ExperimentConfig, Result], None] | None = None,
    ) -> list[Result]:
        """Run experiments sequentially."""
        from paper.scripts.pipeline.stage1 import _run_selection
        from paper.scripts.pipeline.stage2 import _run_evaluation

        run_fn = _run_selection if stage == "rankings" else _run_evaluation

        results: list[Result] = []
        for cfg in configs:
            result = run_fn(cfg, store)
            results.append(result)
            if on_complete:
                on_complete(cfg, result)

        return results
