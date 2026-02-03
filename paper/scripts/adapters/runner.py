"""Runner protocol and implementations for experiment execution.

Provides a clean abstraction for running experiments locally.
For distributed execution, use the API server + pull workers.
"""

from __future__ import annotations

import signal
from collections.abc import Callable, Iterable
from typing import Any, Protocol

from loguru import logger

from paper.scripts.adapters.store import Store
from paper.scripts.pipeline.types import ExperimentConfig, Result, StageType

# Shutdown flag for graceful termination.
# Note: This is intentionally global because signal handlers are process-global.
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
    """

    def run(
        self,
        stage: StageType,
        configs: Iterable[ExperimentConfig],
        store: Store,
        on_complete: Callable[[ExperimentConfig, Result], None] | None = None,
        *,
        collect_results: bool = True,
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
        collect_results : bool, default True
            Whether to accumulate and return results.

        Returns
        -------
        list[Result]
            Results for all configurations.
        """
        ...


class LocalRunner:
    """Local sequential runner for testing and debugging.

    Executes experiments one at a time without any distributed backend.
    Useful for debugging and small-scale runs.
    """

    def run(
        self,
        stage: StageType,
        configs: Iterable[ExperimentConfig],
        store: Store,
        on_complete: Callable[[ExperimentConfig, Result], None] | None = None,
        *,
        collect_results: bool = True,
    ) -> list[Result]:
        """Run experiments sequentially."""
        from paper.scripts.pipeline.stage1 import _run_selection
        from paper.scripts.pipeline.stage2 import _run_evaluation

        run_fn = _run_selection if stage == "rankings" else _run_evaluation

        results: list[Result] = []
        for cfg in configs:
            result = run_fn(cfg, store)
            if collect_results:
                results.append(result)
            if on_complete:
                on_complete(cfg, result)

        return results
