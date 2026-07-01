"""Self-populating FastAPI server for experiment orchestration.

On startup, builds the configured experiment set for both classification and
regression, checks S3 for completed work across all 4 queues (rankings/clf,
rankings/reg, metrics/clf, metrics/reg), and serves remaining work.

Workers call POST /next to get assigned work. The server picks what to
hand out — 90% rankings, 10% metrics, skip empty queues, fallback to
the other.

Queues are materialized and shuffled at startup so workers receive configs
in random order (preventing all workers from hitting the same method/dataset
simultaneously). Configs are serialized on demand when popped via /next.

Metrics queues are periodically refreshed (default: every 5 minutes) so that
rankings completed after startup produce stage 2 work items. The refresh
interval is configurable via the CITREES_REFRESH_INTERVAL env var (seconds).
Set to 0 to disable periodic refresh. Manual refresh is available via
POST /refresh.

Start with:
    uvicorn paper.scripts.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import random
import threading
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.responses import Response

if TYPE_CHECKING:
    from paper.scripts.adapters.store import S3Store
    from paper.scripts.pipeline.grid import ExperimentGrid
    from paper.scripts.pipeline.types import ExperimentConfig


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_config(cfg: Any) -> dict[str, Any]:
    """Serialize an ExperimentConfig to a JSON-safe dict."""
    return {
        "method": {
            "name": cfg.method.name,
            "params": dict(cfg.method.params),
        },
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "task": cfg.task,
    }


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_REFRESH_INTERVAL_SECONDS = int(os.environ.get("CITREES_REFRESH_INTERVAL", "300"))

_TASKS = ("classification", "regression")

# Queue key: "{stage}/{task}" -> e.g. "rankings/classification"
QueueKey = str


@dataclass
class QueueState:
    """Queue backed by a shuffled iterator.

    Configs are materialized and shuffled at startup, then wrapped as an
    iterator. Each call to pop_next() yields the next config in random order.
    """

    _iter: Iterator[Any]
    initial: int = 0
    served: int = 0
    _exhausted: bool = field(default=False, repr=False)

    def pop_next(self) -> dict[str, Any] | None:
        """Return the next pending config as a JSON-safe dict, or None."""
        if self._exhausted:
            return None
        try:
            cfg = next(self._iter)
            self.served += 1
            return _serialize_config(cfg)
        except StopIteration:
            self._exhausted = True
            return None

    @property
    def pending(self) -> int:
        """Approximate number of remaining items."""
        return max(0, self.initial - self.served)


_queues: dict[QueueKey, QueueState] = {}
_grids: dict[str, ExperimentGrid] = {}
_store: S3Store | None = None
_refresh_lock = threading.Lock()


def _key(stage: str, task: str) -> QueueKey:
    return f"{stage}/{task}"


def _needs_metrics_work(
    store: S3Store,
    cfg: ExperimentConfig,
    completed_metrics: set[tuple[str, str, int]],
) -> bool:
    """Return True when Stage 2 is missing required k coverage for a config."""
    if cfg.key not in completed_metrics:
        return True

    from paper.scripts.pipeline.stage2 import (
        get_requested_evaluation_k_values,
        infer_n_features_from_rankings,
        metrics_cover_requested_k_values,
    )

    rankings_df = store.load("rankings", cfg)
    metrics_df = store.load("metrics", cfg)
    required_k_values = get_requested_evaluation_k_values(
        infer_n_features_from_rankings(rankings_df)
    )
    return not metrics_cover_requested_k_values(metrics_df, required_k_values)


def _build_queues() -> None:
    """Build all 4 queues by checking S3 for completed work.

    Materializes pending configs, shuffles them, then wraps as iterators.
    Also saves grids and store reference for later metric queue refreshes.
    """
    global _store  # noqa: PLW0603
    from loguru import logger

    from paper.scripts.adapters.store import S3Store as _S3Store
    from paper.scripts.config.settings import load_config
    from paper.scripts.pipeline.grid import ExperimentGrid

    logger.info("Loading config...")
    config = load_config()
    n_seeds = config.experiment.n_seeds

    logger.info("Connecting to S3...")
    store = _S3Store.from_config()
    _store = store
    _queues.clear()
    _grids.clear()

    for task in _TASKS:
        logger.info("Building queues for {}...", task)
        grid = ExperimentGrid.from_cli(task=task, n_seeds=n_seeds)
        _grids[task] = grid
        grid_size = len(grid)

        logger.info("{}: {} total configs in grid", task, grid_size)

        logger.info("{}: listing completed rankings...", task)
        completed_rankings = store.list_completed("rankings", task)
        logger.info("{}: {} completed rankings, listing metrics...", task, len(completed_rankings))
        completed_metrics = store.list_completed("metrics", task)
        logger.info("{}: {} completed metrics", task, len(completed_metrics))

        # Rankings queue: grid items not yet in S3 (materialized + shuffled)
        rankings_list = list(grid.iter_pending(completed_rankings))

        rankings_initial = len(rankings_list)
        random.shuffle(rankings_list)
        _queues[_key("rankings", task)] = QueueState(
            _iter=iter(rankings_list),
            initial=rankings_initial,
        )

        # Metrics queue: items that have rankings but are missing required k coverage.
        metrics_list = [
            cfg
            for cfg in grid
            if cfg.key in completed_rankings and _needs_metrics_work(store, cfg, completed_metrics)
        ]
        metrics_initial = len(metrics_list)
        random.shuffle(metrics_list)
        _queues[_key("metrics", task)] = QueueState(
            _iter=iter(metrics_list),
            initial=metrics_initial,
        )

        logger.info(
            "{}: ~{} rankings pending, ~{} metrics pending",
            task,
            rankings_initial,
            metrics_initial,
        )

    logger.info("All queues built. Server ready.")


# ---------------------------------------------------------------------------
# Metrics queue refresh
# ---------------------------------------------------------------------------


def _refresh_metrics() -> dict[str, int]:
    """Rescan S3 and rebuild metrics queues with newly completed rankings.

    Returns a summary dict mapping queue keys to their new pending counts.
    Serialized via ``_refresh_lock`` so concurrent periodic + manual calls
    cannot interleave per-task queue updates.
    """
    from loguru import logger

    if _store is None:
        raise RuntimeError("_store not initialized — was _build_queues() called?")

    with _refresh_lock:
        logger.info("Refreshing metrics queues...")
        summary: dict[str, int] = {}

        for task in _TASKS:
            grid = _grids[task]
            completed_rankings = _store.list_completed("rankings", task)
            completed_metrics = _store.list_completed("metrics", task)

            metrics_list = [
                cfg
                for cfg in grid
                if cfg.key in completed_rankings
                and _needs_metrics_work(_store, cfg, completed_metrics)
            ]
            random.shuffle(metrics_list)

            qk = _key("metrics", task)
            old_q = _queues.get(qk)
            old_pending = old_q.pending if old_q else 0

            _queues[qk] = QueueState(
                _iter=iter(metrics_list),
                initial=len(metrics_list),
            )
            summary[qk] = len(metrics_list)

            logger.info(
                "metrics/{}: {} pending (was ~{})",
                task,
                len(metrics_list),
                old_pending,
            )

        logger.info("Metrics refresh complete.")
        return summary


async def _periodic_refresh() -> None:
    """Background task that refreshes metrics queues on an interval."""
    while True:
        await asyncio.sleep(_REFRESH_INTERVAL_SECONDS)
        await asyncio.to_thread(_refresh_metrics)


# ---------------------------------------------------------------------------
# Queue selection logic
# ---------------------------------------------------------------------------


def _pick_next() -> tuple[str, str, dict[str, Any]] | None:
    """Pick the next config to hand out.

    Strategy: 90% rankings, 10% metrics. Within each stage, round-robin
    between clf/reg. Skip empty queues, fallback to the other stage.
    """
    roll = random.random()
    if roll < 0.9:
        primary, fallback = "rankings", "metrics"
    else:
        primary, fallback = "metrics", "rankings"

    for stage in (primary, fallback):
        # Shuffle task order for fairness
        tasks = list(_TASKS)
        random.shuffle(tasks)
        for task in tasks:
            q = _queues.get(_key(stage, task))
            if q and q.pending > 0:
                config = q.pop_next()
                if config is not None:
                    return stage, task, config

    return None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    _build_queues()
    refresh_task: asyncio.Task[None] | None = None
    if _REFRESH_INTERVAL_SECONDS > 0:
        refresh_task = asyncio.create_task(_periodic_refresh())
    yield
    if refresh_task is not None:
        refresh_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await refresh_task


# Single uvicorn worker is the correct choice here: the /next handler is a
# sub-microsecond deque pop with no I/O, so one async process handles 10k+
# req/s. Queue state is module-level (in-process), which is fine since there
# is no need for multi-process parallelism at this throughput.
app = FastAPI(title="citrees experiment queue", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/next")
async def next_config() -> Response:
    result = _pick_next()
    if result is None:
        return Response(status_code=204)
    stage, _task, config = result
    # config already contains "task" from _serialize_config
    return JSONResponse(
        status_code=200,
        content={"stage": stage, **config},
    )


@app.post("/refresh")
async def refresh_metrics() -> dict[str, Any]:
    """Manually trigger a metrics queue refresh from current S3 state."""
    summary = await asyncio.to_thread(_refresh_metrics)
    return {"refreshed": summary}


@app.get("/status")
async def get_status() -> dict[str, Any]:
    return {
        "queues": {
            k: {"pending": q.pending, "initial": q.initial, "served": q.served}
            for k, q in _queues.items()
        },
    }
