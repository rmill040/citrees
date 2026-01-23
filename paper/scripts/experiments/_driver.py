"""Shared driver utilities for Ray experiment scripts.

Centralizes:
- common CLI flags (ray address, grid filters, dry run)
- grid filtering/validation (datasets/methods/seeds)
- Ray initialization (auto cluster vs local mode)
- progress/throughput logging and failure reporting

The goal is to keep Stage 1/Stage 2 scripts focused on computation and artifact IO, not orchestration glue.
"""

from __future__ import annotations

import argparse
import signal
import time
from collections.abc import Callable, Iterable
from contextlib import suppress
from typing import Any

import ray
from loguru import logger

from paper.scripts.experiments._common import parse_csv_ints, parse_csv_list
from paper.scripts.utils.experiment_configs import config_label

# Shutdown flag for graceful termination
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


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--ray-address",
        default="auto",
        help="Ray address (default: auto). Use 'local' for local mode.",
    )
    parser.add_argument("--task-type", choices=["classification", "regression"], default=None)
    parser.add_argument(
        "--source",
        choices=["all", "real", "synthetic"],
        default="all",
        help="Dataset source filter",
    )
    parser.add_argument(
        "--datasets", default=None, help="Comma-separated dataset names (default: all)"
    )
    parser.add_argument(
        "--methods", default=None, help="Comma-separated base method names (default: all)"
    )
    parser.add_argument(
        "--seeds", default=None, help="Comma-separated seed indices (default: 0..n_seeds-1)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned configs and exit")
    parser.add_argument(
        "--dry-run-limit",
        type=int,
        default=50,
        help="Max configs to print in dry-run (default: 50)",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only run configs missing from S3 outputs for the stage",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip configs where output already exists (per-task S3 check for extra safety)",
    )
    return parser


def init_ray(ray_address: str) -> None:
    from pathlib import Path

    register_signal_handlers()
    # Exclude large files/directories to avoid exceeding Ray's 512MB package limit.
    # Also exclude pyproject.toml/uv.lock to prevent Ray from creating a fresh venv
    # (which would be missing ray itself, causing "No module named 'ray'" errors).
    excludes = [
        ".git/",
        ".venv/",
        ".uv-cache/",
        ".ruff_cache/",
        ".pytest_cache/",
        ".mypy_cache/",
        "__pycache__/",
        "paper/data/",  # Large parquet datasets
        "paper/results/",
        "scratch/",
        "pyproject.toml",  # Prevent Ray from creating new venv
        "uv.lock",
        "setup.py",
        "setup.cfg",
    ]
    if ray_address == "local":
        # Local mode: workers run on the same machine and share the current environment.
        # Set CITREES_REPO_ROOT so workers can find data files on the local filesystem
        # (since paper/data is excluded from the package to keep size small).
        repo_root = Path(__file__).resolve().parents[3]
        env_vars = {"CITREES_REPO_ROOT": str(repo_root)}
        ray.init(
            resources={"selection": 4, "evaluation": 4},
            runtime_env={"excludes": excludes, "env_vars": env_vars},
            ignore_reinit_error=True,
        )
    else:
        ray.init(
            address=ray_address,
            runtime_env={"excludes": excludes},
            ignore_reinit_error=True,
        )


def resolve_grid(
    *,
    method_configs: list[dict[str, Any]],
    datasets: list[str],
    n_seeds: int,
    datasets_csv: str | None,
    methods_csv: str | None,
    seeds_csv: str | None,
) -> tuple[list[dict[str, Any]], list[str], list[int]]:
    """Apply CLI filters to the experiment grid with validation."""
    datasets_filter = parse_csv_list(datasets_csv) or None
    methods_filter = parse_csv_list(methods_csv) or None
    seeds_filter = parse_csv_ints(seeds_csv) or None

    if datasets_filter:
        missing = sorted(set(datasets_filter) - set(datasets))
        if missing:
            raise ValueError(f"Unknown datasets: {missing}. Available examples: {datasets[:10]}")
        datasets = [d for d in datasets if d in set(datasets_filter)]

    if methods_filter:
        available_methods = {c.get("method") for c in method_configs}
        missing = sorted(set(methods_filter) - available_methods)
        if missing:
            raise ValueError(f"Unknown methods: {missing}. Available: {sorted(available_methods)}")
        method_configs = [c for c in method_configs if c.get("method") in set(methods_filter)]

    if seeds_filter is None:
        seeds = list(range(n_seeds))
    else:
        bad = [s for s in seeds_filter if s < 0 or s >= n_seeds]
        if bad:
            raise ValueError(f"Seed indices out of range (0..{n_seeds - 1}): {bad}")
        seeds = list(seeds_filter)

    return method_configs, datasets, seeds


def iter_grid(
    method_configs: list[dict[str, Any]],
    datasets: list[str],
    seeds: list[int],
) -> Iterable[tuple[dict[str, Any], str, int]]:
    for method_cfg in method_configs:
        for dataset in datasets:
            for seed in seeds:
                yield method_cfg, dataset, seed


def filter_missing(
    grid: Iterable[tuple[dict[str, Any], str, int]],
    completed: set[tuple[str, str, int]],
) -> list[tuple[dict[str, Any], str, int]]:
    """Filter a grid to configs not present in `completed`."""
    pending: list[tuple[dict[str, Any], str, int]] = []
    for method_cfg, dataset, seed in grid:
        method_id = config_label(method_cfg)
        if (method_id, dataset, seed) not in completed:
            pending.append((method_cfg, dataset, seed))
    return pending


def log_dry_run(
    grid: Iterable[tuple[dict[str, Any], str, int]],
    *,
    stage: str,
    limit: int = 50,
    describe: Callable[[dict[str, Any], str, int], str] | None = None,
) -> None:
    count = 0
    for method_cfg, dataset, seed in grid:
        if count >= limit:
            break
        if describe is None:
            msg = f"method={config_label(method_cfg)}, dataset={dataset}, seed={seed}"
        else:
            msg = describe(method_cfg, dataset, seed)
        logger.info("DRY RUN [{}]: {}", stage, msg)
        count += 1
    if count == 0:
        logger.info("DRY RUN [{}]: no configs to run", stage)


def run_futures(
    futures: list[Any],
    *,
    stage: str,
    success_statuses: set[str],
    skip_statuses: set[str] | None = None,
    batch_size: int = 32,
    progress_every_s: float = 30.0,
) -> tuple[dict[str, int], list[dict[str, Any]], float, list[dict[str, Any]]]:
    """Drain Ray futures with periodic progress logging.

    Returns:
    - status_counts: mapping from status string → count
    - failures: list of task result dicts not in success/skip statuses
    - elapsed_seconds: wall time to drain all futures
    - results: all task result dicts (success + skip + failure)
    """
    global _shutdown_requested
    skip_statuses = skip_statuses or set()

    pending = list(futures)
    status_counts: dict[str, int] = {}
    failures: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    start = time.perf_counter()
    last_log = start

    while pending:
        # Check for shutdown request
        if _shutdown_requested:
            logger.warning(
                "Shutdown requested [{}]: cancelling {} pending tasks", stage, len(pending)
            )
            for future in pending:
                with suppress(Exception):
                    ray.cancel(future, force=False)
            status_counts["cancelled"] = len(pending)
            break

        ready, pending = ray.wait(pending, num_returns=min(batch_size, len(pending)), timeout=10)
        if ready:
            batch = ray.get(ready)
            for result in batch:
                results.append(result)
                status = str(result.get("status", "unknown"))
                status_counts[status] = status_counts.get(status, 0) + 1
                if status not in success_statuses and status not in skip_statuses:
                    failures.append(result)

        now = time.perf_counter()
        if now - last_log >= progress_every_s:
            elapsed = now - start
            done = sum(status_counts.get(s, 0) for s in success_statuses)
            skipped = sum(status_counts.get(s, 0) for s in skip_statuses)
            failed = len(failures)
            rate = done / elapsed if elapsed > 0 else 0.0
            logger.info(
                "Progress [{}]: done={}, skipped={}, failed={}, pending={}, rate={:.3f} configs/s",
                stage,
                done,
                skipped,
                failed,
                len(pending),
                rate,
            )
            last_log = now

    elapsed_total = time.perf_counter() - start
    logger.info(
        "Completed [{}]: counts={}, elapsed_seconds={:.1f}", stage, status_counts, elapsed_total
    )
    return status_counts, failures, float(elapsed_total), results


def log_failures(failures: list[dict[str, Any]], *, stage: str, limit: int = 50) -> None:
    if not failures:
        return
    logger.error("Failures [{}]: count={}", stage, len(failures))
    for i, r in enumerate(failures[:limit]):
        logger.error(
            "Failed [{}] #{idx}: {method}/{dataset}/seed{seed} ({error_type}): {error}",
            stage,
            idx=i + 1,
            method=r.get("method"),
            dataset=r.get("dataset"),
            seed=r.get("seed"),
            error_type=r.get("error_type"),
            error=r.get("error"),
        )
    if len(failures) > limit:
        logger.error("Failures [{}]: ... ({} more)", stage, len(failures) - limit)
