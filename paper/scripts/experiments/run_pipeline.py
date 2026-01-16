"""Run the full Ray experiments pipeline sequentially: Stage 1 → Stage 2.

This is the "one run" entrypoint:
- Stage 1 computes rankings and writes them to S3.
- Stage 2 runs only after Stage 1 completes and consumes the Stage 1 artifacts.

Use `--dry-run` to confirm the grid size without submitting tasks.
"""

from __future__ import annotations

import argparse
from typing import Any

from loguru import logger

from paper.scripts.experiments._common import get_dataset_shape, get_datasets, get_git_sha
from paper.scripts.experiments._driver import (
    build_common_parser,
    init_ray,
    iter_grid,
    log_dry_run,
    log_failures,
    resolve_grid,
    run_futures,
)
from paper.scripts.experiments import ray_eval, ray_feature_selection
from paper.scripts.infra.config import load_config
from paper.scripts.utils.constants import CLF_METHODS, REG_METHODS
from paper.scripts.utils.experiment_configs import config_label, expand_method_configs


def _parse_args() -> argparse.Namespace:
    parser = build_common_parser("Ray pipeline: Stage 1 → Stage 2")
    parser.add_argument("--stage", choices=["all", "stage1", "stage2"], default="all", help="Which stages to run")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    init_ray(args.ray_address)

    task_type = args.task_type or cfg.experiment.type
    methods = CLF_METHODS if task_type == "classification" else REG_METHODS
    method_configs = expand_method_configs(methods)
    datasets = get_datasets(task_type, source=args.source)
    n_seeds = cfg.experiment.n_seeds
    git_sha = get_git_sha()

    method_configs, datasets, seeds = resolve_grid(
        method_configs=method_configs,
        datasets=datasets,
        n_seeds=n_seeds,
        datasets_csv=args.datasets,
        methods_csv=args.methods,
        seeds_csv=args.seeds,
    )

    total = len(method_configs) * len(datasets) * len(seeds)
    logger.info(
        "Pipeline grid: {} configs ({} methods × {} datasets × {} seeds)",
        total,
        len(method_configs),
        len(datasets),
        len(seeds),
    )

    dataset_shapes = {d: get_dataset_shape(d, task_type) for d in datasets}
    evaluation_cpus = ray_eval.evaluation_num_cpus(task_type)
    logger.info("Stage 2 evaluation_cpus={}", evaluation_cpus)

    method_id_to_cfg = {config_label(c): c for c in method_configs}

    def _describe(m: dict[str, Any], d: str, s: int) -> str:
        method_base = m["method"]
        n_samples, n_features = dataset_shapes[d]
        selection_cpus = ray_feature_selection.selection_num_cpus(
            method_base,
            n_samples=n_samples,
            n_features=n_features,
        )
        method_id = config_label(m)
        return (
            f"method={method_id} (base={method_base}), dataset={d}, seed={s}, "
            f"selection_cpus={selection_cpus}, evaluation_cpus={evaluation_cpus}"
        )

    if args.dry_run:
        log_dry_run(iter_grid(method_configs, datasets, seeds), stage="pipeline", describe=_describe)
        return

    # -------------------------------------------------------------------------
    # Stage 1
    # -------------------------------------------------------------------------
    stage1_done: set[tuple[str, str, int]] = set()
    if args.stage in {"all", "stage1"}:
        futures = []
        for method_cfg, dataset, seed in iter_grid(method_configs, datasets, seeds):
            method_base = method_cfg["method"]
            n_samples, n_features = dataset_shapes[dataset]
            selection_cpus = ray_feature_selection.selection_num_cpus(
                method_base, n_samples=n_samples, n_features=n_features
            )
            futures.append(
                ray_feature_selection.process_config.options(num_cpus=selection_cpus).remote(
                    method_cfg,
                    dataset,
                    seed,
                    task_type,
                    selection_cpus,
                    git_sha,
                )
            )

        _counts, failures, _elapsed, results = run_futures(futures, stage="stage1", success_statuses={"done"})
        log_failures(failures, stage="stage1")
        for r in results:
            if r.get("status") == "done":
                stage1_done.add((str(r.get("method")), str(r.get("dataset")), int(r.get("seed"))))

        logger.info("Stage 1 complete: done={}, failed={}", len(stage1_done), len(failures))

    # -------------------------------------------------------------------------
    # Stage 2
    # -------------------------------------------------------------------------
    if args.stage in {"all", "stage2"}:
        if args.stage == "stage2":
            stage2_items = list(iter_grid(method_configs, datasets, seeds))
            logger.info("Stage 2 only: running full grid and skipping configs without rankings")
        else:
            stage2_items: list[tuple[dict[str, Any], str, int]] = []
            for method_id, dataset, seed in sorted(stage1_done):
                method_cfg = method_id_to_cfg.get(method_id)
                if method_cfg is None:
                    raise RuntimeError(f"Internal error: missing method_cfg for method_id={method_id!r}")
                stage2_items.append((method_cfg, dataset, seed))

        futures = [
            ray_eval.process_config.options(num_cpus=evaluation_cpus).remote(
                method_cfg,
                dataset,
                seed,
                task_type,
                evaluation_cpus,
                git_sha,
            )
            for method_cfg, dataset, seed in stage2_items
        ]

        _counts, failures, _elapsed, _results = run_futures(
            futures,
            stage="stage2",
            success_statuses={"done"},
            skip_statuses={"no_rankings"},
        )
        log_failures(failures, stage="stage2")


if __name__ == "__main__":
    main()
