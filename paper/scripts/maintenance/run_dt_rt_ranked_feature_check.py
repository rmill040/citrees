"""Run DT/RT ranked-feature downstream checks locally.

This is a local maintenance helper for rerunning only DT/RT.

This script runs only the single-tree baselines:

- dt: sklearn.tree.DecisionTreeClassifier/Regressor
- rt: sklearn.tree.ExtraTreeClassifier/Regressor, a single randomized tree

It writes per-config Stage 1/Stage 2 artifacts to a local scratch directory and
creates compact score summaries for inspection. It does not write a
paper-facing parquet unless `--canonical-output` is provided explicitly.

Usage:
  UV_CACHE_DIR=./scratch/.uv_cache uv run python \
    paper/scripts/maintenance/run_dt_rt_ranked_feature_check.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from paper.scripts.adapters.data import get_dataset_shape
from paper.scripts.adapters.runner import LocalRunner
from paper.scripts.pipeline.grid import ExperimentGrid
from paper.scripts.pipeline.types import ExperimentConfig, StageType, TaskType

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKDIR = ROOT / "scratch" / "dt_rt_ranked_feature_check"
DEFAULT_METHODS = "dt,rt"
DEFAULT_TASKS = "classification,regression"


@dataclass
class LocalParquetStore:
    """Minimal local artifact store for isolated reruns."""

    base_dir: Path

    def path(self, stage: StageType, config: ExperimentConfig) -> Path:
        return (
            self.base_dir
            / stage
            / config.task
            / config.dataset
            / f"{config.method.label}_seed{config.seed}.parquet"
        )

    def exists(self, stage: StageType, config: ExperimentConfig) -> bool:
        return self.path(stage, config).exists()

    def save(self, stage: StageType, config: ExperimentConfig, df: pd.DataFrame) -> str:
        path = self.path(stage, config)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return str(path)

    def load(self, stage: StageType, config: ExperimentConfig) -> pd.DataFrame:
        path = self.path(stage, config)
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_parquet(path)

    def list_completed(self, stage: StageType, task: TaskType) -> set[tuple[str, str, int]]:
        completed: set[tuple[str, str, int]] = set()
        root = self.base_dir / stage / task
        if not root.exists():
            return completed

        for path in root.glob("*/*_seed*.parquet"):
            method_label, seed_text = path.stem.rsplit("_seed", 1)
            completed.add((method_label, path.parent.name, int(seed_text)))
        return completed


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_tasks(value: str) -> list[TaskType]:
    tasks = _split_csv(value)
    invalid = sorted(set(tasks) - {"classification", "regression"})
    if invalid:
        raise ValueError(f"Invalid task(s): {invalid}")
    return [task for task in tasks]  # type: ignore[return-value]


def _build_configs(args: argparse.Namespace) -> list[ExperimentConfig]:
    excluded_datasets = set(_split_csv(args.exclude_datasets))

    def within_dataset_cell_cap(cfg: ExperimentConfig) -> bool:
        if args.max_dataset_cells is None:
            return True
        n_samples, n_features = get_dataset_shape(cfg.dataset, cfg.task)
        return n_samples * n_features <= args.max_dataset_cells

    configs: list[ExperimentConfig] = []
    for task in _parse_tasks(args.tasks):
        grid = ExperimentGrid.from_cli(
            task=task,
            methods=args.methods,
            datasets=args.datasets or None,
            seeds=args.seeds or None,
            source=args.source,
            n_seeds=args.n_seeds,
            max_configs_per_method=args.max_configs_per_method,
        )
        configs.extend(grid.as_list())

    if excluded_datasets:
        configs = [cfg for cfg in configs if cfg.dataset not in excluded_datasets]

    if args.max_dataset_cells is not None:
        configs = [cfg for cfg in configs if within_dataset_cell_cap(cfg)]

    return configs


def _write_status(path: Path, stage: StageType, results: list) -> None:
    rows = []
    for result in results:
        row = result.to_dict()
        row["stage"] = stage
        rows.append(row)
    frame = pd.DataFrame(rows)
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    frame.to_csv(path, index=False)


def _run_stage(
    *,
    runner: LocalRunner,
    store: LocalParquetStore,
    stage: StageType,
    configs: list[ExperimentConfig],
    status_path: Path,
) -> None:
    tic = time.perf_counter()
    print(f"{stage}: running {len(configs)} configs", flush=True)

    completed = 0

    def on_complete(cfg: ExperimentConfig, result) -> None:
        nonlocal completed
        completed += 1
        if completed == 1 or completed == len(configs) or completed % 10 == 0 or result.is_failure:
            print(
                f"  [{completed}/{len(configs)}] {stage} {result.status} {cfg}",
                flush=True,
            )

    results = runner.run(stage, configs, store, on_complete=on_complete)
    elapsed = time.perf_counter() - tic
    _write_status(status_path, stage, results)

    statuses = pd.Series([result.status for result in results]).value_counts().to_dict()
    print(f"{stage}: statuses={statuses} elapsed_seconds={elapsed:.1f}", flush=True)

    failures = [result for result in results if result.is_failure]
    if failures:
        first = failures[0]
        raise RuntimeError(f"{stage} failed for {first.config}: {first.error_type}: {first.error}")

    if stage == "metrics":
        missing = [result for result in results if result.is_no_rankings]
        if missing:
            raise RuntimeError(f"metrics missing rankings for {len(missing)} config(s)")


def _summarize_metrics(
    store: LocalParquetStore,
    configs: list[ExperimentConfig],
    output_dir: Path,
    canonical_output: Path | None,
) -> None:
    frames = [store.load("metrics", cfg) for cfg in configs if store.exists("metrics", cfg)]
    if not frames:
        print("No metrics artifacts available to summarize", flush=True)
        return

    metrics = pd.concat(frames, ignore_index=True)
    combined_path = output_dir / "dt_rt_metrics_combined.parquet"
    metrics.to_parquet(combined_path, index=False)
    if canonical_output is not None:
        canonical_output.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_parquet(canonical_output, index=False)

    score_frames = []
    clf = metrics[metrics["task"] == "classification"].copy()
    if not clf.empty:
        clf["score"] = clf["balanced_accuracy"]
        score_frames.append(clf)

    reg = metrics[metrics["task"] == "regression"].copy()
    if not reg.empty:
        reg["score"] = reg["r2"]
        score_frames.append(reg)

    if not score_frames:
        print("Metrics artifacts did not contain supported score columns", flush=True)
        return

    scores = pd.concat(score_frames, ignore_index=True)
    summary = (
        scores.groupby(["task", "method_base"], dropna=False)["score"]
        .agg(mean_score="mean", std_score="std", n_rows="size")
        .reset_index()
    )
    summary["rank_position"] = summary.groupby("task")["mean_score"].rank(
        method="min", ascending=False
    )
    summary = summary.sort_values(["task", "rank_position", "method_base"]).reset_index(drop=True)

    by_k = (
        scores.groupby(["task", "method_base", "downstream_model", "k"], dropna=False)["score"]
        .agg(mean_score="mean", std_score="std", n_rows="size")
        .reset_index()
        .sort_values(["task", "method_base", "downstream_model", "k"])
    )

    summary_path = output_dir / "dt_rt_score_summary.csv"
    by_k_path = output_dir / "dt_rt_score_by_k.csv"
    summary.to_csv(summary_path, index=False)
    by_k.to_csv(by_k_path, index=False)

    print(f"Wrote {combined_path}", flush=True)
    if canonical_output is not None:
        print(f"Wrote {canonical_output}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {by_k_path}", flush=True)
    print(summary.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument(
        "--canonical-output",
        type=Path,
        default=None,
        help="Optional parquet to update after metrics are summarized",
    )
    parser.add_argument(
        "--no-canonical-output",
        action="store_true",
        help="Only write scratch summaries; ignore --canonical-output",
    )
    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS)
    parser.add_argument("--methods", type=str, default=DEFAULT_METHODS)
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--exclude-datasets", type=str, default="")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument(
        "--source",
        choices=["all", "real", "synthetic"],
        default="real",
        help="Dataset source filter",
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--max-configs-per-method", type=int, default=None)
    parser.add_argument(
        "--max-dataset-cells",
        type=int,
        default=None,
        help="Optional n_samples * n_features cap for selected configs",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "rankings", "metrics"],
        default="all",
        help="Run both stages, rankings only, or metrics only",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    status_path = workdir / "dt_rt_run_status.csv"

    configs = _build_configs(args)
    if not configs:
        raise RuntimeError("No configs selected")

    print(f"Workdir: {workdir}", flush=True)
    print(f"Methods: {sorted({cfg.method.name for cfg in configs})}", flush=True)
    print(f"Tasks: {sorted({cfg.task for cfg in configs})}", flush=True)
    print(f"Datasets: {len({(cfg.task, cfg.dataset) for cfg in configs})}", flush=True)
    print(f"Seeds: {sorted({cfg.seed for cfg in configs})}", flush=True)
    print(f"Configs: {len(configs)}", flush=True)

    if args.dry_run:
        return

    store = LocalParquetStore(workdir)
    runner = LocalRunner()

    if args.stage in {"all", "rankings"}:
        _run_stage(
            runner=runner,
            store=store,
            stage="rankings",
            configs=configs,
            status_path=status_path,
        )
    if args.stage in {"all", "metrics"}:
        _run_stage(
            runner=runner,
            store=store,
            stage="metrics",
            configs=configs,
            status_path=status_path,
        )
    canonical_output = None
    if not args.no_canonical_output and args.canonical_output is not None:
        canonical_output = args.canonical_output.resolve()
    _summarize_metrics(store, configs, workdir, canonical_output)


if __name__ == "__main__":
    main()
