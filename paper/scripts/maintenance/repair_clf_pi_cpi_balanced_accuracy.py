"""Refresh classification PI/CPI artifacts after switching to balanced accuracy.

Operational repair helper; not part of the current paper-facing rebuild path.

This script reruns only the classification PI/CPI benchmark surface, writes the
per-config Stage 1/Stage 2 outputs to a local scratch directory, and then
replaces the corresponding rows in the canonical aggregate parquets:

- paper/results/clf_rankings.parquet
- paper/results/clf_evaluation.parquet

It is resumable at the per-config parquet level because completed local
artifacts are reused on subsequent invocations.
"""

from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pandas as pd
from joblib import parallel_backend

from paper.scripts.adapters import get_s3_bucket
from paper.scripts.analysis.aggregate import _build_grid_config_keys, aggregate_stage
from paper.scripts.config import load_config
from paper.scripts.adapters.data import get_dataset_metadata, load_dataset
from paper.scripts.pipeline.methods import get_full_method_configs
from paper.scripts.pipeline.stage1 import run_selection
from paper.scripts.pipeline.stage2 import (
    get_requested_evaluation_k_values,
    infer_n_features_from_rankings,
    metrics_cover_requested_k_values,
    run_evaluation,
)
from paper.scripts.pipeline.types import ExperimentConfig, MethodConfig
from paper.scripts.utils.env import get_git_sha, get_library_versions, utc_now_iso

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "paper" / "results"
TARGET_TASK = "classification"
TARGET_METHODS = ("pi", "cpi")
RANKINGS_PATH = RESULTS_DIR / "clf_rankings.parquet"
EVALUATION_PATH = RESULTS_DIR / "clf_evaluation.parquet"


@dataclass
class LocalParquetStore:
    """Minimal local artifact store used for the targeted rerun."""

    base_dir: Path

    def path(self, stage: str, cfg: ExperimentConfig) -> Path:
        return self.base_dir / stage / cfg.task / cfg.dataset / f"{cfg.method.label}_seed{cfg.seed}.parquet"

    def exists(self, stage: str, cfg: ExperimentConfig) -> bool:
        return self.path(stage, cfg).exists()

    def save(self, stage: str, cfg: ExperimentConfig, df: pd.DataFrame) -> str:
        path = self.path(stage, cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return str(path)

    def load(self, stage: str, cfg: ExperimentConfig) -> pd.DataFrame:
        path = self.path(stage, cfg)
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_parquet(path)


def _ensure_base_aggregate(path: Path, *, stage: str) -> None:
    """Materialize the canonical classification aggregate locally if missing."""
    if path.exists():
        return

    config = load_config()
    bucket = get_s3_bucket()
    grid_keys = _build_grid_config_keys(TARGET_TASK)

    print(f"Bootstrapping {path.name} from s3://{bucket}/{stage}/{TARGET_TASK}/", flush=True)
    aggregate_stage(
        bucket=bucket,
        stage=stage,  # type: ignore[arg-type]
        task=TARGET_TASK,  # type: ignore[arg-type]
        output_path=path,
        grid_keys=grid_keys,
        region_name=config.aws_region,
        dry_run=False,
    )


def _ensure_base_aggregates() -> None:
    """Ensure the canonical classification aggregates exist before targeted replacement."""
    _ensure_base_aggregate(RANKINGS_PATH, stage="rankings")
    _ensure_base_aggregate(EVALUATION_PATH, stage="metrics")


def _target_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[(df["task"] == TARGET_TASK) & (df["method_base"].isin(TARGET_METHODS))].copy()


def _resolve_method_configs(method_ids: set[str]) -> list[MethodConfig]:
    configs = get_full_method_configs(list(TARGET_METHODS), TARGET_TASK)
    selected = [cfg for cfg in configs if cfg.label in method_ids]
    missing = sorted(method_ids - {cfg.label for cfg in selected})
    if missing:
        raise RuntimeError(f"Could not resolve method configs for: {missing}")
    return sorted(selected, key=lambda cfg: cfg.label)


def _build_configs(
    *,
    methods: list[MethodConfig],
    datasets: list[str],
    seeds: list[int],
) -> list[ExperimentConfig]:
    return [
        ExperimentConfig(method=method, dataset=dataset, seed=seed, task=TARGET_TASK)
        for method, dataset, seed in product(methods, datasets, seeds)
    ]


def _aggregate_stage(store: LocalParquetStore, stage: str, configs: list[ExperimentConfig]) -> pd.DataFrame:
    frames = [store.load(stage, cfg) for cfg in configs]
    return pd.concat(frames, ignore_index=True)


def _replace_rows(*, existing_path: Path, replacement: pd.DataFrame, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(existing_path, backup_dir / existing_path.name)

    existing = pd.read_parquet(existing_path)
    keep = existing[~((existing["task"] == TARGET_TASK) & (existing["method_base"].isin(TARGET_METHODS)))].copy()
    updated = pd.concat([keep, replacement], ignore_index=True)
    updated.to_parquet(existing_path, index=False)


def _run_selection_local(
    cfg: ExperimentConfig,
    store: LocalParquetStore,
    *,
    backend: str,
    n_jobs: int,
) -> str:
    if store.exists("rankings", cfg):
        return "skipped"

    X, y = load_dataset(cfg.dataset, cfg.task)
    dataset_meta = get_dataset_metadata(cfg.dataset, cfg.task)
    created_at_utc = utc_now_iso()
    tic = time.perf_counter()
    with parallel_backend(backend):
        fold_results = run_selection(
            X,
            y,
            cfg.method.name,
            cfg.task,
            cfg.seed,
            params=cfg.method.params_dict,
            n_jobs=n_jobs,
        )
    elapsed = time.perf_counter() - tic

    for row in fold_results:
        row.update(
            {
                "dataset": cfg.dataset,
                "task": cfg.task,
                "seed": cfg.seed,
                "method_id": cfg.method.label,
                "method": cfg.method.label,
                "method_base": cfg.method.name,
                "artifact_version": 2,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "selection_cpus": n_jobs,
                "elapsed_seconds": float(elapsed),
                "created_at_utc": created_at_utc,
                "git_sha": get_git_sha(),
                "library_versions": get_library_versions(),
                "dataset_source": dataset_meta.get("dataset_source"),
                "dataset_type": dataset_meta.get("dataset_type"),
                "dataset_family": dataset_meta.get("dataset_family"),
                "n_informative": dataset_meta.get("n_informative"),
            }
        )

    store.save("rankings", cfg, pd.DataFrame(fold_results))
    return "done"


def _run_evaluation_local(
    cfg: ExperimentConfig,
    store: LocalParquetStore,
    *,
    backend: str,
    n_jobs: int,
) -> str:
    if not store.exists("rankings", cfg):
        raise FileNotFoundError(f"Missing rankings for {cfg}")

    rankings_df = store.load("rankings", cfg)
    ranking_n_features = infer_n_features_from_rankings(rankings_df)
    requested_k_values = get_requested_evaluation_k_values(ranking_n_features)

    if store.exists("metrics", cfg):
        existing_metrics = store.load("metrics", cfg)
        if metrics_cover_requested_k_values(existing_metrics, requested_k_values):
            return "skipped"

    X, y = load_dataset(cfg.dataset, cfg.task)
    dataset_meta = get_dataset_metadata(cfg.dataset, cfg.task)
    created_at_utc = utc_now_iso()
    tic = time.perf_counter()
    with parallel_backend(backend):
        results = run_evaluation(X, y, rankings_df, cfg, n_jobs=n_jobs)
    elapsed = time.perf_counter() - tic

    for row in results:
        row.update(
            {
                "elapsed_seconds": float(elapsed),
                "created_at_utc": created_at_utc,
                "git_sha": get_git_sha(),
                "library_versions": get_library_versions(),
                "dataset_source": dataset_meta.get("dataset_source"),
                "dataset_type": dataset_meta.get("dataset_type"),
                "dataset_family": dataset_meta.get("dataset_family"),
                "n_informative": dataset_meta.get("n_informative"),
            }
        )

    store.save("metrics", cfg, pd.DataFrame(results))
    return "done"


def _run_stage1_and_stage2(
    store: LocalParquetStore,
    configs: list[ExperimentConfig],
    *,
    backend: str,
    n_jobs: int,
) -> None:
    total = len(configs)
    for idx, cfg in enumerate(configs, start=1):
        print(f"[{idx}/{total}] selection {cfg}", flush=True)
        selection_status = _run_selection_local(cfg, store, backend=backend, n_jobs=n_jobs)
        print(f"    selection status={selection_status}", flush=True)

        print(f"[{idx}/{total}] evaluation {cfg}", flush=True)
        evaluation_status = _run_evaluation_local(cfg, store, backend=backend, n_jobs=n_jobs)
        print(f"    evaluation status={evaluation_status}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=ROOT / "scratch" / "pi_cpi_balacc_rerun",
        help="Local scratch directory for rerun artifacts and backups",
    )
    parser.add_argument("--datasets", type=str, default="", help="Optional comma-separated dataset filter")
    parser.add_argument("--methods", type=str, default="", help="Optional comma-separated method_base filter")
    parser.add_argument("--seeds", type=str, default="", help="Optional comma-separated seed filter")
    parser.add_argument(
        "--skip-replace",
        action="store_true",
        help="Run reruns and keep scratch outputs without replacing canonical parquets",
    )
    parser.add_argument(
        "--backend",
        choices=["threading", "loky"],
        default="threading",
        help="Joblib backend for wrapper-selector parallel work",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel job count passed into Stage 1/Stage 2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_base_aggregates()
    target_rankings = _target_frame(RANKINGS_PATH)
    target_evaluation = _target_frame(EVALUATION_PATH)

    if set(target_rankings["method_id"].unique()) != set(target_evaluation["method_id"].unique()):
        raise RuntimeError("Ranking/evaluation method_id sets do not match")

    method_ids = set(target_rankings["method_id"].unique())
    methods = _resolve_method_configs(method_ids)
    datasets = sorted(target_rankings["dataset"].unique().tolist())
    seeds = sorted(target_rankings["seed"].unique().tolist())

    if args.methods:
        allowed_methods = {part.strip() for part in args.methods.split(",") if part.strip()}
        methods = [cfg for cfg in methods if cfg.name in allowed_methods]
    if args.datasets:
        allowed_datasets = {part.strip() for part in args.datasets.split(",") if part.strip()}
        datasets = [dataset for dataset in datasets if dataset in allowed_datasets]
    if args.seeds:
        allowed_seeds = {int(part.strip()) for part in args.seeds.split(",") if part.strip()}
        seeds = [seed for seed in seeds if seed in allowed_seeds]

    configs = _build_configs(methods=methods, datasets=datasets, seeds=seeds)
    if not configs:
        raise RuntimeError("No configs selected for rerun")

    print(f"Methods: {[cfg.label for cfg in methods]}", flush=True)
    print(f"Datasets: {len(datasets)}", flush=True)
    print(f"Seeds: {seeds}", flush=True)
    print(f"Configs: {len(configs)}", flush=True)

    store = LocalParquetStore(args.workdir.resolve())
    _run_stage1_and_stage2(store, configs, backend=args.backend, n_jobs=args.n_jobs)

    rankings_replacement = _aggregate_stage(store, "rankings", configs)
    evaluation_replacement = _aggregate_stage(store, "metrics", configs)
    print(f"Replacement rankings rows: {len(rankings_replacement)}", flush=True)
    print(f"Replacement evaluation rows: {len(evaluation_replacement)}", flush=True)

    if args.skip_replace:
        print("Skipping canonical parquet replacement", flush=True)
        return

    backup_dir = args.workdir.resolve() / "backups"
    _replace_rows(existing_path=RANKINGS_PATH, replacement=rankings_replacement, backup_dir=backup_dir)
    _replace_rows(existing_path=EVALUATION_PATH, replacement=evaluation_replacement, backup_dir=backup_dir)
    print(f"Updated {RANKINGS_PATH}", flush=True)
    print(f"Updated {EVALUATION_PATH}", flush=True)


if __name__ == "__main__":
    main()
