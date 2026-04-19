"""Build an old-hash -> current-hash alias manifest from git history.

Operational audit helper; not part of the current paper-facing rebuild path.

This script reconstructs a previous config grid from git, normalizes parameter
encodings into the current schema, and reports which old labels are semantically
equivalent to current labels.

It also checks whether the old-label raw artifacts would fill any current-grid
gaps, which determines whether a raw-file migration is useful or merely
cosmetic.

Outputs:
  - paper/results/tables/grid_hash_aliases.csv
  - paper/results/tables/grid_hash_aliases_summary.json

Usage:
  UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/maintenance/audit_hash_alias_manifest.py \
      --old-spec '040f681a^:paper/scripts/pipeline/config.py' --local-dir ../data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from paper.scripts.pipeline.config import get_configs as get_current_configs
from paper.scripts.pipeline.grid import ExperimentGrid

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TABLES_DIR = RESULTS_DIR / "tables"


def _label_from_params(method: str, params: dict[str, Any]) -> str:
    """Compute the method label used by the experiment grid."""
    filtered = {k: v for k, v in sorted(params.items()) if k not in ("method", "random_state")}
    digest = hashlib.md5(json.dumps(filtered, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return f"{method}__{digest}"


def _norm_key(params: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Canonical tuple key for dict comparison."""
    return tuple(
        sorted(
            (k, json.dumps(v, sort_keys=True, default=str))
            for k, v in params.items()
            if k != "random_state"
        )
    )


def _load_old_grid(spec: str) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Execute a historical config module from git and return its get_configs()."""
    source = subprocess.check_output(["git", "show", spec], text=True)
    namespace: dict[str, Any] = {"__name__": "__grid_snapshot__"}
    exec(source, namespace)
    return {task: namespace["get_configs"](task) for task in ("classification", "regression")}


def _normalize_old_params(
    task: Literal["classification", "regression"],
    method: str,
    params: dict[str, Any],
) -> dict[str, Any] | None:
    """Normalize historical params into the current schema when exact equivalence exists."""
    normalized = dict(params)

    bootstrap_method = normalized.pop("bootstrap_method", None)
    if bootstrap_method is not None:
        if bootstrap_method != "classic":
            return None
        normalized["bootstrap"] = True

    if method in {"cif", "cit"}:
        if normalized.get("n_resamples_selector") != "minimum":
            return None
        if normalized.get("n_resamples_splitter") != "minimum":
            return None
        if normalized.get("threshold_method") != "histogram":
            return None
        if normalized.get("max_thresholds") != 256:
            return None

    if task == "classification" and method in {"cif", "cit"} and normalized.get("selector") not in {"mc", "rdc"}:
        return None
    if task == "regression" and method in {"cif", "cit"} and normalized.get("selector") not in {"pc", "rdc"}:
        return None
    if method == "cif" and task == "classification" and normalized.get("sampling_method") != "stratified":
        return None
    if method == "cif" and normalized.get("bootstrap") is not True:
        return None

    return normalized


def _collect_raw_keys(
    local_dir: Path,
    stage: Literal["rankings", "metrics"],
    task: Literal["classification", "regression"],
) -> dict[str, set[tuple[str, int]]]:
    """Collect raw local artifact coverage by method label."""
    base = local_dir / stage / task
    by_label: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for path in base.rglob("*.parquet"):
        stem = path.stem
        if "_seed" not in stem:
            continue
        method_id, seed_text = stem.rsplit("_seed", 1)
        by_label[method_id].add((path.parent.name, int(seed_text)))
    return by_label


def _build_manifest(
    old_grid: dict[str, dict[str, list[dict[str, Any]]]],
    local_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the alias manifest with raw-artifact coverage stats."""
    current_norm: dict[str, dict[tuple[tuple[str, str], ...], str]] = {}
    expected_by_label: dict[str, dict[str, set[tuple[str, int]]]] = {}

    for task in ("classification", "regression"):
        task = task  # type: ignore[assignment]
        current_norm[task] = {}
        expected_by_label[task] = defaultdict(set)

        for method, configs in get_current_configs(task).items():
            for params in configs:
                label = _label_from_params(method, params)
                current_norm[task][_norm_key(params)] = label

        for method_id, dataset, seed in {cfg.key for cfg in ExperimentGrid.from_cli(task, source="all")}:
            expected_by_label[task][method_id].add((dataset, seed))

    raw_rank = {
        task: _collect_raw_keys(local_dir, "rankings", task) for task in ("classification", "regression")
    }
    raw_metric = {
        task: _collect_raw_keys(local_dir, "metrics", task) for task in ("classification", "regression")
    }

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    for task in ("classification", "regression"):
        mapped = 0
        unmapped = 0
        rows_before = len(rows)

        for method, configs in old_grid[task].items():
            for params in configs:
                old_label = _label_from_params(method, params)
                normalized = _normalize_old_params(task, method, params)  # type: ignore[arg-type]
                if normalized is None:
                    unmapped += 1
                    continue

                new_label = current_norm[task].get(_norm_key(normalized))
                if new_label is None or new_label == old_label:
                    continue

                mapped += 1
                expected_pairs = expected_by_label[task].get(new_label, set())
                old_rank_pairs = raw_rank[task].get(old_label, set())
                old_metric_pairs = raw_metric[task].get(old_label, set())
                new_rank_pairs = raw_rank[task].get(new_label, set())
                new_metric_pairs = raw_metric[task].get(new_label, set())

                missing_rank_pairs = expected_pairs - new_rank_pairs
                missing_metric_pairs = expected_pairs - new_metric_pairs
                alias_rank_pairs = old_rank_pairs & expected_pairs
                alias_metric_pairs = old_metric_pairs & expected_pairs

                rows.append(
                    {
                        "task": task,
                        "method_base": method,
                        "old_label": old_label,
                        "new_label": new_label,
                        "old_rank_jobs": len(old_rank_pairs),
                        "old_metric_jobs": len(old_metric_pairs),
                        "new_rank_jobs": len(new_rank_pairs),
                        "new_metric_jobs": len(new_metric_pairs),
                        "expected_jobs": len(expected_pairs),
                        "fills_missing_rank_jobs": len(alias_rank_pairs & missing_rank_pairs),
                        "fills_missing_metric_jobs": len(alias_metric_pairs & missing_metric_pairs),
                        "rank_alias_is_complete": alias_rank_pairs == expected_pairs,
                        "metric_alias_is_complete": alias_metric_pairs == expected_pairs,
                        "sample_alias_rank_pairs": ";".join(
                            f"{dataset}/seed{seed}" for dataset, seed in sorted(alias_rank_pairs)[:10]
                        ),
                    }
                )

        summary[task] = {
            "mapped_aliases": mapped,
            "unmapped_old_configs": unmapped,
            "aliases_with_raw_presence": sum(
                1
                for row in rows[rows_before:]
                if row["old_rank_jobs"] > 0 or row["old_metric_jobs"] > 0
            ),
            "aliases_that_fill_current_rank_gaps": sum(1 for row in rows[rows_before:] if row["fills_missing_rank_jobs"] > 0),
            "aliases_that_fill_current_metric_gaps": sum(
                1 for row in rows[rows_before:] if row["fills_missing_metric_jobs"] > 0
            ),
        }

    frame = pd.DataFrame(rows).sort_values(["task", "method_base", "old_label"]).reset_index(drop=True)
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit old-hash -> current-hash config aliases")
    parser.add_argument(
        "--old-spec",
        default="040f681a^:paper/scripts/pipeline/config.py",
        help="git show spec for the historical config module",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("../data"),
        help="Local data directory containing rankings/ and metrics/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TABLES_DIR,
        help="Directory for alias audit outputs",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    old_grid = _load_old_grid(args.old_spec)
    manifest, summary = _build_manifest(old_grid, args.local_dir.resolve())

    csv_out = output_dir / "grid_hash_aliases.csv"
    json_out = output_dir / "grid_hash_aliases_summary.json"

    manifest.to_csv(csv_out, index=False)
    json_out.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Saved {csv_out}")
    print(f"Saved {json_out}")


if __name__ == "__main__":
    main()
