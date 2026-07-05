"""Build canonical joined data surfaces for paper-facing analyses.

The experiment pipeline historically wrote task-level artifacts, while a few
late single-tree runs landed in separate files. This script consolidates those
pieces into paper-facing surfaces so downstream table and figure builders do not
need method-specific sidecar logic.

Outputs:
  - paper/results/paper_real_evaluation.parquet
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

RESULTS_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "results"
REAL_EVALUATION_OUTPUT: Final[Path] = RESULTS_DIR / "paper_real_evaluation.parquet"
REAL_EVALUATION_INPUTS: Final[tuple[Path, ...]] = (
    RESULTS_DIR / "clf_evaluation.parquet",
    RESULTS_DIR / "reg_evaluation.parquet",
)

EVALUATION_DEDUP_KEYS: Final[tuple[str, ...]] = (
    "task",
    "dataset",
    "dataset_source",
    "method_base",
    "method_id",
    "seed",
    "fold_idx",
    "downstream_model",
    "k",
)


def _load_real_evaluation(path: Path) -> pd.DataFrame:
    """Load one evaluation artifact and keep real-data rows."""
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    if "dataset_source" not in frame.columns:
        raise ValueError(f"{path} has no dataset_source column")
    return frame[frame["dataset_source"] == "real"].copy()


def build_real_evaluation_surface() -> pd.DataFrame:
    """Return the joined real-data evaluation surface used by benchmark analyses."""
    frames = [_load_real_evaluation(path) for path in REAL_EVALUATION_INPUTS]
    joined = pd.concat(frames, ignore_index=True, sort=False)
    dedup_keys = [col for col in EVALUATION_DEDUP_KEYS if col in joined.columns]
    joined = joined.drop_duplicates(subset=dedup_keys, keep="last")
    return joined.sort_values(
        [
            "task",
            "dataset",
            "method_base",
            "method_id",
            "seed",
            "fold_idx",
            "downstream_model",
            "k",
        ],
        kind="stable",
    ).reset_index(drop=True)


def main() -> None:
    """Build and save paper-facing data surfaces."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    real_evaluation = build_real_evaluation_surface()
    real_evaluation.to_parquet(REAL_EVALUATION_OUTPUT, index=False)
    print(f"Saved {REAL_EVALUATION_OUTPUT} {real_evaluation.shape}")
    print(
        real_evaluation.groupby(["task", "method_base"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
