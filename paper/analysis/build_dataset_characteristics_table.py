"""Build the canonical dataset inventory from the packaged parquet tree.

This inventory is intended to be paper-facing. It scans the datasets shipped in
``paper/data`` rather than inferring membership from evaluation parquets, which
avoids stale rows when historical datasets are no longer part of the packaged
benchmark.

Output:
  - paper/results/tables/dataset_characteristics.csv

Usage:
  python paper/analysis/build_dataset_characteristics_table.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PAPER_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PAPER_DIR / "data"
TABLES_DIR = PAPER_DIR / "results" / "tables"


def _strip_task_prefix(stem: str) -> str:
    """Remove the task prefix from a packaged parquet stem."""
    if stem.startswith("clf_") or stem.startswith("reg_"):
        return stem[4:]
    return stem


def _infer_family_and_dtype(dataset: str, source: str) -> tuple[str, str]:
    """Return the paper-facing family and dtype labels for one dataset."""
    if source == "real":
        return "other", "real"

    name = dataset.lower()
    if "bias" in name:
        family = "bias"
    elif "corr_noise" in name:
        family = "confounder"
    elif "friedman" in name:
        family = "friedman"
    elif "heteroscedastic" in name:
        family = "heteroscedastic"
    elif "nonlinear" in name:
        family = "nonlinear"
    elif "redundant" in name:
        family = "redundant"
    elif "toeplitz" in name:
        family = "toeplitz"
    elif "weak" in name:
        family = "weak_signal"
    else:
        family = "standard"
    return family, family


def _count_features(df: pd.DataFrame) -> int:
    """Count feature columns in a packaged dataset frame."""
    if "y" in df.columns:
        return len([col for col in df.columns if col != "y"])
    if df.shape[1] == 0:
        return 0
    return df.shape[1] - 1


def build_inventory_table() -> pd.DataFrame:
    """Scan the packaged parquet tree and return the canonical inventory."""
    rows: list[dict[str, object]] = []

    for task in ("classification", "regression"):
        for source in ("real", "synthetic"):
            base = DATA_DIR / task / source
            for path in sorted(base.glob("*.parquet")):
                df = pd.read_parquet(path)
                dataset = _strip_task_prefix(path.stem)
                family, dtype = _infer_family_and_dtype(dataset, source)
                rows.append(
                    {
                        "dataset": dataset,
                        "n_samples": int(len(df)),
                        "n_features": int(_count_features(df)),
                        "task": task,
                        "source": source,
                        "family": family,
                        "dtype": dtype,
                    }
                )

    return (
        pd.DataFrame(rows)
        .sort_values(["task", "source", "n_features", "dataset"], kind="stable")
        .reset_index(drop=True)
    )


def main() -> None:
    """Write the canonical packaged dataset inventory."""
    parser = argparse.ArgumentParser(description="Build the packaged dataset inventory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TABLES_DIR,
        help="Directory where dataset_characteristics.csv will be written",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory_table()
    out_path = output_dir / "dataset_characteristics.csv"
    inventory.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(inventory)} datasets)")


if __name__ == "__main__":
    main()
