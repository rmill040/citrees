"""Acquire and prepare the public DGRP cardiac phenotype outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PHENOTYPE_URL = "https://cdn.elifesciences.org/articles/82459/elife-82459-fig1-data1-v1.xlsx"
PHENOTYPE_SHA256 = "ed0744c8c9359cc023402bcb78418f544eff62c6a49b4a0cd74b637e8bd2fc99"
PHENOTYPE_FILENAME = "elife-82459-fig1-data1-v1.xlsx"
PHENOTYPE_SHEET = "phenotype_data_REMOVE_OUTLIERS_"
MIN_INDIVIDUALS_PER_LINE = 7
EXPECTED_INDIVIDUAL_ROWS = 2_032
EXPECTED_SOURCE_LINES = 167
EXPECTED_OUTCOME_ROWS = 1_136
EXPECTED_ELIGIBLE_LINES = 166
EXPECTED_COMPLETE_LINES = 154

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "dgrp"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "dgrp"


@dataclass(frozen=True)
class TraitSpec:
    """Source definition for one cardiac outcome."""

    name: str
    column: str
    unit: str


TRAITS = (
    TraitSpec("DI", "DiastolicIntervals_Median", "seconds"),
    TraitSpec("SI", "SystolicIntervals_Median", "seconds"),
    TraitSpec("HP", "Heartperiod_Median", "seconds"),
    TraitSpec("EDD", "DiastolicMeanDiameter", "micrometers"),
    TraitSpec("ESD", "SystolicMeanDiameter", "micrometers"),
    TraitSpec("FS", "FractionalShortening", "dimensionless fraction"),
    TraitSpec("AI", "Heartperiod_StdDevOnMedian", "dimensionless SD/median ratio"),
)

EXPECTED_TRAIT_COUNTS = {
    "DI": (165, 1_914),
    "SI": (166, 1_911),
    "HP": (165, 1_920),
    "EDD": (159, 1_779),
    "ESD": (157, 1_753),
    "FS": (158, 1_767),
    "AI": (166, 1_832),
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(path: Path) -> None:
    """Require the pinned DGRP phenotype workbook."""
    observed = sha256(path)
    if observed != PHENOTYPE_SHA256:
        raise RuntimeError(
            f"DGRP phenotype checksum mismatch for {path}: "
            f"expected {PHENOTYPE_SHA256}, got {observed}"
        )


def acquire_phenotype_workbook(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download the pinned phenotype workbook when it is not already present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / PHENOTYPE_FILENAME
    if destination.exists():
        verify_source(destination)
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        raise RuntimeError(f"Partial DGRP download already exists: {partial}")

    try:
        with (
            urllib.request.urlopen(PHENOTYPE_URL, timeout=60) as response,
            partial.open("xb") as stream,
        ):
            shutil.copyfileobj(response, stream)
        verify_source(partial)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


def validate_individual_phenotypes(frame: pd.DataFrame) -> None:
    """Validate the columns and identities required by the DGRP analysis."""
    required = {"individual_name", "strain_number", *(trait.column for trait in TRAITS)}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"DGRP phenotype data is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("DGRP phenotype data is empty")
    if frame["individual_name"].isna().any() or frame["strain_number"].isna().any():
        raise ValueError("DGRP phenotype identities must be nonmissing")
    if frame["individual_name"].duplicated().any():
        raise ValueError("DGRP phenotype individual_name values must be unique")
    if not frame["strain_number"].astype(str).str.fullmatch(r"dgrp\d+").all():
        raise ValueError("DGRP strain_number values must match dgrp followed by digits")
    for trait in TRAITS:
        values = pd.to_numeric(frame[trait.column], errors="coerce")
        invalid = frame[trait.column].notna() & values.isna()
        if invalid.any():
            raise ValueError(f"DGRP trait {trait.name} contains nonnumeric values")
        finite_values = values[values.notna()].to_numpy(dtype=np.float64)
        if not np.isfinite(finite_values).all():
            raise ValueError(f"DGRP trait {trait.name} contains non-finite values")


def validate_pinned_individual_inventory(frame: pd.DataFrame) -> None:
    """Require the row and line inventory of the pinned public workbook."""
    if len(frame) != EXPECTED_INDIVIDUAL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_INDIVIDUAL_ROWS} DGRP individuals, found {len(frame)}"
        )
    n_lines = frame["strain_number"].nunique()
    if n_lines != EXPECTED_SOURCE_LINES:
        raise RuntimeError(f"Expected {EXPECTED_SOURCE_LINES} DGRP lines, found {n_lines}")


def load_individual_phenotypes(path: Path) -> pd.DataFrame:
    """Load and validate the pinned individual-level phenotype workbook."""
    verify_source(path)
    frame = pd.read_excel(path, sheet_name=PHENOTYPE_SHEET, engine="openpyxl")
    validate_individual_phenotypes(frame)
    validate_pinned_individual_inventory(frame)
    return frame


def build_line_outcomes(
    frame: pd.DataFrame,
    *,
    min_individuals: int = MIN_INDIVIDUALS_PER_LINE,
) -> pd.DataFrame:
    """Aggregate each trait by line after applying its observation-count rule."""
    if min_individuals < 1:
        raise ValueError("min_individuals must be positive")
    validate_individual_phenotypes(frame)

    rows: list[pd.DataFrame] = []
    for trait in TRAITS:
        values = pd.to_numeric(frame[trait.column], errors="coerce")
        grouped = (
            frame.assign(_outcome=values)
            .groupby("strain_number", sort=True)["_outcome"]
            .agg(n_individuals="count", outcome="mean")
            .reset_index()
        )
        grouped = grouped[grouped["n_individuals"] >= min_individuals].copy()
        grouped.insert(1, "trait", trait.name)
        grouped.insert(2, "source_column", trait.column)
        grouped.insert(3, "unit", trait.unit)
        rows.append(grouped)

    outcomes = pd.concat(rows, ignore_index=True)
    if not np.isfinite(outcomes["outcome"].to_numpy(dtype=float)).all():
        raise RuntimeError("DGRP line outcomes contain non-finite values")
    return outcomes


def summarize_line_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Summarize eligible lines and contributing individuals by trait."""
    return (
        outcomes.groupby(["trait", "source_column", "unit"], sort=False)
        .agg(
            n_lines=("strain_number", "nunique"),
            n_individuals=("n_individuals", "sum"),
        )
        .reset_index()
    )


def validate_pinned_line_outcomes(outcomes: pd.DataFrame) -> None:
    """Require the outcome inventory reproduced from the pinned workbook."""
    required = {
        "strain_number",
        "trait",
        "source_column",
        "unit",
        "n_individuals",
        "outcome",
    }
    missing = required - set(outcomes.columns)
    if missing:
        raise RuntimeError(f"DGRP line outcomes are missing columns: {sorted(missing)}")
    if len(outcomes) != EXPECTED_OUTCOME_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_OUTCOME_ROWS} DGRP line-trait outcomes, found {len(outcomes)}"
        )
    if outcomes.duplicated(["strain_number", "trait"]).any():
        raise RuntimeError("DGRP line outcomes contain duplicate line-trait rows")
    if not np.isfinite(outcomes["outcome"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("DGRP line outcomes contain non-finite values")

    summary = summarize_line_outcomes(outcomes)
    observed_counts = {
        str(row.trait): (int(row.n_lines), int(row.n_individuals))
        for row in summary.itertuples(index=False)
    }
    if observed_counts != EXPECTED_TRAIT_COUNTS:
        raise RuntimeError(
            "DGRP trait inventory mismatch: "
            f"expected {EXPECTED_TRAIT_COUNTS}, found {observed_counts}"
        )

    traits_per_line = outcomes.groupby("strain_number")["trait"].nunique()
    if len(traits_per_line) != EXPECTED_ELIGIBLE_LINES:
        raise RuntimeError(
            f"Expected {EXPECTED_ELIGIBLE_LINES} lines with at least one trait, "
            f"found {len(traits_per_line)}"
        )
    n_complete = int((traits_per_line == len(TRAITS)).sum())
    if n_complete != EXPECTED_COMPLETE_LINES:
        raise RuntimeError(
            f"Expected {EXPECTED_COMPLETE_LINES} lines with all traits, found {n_complete}"
        )


def write_results(
    source_path: Path,
    outcomes: pd.DataFrame,
    output_dir: Path,
    *,
    minimum_individuals: int = MIN_INDIVIDUALS_PER_LINE,
) -> None:
    """Write prepared outcomes, summary counts, and a source receipt."""
    verify_source(source_path)
    validate_pinned_line_outcomes(outcomes)
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = output_dir / "line_outcomes.parquet"
    summary_path = output_dir / "trait_summary.csv"
    outcomes.to_parquet(outcomes_path, index=False)
    summarize_line_outcomes(outcomes).to_csv(summary_path, index=False)

    artifacts = (outcomes_path, summary_path)
    receipt = {
        "analysis": "dgrp_phenotypes",
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "source": {
            "url": PHENOTYPE_URL,
            "filename": PHENOTYPE_FILENAME,
            "bytes": source_path.stat().st_size,
            "sha256": sha256(source_path),
            "sheet": PHENOTYPE_SHEET,
        },
        "minimum_individuals_per_line": minimum_individuals,
        "traits": [asdict(trait) for trait in TRAITS],
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in artifacts
        },
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_path = acquire_phenotype_workbook(args.data_dir)
    frame = load_individual_phenotypes(source_path)
    outcomes = build_line_outcomes(frame)
    write_results(
        source_path,
        outcomes,
        args.output_dir,
        minimum_individuals=MIN_INDIVIDUALS_PER_LINE,
    )
    print(f"Wrote DGRP outcomes for {outcomes['strain_number'].nunique()} lines.")


if __name__ == "__main__":
    main()
