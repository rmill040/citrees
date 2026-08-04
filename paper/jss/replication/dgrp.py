"""Acquire and validate the public DGRP phenotype and genotype inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

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

GENOTYPE_RECORD_ID = 5_582_846
GENOTYPE_URL = "https://zenodo.org/api/records/5582846/files/dgrp2.tar.gz/content"
GENOTYPE_ARCHIVE_FILENAME = "dgrp2.tar.gz"
GENOTYPE_ARCHIVE_BYTES = 97_913_546
GENOTYPE_ARCHIVE_MD5 = "77c26d1469b18d7da5e6597b6d466454"
GENOTYPE_ARCHIVE_SHA256 = "ff3c318debf28b02d61293b2b82cd12047273f5d743387d748d1ce308ea4c452"
GENOTYPE_BED_MAGIC = bytes.fromhex("6c1b01")
EXPECTED_GENOTYPE_SAMPLES = 205
EXPECTED_GENOTYPE_VARIANTS = 4_438_427
EXPECTED_GENOTYPE_BED_BYTES = 230_798_207
EXPECTED_GENOTYPE_ONLY_LINES = 38

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "dgrp"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "dgrp"


@dataclass(frozen=True)
class TraitSpec:
    """Source definition for one cardiac outcome."""

    name: str
    column: str
    unit: str


@dataclass(frozen=True)
class GenotypeFileSpec:
    """Pinned metadata for one file in the DGRP PLINK archive."""

    member_name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class GenotypeInventory:
    """Validated identities and dimensions of the DGRP PLINK files."""

    genotype_lines: tuple[str, ...]
    variant_count: int
    bed_bytes: int


@dataclass(frozen=True)
class LineOverlap:
    """Normalized line identities shared by the phenotype and genotype sources."""

    phenotype_lines: tuple[str, ...]
    genotype_lines: tuple[str, ...]
    genotype_only_lines: tuple[str, ...]


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

GENOTYPE_FILE_SPECS = (
    GenotypeFileSpec(
        "input/dgrp2.bed",
        EXPECTED_GENOTYPE_BED_BYTES,
        "2855e4fab69dde2ed0016d503b87a293c436c423bcda8b1021048d173df43ce7",
    ),
    GenotypeFileSpec(
        "input/dgrp2.bim",
        148_523_556,
        "3d1d0e77c90cd135360fbf79113df3d9c94648503c50e0ed1fd54240625909be",
    ),
    GenotypeFileSpec(
        "input/dgrp2.fam",
        5_491,
        "387d760ae033a3d97f261bef8ea1fc256d26c2528c97698b565219bf3df69a2b",
    ),
)

_PHENOTYPE_LINE_PATTERN = re.compile(r"dgrp([0-9]+)")
_GENOTYPE_LINE_PATTERN = re.compile(r"line_([0-9]+)")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_and_sha256(path: Path) -> tuple[str, str]:
    """Return both pinned archive digests after one streaming read."""
    md5_digest = hashlib.md5(usedforsecurity=False)
    sha256_digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            md5_digest.update(chunk)
            sha256_digest.update(chunk)
    return md5_digest.hexdigest(), sha256_digest.hexdigest()


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


def verify_genotype_archive(
    path: Path,
    *,
    expected_bytes: int | None = None,
    expected_md5: str | None = None,
    expected_sha256: str | None = None,
) -> tuple[str, str]:
    """Require the pinned archive and return its observed MD5 and SHA-256."""
    expected_bytes = GENOTYPE_ARCHIVE_BYTES if expected_bytes is None else expected_bytes
    expected_md5 = GENOTYPE_ARCHIVE_MD5 if expected_md5 is None else expected_md5
    expected_sha256 = GENOTYPE_ARCHIVE_SHA256 if expected_sha256 is None else expected_sha256
    observed_bytes = path.stat().st_size
    if observed_bytes != expected_bytes:
        raise RuntimeError(
            f"DGRP genotype archive size mismatch for {path}: "
            f"expected {expected_bytes}, got {observed_bytes}"
        )
    observed_md5, observed_sha256 = _md5_and_sha256(path)
    if observed_md5 != expected_md5:
        raise RuntimeError(
            f"DGRP genotype archive MD5 mismatch for {path}: "
            f"expected {expected_md5}, got {observed_md5}"
        )
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"DGRP genotype archive SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, got {observed_sha256}"
        )
    return observed_md5, observed_sha256


def acquire_genotype_archive(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download the pinned DGRP genotype archive when it is not already present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / GENOTYPE_ARCHIVE_FILENAME
    if destination.exists():
        verify_genotype_archive(destination)
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        raise RuntimeError(f"Partial DGRP genotype download already exists: {partial}")

    try:
        with (
            urllib.request.urlopen(GENOTYPE_URL, timeout=60) as response,
            partial.open("xb") as stream,
        ):
            shutil.copyfileobj(response, stream)
        verify_genotype_archive(partial)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


def _validated_tar_members(
    archive: tarfile.TarFile,
    specs: Sequence[GenotypeFileSpec],
) -> dict[str, tarfile.TarInfo]:
    """Return exact regular archive members after rejecting unsafe paths."""
    expected = {spec.member_name: spec for spec in specs}
    if len(expected) != len(specs):
        raise ValueError("DGRP genotype file specifications contain duplicate paths")

    members = archive.getmembers()
    observed_names = [member.name for member in members]
    if len(observed_names) != len(set(observed_names)):
        raise RuntimeError("DGRP genotype archive contains duplicate member paths")

    for member in members:
        member_path = PurePosixPath(member.name)
        if member_path.is_absolute() or ".." in member_path.parts or "\\" in member.name:
            raise RuntimeError(f"Unsafe DGRP genotype archive member: {member.name}")
        if not member.isfile():
            raise RuntimeError(f"DGRP genotype archive member is not a regular file: {member.name}")

    observed = set(observed_names)
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        extra = sorted(observed - set(expected))
        raise RuntimeError(
            f"DGRP genotype archive member mismatch: missing={missing}, extra={extra}"
        )

    indexed = {member.name: member for member in members}
    for name, spec in expected.items():
        observed_size = indexed[name].size
        if observed_size != spec.size_bytes:
            raise RuntimeError(
                f"DGRP genotype member size mismatch for {name}: "
                f"expected {spec.size_bytes}, got {observed_size}"
            )
    return indexed


def validate_genotype_archive_members(archive_path: Path) -> None:
    """Require the exact three regular files in the pinned genotype archive."""
    with tarfile.open(archive_path, mode="r:gz") as archive:
        _validated_tar_members(archive, GENOTYPE_FILE_SPECS)


def _safe_extract_genotype_members(
    archive_path: Path,
    destination: Path,
    specs: Sequence[GenotypeFileSpec],
) -> tuple[Path, ...]:
    """Extract validated members to fixed destinations without trusting tar paths."""
    if destination.is_symlink():
        raise RuntimeError(f"DGRP extraction destination must not be a symlink: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise RuntimeError(f"DGRP extraction destination must be empty: {destination}")

    written: list[Path] = []
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = _validated_tar_members(archive, specs)
            for spec in specs:
                target = destination.joinpath(*PurePosixPath(spec.member_name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(members[spec.member_name])
                if source is None:
                    raise RuntimeError(
                        f"Unable to read DGRP genotype archive member: {spec.member_name}"
                    )
                with source, target.open("xb") as stream:
                    written.append(target)
                    shutil.copyfileobj(source, stream)

                observed_sha256 = sha256(target)
                if observed_sha256 != spec.sha256:
                    raise RuntimeError(
                        f"DGRP genotype checksum mismatch for {spec.member_name}: "
                        f"expected {spec.sha256}, got {observed_sha256}"
                    )
    except Exception:
        shutil.rmtree(destination)
        destination.mkdir(parents=True)
        raise
    return tuple(written)


def normalize_phenotype_line(value: str) -> str:
    """Map a phenotype identity such as dgrp21 to the PLINK identity line_21."""
    match = _PHENOTYPE_LINE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"DGRP phenotype line must match dgrp followed by digits: {value!r}")
    return f"line_{int(match.group(1))}"


def _normalize_genotype_line(value: str) -> str:
    """Return the canonical numeric identity for one PLINK DGRP line."""
    match = _GENOTYPE_LINE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"DGRP genotype line must match line_ followed by digits: {value!r}")
    return f"line_{int(match.group(1))}"


def _line_sort_key(value: str) -> int:
    """Return the numeric component used for deterministic line ordering."""
    match = _GENOTYPE_LINE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid normalized DGRP genotype line: {value!r}")
    return int(match.group(1))


def validate_fam(
    path: Path,
    *,
    expected_samples: int | None = EXPECTED_GENOTYPE_SAMPLES,
) -> tuple[str, ...]:
    """Validate FAM identities and return normalized genotype lines in file order."""
    genotype_lines: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="ascii", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(
                    f"DGRP FAM row {line_number} must contain 6 fields, found {len(fields)}"
                )
            family_id, individual_id = fields[:2]
            if family_id != individual_id:
                raise ValueError(f"DGRP FAM row {line_number} has different FID and IID values")
            normalized = _normalize_genotype_line(family_id)
            if normalized in seen:
                raise ValueError(f"DGRP FAM contains duplicate line identity: {normalized}")
            seen.add(normalized)
            genotype_lines.append(normalized)

    if expected_samples is not None and len(genotype_lines) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} DGRP genotype samples, found {len(genotype_lines)}"
        )
    return tuple(genotype_lines)


def validate_bim(
    path: Path,
    *,
    expected_variants: int | None = EXPECTED_GENOTYPE_VARIANTS,
) -> int:
    """Validate BIM rows, unique variant IDs, and positive integer positions."""
    variant_ids: set[str] = set()
    variant_count = 0
    with path.open("r", encoding="ascii", newline="", buffering=1024 * 1024) as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(
                    f"DGRP BIM row {line_number} must contain 6 fields, found {len(fields)}"
                )
            variant_id = fields[1]
            if variant_id in variant_ids:
                raise ValueError(f"DGRP BIM contains duplicate variant ID: {variant_id}")
            variant_ids.add(variant_id)

            position_text = fields[3]
            if not position_text.isdecimal() or int(position_text) <= 0:
                raise ValueError(
                    f"DGRP BIM row {line_number} has invalid integer position: {position_text!r}"
                )
            variant_count += 1
            if expected_variants is not None and variant_count > expected_variants:
                raise RuntimeError(
                    f"Expected {expected_variants} DGRP variants, found more than that"
                )

    if expected_variants is not None and variant_count != expected_variants:
        raise RuntimeError(f"Expected {expected_variants} DGRP variants, found {variant_count}")
    return variant_count


def variant_major_bed_bytes(sample_count: int, variant_count: int) -> int:
    """Return the exact PLINK BED size for variant-major two-bit genotypes."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if variant_count < 1:
        raise ValueError("variant_count must be positive")
    bytes_per_variant = (sample_count + 3) // 4
    return len(GENOTYPE_BED_MAGIC) + bytes_per_variant * variant_count


def validate_bed(
    path: Path,
    *,
    sample_count: int,
    variant_count: int,
    expected_magic: bytes = GENOTYPE_BED_MAGIC,
) -> None:
    """Validate PLINK variant-major magic and size derived from data dimensions."""
    expected_bytes = variant_major_bed_bytes(sample_count, variant_count)
    observed_bytes = path.stat().st_size
    if observed_bytes != expected_bytes:
        raise RuntimeError(
            f"DGRP BED size mismatch for {path}: expected {expected_bytes}, got {observed_bytes}"
        )
    with path.open("rb") as stream:
        observed_magic = stream.read(len(expected_magic))
    if observed_magic != expected_magic:
        raise RuntimeError(
            f"DGRP BED magic mismatch for {path}: "
            f"expected {expected_magic.hex()}, got {observed_magic.hex()}"
        )


def _input_paths(input_dir: Path) -> dict[str, Path]:
    """Return expected extracted paths keyed by archive member name."""
    return {
        spec.member_name: input_dir / PurePosixPath(spec.member_name).name
        for spec in GENOTYPE_FILE_SPECS
    }


def validate_genotype_files(input_dir: Path) -> GenotypeInventory:
    """Validate the pinned DGRP PLINK files and inventory."""
    if input_dir.is_symlink() or not input_dir.is_dir():
        raise RuntimeError(f"DGRP genotype input directory is invalid: {input_dir}")

    paths = _input_paths(input_dir)
    expected_names = {path.name for path in paths.values()}
    observed_names = {path.name for path in input_dir.iterdir()}
    if not expected_names.issubset(observed_names):
        missing = sorted(expected_names - observed_names)
        raise RuntimeError(f"DGRP genotype input files are missing: {missing}")

    for spec in GENOTYPE_FILE_SPECS:
        path = paths[spec.member_name]
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"DGRP genotype input must be a regular file: {path}")
        observed_bytes = path.stat().st_size
        if observed_bytes != spec.size_bytes:
            raise RuntimeError(
                f"DGRP genotype file size mismatch for {path.name}: "
                f"expected {spec.size_bytes}, got {observed_bytes}"
            )
        observed_sha256 = sha256(path)
        if observed_sha256 != spec.sha256:
            raise RuntimeError(
                f"DGRP genotype checksum mismatch for {path.name}: "
                f"expected {spec.sha256}, got {observed_sha256}"
            )

    genotype_lines = validate_fam(
        paths["input/dgrp2.fam"],
        expected_samples=EXPECTED_GENOTYPE_SAMPLES,
    )
    variant_count = validate_bim(
        paths["input/dgrp2.bim"],
        expected_variants=EXPECTED_GENOTYPE_VARIANTS,
    )
    validate_bed(
        paths["input/dgrp2.bed"],
        sample_count=len(genotype_lines),
        variant_count=variant_count,
        expected_magic=GENOTYPE_BED_MAGIC,
    )
    return GenotypeInventory(
        genotype_lines=genotype_lines,
        variant_count=variant_count,
        bed_bytes=paths["input/dgrp2.bed"].stat().st_size,
    )


def extract_genotype_archive(
    archive_path: Path,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> Path:
    """Safely install the pinned PLINK files while preserving unrelated inputs."""
    verify_genotype_archive(archive_path)
    validate_genotype_archive_members(archive_path)
    data_dir.mkdir(parents=True, exist_ok=True)
    input_dir = data_dir / "input"
    if input_dir.is_symlink() or (input_dir.exists() and not input_dir.is_dir()):
        raise RuntimeError(f"DGRP genotype input directory is invalid: {input_dir}")
    if input_dir.is_dir():
        expected_paths = tuple(_input_paths(input_dir).values())
        if any(path.exists() or path.is_symlink() for path in expected_paths):
            validate_genotype_files(input_dir)
            return input_dir

    with tempfile.TemporaryDirectory(prefix=".dgrp2-", dir=data_dir) as temporary:
        temporary_root = Path(temporary)
        _safe_extract_genotype_members(
            archive_path,
            temporary_root,
            GENOTYPE_FILE_SPECS,
        )
        temporary_input = temporary_root / "input"
        validate_genotype_files(temporary_input)
        if not input_dir.exists():
            temporary_input.replace(input_dir)
        else:
            installed: list[Path] = []
            try:
                for source in _input_paths(temporary_input).values():
                    destination = input_dir / source.name
                    os.link(source, destination)
                    installed.append(destination)
                validate_genotype_files(input_dir)
            except Exception:
                for path in installed:
                    path.unlink(missing_ok=True)
                raise
    return input_dir


def validate_line_overlap(
    phenotype_lines: Iterable[str],
    genotype_lines: Iterable[str],
) -> LineOverlap:
    """Require every normalized phenotype line to have a genotype sample."""
    normalized_phenotypes = {normalize_phenotype_line(value) for value in phenotype_lines}
    normalized_genotypes_list = [_normalize_genotype_line(value) for value in genotype_lines]
    normalized_genotypes = set(normalized_genotypes_list)
    if not normalized_phenotypes:
        raise ValueError("DGRP phenotype line inventory is empty")
    if not normalized_genotypes:
        raise ValueError("DGRP genotype line inventory is empty")
    if len(normalized_genotypes) != len(normalized_genotypes_list):
        raise ValueError("DGRP genotype line inventory contains duplicate identities")

    missing = normalized_phenotypes - normalized_genotypes
    if missing:
        raise RuntimeError(
            "DGRP phenotype lines are missing from the genotype inventory: "
            f"{sorted(missing, key=_line_sort_key)}"
        )

    genotype_only = normalized_genotypes - normalized_phenotypes
    return LineOverlap(
        phenotype_lines=tuple(sorted(normalized_phenotypes, key=_line_sort_key)),
        genotype_lines=tuple(sorted(normalized_genotypes, key=_line_sort_key)),
        genotype_only_lines=tuple(sorted(genotype_only, key=_line_sort_key)),
    )


def validate_pinned_line_overlap(
    phenotype_lines: Iterable[str],
    genotype_lines: Iterable[str],
) -> LineOverlap:
    """Require the exact overlap between the pinned phenotype and genotype sources."""
    overlap = validate_line_overlap(phenotype_lines, genotype_lines)
    observed_counts = (
        len(overlap.phenotype_lines),
        len(overlap.genotype_lines),
        len(overlap.genotype_only_lines),
    )
    expected_counts = (
        EXPECTED_SOURCE_LINES,
        EXPECTED_GENOTYPE_SAMPLES,
        EXPECTED_GENOTYPE_ONLY_LINES,
    )
    if observed_counts != expected_counts:
        raise RuntimeError(
            "DGRP phenotype-genotype line inventory mismatch: "
            f"expected {expected_counts}, found {observed_counts}"
        )
    return overlap


def build_genotype_source_receipt(
    archive_path: Path,
    input_dir: Path,
    phenotype_lines: Iterable[str],
) -> dict[str, object]:
    """Build deterministic provenance after validating both pinned sources."""
    observed_md5, observed_sha256 = verify_genotype_archive(archive_path)
    validate_genotype_archive_members(archive_path)
    inventory = validate_genotype_files(input_dir)
    overlap = validate_pinned_line_overlap(
        phenotype_lines,
        inventory.genotype_lines,
    )
    paths = _input_paths(input_dir)
    return {
        "schema_version": 1,
        "source": {
            "record_id": GENOTYPE_RECORD_ID,
            "url": GENOTYPE_URL,
            "filename": GENOTYPE_ARCHIVE_FILENAME,
            "bytes": archive_path.stat().st_size,
            "md5": observed_md5,
            "sha256": observed_sha256,
        },
        "files": {
            spec.member_name: {
                "bytes": paths[spec.member_name].stat().st_size,
                "sha256": sha256(paths[spec.member_name]),
            }
            for spec in GENOTYPE_FILE_SPECS
        },
        "inventory": {
            "samples": len(inventory.genotype_lines),
            "variants": inventory.variant_count,
            "bed_bytes": inventory.bed_bytes,
            "bed_magic": GENOTYPE_BED_MAGIC.hex(),
        },
        "line_overlap": {
            "phenotype_source_lines": len(overlap.phenotype_lines),
            "genotype_lines": len(overlap.genotype_lines),
            "shared_lines": len(overlap.phenotype_lines),
            "genotype_only_lines": len(overlap.genotype_only_lines),
            "genotype_only_ids": list(overlap.genotype_only_lines),
        },
    }


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
    genotype_provenance: Mapping[str, object],
    minimum_individuals: int = MIN_INDIVIDUALS_PER_LINE,
) -> None:
    """Write prepared outcomes and phenotype-genotype source provenance."""
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
        "schema_version": 2,
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
        "genotype": dict(genotype_provenance),
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
    genotype_archive = acquire_genotype_archive(args.data_dir)
    genotype_input = extract_genotype_archive(genotype_archive, args.data_dir)
    genotype_provenance = build_genotype_source_receipt(
        genotype_archive,
        genotype_input,
        frame["strain_number"],
    )
    outcomes = build_line_outcomes(frame)
    write_results(
        source_path,
        outcomes,
        args.output_dir,
        genotype_provenance=genotype_provenance,
        minimum_individuals=MIN_INDIVIDUALS_PER_LINE,
    )
    print(f"Wrote DGRP outcomes for {outcomes['strain_number'].nunique()} lines.")


if __name__ == "__main__":
    main()
