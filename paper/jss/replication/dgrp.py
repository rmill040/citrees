"""Acquire and validate the public DGRP phenotype and genotype inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
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
DGRP_MIN_CALL_RATE = 0.95
DGRP_MIN_MINOR_ALLELE_FREQUENCY = 0.04

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


def _git_sha(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_dirty(repo_root: Path) -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


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
    except BaseException:
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
    except BaseException:
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
    except BaseException:
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


def load_bim_metadata(path: Path, variant_indices: Sequence[int]) -> pd.DataFrame:
    """Load genomic metadata for selected BIM rows in variant-index order."""
    indices = np.asarray(variant_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("variant_indices must be a nonempty one-dimensional sequence")
    if np.any(indices < 0) or np.any(np.diff(indices) <= 0):
        raise ValueError("variant_indices must be strictly increasing and nonnegative")
    wanted = set(indices.tolist())
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="ascii", newline="", buffering=1024 * 1024) as stream:
        for variant_index, line in enumerate(stream):
            if variant_index not in wanted:
                continue
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"DGRP BIM row {variant_index + 1} must contain 6 fields")
            chromosome, variant_id, _distance, position, allele_a, allele_b = fields
            if not position.isdecimal() or int(position) <= 0:
                raise ValueError(f"DGRP BIM row {variant_index + 1} has invalid position")
            rows.append(
                {
                    "variant_index": variant_index,
                    "chromosome": chromosome,
                    "variant_id": variant_id,
                    "position": int(position),
                    "allele_a": allele_a,
                    "allele_b": allele_b,
                }
            )
            if len(rows) == len(indices):
                break
    if len(rows) != len(indices):
        raise ValueError("BIM file does not contain every requested variant index")
    return pd.DataFrame(rows)


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


def decode_bed_variants(
    path: Path,
    *,
    sample_count: int,
    variant_count: int,
    start_variant: int,
    n_variants: int,
) -> np.ndarray:
    """Decode a contiguous variant-major PLINK BED block into dosages.

    The returned array has shape ``(n_variants, sample_count)``.  PLINK's
    two-bit missing-value code is represented as ``-1``; observed dosages are
    represented as ``0``, ``1``, or ``2``.
    """
    if start_variant < 0:
        raise ValueError("start_variant must be nonnegative")
    if n_variants < 1:
        raise ValueError("n_variants must be positive")
    if start_variant + n_variants > variant_count:
        raise ValueError("requested BED variant block exceeds variant_count")

    validate_bed(path, sample_count=sample_count, variant_count=variant_count)
    bytes_per_variant = (sample_count + 3) // 4
    offset = len(GENOTYPE_BED_MAGIC) + start_variant * bytes_per_variant
    expected_bytes = n_variants * bytes_per_variant
    with path.open("rb") as stream:
        stream.seek(offset)
        payload = stream.read(expected_bytes)
    if len(payload) != expected_bytes:
        raise RuntimeError(
            f"DGRP BED block is truncated: expected {expected_bytes} bytes, got {len(payload)}"
        )

    packed = np.frombuffer(payload, dtype=np.uint8).reshape(n_variants, bytes_per_variant)
    shifts = (2 * np.arange(4, dtype=np.uint8)).reshape(1, 4)
    codes = ((packed[:, :, None] >> shifts) & 0b11).reshape(n_variants, -1)
    dosage_map = np.array([0, -1, 1, 2], dtype=np.int8)
    return dosage_map[codes[:, :sample_count]]


def summarize_bed_variants(
    path: Path,
    *,
    sample_count: int,
    variant_count: int,
    block_size: int = 16_384,
) -> pd.DataFrame:
    """Summarize every BED variant without materializing the genotype matrix."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    validate_bed(path, sample_count=sample_count, variant_count=variant_count)

    rows: list[pd.DataFrame] = []
    for start_variant in range(0, variant_count, block_size):
        n_variants = min(block_size, variant_count - start_variant)
        dosages = decode_bed_variants(
            path,
            sample_count=sample_count,
            variant_count=variant_count,
            start_variant=start_variant,
            n_variants=n_variants,
        )
        called = dosages >= 0
        n_called = called.sum(axis=1, dtype=np.int64)
        allele_sum = np.where(called, dosages, 0).sum(axis=1, dtype=np.int64)
        allele_frequency = np.full(n_variants, np.nan, dtype=np.float64)
        observed = n_called > 0
        allele_frequency[observed] = allele_sum[observed] / (2.0 * n_called[observed])
        rows.append(
            pd.DataFrame(
                {
                    "variant_index": np.arange(
                        start_variant, start_variant + n_variants, dtype=np.int64
                    ),
                    "n_called": n_called,
                    "call_rate": n_called / sample_count,
                    "allele_frequency": allele_frequency,
                    "minor_allele_frequency": np.minimum(allele_frequency, 1.0 - allele_frequency),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def select_qc_pass_variants(
    summary: pd.DataFrame,
    *,
    min_call_rate: float = DGRP_MIN_CALL_RATE,
    min_minor_allele_frequency: float = DGRP_MIN_MINOR_ALLELE_FREQUENCY,
) -> pd.DataFrame:
    """Return the deterministic variant inventory retained by genotype QC."""
    required = {
        "variant_index",
        "n_called",
        "call_rate",
        "allele_frequency",
        "minor_allele_frequency",
    }
    if set(summary.columns) != required:
        raise ValueError("genotype summary columns differ from the required schema")
    if not 0.0 < min_call_rate <= 1.0:
        raise ValueError("min_call_rate must be in (0, 1]")
    if not 0.0 <= min_minor_allele_frequency <= 0.5:
        raise ValueError("min_minor_allele_frequency must be in [0, 0.5]")
    if summary["variant_index"].duplicated().any():
        raise ValueError("genotype summary contains duplicate variant indices")
    numeric = summary[
        ["n_called", "call_rate", "allele_frequency", "minor_allele_frequency"]
    ].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("genotype summary contains nonnumeric or non-finite values")
    keep = (numeric["call_rate"] >= min_call_rate) & (
        numeric["minor_allele_frequency"] > min_minor_allele_frequency
    )
    return summary.loc[keep.to_numpy()].reset_index(drop=True)


def materialize_filtered_genotypes(
    bed_path: Path,
    retained_variants: pd.DataFrame,
    output_path: Path,
    *,
    sample_count: int,
    variant_count: int,
    block_size: int = 16_384,
) -> dict[str, int]:
    """Write retained dosages to a row-major NumPy memmap in BED order."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    required = {"variant_index"}
    if not required.issubset(retained_variants.columns):
        raise ValueError("retained variant inventory is missing variant_index")
    indices = retained_variants["variant_index"].to_numpy(dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("retained variant inventory must be nonempty")
    if np.any(indices < 0) or np.any(indices >= variant_count):
        raise ValueError("retained variant indices are outside the BED range")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("retained variant indices must be strictly increasing")

    validate_bed(bed_path, sample_count=sample_count, variant_count=variant_count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.int8,
        shape=(len(indices), sample_count),
    )
    cursor = 0
    for start_variant in range(0, variant_count, block_size):
        stop_variant = min(start_variant + block_size, variant_count)
        left = np.searchsorted(indices, start_variant, side="left")
        right = np.searchsorted(indices, stop_variant, side="left")
        if left == right:
            continue
        block = decode_bed_variants(
            bed_path,
            sample_count=sample_count,
            variant_count=variant_count,
            start_variant=start_variant,
            n_variants=stop_variant - start_variant,
        )
        selected = indices[left:right] - start_variant
        matrix[cursor : cursor + len(selected)] = block[selected]
        cursor += len(selected)
    if cursor != len(indices):
        raise RuntimeError("filtered genotype materialization did not cover its inventory")
    matrix.flush()
    del matrix
    return {
        "variant_count": len(indices),
        "sample_count": sample_count,
        "matrix_bytes": len(indices) * sample_count,
    }


def build_line_folds(
    line_ids: Sequence[str],
    *,
    n_splits: int = 5,
    random_state: int = 1718,
) -> tuple[np.ndarray, ...]:
    """Create deterministic, disjoint folds over DGRP lines."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(line_ids) < n_splits:
        raise ValueError("n_splits cannot exceed the number of lines")
    if len(set(line_ids)) != len(line_ids):
        raise ValueError("line_ids must be unique")
    order = np.random.default_rng(random_state).permutation(len(line_ids))
    return tuple(np.asarray(fold, dtype=np.int64) for fold in np.array_split(order, n_splits))


def impute_fold_genotypes(
    genotypes: np.ndarray,
    train_indices: Sequence[int],
    evaluation_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Impute train and evaluation rows using medians learned from training."""
    matrix = np.asarray(genotypes)
    train = np.asarray(train_indices, dtype=np.int64)
    evaluation = np.asarray(evaluation_indices, dtype=np.int64)
    if matrix.ndim != 2:
        raise ValueError("genotypes must be two-dimensional")
    if len(train) < 1 or np.any(train < 0) or np.any(train >= matrix.shape[0]):
        raise ValueError("train_indices must contain valid rows")
    if np.any(evaluation < 0) or np.any(evaluation >= matrix.shape[0]):
        raise ValueError("evaluation_indices must contain valid rows")
    train_matrix = np.asarray(matrix[train], dtype=np.float64)
    evaluation_matrix = np.asarray(matrix[evaluation], dtype=np.float64)
    train_matrix[train_matrix < 0] = np.nan
    evaluation_matrix[evaluation_matrix < 0] = np.nan
    if np.isnan(train_matrix).all(axis=0).any():
        raise ValueError("a training-fold genotype feature has no observed values")
    medians = np.nanmedian(train_matrix, axis=0)
    train_missing = np.isnan(train_matrix)
    evaluation_missing = np.isnan(evaluation_matrix)
    if train_missing.any():
        train_matrix[train_missing] = np.broadcast_to(medians, train_matrix.shape)[train_missing]
    if evaluation_missing.any():
        evaluation_matrix[evaluation_missing] = np.broadcast_to(medians, evaluation_matrix.shape)[
            evaluation_missing
        ]
    return train_matrix, evaluation_matrix


def residualize_fold_outcomes(
    outcomes: np.ndarray,
    covariates: np.ndarray,
    train_indices: Sequence[int],
    evaluation_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Remove covariate effects using coefficients learned on training rows."""
    target = np.asarray(outcomes, dtype=np.float64)
    confounders = np.asarray(covariates, dtype=np.float64)
    train = np.asarray(train_indices, dtype=np.int64)
    evaluation = np.asarray(evaluation_indices, dtype=np.int64)
    if target.ndim != 1 or confounders.ndim != 2 or len(target) != len(confounders):
        raise ValueError(
            "outcomes and covariates must have aligned one-dimensional/two-dimensional shapes"
        )
    if len(train) < confounders.shape[1] + 1:
        raise ValueError("training rows must exceed the covariate-plus-intercept dimension")
    if np.any(train < 0) or np.any(train >= len(target)):
        raise ValueError("train_indices must contain valid rows")
    if np.any(evaluation < 0) or np.any(evaluation >= len(target)):
        raise ValueError("evaluation_indices must contain valid rows")
    if not np.isfinite(target).all() or not np.isfinite(confounders).all():
        raise ValueError("outcomes and covariates must be finite")
    design = np.column_stack((np.ones(len(target)), confounders))
    coefficients, _, _, _ = np.linalg.lstsq(design[train], target[train], rcond=None)
    return target[train] - design[train] @ coefficients, target[evaluation] - design[
        evaluation
    ] @ coefficients


def collapse_ld_redundant_ranking(
    genotypes: np.ndarray,
    ranking: Sequence[int],
    *,
    r2_threshold: float = 0.8,
) -> np.ndarray:
    """Return rank-ordered LD representatives from a complete ranking."""
    matrix = np.asarray(genotypes, dtype=np.float64)
    order = np.asarray(ranking, dtype=np.int64)
    if matrix.ndim != 2:
        raise ValueError("genotypes must be two-dimensional")
    if order.ndim != 1 or len(order) == 0:
        raise ValueError("ranking must be a nonempty one-dimensional sequence")
    if len(np.unique(order)) != len(order) or np.any(order < 0) or np.any(order >= matrix.shape[1]):
        raise ValueError("ranking must contain unique valid feature indices")
    if not 0.0 < r2_threshold <= 1.0:
        raise ValueError("r2_threshold must be in (0, 1]")
    selected = matrix[:, order]
    centered = selected - selected.mean(axis=0)
    norms = np.sqrt((centered * centered).sum(axis=0))
    if not np.all(norms > 0):
        raise ValueError("LD stability requires nonconstant ranked genotypes")
    correlation = (centered.T @ centered) / np.outer(norms, norms)
    representatives: list[int] = []
    for position in range(len(order)):
        if not any(correlation[position, prior] ** 2 >= r2_threshold for prior in representatives):
            representatives.append(position)
    return order[np.asarray(representatives, dtype=np.int64)]


def rank_features_fold_local(
    genotypes: np.ndarray,
    outcomes: np.ndarray,
    train_indices: Sequence[int],
) -> np.ndarray:
    """Rank variants using only a training fold and training-fold imputation."""
    matrix = np.asarray(genotypes)
    target = np.asarray(outcomes, dtype=np.float64)
    indices = np.asarray(train_indices, dtype=np.int64)
    if matrix.ndim != 2 or target.ndim != 1 or matrix.shape[0] != len(target):
        raise ValueError(
            "genotypes and outcomes must have aligned two-dimensional/one-dimensional shapes"
        )
    if len(indices) < 2 or np.any(indices < 0) or np.any(indices >= len(target)):
        raise ValueError("train_indices must contain at least two valid rows")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("train_indices must be unique")
    y = target[indices]
    if not np.isfinite(y).all():
        raise ValueError("training outcomes must be finite")

    x, _ = impute_fold_genotypes(matrix, indices, indices)

    centered_x = x - x.mean(axis=0)
    centered_y = y - y.mean()
    denominator = np.sqrt((centered_x * centered_x).sum(axis=0) * (centered_y * centered_y).sum())
    scores = np.zeros(matrix.shape[1], dtype=np.float64)
    valid = denominator > 0
    scores[valid] = np.abs(
        (centered_x[:, valid] * centered_y[:, None]).sum(axis=0) / denominator[valid]
    )
    return np.lexsort((np.arange(matrix.shape[1]), -scores))


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
            except BaseException:
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


def load_line_covariates(
    database_path: Path,
    *,
    gwas_analysis_id: int,
    genotype_lines: Iterable[str],
) -> pd.DataFrame:
    """Load and validate line-level covariates from the pinned DGRP database."""
    if gwas_analysis_id < 1:
        raise ValueError("gwas_analysis_id must be positive")
    expected_lines = tuple(_normalize_genotype_line(value) for value in genotype_lines)
    if not expected_lines or len(set(expected_lines)) != len(expected_lines):
        raise ValueError("genotype_lines must be nonempty and unique")

    query = """
        SELECT strain_number, covariable_name, covariable_value
        FROM gwas_covariable_value
        WHERE associated_gwasanalysis_id = ?
        ORDER BY strain_number, covariable_name
    """
    with sqlite3.connect(database_path) as connection:
        frame = pd.read_sql_query(query, connection, params=(gwas_analysis_id,))
    required = {"strain_number", "covariable_name", "covariable_value"}
    if set(frame.columns) != required:
        raise RuntimeError("DGRP covariate table has an unexpected schema")
    if frame.empty:
        raise RuntimeError(f"No DGRP covariates found for GWAS analysis {gwas_analysis_id}")
    frame = frame[frame["strain_number"].isin(expected_lines)].copy()
    if frame.empty:
        raise RuntimeError("DGRP covariates do not cover the requested genotype lines")
    if frame[["strain_number", "covariable_name"]].duplicated().any():
        raise RuntimeError("DGRP covariates contain duplicate line-variable pairs")
    if set(frame["strain_number"]) != set(expected_lines):
        raise RuntimeError("DGRP covariates do not cover the requested genotype lines")
    values = pd.to_numeric(frame["covariable_value"], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("DGRP covariates contain nonnumeric or non-finite values")
    result = frame.assign(covariable_value=values.astype(np.float64)).pivot(
        index="strain_number",
        columns="covariable_name",
        values="covariable_value",
    )
    if result.isna().any().any():
        raise RuntimeError("DGRP covariates contain incomplete line coverage")
    return result.reindex(expected_lines).reset_index()


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
    repo_root = Path(__file__).resolve().parents[3]
    source_files = (
        Path(__file__).resolve(),
        repo_root / "pyproject.toml",
        repo_root / "uv.lock",
    )
    receipt = {
        "analysis": "dgrp_phenotypes",
        "schema_version": 3,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(repo_root),
        "git_dirty": _git_dirty(repo_root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_sha256": {str(path.relative_to(repo_root)): sha256(path) for path in source_files},
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "openpyxl", "pandas")
        },
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
