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
import time
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import combinations, product
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.linear_model import Ridge

from citrees import (
    ConditionalInferenceForestRegressor,
    ConditionalInferenceTreeRegressor,
)

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
COVARIATE_ARCHIVE_URL = (
    "https://zenodo.org/api/records/5582846/files/Phenosnip.sqlite.tar.gz/content"
)
COVARIATE_ARCHIVE_FILENAME = "Phenosnip.sqlite.tar.gz"
COVARIATE_ARCHIVE_BYTES = 1_321_171_772
COVARIATE_ARCHIVE_SHA256 = "2a090b4e1168361123d8214c200ba1e52c8df78750aa5e915388cea5a0c94e37"
COVARIATE_DATABASE_MEMBER = "input/Phenosnip.201612.sqlite"
COVARIATE_DATABASE_FILENAME = "Phenosnip.201612.sqlite"
COVARIATE_DATABASE_BYTES = 7_216_261_120
COVARIATE_DATABASE_SHA256 = "fa7d7285213c1a4264a9ed5b3b791e8012b6ed817e19ccb1ecff5718597398ac"
DGRP_GWAS_ANALYSIS_ID = 7
DGRP_COVARIATES = tuple(f"cov{index}" for index in range(1, 18))
DGRP_PRIMARY_TRAIT = "EDD"
DGRP_SECONDARY_TRAITS = ("DI", "SI", "HP", "ESD", "FS", "AI")
DGRP_METHODS = ("marginal", "ridge", "cit", "cif")
DGRP_PREDICTION_K = (5, 10, 25, 50, 100)
DGRP_STABILITY_K = (5, 10, 25)
DGRP_LD_R2_THRESHOLD = 0.8
DGRP_RDC_N_PROJECTIONS = 10
DGRP_RIDGE_ALPHA = 1.0
DGRP_SCREEN_BLOCK_SIZE = 16_384
DGRP_HOLM_ALPHA = 0.05
DGRP_HOLM_FAMILY = "dgrp_secondary_trait_cif_top10_prediction"
DGRP_CANDIDATE_SCREEN = "covariate_adjusted_marginal_absolute_correlation"
DGRP_SCREEN_TIE_BREAKER = "variant_index_ascending"
DGRP_RANKING_TIE_BREAKER = "candidate_screen_rank"
DGRP_INFERENCE_TEST = "corrected_repeated_kfold_t"
DGRP_RANKING_METHODS = {
    "marginal": "candidate_screen_score",
    "ridge": "absolute_ridge_coefficient",
    "cit": "cit_feature_importance",
    "cif": "cif_feature_importance",
}
DGRP_PROFILE_INFERENCE_STATUS = {
    "smoke": "pipeline_validation",
    "quick": "exploratory",
    "full": "manuscript_inference",
}
DGRP_INFERENCE_UNIT = "outer_fold_mean_squared_error_difference"
DGRP_INFERENCE_ESTIMAND = "cif_minus_marginal_top10_ridge_mean_squared_error"
DGRP_DOWNSTREAM_MODEL = "ridge_alpha_1"
DGRP_PREDICTION_SCALE = "training_fold_residual_outcome_units"
DGRP_NUMERIC_RTOL = 1e-10
DGRP_NUMERIC_ATOL = 1e-12
DGRP_BASELINE_ATOL = 1e-10

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "dgrp"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "dgrp"
SPECIFICATION_PATH = Path(__file__).with_name("dgrp-specification.json")
DERIVED_GENOTYPE_DIRECTORY = "derived-genotypes"
DERIVED_GENOTYPE_MATRIX = "filtered-genotypes.npy"
DERIVED_VARIANT_INVENTORY = "retained-variants.parquet"
DERIVED_GENOTYPE_RECEIPT = "receipt.json"

Profile = Literal["smoke", "quick", "full"]
type IndexSequence = Sequence[int] | np.ndarray
type FloatSequence = Sequence[float] | np.ndarray


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


@dataclass(frozen=True)
class DGRPSettings:
    """Fixed workload settings for one replication profile."""

    profile: Profile
    traits: tuple[str, ...]
    n_repeats: int
    n_splits: int
    candidate_count: int
    n_trees: int
    prediction_k: tuple[int, ...]
    stability_k: tuple[int, ...]


@dataclass(frozen=True)
class CorrectedRepeatedKFoldResult:
    """Corrected repeated k-fold comparison statistics."""

    n_fold_differences: int
    mean_difference: float
    sample_variance: float
    mean_test_train_ratio: float
    variance_correction: float
    standard_error: float
    degrees_of_freedom: int
    test_statistic: float
    lower_tail_p_value: float


DGRP_RESULT_SCHEMAS: dict[str, tuple[str, ...]] = {
    "line_outcomes": (
        "strain_number",
        "trait",
        "source_column",
        "unit",
        "n_individuals",
        "outcome",
    ),
    "trait_summary": (
        "trait",
        "source_column",
        "unit",
        "n_lines",
        "n_individuals",
    ),
    "fold_assignments": (
        "profile",
        "trait",
        "repeat",
        "fold",
        "split_seed",
        "line_id",
        "role",
    ),
    "screening_rankings": (
        "profile",
        "trait",
        "repeat",
        "fold",
        "split_seed",
        "model_seed",
        "method",
        "candidate_screen",
        "candidate_count",
        "screen_tie_breaker",
        "ranking_method",
        "ranking_tie_breaker",
        "rank",
        "candidate_rank",
        "variant_index",
        "chromosome",
        "variant_id",
        "position",
        "allele_a",
        "allele_b",
        "score",
    ),
    "predictions": (
        "profile",
        "trait",
        "repeat",
        "fold",
        "split_seed",
        "model_seed",
        "method",
        "candidate_screen",
        "candidate_count",
        "screen_tie_breaker",
        "ranking_method",
        "ranking_tie_breaker",
        "k",
        "line_id",
        "prediction_scale",
        "observed",
        "prediction",
        "squared_error",
    ),
    "fold_metrics": (
        "profile",
        "trait",
        "repeat",
        "fold",
        "split_seed",
        "model_seed",
        "method",
        "candidate_screen",
        "candidate_count",
        "screen_tie_breaker",
        "ranking_method",
        "ranking_tie_breaker",
        "k",
        "downstream_model",
        "n_training",
        "n_evaluation",
        "training_baseline",
        "mean_squared_error",
        "predictive_r2",
    ),
    "ld_groups": (
        "profile",
        "trait",
        "repeat",
        "fold",
        "split_seed",
        "model_seed",
        "method",
        "rank",
        "variant_index",
        "chromosome",
        "variant_id",
        "ld_group_id",
        "representative_variant_index",
        "representative_variant_id",
        "representative_rank",
        "r2_to_representative",
        "is_representative",
    ),
    "stability_summary": (
        "profile",
        "trait",
        "repeat",
        "method",
        "fold_a",
        "fold_b",
        "k",
        "match_type",
        "matches",
        "union_size",
        "jaccard",
    ),
    "stability_matches": (
        "profile",
        "trait",
        "repeat",
        "method",
        "fold_a",
        "fold_b",
        "k",
        "match_type",
        "rank_a",
        "variant_index_a",
        "rank_b",
        "variant_index_b",
        "chromosome",
        "r2",
    ),
    "primary_inference": (
        "profile",
        "trait",
        "hypothesis_id",
        "contrast",
        "method_a",
        "method_b",
        "k",
        "metric",
        "alternative",
        "candidate_screen",
        "candidate_count",
        "screen_tie_breaker",
        "method_a_ranking",
        "method_b_ranking",
        "ranking_tie_breaker",
        "downstream_model",
        "test",
        "unit_of_analysis",
        "estimand",
        "inference_status",
        "n_repeats",
        "n_splits",
        "n_fold_differences",
        "n_lines",
        "mean_difference",
        "sample_variance",
        "mean_test_train_ratio",
        "variance_correction",
        "standard_error",
        "degrees_of_freedom",
        "test_statistic",
        "raw_p_value",
    ),
    "secondary_holm": (
        "profile",
        "family",
        "family_size",
        "hypothesis_id",
        "trait",
        "contrast",
        "method_a",
        "method_b",
        "k",
        "metric",
        "alternative",
        "candidate_screen",
        "candidate_count",
        "screen_tie_breaker",
        "method_a_ranking",
        "method_b_ranking",
        "ranking_tie_breaker",
        "downstream_model",
        "test",
        "unit_of_analysis",
        "estimand",
        "inference_status",
        "test_defined",
        "n_repeats",
        "n_splits",
        "n_fold_differences",
        "n_lines",
        "mean_difference",
        "sample_variance",
        "mean_test_train_ratio",
        "variance_correction",
        "standard_error",
        "degrees_of_freedom",
        "test_statistic",
        "raw_p_value",
        "holm_input_p_value",
        "holm_adjusted_p_value",
        "holm_reject_0_05",
    ),
}


TRAITS = (
    TraitSpec("DI", "DiastolicIntervals_Median", "seconds"),
    TraitSpec("SI", "SystolicIntervals_Median", "seconds"),
    TraitSpec("HP", "Heartperiod_Median", "seconds"),
    TraitSpec("EDD", "DiastolicMeanDiameter", "micrometers"),
    TraitSpec("ESD", "SystolicMeanDiameter", "micrometers"),
    TraitSpec("FS", "FractionalShortening", "dimensionless fraction"),
    TraitSpec("AI", "Heartperiod_StdDevOnMedian", "dimensionless SD/median ratio"),
)


def _settings(profile: Profile) -> DGRPSettings:
    """Return the fixed DGRP workload for a replication profile."""
    if profile == "smoke":
        return DGRPSettings(
            profile=profile,
            traits=(DGRP_PRIMARY_TRAIT,),
            n_repeats=2,
            n_splits=2,
            candidate_count=25,
            n_trees=2,
            prediction_k=(5, 10, 25),
            stability_k=DGRP_STABILITY_K,
        )
    if profile == "quick":
        return DGRPSettings(
            profile=profile,
            traits=(DGRP_PRIMARY_TRAIT, *DGRP_SECONDARY_TRAITS),
            n_repeats=3,
            n_splits=3,
            candidate_count=100,
            n_trees=5,
            prediction_k=DGRP_PREDICTION_K,
            stability_k=DGRP_STABILITY_K,
        )
    if profile == "full":
        return DGRPSettings(
            profile=profile,
            traits=(DGRP_PRIMARY_TRAIT, *DGRP_SECONDARY_TRAITS),
            n_repeats=10,
            n_splits=5,
            candidate_count=100,
            n_trees=10,
            prediction_k=DGRP_PREDICTION_K,
            stability_k=DGRP_STABILITY_K,
        )
    raise ValueError(f"unknown DGRP profile: {profile!r}")


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


def _verify_sized_sha256(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    source_name: str,
) -> None:
    """Require a pinned file size and SHA-256."""
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{source_name} is not a regular file: {path}")
    observed_bytes = path.stat().st_size
    if observed_bytes != expected_bytes:
        raise RuntimeError(
            f"{source_name} size mismatch for {path}: "
            f"expected {expected_bytes}, got {observed_bytes}"
        )
    observed_sha256 = sha256(path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"{source_name} SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, got {observed_sha256}"
        )


def verify_covariate_archive(path: Path) -> None:
    """Require the pinned Phenosnip SQLite archive."""
    _verify_sized_sha256(
        path,
        expected_bytes=COVARIATE_ARCHIVE_BYTES,
        expected_sha256=COVARIATE_ARCHIVE_SHA256,
        source_name="DGRP covariate archive",
    )


def verify_covariate_database(path: Path) -> None:
    """Require the pinned extracted Phenosnip SQLite database."""
    _verify_sized_sha256(
        path,
        expected_bytes=COVARIATE_DATABASE_BYTES,
        expected_sha256=COVARIATE_DATABASE_SHA256,
        source_name="DGRP covariate database",
    )


def acquire_covariate_archive(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download the pinned Phenosnip archive when it is not already present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / COVARIATE_ARCHIVE_FILENAME
    if destination.exists():
        verify_covariate_archive(destination)
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        raise RuntimeError(f"Partial DGRP covariate download already exists: {partial}")
    try:
        with (
            urllib.request.urlopen(COVARIATE_ARCHIVE_URL, timeout=60) as response,
            partial.open("xb") as stream,
        ):
            shutil.copyfileobj(response, stream)
        verify_covariate_archive(partial)
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return destination


def extract_covariate_database(archive_path: Path, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Extract the single pinned SQLite member to the shared input directory."""
    verify_covariate_archive(archive_path)
    input_dir = data_dir / "input"
    if input_dir.is_symlink():
        raise RuntimeError(f"DGRP input directory must not be a symlink: {input_dir}")
    input_dir.mkdir(parents=True, exist_ok=True)
    destination = input_dir / COVARIATE_DATABASE_FILENAME
    if destination.exists():
        verify_covariate_database(destination)
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        raise RuntimeError(f"Partial DGRP covariate extraction already exists: {partial}")
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) != 1:
                raise RuntimeError("DGRP covariate archive must contain exactly one member")
            member = members[0]
            member_path = PurePosixPath(member.name)
            if (
                member.name != COVARIATE_DATABASE_MEMBER
                or member_path.is_absolute()
                or ".." in member_path.parts
                or "\\" in member.name
                or not member.isfile()
                or member.size != COVARIATE_DATABASE_BYTES
            ):
                raise RuntimeError("DGRP covariate archive member differs from the pinned database")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("Unable to read the DGRP covariate database member")
            with source, partial.open("xb") as stream:
                shutil.copyfileobj(source, stream)
        verify_covariate_database(partial)
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


def load_bim_metadata(path: Path, variant_indices: IndexSequence) -> pd.DataFrame:
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


def _derived_genotype_paths(data_dir: Path) -> tuple[Path, Path, Path]:
    root = data_dir / DERIVED_GENOTYPE_DIRECTORY
    return (
        root / DERIVED_GENOTYPE_MATRIX,
        root / DERIVED_VARIANT_INVENTORY,
        root / DERIVED_GENOTYPE_RECEIPT,
    )


def validate_derived_genotypes(
    data_dir: Path,
    input_dir: Path,
) -> tuple[Path, Path, dict[str, object]]:
    """Validate and return the cached filtered genotype matrix and inventory."""
    matrix_path, variants_path, receipt_path = _derived_genotype_paths(data_dir)
    root = receipt_path.parent
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"DGRP derived genotype directory is invalid: {root}")
    expected_names = {
        DERIVED_GENOTYPE_MATRIX,
        DERIVED_VARIANT_INVENTORY,
        DERIVED_GENOTYPE_RECEIPT,
    }
    observed_names = {path.name for path in root.iterdir() if path.is_file()}
    if observed_names != expected_names:
        raise RuntimeError(
            "DGRP derived genotype inventory differs: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"extra={sorted(observed_names - expected_names)}"
        )
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    if not isinstance(receipt, dict) or receipt.get("schema") != "citrees-jss-dgrp-genotypes-v1":
        raise RuntimeError("DGRP derived genotype receipt has an invalid schema")
    paths = _input_paths(input_dir)
    expected_sources = {name: sha256(path) for name, path in paths.items()}
    if receipt.get("source_sha256") != expected_sources:
        raise RuntimeError("DGRP derived genotype source hashes differ")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        DERIVED_GENOTYPE_MATRIX,
        DERIVED_VARIANT_INVENTORY,
    }:
        raise RuntimeError("DGRP derived genotype artifact metadata differs")
    for path in (matrix_path, variants_path):
        metadata = artifacts[path.name]
        if (
            not isinstance(metadata, dict)
            or metadata.get("bytes") != path.stat().st_size
            or metadata.get("sha256") != sha256(path)
        ):
            raise RuntimeError(f"DGRP derived genotype artifact differs: {path.name}")
    variants = pd.read_parquet(variants_path)
    expected_columns = (
        "variant_index",
        "n_called",
        "call_rate",
        "allele_frequency",
        "minor_allele_frequency",
        "chromosome",
        "variant_id",
        "position",
        "allele_a",
        "allele_b",
    )
    if tuple(variants.columns) != expected_columns:
        raise RuntimeError("DGRP retained variant inventory has an invalid schema")
    if variants.empty or variants["variant_index"].duplicated().any():
        raise RuntimeError("DGRP retained variant inventory is empty or duplicated")
    matrix = np.load(matrix_path, mmap_mode="r")
    expected_shape = (len(variants), EXPECTED_GENOTYPE_SAMPLES)
    if matrix.dtype != np.int8 or matrix.shape != expected_shape:
        raise RuntimeError(
            f"DGRP derived genotype matrix differs: expected int8 {expected_shape}, "
            f"got {matrix.dtype} {matrix.shape}"
        )
    del matrix
    return matrix_path, variants_path, receipt


def prepare_derived_genotypes(
    data_dir: Path,
    input_dir: Path,
) -> tuple[Path, Path, dict[str, object]]:
    """Build or validate the deterministic genotype QC cache."""
    matrix_path, variants_path, receipt_path = _derived_genotype_paths(data_dir)
    root = receipt_path.parent
    if root.exists():
        return validate_derived_genotypes(data_dir, input_dir)

    inventory = validate_genotype_files(input_dir)
    paths = _input_paths(input_dir)
    started = time.perf_counter()
    summary = summarize_bed_variants(
        paths["input/dgrp2.bed"],
        sample_count=len(inventory.genotype_lines),
        variant_count=inventory.variant_count,
    )
    retained = select_qc_pass_variants(summary)
    del summary
    metadata = load_bim_metadata(
        paths["input/dgrp2.bim"],
        retained["variant_index"].to_numpy(dtype=np.int64),
    )
    if not np.array_equal(
        retained["variant_index"].to_numpy(dtype=np.int64),
        metadata["variant_index"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("DGRP retained variant metadata order differs from QC order")
    retained = pd.concat(
        (
            retained.reset_index(drop=True),
            metadata.drop(columns="variant_index").reset_index(drop=True),
        ),
        axis=1,
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dgrp-derived-", dir=data_dir) as temporary:
        staging = Path(temporary) / DERIVED_GENOTYPE_DIRECTORY
        staging.mkdir()
        staged_matrix = staging / DERIVED_GENOTYPE_MATRIX
        staged_variants = staging / DERIVED_VARIANT_INVENTORY
        materialize_filtered_genotypes(
            paths["input/dgrp2.bed"],
            retained,
            staged_matrix,
            sample_count=len(inventory.genotype_lines),
            variant_count=inventory.variant_count,
        )
        retained.to_parquet(staged_variants, index=False)
        source_hashes = {name: sha256(path) for name, path in paths.items()}
        receipt = {
            "schema": "citrees-jss-dgrp-genotypes-v1",
            "call_rate_threshold": DGRP_MIN_CALL_RATE,
            "minor_allele_frequency_threshold": DGRP_MIN_MINOR_ALLELE_FREQUENCY,
            "minor_allele_frequency_operator": "strict_greater_than",
            "sample_count": len(inventory.genotype_lines),
            "source_variant_count": inventory.variant_count,
            "retained_variant_count": len(retained),
            "matrix_orientation": "variants_by_fam_samples",
            "elapsed_seconds": time.perf_counter() - started,
            "source_sha256": source_hashes,
            "artifacts": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in (staged_matrix, staged_variants)
            },
        }
        (staging / DERIVED_GENOTYPE_RECEIPT).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        staging.rename(root)
    return validate_derived_genotypes(data_dir, input_dir)


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
    train_indices: IndexSequence,
    evaluation_indices: IndexSequence,
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
    train_indices: IndexSequence,
    evaluation_indices: IndexSequence,
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
    return (
        target[train] - design[train] @ coefficients,
        target[evaluation] - design[evaluation] @ coefficients,
    )


def residualize_fold_genotypes(
    genotypes: np.ndarray,
    covariates: np.ndarray,
    train_indices: IndexSequence,
    evaluation_indices: IndexSequence,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove covariate effects from genotypes using training-fitted coefficients."""
    matrix = np.asarray(genotypes, dtype=np.float64)
    confounders = np.asarray(covariates, dtype=np.float64)
    train = np.asarray(train_indices, dtype=np.int64)
    evaluation = np.asarray(evaluation_indices, dtype=np.int64)
    if matrix.ndim != 2 or confounders.ndim != 2 or len(matrix) != len(confounders):
        raise ValueError("genotypes and covariates must be aligned two-dimensional arrays")
    if len(train) < confounders.shape[1] + 1:
        raise ValueError("training rows must exceed the covariate-plus-intercept dimension")
    if np.any(train < 0) or np.any(train >= len(matrix)):
        raise ValueError("train_indices must contain valid rows")
    if np.any(evaluation < 0) or np.any(evaluation >= len(matrix)):
        raise ValueError("evaluation_indices must contain valid rows")
    if not np.isfinite(matrix).all() or not np.isfinite(confounders).all():
        raise ValueError("genotypes and covariates must be finite")
    design = np.column_stack((np.ones(len(matrix)), confounders))
    coefficients, _, _, _ = np.linalg.lstsq(design[train], matrix[train], rcond=None)
    return (
        matrix[train] - design[train] @ coefficients,
        matrix[evaluation] - design[evaluation] @ coefficients,
    )


def collapse_ld_redundant_ranking(
    genotypes: np.ndarray,
    ranking: IndexSequence,
    chromosomes: Sequence[str],
    *,
    r2_threshold: float = 0.8,
) -> pd.DataFrame:
    """Assign ranked features to chromosome-scoped LD groups."""
    matrix = np.asarray(genotypes, dtype=np.float64)
    order = np.asarray(ranking, dtype=np.int64)
    chromosome = np.asarray(chromosomes, dtype=str)
    if matrix.ndim != 2:
        raise ValueError("genotypes must be two-dimensional")
    if order.ndim != 1 or len(order) == 0:
        raise ValueError("ranking must be a nonempty one-dimensional sequence")
    if len(np.unique(order)) != len(order) or np.any(order < 0) or np.any(order >= matrix.shape[1]):
        raise ValueError("ranking must contain unique valid feature indices")
    if chromosome.ndim != 1 or len(chromosome) != matrix.shape[1]:
        raise ValueError("chromosomes must contain one label per genotype feature")
    if np.any(np.char.str_len(chromosome) == 0):
        raise ValueError("chromosome labels must be nonempty")
    if not 0.0 < r2_threshold <= 1.0:
        raise ValueError("r2_threshold must be in (0, 1]")
    selected = matrix[:, order]
    centered = selected - selected.mean(axis=0)
    norms = np.sqrt((centered * centered).sum(axis=0))
    correlation = np.zeros((len(order), len(order)), dtype=np.float64)
    valid = norms > 0
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices):
        valid_centered = centered[:, valid_indices]
        valid_norms = norms[valid_indices]
        correlation[np.ix_(valid_indices, valid_indices)] = (
            valid_centered.T @ valid_centered
        ) / np.outer(valid_norms, valid_norms)
    np.fill_diagonal(correlation, 1.0)
    representative_positions: list[int] = []
    group_ids = np.empty(len(order), dtype=np.int64)
    representative_feature_indices = np.empty(len(order), dtype=np.int64)
    representative_ranks = np.empty(len(order), dtype=np.int64)
    r2_values = np.empty(len(order), dtype=np.float64)
    for position in range(len(order)):
        matched_position: int | None = None
        for prior in representative_positions:
            if (
                chromosome[order[position]] == chromosome[order[prior]]
                and correlation[position, prior] ** 2 >= r2_threshold
            ):
                matched_position = prior
                break
        if matched_position is None:
            matched_position = position
            representative_positions.append(position)
        group_id = representative_positions.index(matched_position) + 1
        group_ids[position] = group_id
        representative_feature_indices[position] = order[matched_position]
        representative_ranks[position] = matched_position + 1
        r2_values[position] = correlation[position, matched_position] ** 2
    return pd.DataFrame(
        {
            "feature_index": order,
            "rank": np.arange(1, len(order) + 1, dtype=np.int64),
            "chromosome": chromosome[order],
            "ld_group_id": group_ids,
            "representative_feature_index": representative_feature_indices,
            "representative_rank": representative_ranks,
            "r2_to_representative": r2_values,
            "is_representative": order == representative_feature_indices,
        }
    )


def rank_features_fold_local(
    genotypes: np.ndarray,
    outcomes: np.ndarray,
    covariates: np.ndarray,
    train_indices: IndexSequence,
) -> np.ndarray:
    """Rank covariate-adjusted variants using one training fold."""
    matrix = np.asarray(genotypes)
    target = np.asarray(outcomes, dtype=np.float64)
    confounders = np.asarray(covariates, dtype=np.float64)
    indices = np.asarray(train_indices, dtype=np.int64)
    if (
        matrix.ndim != 2
        or target.ndim != 1
        or confounders.ndim != 2
        or matrix.shape[0] != len(target)
        or len(confounders) != len(target)
    ):
        raise ValueError("genotypes, outcomes, and covariates must have aligned shapes")
    if (
        len(indices) < confounders.shape[1] + 1
        or np.any(indices < 0)
        or np.any(indices >= len(target))
    ):
        raise ValueError("train_indices must exceed the covariate dimension and contain valid rows")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("train_indices must be unique")
    x_imputed, _ = impute_fold_genotypes(matrix, indices, indices)
    x, _ = residualize_fold_genotypes(
        x_imputed,
        confounders[indices],
        np.arange(len(indices)),
        np.array([], dtype=np.int64),
    )
    y, _ = residualize_fold_outcomes(
        target,
        confounders,
        indices,
        np.array([], dtype=np.int64),
    )
    centered_x = x - x.mean(axis=0)
    centered_y = y - y.mean()
    denominator = np.sqrt((centered_x * centered_x).sum(axis=0) * (centered_y * centered_y).sum())
    scores = np.zeros(matrix.shape[1], dtype=np.float64)
    valid = denominator > 0
    scores[valid] = np.abs(
        (centered_x[:, valid] * centered_y[:, None]).sum(axis=0) / denominator[valid]
    )
    return np.lexsort((np.arange(matrix.shape[1]), -scores))


def screen_fold_candidates(
    genotypes: np.ndarray,
    variant_indices: IndexSequence,
    sample_indices: IndexSequence,
    outcomes: np.ndarray,
    covariates: np.ndarray,
    train_indices: IndexSequence,
    *,
    candidate_count: int,
    block_size: int = DGRP_SCREEN_BLOCK_SIZE,
) -> pd.DataFrame:
    """Screen all retained variants by fold-local partial absolute correlation."""
    matrix = np.asarray(genotypes)
    variants = np.asarray(variant_indices, dtype=np.int64)
    samples = np.asarray(sample_indices, dtype=np.int64)
    target = np.asarray(outcomes, dtype=np.float64)
    confounders = np.asarray(covariates, dtype=np.float64)
    train = np.asarray(train_indices, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != len(variants):
        raise ValueError("genotypes and variant_indices must be aligned")
    if (
        samples.ndim != 1
        or len(samples) != len(target)
        or len(confounders) != len(target)
        or confounders.ndim != 2
    ):
        raise ValueError("sample_indices, outcomes, and covariates must be aligned")
    if np.any(samples < 0) or np.any(samples >= matrix.shape[1]):
        raise ValueError("sample_indices contain rows outside the genotype matrix")
    if len(train) < confounders.shape[1] + 1:
        raise ValueError("training rows must exceed the covariate-plus-intercept dimension")
    if candidate_count < 1 or candidate_count > len(variants):
        raise ValueError("candidate_count must be within the retained variant count")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    evaluation = np.array([], dtype=np.int64)
    adjusted_y, _ = residualize_fold_outcomes(target, confounders, train, evaluation)
    centered_y = adjusted_y - adjusted_y.mean()
    y_norm = float(np.linalg.norm(centered_y))
    if y_norm == 0.0:
        raise ValueError("training-fold adjusted outcome is constant")
    design = np.column_stack((np.ones(len(train)), confounders[train]))
    left_vectors, singular_values, _ = np.linalg.svd(design, full_matrices=False)
    tolerance = singular_values.max(initial=0.0) * max(design.shape) * np.finfo(float).eps
    basis = left_vectors[:, singular_values > tolerance]

    scores = np.zeros(len(variants), dtype=np.float64)
    for start in range(0, len(variants), block_size):
        stop = min(start + block_size, len(variants))
        block = np.asarray(matrix[start:stop, samples], dtype=np.float64).T
        block[block < 0] = np.nan
        train_block = block[train]
        if np.isnan(train_block).all(axis=0).any():
            raise ValueError("a training-fold genotype feature has no observed values")
        medians = np.nanmedian(train_block, axis=0)
        missing = np.isnan(train_block)
        if missing.any():
            train_block[missing] = np.broadcast_to(medians, train_block.shape)[missing]
        adjusted_x = train_block - basis @ (basis.T @ train_block)
        adjusted_x -= adjusted_x.mean(axis=0)
        x_norm = np.linalg.norm(adjusted_x, axis=0)
        valid = x_norm > 0
        block_scores = np.zeros(stop - start, dtype=np.float64)
        block_scores[valid] = np.abs(adjusted_x[:, valid].T @ centered_y / (x_norm[valid] * y_norm))
        scores[start:stop] = block_scores
    order = np.lexsort((variants, -scores))[:candidate_count]
    return pd.DataFrame(
        {
            "variant_row": order.astype(np.int64),
            "variant_index": variants[order],
            "candidate_rank": np.arange(1, candidate_count + 1, dtype=np.int64),
            "screen_score": scores[order],
        }
    )


def prepare_fold_candidate_matrices(
    genotypes: np.ndarray,
    candidate_rows: IndexSequence,
    sample_indices: IndexSequence,
    outcomes: np.ndarray,
    covariates: np.ndarray,
    train_indices: IndexSequence,
    evaluation_indices: IndexSequence,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return adjusted fold arrays and imputed training genotypes for LD."""
    matrix = np.asarray(genotypes)
    candidates = np.asarray(candidate_rows, dtype=np.int64)
    samples = np.asarray(sample_indices, dtype=np.int64)
    train = np.asarray(train_indices, dtype=np.int64)
    evaluation = np.asarray(evaluation_indices, dtype=np.int64)
    raw = np.asarray(matrix[candidates][:, samples], dtype=np.float64).T
    train_raw, evaluation_raw = impute_fold_genotypes(raw, train, evaluation)
    design = np.column_stack((np.ones(len(outcomes)), np.asarray(covariates, dtype=np.float64)))
    coefficients, _, _, _ = np.linalg.lstsq(design[train], train_raw, rcond=None)
    train_x = train_raw - design[train] @ coefficients
    evaluation_x = evaluation_raw - design[evaluation] @ coefficients
    train_y, evaluation_y = residualize_fold_outcomes(
        outcomes,
        covariates,
        train,
        evaluation,
    )
    return train_x, evaluation_x, train_y, evaluation_y, train_raw


def _complete_ranking(scores: np.ndarray, candidate_ranks: np.ndarray) -> np.ndarray:
    """Return a deterministic total order with screening rank as the tie break."""
    values = np.asarray(scores, dtype=np.float64)
    fallback = np.asarray(candidate_ranks, dtype=np.int64)
    if values.ndim != 1 or fallback.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("ranking scores and candidate ranks must be aligned and finite")
    if sorted(fallback.tolist()) != list(range(1, len(fallback) + 1)):
        raise ValueError("candidate ranks must be a complete one-based ordering")
    return np.lexsort((fallback, -values))


def fit_fold_rankings(
    train_x: np.ndarray,
    train_y: np.ndarray,
    candidate_ranks: IndexSequence,
    marginal_scores: FloatSequence,
    *,
    n_trees: int,
    random_state: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fit the four prespecified ranking methods on one adjusted training fold."""
    matrix = np.asarray(train_x, dtype=np.float64)
    target = np.asarray(train_y, dtype=np.float64)
    fallback = np.asarray(candidate_ranks, dtype=np.int64)
    marginal = np.asarray(marginal_scores, dtype=np.float64)
    if matrix.ndim != 2 or target.ndim != 1 or len(matrix) != len(target):
        raise ValueError("training genotypes and outcomes must be aligned")
    if matrix.shape[1] != len(fallback) or marginal.shape != fallback.shape:
        raise ValueError("candidate ranks and marginal scores must match the feature count")
    if n_trees < 1:
        raise ValueError("n_trees must be positive")

    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales == 0.0] = 1.0
    standardized = (matrix - means) / scales
    ridge = Ridge(alpha=DGRP_RIDGE_ALPHA)
    ridge.fit(standardized, target)
    ridge_scores = np.abs(np.asarray(ridge.coef_, dtype=np.float64))

    tree = ConditionalInferenceTreeRegressor(
        selector="rdc",
        rdc_n_projections=DGRP_RDC_N_PROJECTIONS,
        random_state=random_state,
        verbose=0,
    )
    tree.fit(matrix, target)
    tree_scores = np.asarray(tree.feature_importances_, dtype=np.float64)

    forest = ConditionalInferenceForestRegressor(
        n_estimators=n_trees,
        selector="rdc",
        rdc_n_projections=DGRP_RDC_N_PROJECTIONS,
        n_jobs=1,
        random_state=random_state,
        verbose=0,
    )
    forest.fit(matrix, target)
    forest_scores = np.asarray(forest.feature_importances_, dtype=np.float64)

    score_by_method = {
        "marginal": marginal,
        "ridge": ridge_scores,
        "cit": tree_scores,
        "cif": forest_scores,
    }
    return {
        method: (_complete_ranking(scores, fallback), scores)
        for method, scores in score_by_method.items()
    }


def ridge_predictions(
    train_x: np.ndarray,
    evaluation_x: np.ndarray,
    train_y: np.ndarray,
    ranking: IndexSequence,
    k: int,
) -> np.ndarray:
    """Fit the fixed ridge predictor on the first k ranked adjusted genotypes."""
    order = np.asarray(ranking, dtype=np.int64)
    if k < 1 or k > len(order):
        raise ValueError("k must be within the ranking length")
    selected = order[:k]
    means = train_x[:, selected].mean(axis=0)
    scales = train_x[:, selected].std(axis=0)
    scales[scales == 0.0] = 1.0
    train_standardized = (train_x[:, selected] - means) / scales
    evaluation_standardized = (evaluation_x[:, selected] - means) / scales
    model = Ridge(alpha=DGRP_RIDGE_ALPHA)
    model.fit(train_standardized, train_y)
    return np.asarray(model.predict(evaluation_standardized), dtype=np.float64)


def _pairwise_complete_r2(left: np.ndarray, right: np.ndarray) -> float:
    observed = (left >= 0) & (right >= 0)
    if observed.sum() < 3:
        return 0.0
    x = np.asarray(left[observed], dtype=np.float64)
    y = np.asarray(right[observed], dtype=np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator == 0.0:
        return 0.0
    correlation = float(x @ y / denominator)
    return correlation * correlation


def match_ranked_variants(
    genotypes: np.ndarray,
    variants: pd.DataFrame,
    left_rows: IndexSequence,
    right_rows: IndexSequence,
    sample_indices: IndexSequence,
    *,
    r2_threshold: float = DGRP_LD_R2_THRESHOLD,
) -> list[tuple[int, int, float]]:
    """Return a deterministic maximum-cardinality chromosome-aware LD matching."""
    left = np.asarray(left_rows, dtype=np.int64)
    right = np.asarray(right_rows, dtype=np.int64)
    samples = np.asarray(sample_indices, dtype=np.int64)
    if len(np.unique(left)) != len(left) or len(np.unique(right)) != len(right):
        raise ValueError("ranked LD sets must contain unique variant rows")
    if not 0.0 < r2_threshold <= 1.0:
        raise ValueError("r2_threshold must be in (0, 1]")
    chromosomes = variants["chromosome"].astype(str).to_numpy()
    variant_ids = variants["variant_index"].to_numpy(dtype=np.int64)
    neighbors: list[list[tuple[int, float]]] = []
    for left_position, left_row in enumerate(left):
        candidates: list[tuple[int, float]] = []
        for right_position, right_row in enumerate(right):
            if chromosomes[left_row] != chromosomes[right_row]:
                continue
            if variant_ids[left_row] == variant_ids[right_row]:
                r2 = 1.0
            else:
                r2 = _pairwise_complete_r2(
                    np.asarray(genotypes[left_row, samples]),
                    np.asarray(genotypes[right_row, samples]),
                )
            if r2 >= r2_threshold:
                candidates.append((right_position, r2))
        candidates.sort(
            key=lambda item: (
                variant_ids[left[left_position]] != variant_ids[right[item[0]]],
                -item[1],
                item[0],
            )
        )
        neighbors.append(candidates)

    right_match: dict[int, int] = {}

    def augment(left_position: int, visited: set[int]) -> bool:
        for right_position, _ in neighbors[left_position]:
            if right_position in visited:
                continue
            visited.add(right_position)
            prior = right_match.get(right_position)
            if prior is None or augment(prior, visited):
                right_match[right_position] = left_position
                return True
        return False

    for left_position in range(len(left)):
        augment(left_position, set())
    edge_r2 = {
        (left_position, right_position): r2
        for left_position, entries in enumerate(neighbors)
        for right_position, r2 in entries
    }
    return sorted(
        (
            left_position,
            right_position,
            edge_r2[(left_position, right_position)],
        )
        for right_position, left_position in right_match.items()
    )


def corrected_repeated_kfold_test(
    differences: FloatSequence,
    test_train_ratios: FloatSequence,
) -> CorrectedRepeatedKFoldResult:
    """Test a lower mean fold difference with corrected repeated-CV variance."""
    values = np.asarray(differences, dtype=np.float64)
    ratios = np.asarray(test_train_ratios, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) < 4
        or ratios.shape != values.shape
        or not np.isfinite(values).all()
        or not np.isfinite(ratios).all()
        or np.any(ratios <= 0.0)
    ):
        raise ValueError(
            "fold differences and positive test-to-train ratios must contain "
            "at least four aligned finite values"
        )
    mean_difference = float(values.mean())
    sample_variance = float(values.var(ddof=1))
    if sample_variance <= 0.0:
        raise ValueError("corrected repeated-CV inference requires positive fold variance")
    mean_ratio = float(ratios.mean())
    correction = 1.0 / len(values) + mean_ratio
    standard_error = float(np.sqrt(correction * sample_variance))
    statistic = mean_difference / standard_error
    degrees_of_freedom = len(values) - 1
    p_value = float(student_t.cdf(statistic, df=degrees_of_freedom))
    return CorrectedRepeatedKFoldResult(
        n_fold_differences=len(values),
        mean_difference=mean_difference,
        sample_variance=sample_variance,
        mean_test_train_ratio=mean_ratio,
        variance_correction=correction,
        standard_error=standard_error,
        degrees_of_freedom=degrees_of_freedom,
        test_statistic=statistic,
        lower_tail_p_value=p_value,
    )


def holm_adjust(p_values: FloatSequence) -> np.ndarray:
    """Return Holm-adjusted p-values in their original order."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be a finite one-dimensional probability vector")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    adjusted = np.empty(len(values), dtype=np.float64)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


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
    if set(result.columns) != set(DGRP_COVARIATES):
        raise RuntimeError("DGRP covariate names differ from the frozen specification")
    return result.reindex(index=expected_lines, columns=DGRP_COVARIATES).reset_index()


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


def _frame_from_rows(name: str, rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=DGRP_RESULT_SCHEMAS[name])


def _trait_arrays(
    trait: str,
    outcomes: pd.DataFrame,
    covariates: pd.DataFrame,
    genotype_lines: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    trait_outcomes = outcomes[outcomes["trait"] == trait].copy()
    if trait_outcomes.empty:
        raise ValueError(f"DGRP outcomes omit requested trait {trait!r}")
    trait_outcomes["line_id"] = trait_outcomes["strain_number"].map(normalize_phenotype_line)
    trait_outcomes = trait_outcomes.sort_values(
        "line_id",
        key=lambda values: values.map(_line_sort_key),
    )
    line_ids = tuple(trait_outcomes["line_id"].astype(str))
    if len(set(line_ids)) != len(line_ids):
        raise ValueError(f"DGRP trait {trait} contains duplicate line outcomes")
    genotype_position = {line_id: index for index, line_id in enumerate(genotype_lines)}
    if not set(line_ids).issubset(genotype_position):
        missing = sorted(set(line_ids) - set(genotype_position), key=_line_sort_key)
        raise ValueError(f"DGRP trait {trait} has lines absent from genotypes: {missing}")
    covariate_frame = covariates.set_index("strain_number")
    missing_covariates = sorted(set(line_ids) - set(covariate_frame.index), key=_line_sort_key)
    if missing_covariates:
        raise ValueError(
            f"DGRP trait {trait} has lines absent from covariates: {missing_covariates}"
        )
    if tuple(column for column in covariate_frame.columns) != DGRP_COVARIATES:
        raise ValueError("DGRP covariate columns differ from the frozen specification")
    sample_indices = np.asarray(
        [genotype_position[line_id] for line_id in line_ids], dtype=np.int64
    )
    target = trait_outcomes["outcome"].to_numpy(dtype=np.float64)
    confounders = covariate_frame.loc[list(line_ids), list(DGRP_COVARIATES)].to_numpy(
        dtype=np.float64
    )
    return line_ids, sample_indices, target, confounders


def _append_stability_rows(
    *,
    profile: Profile,
    trait: str,
    repeat: int,
    method: str,
    fold_a: int,
    fold_b: int,
    k: int,
    left_rows: np.ndarray,
    right_rows: np.ndarray,
    shared_samples: np.ndarray,
    genotypes: np.ndarray,
    variants: pd.DataFrame,
    summary_rows: list[dict[str, object]],
    match_rows: list[dict[str, object]],
) -> None:
    variant_indices = variants["variant_index"].to_numpy(dtype=np.int64)
    chromosomes = variants["chromosome"].astype(str).to_numpy()
    right_by_variant = {
        int(variant_indices[row]): position for position, row in enumerate(right_rows)
    }
    exact_matches = [
        (left_position, right_by_variant[int(variant_indices[row])], 1.0)
        for left_position, row in enumerate(left_rows)
        if int(variant_indices[row]) in right_by_variant
    ]
    ld_matches = match_ranked_variants(
        genotypes,
        variants,
        left_rows,
        right_rows,
        shared_samples,
    )
    for match_type, matches in (
        ("exact_snp", exact_matches),
        ("ld_r2_ge_0.8", ld_matches),
    ):
        union_size = 2 * k - len(matches)
        summary_rows.append(
            {
                "profile": profile,
                "trait": trait,
                "repeat": repeat,
                "method": method,
                "fold_a": fold_a,
                "fold_b": fold_b,
                "k": k,
                "match_type": match_type,
                "matches": len(matches),
                "union_size": union_size,
                "jaccard": len(matches) / union_size,
            }
        )
        for left_position, right_position, r2 in matches:
            left_row = left_rows[left_position]
            right_row = right_rows[right_position]
            match_rows.append(
                {
                    "profile": profile,
                    "trait": trait,
                    "repeat": repeat,
                    "method": method,
                    "fold_a": fold_a,
                    "fold_b": fold_b,
                    "k": k,
                    "match_type": match_type,
                    "rank_a": left_position + 1,
                    "variant_index_a": int(variant_indices[left_row]),
                    "rank_b": right_position + 1,
                    "variant_index_b": int(variant_indices[right_row]),
                    "chromosome": chromosomes[left_row],
                    "r2": r2,
                }
            )


def _prediction_test_row(
    predictions: pd.DataFrame,
    trait: str,
    *,
    settings: DGRPSettings,
) -> dict[str, object] | None:
    selected = predictions[
        (predictions["trait"] == trait)
        & (predictions["k"] == 10)
        & predictions["method"].isin(("marginal", "cif"))
    ]
    if selected.empty:
        return None
    pivot = selected.pivot(
        index=["repeat", "fold", "line_id"],
        columns="method",
        values="squared_error",
    )
    if tuple(sorted(pivot.columns)) != ("cif", "marginal") or pivot.isna().any().any():
        raise ValueError(f"DGRP prediction contrast is incomplete for trait {trait}")
    indexed = pivot.reset_index()
    if (
        indexed.groupby(["repeat", "line_id"], sort=True).size().ne(1).any()
        or sorted(indexed["repeat"].unique().tolist()) != list(range(settings.n_repeats))
        or indexed.groupby("repeat", sort=True)["fold"]
        .apply(lambda values: sorted(values.unique().tolist()))
        .apply(lambda values: values != list(range(settings.n_splits)))
        .any()
    ):
        raise ValueError(f"DGRP prediction folds are incomplete or overlapping for trait {trait}")
    indexed["difference"] = indexed["cif"] - indexed["marginal"]
    grouped = indexed.groupby(["repeat", "fold"], sort=True)
    fold_differences = grouped["difference"].mean()
    evaluation_counts = grouped.size().astype(np.float64)
    lines_per_repeat = indexed.groupby("repeat", sort=True)["line_id"].nunique()
    if (
        len(fold_differences) != settings.n_repeats * settings.n_splits
        or lines_per_repeat.nunique() != 1
    ):
        raise ValueError(f"DGRP repeated-CV inventory is incomplete for trait {trait}")
    n_lines = int(lines_per_repeat.iloc[0])
    training_counts = n_lines - evaluation_counts.to_numpy(dtype=np.float64)
    if np.any(training_counts <= 0):
        raise ValueError(f"DGRP training fold is empty for trait {trait}")
    test_train_ratios = evaluation_counts.to_numpy(dtype=np.float64) / np.asarray(
        training_counts,
        dtype=np.float64,
    )
    result = corrected_repeated_kfold_test(
        fold_differences.to_numpy(dtype=np.float64),
        test_train_ratios,
    )
    return {
        "profile": settings.profile,
        "trait": trait,
        "hypothesis_id": f"{trait.lower()}_cif_vs_marginal_top10_squared_error",
        "contrast": "cif_minus_marginal",
        "method_a": "cif",
        "method_b": "marginal",
        "k": 10,
        "metric": "held_out_mean_squared_error",
        "alternative": "cif_lower",
        "candidate_screen": DGRP_CANDIDATE_SCREEN,
        "candidate_count": settings.candidate_count,
        "screen_tie_breaker": DGRP_SCREEN_TIE_BREAKER,
        "method_a_ranking": DGRP_RANKING_METHODS["cif"],
        "method_b_ranking": DGRP_RANKING_METHODS["marginal"],
        "ranking_tie_breaker": DGRP_RANKING_TIE_BREAKER,
        "downstream_model": DGRP_DOWNSTREAM_MODEL,
        "test": DGRP_INFERENCE_TEST,
        "unit_of_analysis": DGRP_INFERENCE_UNIT,
        "estimand": DGRP_INFERENCE_ESTIMAND,
        "inference_status": DGRP_PROFILE_INFERENCE_STATUS[settings.profile],
        "n_repeats": settings.n_repeats,
        "n_splits": settings.n_splits,
        "n_fold_differences": result.n_fold_differences,
        "n_lines": n_lines,
        "mean_difference": result.mean_difference,
        "sample_variance": result.sample_variance,
        "mean_test_train_ratio": result.mean_test_train_ratio,
        "variance_correction": result.variance_correction,
        "standard_error": result.standard_error,
        "degrees_of_freedom": result.degrees_of_freedom,
        "test_statistic": result.test_statistic,
        "raw_p_value": result.lower_tail_p_value,
    }


def build_inference_tables(
    predictions: pd.DataFrame,
    settings: DGRPSettings,
    *,
    base_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the primary EDD test and fixed six-trait Holm family."""
    if not isinstance(base_seed, int):
        raise TypeError("DGRP base seed must be an integer")
    primary = _prediction_test_row(
        predictions,
        DGRP_PRIMARY_TRAIT,
        settings=settings,
    )
    if primary is None:
        raise ValueError("DGRP primary EDD prediction contrast is undefined")
    primary_frame = _frame_from_rows("primary_inference", [primary])

    secondary_rows: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    for trait in DGRP_SECONDARY_TRAITS:
        test = _prediction_test_row(
            predictions,
            trait,
            settings=settings,
        )
        test_defined = test is not None
        if test is None:
            test = {
                "profile": settings.profile,
                "trait": trait,
                "hypothesis_id": f"{trait.lower()}_cif_vs_marginal_top10_squared_error",
                "contrast": "cif_minus_marginal",
                "method_a": "cif",
                "method_b": "marginal",
                "k": 10,
                "metric": "held_out_mean_squared_error",
                "alternative": "cif_lower",
                "candidate_screen": DGRP_CANDIDATE_SCREEN,
                "candidate_count": settings.candidate_count,
                "screen_tie_breaker": DGRP_SCREEN_TIE_BREAKER,
                "method_a_ranking": DGRP_RANKING_METHODS["cif"],
                "method_b_ranking": DGRP_RANKING_METHODS["marginal"],
                "ranking_tie_breaker": DGRP_RANKING_TIE_BREAKER,
                "downstream_model": DGRP_DOWNSTREAM_MODEL,
                "test": DGRP_INFERENCE_TEST,
                "unit_of_analysis": DGRP_INFERENCE_UNIT,
                "estimand": DGRP_INFERENCE_ESTIMAND,
                "inference_status": DGRP_PROFILE_INFERENCE_STATUS[settings.profile],
                "n_repeats": settings.n_repeats,
                "n_splits": settings.n_splits,
                "n_fold_differences": 0,
                "n_lines": 0,
                "mean_difference": np.nan,
                "sample_variance": np.nan,
                "mean_test_train_ratio": np.nan,
                "variance_correction": np.nan,
                "standard_error": np.nan,
                "degrees_of_freedom": 0,
                "test_statistic": np.nan,
                "raw_p_value": np.nan,
            }
        raw_p_value = test["raw_p_value"]
        if test_defined and not isinstance(
            raw_p_value,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("DGRP raw p-value must be numeric")
        holm_input = float(cast(float, raw_p_value)) if test_defined else 1.0
        raw_p_values.append(holm_input)
        secondary_rows.append(
            {
                "profile": settings.profile,
                "family": DGRP_HOLM_FAMILY,
                "family_size": len(DGRP_SECONDARY_TRAITS),
                "hypothesis_id": test["hypothesis_id"],
                "trait": trait,
                "contrast": test["contrast"],
                "method_a": test["method_a"],
                "method_b": test["method_b"],
                "k": test["k"],
                "metric": test["metric"],
                "alternative": test["alternative"],
                "candidate_screen": test["candidate_screen"],
                "candidate_count": test["candidate_count"],
                "screen_tie_breaker": test["screen_tie_breaker"],
                "method_a_ranking": test["method_a_ranking"],
                "method_b_ranking": test["method_b_ranking"],
                "ranking_tie_breaker": test["ranking_tie_breaker"],
                "downstream_model": test["downstream_model"],
                "test": test["test"],
                "unit_of_analysis": test["unit_of_analysis"],
                "estimand": test["estimand"],
                "inference_status": test["inference_status"],
                "test_defined": test_defined,
                "n_repeats": test["n_repeats"],
                "n_splits": test["n_splits"],
                "n_fold_differences": test["n_fold_differences"],
                "n_lines": test["n_lines"],
                "mean_difference": test["mean_difference"],
                "sample_variance": test["sample_variance"],
                "mean_test_train_ratio": test["mean_test_train_ratio"],
                "variance_correction": test["variance_correction"],
                "standard_error": test["standard_error"],
                "degrees_of_freedom": test["degrees_of_freedom"],
                "test_statistic": test["test_statistic"],
                "raw_p_value": test["raw_p_value"],
                "holm_input_p_value": holm_input,
            }
        )
    adjusted = holm_adjust(raw_p_values)
    for row, adjusted_p_value in zip(secondary_rows, adjusted, strict=True):
        row["holm_adjusted_p_value"] = adjusted_p_value
        row["holm_reject_0_05"] = bool(adjusted_p_value <= DGRP_HOLM_ALPHA)
    return primary_frame, _frame_from_rows("secondary_holm", secondary_rows)


def _frame_key_set(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> set[tuple[object, ...]]:
    """Return unique row keys for the requested columns."""
    return set(frame.loc[:, list(columns)].itertuples(index=False, name=None))


def _require_frame_equal(
    name: str,
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_by: Sequence[str],
) -> None:
    """Require two result frames to agree after deterministic sorting."""
    observed_sorted = observed.sort_values(list(sort_by)).reset_index(drop=True)
    expected_sorted = expected.sort_values(list(sort_by)).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            observed_sorted,
            expected_sorted,
            check_dtype=False,
            check_exact=False,
            rtol=DGRP_NUMERIC_RTOL,
            atol=DGRP_NUMERIC_ATOL,
        )
    except AssertionError as error:
        raise ValueError(f"DGRP {name} differs from recomputed values") from error


def _validate_assignment_inventory(
    assignments: pd.DataFrame,
    settings: DGRPSettings,
    base_seed: int | None,
) -> tuple[
    int,
    dict[str, set[str]],
    dict[tuple[str, int, int], set[str]],
]:
    """Validate repeated folds and return their line inventories."""
    if set(assignments["trait"]) != set(settings.traits):
        raise ValueError("DGRP fold assignments differ from the profile trait inventory")
    if not assignments["profile"].eq(settings.profile).all():
        raise ValueError("DGRP fold assignment profile differs")
    first = assignments[(assignments["trait"] == settings.traits[0]) & (assignments["repeat"] == 0)]
    if first.empty or first["split_seed"].nunique() != 1:
        raise ValueError("DGRP first split seed is missing or ambiguous")
    inferred_seed = int(first["split_seed"].iloc[0])
    if base_seed is None:
        base_seed = inferred_seed
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("DGRP base seed must be an integer")
    if inferred_seed != base_seed:
        raise ValueError("DGRP first split seed differs from the base seed")

    line_inventory: dict[str, set[str]] = {}
    evaluation_sets: dict[tuple[str, int, int], set[str]] = {}
    for trait_position, trait in enumerate(settings.traits):
        trait_rows = assignments[assignments["trait"] == trait]
        lines = set(trait_rows["line_id"].astype(str))
        if not lines:
            raise ValueError(f"DGRP trait {trait} has no assigned lines")
        line_inventory[trait] = lines
        fingerprints: set[tuple[tuple[str, ...], ...]] = set()
        for repeat in range(settings.n_repeats):
            repeated = trait_rows[trait_rows["repeat"] == repeat]
            expected_split_seed = base_seed + trait_position * 10_000 + repeat
            if (
                len(repeated) != len(lines) * settings.n_splits
                or repeated.duplicated(["fold", "line_id"]).any()
                or set(repeated["fold"]) != set(range(settings.n_splits))
                or set(repeated["line_id"].astype(str)) != lines
                or not repeated["split_seed"].eq(expected_split_seed).all()
                or not set(repeated["role"]).issubset({"training", "evaluation"})
            ):
                raise ValueError(
                    f"DGRP assignment inventory is incomplete for {trait} repeat {repeat}"
                )
            role_counts = pd.crosstab(repeated["line_id"], repeated["role"]).reindex(
                columns=["training", "evaluation"],
                fill_value=0,
            )
            if (
                not role_counts["evaluation"].eq(1).all()
                or not role_counts["training"].eq(settings.n_splits - 1).all()
            ):
                raise ValueError(f"DGRP line roles are invalid for {trait} repeat {repeat}")
            partition: list[tuple[str, ...]] = []
            for fold in range(settings.n_splits):
                folded = repeated[repeated["fold"] == fold]
                if set(folded["line_id"].astype(str)) != lines:
                    raise ValueError(
                        f"DGRP fold inventory is incomplete for {trait} repeat {repeat} fold {fold}"
                    )
                evaluation = set(
                    folded.loc[
                        folded["role"] == "evaluation",
                        "line_id",
                    ].astype(str)
                )
                if not evaluation or len(evaluation) == len(lines):
                    raise ValueError(
                        f"DGRP evaluation fold is invalid for {trait} repeat {repeat} fold {fold}"
                    )
                evaluation_sets[(trait, repeat, fold)] = evaluation
                partition.append(tuple(sorted(evaluation)))
            fingerprints.add(tuple(sorted(partition)))
        if len(fingerprints) != settings.n_repeats:
            raise ValueError(f"DGRP repeated partitions are duplicated for trait {trait}")
    return base_seed, line_inventory, evaluation_sets


def _validate_execution_seeds(
    frame: pd.DataFrame,
    name: str,
    settings: DGRPSettings,
    base_seed: int,
) -> None:
    """Require every row to carry its deterministic split and model seeds."""
    trait_positions = {trait: position for position, trait in enumerate(settings.traits)}
    positions = frame["trait"].map(trait_positions)
    if positions.isna().any():
        raise ValueError(f"DGRP {name} contains an unknown trait")
    expected_split = (
        base_seed
        + positions.to_numpy(dtype=np.int64) * 10_000
        + frame["repeat"].to_numpy(dtype=np.int64)
    )
    expected_model = (
        base_seed
        + positions.to_numpy(dtype=np.int64) * 100_000
        + frame["repeat"].to_numpy(dtype=np.int64) * 1_000
        + frame["fold"].to_numpy(dtype=np.int64)
    )
    if not np.array_equal(
        frame["split_seed"].to_numpy(dtype=np.int64),
        expected_split,
    ) or not np.array_equal(
        frame["model_seed"].to_numpy(dtype=np.int64),
        expected_model,
    ):
        raise ValueError(f"DGRP {name} execution seeds differ from the fixed formulas")


def _validate_ranking_inventory(
    rankings: pd.DataFrame,
    settings: DGRPSettings,
    base_seed: int,
) -> None:
    """Require the exact shared-candidate ranking product."""
    key = ["trait", "repeat", "fold", "method"]
    expected_keys = set(
        product(
            settings.traits,
            range(settings.n_repeats),
            range(settings.n_splits),
            DGRP_METHODS,
        )
    )
    if _frame_key_set(rankings, key) != expected_keys:
        raise ValueError("DGRP ranking group inventory differs from the fixed product")
    groups = rankings.groupby(key, sort=False)
    expected_order = list(range(1, settings.candidate_count + 1))
    if (
        not groups.size().eq(settings.candidate_count).all()
        or not groups["rank"].apply(lambda values: sorted(values.tolist()) == expected_order).all()
        or not groups["candidate_rank"]
        .apply(lambda values: sorted(values.tolist()) == expected_order)
        .all()
        or groups["variant_index"].nunique().ne(settings.candidate_count).any()
    ):
        raise ValueError("DGRP rankings are not complete one-based candidate orders")
    scores = rankings["score"].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("DGRP ranking scores must be finite")
    for raw_key, frame in groups:
        ranking_key = cast(tuple[str, int, int, str], raw_key)
        expected_positions = np.lexsort(
            (
                frame["candidate_rank"].to_numpy(dtype=np.int64),
                -frame["score"].to_numpy(dtype=np.float64),
            )
        )
        expected_candidate_order = frame.iloc[expected_positions]["candidate_rank"].to_numpy(
            dtype=np.int64
        )
        observed_candidate_order = frame.sort_values("rank")["candidate_rank"].to_numpy(
            dtype=np.int64
        )
        if not np.array_equal(observed_candidate_order, expected_candidate_order):
            raise ValueError(f"DGRP ranking order differs from scores for {ranking_key}")

    screen_groups = rankings[rankings["method"] == "marginal"].groupby(
        ["trait", "repeat", "fold"],
        sort=False,
    )
    for raw_key, frame in screen_groups:
        screen_key = cast(tuple[str, int, int], raw_key)
        ordered = frame.sort_values("candidate_rank")
        expected_positions = np.lexsort(
            (
                ordered["variant_index"].to_numpy(dtype=np.int64),
                -ordered["score"].to_numpy(dtype=np.float64),
            )
        )
        if not np.array_equal(expected_positions, np.arange(settings.candidate_count)):
            raise ValueError(f"DGRP candidate screen order differs from scores for {screen_key}")

    candidate_key = ["trait", "repeat", "fold", "candidate_rank"]
    candidate_groups = rankings.groupby(candidate_key, sort=False)
    identity_columns = (
        "variant_index",
        "chromosome",
        "variant_id",
        "position",
        "allele_a",
        "allele_b",
    )
    if (
        candidate_groups.size().ne(len(DGRP_METHODS)).any()
        or candidate_groups["method"].nunique().ne(len(DGRP_METHODS)).any()
    ):
        raise ValueError("DGRP methods do not share one complete candidate mapping")
    for column in identity_columns:
        if candidate_groups[column].nunique(dropna=False).ne(1).any():
            raise ValueError(f"DGRP shared candidate identity differs for column {column}")
    expected_methods = rankings["method"].map(DGRP_RANKING_METHODS)
    if (
        not rankings["profile"].eq(settings.profile).all()
        or not rankings["candidate_screen"].eq(DGRP_CANDIDATE_SCREEN).all()
        or not rankings["candidate_count"].eq(settings.candidate_count).all()
        or not rankings["screen_tie_breaker"].eq(DGRP_SCREEN_TIE_BREAKER).all()
        or not rankings["ranking_tie_breaker"].eq(DGRP_RANKING_TIE_BREAKER).all()
        or expected_methods.isna().any()
        or not rankings["ranking_method"].eq(expected_methods).all()
    ):
        raise ValueError("DGRP ranking method metadata differs from the specification")
    _validate_execution_seeds(rankings, "rankings", settings, base_seed)


def _validate_prediction_inventory(
    predictions: pd.DataFrame,
    settings: DGRPSettings,
    base_seed: int,
    evaluation_sets: Mapping[tuple[str, int, int], set[str]],
) -> dict[tuple[str, int, int, str, int], pd.DataFrame]:
    """Validate prediction keys, line membership, metadata, and losses."""
    key = ["trait", "repeat", "fold", "method", "k"]
    expected_keys = set(
        product(
            settings.traits,
            range(settings.n_repeats),
            range(settings.n_splits),
            DGRP_METHODS,
            settings.prediction_k,
        )
    )
    if _frame_key_set(predictions, key) != expected_keys:
        raise ValueError("DGRP prediction group inventory differs from the fixed product")
    if predictions.duplicated([*key, "line_id"]).any():
        raise ValueError("DGRP prediction keys are duplicated")
    numeric = predictions[["observed", "prediction", "squared_error"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("DGRP predictions contain non-finite values")
    expected_errors = (
        predictions["observed"].to_numpy(dtype=np.float64)
        - predictions["prediction"].to_numpy(dtype=np.float64)
    ) ** 2
    if not np.allclose(
        predictions["squared_error"].to_numpy(dtype=np.float64),
        expected_errors,
        rtol=DGRP_NUMERIC_RTOL,
        atol=DGRP_NUMERIC_ATOL,
    ):
        raise ValueError("DGRP prediction squared errors differ from recomputation")
    observed_groups = predictions.groupby(
        ["trait", "repeat", "fold", "line_id"],
        sort=False,
    )["observed"]
    if observed_groups.nunique(dropna=False).ne(1).any():
        raise ValueError("DGRP observed outcomes differ across methods or feature counts")
    expected_methods = predictions["method"].map(DGRP_RANKING_METHODS)
    if (
        not predictions["profile"].eq(settings.profile).all()
        or not predictions["candidate_screen"].eq(DGRP_CANDIDATE_SCREEN).all()
        or not predictions["candidate_count"].eq(settings.candidate_count).all()
        or not predictions["screen_tie_breaker"].eq(DGRP_SCREEN_TIE_BREAKER).all()
        or not predictions["ranking_tie_breaker"].eq(DGRP_RANKING_TIE_BREAKER).all()
        or not predictions["prediction_scale"].eq(DGRP_PREDICTION_SCALE).all()
        or expected_methods.isna().any()
        or not predictions["ranking_method"].eq(expected_methods).all()
    ):
        raise ValueError("DGRP prediction method metadata differs from the specification")
    _validate_execution_seeds(predictions, "predictions", settings, base_seed)

    grouped: dict[tuple[str, int, int, str, int], pd.DataFrame] = {}
    for raw_key, frame in predictions.groupby(key, sort=False):
        group_key = cast(tuple[str, int, int, str, int], raw_key)
        expected_lines = evaluation_sets[group_key[:3]]
        if set(frame["line_id"].astype(str)) != expected_lines:
            raise ValueError(f"DGRP prediction lines differ for group {group_key}")
        grouped[group_key] = frame
    return grouped


def _validate_fold_metrics(
    metrics: pd.DataFrame,
    settings: DGRPSettings,
    base_seed: int,
    prediction_groups: Mapping[tuple[str, int, int, str, int], pd.DataFrame],
    line_inventory: Mapping[str, set[str]],
) -> None:
    """Recompute every fold metric from held-out predictions."""
    key = ["trait", "repeat", "fold", "method", "k"]
    if _frame_key_set(metrics, key) != set(prediction_groups) or metrics.duplicated(key).any():
        raise ValueError("DGRP fold metric inventory differs from predictions")
    expected_methods = metrics["method"].map(DGRP_RANKING_METHODS)
    if (
        not metrics["profile"].eq(settings.profile).all()
        or not metrics["candidate_screen"].eq(DGRP_CANDIDATE_SCREEN).all()
        or not metrics["candidate_count"].eq(settings.candidate_count).all()
        or not metrics["screen_tie_breaker"].eq(DGRP_SCREEN_TIE_BREAKER).all()
        or not metrics["ranking_tie_breaker"].eq(DGRP_RANKING_TIE_BREAKER).all()
        or expected_methods.isna().any()
        or not metrics["ranking_method"].eq(expected_methods).all()
        or not metrics["downstream_model"].eq(DGRP_DOWNSTREAM_MODEL).all()
    ):
        raise ValueError("DGRP fold metric metadata differs from the specification")
    _validate_execution_seeds(metrics, "fold metrics", settings, base_seed)
    baselines = metrics["training_baseline"].to_numpy(dtype=np.float64)
    if not np.isfinite(baselines).all() or np.any(np.abs(baselines) > DGRP_BASELINE_ATOL):
        raise ValueError("DGRP training baselines are invalid for residual outcomes")
    if (
        metrics.groupby(["trait", "repeat", "fold"])["training_baseline"]
        .nunique(dropna=False)
        .ne(1)
        .any()
    ):
        raise ValueError("DGRP training baselines differ within folds")

    for raw_key, row in metrics.set_index(key).iterrows():
        group_key = cast(tuple[str, int, int, str, int], raw_key)
        predictions = prediction_groups[group_key]
        expected_evaluation = len(predictions)
        expected_training = len(line_inventory[group_key[0]]) - expected_evaluation
        errors = predictions["squared_error"].to_numpy(dtype=np.float64)
        baseline = float(row["training_baseline"])
        denominator = float(
            np.sum((predictions["observed"].to_numpy(dtype=np.float64) - baseline) ** 2)
        )
        expected_r2 = 1.0 - float(errors.sum()) / denominator if denominator > 0.0 else np.nan
        if (
            int(row["n_training"]) != expected_training
            or int(row["n_evaluation"]) != expected_evaluation
            or not np.isclose(
                float(row["mean_squared_error"]),
                float(errors.mean()),
                rtol=DGRP_NUMERIC_RTOL,
                atol=DGRP_NUMERIC_ATOL,
            )
            or not (
                np.isnan(expected_r2)
                and pd.isna(row["predictive_r2"])
                or np.isclose(
                    float(row["predictive_r2"]),
                    expected_r2,
                    rtol=DGRP_NUMERIC_RTOL,
                    atol=DGRP_NUMERIC_ATOL,
                )
            )
        ):
            raise ValueError(f"DGRP fold metrics differ from predictions for {group_key}")


def _validate_ld_groups(
    ld_groups: pd.DataFrame,
    rankings: pd.DataFrame,
    settings: DGRPSettings,
    base_seed: int,
) -> None:
    """Require LD rows to preserve ranking and representative identities."""
    key = ["trait", "repeat", "fold", "method"]
    if _frame_key_set(ld_groups, key) != _frame_key_set(rankings, key):
        raise ValueError("DGRP LD group inventory differs from rankings")
    if (
        ld_groups.groupby(key).size().ne(settings.candidate_count).any()
        or ld_groups.duplicated([*key, "rank"]).any()
    ):
        raise ValueError("DGRP LD groups do not contain every ranked candidate")
    _validate_execution_seeds(ld_groups, "LD groups", settings, base_seed)
    ranking_identity = rankings.loc[
        :,
        [*key, "rank", "variant_index", "chromosome", "variant_id"],
    ]
    merged = ld_groups.merge(
        ranking_identity,
        on=[*key, "rank"],
        how="outer",
        suffixes=("_ld", "_ranking"),
        validate="one_to_one",
        indicator=True,
    )
    if (
        not merged["_merge"].eq("both").all()
        or not merged["variant_index_ld"].eq(merged["variant_index_ranking"]).all()
        or not merged["chromosome_ld"].eq(merged["chromosome_ranking"]).all()
        or not merged["variant_id_ld"].eq(merged["variant_id_ranking"]).all()
    ):
        raise ValueError("DGRP LD rows differ from ranked variant identities")
    representatives = rankings.loc[
        :,
        [*key, "rank", "variant_index", "variant_id"],
    ].rename(
        columns={
            "rank": "representative_rank",
            "variant_index": "expected_representative_variant_index",
            "variant_id": "expected_representative_variant_id",
        }
    )
    resolved = ld_groups.merge(
        representatives,
        on=[*key, "representative_rank"],
        how="left",
        validate="many_to_one",
    )
    r2 = resolved["r2_to_representative"].to_numpy(dtype=np.float64)
    expected_representative = resolved["rank"].eq(resolved["representative_rank"]) & resolved[
        "variant_index"
    ].eq(resolved["representative_variant_index"])
    group_key = [*key, "ld_group_id"]
    if (
        not ld_groups["profile"].eq(settings.profile).all()
        or resolved["expected_representative_variant_index"].isna().any()
        or not resolved["representative_variant_index"]
        .eq(resolved["expected_representative_variant_index"])
        .all()
        or not resolved["representative_variant_id"]
        .eq(resolved["expected_representative_variant_id"])
        .all()
        or not np.isfinite(r2).all()
        or np.any((r2 < DGRP_LD_R2_THRESHOLD) | (r2 > 1.0 + DGRP_NUMERIC_ATOL))
        or not resolved["is_representative"].eq(expected_representative).all()
        or not np.allclose(
            r2[expected_representative.to_numpy(dtype=bool)],
            1.0,
            rtol=DGRP_NUMERIC_RTOL,
            atol=DGRP_NUMERIC_ATOL,
        )
        or resolved["representative_rank"].gt(resolved["rank"]).any()
        or resolved.groupby(group_key)["is_representative"].sum().ne(1).any()
        or resolved.groupby(group_key)["representative_variant_index"].nunique().ne(1).any()
        or resolved.groupby(group_key)["chromosome"].nunique().ne(1).any()
    ):
        raise ValueError("DGRP LD representative metadata is inconsistent")


def _validate_stability_tables(
    summaries: pd.DataFrame,
    matches: pd.DataFrame,
    rankings: pd.DataFrame,
    settings: DGRPSettings,
) -> None:
    """Validate stability inventories, counts, and ranking references."""
    key = ["trait", "repeat", "method", "fold_a", "fold_b", "k", "match_type"]
    expected_keys = {
        (trait, repeat, method, fold_a, fold_b, k, match_type)
        for trait, repeat, method, k, match_type in product(
            settings.traits,
            range(settings.n_repeats),
            DGRP_METHODS,
            settings.stability_k,
            ("exact_snp", "ld_r2_ge_0.8"),
        )
        for fold_a, fold_b in combinations(range(settings.n_splits), 2)
    }
    if _frame_key_set(summaries, key) != expected_keys or summaries.duplicated(key).any():
        raise ValueError("DGRP stability summary inventory differs from the fixed product")
    if not summaries["profile"].eq(settings.profile).all() or (
        not matches.empty and not matches["profile"].eq(settings.profile).all()
    ):
        raise ValueError("DGRP stability profile differs")
    if not _frame_key_set(matches, key).issubset(expected_keys):
        raise ValueError("DGRP stability matches contain an unexpected group")
    if matches.duplicated([*key, "rank_a"]).any() or matches.duplicated([*key, "rank_b"]).any():
        raise ValueError("DGRP stability matching is not one-to-one")
    match_counts = matches.groupby(key).size() if not matches.empty else pd.Series(dtype=int)
    for raw_key, row in summaries.set_index(key).iterrows():
        group_key = cast(tuple[str, int, str, int, int, int, str], raw_key)
        count = int(match_counts.get(group_key, 0))
        k = group_key[5]
        union_size = 2 * k - count
        if (
            int(row["matches"]) != count
            or int(row["union_size"]) != union_size
            or not np.isclose(
                float(row["jaccard"]),
                count / union_size,
                rtol=DGRP_NUMERIC_RTOL,
                atol=DGRP_NUMERIC_ATOL,
            )
        ):
            raise ValueError(f"DGRP stability summary differs for {group_key}")

    ranking_lookup = {
        (
            str(row.trait),
            int(row.repeat),
            int(row.fold),
            str(row.method),
            int(row.rank),
        ): (int(row.variant_index), str(row.chromosome))
        for row in rankings.itertuples(index=False)
    }
    ranking_orders = {
        cast(tuple[str, int, int, str], raw_key): tuple(
            frame.sort_values("rank")["variant_index"].to_numpy(dtype=np.int64)
        )
        for raw_key, frame in rankings.groupby(
            ["trait", "repeat", "fold", "method"],
            sort=False,
        )
    }
    match_groups = {
        cast(tuple[str, int, str, int, int, int, str], raw_key): frame
        for raw_key, frame in matches.groupby(key, sort=False)
    }
    for group_key in expected_keys:
        if group_key[6] != "exact_snp":
            continue
        trait, repeat, method, fold_a, fold_b, k, _ = group_key
        exact_left = ranking_orders[(trait, repeat, fold_a, method)][:k]
        exact_right = ranking_orders[(trait, repeat, fold_b, method)][:k]
        right_ranks = {
            variant_index: rank for rank, variant_index in enumerate(exact_right, start=1)
        }
        expected_matches = {
            (rank_a, int(variant_index), right_ranks[variant_index], int(variant_index))
            for rank_a, variant_index in enumerate(exact_left, start=1)
            if variant_index in right_ranks
        }
        observed = match_groups.get(group_key)
        observed_matches = (
            set(
                observed[["rank_a", "variant_index_a", "rank_b", "variant_index_b"]].itertuples(
                    index=False, name=None
                )
            )
            if observed is not None
            else set()
        )
        if observed_matches != expected_matches:
            raise ValueError(f"DGRP exact stability matches differ for {group_key}")

    for row in matches.itertuples(index=False):
        left_reference = ranking_lookup.get(
            (row.trait, row.repeat, row.fold_a, row.method, row.rank_a)
        )
        right_reference = ranking_lookup.get(
            (row.trait, row.repeat, row.fold_b, row.method, row.rank_b)
        )
        if (
            left_reference is None
            or right_reference is None
            or left_reference[0] != row.variant_index_a
            or right_reference[0] != row.variant_index_b
            or left_reference[1] != row.chromosome
            or right_reference[1] != row.chromosome
            or row.rank_a < 1
            or row.rank_a > row.k
            or row.rank_b < 1
            or row.rank_b > row.k
            or not np.isfinite(row.r2)
            or row.r2 < DGRP_LD_R2_THRESHOLD
            or row.r2 > 1.0 + DGRP_NUMERIC_ATOL
            or (
                row.match_type == "exact_snp"
                and (
                    row.variant_index_a != row.variant_index_b
                    or not np.isclose(row.r2, 1.0, atol=DGRP_NUMERIC_ATOL)
                )
            )
        ):
            raise ValueError("DGRP stability match differs from ranked variants")


def validate_dgrp_results(
    results: Mapping[str, pd.DataFrame],
    settings: DGRPSettings,
    *,
    base_seed: int | None = None,
) -> None:
    """Recompute the complete DGRP result contract before publication."""
    if set(results) != set(DGRP_RESULT_SCHEMAS):
        raise ValueError("DGRP result table inventory differs from the required schemas")
    for name, schema in DGRP_RESULT_SCHEMAS.items():
        if tuple(results[name].columns) != schema:
            raise ValueError(f"DGRP table {name} differs from its required schema")
    core_names = ("fold_assignments", "screening_rankings", "predictions", "fold_metrics")
    if any(results[name].empty for name in core_names):
        raise ValueError("DGRP core analysis tables must be nonempty")

    outcomes = results["line_outcomes"]
    if (
        outcomes.duplicated(["trait", "strain_number"]).any()
        or not np.isfinite(outcomes["outcome"].to_numpy(dtype=np.float64)).all()
    ):
        raise ValueError("DGRP line outcomes contain duplicate or invalid values")
    expected_summary = summarize_line_outcomes(outcomes).loc[
        :,
        DGRP_RESULT_SCHEMAS["trait_summary"],
    ]
    _require_frame_equal(
        "trait summary",
        results["trait_summary"],
        expected_summary,
        sort_by=["trait"],
    )
    expected_line_inventory: dict[str, set[str]] = {}
    for trait in settings.traits:
        trait_outcomes = outcomes[outcomes["trait"] == trait]
        if trait_outcomes.empty:
            raise ValueError(f"DGRP line outcomes omit profile trait {trait}")
        normalized_lines = {
            normalize_phenotype_line(str(value)) for value in trait_outcomes["strain_number"]
        }
        if len(normalized_lines) != len(trait_outcomes):
            raise ValueError(f"DGRP line outcomes collapse duplicate identities for trait {trait}")
        expected_line_inventory[trait] = normalized_lines

    resolved_seed, line_inventory, evaluation_sets = _validate_assignment_inventory(
        results["fold_assignments"],
        settings,
        base_seed,
    )
    if line_inventory != expected_line_inventory:
        raise ValueError("DGRP fold assignment lines differ from line outcomes")
    _validate_ranking_inventory(
        results["screening_rankings"],
        settings,
        resolved_seed,
    )
    prediction_groups = _validate_prediction_inventory(
        results["predictions"],
        settings,
        resolved_seed,
        evaluation_sets,
    )
    _validate_fold_metrics(
        results["fold_metrics"],
        settings,
        resolved_seed,
        prediction_groups,
        line_inventory,
    )
    _validate_ld_groups(
        results["ld_groups"],
        results["screening_rankings"],
        settings,
        resolved_seed,
    )
    _validate_stability_tables(
        results["stability_summary"],
        results["stability_matches"],
        results["screening_rankings"],
        settings,
    )

    expected_primary, expected_secondary = build_inference_tables(
        results["predictions"],
        settings,
        base_seed=resolved_seed,
    )
    _require_frame_equal(
        "primary inference",
        results["primary_inference"],
        expected_primary,
        sort_by=["trait"],
    )
    _require_frame_equal(
        "secondary Holm inference",
        results["secondary_holm"],
        expected_secondary,
        sort_by=["trait"],
    )


def run_dgrp_analysis(
    profile: Profile,
    genotypes: np.ndarray,
    variants: pd.DataFrame,
    genotype_lines: Sequence[str],
    outcomes: pd.DataFrame,
    covariates: pd.DataFrame,
    *,
    base_seed: int = 1718,
) -> dict[str, pd.DataFrame]:
    """Run the complete fold-local DGRP predictive screening analysis."""
    settings = _settings(profile)
    matrix = np.asarray(genotypes)
    if matrix.ndim != 2 or matrix.shape != (len(variants), len(genotype_lines)):
        raise ValueError("DGRP genotype matrix, variants, and line inventory are misaligned")
    required_variant_columns = {
        "variant_index",
        "chromosome",
        "variant_id",
        "position",
        "allele_a",
        "allele_b",
    }
    if not required_variant_columns.issubset(variants.columns):
        raise ValueError("DGRP retained variants omit genomic identity columns")
    if variants["variant_index"].duplicated().any() or variants["variant_id"].duplicated().any():
        raise ValueError("DGRP retained variant identities must be unique")

    assignment_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    ld_group_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    stability_match_rows: list[dict[str, object]] = []
    ranking_state: dict[tuple[str, int, int, str], np.ndarray] = {}
    fold_state: dict[tuple[str, int, int], np.ndarray] = {}

    variant_indices = variants["variant_index"].to_numpy(dtype=np.int64)
    for trait_position, trait in enumerate(settings.traits):
        line_ids, sample_indices, target, confounders = _trait_arrays(
            trait,
            outcomes,
            covariates,
            genotype_lines,
        )
        all_indices = np.arange(len(line_ids), dtype=np.int64)
        for repeat in range(settings.n_repeats):
            split_seed = base_seed + trait_position * 10_000 + repeat
            folds = build_line_folds(
                line_ids,
                n_splits=settings.n_splits,
                random_state=split_seed,
            )
            for fold, evaluation_indices in enumerate(folds):
                model_seed = base_seed + trait_position * 100_000 + repeat * 1_000 + fold
                train_indices = np.setdiff1d(
                    all_indices,
                    evaluation_indices,
                    assume_unique=True,
                )
                fold_state[(trait, repeat, fold)] = train_indices
                evaluation_set = set(evaluation_indices.tolist())
                assignment_rows.extend(
                    {
                        "profile": profile,
                        "trait": trait,
                        "repeat": repeat,
                        "fold": fold,
                        "split_seed": split_seed,
                        "line_id": line_id,
                        "role": "evaluation" if index in evaluation_set else "training",
                    }
                    for index, line_id in enumerate(line_ids)
                )
                candidates = screen_fold_candidates(
                    matrix,
                    variant_indices,
                    sample_indices,
                    target,
                    confounders,
                    train_indices,
                    candidate_count=settings.candidate_count,
                )
                candidate_rows = candidates["variant_row"].to_numpy(dtype=np.int64)
                candidate_metadata = variants.iloc[candidate_rows].reset_index(drop=True)
                train_x, evaluation_x, train_y, evaluation_y, train_raw = (
                    prepare_fold_candidate_matrices(
                        matrix,
                        candidate_rows,
                        sample_indices,
                        target,
                        confounders,
                        train_indices,
                        evaluation_indices,
                    )
                )
                rankings = fit_fold_rankings(
                    train_x,
                    train_y,
                    candidates["candidate_rank"].to_numpy(dtype=np.int64),
                    candidates["screen_score"].to_numpy(dtype=np.float64),
                    n_trees=settings.n_trees,
                    random_state=model_seed,
                )
                candidate_rank_values = candidates["candidate_rank"].to_numpy(dtype=np.int64)
                for method, (order, scores) in rankings.items():
                    ranked_rows = candidate_rows[order]
                    ranking_state[(trait, repeat, fold, method)] = ranked_rows
                    method_metadata = {
                        "candidate_screen": DGRP_CANDIDATE_SCREEN,
                        "candidate_count": settings.candidate_count,
                        "screen_tie_breaker": DGRP_SCREEN_TIE_BREAKER,
                        "ranking_method": DGRP_RANKING_METHODS[method],
                        "ranking_tie_breaker": DGRP_RANKING_TIE_BREAKER,
                    }
                    for rank_position, candidate_position in enumerate(order, start=1):
                        metadata = candidate_metadata.iloc[candidate_position]
                        ranking_rows.append(
                            {
                                "profile": profile,
                                "trait": trait,
                                "repeat": repeat,
                                "fold": fold,
                                "split_seed": split_seed,
                                "model_seed": model_seed,
                                "method": method,
                                **method_metadata,
                                "rank": rank_position,
                                "candidate_rank": int(candidate_rank_values[candidate_position]),
                                "variant_index": int(metadata["variant_index"]),
                                "chromosome": str(metadata["chromosome"]),
                                "variant_id": str(metadata["variant_id"]),
                                "position": int(metadata["position"]),
                                "allele_a": str(metadata["allele_a"]),
                                "allele_b": str(metadata["allele_b"]),
                                "score": float(scores[candidate_position]),
                            }
                        )
                    groups = collapse_ld_redundant_ranking(
                        train_raw,
                        order,
                        candidate_metadata["chromosome"].astype(str),
                    )
                    for group in groups.itertuples(index=False):
                        metadata = candidate_metadata.iloc[group.feature_index]
                        representative = candidate_metadata.iloc[group.representative_feature_index]
                        ld_group_rows.append(
                            {
                                "profile": profile,
                                "trait": trait,
                                "repeat": repeat,
                                "fold": fold,
                                "split_seed": split_seed,
                                "model_seed": model_seed,
                                "method": method,
                                "rank": int(group.rank),
                                "variant_index": int(metadata["variant_index"]),
                                "chromosome": str(metadata["chromosome"]),
                                "variant_id": str(metadata["variant_id"]),
                                "ld_group_id": (
                                    f"{trait}:{repeat}:{fold}:{method}:{int(group.ld_group_id)}"
                                ),
                                "representative_variant_index": int(
                                    representative["variant_index"]
                                ),
                                "representative_variant_id": str(representative["variant_id"]),
                                "representative_rank": int(group.representative_rank),
                                "r2_to_representative": float(group.r2_to_representative),
                                "is_representative": bool(group.is_representative),
                            }
                        )
                    for k in settings.prediction_k:
                        predictions = ridge_predictions(
                            train_x,
                            evaluation_x,
                            train_y,
                            order,
                            k,
                        )
                        errors = (evaluation_y - predictions) ** 2
                        denominator = float(np.sum((evaluation_y - float(train_y.mean())) ** 2))
                        predictive_r2 = (
                            1.0 - float(errors.sum()) / denominator if denominator > 0.0 else np.nan
                        )
                        metric_rows.append(
                            {
                                "profile": profile,
                                "trait": trait,
                                "repeat": repeat,
                                "fold": fold,
                                "split_seed": split_seed,
                                "model_seed": model_seed,
                                "method": method,
                                **method_metadata,
                                "k": k,
                                "downstream_model": DGRP_DOWNSTREAM_MODEL,
                                "n_training": len(train_indices),
                                "n_evaluation": len(evaluation_indices),
                                "training_baseline": float(train_y.mean()),
                                "mean_squared_error": float(errors.mean()),
                                "predictive_r2": predictive_r2,
                            }
                        )
                        prediction_rows.extend(
                            {
                                "profile": profile,
                                "trait": trait,
                                "repeat": repeat,
                                "fold": fold,
                                "split_seed": split_seed,
                                "model_seed": model_seed,
                                "method": method,
                                **method_metadata,
                                "k": k,
                                "line_id": line_ids[line_index],
                                "prediction_scale": DGRP_PREDICTION_SCALE,
                                "observed": float(evaluation_y[position]),
                                "prediction": float(predictions[position]),
                                "squared_error": float(errors[position]),
                            }
                            for position, line_index in enumerate(evaluation_indices)
                        )

            for fold_a, fold_b in combinations(range(settings.n_splits), 2):
                shared_train = np.intersect1d(
                    fold_state[(trait, repeat, fold_a)],
                    fold_state[(trait, repeat, fold_b)],
                    assume_unique=True,
                )
                shared_samples = sample_indices[shared_train]
                for method in DGRP_METHODS:
                    for k in settings.stability_k:
                        _append_stability_rows(
                            profile=profile,
                            trait=trait,
                            repeat=repeat,
                            method=method,
                            fold_a=fold_a,
                            fold_b=fold_b,
                            k=k,
                            left_rows=ranking_state[(trait, repeat, fold_a, method)][:k],
                            right_rows=ranking_state[(trait, repeat, fold_b, method)][:k],
                            shared_samples=shared_samples,
                            genotypes=matrix,
                            variants=variants,
                            summary_rows=stability_rows,
                            match_rows=stability_match_rows,
                        )

    predictions_frame = _frame_from_rows("predictions", prediction_rows)
    primary, secondary = build_inference_tables(
        predictions_frame,
        settings,
        base_seed=base_seed,
    )
    results = {
        "line_outcomes": outcomes.loc[:, DGRP_RESULT_SCHEMAS["line_outcomes"]].copy(),
        "trait_summary": summarize_line_outcomes(outcomes).loc[
            :, DGRP_RESULT_SCHEMAS["trait_summary"]
        ],
        "fold_assignments": _frame_from_rows("fold_assignments", assignment_rows),
        "screening_rankings": _frame_from_rows("screening_rankings", ranking_rows),
        "predictions": predictions_frame,
        "fold_metrics": _frame_from_rows("fold_metrics", metric_rows),
        "ld_groups": _frame_from_rows("ld_groups", ld_group_rows),
        "stability_summary": _frame_from_rows("stability_summary", stability_rows),
        "stability_matches": _frame_from_rows(
            "stability_matches",
            stability_match_rows,
        ),
        "primary_inference": primary,
        "secondary_holm": secondary,
    }
    validate_dgrp_results(results, settings, base_seed=base_seed)
    return results


def write_results(
    results: Mapping[str, pd.DataFrame],
    output_dir: Path,
    *,
    profile: Profile,
    base_seed: int,
    input_sha256: Mapping[str, str],
    genotype_provenance: Mapping[str, object],
    derived_genotype_receipt: Mapping[str, object],
    elapsed_seconds: float,
) -> Path:
    """Atomically write DGRP tables and their complete execution receipt."""
    settings = _settings(profile)
    validate_dgrp_results(results, settings, base_seed=base_seed)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"DGRP output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[3]
    source_files = (
        Path(__file__).resolve(),
        SPECIFICATION_PATH,
        repo_root / "pyproject.toml",
        repo_root / "uv.lock",
    )
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-",
        dir=output_dir.parent,
    ) as temporary:
        staging = Path(temporary) / output_dir.name
        staging.mkdir()
        artifacts: list[Path] = []
        table_metadata: dict[str, dict[str, object]] = {}
        serialized_results: dict[str, pd.DataFrame] = {}
        for name, frame in results.items():
            parquet_path = staging / f"{name}.parquet"
            frame.to_parquet(parquet_path, index=False)
            serialized_results[name] = pd.read_parquet(parquet_path)
            artifacts.append(parquet_path)
            table_metadata[name] = {
                "artifact": parquet_path.name,
                "rows": len(frame),
                "columns": list(frame.columns),
                "sha256": sha256(parquet_path),
            }
        validate_dgrp_results(
            serialized_results,
            settings,
            base_seed=base_seed,
        )
        csv_sort_keys = {
            "trait_summary": ["trait"],
            "fold_metrics": ["trait", "repeat", "fold", "method", "k"],
            "stability_summary": [
                "trait",
                "repeat",
                "method",
                "fold_a",
                "fold_b",
                "k",
                "match_type",
            ],
            "primary_inference": ["trait"],
            "secondary_holm": ["trait"],
        }
        for name, sort_keys in csv_sort_keys.items():
            csv_path = staging / f"{name}.csv"
            results[name].to_csv(csv_path, index=False)
            _require_frame_equal(
                f"{name} CSV",
                pd.read_csv(csv_path),
                results[name],
                sort_by=sort_keys,
            )
            artifacts.append(csv_path)
        receipt = {
            "analysis": "dgrp",
            "schema_version": 2,
            "semantic_validation": "citrees-jss-dgrp-results-v1",
            "profile": profile,
            "base_seed": base_seed,
            "settings": asdict(settings),
            "specification": {
                "path": str(SPECIFICATION_PATH.relative_to(repo_root)),
                "sha256": sha256(SPECIFICATION_PATH),
            },
            "created_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "git_sha": _git_sha(repo_root),
            "git_dirty": _git_dirty(repo_root),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "source_sha256": {
                str(path.relative_to(repo_root)): sha256(path) for path in source_files
            },
            "input_sha256": dict(sorted(input_sha256.items())),
            "versions": {
                package: importlib.metadata.version(package)
                for package in (
                    "citrees",
                    "numpy",
                    "openpyxl",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                )
            },
            "sources": {
                "phenotype": {
                    "url": PHENOTYPE_URL,
                    "filename": PHENOTYPE_FILENAME,
                    "sheet": PHENOTYPE_SHEET,
                    "sha256": PHENOTYPE_SHA256,
                },
                "genotype": dict(genotype_provenance),
                "covariates": {
                    "gwas_analysis_id": DGRP_GWAS_ANALYSIS_ID,
                    "archive_url": COVARIATE_ARCHIVE_URL,
                    "archive_filename": COVARIATE_ARCHIVE_FILENAME,
                    "archive_sha256": COVARIATE_ARCHIVE_SHA256,
                    "database_filename": COVARIATE_DATABASE_FILENAME,
                    "database_sha256": COVARIATE_DATABASE_SHA256,
                },
                "derived_genotypes": dict(derived_genotype_receipt),
            },
            "schemas": {name: list(columns) for name, columns in DGRP_RESULT_SCHEMAS.items()},
            "tables": table_metadata,
            "artifacts": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in sorted(artifacts)
            },
        }
        (staging / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        staging.rename(output_dir)
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("smoke", "quick", "full"),
        default="quick",
        help="DGRP replication workload.",
    )
    parser.add_argument("--seed", type=int, default=1718, help="Base random seed.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profile: Profile = args.profile
    repo_root = Path(__file__).resolve().parents[3]
    if profile == "full" and _git_dirty(repo_root):
        raise RuntimeError("The full DGRP profile requires a clean source tree")
    started = time.perf_counter()
    source_path = acquire_phenotype_workbook(args.data_dir)
    frame = load_individual_phenotypes(source_path)
    genotype_archive = acquire_genotype_archive(args.data_dir)
    genotype_input = extract_genotype_archive(genotype_archive, args.data_dir)
    covariate_archive = acquire_covariate_archive(args.data_dir)
    covariate_database = extract_covariate_database(covariate_archive, args.data_dir)
    genotype_provenance = build_genotype_source_receipt(
        genotype_archive,
        genotype_input,
        frame["strain_number"],
    )
    outcomes = build_line_outcomes(frame)
    validate_pinned_line_outcomes(outcomes)
    genotype_inventory = validate_genotype_files(genotype_input)
    covariates = load_line_covariates(
        covariate_database,
        gwas_analysis_id=DGRP_GWAS_ANALYSIS_ID,
        genotype_lines=genotype_inventory.genotype_lines,
    )
    matrix_path, variants_path, derived_receipt = prepare_derived_genotypes(
        args.data_dir,
        genotype_input,
    )
    genotypes = np.load(matrix_path, mmap_mode="r")
    variants = pd.read_parquet(variants_path)
    results = run_dgrp_analysis(
        profile,
        genotypes,
        variants,
        genotype_inventory.genotype_lines,
        outcomes,
        covariates,
        base_seed=args.seed,
    )
    extracted_paths = _input_paths(genotype_input)
    genotype_files = cast(
        Mapping[str, Mapping[str, object]],
        genotype_provenance["files"],
    )
    derived_artifacts = cast(
        Mapping[str, Mapping[str, object]],
        derived_receipt["artifacts"],
    )
    input_sha256 = {
        PHENOTYPE_FILENAME: PHENOTYPE_SHA256,
        GENOTYPE_ARCHIVE_FILENAME: GENOTYPE_ARCHIVE_SHA256,
        COVARIATE_ARCHIVE_FILENAME: COVARIATE_ARCHIVE_SHA256,
        COVARIATE_DATABASE_FILENAME: COVARIATE_DATABASE_SHA256,
        **{
            path.name: str(genotype_files[name]["sha256"]) for name, path in extracted_paths.items()
        },
        DERIVED_GENOTYPE_MATRIX: str(derived_artifacts[DERIVED_GENOTYPE_MATRIX]["sha256"]),
        DERIVED_VARIANT_INVENTORY: str(derived_artifacts[DERIVED_VARIANT_INVENTORY]["sha256"]),
    }
    output = write_results(
        results,
        args.output_dir,
        profile=profile,
        base_seed=args.seed,
        input_sha256=input_sha256,
        genotype_provenance=genotype_provenance,
        derived_genotype_receipt=derived_receipt,
        elapsed_seconds=time.perf_counter() - started,
    )
    print(f"Wrote verified DGRP {profile} artifacts to {output}.")


if __name__ == "__main__":
    main()
