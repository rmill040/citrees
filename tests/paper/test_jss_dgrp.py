"""Tests for the JSS DGRP phenotype and genotype input validation."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper.jss.replication import dgrp
from paper.jss.replication.dgrp import (
    EXPECTED_COMPLETE_LINES,
    EXPECTED_ELIGIBLE_LINES,
    EXPECTED_INDIVIDUAL_ROWS,
    EXPECTED_OUTCOME_ROWS,
    EXPECTED_SOURCE_LINES,
    EXPECTED_TRAIT_COUNTS,
    TRAITS,
    build_line_outcomes,
    sha256,
    summarize_line_outcomes,
    validate_individual_phenotypes,
    validate_pinned_individual_inventory,
    validate_pinned_line_outcomes,
)

pytestmark = pytest.mark.paper


def _write_tar(path: Path, members: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        for name, payload in members:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.mtime = 0
            archive.addfile(member, io.BytesIO(payload))


def _genotype_specs(
    members: list[tuple[str, bytes]],
) -> tuple[dgrp.GenotypeFileSpec, ...]:
    return tuple(
        dgrp.GenotypeFileSpec(
            member_name=name,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        for name, payload in members
    )


def _phenotypes() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for line, n_rows in (("dgrp1", 7), ("dgrp2", 6), ("dgrp3", 8)):
        for index in range(n_rows):
            row: dict[str, object] = {
                "individual_name": f"{line}_{index}",
                "strain_number": line,
            }
            for trait_index, trait in enumerate(TRAITS):
                row[trait.column] = float(index + trait_index)
            if line == "dgrp3" and index >= 6:
                row[TRAITS[0].column] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def test_line_outcomes_apply_trait_specific_minimum_counts() -> None:
    outcomes = build_line_outcomes(_phenotypes(), min_individuals=7)
    summary = summarize_line_outcomes(outcomes).set_index("trait")

    assert set(outcomes["strain_number"]) == {"dgrp1", "dgrp3"}
    assert summary.loc[TRAITS[0].name, "n_lines"] == 1
    assert summary.loc[TRAITS[0].name, "n_individuals"] == 7
    assert summary.loc[TRAITS[1].name, "n_lines"] == 2
    assert summary.loc[TRAITS[1].name, "n_individuals"] == 15


def test_load_line_covariates_validates_and_pivots_source(tmp_path: Path) -> None:
    database = tmp_path / "dgrp.sqlite"
    import sqlite3

    with sqlite3.connect(database) as connection:
        connection.execute(
            "create table gwas_covariable_value ("
            "associated_gwasanalysis_id integer, strain_number text, "
            "covariable_name text, covariable_value text)"
        )
        connection.executemany(
            "insert into gwas_covariable_value values (?, ?, ?, ?)",
            [
                (7, "line_2", "cov2", "1"),
                (7, "line_1", "cov1", "0"),
                (7, "line_1", "cov2", "2"),
                (7, "line_2", "cov1", "1"),
                (7, "line_3", "cov1", "1"),
                (7, "line_3", "cov2", "2"),
            ],
        )
        connection.commit()

    observed = dgrp.load_line_covariates(
        database,
        gwas_analysis_id=7,
        genotype_lines=["line_2", "line_1"],
    )
    assert observed.columns.tolist() == ["strain_number", "cov1", "cov2"]
    assert observed.to_dict("records") == [
        {"strain_number": "line_2", "cov1": 1.0, "cov2": 1.0},
        {"strain_number": "line_1", "cov1": 0.0, "cov2": 2.0},
    ]


def test_phenotype_validation_rejects_missing_and_duplicate_identities() -> None:
    frame = _phenotypes()
    with pytest.raises(ValueError, match="missing columns"):
        validate_individual_phenotypes(frame.drop(columns=[TRAITS[0].column]))

    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="must be unique"):
        validate_individual_phenotypes(duplicated)


def test_phenotype_validation_rejects_nonfinite_traits() -> None:
    frame = _phenotypes()
    frame.loc[0, TRAITS[0].column] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        validate_individual_phenotypes(frame)


def test_pinned_inventory_validators_reject_incomplete_data() -> None:
    frame = _phenotypes()
    with pytest.raises(RuntimeError, match=str(EXPECTED_INDIVIDUAL_ROWS)):
        validate_pinned_individual_inventory(frame)

    outcomes = build_line_outcomes(frame, min_individuals=7)
    with pytest.raises(RuntimeError, match=str(EXPECTED_OUTCOME_ROWS)):
        validate_pinned_line_outcomes(outcomes)


def test_pinned_inventory_constants_match_reproduced_public_counts() -> None:
    assert EXPECTED_SOURCE_LINES == 167
    assert EXPECTED_ELIGIBLE_LINES == 166
    assert EXPECTED_COMPLETE_LINES == 154
    assert dgrp.GENOTYPE_RECORD_ID == 5_582_846
    assert dgrp.GENOTYPE_URL == "https://zenodo.org/api/records/5582846/files/dgrp2.tar.gz/content"
    assert dgrp.GENOTYPE_ARCHIVE_BYTES == 97_913_546
    assert dgrp.GENOTYPE_ARCHIVE_MD5 == "77c26d1469b18d7da5e6597b6d466454"
    assert (
        dgrp.GENOTYPE_ARCHIVE_SHA256
        == "ff3c318debf28b02d61293b2b82cd12047273f5d743387d748d1ce308ea4c452"
    )
    assert dgrp.EXPECTED_GENOTYPE_SAMPLES == 205
    assert dgrp.EXPECTED_GENOTYPE_VARIANTS == 4_438_427
    assert dgrp.EXPECTED_GENOTYPE_BED_BYTES == 230_798_207
    assert dgrp.EXPECTED_GENOTYPE_ONLY_LINES == 38
    assert {spec.member_name: spec.size_bytes for spec in dgrp.GENOTYPE_FILE_SPECS} == {
        "input/dgrp2.bed": 230_798_207,
        "input/dgrp2.bim": 148_523_556,
        "input/dgrp2.fam": 5_491,
    }
    assert {spec.member_name: spec.sha256 for spec in dgrp.GENOTYPE_FILE_SPECS} == {
        "input/dgrp2.bed": ("2855e4fab69dde2ed0016d503b87a293c436c423bcda8b1021048d173df43ce7"),
        "input/dgrp2.bim": ("3d1d0e77c90cd135360fbf79113df3d9c94648503c50e0ed1fd54240625909be"),
        "input/dgrp2.fam": ("387d760ae033a3d97f261bef8ea1fc256d26c2528c97698b565219bf3df69a2b"),
    }
    assert EXPECTED_TRAIT_COUNTS == {
        "DI": (165, 1_914),
        "SI": (166, 1_911),
        "HP": (165, 1_920),
        "EDD": (159, 1_779),
        "ESD": (157, 1_753),
        "FS": (158, 1_767),
        "AI": (166, 1_832),
    }


def test_line_outcomes_reject_nonpositive_minimum() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        build_line_outcomes(_phenotypes(), min_individuals=0)


def test_sha256_reads_file_in_binary_mode(tmp_path: Path) -> None:
    path = tmp_path / "source.bin"
    path.write_bytes(b"citrees-dgrp")
    assert sha256(path) == "c4bee03cf993073a25060de49232204f57fa4f7bfec4385b9629b4a5591a44e2"


def test_genotype_archive_rejects_size_and_checksum_mismatches(tmp_path: Path) -> None:
    path = tmp_path / "dgrp2.tar.gz"
    payload = b"synthetic DGRP archive"
    path.write_bytes(payload)
    expected_md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    expected_sha256 = hashlib.sha256(payload).hexdigest()

    assert dgrp.verify_genotype_archive(
        path,
        expected_bytes=len(payload),
        expected_md5=expected_md5,
        expected_sha256=expected_sha256,
    ) == (
        expected_md5,
        expected_sha256,
    )
    with pytest.raises(RuntimeError, match="size mismatch"):
        dgrp.verify_genotype_archive(
            path,
            expected_bytes=len(payload) + 1,
            expected_md5=expected_md5,
            expected_sha256=expected_sha256,
        )
    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        dgrp.verify_genotype_archive(
            path,
            expected_bytes=len(payload),
            expected_md5="0" * 32,
            expected_sha256=expected_sha256,
        )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        dgrp.verify_genotype_archive(
            path,
            expected_bytes=len(payload),
            expected_md5=expected_md5,
            expected_sha256="0" * 64,
        )


def test_genotype_archive_acquisition_verifies_download_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"synthetic pinned archive"
    source_url = "https://example.test/dgrp2.tar.gz"
    calls: list[tuple[str, int]] = []

    def fake_urlopen(url: str, timeout: int) -> io.BytesIO:
        calls.append((url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(dgrp, "GENOTYPE_URL", source_url)
    monkeypatch.setattr(dgrp, "GENOTYPE_ARCHIVE_BYTES", len(payload))
    monkeypatch.setattr(
        dgrp,
        "GENOTYPE_ARCHIVE_MD5",
        hashlib.md5(payload, usedforsecurity=False).hexdigest(),
    )
    monkeypatch.setattr(
        dgrp,
        "GENOTYPE_ARCHIVE_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(dgrp.urllib.request, "urlopen", fake_urlopen)

    destination = dgrp.acquire_genotype_archive(tmp_path)
    assert destination.read_bytes() == payload
    assert calls == [(source_url, 60)]
    assert not destination.with_suffix(destination.suffix + ".part").exists()

    assert dgrp.acquire_genotype_archive(tmp_path) == destination
    assert calls == [(source_url, 60)]


@pytest.mark.parametrize(
    ("acquire", "filename"),
    [
        (dgrp.acquire_phenotype_workbook, dgrp.PHENOTYPE_FILENAME),
        (dgrp.acquire_genotype_archive, dgrp.GENOTYPE_ARCHIVE_FILENAME),
    ],
)
def test_acquisition_cleans_partial_download_after_keyboard_interrupt(
    acquire: Callable[[Path], Path],
    filename: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InterruptedResponse(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            raise KeyboardInterrupt

    monkeypatch.setattr(
        dgrp.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: InterruptedResponse(),
    )

    with pytest.raises(KeyboardInterrupt):
        acquire(tmp_path)

    partial = (tmp_path / filename).with_suffix((tmp_path / filename).suffix + ".part")
    assert not partial.exists()


def test_safe_tar_extraction_accepts_only_exact_regular_members(tmp_path: Path) -> None:
    members = [
        ("input/dgrp2.bed", b"\x6c\x1b\x01\x00"),
        ("input/dgrp2.bim", b"1 variant 0 1 A G\n"),
        ("input/dgrp2.fam", b"line_1 line_1 0 0 2 -9\n"),
    ]
    specs = _genotype_specs(members)
    archive_path = tmp_path / "valid.tar.gz"
    _write_tar(archive_path, members)

    extracted = dgrp._safe_extract_genotype_members(
        archive_path,
        tmp_path / "valid",
        specs,
    )
    assert [path.relative_to(tmp_path / "valid").as_posix() for path in extracted] == [
        name for name, _ in members
    ]
    assert [path.read_bytes() for path in extracted] == [payload for _, payload in members]

    traversal_path = tmp_path / "traversal.tar.gz"
    _write_tar(traversal_path, [("../dgrp2.bed", b"unsafe")])
    with pytest.raises(RuntimeError, match="Unsafe"):
        dgrp._safe_extract_genotype_members(
            traversal_path,
            tmp_path / "traversal",
            specs,
        )

    extra_path = tmp_path / "extra.tar.gz"
    _write_tar(extra_path, [*members, ("input/extra.txt", b"extra")])
    with pytest.raises(RuntimeError, match="member mismatch"):
        dgrp._safe_extract_genotype_members(
            extra_path,
            tmp_path / "extra",
            specs,
        )

    bad_size_specs = (
        dgrp.GenotypeFileSpec(
            member_name=specs[0].member_name,
            size_bytes=specs[0].size_bytes + 1,
            sha256=specs[0].sha256,
        ),
        *specs[1:],
    )
    with pytest.raises(RuntimeError, match="member size mismatch"):
        dgrp._safe_extract_genotype_members(
            archive_path,
            tmp_path / "bad-size",
            bad_size_specs,
        )

    bad_hash_specs = (
        dgrp.GenotypeFileSpec(
            member_name=specs[0].member_name,
            size_bytes=specs[0].size_bytes,
            sha256="0" * 64,
        ),
        *specs[1:],
    )
    retry_destination = tmp_path / "bad-hash"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        dgrp._safe_extract_genotype_members(
            archive_path,
            retry_destination,
            bad_hash_specs,
        )
    assert list(retry_destination.iterdir()) == []
    retried = dgrp._safe_extract_genotype_members(
        archive_path,
        retry_destination,
        specs,
    )
    assert [path.read_bytes() for path in retried] == [payload for _, payload in members]


def test_bed_validation_rejects_wrong_size_and_magic(tmp_path: Path) -> None:
    path = tmp_path / "dgrp2.bed"
    path.write_bytes(dgrp.GENOTYPE_BED_MAGIC + b"\x00" * 5)
    assert dgrp.variant_major_bed_bytes(205, 4_438_427) == 230_798_207
    dgrp.validate_bed(path, sample_count=2, variant_count=5)

    with pytest.raises(RuntimeError, match="size mismatch"):
        dgrp.validate_bed(path, sample_count=2, variant_count=6)

    path.write_bytes(b"\x00\x00\x00" + b"\x00" * 5)
    with pytest.raises(RuntimeError, match="magic mismatch"):
        dgrp.validate_bed(path, sample_count=2, variant_count=5)


def test_bed_decoder_reads_variant_blocks_and_missing_values(tmp_path: Path) -> None:
    path = tmp_path / "dgrp2.bed"
    matrix = np.array(
        [
            [0, 1, 2, -1, 1],
            [2, 0, -1, 1, 2],
            [1, 2, 0, 1, -1],
        ],
        dtype=np.int8,
    )
    code_for_dosage = {0: 0, 1: 2, 2: 3, -1: 1}
    packed = bytearray(dgrp.GENOTYPE_BED_MAGIC)
    for row in matrix:
        encoded = [code_for_dosage[int(value)] for value in row]
        for offset in range(0, len(encoded), 4):
            packed.append(
                sum(code << (2 * index) for index, code in enumerate(encoded[offset : offset + 4]))
            )
    path.write_bytes(bytes(packed))

    observed = dgrp.decode_bed_variants(
        path,
        sample_count=5,
        variant_count=3,
        start_variant=1,
        n_variants=2,
    )
    np.testing.assert_array_equal(observed, matrix[1:])


def test_bed_decoder_rejects_invalid_blocks(tmp_path: Path) -> None:
    path = tmp_path / "dgrp2.bed"
    path.write_bytes(dgrp.GENOTYPE_BED_MAGIC + b"\x00")

    with pytest.raises(ValueError, match="nonnegative"):
        dgrp.decode_bed_variants(
            path,
            sample_count=1,
            variant_count=1,
            start_variant=-1,
            n_variants=1,
        )
    with pytest.raises(ValueError, match="exceeds"):
        dgrp.decode_bed_variants(
            path,
            sample_count=1,
            variant_count=1,
            start_variant=1,
            n_variants=1,
        )


def test_bed_summary_streams_blocks_and_computes_qc_metrics(tmp_path: Path) -> None:
    path = tmp_path / "dgrp2.bed"
    matrix = np.array(
        [
            [0, 1, 2, -1],
            [2, 0, -1, 1],
            [1, 2, 0, 1],
        ],
        dtype=np.int8,
    )
    code_for_dosage = {0: 0, 1: 2, 2: 3, -1: 1}
    packed = bytearray(dgrp.GENOTYPE_BED_MAGIC)
    for row in matrix:
        packed.append(
            sum(code_for_dosage[int(value)] << (2 * index) for index, value in enumerate(row))
        )
    path.write_bytes(bytes(packed))

    observed = dgrp.summarize_bed_variants(
        path,
        sample_count=4,
        variant_count=3,
        block_size=2,
    )
    np.testing.assert_array_equal(observed["n_called"], [3, 3, 4])
    np.testing.assert_allclose(observed["call_rate"], [0.75, 0.75, 1.0])
    np.testing.assert_allclose(observed["allele_frequency"], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(observed["minor_allele_frequency"], [0.5, 0.5, 0.5])


def test_select_qc_pass_variants_applies_named_thresholds() -> None:
    summary = pd.DataFrame(
        {
            "variant_index": [3, 7, 11],
            "n_called": [4, 3, 4],
            "call_rate": [1.0, 0.75, 1.0],
            "allele_frequency": [0.5, 0.5, 0.97],
            "minor_allele_frequency": [0.5, 0.5, 0.03],
        }
    )
    observed = dgrp.select_qc_pass_variants(summary)
    assert observed["variant_index"].tolist() == [3]

    with pytest.raises(ValueError, match="columns differ"):
        dgrp.select_qc_pass_variants(summary.drop(columns="call_rate"))
    with pytest.raises(ValueError, match="min_call_rate"):
        dgrp.select_qc_pass_variants(summary, min_call_rate=0.0)


def test_select_qc_pass_variants_uses_frozen_strict_boundaries() -> None:
    summary = pd.DataFrame(
        {
            "variant_index": [0, 1, 2, 3],
            "n_called": [19, 19, 20, 18],
            "call_rate": [0.95, 0.95, 0.96, 0.90],
            "allele_frequency": [0.04, 0.040001, 0.96, 0.5],
            "minor_allele_frequency": [0.04, 0.040001, 0.04, 0.5],
        }
    )

    observed = dgrp.select_qc_pass_variants(summary)

    assert observed["variant_index"].tolist() == [1]


def test_materialize_filtered_genotypes_preserves_inventory_order(tmp_path: Path) -> None:
    bed_path = tmp_path / "dgrp2.bed"
    output_path = tmp_path / "filtered.npy"
    matrix = np.array(
        [
            [0, 1, 2],
            [2, 0, -1],
            [1, 2, 0],
            [0, -1, 1],
        ],
        dtype=np.int8,
    )
    code_for_dosage = {0: 0, 1: 2, 2: 3, -1: 1}
    payload = bytearray(dgrp.GENOTYPE_BED_MAGIC)
    for row in matrix:
        payload.append(
            sum(code_for_dosage[int(value)] << (2 * index) for index, value in enumerate(row))
        )
    bed_path.write_bytes(bytes(payload))
    inventory = pd.DataFrame({"variant_index": [0, 2, 3]})

    metadata = dgrp.materialize_filtered_genotypes(
        bed_path,
        inventory,
        output_path,
        sample_count=3,
        variant_count=4,
        block_size=2,
    )
    observed = np.load(output_path, mmap_mode="r")
    assert metadata == {"variant_count": 3, "sample_count": 3, "matrix_bytes": 9}
    np.testing.assert_array_equal(observed, matrix[[0, 2, 3]])

    with pytest.raises(ValueError, match="strictly increasing"):
        dgrp.materialize_filtered_genotypes(
            bed_path,
            pd.DataFrame({"variant_index": [2, 1]}),
            tmp_path / "invalid.npy",
            sample_count=3,
            variant_count=4,
        )


def test_build_line_folds_are_deterministic_and_disjoint() -> None:
    lines = [f"line_{index}" for index in range(11)]
    first = dgrp.build_line_folds(lines, n_splits=5, random_state=9)
    second = dgrp.build_line_folds(lines, n_splits=5, random_state=9)
    assert [fold.tolist() for fold in first] == [fold.tolist() for fold in second]
    assert sorted(np.concatenate(first).tolist()) == list(range(len(lines)))
    assert sum(len(fold) for fold in first) == len(lines)


def test_impute_fold_genotypes_uses_training_medians_for_evaluation_rows() -> None:
    genotypes = np.array([[0, -1], [2, 4], [-1, 8]], dtype=np.int8)
    train, evaluation = dgrp.impute_fold_genotypes(genotypes, [0, 1], [2])
    np.testing.assert_array_equal(train, [[0.0, 4.0], [2.0, 4.0]])
    np.testing.assert_array_equal(evaluation, [[1.0, 8.0]])


def test_residualize_fold_outcomes_learns_coefficients_on_training_rows() -> None:
    covariates = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    outcomes = 10.0 + 2.0 * covariates[:, 0] + np.array([1.0, -3.0, 3.0, -1.0, 2.0])
    train_residuals, evaluation_residuals = dgrp.residualize_fold_outcomes(
        outcomes,
        covariates,
        [0, 1, 2, 3],
        [4],
    )
    np.testing.assert_allclose(train_residuals, [1.0, -3.0, 3.0, -1.0])
    np.testing.assert_allclose(evaluation_residuals, [2.0])


def test_collapse_ld_redundant_ranking_keeps_first_of_correlated_features() -> None:
    genotypes = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 2.0, 0.0],
            [2.0, 2.0, 1.0, 1.0],
            [3.0, 3.0, 0.0, 0.0],
        ]
    )
    representatives = dgrp.collapse_ld_redundant_ranking(
        genotypes,
        [1, 0, 2, 3],
        r2_threshold=0.8,
    )
    assert representatives.tolist() == [1, 2, 3]


def test_fold_local_ranking_imputes_training_only_and_returns_full_order() -> None:
    genotypes = np.array(
        [
            [0, 0, 0],
            [1, -1, 1],
            [2, 2, 0],
            [0, 1, 1],
            [2, 2, 2],
        ],
        dtype=np.int8,
    )
    outcomes = np.array([0.0, 1.0, 2.0, 0.1, 1.9])
    ranking = dgrp.rank_features_fold_local(genotypes, outcomes, [0, 1, 2, 3])
    assert ranking.tolist() == [0, 1, 2]

    with pytest.raises(ValueError, match="no observed"):
        dgrp.rank_features_fold_local(
            np.array([[0, -1], [1, -1]], dtype=np.int8),
            np.array([0.0, 1.0]),
            [0, 1],
        )


def test_bim_validation_rejects_duplicate_ids_and_invalid_positions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dgrp2.bim"
    path.write_text(
        "1 variant_1 0 10 A G\n1 variant_2 0 20 C T\n",
        encoding="ascii",
    )
    assert dgrp.validate_bim(path, expected_variants=2) == 2

    path.write_text(
        "1 duplicate 0 10 A G\n1 duplicate 0 20 C T\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="duplicate variant ID"):
        dgrp.validate_bim(path, expected_variants=2)

    path.write_text("1 variant 0 1.5 A G\n", encoding="ascii")
    with pytest.raises(ValueError, match="invalid integer position"):
        dgrp.validate_bim(path, expected_variants=1)


def test_load_bim_metadata_preserves_selected_variant_indices(tmp_path: Path) -> None:
    path = tmp_path / "dgrp2.bim"
    path.write_text(
        "1 variant_1 0 10 A G\n1 variant_2 0 20 C T\n2 variant_3 0 30 G A\n",
        encoding="ascii",
    )
    observed = dgrp.load_bim_metadata(path, [0, 2])
    assert observed.to_dict("records") == [
        {
            "variant_index": 0,
            "chromosome": "1",
            "variant_id": "variant_1",
            "position": 10,
            "allele_a": "A",
            "allele_b": "G",
        },
        {
            "variant_index": 2,
            "chromosome": "2",
            "variant_id": "variant_3",
            "position": 30,
            "allele_a": "G",
            "allele_b": "A",
        },
    ]


def test_fam_validation_requires_matching_line_identities(tmp_path: Path) -> None:
    path = tmp_path / "dgrp2.fam"
    path.write_text(
        "line_21 line_21 0 0 2 -9\nline_42 line_42 0 0 2 -9\n",
        encoding="ascii",
    )
    assert dgrp.validate_fam(path, expected_samples=2) == ("line_21", "line_42")

    path.write_text("line_21 line_22 0 0 2 -9\n", encoding="ascii")
    with pytest.raises(ValueError, match="different FID and IID"):
        dgrp.validate_fam(path, expected_samples=1)

    path.write_text("dgrp21 dgrp21 0 0 2 -9\n", encoding="ascii")
    with pytest.raises(ValueError, match="line_ followed by digits"):
        dgrp.validate_fam(path, expected_samples=1)


def test_line_normalization_and_overlap_retain_genotype_only_lines() -> None:
    assert dgrp.normalize_phenotype_line("dgrp21") == "line_21"
    overlap = dgrp.validate_line_overlap(
        ["dgrp42", "dgrp21", "dgrp21"],
        ["line_7", "line_21", "line_42"],
    )
    assert overlap.phenotype_lines == ("line_21", "line_42")
    assert overlap.genotype_lines == ("line_7", "line_21", "line_42")
    assert overlap.genotype_only_lines == ("line_7",)

    with pytest.raises(RuntimeError, match="line_99"):
        dgrp.validate_line_overlap(
            ["dgrp21", "dgrp99"],
            ["line_21", "line_42"],
        )


def test_genotype_source_receipt_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members = [
        ("input/dgrp2.bed", dgrp.GENOTYPE_BED_MAGIC + b"\x00"),
        ("input/dgrp2.bim", b"1 variant 0 1 A G\n"),
        (
            "input/dgrp2.fam",
            b"line_1 line_1 0 0 2 -9\nline_2 line_2 0 0 2 -9\n",
        ),
    ]
    specs = _genotype_specs(members)
    archive_path = tmp_path / dgrp.GENOTYPE_ARCHIVE_FILENAME
    _write_tar(archive_path, members)
    archive_payload = archive_path.read_bytes()
    archive_md5 = hashlib.md5(archive_payload, usedforsecurity=False).hexdigest()
    archive_sha256 = hashlib.sha256(archive_payload).hexdigest()
    extraction_root = tmp_path / "extracted"
    dgrp._safe_extract_genotype_members(archive_path, extraction_root, specs)
    (extraction_root / "input" / "other-source.sqlite").write_bytes(b"unrelated")

    monkeypatch.setattr(dgrp, "GENOTYPE_FILE_SPECS", specs)
    monkeypatch.setattr(dgrp, "GENOTYPE_ARCHIVE_BYTES", len(archive_payload))
    monkeypatch.setattr(dgrp, "GENOTYPE_ARCHIVE_MD5", archive_md5)
    monkeypatch.setattr(dgrp, "GENOTYPE_ARCHIVE_SHA256", archive_sha256)
    monkeypatch.setattr(dgrp, "EXPECTED_GENOTYPE_SAMPLES", 2)
    monkeypatch.setattr(dgrp, "EXPECTED_GENOTYPE_VARIANTS", 1)
    monkeypatch.setattr(dgrp, "EXPECTED_GENOTYPE_BED_BYTES", 4)
    monkeypatch.setattr(dgrp, "EXPECTED_SOURCE_LINES", 1)
    monkeypatch.setattr(dgrp, "EXPECTED_GENOTYPE_ONLY_LINES", 1)
    observed_md5 = "1" * 32
    observed_sha256 = "2" * 64
    verification_calls: list[Path] = []

    def verify_archive(path: Path) -> tuple[str, str]:
        verification_calls.append(path)
        return observed_md5, observed_sha256

    monkeypatch.setattr(dgrp, "verify_genotype_archive", verify_archive)

    first = dgrp.build_genotype_source_receipt(
        archive_path,
        extraction_root / "input",
        ["dgrp1", "dgrp1"],
    )
    second = dgrp.build_genotype_source_receipt(
        archive_path,
        extraction_root / "input",
        ["dgrp1", "dgrp1"],
    )

    assert first == second
    assert verification_calls == [archive_path, archive_path]
    assert first["source"] == {
        "record_id": 5_582_846,
        "url": dgrp.GENOTYPE_URL,
        "filename": "dgrp2.tar.gz",
        "bytes": len(archive_payload),
        "md5": observed_md5,
        "sha256": observed_sha256,
    }
    assert first["inventory"] == {
        "samples": 2,
        "variants": 1,
        "bed_bytes": 4,
        "bed_magic": "6c1b01",
    }
    assert first["line_overlap"] == {
        "phenotype_source_lines": 1,
        "genotype_lines": 2,
        "shared_lines": 1,
        "genotype_only_lines": 1,
        "genotype_only_ids": ["line_2"],
    }


def test_genotype_extraction_preserves_sqlite_and_rejects_malformed_plink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members = [
        ("input/dgrp2.bed", dgrp.GENOTYPE_BED_MAGIC + b"\x00"),
        ("input/dgrp2.bim", b"1 variant 0 1 A G\n"),
        (
            "input/dgrp2.fam",
            b"line_1 line_1 0 0 2 -9\nline_2 line_2 0 0 2 -9\n",
        ),
    ]
    specs = _genotype_specs(members)
    data_dir = tmp_path / "data"
    input_dir = data_dir / "input"
    input_dir.mkdir(parents=True)
    sqlite_path = input_dir / "Phenosnip.201612.sqlite"
    sqlite_payload = b"preserve this source"
    sqlite_path.write_bytes(sqlite_payload)
    archive_path = data_dir / dgrp.GENOTYPE_ARCHIVE_FILENAME
    _write_tar(archive_path, members)
    archive_payload = archive_path.read_bytes()

    monkeypatch.setattr(dgrp, "GENOTYPE_FILE_SPECS", specs)
    monkeypatch.setattr(dgrp, "GENOTYPE_ARCHIVE_BYTES", len(archive_payload))
    monkeypatch.setattr(
        dgrp,
        "GENOTYPE_ARCHIVE_MD5",
        hashlib.md5(archive_payload, usedforsecurity=False).hexdigest(),
    )
    monkeypatch.setattr(
        dgrp,
        "GENOTYPE_ARCHIVE_SHA256",
        hashlib.sha256(archive_payload).hexdigest(),
    )
    monkeypatch.setattr(dgrp, "EXPECTED_GENOTYPE_SAMPLES", 2)
    monkeypatch.setattr(dgrp, "EXPECTED_GENOTYPE_VARIANTS", 1)

    assert dgrp.extract_genotype_archive(archive_path, data_dir) == input_dir
    assert sqlite_path.read_bytes() == sqlite_payload
    assert {path.name for path in input_dir.iterdir()} == {
        "Phenosnip.201612.sqlite",
        "dgrp2.bed",
        "dgrp2.bim",
        "dgrp2.fam",
    }

    malformed_fam = input_dir / "dgrp2.fam"
    malformed_fam.write_bytes(b"malformed\n")
    with pytest.raises(RuntimeError, match="file size mismatch"):
        dgrp.extract_genotype_archive(archive_path, data_dir)
    assert malformed_fam.read_bytes() == b"malformed\n"
    assert sqlite_path.read_bytes() == sqlite_payload


def test_genotype_extraction_rolls_back_after_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members = [
        ("input/dgrp2.bed", dgrp.GENOTYPE_BED_MAGIC + b"\x00"),
        ("input/dgrp2.bim", b"1 variant 0 1 A G\n"),
        (
            "input/dgrp2.fam",
            b"line_1 line_1 0 0 2 -9\nline_2 line_2 0 0 2 -9\n",
        ),
    ]
    specs = _genotype_specs(members)
    data_dir = tmp_path / "data"
    input_dir = data_dir / "input"
    input_dir.mkdir(parents=True)
    unrelated = input_dir / "Phenosnip.201612.sqlite"
    unrelated.write_bytes(b"preserve this source")
    archive_path = data_dir / dgrp.GENOTYPE_ARCHIVE_FILENAME
    _write_tar(archive_path, members)
    archive_payload = archive_path.read_bytes()

    monkeypatch.setattr(dgrp, "GENOTYPE_FILE_SPECS", specs)
    monkeypatch.setattr(dgrp, "GENOTYPE_ARCHIVE_BYTES", len(archive_payload))
    monkeypatch.setattr(
        dgrp,
        "GENOTYPE_ARCHIVE_MD5",
        hashlib.md5(archive_payload, usedforsecurity=False).hexdigest(),
    )
    monkeypatch.setattr(
        dgrp,
        "GENOTYPE_ARCHIVE_SHA256",
        hashlib.sha256(archive_payload).hexdigest(),
    )
    monkeypatch.setattr(dgrp, "EXPECTED_GENOTYPE_SAMPLES", 2)
    monkeypatch.setattr(dgrp, "EXPECTED_GENOTYPE_VARIANTS", 1)

    real_link = dgrp.os.link
    links = 0

    def interrupt_second_link(source: Path, destination: Path) -> None:
        nonlocal links
        links += 1
        if links == 2:
            raise KeyboardInterrupt
        real_link(source, destination)

    monkeypatch.setattr(dgrp.os, "link", interrupt_second_link)

    with pytest.raises(KeyboardInterrupt):
        dgrp.extract_genotype_archive(archive_path, data_dir)

    assert {path.name for path in input_dir.iterdir()} == {unrelated.name}
    assert unrelated.read_bytes() == b"preserve this source"


def test_main_invokes_genotype_pipeline_and_persists_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    phenotype_source = data_dir / dgrp.PHENOTYPE_FILENAME
    phenotype_source.write_bytes(b"synthetic phenotype source")
    genotype_archive = data_dir / dgrp.GENOTYPE_ARCHIVE_FILENAME
    genotype_archive.write_bytes(b"synthetic genotype source")
    genotype_input = data_dir / "input"
    genotype_input.mkdir()
    frame = _phenotypes()
    expected_provenance = {
        "schema_version": 1,
        "source": {"sha256": "a" * 64},
        "inventory": {"samples": 3, "variants": 2},
        "line_overlap": {
            "phenotype_source_lines": 3,
            "genotype_lines": 3,
            "shared_lines": 3,
            "genotype_only_lines": 0,
            "genotype_only_ids": [],
        },
    }
    calls: list[str] = []

    def acquire_phenotypes(observed_data_dir: Path) -> Path:
        assert observed_data_dir == data_dir
        calls.append("acquire_phenotypes")
        return phenotype_source

    def load_phenotypes(observed_source: Path) -> pd.DataFrame:
        assert observed_source == phenotype_source
        calls.append("load_phenotypes")
        return frame

    def acquire_genotypes(observed_data_dir: Path) -> Path:
        assert observed_data_dir == data_dir
        calls.append("acquire_genotypes")
        return genotype_archive

    def extract_genotypes(observed_archive: Path, observed_data_dir: Path) -> Path:
        assert observed_archive == genotype_archive
        assert observed_data_dir == data_dir
        calls.append("extract_genotypes")
        return genotype_input

    def build_provenance(
        observed_archive: Path,
        observed_input: Path,
        phenotype_lines: pd.Series,
    ) -> dict[str, object]:
        assert observed_archive == genotype_archive
        assert observed_input == genotype_input
        assert set(phenotype_lines) == {"dgrp1", "dgrp2", "dgrp3"}
        calls.append("build_provenance")
        return expected_provenance

    monkeypatch.setattr(dgrp, "acquire_phenotype_workbook", acquire_phenotypes)
    monkeypatch.setattr(dgrp, "load_individual_phenotypes", load_phenotypes)
    monkeypatch.setattr(dgrp, "acquire_genotype_archive", acquire_genotypes)
    monkeypatch.setattr(dgrp, "extract_genotype_archive", extract_genotypes)
    monkeypatch.setattr(dgrp, "build_genotype_source_receipt", build_provenance)
    monkeypatch.setattr(dgrp, "verify_source", lambda path: None)
    monkeypatch.setattr(dgrp, "validate_pinned_line_outcomes", lambda outcomes: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dgrp.py",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    dgrp.main()

    assert calls == [
        "acquire_phenotypes",
        "load_phenotypes",
        "acquire_genotypes",
        "extract_genotypes",
        "build_provenance",
    ]
    receipt = json.loads((output_dir / "receipt.json").read_text(encoding="ascii"))
    assert receipt["schema_version"] == 3
    assert receipt["genotype"] == expected_provenance
    repo_root = Path(dgrp.__file__).resolve().parents[3]
    assert receipt["git_sha"] == dgrp._git_sha(repo_root)
    assert isinstance(receipt["git_dirty"], bool)
    assert receipt["source_sha256"]["paper/jss/replication/dgrp.py"] == sha256(
        Path(dgrp.__file__).resolve()
    )
    assert set(receipt["versions"]) == {"numpy", "openpyxl", "pandas"}
    assert (output_dir / "line_outcomes.parquet").is_file()
    assert (output_dir / "trait_summary.csv").is_file()
    assert capsys.readouterr().out == "Wrote DGRP outcomes for 2 lines.\n"
