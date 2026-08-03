"""Tests for the JSS DGRP phenotype preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
