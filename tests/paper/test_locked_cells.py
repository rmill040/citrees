"""The locked-cell list names exactly the RDC cells that terminate EC2 c6a hosts."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

CONFIG = Path(__file__).resolve().parents[2] / "paper" / "benchmark" / "config"

pytestmark = pytest.mark.paper


def _rows(name: str) -> list[dict[str, str]]:
    with (CONFIG / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_locked_cells_cover_hard_lock_exclusions_and_censored_extension_cells() -> None:
    locked = {
        (r["task"], r["dataset"], r["method_id"], r["seed"]) for r in _rows("locked_cells.csv")
    }
    hard_lock = {
        (r["task"], r["dataset"], r["method_id"], r["seed"])
        for r in _rows("maxt_extension_exclusions.csv")
        if r["reason"] == "rdc_host_hard_lock"
    }
    censored = {
        ("classification", dataset, "cif_maxt__19b65c92023416b0", str(seed))
        for dataset, seed in [
            ("gisette", 3),
            ("isolet", 0),
            ("isolet", 1),
            ("isolet", 3),
            ("isolet", 4),
            ("letter", 2),
            ("letter", 3),
        ]
    }
    assert hard_lock <= locked
    assert censored <= locked
    assert locked == hard_lock | censored
    assert len(locked) == 28


def test_locked_cells_are_all_rdc_classification_cells_with_one_reason() -> None:
    rows = _rows("locked_cells.csv")
    assert {r["task"] for r in rows} == {"classification"}
    assert {r["dataset"] for r in rows} == {"gisette", "isolet", "letter"}
    assert {r["reason"] for r in rows} == {"ec2_host_fault_confirmed_2026-09-13"}
