"""Tests for the isolated JSS runtime and peak-memory analysis."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from paper.jss.replication import performance
from paper.jss.replication.performance import (
    PERFORMANCE_RAW_SCHEMA,
    PERFORMANCE_SUMMARY_SCHEMA,
    PerformanceCell,
    PerformanceSettings,
    _cell_from_payload,
    _peak_rss_bytes,
    _run_cell_subprocess,
    _settings,
    build_performance_grid,
    run_performance,
    summarize_performance,
    validate_performance_results,
    write_results,
)

pytestmark = pytest.mark.paper

R_AVAILABLE = shutil.which("Rscript") is not None and importlib.util.find_spec("rpy2") is not None


def _fake_runner(cell: PerformanceCell, timeout_seconds: int) -> dict[str, object]:
    assert timeout_seconds > 0
    method_scale = {"citrees": 3.0, "partykit": 2.0, "sklearn": 1.0}[cell.method]
    elapsed = method_scale * (cell.repeat + 1) * (cell.n_samples / 100.0)
    baseline = 100_000_000 + cell.data_seed % 10_000
    increment = int(method_scale * cell.n_features * 1_000)
    return {
        **asdict(cell),
        "worker_pid": int(cell.cell_id[:8], 16) + 1,
        "input_sha256": hashlib.sha256(str(cell.data_seed).encode("ascii")).hexdigest(),
        "elapsed_seconds": elapsed,
        "baseline_peak_rss_bytes": baseline,
        "peak_rss_bytes": baseline + increment,
        "incremental_peak_rss_bytes": increment,
        "fit_result_size": cell.n_estimators,
    }


def _smoke_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    return run_performance("smoke", base_seed=7, runner=_fake_runner)


def test_profiles_define_explicit_controlled_workloads() -> None:
    assert _settings("smoke") == PerformanceSettings(
        repeats=1,
        baseline_n_samples=96,
        baseline_n_features=6,
        baseline_n_resamples=39,
        baseline_n_estimators=3,
        sample_sizes=(),
        predictor_counts=(),
        permutation_budgets=(),
        forest_sizes=(),
        cell_timeout_seconds=600,
    )
    assert _settings("quick").repeats == 2
    assert _settings("quick").sample_sizes == (250, 1_000)
    assert _settings("full").repeats == 10
    assert _settings("full").permutation_budgets == (99, 9_999)
    with pytest.raises(ValueError, match="unknown performance profile"):
        _settings("unknown")  # type: ignore[arg-type]


def test_grid_is_exact_unique_and_paired_across_methods() -> None:
    smoke = build_performance_grid("smoke", base_seed=7)
    quick = build_performance_grid("quick", base_seed=7)
    full = build_performance_grid("full", base_seed=7)

    assert len(smoke) == 20
    assert len(quick) == 192
    assert len(full) == 960
    assert len({cell.cell_id for cell in full}) == len(full)
    assert {cell.method for cell in smoke} == {"citrees", "partykit", "sklearn"}
    assert {cell.model_family for cell in smoke} == {"tree", "forest"}
    assert {cell.task for cell in smoke} == {"classification", "regression"}

    paired: dict[tuple[str, int, int, int], set[tuple[int, int]]] = {}
    for cell in quick:
        key = (cell.task, cell.n_samples, cell.n_features, cell.repeat)
        paired.setdefault(key, set()).add((cell.data_seed, cell.model_seed))
    assert all(len(seeds) == 1 for seeds in paired.values())


def test_worker_payload_identity_rejects_any_cell_mutation() -> None:
    cell = build_performance_grid("smoke", base_seed=7)[0]
    assert _cell_from_payload(asdict(cell)) == cell

    corrupted = asdict(cell)
    corrupted["n_samples"] = cell.n_samples + 1
    with pytest.raises(ValueError, match="identity does not match"):
        _cell_from_payload(corrupted)

    missing = asdict(cell)
    missing.pop("selector")
    with pytest.raises(ValueError, match="worker cell differs"):
        _cell_from_payload(missing)


def test_peak_rss_units_are_explicit_and_platform_normalized() -> None:
    assert _peak_rss_bytes(2_048, system="Darwin") == 2_048
    assert _peak_rss_bytes(2_048, system="Linux") == 2_097_152
    with pytest.raises(ValueError, match="nonnegative"):
        _peak_rss_bytes(-1, system="Linux")
    with pytest.raises(RuntimeError, match="unsupported"):
        _peak_rss_bytes(1, system="Windows")


def test_fake_run_recomputes_complete_summary_and_reference_ratios() -> None:
    raw, summary = _smoke_results()

    assert tuple(raw.columns) == PERFORMANCE_RAW_SCHEMA
    assert tuple(summary.columns) == PERFORMANCE_SUMMARY_SCHEMA
    assert len(raw) == 20
    assert len(summary) == 20
    assert summary["elapsed_ratio_to_reference"].eq(1.0).all()
    assert summary["peak_rss_ratio_to_reference"].eq(1.0).all()
    validate_performance_results(raw, summary, profile="smoke", base_seed=7)
    pd.testing.assert_frame_equal(summary, summarize_performance(raw))


@pytest.mark.parametrize(
    ("table_name", "column", "value", "message"),
    [
        ("raw", "elapsed_seconds", 0.0, "elapsed times"),
        ("raw", "incremental_peak_rss_bytes", 1, "incremental peak RSS"),
        ("raw", "input_sha256", "invalid", "input hashes"),
        ("summary", "median_elapsed_seconds", 999.0, "differs from"),
    ],
)
def test_validation_rejects_corrupted_measurements(
    table_name: str,
    column: str,
    value: object,
    message: str,
) -> None:
    raw, summary = _smoke_results()
    raw = raw.copy(deep=True)
    summary = summary.copy(deep=True)
    target = raw if table_name == "raw" else summary
    target.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        validate_performance_results(raw, summary, profile="smoke", base_seed=7)


def test_writer_records_controls_sources_and_artifact_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, summary = _smoke_results()
    monkeypatch.setattr(
        performance,
        "_r_environment",
        lambda: {"r": "R fixture", "partykit": "partykit fixture"},
    )
    write_results(
        raw,
        summary,
        tmp_path,
        profile="smoke",
        base_seed=7,
        elapsed_seconds=1.25,
    )

    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="ascii"))
    assert receipt["analysis"] == "performance"
    assert receipt["schema_version"] == 2
    assert receipt["profile"] == "smoke"
    assert receipt["base_seed"] == 7
    assert receipt["controls"]["process_isolation"] == ("one_fresh_process_per_measured_cell")
    assert receipt["controls"]["thread_environment"] == performance.THREAD_ENVIRONMENT
    assert receipt["tables"]["performance_raw"]["rows"] == 20
    assert "paper/jss/replication/performance.py" in receipt["source_sha256"]
    assert "paper/benchmark/pipeline/r_methods.py" in receipt["source_sha256"]
    assert "citrees/_tree.py" in receipt["source_sha256"]
    assert "uv.lock" in receipt["source_sha256"]
    assert set(receipt["artifacts"]) == {
        "performance_raw.parquet",
        "performance_summary.parquet",
        "performance_summary.csv",
    }
    for artifact, metadata in receipt["artifacts"].items():
        path = tmp_path / artifact
        assert metadata["bytes"] == path.stat().st_size
        assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_sklearn_cells_run_in_fresh_processes() -> None:
    cell = next(
        cell
        for cell in build_performance_grid("smoke", base_seed=7)
        if cell.method == "sklearn"
        and cell.task == "classification"
        and cell.model_family == "tree"
    )
    first = _run_cell_subprocess(cell, 120)
    second = _run_cell_subprocess(cell, 120)

    assert first["worker_pid"] != second["worker_pid"]
    assert first["input_sha256"] == second["input_sha256"]
    assert first["fit_result_size"] >= 1
    assert second["fit_result_size"] >= 1
    assert first["peak_rss_bytes"] >= first["baseline_peak_rss_bytes"]
    assert second["peak_rss_bytes"] >= second["baseline_peak_rss_bytes"]


@pytest.mark.skipif(not R_AVAILABLE, reason="R and rpy2 are required")
def test_partykit_fit_only_cell_runs_in_an_isolated_process() -> None:
    cell = next(
        cell
        for cell in build_performance_grid("smoke", base_seed=7)
        if cell.method == "partykit" and cell.task == "regression" and cell.model_family == "tree"
    )
    result = _run_cell_subprocess(cell, 180)

    assert result["input_sha256"]
    assert result["elapsed_seconds"] > 0.0
    assert result["fit_result_size"] == 1
    assert result["peak_rss_bytes"] >= result["baseline_peak_rss_bytes"]


def test_summary_rejects_missing_method_reference() -> None:
    raw, _summary = _smoke_results()
    without_reference = raw.copy(deep=True)
    target = (
        raw["method"].eq("sklearn")
        & raw["task"].eq("classification")
        & raw["model_family"].eq("tree")
    )
    without_reference.loc[target, "axis"] = "sample_size"
    without_reference.loc[target, "axis_value"] = "96"
    with pytest.raises(ValueError, match="missing a method reference"):
        summarize_performance(without_reference)


def test_validation_rejects_changed_cell_specification_and_reused_process() -> None:
    raw, summary = _smoke_results()

    changed = raw.copy(deep=True)
    changed.loc[0, "n_samples"] = int(changed.loc[0, "n_samples"]) + 1
    with pytest.raises(ValueError, match="cell specification differs"):
        validate_performance_results(changed, summary, profile="smoke", base_seed=7)

    reused = raw.copy(deep=True)
    reused.loc[1, "worker_pid"] = reused.loc[0, "worker_pid"]
    with pytest.raises(ValueError, match="distinct worker processes"):
        validate_performance_results(reused, summary, profile="smoke", base_seed=7)
