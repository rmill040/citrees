"""Regression tests for the paper package surface."""

from __future__ import annotations

import importlib.util
import math
import re
import subprocess
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.paper

ROOT = Path(__file__).resolve().parents[2]
ARXIV_DIR = ROOT / "paper" / "arxiv"


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def _one_line(text: str) -> str:
    return " ".join(text.split())


def test_arxiv_title_block_is_stable_for_reproducible_builds():
    """The manuscript title block should not vary with the current date."""
    text = _read("paper/arxiv/main.tex")
    assert r"\date{\today}" not in text


def test_tables_manifest_lists_only_joss_cited_tables():
    """The tracked table manifest should stay limited to JOSS-cited CSVs."""
    tables_manifest = _read("paper/results/tables/README.md")

    assert "paper_benchmark_method_aggregate.csv" in tables_manifest
    assert "paper_presentation_practical_controls_summary.csv" in tables_manifest
    assert "paper_benchmark_best_configs.csv" not in tables_manifest
    assert "paper_mechanism_" not in tables_manifest


def test_results_readme_demotes_non_packaged_outputs():
    """The results README should demote anything outside the manifest."""
    results_readme = _read("paper/results/README.md")

    assert "Do not add broad analysis tables here" in results_readme


def test_readmes_and_appendices_do_not_reference_stale_appendix_layout():
    """Paper-facing docs should reflect the current appendix structure."""
    appendix_methods = _read("paper/arxiv/appendices/appendix_D_methods.tex")
    arxiv_readme = _read("paper/arxiv/README.md")
    paper_readme = _read("paper/README.md")

    assert "Appendices~A--G" not in appendix_methods
    assert "one appendix per claim" not in arxiv_readme
    assert "rewriting scaffolds" not in paper_readme


def test_citrees_exp_entrypoint_targets_packaged_wheel_sources():
    """The wheel should ship the experiment CLI modules named by its console script."""
    pyproject = tomllib.loads(_read("pyproject.toml"))
    wheel_includes = set(pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["only-include"])

    assert pyproject["project"]["scripts"]["citrees-exp"] == "paper.benchmark.cli.entrypoint:main"
    assert "citrees" in wheel_includes
    assert {
        "paper/benchmark/adapters",
        "paper/benchmark/api",
        "paper/benchmark/cli",
        "paper/benchmark/config",
        "paper/benchmark/experiments",
        "paper/benchmark/infra",
        "paper/benchmark/pipeline",
        "paper/benchmark/utils",
    }.issubset(wheel_includes)
    assert "paper/benchmark" not in wheel_includes
    assert "paper/analysis" not in wheel_includes
    assert "paper/maintenance" not in wheel_includes
    assert "paper/theory" not in wheel_includes


def test_built_wheel_excludes_paper_research_residue(tmp_path):
    """The actual wheel should include the experiment CLI without paper artifacts."""
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    wheels = list(tmp_path.glob("citrees-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())

    assert any(name.startswith("citrees/") for name in names)
    assert any(name.startswith("paper/benchmark/cli/") for name in names)
    assert any(name.startswith("paper/benchmark/pipeline/") for name in names)
    assert not any(name.startswith("paper/analysis/") for name in names)
    assert not any(name.startswith("paper/maintenance/") for name in names)
    assert not any(name.startswith("paper/theory/") for name in names)
    assert not any(name.startswith("paper/data/") for name in names)
    assert not any(name.startswith("paper/results/") for name in names)


def test_generated_paper_artifacts_are_excluded_from_sdist(tmp_path):
    """Generated experiment data/results should not ship in the Python sdist."""
    pyproject = tomllib.loads(_read("pyproject.toml"))

    sdist_exclude = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]
    assert "/paper/arxiv/*.pdf" in sdist_exclude
    assert "/paper/data" in sdist_exclude
    assert "/paper/results" in sdist_exclude

    result = subprocess.run(
        ["uv", "build", "--sdist", "--out-dir", str(tmp_path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    sdists = list(tmp_path.glob("citrees-*.tar.gz"))
    assert len(sdists) == 1
    with tarfile.open(sdists[0]) as sdist:
        names = set(sdist.getnames())

    assert any("/citrees/" in name for name in names)
    assert not any("/paper/data/" in name for name in names)
    assert not any("/paper/results/" in name for name in names)
    assert not any("__pycache__" in name for name in names)


def test_joss_benchmark_claims_match_canonical_result_tables():
    """The JOSS-facing benchmark claims should stay tied to the locked CSVs."""
    method_summary = pd.read_csv(ROOT / "paper/results/tables/paper_benchmark_method_aggregate.csv")
    paper = _one_line(_read("paper/joss/paper.md"))
    cif = method_summary[method_summary["method_base"] == "cif"].set_index("task")

    classification_rank = int(cif.loc["classification", "rank_position"])
    classification_methods = int(
        method_summary[method_summary["task"] == "classification"]["method_base"].nunique()
    )
    regression_rank = int(cif.loc["regression", "rank_position"])
    regression_methods = int(
        method_summary[method_summary["task"] == "regression"]["method_base"].nunique()
    )

    assert classification_rank == 4
    assert int(cif.loc["classification", "n_datasets"]) == 22
    assert classification_methods == 17
    assert regression_rank == 3
    assert int(cif.loc["regression", "n_datasets"]) == 8
    assert regression_methods == 18
    assert f"fourth among {classification_methods} classification methods" in paper
    assert f"third among {regression_methods} regression methods" in paper


def test_joss_runtime_claim_matches_practical_controls_summary():
    """The adaptive-stopping runtime sentence should match the locked summary table."""
    controls = pd.read_csv(
        ROOT / "paper/results/tables/paper_presentation_practical_controls_summary.csv"
    )
    paper = _one_line(_read("paper/joss/paper.md"))
    rows = controls[
        (controls["family"] == "mirrored")
        & (controls["reference_variant"] == "cif_default")
        & (controls["variant"] == "cif_no_adaptive")
    ]

    assert set(rows["task"]) == {"classification", "regression"}
    assert set(rows["dataset_group"]) == {"real", "synthetic"}
    assert rows["runtime_ratio_vs_default"].min() >= 3.95
    assert rows["runtime_ratio_vs_default"].max() <= 8.45
    assert rows["downstream_delta_vs_default"].abs().max() <= 0.006
    runtime_min = rows["runtime_ratio_vs_default"].min()
    runtime_max = rows["runtime_ratio_vs_default"].max()
    score_delta_max = rows["downstream_delta_vs_default"].abs().max()
    score_delta_bound = math.ceil(score_delta_max * 1000) / 1000
    assert f"{runtime_min:.1f}--{runtime_max:.1f} times slower" in paper
    assert f"{score_delta_bound:.3f}" in paper


def _load_arxiv_bundle_module():
    module_path = ROOT / "paper" / "analysis" / "build_arxiv_source_bundle.py"
    spec = importlib.util.spec_from_file_location("build_arxiv_source_bundle", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arxiv_bundle_keeps_referenced_figures_local():
    """The manuscript should not depend on paper/results for its figures."""
    main_tex = _read("paper/arxiv/main.tex")
    bundler = _load_arxiv_bundle_module()
    referenced = {
        path.relative_to(ARXIV_DIR).as_posix() for path in bundler.collect_referenced_figures()
    }

    assert r"\graphicspath{{figures/}}" in main_tex
    for relpath in referenced:
        assert (ARXIV_DIR / relpath).exists(), f"Missing local arXiv figure {relpath}"


def test_arxiv_source_bundle_membership_includes_bibliography_and_excludes_junk():
    """The deterministic source bundle should include BibTeX and avoid build products."""
    bundler = _load_arxiv_bundle_module()
    members = {path.relative_to(ARXIV_DIR).as_posix() for path in bundler.bundle_members()}
    referenced = {
        path.relative_to(ARXIV_DIR).as_posix() for path in bundler.collect_referenced_figures()
    }
    bundled_figures = {member for member in members if member.startswith("figures/")}

    assert "main.tex" in members
    assert "references.bib" in members
    assert "main.bbl" not in members
    assert "main.bbl" not in bundler.STATIC_FILES
    assert bundled_figures == referenced
    assert all(not member.startswith("scratch/") for member in members)
    assert all(not member.startswith("build/") for member in members)


def test_arxiv_referenced_figures_are_tracked_by_git():
    """A committed manuscript should not reference untracked local-only figures."""
    bundler = _load_arxiv_bundle_module()
    referenced = [
        path.relative_to(ROOT).as_posix() for path in bundler.collect_referenced_figures()
    ]

    for relpath in referenced:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relpath],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        assert result.returncode == 0, f"{relpath} is referenced by TeX but is not tracked"


def test_arxiv_figure_writes_are_explicit_opt_in():
    """Active figure builders should not mutate frozen arXiv figure copies by default."""
    old_dual_write = "for out_dir in (FIGURES_DIR, ARXIV_FIGURES_DIR)"
    for path in (ROOT / "paper/analysis").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert old_dual_write not in text, (
            f"{path.relative_to(ROOT)} still writes arXiv figures unconditionally"
        )
        if "ARXIV_FIGURES_DIR" in text:
            assert "add_write_arxiv_argument(parser)" in text, (
                f"{path.relative_to(ROOT)} references arXiv figures without the opt-in flag helper"
            )
            assert "figure_output_dirs(" in text, (
                f"{path.relative_to(ROOT)} should route figure writes through the output helper"
            )


def test_existing_arxiv_source_zip_is_not_stale():
    """If a local source zip exists, it should exactly match the current source inputs."""
    archive_path = ARXIV_DIR / "build" / "citrees-arxiv-source.zip"
    if not archive_path.exists():
        pytest.skip("No local arXiv source zip to check")

    bundler = _load_arxiv_bundle_module()
    expected_paths = bundler.bundle_members()
    expected_members = {path.relative_to(ARXIV_DIR).as_posix() for path in expected_paths}
    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        assert members == expected_members
        for path in expected_paths:
            relpath = path.relative_to(ARXIV_DIR).as_posix()
            assert archive.read(relpath) == path.read_bytes(), (
                f"{relpath} is stale in the local arXiv source zip"
            )


def test_paper_markdown_does_not_reference_missing_scripts():
    """Paper markdown should not point at deleted helper scripts."""
    pattern = re.compile(
        r"paper/(?:analysis|benchmark|data_generation|maintenance|theory)/[A-Za-z0-9_./-]+\.py"
    )

    for path in (ROOT / "paper").rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for relpath in pattern.findall(text):
            assert (ROOT / relpath).exists(), (
                f"{path.relative_to(ROOT)} references missing script {relpath}"
            )


def test_live_paper_markdown_does_not_reference_missing_artifacts():
    """Live paper markdown should not point at deleted paper artifacts."""
    pattern = re.compile(
        r"paper/(?:analysis|benchmark|data_generation|maintenance|results|theory)/[A-Za-z0-9_./-]+\.(?:csv|json|md|parquet|png|py|tex)"
    )
    intentionally_untracked = {
        "paper/results/tables/cif_mechanism_ablation_metrics_flat.csv",
    }

    for path in (ROOT / "paper").rglob("*.md"):
        if ARXIV_DIR in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for relpath in pattern.findall(text):
            if relpath in intentionally_untracked:
                continue
            assert (ROOT / relpath).exists(), (
                f"{path.relative_to(ROOT)} references missing artifact {relpath}"
            )
