"""Regression tests for the paper package surface and authority docs."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import tomllib
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.paper

ROOT = Path(__file__).resolve().parents[2]
ARXIV_DIR = ROOT / "paper" / "arxiv"


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def _one_line(text: str) -> str:
    return " ".join(text.split())


def test_key_readmes_do_not_reference_removed_markdown_docs():
    """Current paper READMEs should not point at removed support-doc files."""
    stale_names = (
        "drafts.md",
        "figures-plan.md",
        "analysis-lockdown-plan.md",
        "claims-index.md",
        "paper-verification-checklist.md",
        "paper-verification-log.md",
    )
    for relpath in (
        "paper/README.md",
        "paper/arxiv/README.md",
        "paper/results/README.md",
        "paper/results/tables/README.md",
    ):
        text = _read(relpath)
        for stale_name in stale_names:
            assert stale_name not in text, f"{relpath} still references removed doc {stale_name}"


def test_canonical_manifests_do_not_use_broad_wildcards_for_closed_table_families():
    """Closed paper-facing table manifests should enumerate specific files."""
    forbidden_patterns = (
        "paper_benchmark_*.csv",
        "paper_high_p_*.csv",
        "paper_heterogeneity_*.csv",
        "paper_mechanism_*.csv",
        "top_ranking_*.csv",
        "synthetic_topk_*.csv",
    )
    for relpath in ("paper/README.md", "paper/results/tables/README.md"):
        text = _read(relpath)
        for pattern in forbidden_patterns:
            assert pattern not in text, f"{relpath} should not canonize wildcard family {pattern}"


def test_arxiv_title_block_is_stable_for_reproducible_builds():
    """The manuscript title block should not vary with the current date."""
    text = _read("paper/arxiv/main.tex")
    assert r"\date{\today}" not in text


def test_results_authority_docs_keep_breadth_canonical_and_calibration_supporting_only():
    """Authority docs should preserve the current heterogeneity/calibration freeze rules."""
    finalization = _read("paper/docs/results-finalization.md")
    tables_manifest = _read("paper/results/tables/README.md")

    assert "paper_benchmark_best_configs.csv" in finalization
    assert "paper_benchmark_selected_config_details.csv" in finalization
    assert "paper_benchmark_fixed_panel_aggregate.csv" in finalization
    assert "paper_benchmark_fixed_panel_membership.csv" in finalization
    assert "cif_mechanism_ablation_pairwise_vs_default.csv" in finalization
    assert "k_trajectory_ranks.csv" in finalization
    assert "regression_k_trajectory_ranks.csv" in finalization
    assert "paper_heterogeneity_cif_pairwise_breadth.csv" in finalization
    assert "calibration_summary.csv" in finalization
    assert "appendix/supporting-only" in finalization
    assert "refreshed fixed-`B` Stage A calibration" in finalization
    assert "paper_benchmark_best_configs.csv" in tables_manifest
    assert "paper_benchmark_selected_config_details.csv" in tables_manifest
    assert "paper_benchmark_fixed_panel_aggregate.csv" in tables_manifest
    assert "paper_benchmark_fixed_panel_membership.csv" in tables_manifest
    assert "cif_mechanism_ablation_pairwise_vs_default.csv" in tables_manifest
    assert "k_trajectory_ranks.csv" in tables_manifest
    assert "regression_k_trajectory_ranks.csv" in tables_manifest
    assert "paper_heterogeneity_cif_pairwise_breadth.csv" in tables_manifest
    assert "paper_heterogeneity_cif_pairwise_summary.csv" in tables_manifest
    assert "superseded by `paper_heterogeneity_cif_pairwise_breadth.csv`" in tables_manifest
    assert "Supporting-Only (Refreshed)" in tables_manifest


def test_experiments_doc_demotes_non_packaged_outputs():
    """The experiment runbook should demote anything outside the packaged outputs."""
    experiments = _read("paper/docs/experiments.md")
    results_readme = _read("paper/results/README.md")

    assert "Supporting Studies Used In The Paper" in experiments
    assert "mirrored practical-knob ablations" in experiments
    assert "threshold-search ablation" in experiments
    assert (
        "treat anything outside those locked outputs as exploratory or historical by"
        in _one_line(experiments)
    )
    assert "Exploratory artifacts should not drive manuscript claims" in results_readme


def test_documented_rebuild_path_includes_main_text_trajectory_generator():
    """The closed rebuild path should regenerate trajectory figures and rank tables."""
    generator = "paper/scripts/analysis/fig_benchmark_k_trajectory.py"

    assert generator in _read("paper/README.md")
    assert generator in _read("paper/docs/experiments.md")


def test_readmes_and_appendices_do_not_reference_stale_appendix_layout():
    """Paper-facing docs should reflect the current appendix structure."""
    appendix_methods = _read("paper/arxiv/appendices/appendix_D_methods.tex")
    arxiv_readme = _read("paper/arxiv/README.md")
    paper_readme = _read("paper/README.md")

    assert "Appendices~A--G" not in appendix_methods
    assert "one appendix per claim" not in arxiv_readme
    assert "rewriting scaffolds" not in paper_readme


def test_experiments_doc_points_to_current_operational_entrypoints():
    """The experiment runbook should point to the current infra entrypoints."""
    experiments = _read("paper/docs/experiments.md")
    infrastructure = _read("paper/docs/infrastructure.md")

    assert "This file intentionally omits the old EC2 launcher cookbook." in experiments
    assert "citrees-exp --help" in experiments
    assert "paper/scripts/infra/" in experiments
    assert "paper/docs/infrastructure.md" in experiments
    assert "citrees-exp infra launch-api" in infrastructure
    assert "citrees-exp infra launch-workers --count 5" in infrastructure


def test_citrees_exp_entrypoint_targets_packaged_wheel_sources():
    """The wheel should ship the experiment CLI modules named by its console script."""
    pyproject = tomllib.loads(_read("pyproject.toml"))

    assert pyproject["project"]["scripts"]["citrees-exp"] == "paper.scripts.cli.entrypoint:main"
    assert "citrees" in pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["only-include"]
    assert (
        "paper/scripts" in pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["only-include"]
    )


def test_generated_paper_artifacts_are_excluded_from_sdist():
    """Generated experiment data/results should not ship in the Python sdist."""
    pyproject = tomllib.loads(_read("pyproject.toml"))

    sdist_exclude = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]
    assert "/paper/data" in sdist_exclude
    assert "/paper/results" in sdist_exclude


def _load_arxiv_bundle_module():
    module_path = ROOT / "paper" / "scripts" / "analysis" / "build_arxiv_source_bundle.py"
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
    pattern = re.compile(r"paper/scripts/[A-Za-z0-9_./-]+\.py")

    for path in (ROOT / "paper").rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for relpath in pattern.findall(text):
            assert (ROOT / relpath).exists(), (
                f"{path.relative_to(ROOT)} references missing script {relpath}"
            )
