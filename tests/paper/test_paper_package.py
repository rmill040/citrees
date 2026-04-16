"""Regression tests for the paper package surface and authority docs."""

from __future__ import annotations

from pathlib import Path
import re

import pytest

pytestmark = pytest.mark.paper

ROOT = Path(__file__).resolve().parents[2]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


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
    assert "paper_heterogeneity_cif_pairwise_breadth.csv" in finalization
    assert "calibration_summary.csv" in finalization
    assert "appendix/supporting-only" in finalization
    assert "refreshed fixed-`B` Stage A calibration" in finalization
    assert "paper_benchmark_best_configs.csv" in tables_manifest
    assert "paper_benchmark_selected_config_details.csv" in tables_manifest
    assert "paper_benchmark_fixed_panel_aggregate.csv" in tables_manifest
    assert "paper_benchmark_fixed_panel_membership.csv" in tables_manifest
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
    assert "treat anything outside those locked outputs as exploratory or historical by" in experiments
    assert "Exploratory artifacts should not drive manuscript claims" in results_readme


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


def test_arxiv_bundle_keeps_main_text_figures_local():
    """The manuscript should not depend on paper/results for its main figures."""
    main_tex = _read("paper/arxiv/main.tex")

    assert r"\graphicspath{{figures/}}" in main_tex
    for relpath in (
        "paper/arxiv/figures/k_trajectory.png",
        "paper/arxiv/figures/synthetic_topk_focus_curves.png",
        "paper/arxiv/figures/paper_mechanism_grid_forest_classification_feature_counts_p1000_i2_1000trees.png",
    ):
        assert (ROOT / relpath).exists(), f"Missing local arXiv figure {relpath}"


def test_paper_markdown_does_not_reference_missing_scripts():
    """Paper markdown should not point at deleted helper scripts."""
    pattern = re.compile(r"paper/scripts/[A-Za-z0-9_./-]+\.py")

    for path in (ROOT / "paper").rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for relpath in pattern.findall(text):
            assert (ROOT / relpath).exists(), f"{path.relative_to(ROOT)} references missing script {relpath}"
