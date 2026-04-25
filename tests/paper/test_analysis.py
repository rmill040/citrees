"""Tests for paper/scripts/analysis/* statistical functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paper.scripts.analysis.analyze_synthetic_ground_truth import dataset_type_from_config
from paper.scripts.analysis.benchmark_common import select_best_task_configs
from paper.scripts.analysis.build_benchmark_package_tables import (
    _build_benchmark_spread,
    _build_complete_case_membership,
    _build_config_selection_audit,
    _build_fixed_panel_aggregate,
    _build_fixed_panel_membership,
    _build_pairwise_aggregate,
)
from paper.scripts.analysis.build_high_p_saturation_tables import (
    build_cif_best_observed_k_summary,
    build_delta_vs_endpoint_cells,
    build_endpoint_method_presence,
)
from paper.scripts.analysis.build_knob_ablation_summary_tables import build_knob_ablation_summary
from paper.scripts.analysis.build_manuscript_summary_tables import (
    build_benchmark_presentation_summary,
    build_practical_controls_presentation_summary,
)
from paper.scripts.analysis.build_mechanism_summary_tables import (
    FixedDesignSpec,
    build_single_tree_split_model,
    collect_cit_split_features,
    count_split_features_from_tree,
    informative_coverage_probability,
    make_fixed_dataset,
    resolve_max_features_count,
)
from paper.scripts.analysis.build_synthetic_topk_tables import (
    _build_row_metrics as build_topk_row_metrics,
)
from paper.scripts.analysis.build_synthetic_topk_tables import (
    _summarize_curve_over_k as summarize_topk_curve_over_k,
)
from paper.scripts.analysis.build_threshold_ablation_summary_tables import (
    build_threshold_ablation_summary,
)
from paper.scripts.analysis.build_top_ranking_tables import (
    _summarize_curve_over_k as summarize_top_ranking_curve_over_k,
)
from paper.scripts.analysis.config_resolution import resolve_method_config_details
from paper.scripts.analysis.run_mechanism_dimension_sweep_shard import (
    _build_specs as build_mechanism_shard_specs,
)
from paper.scripts.analysis.stats import (
    bootstrap_ci,
    cohens_d,
    compute_noise_selection_rate,
    friedman_test,
    interpret_cohens_d,
    pairwise_wilcoxon_holm,
)

pytestmark = pytest.mark.paper


class TestConfigResolution:
    """Tests for benchmark config expansion helpers."""

    def test_resolves_selected_cit_cif_configs(self):
        """Known benchmark-selected CIT/CIF IDs should expand to their concrete params."""
        best = pd.DataFrame(
            [
                {
                    "task": "classification",
                    "metric": "balanced_accuracy",
                    "method_base": "cit",
                    "method_id": "cit__bc8d0ddbb26c15b5",
                    "task_global_mean_metric": 0.77,
                },
                {
                    "task": "regression",
                    "metric": "r2",
                    "method_base": "cif",
                    "method_id": "cif__a01a3672a449d400",
                    "task_global_mean_metric": 0.29,
                },
            ]
        )
        resolved = resolve_method_config_details(best)

        clf_cit = resolved[resolved["method_id"] == "cit__bc8d0ddbb26c15b5"].iloc[0]
        reg_cif = resolved[resolved["method_id"] == "cif__a01a3672a449d400"].iloc[0]

        assert bool(clf_cit["config_resolved"])
        assert clf_cit["selector"] == "mc"
        assert not bool(clf_cit["honesty"])

        assert bool(reg_cif["config_resolved"])
        assert reg_cif["selector"] == "pc"
        assert bool(reg_cif["honesty"])


class TestBenchmarkCommon:
    """Tests for benchmark-contract helper functions."""

    def test_select_best_task_configs_uses_cell_means_not_raw_row_weighting(self):
        """Uneven raw-row coverage should not change the selected config."""
        df = pd.DataFrame(
            [
                # Config A wins on a raw-row mean because its strong cell is
                # duplicated, but loses on the paper contract's cell-mean rule.
                {
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "dataset": "d1",
                    "downstream_model": "lr",
                    "k": 5,
                    "balanced_accuracy": 0.95,
                },
                {
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "dataset": "d1",
                    "downstream_model": "lr",
                    "k": 5,
                    "balanced_accuracy": 0.95,
                },
                {
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "dataset": "d2",
                    "downstream_model": "lr",
                    "k": 5,
                    "balanced_accuracy": 0.10,
                },
                # Config B should be selected after averaging within cells.
                {
                    "method_base": "cif",
                    "method_id": "cif__b",
                    "dataset": "d1",
                    "downstream_model": "lr",
                    "k": 5,
                    "balanced_accuracy": 0.70,
                },
                {
                    "method_base": "cif",
                    "method_id": "cif__b",
                    "dataset": "d2",
                    "downstream_model": "lr",
                    "k": 5,
                    "balanced_accuracy": 0.70,
                },
                # A second method_base ensures the helper still behaves like a
                # per-family selector.
                {
                    "method_base": "rf",
                    "method_id": "rf__a",
                    "dataset": "d1",
                    "downstream_model": "lr",
                    "k": 5,
                    "balanced_accuracy": 0.80,
                },
                {
                    "method_base": "rf",
                    "method_id": "rf__a",
                    "dataset": "d2",
                    "downstream_model": "lr",
                    "k": 5,
                    "balanced_accuracy": 0.80,
                },
            ]
        )

        _, best = select_best_task_configs(df, "balanced_accuracy")
        selected = dict(zip(best["method_base"], best["method_id"], strict=False))

        assert selected["cif"] == "cif__b"
        assert selected["rf"] == "rf__a"

    def test_build_complete_case_membership_marks_missing_methods(self):
        """The trajectory membership surface should expose incomplete cells explicitly."""
        scores = pd.DataFrame(
            [
                {"downstream_model": "lr", "k": 5, "dataset": "d1", "method_base": "cif"},
                {"downstream_model": "lr", "k": 5, "dataset": "d1", "method_base": "rf"},
                {"downstream_model": "lr", "k": 5, "dataset": "d2", "method_base": "cif"},
            ]
        )

        membership = _build_complete_case_membership(scores, "classification")
        d1 = membership[membership["dataset"] == "d1"].iloc[0]
        d2 = membership[membership["dataset"] == "d2"].iloc[0]

        assert d1["expected_methods"] == 2
        assert bool(d1["is_complete_case"])
        assert d2["n_methods_present"] == 1
        assert not bool(d2["is_complete_case"])

    def test_build_benchmark_spread_summarizes_per_k_and_pooled_cells(self):
        """The benchmark spread surface should expose standard-k and pooled dispersion."""
        scores = pd.DataFrame(
            [
                {
                    "dataset": "d1",
                    "downstream_model": "lr",
                    "k": 5,
                    "method_base": "cif",
                    "dataset_mean_score": 0.8,
                },
                {
                    "dataset": "d1",
                    "downstream_model": "lr",
                    "k": 5,
                    "method_base": "rf",
                    "dataset_mean_score": 0.6,
                },
                {
                    "dataset": "d2",
                    "downstream_model": "lr",
                    "k": 5,
                    "method_base": "cif",
                    "dataset_mean_score": 0.7,
                },
                {
                    "dataset": "d2",
                    "downstream_model": "lr",
                    "k": 5,
                    "method_base": "rf",
                    "dataset_mean_score": 0.5,
                },
                {
                    "dataset": "d1",
                    "downstream_model": "lr",
                    "k": 10,
                    "method_base": "cif",
                    "dataset_mean_score": 0.9,
                },
                {
                    "dataset": "d1",
                    "downstream_model": "lr",
                    "k": 10,
                    "method_base": "rf",
                    "dataset_mean_score": 0.4,
                },
                {
                    "dataset": "d2",
                    "downstream_model": "lr",
                    "k": 10,
                    "method_base": "cif",
                    "dataset_mean_score": 0.8,
                },
                {
                    "dataset": "d2",
                    "downstream_model": "lr",
                    "k": 10,
                    "method_base": "rf",
                    "dataset_mean_score": 0.3,
                },
            ]
        )

        spread = _build_benchmark_spread(scores, "classification")
        k5 = spread[(spread["comparison_scope"] == "standard_k") & (spread["k"] == 5)].iloc[0]
        pooled = spread[spread["comparison_scope"] == "all_standard_k"].iloc[0]

        assert k5["n_dataset_downstream_cells"] == 2
        assert k5["mean_range"] == pytest.approx(0.2)
        assert pooled["n_dataset_downstream_cells"] == 4
        assert pooled["mean_range"] == pytest.approx(0.35)

    def test_build_pairwise_aggregate_compares_all_methods_to_all_methods(self):
        """The canonical pairwise benchmark surface should not be CIF-only."""
        scores = pd.DataFrame(
            [
                {
                    "dataset": "d1",
                    "downstream_model": "lr",
                    "k": 5,
                    "method_base": "cif",
                    "dataset_mean_score": 0.80,
                },
                {
                    "dataset": "d1",
                    "downstream_model": "lr",
                    "k": 5,
                    "method_base": "cit",
                    "dataset_mean_score": 0.70,
                },
                {
                    "dataset": "d1",
                    "downstream_model": "lr",
                    "k": 5,
                    "method_base": "dt",
                    "dataset_mean_score": 0.75,
                },
                {
                    "dataset": "d2",
                    "downstream_model": "lr",
                    "k": 5,
                    "method_base": "cif",
                    "dataset_mean_score": 0.60,
                },
                {
                    "dataset": "d2",
                    "downstream_model": "lr",
                    "k": 5,
                    "method_base": "cit",
                    "dataset_mean_score": 0.65,
                },
                {
                    "dataset": "d2",
                    "downstream_model": "lr",
                    "k": 5,
                    "method_base": "dt",
                    "dataset_mean_score": 0.55,
                },
            ]
        )

        pairwise = _build_pairwise_aggregate(scores, "classification")
        pairs = set(zip(pairwise["focus_method"], pairwise["baseline"], strict=False))

        assert pairs == {
            ("cif", "cit"),
            ("cif", "dt"),
            ("cit", "cif"),
            ("cit", "dt"),
            ("dt", "cif"),
            ("dt", "cit"),
        }

    def test_build_config_selection_audit_reports_runner_up_gap(self):
        """Selected-config details should expose family grid size and runner-up separation."""
        df = pd.DataFrame(
            [
                {
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "dataset": "d1",
                    "downstream_model": "lr",
                    "k": 5,
                    "balanced_accuracy": 0.80,
                },
                {
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "dataset": "d2",
                    "downstream_model": "lr",
                    "k": 5,
                    "balanced_accuracy": 0.70,
                },
                {
                    "method_base": "cif",
                    "method_id": "cif__b",
                    "dataset": "d1",
                    "downstream_model": "lr",
                    "k": 5,
                    "balanced_accuracy": 0.76,
                },
                {
                    "method_base": "cif",
                    "method_id": "cif__b",
                    "dataset": "d2",
                    "downstream_model": "lr",
                    "k": 5,
                    "balanced_accuracy": 0.75,
                },
                {
                    "method_base": "rf",
                    "method_id": "rf__a",
                    "dataset": "d1",
                    "downstream_model": "lr",
                    "k": 5,
                    "balanced_accuracy": 0.65,
                },
                {
                    "method_base": "rf",
                    "method_id": "rf__a",
                    "dataset": "d2",
                    "downstream_model": "lr",
                    "k": 5,
                    "balanced_accuracy": 0.66,
                },
            ]
        )

        audit = _build_config_selection_audit(df, "classification", "balanced_accuracy")
        cif = audit[audit["method_base"] == "cif"].iloc[0]
        rf = audit[audit["method_base"] == "rf"].iloc[0]

        assert cif["candidate_config_count"] == 4
        assert cif["method_id"] == "cif__b"
        assert cif["runner_up_method_id"] == "cif__a"
        assert cif["selected_minus_runner_up"] == pytest.approx(0.005)
        assert rf["candidate_config_count"] == 1
        assert pd.isna(rf["runner_up_method_id"])

    def test_build_fixed_panel_membership_requires_all_downstream_k_cells(self):
        """Fixed-panel membership should keep only datasets complete in every standard cell."""
        scores = pd.DataFrame(
            [
                {"downstream_model": "lr", "k": 5, "dataset": "d1", "method_base": "cif"},
                {"downstream_model": "lr", "k": 5, "dataset": "d1", "method_base": "rf"},
                {"downstream_model": "lr", "k": 10, "dataset": "d1", "method_base": "cif"},
                {"downstream_model": "lr", "k": 10, "dataset": "d1", "method_base": "rf"},
                {"downstream_model": "svm", "k": 5, "dataset": "d1", "method_base": "cif"},
                {"downstream_model": "svm", "k": 5, "dataset": "d1", "method_base": "rf"},
                {"downstream_model": "svm", "k": 10, "dataset": "d1", "method_base": "cif"},
                {"downstream_model": "svm", "k": 10, "dataset": "d1", "method_base": "rf"},
                {"downstream_model": "lr", "k": 5, "dataset": "d2", "method_base": "cif"},
                {"downstream_model": "lr", "k": 5, "dataset": "d2", "method_base": "rf"},
                {"downstream_model": "lr", "k": 10, "dataset": "d2", "method_base": "cif"},
                {"downstream_model": "lr", "k": 10, "dataset": "d2", "method_base": "rf"},
                {"downstream_model": "svm", "k": 5, "dataset": "d2", "method_base": "cif"},
                {"downstream_model": "svm", "k": 5, "dataset": "d2", "method_base": "rf"},
            ]
        )

        membership = _build_fixed_panel_membership(scores, "classification")
        d1 = membership[membership["dataset"] == "d1"].iloc[0]
        d2 = membership[membership["dataset"] == "d2"].iloc[0]

        assert d1["required_complete_cells"] == 4
        assert d1["n_complete_case_cells"] == 4
        assert bool(d1["is_fixed_panel"])
        assert d2["n_complete_case_cells"] == 3
        assert not bool(d2["is_fixed_panel"])

    def test_build_fixed_panel_aggregate_compares_to_headline_summary(self):
        """Fixed-panel aggregate should report deltas relative to the headline surface."""
        scores = pd.DataFrame(
            [
                {
                    "downstream_model": "lr",
                    "k": 5,
                    "dataset": "d1",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "dataset_mean_score": 0.9,
                },
                {
                    "downstream_model": "lr",
                    "k": 5,
                    "dataset": "d1",
                    "method_base": "rf",
                    "method_id": "rf__a",
                    "dataset_mean_score": 0.7,
                },
                {
                    "downstream_model": "lr",
                    "k": 5,
                    "dataset": "d1",
                    "method_base": "et",
                    "method_id": "et__a",
                    "dataset_mean_score": 0.6,
                },
                {
                    "downstream_model": "lr",
                    "k": 10,
                    "dataset": "d1",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "dataset_mean_score": 0.8,
                },
                {
                    "downstream_model": "lr",
                    "k": 10,
                    "dataset": "d1",
                    "method_base": "rf",
                    "method_id": "rf__a",
                    "dataset_mean_score": 0.6,
                },
                {
                    "downstream_model": "lr",
                    "k": 10,
                    "dataset": "d1",
                    "method_base": "et",
                    "method_id": "et__a",
                    "dataset_mean_score": 0.5,
                },
                {
                    "downstream_model": "svm",
                    "k": 5,
                    "dataset": "d1",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "dataset_mean_score": 0.85,
                },
                {
                    "downstream_model": "svm",
                    "k": 5,
                    "dataset": "d1",
                    "method_base": "rf",
                    "method_id": "rf__a",
                    "dataset_mean_score": 0.65,
                },
                {
                    "downstream_model": "svm",
                    "k": 5,
                    "dataset": "d1",
                    "method_base": "et",
                    "method_id": "et__a",
                    "dataset_mean_score": 0.55,
                },
                {
                    "downstream_model": "svm",
                    "k": 10,
                    "dataset": "d1",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "dataset_mean_score": 0.75,
                },
                {
                    "downstream_model": "svm",
                    "k": 10,
                    "dataset": "d1",
                    "method_base": "rf",
                    "method_id": "rf__a",
                    "dataset_mean_score": 0.55,
                },
                {
                    "downstream_model": "svm",
                    "k": 10,
                    "dataset": "d1",
                    "method_base": "et",
                    "method_id": "et__a",
                    "dataset_mean_score": 0.45,
                },
                {
                    "downstream_model": "lr",
                    "k": 5,
                    "dataset": "d2",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "dataset_mean_score": 0.6,
                },
                {
                    "downstream_model": "lr",
                    "k": 5,
                    "dataset": "d2",
                    "method_base": "rf",
                    "method_id": "rf__a",
                    "dataset_mean_score": 0.8,
                },
                {
                    "downstream_model": "lr",
                    "k": 5,
                    "dataset": "d2",
                    "method_base": "et",
                    "method_id": "et__a",
                    "dataset_mean_score": 0.7,
                },
                {
                    "downstream_model": "lr",
                    "k": 10,
                    "dataset": "d2",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "dataset_mean_score": 0.5,
                },
                {
                    "downstream_model": "lr",
                    "k": 10,
                    "dataset": "d2",
                    "method_base": "rf",
                    "method_id": "rf__a",
                    "dataset_mean_score": 0.7,
                },
                {
                    "downstream_model": "lr",
                    "k": 10,
                    "dataset": "d2",
                    "method_base": "et",
                    "method_id": "et__a",
                    "dataset_mean_score": 0.6,
                },
            ]
        )

        headline_ranked = scores.copy()
        headline_ranked["rank"] = headline_ranked.groupby(["downstream_model", "k", "dataset"])[
            "dataset_mean_score"
        ].rank(
            ascending=False,
            method="average",
        )
        headline_by_dataset = headline_ranked.groupby(
            ["dataset", "method_base", "method_id"], as_index=False
        ).agg(
            n_cells=("rank", "size"),
            mean_rank=("rank", "mean"),
            mean_score=("dataset_mean_score", "mean"),
        )
        headline_aggregate = headline_by_dataset.groupby(
            ["method_base", "method_id"], as_index=False
        ).agg(
            n_datasets=("dataset", "nunique"),
            mean_dataset_cells=("n_cells", "mean"),
            mean_rank=("mean_rank", "mean"),
            median_rank=("mean_rank", "median"),
            mean_score=("mean_score", "mean"),
        )
        headline_aggregate.insert(0, "task", "classification")
        headline_aggregate.insert(1, "metric", "balanced_accuracy")
        headline_aggregate["support_type"] = "dataset_mean_over_all_complete_case_downstream_k"
        headline_aggregate["rank_position"] = headline_aggregate["mean_rank"].rank(
            ascending=True, method="average"
        )

        _, fixed = _build_fixed_panel_aggregate(
            scores, "classification", "balanced_accuracy", headline_aggregate
        )
        cif = fixed[fixed["method_base"] == "cif"].iloc[0]

        assert cif["n_datasets"] == 1
        assert cif["headline_n_datasets"] == 2
        assert cif["rank_position"] == 1
        assert cif["headline_rank_position"] == 2
        assert cif["delta_mean_rank_vs_headline"] == pytest.approx(-1.0)


class TestHighPEndpointTables:
    """Tests for high-p endpoint summary helpers."""

    def test_build_cif_best_observed_k_summary_buckets_first_optimum(self):
        """The first best observed k should be bucketed correctly."""
        scores = pd.DataFrame(
            [
                # under_100 optimum
                {
                    "task": "classification",
                    "dataset": "d1",
                    "downstream_model": "lr",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "k": 50,
                    "dataset_mean_score": 0.90,
                    "endpoint_k": 500,
                },
                {
                    "task": "classification",
                    "dataset": "d1",
                    "downstream_model": "lr",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "k": 100,
                    "dataset_mean_score": 0.80,
                    "endpoint_k": 500,
                },
                # k=100 optimum
                {
                    "task": "classification",
                    "dataset": "d2",
                    "downstream_model": "lr",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "k": 100,
                    "dataset_mean_score": 0.91,
                    "endpoint_k": 500,
                },
                {
                    "task": "classification",
                    "dataset": "d2",
                    "downstream_model": "lr",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "k": 300,
                    "dataset_mean_score": 0.70,
                    "endpoint_k": 500,
                },
                # intermediate optimum
                {
                    "task": "classification",
                    "dataset": "d3",
                    "downstream_model": "lr",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "k": 100,
                    "dataset_mean_score": 0.88,
                    "endpoint_k": 500,
                },
                {
                    "task": "classification",
                    "dataset": "d3",
                    "downstream_model": "lr",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "k": 300,
                    "dataset_mean_score": 0.95,
                    "endpoint_k": 500,
                },
                # endpoint optimum
                {
                    "task": "classification",
                    "dataset": "d4",
                    "downstream_model": "lr",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "k": 100,
                    "dataset_mean_score": 0.89,
                    "endpoint_k": 500,
                },
                {
                    "task": "classification",
                    "dataset": "d4",
                    "downstream_model": "lr",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "k": 500,
                    "dataset_mean_score": 0.96,
                    "endpoint_k": 500,
                },
            ]
        )

        summary, examples = build_cif_best_observed_k_summary(scores, "cif")

        overall = summary[summary["downstream_model"] == "all"].iloc[0]
        assert overall["n_cells"] == 4
        assert overall["under_100_cells"] == 1
        assert overall["k100_cells"] == 1
        assert overall["intermediate_cells"] == 1
        assert overall["endpoint_cells"] == 1
        assert overall["under_100_share"] == pytest.approx(0.25)
        assert overall["k100_share"] == pytest.approx(0.25)
        assert overall["intermediate_share"] == pytest.approx(0.25)
        assert overall["endpoint_share"] == pytest.approx(0.25)

        assert examples["best_k_bucket"].tolist() == [
            "under_100",
            "k100",
            "between_100_and_endpoint",
            "endpoint",
        ]

    def test_build_endpoint_method_presence_exposes_missing_endpoint_method(self):
        """Method-level endpoint presence should surface missing k=p rows."""
        scores = pd.DataFrame(
            [
                {
                    "task": "classification",
                    "dataset": "d1",
                    "downstream_model": "lr",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "k": 100,
                    "dataset_mean_score": 0.7,
                    "endpoint_k": 500,
                },
                {
                    "task": "classification",
                    "dataset": "d1",
                    "downstream_model": "lr",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "k": 500,
                    "dataset_mean_score": 0.8,
                    "endpoint_k": 500,
                },
                {
                    "task": "classification",
                    "dataset": "d1",
                    "downstream_model": "lr",
                    "method_base": "rf",
                    "method_id": "rf__a",
                    "k": 100,
                    "dataset_mean_score": 0.6,
                    "endpoint_k": 500,
                },
            ]
        )

        presence = build_endpoint_method_presence(scores)
        cif = presence[presence["method_base"] == "cif"].iloc[0]
        rf = presence[presence["method_base"] == "rf"].iloc[0]

        assert bool(cif["has_endpoint_row"])
        assert not bool(cif["missing_endpoint_row"])
        assert not bool(rf["has_endpoint_row"])
        assert bool(rf["missing_endpoint_row"])

    def test_build_delta_vs_endpoint_cells_keeps_best_observed_flag(self):
        """The cell-level high-p surface should keep endpoint deltas and best-k flags."""
        scores = pd.DataFrame(
            [
                {
                    "task": "regression",
                    "dataset": "d1",
                    "downstream_model": "ridge",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "k": 100,
                    "dataset_mean_score": 0.8,
                    "endpoint_k": 500,
                },
                {
                    "task": "regression",
                    "dataset": "d1",
                    "downstream_model": "ridge",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "k": 500,
                    "dataset_mean_score": 0.7,
                    "endpoint_k": 500,
                },
            ]
        )

        cells = build_delta_vs_endpoint_cells(scores)
        k100 = cells[cells["k"] == 100].iloc[0]
        endpoint = cells[cells["k"] == 500].iloc[0]

        assert k100["score_minus_endpoint"] == pytest.approx(0.1)
        assert bool(k100["is_standard_k"])
        assert not bool(k100["is_endpoint"])
        assert bool(k100["is_best_over_observed_k"])
        assert bool(endpoint["is_endpoint"])


class TestSyntheticTrendSummaries:
    """Tests for across-k synthetic trend surfaces."""

    def test_top_ranking_curve_summary_uses_mean_over_standard_k(self):
        """Top-ranking trend summary should rank methods by the head-of-list mean, not a single k."""
        df = pd.DataFrame(
            [
                {
                    "task": "classification",
                    "dataset": "d1",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "top1_hit": 1,
                    "any_hit@2": 1,
                    "mrr_true": 1.0,
                    "first_true_rank": 1,
                    "precision@1": 0.70,
                    "precision@2": 0.60,
                    "precision@5": 0.70,
                    "precision@10": 0.60,
                    "precision@25": 0.50,
                    "precision@50": 0.40,
                    "precision@100": 0.30,
                    "recall@1": 0.20,
                    "recall@2": 0.30,
                    "recall@5": 0.20,
                    "recall@10": 0.30,
                    "recall@25": 0.40,
                    "recall@50": 0.50,
                    "recall@100": 0.60,
                    "f1@1": 0.30,
                    "f1@2": 0.35,
                    "f1@5": 0.30,
                    "f1@10": 0.35,
                    "f1@25": 0.40,
                    "f1@50": 0.45,
                    "f1@100": 0.50,
                },
                {
                    "task": "classification",
                    "dataset": "d1",
                    "method_base": "rf",
                    "method_id": "rf__a",
                    "top1_hit": 0,
                    "any_hit@2": 1,
                    "mrr_true": 0.5,
                    "first_true_rank": 2,
                    "precision@1": 0.60,
                    "precision@2": 0.59,
                    "precision@5": 0.60,
                    "precision@10": 0.59,
                    "precision@25": 0.58,
                    "precision@50": 0.57,
                    "precision@100": 0.56,
                    "recall@1": 0.25,
                    "recall@2": 0.35,
                    "recall@5": 0.25,
                    "recall@10": 0.35,
                    "recall@25": 0.45,
                    "recall@50": 0.55,
                    "recall@100": 0.65,
                    "f1@1": 0.35,
                    "f1@2": 0.40,
                    "f1@5": 0.35,
                    "f1@10": 0.40,
                    "f1@25": 0.45,
                    "f1@50": 0.50,
                    "f1@100": 0.55,
                },
            ]
        )

        summary = summarize_top_ranking_curve_over_k(df)
        cif = summary[summary["method_base"] == "cif"].iloc[0]
        rf = summary[summary["method_base"] == "rf"].iloc[0]

        assert cif["mean_precision_over_head_k_1_2"] == pytest.approx(0.65)
        assert rf["mean_precision_over_head_k_1_2"] == pytest.approx(0.595)
        assert cif["curve_rank_position"] < rf["curve_rank_position"]

    def test_topk_curve_summary_tracks_curve_means_and_endpoints(self):
        """Top-k composition curve summary should expose across-k means and endpoint deltas."""
        df = pd.DataFrame(
            [
                {
                    "task": "classification",
                    "dataset": "d1",
                    "dataset_type": "standard_easy",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "k": 5,
                    "informative_share": 0.80,
                    "signal_or_redundant_share": 0.90,
                    "pure_noise_share": 0.10,
                    "correlated_noise_share": 0.00,
                    "missing_share": 0.00,
                    "dataset_size_cap_share": 0.00,
                },
                {
                    "task": "classification",
                    "dataset": "d1",
                    "dataset_type": "standard_easy",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "k": 10,
                    "informative_share": 0.70,
                    "signal_or_redundant_share": 0.80,
                    "pure_noise_share": 0.20,
                    "correlated_noise_share": 0.00,
                    "missing_share": 0.00,
                    "dataset_size_cap_share": 0.00,
                },
                {
                    "task": "classification",
                    "dataset": "d1",
                    "dataset_type": "standard_easy",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "k": 25,
                    "informative_share": 0.60,
                    "signal_or_redundant_share": 0.70,
                    "pure_noise_share": 0.30,
                    "correlated_noise_share": 0.00,
                    "missing_share": 0.00,
                    "dataset_size_cap_share": 0.00,
                },
                {
                    "task": "classification",
                    "dataset": "d1",
                    "dataset_type": "standard_easy",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "k": 50,
                    "informative_share": 0.50,
                    "signal_or_redundant_share": 0.60,
                    "pure_noise_share": 0.40,
                    "correlated_noise_share": 0.00,
                    "missing_share": 0.00,
                    "dataset_size_cap_share": 0.00,
                },
                {
                    "task": "classification",
                    "dataset": "d1",
                    "dataset_type": "standard_easy",
                    "method_base": "cif",
                    "method_id": "cif__a",
                    "k": 100,
                    "informative_share": 0.40,
                    "signal_or_redundant_share": 0.50,
                    "pure_noise_share": 0.50,
                    "correlated_noise_share": 0.00,
                    "missing_share": 0.00,
                    "dataset_size_cap_share": 0.00,
                },
            ]
        )

        summary = summarize_topk_curve_over_k(df, ["task", "method_base", "method_id"])
        row = summary.iloc[0]

        assert row["mean_informative_share_over_standard_k"] == pytest.approx(0.60)
        assert row["mean_pure_noise_share_over_standard_k"] == pytest.approx(0.30)
        assert row["informative_share_k5"] == pytest.approx(0.80)
        assert row["informative_share_k100"] == pytest.approx(0.40)
        assert row["informative_share_delta_k100_minus_k5"] == pytest.approx(-0.40)


class TestPaperFacingAblations:
    """Tests for paper-facing ablation summary builders."""

    def test_knob_ablation_summary_normalizes_task_labels(self):
        """Mirrored-knob summary should emit classification/regression task labels."""
        ablation = pd.DataFrame(
            [
                {
                    "task": "clf",
                    "dataset_type": "real_demo",
                    "variant": "cif_default",
                    "elapsed_seconds_mean": 10.0,
                    "precision_at_10_mean": 0.5,
                    "precision_at_5_mean": 0.6,
                    "precision_at_25_mean": 0.4,
                    "precision_at_50_mean": 0.3,
                    "precision_at_100_mean": 0.2,
                    "max_depth_mean": 3.0,
                    "mean_features_used_mean": 5.0,
                    "n_estimators_actual_mean": 100.0,
                    "ds_lr_k5_mean": 0.7,
                    "ds_svm_k5_mean": 0.6,
                    "ds_knn_k5_mean": 0.5,
                },
                {
                    "task": "clf",
                    "dataset_type": "real_demo",
                    "variant": "cif_no_adaptive",
                    "elapsed_seconds_mean": 20.0,
                    "precision_at_10_mean": 0.4,
                    "precision_at_5_mean": 0.5,
                    "precision_at_25_mean": 0.3,
                    "precision_at_50_mean": 0.2,
                    "precision_at_100_mean": 0.1,
                    "max_depth_mean": 4.0,
                    "mean_features_used_mean": 6.0,
                    "n_estimators_actual_mean": 100.0,
                    "ds_lr_k5_mean": 0.65,
                    "ds_svm_k5_mean": 0.55,
                    "ds_knn_k5_mean": 0.45,
                },
            ]
        )

        summary = build_knob_ablation_summary(ablation)
        assert set(summary["task"]) == {"classification"}
        default = summary[summary["variant"] == "cif_default"].iloc[0]
        alt = summary[summary["variant"] == "cif_no_adaptive"].iloc[0]
        assert default["mean_precision_over_standard_k"] == pytest.approx(0.4)
        assert alt["delta_precision_over_standard_k"] == pytest.approx(-0.1)

    def test_threshold_ablation_summary_normalizes_task_labels(self):
        """Threshold summary should emit classification/regression task labels."""
        ablation = pd.DataFrame(
            [
                {
                    "task": "reg",
                    "dataset_type": "real_demo",
                    "variant": "histogram_256",
                    "elapsed_seconds_mean": 10.0,
                    "mean_depth_mean": 2.0,
                    "max_depth_mean": 3.0,
                    "mean_features_used_mean": 4.0,
                    "precision_at_10_mean": 0.4,
                    "precision_at_5_mean": 0.5,
                    "precision_at_25_mean": 0.3,
                    "precision_at_50_mean": 0.2,
                    "precision_at_100_mean": 0.1,
                    "confounder_rate_at_10_mean": 0.2,
                    "ridge_r2_mean": 0.5,
                    "svr_r2_mean": 0.4,
                    "knn_r2_mean": 0.3,
                    "f1_at_10_mean": 0.2,
                    "spread_at_10_mean": 0.1,
                },
                {
                    "task": "reg",
                    "dataset_type": "real_demo",
                    "variant": "exact_all",
                    "elapsed_seconds_mean": 20.0,
                    "mean_depth_mean": 3.0,
                    "max_depth_mean": 4.0,
                    "mean_features_used_mean": 5.0,
                    "precision_at_10_mean": 0.45,
                    "precision_at_5_mean": 0.55,
                    "precision_at_25_mean": 0.35,
                    "precision_at_50_mean": 0.25,
                    "precision_at_100_mean": 0.15,
                    "confounder_rate_at_10_mean": 0.25,
                    "ridge_r2_mean": 0.45,
                    "svr_r2_mean": 0.35,
                    "knn_r2_mean": 0.25,
                    "f1_at_10_mean": 0.25,
                    "spread_at_10_mean": 0.15,
                },
            ]
        )

        summary = build_threshold_ablation_summary(ablation)
        assert set(summary["task"]) == {"regression"}
        default = summary[summary["variant"] == "histogram_256"].iloc[0]
        alt = summary[summary["variant"] == "exact_all"].iloc[0]
        assert default["mean_precision_over_standard_k"] == pytest.approx(0.3)
        assert alt["delta_precision_over_standard_k_vs_default"] == pytest.approx(0.05)


class TestScreeningMechanismHelpers:
    """Tests for fixed-design mechanism helpers."""

    def test_collect_cit_split_features_walks_internal_nodes(self):
        """CIT split-feature traversal should visit each internal node once."""
        tree = {
            "feature": 4,
            "left_child": {"feature": 1, "left_child": {"value": 0}, "right_child": {"value": 1}},
            "right_child": {"value": 1},
        }

        assert collect_cit_split_features(tree) == [4, 1]

    def test_count_split_features_from_tree_handles_cit_and_sklearn(self):
        """Split counting should work for both CIT and sklearn-style trees."""
        rng = np.random.RandomState(0)
        X = rng.normal(size=(64, 2))
        y = (X[:, 0] > 0.0).astype(int)

        cit = build_single_tree_split_model("classification", "cit", seed=0)
        cit.fit(X, y)
        cit_counts = count_split_features_from_tree(cit, p=2)

        dt = build_single_tree_split_model("classification", "dt", seed=0)
        dt.fit(X, y)
        dt_counts = count_split_features_from_tree(dt, p=2)

        assert cit_counts.shape == (2,)
        assert dt_counts.shape == (2,)
        assert cit_counts.sum() >= 1
        assert dt_counts.sum() >= 1

    def test_symmetric_gaussian_generator_honors_n_informative(self):
        """The symmetric Gaussian design should inject signal into all informative features."""
        spec = FixedDesignSpec(
            name="sym",
            kind="symmetric_two_signal_gaussian",
            n_samples=200,
            n_features=20,
            n_informative=5,
            dataset_seed=123,
        )
        X, y, informative = make_fixed_dataset(spec)

        mean_diffs = np.abs(X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0))
        top_features = set(np.argsort(mean_diffs)[-5:].tolist())

        assert len(informative) == 5
        assert set(informative).issubset(top_features)


class TestSyntheticTopKComposition:
    """Tests for synthetic top-k composition helpers."""

    def test_dataset_type_from_config_splits_standard_easy_and_hard(self):
        """Standard easy/hard suites should not be collapsed into one bucket."""
        assert (
            dataset_type_from_config(
                {"name": "synthetic_p100_k10_n1000_sep2.0", "n_features": 100, "n_informative": 10}
            )
            == "standard_easy"
        )
        assert (
            dataset_type_from_config(
                {"name": "synthetic_p1000_k5_n200_sep0.5", "n_features": 1000, "n_informative": 5}
            )
            == "standard_hard"
        )

    def test_build_row_metrics_caps_denominator_at_dataset_size(self):
        """Requested k above p should not create artificial missing-share penalties."""
        groups = {
            "informative": {0, 1},
            "redundant": {2},
            "explicit_noise": set(),
            "correlated_noise": set(),
            "background_null": set(range(3, 50)),
            "n_features_final": 50,
        }
        ranking = list(range(50))

        row = build_topk_row_metrics(ranking, groups, k=100)

        assert row["effective_k"] == 50
        assert row["returned_count"] == 50
        assert row["missing_count"] == 0
        assert row["dataset_size_cap_count"] == 50
        assert row["returned_share"] == pytest.approx(1.0)
        assert row["missing_share"] == pytest.approx(0.0)
        assert row["dataset_size_cap_share"] == pytest.approx(0.5)
        assert row["informative_share"] == pytest.approx(2 / 50)
        assert row["redundant_share"] == pytest.approx(1 / 50)


class TestScreeningMechanismTables:
    """Tests for fixed-design screening mechanism helpers."""

    def test_informative_coverage_probability_matches_sqrt_case(self):
        """The p=100, two-signal sqrt case should match the scratch audit value."""
        prob = informative_coverage_probability(p=100, sampled_features=10, n_informative=2)
        assert prob == pytest.approx(0.1909090909)

    def test_resolve_max_features_count_handles_none_and_sqrt(self):
        """The helper should match the paper-facing candidate-set interpretation."""
        assert resolve_max_features_count(100, None) == 100
        assert resolve_max_features_count(100, "sqrt") == 10
        assert resolve_max_features_count(100, 20) == 20

    def test_make_fixed_dataset_tracks_two_informative_features(self):
        """Both fixed designs should expose exactly two shuffled informative indices."""
        easy = FixedDesignSpec(name="easy", kind="easy_shuffled_classification")
        gaussian = FixedDesignSpec(name="gaussian", kind="symmetric_two_signal_gaussian")

        X_easy, y_easy, informative_easy = make_fixed_dataset(easy)
        X_gauss, y_gauss, informative_gauss = make_fixed_dataset(gaussian)

        assert X_easy.shape == (1000, 100)
        assert X_gauss.shape == (1000, 100)
        assert y_easy.shape == (1000,)
        assert y_gauss.shape == (1000,)
        assert len(informative_easy) == 2
        assert len(informative_gauss) == 2
        assert len(set(informative_easy)) == 2
        assert len(set(informative_gauss)) == 2

    def test_make_fixed_dataset_supports_one_informative_classification_feature(self):
        """The fixed classification generator should handle n_informative=1."""
        spec = FixedDesignSpec(
            name="easy_i1",
            kind="easy_shuffled_classification",
            n_samples=250,
            n_features=100,
            n_informative=1,
            dataset_seed=123,
        )

        X, y, informative = make_fixed_dataset(spec)

        assert X.shape == (250, 100)
        assert y.shape == (250,)
        assert len(informative) == 1
        assert informative[0] in range(100)

    def test_mechanism_shard_builder_supports_multi_informative_grid(self):
        """The shard runner should build one spec per (p, n_informative) pair."""
        specs = build_mechanism_shard_specs("classification", [100, 500], [1, 2, 5])

        assert len(specs) == 6
        assert [spec.name for spec in specs] == [
            "make_classification_n250_p100_i1",
            "make_classification_n250_p500_i1",
            "make_classification_n250_p100_i2",
            "make_classification_n250_p500_i2",
            "make_classification_n250_p100_i5",
            "make_classification_n250_p500_i5",
        ]
        assert {(spec.n_features, spec.n_informative) for spec in specs} == {
            (100, 1),
            (500, 1),
            (100, 2),
            (500, 2),
            (100, 5),
            (500, 5),
        }


class TestComputeNoiseSelectionRate:
    """Tests for the noise selection rate metric."""

    def test_perfect_selection_no_noise(self):
        """No noise in top-k should return 0.0."""
        ranking = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        noise_indices = [10, 11, 12]
        assert compute_noise_selection_rate(ranking, noise_indices, k=5) == 0.0

    def test_worst_case_all_noise(self):
        """All noise in top-k should return 1.0."""
        ranking = [10, 11, 12, 0, 1, 2]
        noise_indices = [10, 11, 12]
        assert compute_noise_selection_rate(ranking, noise_indices, k=3) == 1.0

    def test_mixed_selection_half(self):
        """Mixed selection should return correct fraction."""
        ranking = [0, 10, 1, 11, 2]
        noise_indices = [10, 11, 12]
        assert compute_noise_selection_rate(ranking, noise_indices, k=4) == 0.5

    def test_mixed_selection_one_third(self):
        """One noise feature in top-3 should return 1/3."""
        ranking = [0, 1, 10, 2, 3]
        noise_indices = [10, 11, 12]
        result = compute_noise_selection_rate(ranking, noise_indices, k=3)
        assert result == pytest.approx(1 / 3)

    def test_edge_case_k_zero(self):
        """k=0 should return 0.0 (no features to evaluate)."""
        ranking = [10, 11, 12]
        noise_indices = [10, 11, 12]
        assert compute_noise_selection_rate(ranking, noise_indices, k=0) == 0.0

    def test_edge_case_empty_noise_indices(self):
        """Empty noise_indices should return 0.0 (no noise to find)."""
        ranking = [0, 1, 2, 3, 4]
        assert compute_noise_selection_rate(ranking, [], k=3) == 0.0

    def test_edge_case_k_larger_than_ranking(self):
        """k larger than ranking length uses k as denominator."""
        ranking = [0, 10, 1]
        noise_indices = [10, 11, 12]
        result = compute_noise_selection_rate(ranking, noise_indices, k=10)
        assert result == pytest.approx(0.1)

    def test_noise_not_in_ranking(self):
        """Noise indices not appearing in ranking should not be counted."""
        ranking = [0, 1, 2, 3, 4]
        noise_indices = [100, 101, 102]
        assert compute_noise_selection_rate(ranking, noise_indices, k=5) == 0.0

    def test_partial_noise_overlap(self):
        """Only noise indices that appear in top-k should be counted."""
        ranking = [0, 10, 1, 2, 3]
        noise_indices = [10, 11, 12]
        assert compute_noise_selection_rate(ranking, noise_indices, k=5) == 0.2

    def test_deterministic_output(self):
        """Same inputs should always produce same output."""
        ranking = [0, 10, 1, 11, 2, 12]
        noise_indices = [10, 11, 12]
        k = 4
        result1 = compute_noise_selection_rate(ranking, noise_indices, k)
        result2 = compute_noise_selection_rate(ranking, noise_indices, k)
        assert result1 == result2

    def test_realistic_rf_vs_citrees_scenario(self):
        """Simulated RF should have higher noise rate than citrees."""
        np.random.seed(42)
        n_features = 50
        n_informative = 10
        informative_indices = list(range(n_informative))
        noise_indices = list(range(n_informative, n_features))

        rf_ranking = (
            [30, 35, 40, 0, 1, 2, 3, 4, 5, 6]
            + list(range(7, 30))
            + [31, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        )
        citrees_ranking = informative_indices + noise_indices

        rf_rate = compute_noise_selection_rate(rf_ranking, noise_indices, k=10)
        citrees_rate = compute_noise_selection_rate(citrees_ranking, noise_indices, k=10)

        assert citrees_rate == 0.0
        assert rf_rate > 0.0
        assert citrees_rate < rf_rate


class TestPairwiseWilcoxonHolm:
    """Tests for pairwise Wilcoxon signed-rank test with Holm correction."""

    def test_detects_significant_difference(self):
        """Should detect difference between clearly different distributions."""
        np.random.seed(42)
        n = 30
        data = pd.DataFrame(
            {
                "A_precision@10": np.random.normal(0.8, 0.05, n),
                "B_precision@10": np.random.normal(0.5, 0.05, n),
            }
        )
        result = pairwise_wilcoxon_holm(data, ["A", "B"], "precision@10")
        assert len(result) == 1
        assert result.iloc[0]["significant"]
        assert result.iloc[0]["p_value_corrected"] < 0.05

    def test_no_false_positive_similar_distributions(self):
        """Should not find significance when distributions are similar."""
        np.random.seed(123)
        n = 30
        data = pd.DataFrame(
            {
                "A_precision@10": np.random.normal(0.7, 0.1, n),
                "B_precision@10": np.random.normal(0.7, 0.1, n),
            }
        )
        result = pairwise_wilcoxon_holm(data, ["A", "B"], "precision@10")
        if not result.empty:
            assert result.iloc[0]["p_value_corrected"] > 0.01

    def test_holm_correction_increases_pvalues(self):
        """Holm-corrected p-values should be >= raw p-values."""
        np.random.seed(42)
        n = 30
        data = pd.DataFrame(
            {
                "A_precision@10": np.random.normal(0.8, 0.1, n),
                "B_precision@10": np.random.normal(0.6, 0.1, n),
                "C_precision@10": np.random.normal(0.7, 0.1, n),
            }
        )
        result = pairwise_wilcoxon_holm(data, ["A", "B", "C"], "precision@10")
        for _, row in result.iterrows():
            assert row["p_value_corrected"] >= row["p_value"]

    def test_empty_for_insufficient_data(self):
        """Should return empty DataFrame when n < 10."""
        data = pd.DataFrame(
            {
                "A_precision@10": [0.8, 0.7, 0.9],
                "B_precision@10": [0.5, 0.6, 0.4],
            }
        )
        result = pairwise_wilcoxon_holm(data, ["A", "B"], "precision@10")
        assert result.empty

    def test_output_columns(self):
        """Should have expected output columns."""
        np.random.seed(42)
        n = 30
        data = pd.DataFrame(
            {
                "A_precision@10": np.random.normal(0.8, 0.1, n),
                "B_precision@10": np.random.normal(0.6, 0.1, n),
            }
        )
        result = pairwise_wilcoxon_holm(data, ["A", "B"], "precision@10")
        expected_cols = [
            "method1",
            "method2",
            "statistic",
            "p_value",
            "p_value_corrected",
            "significant",
            "n_pairs",
        ]
        assert list(result.columns) == expected_cols

    def test_correct_number_of_pairs(self):
        """Should generate C(n,2) pairs for n methods."""
        np.random.seed(42)
        n = 30
        methods = ["rf", "cif", "boruta", "pi"]
        data = pd.DataFrame(
            {
                f"{m}_precision@10": np.random.normal(0.5 + 0.1 * i, 0.05, n)
                for i, m in enumerate(methods)
            }
        )
        result = pairwise_wilcoxon_holm(data, methods, "precision@10")
        expected_pairs = len(methods) * (len(methods) - 1) // 2
        assert len(result) == expected_pairs


class TestCohensD:
    """Tests for Cohen's d effect size calculation."""

    def test_identical_distributions_zero(self):
        """Identical distributions should have d ≈ 0."""
        g1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(cohens_d(g1, g2)) < 0.001

    def test_one_sd_difference(self):
        """1 SD difference should give d ≈ 1."""
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 1000)
        g2 = np.random.normal(1, 1, 1000)
        d = cohens_d(g1, g2)
        assert abs(d + 1.0) < 0.15

    def test_sign_direction(self):
        """d should be positive when g1 > g2, negative when g1 < g2."""
        np.random.seed(42)
        g1 = np.random.normal(10, 1, 100)
        g2 = np.random.normal(5, 1, 100)
        d = cohens_d(g1, g2)
        assert d > 0

        d_reversed = cohens_d(g2, g1)
        assert d_reversed < 0

    def test_zero_variance_returns_zero(self):
        """Zero variance should return 0.0 to avoid division by zero."""
        g1 = np.array([5.0, 5.0, 5.0, 5.0])
        g2 = np.array([5.0, 5.0, 5.0, 5.0])
        assert cohens_d(g1, g2) == 0.0


class TestInterpretCohensD:
    """Tests for Cohen's d interpretation."""

    def test_negligible(self):
        """d < 0.2 should be negligible."""
        assert interpret_cohens_d(0.0) == "negligible"
        assert interpret_cohens_d(0.1) == "negligible"
        assert interpret_cohens_d(0.19) == "negligible"
        assert interpret_cohens_d(-0.1) == "negligible"

    def test_small(self):
        """0.2 <= d < 0.5 should be small."""
        assert interpret_cohens_d(0.2) == "small"
        assert interpret_cohens_d(0.35) == "small"
        assert interpret_cohens_d(0.49) == "small"
        assert interpret_cohens_d(-0.3) == "small"

    def test_medium(self):
        """0.5 <= d < 0.8 should be medium."""
        assert interpret_cohens_d(0.5) == "medium"
        assert interpret_cohens_d(0.65) == "medium"
        assert interpret_cohens_d(0.79) == "medium"
        assert interpret_cohens_d(-0.6) == "medium"

    def test_large(self):
        """d >= 0.8 should be large."""
        assert interpret_cohens_d(0.8) == "large"
        assert interpret_cohens_d(1.0) == "large"
        assert interpret_cohens_d(2.5) == "large"
        assert interpret_cohens_d(-1.2) == "large"


class TestBootstrapCI:
    """Tests for bootstrap confidence interval calculation."""

    def test_ci_contains_sample_mean(self):
        """CI should contain sample mean."""
        scores = np.array([0.8, 0.82, 0.79, 0.81, 0.83, 0.78, 0.80, 0.84, 0.77, 0.82])
        lo, hi = bootstrap_ci(scores)
        mean = np.mean(scores)
        assert lo <= mean <= hi

    def test_larger_samples_narrower_ci(self):
        """Larger samples should produce narrower CI."""
        np.random.seed(42)
        small_sample = np.random.normal(0.8, 0.1, 10)
        large_sample = np.random.normal(0.8, 0.1, 100)

        lo_s, hi_s = bootstrap_ci(small_sample)
        lo_l, hi_l = bootstrap_ci(large_sample)

        width_small = hi_s - lo_s
        width_large = hi_l - lo_l
        assert width_large < width_small

    def test_deterministic_with_seed(self):
        """Same seed should produce identical results."""
        scores = np.array([0.8, 0.82, 0.79, 0.81, 0.83])
        ci1 = bootstrap_ci(scores, random_state=42)
        ci2 = bootstrap_ci(scores, random_state=42)
        assert ci1 == ci2

    def test_different_seeds_different_results(self):
        """Different seeds should produce different results."""
        scores = np.array([0.8, 0.82, 0.79, 0.81, 0.83])
        ci1 = bootstrap_ci(scores, random_state=42)
        ci2 = bootstrap_ci(scores, random_state=123)
        assert ci1 != ci2

    def test_high_confidence_wider_ci(self):
        """Higher confidence level should produce wider CI."""
        scores = np.array([0.8, 0.82, 0.79, 0.81, 0.83, 0.78, 0.80, 0.84])
        lo_95, hi_95 = bootstrap_ci(scores, ci=0.95)
        lo_99, hi_99 = bootstrap_ci(scores, ci=0.99)

        width_95 = hi_95 - lo_95
        width_99 = hi_99 - lo_99
        assert width_99 > width_95


class TestFriedmanTest:
    """Tests for Friedman test."""

    @pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide")
    def test_uses_complete_case_rows(self):
        """Friedman test should use only complete cases."""
        data = pd.DataFrame(
            {
                "a_m": [1.0, 2.0, np.nan, 4.0],
                "b_m": [1.0, np.nan, 3.0, 4.0],
                "c_m": [1.0, 2.0, 3.0, 4.0],
            }
        )
        chi2, p, n_datasets, k_methods = friedman_test(data, ["a", "b", "c"], "m")
        assert k_methods == 3
        assert n_datasets == 2
        assert np.isfinite(chi2) or np.isnan(chi2)

    @pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide")
    def test_pairwise_wilcoxon_uses_aligned_pairs(self):
        """Pairwise Wilcoxon should use aligned pairs."""
        n = 12
        a = np.arange(n, dtype=float)
        b = np.arange(n, dtype=float)
        c = np.arange(n, dtype=float)
        a[1] = np.nan
        b[2] = np.nan
        data = pd.DataFrame({"a_m": a, "b_m": b, "c_m": c})

        results = pairwise_wilcoxon_holm(data, ["a", "b", "c"], "m")
        assert not results.empty

        ab = results[(results["method1"] == "a") & (results["method2"] == "b")].iloc[0]
        assert ab["n_pairs"] == n - 2


class TestPresentationSummaryTables:
    """Tests for compact paper-presentation summary tables."""

    def test_build_benchmark_presentation_summary(self):
        aggregate = pd.DataFrame(
            [
                {
                    "task": "classification",
                    "method_base": "lgbm",
                    "mean_rank": 4.6,
                    "mean_score": 0.83,
                    "rank_position": 1,
                },
                {
                    "task": "classification",
                    "method_base": "cif",
                    "mean_rank": 5.4,
                    "mean_score": 0.82,
                    "rank_position": 4,
                },
                {
                    "task": "regression",
                    "method_base": "et",
                    "mean_rank": 5.8,
                    "mean_score": 0.35,
                    "rank_position": 1,
                },
                {
                    "task": "regression",
                    "method_base": "cif",
                    "mean_rank": 6.2,
                    "mean_score": 0.33,
                    "rank_position": 2,
                },
            ]
        )
        heterogeneity = pd.DataFrame(
            [
                {
                    "task": "classification",
                    "method_base": "cif",
                    "n_datasets": 22,
                    "top1_share": 3 / 22,
                    "top3_share": 7 / 22,
                    "top_half_share": 21 / 22,
                },
                {
                    "task": "regression",
                    "method_base": "cif",
                    "n_datasets": 8,
                    "top1_share": 1 / 8,
                    "top3_share": 2 / 8,
                    "top_half_share": 1.0,
                },
            ]
        )
        pairwise = pd.DataFrame(
            [
                {
                    "task": "classification",
                    "focus_method": "cif",
                    "baseline": "r_ctree",
                    "n_datasets": 22,
                    "wins": 22,
                    "mean_delta": 0.0876,
                },
                {
                    "task": "classification",
                    "focus_method": "cif",
                    "baseline": "r_cforest",
                    "n_datasets": 22,
                    "wins": 19,
                    "mean_delta": 0.0715,
                },
                {
                    "task": "classification",
                    "focus_method": "cif",
                    "baseline": "cit",
                    "n_datasets": 23,
                    "wins": 21,
                    "mean_delta": 0.0301,
                },
                {
                    "task": "regression",
                    "focus_method": "cif",
                    "baseline": "r_ctree",
                    "n_datasets": 8,
                    "wins": 7,
                    "mean_delta": 0.2265,
                },
                {
                    "task": "regression",
                    "focus_method": "cif",
                    "baseline": "r_cforest",
                    "n_datasets": 8,
                    "wins": 7,
                    "mean_delta": 0.7211,
                },
                {
                    "task": "regression",
                    "focus_method": "cif",
                    "baseline": "cit",
                    "n_datasets": 8,
                    "wins": 6,
                    "mean_delta": 0.3598,
                },
            ]
        )
        membership = pd.DataFrame(
            [
                {
                    "task": "classification",
                    "downstream_model": "lr",
                    "k": 5,
                    "dataset": "d1",
                    "is_complete_case": True,
                },
                {
                    "task": "classification",
                    "downstream_model": "svm",
                    "k": 5,
                    "dataset": "d1",
                    "is_complete_case": True,
                },
                {
                    "task": "classification",
                    "downstream_model": "lr",
                    "k": 100,
                    "dataset": "d1",
                    "is_complete_case": True,
                },
                {
                    "task": "classification",
                    "downstream_model": "svm",
                    "k": 100,
                    "dataset": "d1",
                    "is_complete_case": False,
                },
                {
                    "task": "regression",
                    "downstream_model": "ridge",
                    "k": 5,
                    "dataset": "r1",
                    "is_complete_case": True,
                },
                {
                    "task": "regression",
                    "downstream_model": "svr",
                    "k": 5,
                    "dataset": "r1",
                    "is_complete_case": True,
                },
                {
                    "task": "regression",
                    "downstream_model": "ridge",
                    "k": 100,
                    "dataset": "r1",
                    "is_complete_case": True,
                },
                {
                    "task": "regression",
                    "downstream_model": "svr",
                    "k": 100,
                    "dataset": "r1",
                    "is_complete_case": True,
                },
            ]
        )
        spread = pd.DataFrame(
            [
                {
                    "task": "classification",
                    "comparison_scope": "standard_k",
                    "k": 5,
                    "mean_range": 0.25,
                },
                {
                    "task": "classification",
                    "comparison_scope": "standard_k",
                    "k": 100,
                    "mean_range": 0.18,
                },
                {
                    "task": "regression",
                    "comparison_scope": "standard_k",
                    "k": 5,
                    "mean_range": 1.05,
                },
                {
                    "task": "regression",
                    "comparison_scope": "standard_k",
                    "k": 100,
                    "mean_range": 1.88,
                },
            ]
        )

        summary = build_benchmark_presentation_summary(
            aggregate, heterogeneity, pairwise, membership, spread
        )

        clf = summary[summary["task"] == "classification"].iloc[0]
        reg = summary[summary["task"] == "regression"].iloc[0]
        assert clf["cif_rank_position"] == 4
        assert clf["best_method_base"] == "lgbm"
        assert clf["n_complete_case_datasets_k5"] == 1
        assert clf["n_complete_case_datasets_k100"] == 0
        assert clf["cif_top3_datasets"] == 7
        assert clf["cif_positive_vs_r_ctree_datasets"] == 22
        assert clf["mean_cross_method_range_k100"] == pytest.approx(0.18)
        assert reg["cif_rank_position"] == 2
        assert reg["cif_top_half_datasets"] == 8
        assert reg["cif_mean_delta_vs_r_cforest"] == pytest.approx(0.7211)

    def test_build_practical_controls_presentation_summary(self):
        mirrored = pd.DataFrame(
            [
                {
                    "task": "classification",
                    "dataset_group": "synthetic",
                    "variant": "cif_no_adaptive",
                    "runtime_ratio_vs_default_median": 6.2,
                    "delta_downstream_score": -0.005,
                    "delta_precision_over_standard_k": -0.001,
                    "delta_max_depth": 2.0,
                    "delta_features_used": 0.7,
                },
                {
                    "task": "regression",
                    "dataset_group": "real",
                    "variant": "cif_no_bonferroni",
                    "runtime_ratio_vs_default_median": 0.02,
                    "delta_downstream_score": 0.001,
                    "delta_precision_over_standard_k": pd.NA,
                    "delta_max_depth": 3.0,
                    "delta_features_used": 1.6,
                },
            ]
        )
        threshold = pd.DataFrame(
            [
                {
                    "task": "classification",
                    "dataset_group": "synthetic",
                    "variant": "histogram_32",
                    "elapsed_seconds_ratio_vs_default": 0.04,
                    "delta_real_downstream_vs_default": 0.002,
                    "delta_precision_over_standard_k_vs_default": 0.015,
                    "delta_depth_vs_default": 0.1,
                    "delta_features_used_vs_default": 0.2,
                },
                {
                    "task": "regression",
                    "dataset_group": "real",
                    "variant": "exact_all",
                    "elapsed_seconds_ratio_vs_default": 10.7,
                    "delta_real_downstream_vs_default": 0.006,
                    "delta_precision_over_standard_k_vs_default": pd.NA,
                    "delta_depth_vs_default": 0.1,
                    "delta_features_used_vs_default": 0.4,
                },
            ]
        )

        summary = build_practical_controls_presentation_summary(mirrored, threshold)

        assert set(summary["variant"]) == {
            "cif_no_adaptive",
            "cif_no_bonferroni",
            "histogram_32",
            "exact_all",
        }
        adaptive = summary[summary["variant"] == "cif_no_adaptive"].iloc[0]
        exact = summary[summary["variant"] == "exact_all"].iloc[0]
        assert adaptive["family"] == "mirrored"
        assert adaptive["reference_variant"] == "cif_default"
        assert adaptive["curve_recovery_delta_vs_default"] == pytest.approx(-0.001)
        assert exact["family"] == "threshold"
        assert exact["reference_variant"] == "histogram_256"
        assert exact["runtime_ratio_vs_default"] == pytest.approx(10.7)
