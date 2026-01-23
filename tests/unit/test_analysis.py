"""Unit tests for paper/scripts/analysis.py statistical functions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add paper/scripts to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "paper" / "scripts"))

import pandas as pd
from analysis.stats import (
    bootstrap_ci,
    cohens_d,
    compute_noise_selection_rate,
    interpret_cohens_d,
    pairwise_wilcoxon_holm,
)

# ==============================================================================
# Tests for compute_noise_selection_rate
# ==============================================================================


class TestComputeNoiseSelectionRate:
    """Tests for the noise selection rate metric."""

    def test_perfect_selection_no_noise(self):
        """No noise in top-k should return 0.0."""
        ranking = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Informative first
        noise_indices = [10, 11, 12]  # Noise features
        assert compute_noise_selection_rate(ranking, noise_indices, k=5) == 0.0

    def test_worst_case_all_noise(self):
        """All noise in top-k should return 1.0."""
        ranking = [10, 11, 12, 0, 1, 2]  # Noise first
        noise_indices = [10, 11, 12]
        assert compute_noise_selection_rate(ranking, noise_indices, k=3) == 1.0

    def test_mixed_selection_half(self):
        """Mixed selection should return correct fraction."""
        ranking = [0, 10, 1, 11, 2]  # Alternating
        noise_indices = [10, 11, 12]
        # top-4 = {0, 10, 1, 11}, noise = {10, 11} → 2/4 = 0.5
        assert compute_noise_selection_rate(ranking, noise_indices, k=4) == 0.5

    def test_mixed_selection_one_third(self):
        """One noise feature in top-3 should return 1/3."""
        ranking = [0, 1, 10, 2, 3]
        noise_indices = [10, 11, 12]
        # top-3 = {0, 1, 10}, noise = {10} → 1/3
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
        # ranking[:10] = [0, 10, 1], noise in that = {10} → 1 noise feature
        # rate = 1/k = 1/10 = 0.1 (k is always the denominator)
        result = compute_noise_selection_rate(ranking, noise_indices, k=10)
        assert result == pytest.approx(0.1)

    def test_noise_not_in_ranking(self):
        """Noise indices not appearing in ranking should not be counted."""
        ranking = [0, 1, 2, 3, 4]
        noise_indices = [100, 101, 102]  # Indices not in ranking
        # No overlap → 0.0
        assert compute_noise_selection_rate(ranking, noise_indices, k=5) == 0.0

    def test_partial_noise_overlap(self):
        """Only noise indices that appear in top-k should be counted."""
        ranking = [0, 10, 1, 2, 3]  # Only 10 is noise
        noise_indices = [10, 11, 12]  # 11, 12 not in top-5
        # top-5 = {0, 10, 1, 2, 3}, noise in top-5 = {10} → 1/5 = 0.2
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

        # RF: biased ranking with noise in top positions
        rf_ranking = (
            [30, 35, 40, 0, 1, 2, 3, 4, 5, 6]
            + list(range(7, 30))
            + [31, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        )

        # citrees: unbiased ranking with informative first
        citrees_ranking = informative_indices + noise_indices

        rf_rate = compute_noise_selection_rate(rf_ranking, noise_indices, k=10)
        citrees_rate = compute_noise_selection_rate(citrees_ranking, noise_indices, k=10)

        # citrees should have 0% noise (all informative in top-10)
        assert citrees_rate == 0.0
        # RF should have noise (3 noise features in top-10)
        assert rf_rate > 0.0
        # citrees should be strictly better
        assert citrees_rate < rf_rate


# ==============================================================================
# Tests for pairwise_wilcoxon_holm
# ==============================================================================


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
            # Similar distributions should have high p-value
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
                "A_precision@10": [0.8, 0.7, 0.9],  # Only 3 samples
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
        methods = ["rf", "cif", "boruta", "shap"]
        data = pd.DataFrame(
            {
                f"{m}_precision@10": np.random.normal(0.5 + 0.1 * i, 0.05, n)
                for i, m in enumerate(methods)
            }
        )
        result = pairwise_wilcoxon_holm(data, methods, "precision@10")
        expected_pairs = len(methods) * (len(methods) - 1) // 2  # C(4,2) = 6
        assert len(result) == expected_pairs


# ==============================================================================
# Tests for cohens_d
# ==============================================================================


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
        # g1 < g2, so d should be negative and close to -1
        assert abs(d + 1.0) < 0.15

    def test_sign_direction(self):
        """d should be positive when g1 > g2, negative when g1 < g2."""
        np.random.seed(42)
        g1 = np.random.normal(10, 1, 100)
        g2 = np.random.normal(5, 1, 100)
        d = cohens_d(g1, g2)
        assert d > 0  # g1 > g2

        d_reversed = cohens_d(g2, g1)
        assert d_reversed < 0  # g2 < g1

    def test_zero_variance_returns_zero(self):
        """Zero variance should return 0.0 to avoid division by zero."""
        g1 = np.array([5.0, 5.0, 5.0, 5.0])
        g2 = np.array([5.0, 5.0, 5.0, 5.0])
        assert cohens_d(g1, g2) == 0.0


# ==============================================================================
# Tests for interpret_cohens_d
# ==============================================================================


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


# ==============================================================================
# Tests for bootstrap_ci
# ==============================================================================


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
