"""Unit tests for paper/scripts/analysis.py statistical functions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add paper/scripts to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "paper" / "scripts"))

from analysis import compute_noise_selection_rate


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
        rf_ranking = [30, 35, 40, 0, 1, 2, 3, 4, 5, 6] + list(range(7, 30)) + [31, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49]

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
