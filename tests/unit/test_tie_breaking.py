"""Tests for uniform tie-breaking via reservoir sampling.

When multiple features or thresholds have identical p-values or metrics,
selection should be uniform among all ties (each tied candidate has 1/k
probability where k is the number of ties).

This is implemented via reservoir sampling in _select_best_feature() and
_select_best_split().
"""

import numpy as np
import pytest
from collections import Counter
from unittest.mock import patch

from citrees import ConditionalInferenceTreeClassifier, ConditionalInferenceTreeRegressor


class TestReservoirSamplingTieBreaking:
    """Verify uniform tie-breaking when p-values or metrics are equal."""

    def test_metric_tie_breaking_with_identical_features(self):
        """When features are identical copies, tie-breaking should be uniform.

        Uses n_resamples_selector=None to test metric-only branch.
        """
        rng = np.random.default_rng(42)
        n_samples = 100
        n_features = 5

        # Create ONE informative feature, then copy it n_features times
        # This guarantees all features have EXACTLY the same selector score
        base_feature = rng.standard_normal(n_samples)
        X = np.column_stack([base_feature] * n_features)
        y = (base_feature > 0).astype(int)  # Target correlates with all features equally

        # Run many trials with different random_state values
        n_trials = 500
        root_features = []

        for seed in range(n_trials):
            clf = ConditionalInferenceTreeClassifier(
                n_resamples_selector=None,  # Metric-only mode
                n_resamples_splitter=None,
                max_depth=1,
                random_state=seed,
                verbose=0,
            )
            clf.fit(X, y)

            # Get the feature used at the root
            if "feature" in clf.tree_:
                root_features.append(clf.tree_["feature"])

        # Count how often each feature was selected
        counts = Counter(root_features)

        # With 5 identical features, each should be selected ~20% of the time
        # Use loose bounds to avoid flaky tests
        for feature_idx in range(n_features):
            proportion = counts.get(feature_idx, 0) / len(root_features)
            assert proportion > 0.10, (
                f"Feature {feature_idx} selected only {proportion:.1%} of the time "
                f"(expected ~20% for uniform distribution)"
            )
            assert proportion < 0.35, (
                f"Feature {feature_idx} selected {proportion:.1%} of the time "
                f"(too concentrated for uniform distribution)"
            )

    def test_metric_tie_breaking_regressor_with_identical_features(self):
        """When features are identical copies, tie-breaking should be uniform.

        Uses n_resamples_selector=None to test metric-only branch for regression.
        """
        rng = np.random.default_rng(123)
        n_samples = 100
        n_features = 5

        # Create identical features
        base_feature = rng.standard_normal(n_samples)
        X = np.column_stack([base_feature] * n_features)
        y = base_feature + 0.1 * rng.standard_normal(n_samples)

        n_trials = 500
        root_features = []

        for seed in range(n_trials):
            reg = ConditionalInferenceTreeRegressor(
                n_resamples_selector=None,
                n_resamples_splitter=None,
                max_depth=1,
                random_state=seed,
                verbose=0,
            )
            reg.fit(X, y)

            if "feature" in reg.tree_:
                root_features.append(reg.tree_["feature"])

        counts = Counter(root_features)

        # Check for approximate uniformity
        for feature_idx in range(n_features):
            proportion = counts.get(feature_idx, 0) / len(root_features)
            assert proportion > 0.10, (
                f"Feature {feature_idx} selected only {proportion:.1%} of the time"
            )
            assert proportion < 0.35, (
                f"Feature {feature_idx} selected {proportion:.1%} of the time"
            )

    def test_tie_breaking_reproducible_with_fixed_seed(self):
        """Same random_state should produce identical tie-breaking decisions."""
        rng = np.random.default_rng(999)
        n_samples = 100
        n_features = 5

        # Identical features to ensure ties
        base_feature = rng.standard_normal(n_samples)
        X = np.column_stack([base_feature] * n_features)
        y = (base_feature > 0).astype(int)

        # Fit twice with same random_state
        clf1 = ConditionalInferenceTreeClassifier(
            n_resamples_selector=None,
            n_resamples_splitter=None,
            max_depth=1,
            random_state=42,
            verbose=0,
        )
        clf1.fit(X, y)

        clf2 = ConditionalInferenceTreeClassifier(
            n_resamples_selector=None,
            n_resamples_splitter=None,
            max_depth=1,
            random_state=42,
            verbose=0,
        )
        clf2.fit(X, y)

        # Root feature should be identical
        assert clf1.tree_.get("feature") == clf2.tree_.get("feature"), (
            "Root feature with same random_state should be identical"
        )

    def test_different_seeds_produce_different_selections(self):
        """Different random_state values produce different tie-breaking outcomes."""
        rng = np.random.default_rng(777)
        n_samples = 100
        n_features = 5

        # Identical features to ensure ties
        base_feature = rng.standard_normal(n_samples)
        X = np.column_stack([base_feature] * n_features)
        y = (base_feature > 0).astype(int)

        # Try many different seeds and collect unique root features
        unique_root_features = set()
        for seed in range(100):
            clf = ConditionalInferenceTreeClassifier(
                n_resamples_selector=None,
                n_resamples_splitter=None,
                max_depth=1,
                random_state=seed,
                verbose=0,
            )
            clf.fit(X, y)
            if "feature" in clf.tree_:
                unique_root_features.add(clf.tree_["feature"])

        # With uniform tie-breaking across 5 features, we should see multiple
        assert len(unique_root_features) > 1, (
            "Expected different seeds to produce different root features due to "
            f"random tie-breaking, but only saw features: {unique_root_features}"
        )


class TestReservoirSamplingMathematical:
    """Test the mathematical properties of reservoir sampling directly."""

    def test_reservoir_sampling_uniform_distribution(self):
        """Verify reservoir sampling gives uniform distribution over ties."""
        # Simulate the reservoir sampling logic directly
        rng = np.random.default_rng(12345)
        n_candidates = 5
        n_trials = 10000

        winners = []
        for _ in range(n_trials):
            # Simulate reservoir sampling over n_candidates ties
            best = 0
            for k in range(1, n_candidates):
                # k is 0-indexed candidate, n_ties is k+1
                n_ties = k + 1
                if rng.random() < 1.0 / n_ties:
                    best = k
            winners.append(best)

        counts = Counter(winners)

        # Each candidate should win ~20% of the time
        expected = n_trials / n_candidates
        for candidate in range(n_candidates):
            actual = counts.get(candidate, 0)
            # Allow 20% deviation from expected
            assert abs(actual - expected) < 0.2 * expected, (
                f"Candidate {candidate}: expected ~{expected:.0f}, got {actual}"
            )

    def test_reservoir_sampling_three_ties(self):
        """Test reservoir sampling with exactly 3 ties."""
        rng = np.random.default_rng(54321)
        n_candidates = 3
        n_trials = 10000

        winners = []
        for _ in range(n_trials):
            best = 0
            n_ties = 1
            for k in range(1, n_candidates):
                n_ties += 1
                if rng.random() < 1.0 / n_ties:
                    best = k
            winners.append(best)

        counts = Counter(winners)

        # Each of 3 candidates should win ~33% of the time
        expected = n_trials / n_candidates
        for candidate in range(n_candidates):
            actual = counts.get(candidate, 0)
            assert abs(actual - expected) < 0.15 * expected, (
                f"Candidate {candidate}: expected ~{expected:.0f}, got {actual}"
            )
