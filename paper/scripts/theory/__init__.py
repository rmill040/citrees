"""
Theory module for feature muting power analysis.

This module provides tools for:
- Exact power function calculations for correlation tests
- Gap region boundary computation
- Sample size randomness integration
- Soft/noisy gate model analysis
- Multi-feature competition analysis
- Tree depth propagation analysis
"""

from .theoretical_predictions import (
    # Core power functions
    exact_r_critical,
    power_correlation_test,
    root_power,
    gate_power,
    # Gap region
    GapRegion,
    find_gap_region,
    gap_region_table,
    minimum_n_for_gap,
    # Marginal power
    marginal_gate_power_exact,
    marginal_gate_power_approx,
    # Soft gate
    soft_gate_prob,
    calibrate_soft_gate,
    # Noisy gate
    noisy_gate_true_positive_rate,
    noisy_gate_correlations,
    # Multi-feature
    prob_feature_selected,
    # Tree depth
    critical_depth,
    power_at_depth,
    # Constants
    RHO_GATE,
    RHO_ROOT_COEF,
)

__all__ = [
    "exact_r_critical",
    "power_correlation_test",
    "root_power",
    "gate_power",
    "GapRegion",
    "find_gap_region",
    "gap_region_table",
    "minimum_n_for_gap",
    "marginal_gate_power_exact",
    "marginal_gate_power_approx",
    "soft_gate_prob",
    "calibrate_soft_gate",
    "noisy_gate_true_positive_rate",
    "noisy_gate_correlations",
    "prob_feature_selected",
    "critical_depth",
    "power_at_depth",
    "RHO_GATE",
    "RHO_ROOT_COEF",
]
