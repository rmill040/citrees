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

from .muting_power_theory import (
    # Constants
    RHO_GATE,
    RHO_ROOT_COEF,
    # Gap region
    GapRegion,
    calibrate_soft_gate,
    # Tree depth
    critical_depth,
    # Core power functions
    exact_r_critical,
    find_gap_region,
    gap_region_table,
    gate_power,
    marginal_gate_power_approx,
    # Marginal power
    marginal_gate_power_exact,
    minimum_n_for_gap,
    noisy_gate_correlations,
    # Noisy gate
    noisy_gate_true_positive_rate,
    power_at_depth,
    power_correlation_test,
    # Multi-feature
    prob_feature_selected,
    root_power,
    # Soft gate
    soft_gate_prob,
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
