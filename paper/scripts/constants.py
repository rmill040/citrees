"""Shared constants for experiment scripts.

Single source of truth for method lists, defaults, and OpenML dataset IDs.
"""

from __future__ import annotations

import os

# =============================================================================
# Random Seeds
# =============================================================================

RANDOM_STATE = 1718
N_SEEDS = 10
N_SPLITS = 5

# =============================================================================
# Timeouts
# =============================================================================

STALE_TIMEOUT_MINUTES = 30  # Stage 1 (feature selection)
EVAL_STALE_TIMEOUT_MINUTES = 60  # Stage 2 (evaluation)

# =============================================================================
# AWS Configuration
# =============================================================================

S3_BUCKET = os.environ.get("S3_BUCKET", "citrees-results")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# =============================================================================
# Feature Selection Methods
# =============================================================================

# Classification methods (18 total)
CLF_METHODS = [
    # Filter methods
    "mc",  # Multiple correlation (ANOVA-based)
    "mi",  # Mutual information
    "rdc",  # Randomized dependence coefficient
    "mrmr",  # Minimum Redundancy Maximum Relevance
    # Permutation test methods
    "ptest_mc",
    "ptest_mi",
    "ptest_rdc",
    # Embedding methods (tree-based)
    "cit",  # Conditional Inference Tree
    "cif",  # Conditional Inference Forest
    "rf",  # Random Forest
    "et",  # Extra Trees
    "xgb",  # XGBoost
    "lgbm",  # LightGBM
    "cat",  # CatBoost
    # Wrapper methods
    "boruta",
    "pi",  # Permutation importance
    "shap",
    "rfe",  # Recursive Feature Elimination
]

# Regression methods (16 total)
REG_METHODS = [
    # Filter methods
    "pc",  # Pearson correlation
    "dc",  # Distance correlation
    "rdc",  # Randomized dependence coefficient
    # Permutation test methods
    "ptest_pc",
    "ptest_dc",
    "ptest_rdc",
    # Embedding methods (tree-based)
    "cit",
    "cif",
    "rf",
    "et",
    "xgb",
    "lgbm",
    "cat",
    # Wrapper methods
    "boruta",
    "pi",
    "rfe",  # Recursive Feature Elimination
]

# Embedding methods (have feature_importances_ and can make predictions)
EMBEDDING_METHODS = {"cit", "cif", "rf", "et", "xgb", "lgbm", "cat"}

# =============================================================================
# Downstream Models
# =============================================================================

CLF_DOWNSTREAM_MODELS = ["lr", "svm", "knn"]
REG_DOWNSTREAM_MODELS = ["ridge", "svr", "knn"]

# =============================================================================
# OpenML Dataset IDs
# =============================================================================

OPENML_IDS = {
    # Classification
    "adult": 1590,
    "credit-g": 31,
    "diabetes": 37,
    "electricity": 44120,
    "covertype": 44121,
    "madelon": 1485,
    "gisette": 41026,
    # Regression
    "california_housing": 44027,
    "cpu_act": 44132,
    "wine_quality": 44136,
}

# =============================================================================
# Default Hyperparameters
# =============================================================================

DEFAULT_PARAMS = {
    "rf": {"n_estimators": 100, "n_jobs": -1, "random_state": RANDOM_STATE},
    "et": {"n_estimators": 100, "n_jobs": -1, "random_state": RANDOM_STATE},
    "xgb": {"n_estimators": 100, "n_jobs": -1, "random_state": RANDOM_STATE, "verbosity": 0},
    "lgbm": {"n_estimators": 100, "n_jobs": -1, "random_state": RANDOM_STATE, "verbose": -1},
    "cat": {"n_estimators": 100, "random_state": RANDOM_STATE, "verbose": 0},
    "cit": {"random_state": RANDOM_STATE},
    "cif": {"n_estimators": 100, "n_jobs": -1, "random_state": RANDOM_STATE},
}
