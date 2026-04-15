"""Shared constants for experiment scripts.

Single source of truth for random seeds, timeouts, and downstream models.
Method lists are defined in paper.scripts.pipeline.methods.
"""

from __future__ import annotations

import os

# =============================================================================
# Random Seeds
# =============================================================================

RANDOM_STATE = 1718
N_SEEDS = 5
N_SPLITS = 5

# =============================================================================
# AWS Configuration
# =============================================================================

# S3_BUCKET: Derived as citrees-{account_id} by aws.py, required for distributed runs
S3_BUCKET = os.environ.get("S3_BUCKET", "")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# =============================================================================
# Downstream Models
# =============================================================================

CLF_DOWNSTREAM_MODELS = ["lr", "svm", "knn"]
REG_DOWNSTREAM_MODELS = ["ridge", "svr", "knn"]
EVALUATION_K_VALUES = [5, 10, 25, 50, 100]
HIGH_P_EVALUATION_P_THRESHOLD = max(EVALUATION_K_VALUES)
HIGH_P_EVALUATION_EXTRA_K_VALUES = [150, 200, 300, 500, 750, 1000]
HIGH_P_EVALUATION_EXTRA_K_FRACTIONS = [0.25, 0.5, 0.75]

# =============================================================================
# OpenML Dataset IDs
# =============================================================================

OPENML_IDS = {
    # Classification
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
    "xgb": {"n_estimators": 500, "n_jobs": -1, "random_state": RANDOM_STATE, "verbosity": 0},
    "lgbm": {"n_estimators": 500, "n_jobs": -1, "random_state": RANDOM_STATE, "verbose": -1},
    "cat": {"n_estimators": 500, "random_state": RANDOM_STATE, "verbose": 0},
    "cit": {"random_state": RANDOM_STATE},
    "cif": {"n_estimators": 100, "n_jobs": -1, "random_state": RANDOM_STATE},
}
