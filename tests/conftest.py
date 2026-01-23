import os
import sys

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_friedman1, make_regression

# Conditional JIT disabling:
# - JIT is DISABLED when running with coverage (--cov flag) for accurate line tracking
# - JIT is ENABLED by default for fast tests that validate compiled code
# - Users can explicitly control JIT via NUMBA_DISABLE_JIT environment variable

if "NUMBA_DISABLE_JIT" not in os.environ and any("--cov" in arg for arg in sys.argv):
    # Disable JIT when running with coverage
    os.environ["NUMBA_DISABLE_JIT"] = "1"
# Otherwise, JIT stays enabled (Numba's default behavior)
# If NUMBA_DISABLE_JIT is set explicitly, respect the user's choice


# =============================================================================
# REGRESSION TEST FIXTURES
# =============================================================================


@pytest.fixture
def regression_data_friedman1():
    """Nonlinear regression dataset using Friedman #1.

    y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4 + noise
    Features 0-4 are informative, 5-19 are noise.
    """
    X, y = make_friedman1(n_samples=200, n_features=20, noise=1.0, random_state=42)
    return X, y


@pytest.fixture
def regression_data_correlated():
    """Regression dataset with correlated features.

    5 informative + 5 correlated (r=0.9) + 10 noise features.
    """
    rng = np.random.default_rng(42)
    n_samples = 200
    n_informative = 5
    n_correlated = 5
    n_noise = 10

    # Generate informative features
    X_informative = rng.standard_normal((n_samples, n_informative))

    # Generate correlated features (r=0.9 with informative)
    corr = 0.9
    X_correlated = corr * X_informative + np.sqrt(1 - corr**2) * rng.standard_normal(
        (n_samples, n_informative)
    )

    # Generate noise features
    X_noise = rng.standard_normal((n_samples, n_noise))

    # Combine all features
    X = np.hstack([X_informative, X_correlated[:, :n_correlated], X_noise])

    # Generate target (linear combination of informative features)
    beta = rng.standard_normal(n_informative)
    y = X_informative @ beta + rng.standard_normal(n_samples) * 0.5

    return X, y


@pytest.fixture
def regression_data_heteroscedastic():
    """Heteroscedastic regression dataset.

    noise_scale = 1 + 2*|x_0|, so variance grows with the first feature.
    """
    rng = np.random.default_rng(42)
    n_samples = 200
    n_features = 10
    n_informative = 5

    X = rng.standard_normal((n_samples, n_features))

    # Linear signal
    beta = rng.standard_normal(n_informative)
    signal = X[:, :n_informative] @ beta

    # Heteroscedastic noise
    noise_scale = 1 + 2 * np.abs(X[:, 0])
    noise = rng.standard_normal(n_samples) * noise_scale
    y = signal + noise

    return X, y


@pytest.fixture
def regression_data_standard():
    """Standard linear regression dataset."""
    X, y = make_regression(
        n_samples=200,
        n_features=20,
        n_informative=10,
        noise=1.0,
        random_state=42,
    )
    return X, y


# =============================================================================
# CLASSIFICATION TEST FIXTURES
# =============================================================================


@pytest.fixture
def classification_data_simple():
    """Simple binary classification dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    return X, y


@pytest.fixture
def classification_data_multiclass():
    """Multiclass classification dataset."""
    X, y = make_classification(
        n_samples=150,
        n_features=10,
        n_informative=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    return X, y
