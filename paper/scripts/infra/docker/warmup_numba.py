"""Pre-compile Numba caches for dcor and citrees during Docker build.

Run once at image build time so that workers load cached artifacts
instead of racing to JIT-compile simultaneously (which causes LLVM
SIGABRT on concurrent dcor @guvectorize compilation).
"""

import numpy as np


def warmup_dcor() -> None:
    """Trigger dcor's internal @guvectorize compilation."""
    from dcor import distance_correlation, distance_covariance

    rng = np.random.default_rng(0)
    x = rng.standard_normal(50)
    y = rng.standard_normal(50)
    distance_correlation(x, y)
    distance_covariance(x, y)
    print("  dcor: OK")


def warmup_citrees() -> None:
    """Import citrees modules to trigger @njit(cache=True) compilation."""
    import citrees._selector  # noqa: F401
    import citrees._splitter  # noqa: F401
    import citrees._sequential  # noqa: F401
    import citrees._threshold_method  # noqa: F401
    import citrees._utils  # noqa: F401

    print("  citrees imports: OK")

    # Trigger actual compilation by calling representative functions
    from citrees import (
        ConditionalInferenceTreeClassifier,
        ConditionalInferenceTreeRegressor,
    )

    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    y_cls = rng.integers(0, 2, size=100)
    y_reg = rng.standard_normal(100)

    clf = ConditionalInferenceTreeClassifier(random_state=0)
    clf.fit(X, y_cls)
    clf.predict(X)
    print("  citrees classifier fit/predict: OK")

    reg = ConditionalInferenceTreeRegressor(random_state=0)
    reg.fit(X, y_reg)
    reg.predict(X)
    print("  citrees regressor fit/predict: OK")


if __name__ == "__main__":
    print("Warming up Numba caches...")
    warmup_dcor()
    warmup_citrees()
    print("Numba cache warmup complete.")
