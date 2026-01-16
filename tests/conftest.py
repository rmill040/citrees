import os
import sys

# Conditional JIT disabling:
# - JIT is DISABLED when running with coverage (--cov flag) for accurate line tracking
# - JIT is ENABLED by default for fast tests that validate compiled code
# - Users can explicitly control JIT via NUMBA_DISABLE_JIT environment variable

if "NUMBA_DISABLE_JIT" not in os.environ:
    # Check if pytest-cov is active by looking for --cov in args
    if any("--cov" in arg for arg in sys.argv):
        # Disable JIT when running with coverage
        os.environ["NUMBA_DISABLE_JIT"] = "1"
    # Otherwise, JIT stays enabled (Numba's default behavior)
# If NUMBA_DISABLE_JIT is set explicitly, respect the user's choice
