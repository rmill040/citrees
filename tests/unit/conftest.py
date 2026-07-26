"""Shared helpers for unit tests.

The JIT parity helpers let each module's test file verify its own Numba
kernels: ``@njit`` functions compile to machine code, so the compiled kernel
and the Python function it was written as are two implementations that can
silently diverge. The suite runs one or the other depending on
``NUMBA_DISABLE_JIT`` (coverage runs disable it), so neither mode alone proves
they agree.
"""

from typing import Any

import numpy as np
import pytest
from numba.core.registry import CPUDispatcher


def _assert_equal(jit_result: Any, py_result: Any) -> None:
    """Compare kernel results, handling scalars, arrays, and tuples of arrays.

    Tolerances are loose enough to absorb ``fastmath=True`` reassociation,
    which may round differently per architecture, while still failing on any
    real divergence between the compiled and Python implementations.
    """
    if isinstance(jit_result, tuple):
        assert len(jit_result) == len(py_result)
        for a, b in zip(jit_result, py_result, strict=True):
            _assert_equal(a, b)
    elif isinstance(jit_result, np.ndarray):
        np.testing.assert_allclose(jit_result, py_result, rtol=1e-8, atol=1e-8)
    else:
        assert jit_result == pytest.approx(py_result, rel=1e-8, abs=1e-8)


@pytest.fixture
def assert_jit_parity():
    """Assert a compiled Numba kernel and its ``py_func`` return identical values."""

    def check(fn: Any, *args: Any, **kwargs: Any) -> None:
        assert isinstance(fn, CPUDispatcher), f"{fn} is not JIT compiled"
        _assert_equal(fn(*args, **kwargs), fn.py_func(*args, **kwargs))

    return check


@pytest.fixture
def assert_all_kernels_covered():
    """Assert every ``@njit`` kernel in a module has a parity case.

    Fails when a new Numba kernel is added to the module without a
    corresponding entry in that test file's kernel table.
    """

    def check(module: Any, covered: set[Any]) -> None:
        missing = [
            f"{module.__name__}.{name}"
            for name in dir(module)
            if isinstance(getattr(module, name), CPUDispatcher)
            and getattr(module, name) not in covered
        ]
        assert not missing, f"Numba kernels without a JIT/py_func parity case: {sorted(missing)}"

    return check
