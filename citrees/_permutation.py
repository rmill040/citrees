"""Internal permutation-test result collection."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

type PermutationStage = Literal["selector", "splitter"]
type PermutationTestResult = tuple[float, int]
type PermutationCounts = dict[PermutationStage, int]

_ACTIVE_COUNTS: ContextVar[PermutationCounts | None] = ContextVar(
    "citrees_active_permutation_counts",
    default=None,
)


@contextmanager
def collect_permutation_counts() -> Iterator[PermutationCounts]:
    """Collect actual selector and splitter permutations in the current fit."""
    counts: PermutationCounts = {"selector": 0, "splitter": 0}
    token = _ACTIVE_COUNTS.set(counts)
    try:
        yield counts
    finally:
        _ACTIVE_COUNTS.reset(token)


def record_permutation_test(
    stage: PermutationStage,
    result: PermutationTestResult,
) -> float:
    """Record one test result when collection is active and return its p-value."""
    p_value, n_permutations = result
    counts = _ACTIVE_COUNTS.get()
    if counts is not None:
        counts[stage] += n_permutations
    return float(p_value)
