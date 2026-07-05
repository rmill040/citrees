"""Shared evaluation metrics for feature selection experiments."""

from __future__ import annotations


def precision_at_k(ranking: list[int], true_indices: list[int], k: int) -> float:
    """Compute precision@k.

    Parameters
    ----------
    ranking : list[int]
        Feature indices sorted by importance (best first).
    true_indices : list[int]
        Ground truth informative feature indices.
    k : int
        Number of top features to consider.

    Returns
    -------
    float
        Precision@k: |top_k ∩ true| / k
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if k == 0:
        return 0.0
    top_k = set(ranking[:k])
    true_set = set(true_indices)
    return len(top_k & true_set) / k


def recall_at_k(ranking: list[int], true_indices: list[int], k: int) -> float:
    """Compute recall@k.

    Parameters
    ----------
    ranking : list[int]
        Feature indices sorted by importance (best first).
    true_indices : list[int]
        Ground truth informative feature indices.
    k : int
        Number of top features to consider.

    Returns
    -------
    float
        Recall@k: |top_k ∩ true| / |true|
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if len(true_indices) == 0:
        return 0.0
    top_k = set(ranking[:k])
    true_set = set(true_indices)
    return len(top_k & true_set) / len(true_set)


def f1_at_k(ranking: list[int], true_indices: list[int], k: int) -> float:
    """Compute F1@k.

    Parameters
    ----------
    ranking : list[int]
        Feature indices sorted by importance (best first).
    true_indices : list[int]
        Ground truth informative feature indices.
    k : int
        Number of top features to consider.

    Returns
    -------
    float
        F1@k: harmonic mean of precision@k and recall@k
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    p = precision_at_k(ranking, true_indices, k)
    r = recall_at_k(ranking, true_indices, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def jaccard_at_k(ranking: list[int], true_indices: list[int], k: int) -> float:
    """Compute Jaccard similarity at k.

    Parameters
    ----------
    ranking : list[int]
        Feature indices sorted by importance (best first).
    true_indices : list[int]
        Ground truth informative feature indices.
    k : int
        Number of top features to consider.

    Returns
    -------
    float
        Jaccard@k: |top_k ∩ true| / |top_k ∪ true|
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    top_k = set(ranking[:k])
    true_set = set(true_indices)
    intersection = len(top_k & true_set)
    union = len(top_k | true_set)
    if union == 0:
        return 0.0
    return intersection / union
