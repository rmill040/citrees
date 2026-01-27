import numpy as np
from sklearn.metrics import roc_auc_score

from paper.scripts.pipeline.stage2 import compute_roc_auc


def test_compute_roc_auc_binary_labels():
    # Labels are {1, 2}; proba column 1 corresponds to class 2.
    y_true = np.array([1, 2, 1, 2, 2, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    classes = np.array([1, 2])

    expected = compute_roc_auc(y_true, y_proba, classes)
    # Manually binarize using class 2 as positive.
    y_bin = (y_true == 2).astype(int)
    manual = roc_auc_score(y_bin, y_proba)
    assert np.isfinite(expected)
    assert expected == manual


def test_compute_roc_auc_single_class_returns_nan():
    y_true = np.array([1, 1, 1, 1])
    y_proba = np.array([0.2, 0.3, 0.1, 0.4])
    classes = np.array([1, 2])
    result = compute_roc_auc(y_true, y_proba, classes)
    assert np.isnan(result)


def test_compute_roc_auc_multiclass_missing_class_returns_nan():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.6, 0.3, 0.1],
            [0.2, 0.7, 0.1],
        ]
    )
    classes = np.array([0, 1, 2])
    result = compute_roc_auc(y_true, y_proba, classes)
    assert np.isnan(result)
