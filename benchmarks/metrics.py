import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


@dataclass
class MetricResult:
    name: str
    value: float
    higher_is_better: bool


CLASSIFICATION_METRICS: dict[str, tuple[Callable, bool]] = {
    "accuracy": (accuracy_score, True),
    "balanced_accuracy": (balanced_accuracy_score, True),
    "f1_macro": (lambda y, yh: f1_score(y, yh, average="macro"), True),
    "f1_weighted": (lambda y, yh: f1_score(y, yh, average="weighted"), True),
}

CLASSIFICATION_PROBA_METRICS: dict[str, tuple[Callable, bool]] = {
    "roc_auc": (lambda y, p: roc_auc_score(y, p[:, 1]) if p.shape[1] == 2 else roc_auc_score(y, p, multi_class="ovr"), True),
    "log_loss": (log_loss, False),
}

REGRESSION_METRICS: dict[str, tuple[Callable, bool]] = {
    "mse": (mean_squared_error, False),
    "rmse": (lambda y, yh: np.sqrt(mean_squared_error(y, yh)), False),
    "mae": (mean_absolute_error, False),
    "r2": (r2_score, True),
}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    task: str = "classification",
) -> dict[str, MetricResult]:
    results = {}

    if task == "classification":
        for name, (fn, higher) in CLASSIFICATION_METRICS.items():
            try:
                value = fn(y_true, y_pred)
                results[name] = MetricResult(name, float(value), higher)
            except Exception:
                pass

        if y_proba is not None:
            for name, (fn, higher) in CLASSIFICATION_PROBA_METRICS.items():
                try:
                    value = fn(y_true, y_proba)
                    results[name] = MetricResult(name, float(value), higher)
                except Exception:
                    pass
    else:
        for name, (fn, higher) in REGRESSION_METRICS.items():
            try:
                value = fn(y_true, y_pred)
                results[name] = MetricResult(name, float(value), higher)
            except Exception:
                pass

    return results


def importance_correlation(
    estimated: np.ndarray,
    true: np.ndarray,
) -> float:
    return float(np.corrcoef(estimated, true)[0, 1])


class Timer:
    def __init__(self):
        self.start_time: float | None = None
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any):
        self.elapsed = time.perf_counter() - self.start_time


METRICS = {
    "classification": list(CLASSIFICATION_METRICS.keys()) + list(CLASSIFICATION_PROBA_METRICS.keys()),
    "regression": list(REGRESSION_METRICS.keys()),
}
