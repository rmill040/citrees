import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone

from benchmarks.datasets import Dataset, get_dataset
from benchmarks.metrics import Timer, compute_metrics, importance_correlation


@dataclass
class ExperimentResult:
    dataset: str
    model: str
    fold: int
    metrics: dict[str, float]
    fit_time: float
    predict_time: float
    importance_corr: float | None = None
    config: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ExperimentConfig:
    datasets: list[str]
    models: dict[str, BaseEstimator]
    n_splits: int = 5
    seed: int = 42
    output_dir: Path = Path("benchmarks/results")


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: list[ExperimentResult] = []

    def run_single(
        self,
        dataset: Dataset,
        model: BaseEstimator,
        model_name: str,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        fold: int,
    ) -> ExperimentResult:
        X_train, X_test = dataset.X[train_idx], dataset.X[test_idx]
        y_train, y_test = dataset.y[train_idx], dataset.y[test_idx]

        m = clone(model)

        with Timer() as fit_timer:
            m.fit(X_train, y_train)

        with Timer() as pred_timer:
            y_pred = m.predict(X_test)

        y_proba = None
        if hasattr(m, "predict_proba") and dataset.task == "classification":
            y_proba = m.predict_proba(X_test)

        metrics = compute_metrics(y_test, y_pred, y_proba, dataset.task)

        imp_corr = None
        if dataset.true_importances is not None and hasattr(m, "feature_importances_"):
            imp_corr = importance_correlation(m.feature_importances_, dataset.true_importances)

        return ExperimentResult(
            dataset=dataset.name,
            model=model_name,
            fold=fold,
            metrics={k: v.value for k, v in metrics.items()},
            fit_time=fit_timer.elapsed,
            predict_time=pred_timer.elapsed,
            importance_corr=imp_corr,
            config=m.get_params() if hasattr(m, "get_params") else {},
        )

    def run(self) -> list[ExperimentResult]:
        self.results = []

        for ds_name in self.config.datasets:
            dataset = get_dataset(ds_name)
            splits = dataset.cv_splits(self.config.n_splits, self.config.seed)

            for model_name, model in self.config.models.items():
                for fold, (train_idx, test_idx) in enumerate(splits):
                    result = self.run_single(dataset, model, model_name, train_idx, test_idx, fold)
                    self.results.append(result)
                    print(f"{ds_name} | {model_name} | fold {fold} | {result.metrics}")

        return self.results

    def save(self, name: str | None = None):
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        name = name or datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.config.output_dir / f"{name}.json"

        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2, default=str)

        return path
