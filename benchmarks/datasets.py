from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    task: Literal["classification", "regression"]
    feature_names: list[str] | None = None
    n_classes: int | None = None
    n_informative: int | None = None
    true_importances: np.ndarray | None = None
    source: str | None = None

    n_samples: int = field(init=False)
    n_features: int = field(init=False)

    def __post_init__(self):
        self.n_samples, self.n_features = self.X.shape
        if self.task == "classification" and self.n_classes is None:
            self.n_classes = len(np.unique(self.y))
        if self.feature_names is None:
            self.feature_names = [f"X{i}" for i in range(self.n_features)]

    def cv_splits(self, n_splits: int = 5, seed: int = 42):
        if self.task == "classification":
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(cv.split(self.X, self.y))


def make_synthetic_clf(
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 5,
    n_redundant: int = 5,
    n_classes: int = 2,
    noise: float = 0.1,
    correlation: float = 0.8,
    seed: int = 42,
) -> Dataset:
    rng = np.random.default_rng(seed)

    X_inf = rng.standard_normal((n_samples, n_informative))

    X_red = np.zeros((n_samples, n_redundant))
    for i in range(n_redundant):
        parent = i % n_informative
        eps = rng.standard_normal(n_samples) * np.sqrt(1 - correlation**2)
        X_red[:, i] = correlation * X_inf[:, parent] + eps

    n_noise = n_features - n_informative - n_redundant
    X_noise = rng.standard_normal((n_samples, max(0, n_noise)))

    X = np.hstack([X_inf, X_red, X_noise])

    signal = X_inf @ rng.standard_normal(n_informative)
    signal += rng.standard_normal(n_samples) * noise

    if n_classes == 2:
        y = (signal > np.median(signal)).astype(int)
    else:
        thresholds = np.percentile(signal, np.linspace(0, 100, n_classes + 1)[1:-1])
        y = np.digitize(signal, thresholds)

    true_imp = np.zeros(n_features)
    true_imp[:n_informative] = 1.0 / n_informative
    true_imp[n_informative : n_informative + n_redundant] = 0.5 / max(1, n_redundant)
    true_imp /= true_imp.sum()

    return Dataset(
        name=f"synthetic_clf_n{n_samples}_p{n_features}",
        X=X,
        y=y,
        task="classification",
        n_classes=n_classes,
        n_informative=n_informative,
        true_importances=true_imp,
        source="synthetic",
    )


def make_synthetic_reg(
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 5,
    n_redundant: int = 5,
    noise: float = 1.0,
    correlation: float = 0.8,
    heteroscedastic: bool = False,
    seed: int = 42,
) -> Dataset:
    rng = np.random.default_rng(seed)

    X_inf = rng.standard_normal((n_samples, n_informative))

    X_red = np.zeros((n_samples, n_redundant))
    for i in range(n_redundant):
        parent = i % n_informative
        eps = rng.standard_normal(n_samples) * np.sqrt(1 - correlation**2)
        X_red[:, i] = correlation * X_inf[:, parent] + eps

    n_noise = n_features - n_informative - n_redundant
    X_noise = rng.standard_normal((n_samples, max(0, n_noise)))

    X = np.hstack([X_inf, X_red, X_noise])

    y = X_inf @ (rng.standard_normal(n_informative) * 2)

    if heteroscedastic:
        noise_scale = 0.5 + np.abs(X[:, 0])
        y += rng.standard_normal(n_samples) * noise_scale * noise
    else:
        y += rng.standard_normal(n_samples) * noise

    true_imp = np.zeros(n_features)
    true_imp[:n_informative] = 1.0 / n_informative
    true_imp[n_informative : n_informative + n_redundant] = 0.5 / max(1, n_redundant)
    true_imp /= true_imp.sum()

    return Dataset(
        name=f"synthetic_reg_n{n_samples}_p{n_features}",
        X=X,
        y=y,
        task="regression",
        n_informative=n_informative,
        true_importances=true_imp,
        source="synthetic",
    )


OPENML_IDS = {
    # classification
    "adult": 1590,
    "credit-g": 31,
    "diabetes": 37,
    "electricity": 44120,
    "covertype": 44121,
    "madelon": 1485,
    "gisette": 41026,
    # regression
    "california_housing": 44027,
    "cpu_act": 44132,
    "wine_quality": 44136,
}


def load_openml(name: str, cache_dir: Path | None = None) -> Dataset:
    import openml
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    cache_dir = cache_dir or Path.home() / ".cache" / "citrees"
    cache_path = cache_dir / f"{name}.npz"

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return Dataset(
            name=name,
            X=data["X"],
            y=data["y"],
            task=str(data["task"]),
            source="openml",
        )

    dataset_id = OPENML_IDS.get(name)
    if dataset_id is None:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(OPENML_IDS.keys())}")

    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    X = np.array(X)
    y = np.array(y)

    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)

    task = "classification" if len(np.unique(y)) <= 100 else "regression"

    X = StandardScaler().fit_transform(X)
    X = np.nan_to_num(X, nan=0.0)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X=X, y=y, task=task)

    return Dataset(name=name, X=X, y=y, task=task, source="openml")


DATASETS = {
    "synthetic_clf_easy": lambda: make_synthetic_clf(n_features=20, n_informative=5, n_redundant=0),
    "synthetic_clf_hard": lambda: make_synthetic_clf(
        n_features=100, n_informative=10, n_redundant=20, correlation=0.9
    ),
    "synthetic_clf_highdim": lambda: make_synthetic_clf(
        n_samples=500, n_features=500, n_informative=10, n_redundant=50
    ),
    "synthetic_reg_easy": lambda: make_synthetic_reg(n_features=20, n_informative=5, n_redundant=0),
    "synthetic_reg_hard": lambda: make_synthetic_reg(
        n_features=100, n_informative=10, n_redundant=20, correlation=0.9
    ),
    "synthetic_reg_hetero": lambda: make_synthetic_reg(heteroscedastic=True),
}


def get_dataset(name: str) -> Dataset:
    if name in DATASETS:
        return DATASETS[name]()
    if name in OPENML_IDS:
        return load_openml(name)
    raise ValueError(f"Unknown dataset: {name}")


def list_datasets() -> list[str]:
    return sorted(list(DATASETS.keys()) + list(OPENML_IDS.keys()))


class DatasetRegistry:
    get = staticmethod(get_dataset)
    list = staticmethod(list_datasets)
