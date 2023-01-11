import warnings
from abc import ABCMeta, abstractmethod
from math import ceil
from multiprocessing import cpu_count
from typing import Optional, Union

import numpy as np
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder

from ._selector import ClassifierSelectors, ClassifierSelectorTests, RegressorSelectors, RegressorSelectorTests
from ._splitter import ClassifierSplitters, ClassifierSplitterTests, RegressorSplitters, RegressorSplitterTests
from ._threshold_method import ThresholdMethods
from ._tree import _PVAL_PRECISION, ConditonalInferenceTreeClassifier
from ._utils import calculate_max_value


class BaseConditionalInferenceForest(BaseEstimator, metaclass=ABCMeta):
    """Base class for conditional inference forests.

    Warning: This class should not be used directly. Use derived classes instead.
    """

    @abstractmethod
    def __init__(
        self,
        *,
        n_estimators: int,
        selector: str,
        splitter: str,
        alpha_selector: float,
        alpha_splitter: float,
        adjust_alpha_selector: bool,
        adjust_alpha_splitter: bool,
        n_resamples_selector: Union[str, int],
        n_resamples_splitter: Union[str, int],
        early_stopping_selector: bool,
        early_stopping_splitter: bool,
        feature_muting: bool,
        threshold_method: str,
        max_thresholds: Optional[Union[str, float, int]],
        max_depth: Optional[int],
        max_features: Optional[Union[str, float, int]],
        min_samples_split: int,
        min_samples_leaf: int,
        min_impurity_decrease: float,
        bootstrap_method: Optional[str],
        max_samples: Optional[Union[int, float]],
        class_weight: None,
        n_jobs: int,
        random_state: Optional[int],
        verbose: int,
    ) -> None:
        self.n_estimators = n_estimators
        self.selector = selector
        self.splitter = splitter
        self.alpha_selector = alpha_selector
        self.alpha_splitter = alpha_splitter
        self.adjust_alpha_selector = adjust_alpha_selector
        self.adjust_alpha_splitter = adjust_alpha_splitter
        self.n_resamples_selector = n_resamples_selector
        self.n_resamples_splitter = n_resamples_splitter
        self.early_stopping_selector = early_stopping_selector
        self.early_stopping_splitter = early_stopping_splitter
        self.feature_muting = feature_muting
        self.threshold_method = threshold_method
        self.max_thresholds = max_thresholds
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap_method = bootstrap_method
        self.max_samples = max_samples
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseConditionalInferenceForest":
        """ADD HERE.

        Parameters
        ----------

        Returns
        -------
        """
        # Define types for mypy
        self._splitter_test: Any
        self._selector_test: Any
        self._min_samples_leaf: int

        if self._estimator_type == "classifier":
            selector_registry = ClassifierSelectors
            splitter_registry = ClassifierSplitters
        else:
            selector_registry = RegressorSelectors
            splitter_registry = RegressorSplitters

        for attribute, registry in zip(
            ["selector", "splitter", "threshold_method"], [selector_registry, splitter_registry, ThresholdMethods]
        ):
            value = getattr(self, attribute)
            if value not in registry.keys():
                raise ValueError(
                    f"{attribute} ({value}) not supported for ({self._estimator_type}) estimators, expected one of: "
                    f"{registry.keys()}"
                )
            setattr(self, f"_{attribute}", registry[value])

        for attribute in ["alpha_selector", "alpha_splitter"]:
            value = getattr(self, attribute)
            if not 0 < value <= 1.0:
                raise ValueError(f"{attribute} ({value}) should be in range (0, 1.0]")
            setattr(self, f"_{attribute}", value)

        for attribute in [
            "adjust_alpha_selector",
            "adjust_alpha_splitter",
            "early_stopping_selector",
            "early_stopping_splitter",
            "feature_muting",
        ]:
            value = getattr(self, attribute)
            if type(value) != bool:
                raise TypeError(f"{attribute} type ({value}) should be bool")
            setattr(self, f"_{attribute}", value)

        for attribute in ["n_resamples_selector", "n_resamples_splitter"]:
            value = getattr(self, attribute)
            if type(value) == str:
                supported = ["minimum", "auto"]
                if value not in supported:
                    raise ValueError(f"{attribute} ({value}) not supported, expected one of: {supported}")
                alpha = getattr(self, "alpha_selector") if "selector" in attribute else getattr(self, "alpha_splitter")
                lower_limit = ceil(1 / alpha)
                if value == "minimum":
                    value = lower_limit
                else:
                    # Approximate upper limit
                    z = norm.ppf(1 - alpha)
                    upper_limit = ceil(z * z * (alpha * (1 - alpha)) / (_PVAL_PRECISION * _PVAL_PRECISION))
                    value = max(lower_limit, upper_limit)
            else:
                value = max(2, int(value))
            setattr(self, f"_{attribute}", value)

        for attribute, dtype, lower_limit in zip(  # type: ignore
            ["min_samples_split", "min_samples_leaf", "min_impurity_decrease"],
            [int, int, float],
            [2, 1, 0.0],
        ):
            value = getattr(self, attribute)
            if type(value) != dtype:
                warnings.warn(f"{attribute} data type ({type(value)}) should be {dtype}, attempting to cast data type")
                value = dtype(value)
            if value < lower_limit:
                raise ValueError(f"{attribute} ({value}) should be >= {lower_limit}")
            setattr(self, f"_{attribute}", value)

        if self._min_samples_leaf >= self._min_samples_split:
            warnings.warn(
                f"min_samples_leaf ({self._min_samples_leaf}) should be < min_samples_split "
                f"({self._min_samples_split}), setting min_samples_leaf = min_samples_split - 1"
            )
            self._min_samples_leaf = self._min_samples_split - 1

        value = self.max_depth
        if type(value) in (int, float):
            if value < 1:
                raise ValueError(f"max_depth ({value}) should be > 1")
        elif value is None:
            value = np.inf
        setattr(self, "_max_depth", value)

        self._random_state = int(np.random.randint(1, 100_000)) if self.random_state is None else self.random_state
        self._verbose = min(abs(int(self.verbose)), 3)

        X = X.astype(float)
        if self._estimator_type == "classifier":
            self._label_encoder = LabelEncoder()
            y = self._label_encoder.fit_transform(y)
        self.feature_names_in_ = None
        if not isinstance(X, np.ndarray):
            if isinstance(X, (list, tuple)):
                X = np.array(X)
            elif hasattr(X, "values"):
                self.feature_names_in_ = X.columns.tolist()
                X = X.values
            else:
                raise ValueError(
                    f"Unsupported type for X, got ({type(X)}) but expected np.ndarray, list, tuple, or pandas data "
                    "structure"
                )

        if X.ndim == 1:
            X = X[:, None]
        elif X.ndim > 2:
            raise ValueError(
                f"Arrays with more than 2 dimensions are not supported for X, detected ({X.ndim}) dimensions"
            )

        if not isinstance(y, np.ndarray):
            if isinstance(y, (list, tuple)):
                y = np.array(y)
            elif hasattr(y, "values"):
                y = y.values
            else:
                raise ValueError(
                    f"Unsupported type for y, got ({type(y)}) but expected np.ndarray, list, tuple, or pandas data "
                    "structure"
                )

        if y.ndim == 2:
            y = y.ravel()
        elif y.ndim > 2:
            raise ValueError(f"Multi-output labels are not supported for y, detected ({y.ndim - 1}) outputs")

        if len(X) != len(y):
            raise ValueError(f"Different number of samples between X ({len(X)}) and y ({len(y)})")

        self.n_features_in_ = X.shape[1]
        if self.feature_names_in_ is None:
            self.feature_names_in_ = [f"f{j}" for j in range(self.n_features_in_)]

        for attribute in ["max_features", "max_thresholds"]:
            value = getattr(self, attribute)
            if type(value) == str:
                supported = ["sqrt", "log2"]
                if value not in supported:
                    raise ValueError(f"{attribute} ({value}) not supported, expected one of: {supported}")
            elif type(value) == float:
                if not 0 < value <= 1.0:
                    raise ValueError(f"{attribute} ({value}) should be in range (0, 1.0]")
            elif type(value) == int:
                if value < 1:
                    raise ValueError(f"{attribute} ({value}) should be >= 1")
            if attribute == "max_features":
                value = (
                    calculate_max_value(n_values=self.n_features_in_, desired_max=value)
                    if value is not None
                    else self.n_features_in_
                )
            setattr(self, f"_{attribute}", value)

        # Private and fitted estimator attributes
        self._available_features = np.arange(self.n_features_in_, dtype=int)
        if self._estimator_type == "classifier":
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            self._selector_test = ClassifierSelectorTests[self.selector]
            self._selector_kwargs = {
                "n_classes": self.n_classes_,
                "n_resamples": self._n_resamples_selector,
                "early_stopping": self._early_stopping_selector,
                "alpha": self._alpha_selector,
                "random_state": self._random_state,
            }
            self._splitter_test = ClassifierSplitterTests[self.splitter]
            self._splitter_kwargs = {
                "n_resamples": self._n_resamples_splitter,
                "early_stopping": self._early_stopping_splitter,
                "alpha": self._alpha_splitter,
                "random_state": self._random_state,
            }
        else:
            # TODO:
            self._selector_test = RegressorSelectorTests[self.selector]
            self._selector_kwargs = {}
            self._splitter_test = RegressorSplitterTests[self.splitter]
            self._splitter_kwargs = {}

        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)

        # Start recursion
        self.tree_ = self._build_tree(X, y)

        return self

        # if self.n_estimators < 1:
        #     raise ValueError(
        #         f"n_estimators ({self.n_estimators}) should be > 1"
        #     )
        # self._n_estimators = self.n_estimators

        # if self.n_jobs is None:
        #     self._n_jobs = 1
        # else:
        #     max_cpus = cpu_count()
        #     value = min(self.n_jobs, max_cpus)
        #     if value < 0:
        #         cpus = np.arange(1, max_cpus + 1)
        #         if abs(value) > max_cpus:
        #             value = max_cpus
        #         else:
        #             value = cpus[value]
        #     self._n_jobs = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ADD HERE.

        Parameters
        ----------

        Returns
        -------
        """
        pass
