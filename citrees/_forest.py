import warnings
from abc import ABCMeta, abstractmethod
from math import ceil
from multiprocessing import cpu_count
from typing import Optional, Literal, Union

import numpy as np
from pydantic import NonNegativeInt, validator
from pydantic.main import ModelField, ModelMetaclass
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder

from ._threshold_method import ThresholdMethods
from ._tree import BaseConditionalInferenceTreeEstimator, BaseConditionalInferenceTreeParameters, ConditonalInferenceTreeClassifier, ProbabilityFloat
from ._utils import calculate_max_value


class BaseConditionalInferenceForestParameters(BaseConditionalInferenceTreeParameters):
    """Model for BaseConditionalInferenceForest parameters."""

    n_estimators: int
    bootstrap_method: Optional[Literal["bayesian", "classic"]]
    max_samples: Optional[Union[NonNegativeInt, ProbabilityFloat]]
    n_jobs: int


class ConditionalInferenceForestClassifierParameters(BaseConditionalInferenceForestParameters):
    """Model for ConditionalInferenceForestClassifier parameters."""

    class_weight: Optional[Literal["balanced", "balanced_subsample"]]


class BaseConditionalInferenceForest(BaseConditionalInferenceTreeEstimator, metaclass=ABCMeta):
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
        feature_scanning: bool,
        threshold_scanning: bool,
        threshold_method: str,
        max_thresholds: Optional[Union[str, float, int]],
        max_depth: Optional[int],
        max_features: Optional[Union[str, float, int]],
        min_samples_split: int,
        min_samples_leaf: int,
        min_impurity_decrease: float,
        bootstrap_method: Optional[str],
        max_samples: Optional[Union[int, float]],
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
        self.feature_scanning = feature_scanning
        self.threshold_scanning = threshold_scanning
        self.threshold_method = threshold_method
        self.max_thresholds = max_thresholds
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap_method = bootstrap_method
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self._validate_params({**self.get_params(), "estimator_type": self._estimator_type})

    @property
    def _parameter_validator(self) -> ModelMetaclass:
        """Model for hyperparameter validation."""
        return BaseConditionalInferenceForestParameters

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseConditionalInferenceForest":
        """ADD HERE.

        Parameters
        ----------

        Returns
        -------
        """
        X, y = self._validate_data(X, y, self._estimator_type)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ADD HERE.

        Parameters
        ----------

        Returns
        -------
        """
        pass
