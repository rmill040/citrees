from abc import ABCMeta, abstractmethod
from multiprocessing import cpu_count
from typing import Literal, Optional, Union

import numpy as np
from joblib import delayed, Parallel
from pydantic import PositiveInt
from pydantic.main import ModelMetaclass
from sklearn.base import ClassifierMixin, clone, RegressorMixin
from sklearn.preprocessing import LabelEncoder

from ._tree import (
    BaseConditionalInferenceTreeEstimator,
    BaseConditionalInferenceTreeParameters,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
    ProbabilityFloat,
)
from ._utils import balanced_bootstrap_sample, calculate_max_value, classic_bootstrap_sample, stratify_bootstrap_sample


def _parallel_fit_classifier(
    *,
    tree: ConditionalInferenceTreeClassifier,
    X: np.ndarray,
    y: np.ndarray,
    idx: Optional[np.ndarray],
    n: Optional[int],
    tree_idx: int,
    n_estimators: int,
    bootstrap_method: Optional[str],
    bayesian_bootstrap: bool,
    sampling_method: Optional[str],
    verbose: int,
    random_state: int,
) -> ConditionalInferenceTreeClassifier:
    """Build classification trees in parallel.

    Note: This function can't go locally in a class, because joblib complains that it cannot pickle it when placed there

    Parameters
    ----------

    Returns
    -------
    """
    if verbose:
        if verbose == 1:
            denom = 20
        elif verbose == 2:
            denom = 10
        else:
            denom = 1
        if tree_idx % denom == 0:
            print(f"Building tree {tree_idx}/{n_estimators}")

    # Bootstrap sample if specified
    if bootstrap_method:
        kwargs = {"bayesian_bootstrap": bayesian_bootstrap, "random_state": tree.random_state}
        if sampling_method in ["balanced", "stratify"]:
            boot_idx = (
                balanced_bootstrap_sample(idx_classes=idx, n=n, **kwargs)  # type: ignore
                if sampling_method == "balanced"
                else stratify_bootstrap_sample(idx_classes=idx, **kwargs)  # type: ignore
            )
            boot_idx = np.concatenate(idx)
        else:
            boot_idx = classic_bootstrap_sample(idx=idx, n=n, **kwargs)  # type: ignore
        tree.fit(X[boot_idx], y[boot_idx])
    else:
        tree.fit(X, y)

    return tree


def _parallel_fit_regressor(
    *,
    tree: ConditionalInferenceTreeRegressor,
    X: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    n: int,
    tree_idx: int,
    n_estimators: int,
    bootstrap_method: Optional[str],
    bayesian_bootstrap: bool,
    verbose: int,
    random_state: int,
) -> ConditionalInferenceTreeRegressor:
    """Build regression trees in parallel.

    Note: This function can't go locally in a class, because joblib complains that it cannot pickle it when placed there

    Parameters
    ----------

    Returns
    -------
    """
    if verbose:
        if verbose == 1:
            denom = 20
        elif verbose == 2:
            denom = 10
        else:
            denom = 1
        if tree_idx % denom == 0:
            print(f"Building tree {tree_idx}/{n_estimators}")

    # Bootstrap sample if specified
    if bootstrap_method:
        boot_idx = classic_bootstrap_sample(
            idx=idx, n=n, bayesian_bootstrap=bayesian_bootstrap, random_state=random_state
        )
        tree.fit(X[boot_idx], y[boot_idx])
    else:
        tree.fit(X, y)

    return tree


class BaseConditionalInferenceForestParameters(BaseConditionalInferenceTreeParameters):
    """Model for BaseConditionalInferenceForest parameters."""

    n_estimators: PositiveInt
    bootstrap_method: Optional[Literal["bayesian", "classic"]]
    max_samples: Optional[Union[PositiveInt, ProbabilityFloat]]
    n_jobs: Optional[int]


class ConditionalInferenceForestClassifierParameters(BaseConditionalInferenceForestParameters):
    """Model for ConditionalInferenceForestClassifier parameters."""

    sampling_method: Optional[Literal["balanced", "stratify"]]


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
        n_jobs: Optional[int],
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
        """Model for hyperparameter validation.

        Returns
        -------
        ModelMetaclass
            Model to validate hyperparameters.
        """
        return BaseConditionalInferenceForestParameters

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseConditionalInferenceForest":
        """Train estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.

        y : array-like of shape (n_samples,)
            Training target.

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = self._validate_data(X, y, self._estimator_type)
        n, p = X.shape

        # Private attributes for all parameters - for consistency to reference across other classes and methods
        self._random_state = int(np.random.randint(1, 100_000)) if self.random_state is None else self.random_state
        self._verbose = min(self.verbose, 3)

        if self._estimator_type == "classifier":
            n_classes = len(np.unique(y))
            self._label_encoder = LabelEncoder()
            y = self._label_encoder.fit_transform(y)

        self._max_samples = calculate_max_value(n_values=n, desired_max=self.max_values) if self.max_samples else n

        max_cpus = cpu_count()
        value = 1 if self.n_jobs is None else self.n_jobs
        value = min(value, max_cpus)
        if value < 0:
            cpus = np.arange(1, max_cpus + 1)
            value = max_cpus if abs(value) > max_cpus else cpus[value]
        self._n_jobs = value

        # Set rest of parameters as private attributes
        for param, value in self.get_params().items():
            p_param = f"_{param}"
            if not hasattr(self, p_param):
                setattr(self, p_param, value)

        # Fitted attributes
        self.feature_importances_ = np.zeros(p, dtype=float)
        self.n_features_in_ = p
        if self._estimator_type == "classifier":
            base_estimator = ConditionalInferenceTreeClassifier
            self.classes_ = np.unique(y)
            self.n_classes_ = n_classes
        else:
            base_estimator = ConditionalInferenceTreeRegressor

        base_estimator = base_estimator(
            **{param: getattr(self, f"_{param}") for param in base_estimator._get_param_names()}
        )
        self.estimators_ = []
        for j in range(self._n_estimators):
            base_estimator = clone(base_estimator)
            base_estimator._random_state += j
            self.estimators_.append(base_estimator)

        # Train estimators
        bayesian_bootstrap = self._bootstrap_method == "bayesian"
        if self._estimator_type == "classifier":
            n = None
            if self._sampling_method in ["balanced", "stratify"]:
                idx = [np.where(y == j)[0] for j in range(self.n_classes_)]
                if self._sampling_method == "balanced":
                    n = np.bincount(y).min()
            else:
                n = len(y)
                idx = np.arange(n, dtype=int)

            self.estimators_ = Parallel(n_jobs=self._n_jobs, verbose=self._verbose, backend="loky")(
                delayed(_parallel_fit_classifier)(
                    tree=tree,
                    X=X,
                    y=y,
                    idx=idx,
                    n=n,
                    tree_idx=tree_idx,
                    n_estimators=self._n_estimators,
                    bootstrap_method=self._bootstrap_method,
                    bayesian_bootstrap=bayesian_bootstrap,
                    sampling_method=self._sampling_method,
                    verbose=self._verbose,
                    random_state=tree.random_state,
                )
                for tree_idx, tree in enumerate(self.estimators_, 1)
            )
        else:
            n = len(y)
            idx = np.arange(n, dtype=int)
            self.estimators_ = Parallel(n_jobs=self._n_jobs, verbose=self._verbose, backend="loky")(
                delayed(_parallel_fit_regressor)(
                    tree=tree,
                    X=X,
                    y=y,
                    idx=idx,
                    n=n,
                    tree_idx=tree_idx,
                    n_estimators=self._n_estimators,
                    bootstrap_method=self._bootstrap_method,
                    bayesian_bootstrap=bayesian_bootstrap,
                    verbose=self._verbose,
                    random_state=tree.random_state,
                )
                for tree_idx, tree in enumerate(self.estimators, 1)
            )

        # Aggregate feature importances
        for estimator in self.estimators_:
            self.feature_importances_ += estimator.feature_importances_
        self.feature_importances_ /= self.feature_importances_.sum()

        return self

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target."""
        pass


class ConditionalInferenceForestClassifier(BaseConditionalInferenceForest, ClassifierMixin):
    """Conditional inference forest classifier."""

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        selector: str = "mc",
        splitter: str = "gini",
        alpha_selector: float = 0.05,
        alpha_splitter: float = 0.05,
        adjust_alpha_selector: bool = True,
        adjust_alpha_splitter: bool = True,
        n_resamples_selector: Union[str, int] = "auto",
        n_resamples_splitter: Union[str, int] = "auto",
        early_stopping_selector: bool = True,
        early_stopping_splitter: bool = True,
        feature_muting: bool = True,
        feature_scanning: bool = True,
        threshold_scanning: bool = True,
        threshold_method: str = "exact",
        max_thresholds: Optional[Union[str, float, int]] = None,
        max_features: Optional[Union[str, float, int]] = "sqrt",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        bootstrap_method: Optional[str] = "bayesian",
        sampling_method: Optional[str] = "stratify",
        max_samples: Optional[Union[int, float]] = None,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: int = 1,
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
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap_method = bootstrap_method
        self.sampling_method = sampling_method
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self._validate_params({**self.get_params(), "estimator_type": self._estimator_type})

    @property
    def _parameter_validator(self) -> ModelMetaclass:
        """Model for hyperparameter validation.

        Returns
        -------
        ModelMetaclass
            Model to validate hyperparameters.
        """
        return ConditionalInferenceForestClassifierParameters

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Features.

        Returns
        -------
        np.ndarray
            Predicted class probabilities.
        """
        X = self._validate_data_predict(X)

        y_hat = np.zeros((X.shape[0], self.n_classes_))
        for estimator in self.estimators_:
            y_hat += estimator.predict_proba(X)  # type: ignore

        # Average probabilities
        y_hat /= self._n_estimators
        return y_hat

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target.

        Parameters
        ----------
        X : np.ndarray
            Features.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        X = self._validate_data_predict(X)

        y_hat = self.predict_proba(X)
        return np.argmax(y_hat, axis=1)


class ConditionalInferenceForestRegressor(BaseConditionalInferenceForest, RegressorMixin):
    """Conditional inference forest regressor."""

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        selector: str = "pc",
        splitter: str = "mse",
        alpha_selector: float = 0.05,
        alpha_splitter: float = 0.05,
        adjust_alpha_selector: bool = True,
        adjust_alpha_splitter: bool = True,
        n_resamples_selector: Union[str, int] = "auto",
        n_resamples_splitter: Union[str, int] = "auto",
        early_stopping_selector: bool = True,
        early_stopping_splitter: bool = True,
        feature_muting: bool = True,
        feature_scanning: bool = True,
        threshold_scanning: bool = True,
        threshold_method: str = "exact",
        max_thresholds: Optional[Union[str, float, int]] = None,
        max_features: Optional[Union[str, float, int]] = "sqrt",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        bootstrap_method: Optional[str] = "bayesian",
        max_samples: Optional[Union[int, float]] = None,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            selector=selector,
            splitter=splitter,
            alpha_selector=alpha_selector,
            alpha_splitter=alpha_splitter,
            adjust_alpha_selector=adjust_alpha_selector,
            adjust_alpha_splitter=adjust_alpha_splitter,
            n_resamples_selector=n_resamples_selector,
            n_resamples_splitter=n_resamples_splitter,
            threshold_method=threshold_method,
            max_thresholds=max_thresholds,
            early_stopping_selector=early_stopping_selector,
            early_stopping_splitter=early_stopping_splitter,
            feature_muting=feature_muting,
            feature_scanning=feature_scanning,
            threshold_scanning=threshold_scanning,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap_method=bootstrap_method,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target.

        Parameters
        ----------
        X : np.ndarray
            Features.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        X = self._validate_data_predict(X)

        y_hat = np.zeros(X.shape[0])
        for estimator in self.estimators_:
            y_hat += estimator.predict(X)  # type: ignore

        y_hat /= self._n_estimators_

        return y_hat
