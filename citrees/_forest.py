import warnings
from abc import ABCMeta, abstractmethod
from multiprocessing import cpu_count

import numpy as np
from joblib import Parallel, delayed
from pydantic import BaseModel, model_validator
from scipy.sparse import csr_matrix, hstack
from sklearn.base import ClassifierMixin, RegressorMixin, clone
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

from citrees._tree import (
    BaseConditionalInferenceTreeEstimator,
    BaseConditionalInferenceTreeParameters,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
)
from citrees._types import (
    BootstrapMethod,
    BootstrapMethodOption,
    EstimatorType,
    PositiveInt,
    ProbabilityFloat,
    SamplingMethod,
    SamplingMethodOption,
)
from citrees._utils import (
    calculate_max_value,
    classic_bootstrap_sample,
    oversample_bootstrap_sample,
    stratified_bootstrap_sample,
    undersample_bootstrap_sample,
)

# Defines how often to print status during parallel tree training
_PRINT_FACTOR = {
    1: 20,
    2: 10,
    3: 1,
}


def _parallel_fit_classifier(
    *,
    estimator: ConditionalInferenceTreeClassifier,
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    estimator_idx: int,
    n_estimators: int,
    bootstrap_method: str | None,
    sampling_method: str | None,
    verbose: int,
) -> ConditionalInferenceTreeClassifier:
    """Build classification trees in parallel.

    Note: This function can't go locally in a class, because joblib complains that it cannot pickle it when placed there

    Parameters
    ----------
    estimator : ConditionalInferenceTreeClassifier
        Instantiated estimator.

    X : np.ndarray
        Features.

    y : np.ndarray
        Target.

    max_samples : int
        Maximum number of samples in a bootstrap sample.

    estimator_idx : int
        Tree index in ensemble.

    n_estimators : int
        Number of parallel trees to grow.

    bootstrap_method : str or None
        Type of bootstrap to use.

    sampling_method : str or None
        Type of sampling to use during bootstrap.

    verbose : int
        Controls verbosity of fitting.

    Returns
    -------
    ConditionalInferenceTreeClassifier
        Fitted estimator.
    """
    if verbose and estimator_idx % _PRINT_FACTOR[verbose] == 0:
        print(f"Building tree {estimator_idx}/{n_estimators}")

    # Bootstrap sample if specified
    if bootstrap_method:
        kwargs = {
            "max_samples": max_samples,
            "bayesian_bootstrap": bootstrap_method == BootstrapMethod.BAYESIAN,
            "random_state": estimator.random_state,
        }
        if sampling_method == SamplingMethod.STRATIFIED:
            boot_idx = stratified_bootstrap_sample(y=y, **kwargs)  # type: ignore[arg-type]
        elif sampling_method == SamplingMethod.UNDERSAMPLE:
            boot_idx = undersample_bootstrap_sample(y=y, **kwargs)  # type: ignore[arg-type]
        elif sampling_method == SamplingMethod.OVERSAMPLE:
            boot_idx = oversample_bootstrap_sample(y=y, **kwargs)  # type: ignore[arg-type]
        else:
            boot_idx = classic_bootstrap_sample(y=y, **kwargs)  # type: ignore[arg-type]
        estimator.fit(X[boot_idx], y[boot_idx])
    else:
        estimator.fit(X, y)

    return estimator


def _parallel_fit_regressor(
    *,
    estimator: ConditionalInferenceTreeRegressor,
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    estimator_idx: int,
    n_estimators: int,
    bootstrap_method: str | None,
    verbose: int,
) -> ConditionalInferenceTreeRegressor:
    """Build regression trees in parallel.

    Note: This function can't go locally in a class, because joblib complains that it cannot pickle it when placed there

    Parameters
    ----------
    estimator : ConditionalInferenceTreeRegressor
        Instantiated estimator.

    X : np.ndarray
        Features.

    y : np.ndarray
        Target.

    max_samples : int
        Maximum number of samples in a bootstrap sample.

    estimator_idx : int
        Tree index in ensemble.

    n_estimators : int
        Number of parallel trees to grow.

    bootstrap_method : str or None
        Type of bootstrap to use.

    verbose : int
        Controls verbosity of fitting.

    Returns
    -------
    ConditionalInferenceTreeRegressor
        Fitted estimator.
    """
    if verbose and estimator_idx % _PRINT_FACTOR[verbose] == 0:
        print(f"Building tree {estimator_idx}/{n_estimators}")

    # Bootstrap sample if specified
    if bootstrap_method:
        boot_idx = classic_bootstrap_sample(
            y=y,
            max_samples=max_samples,
            bayesian_bootstrap=bootstrap_method == BootstrapMethod.BAYESIAN,
            random_state=estimator.random_state,  # type: ignore[arg-type]
        )
        estimator.fit(X[boot_idx], y[boot_idx])
    else:
        estimator.fit(X, y)

    return estimator


class BaseConditionalInferenceForestParameters(BaseConditionalInferenceTreeParameters):
    """Model for BaseConditionalInferenceForest parameters."""

    n_estimators: PositiveInt
    bootstrap_method: BootstrapMethodOption
    max_samples: PositiveInt | ProbabilityFloat | None
    n_jobs: int | None
    oob_score: bool

    @model_validator(mode="after")
    def validate_bootstrap_constraints(self) -> "BaseConditionalInferenceForestParameters":
        if self.bootstrap_method is None:
            if self.max_samples is not None:
                raise ValueError("max_samples must be None when bootstrap_method=None")
            if self.oob_score:
                raise ValueError("oob_score requires bootstrap_method to be set (bootstrap enabled)")
        return self


class ConditionalInferenceForestClassifierParameters(BaseConditionalInferenceForestParameters):
    """Model for ConditionalInferenceForestClassifier parameters."""

    sampling_method: SamplingMethodOption

    @model_validator(mode="after")
    def validate_sampling_method_constraints(
        self,
    ) -> "ConditionalInferenceForestClassifierParameters":
        if self.bootstrap_method is None and self.sampling_method is not None:
            raise ValueError("sampling_method must be None when bootstrap_method=None")
        return self


class BaseConditionalInferenceForest(BaseConditionalInferenceTreeEstimator, metaclass=ABCMeta):
    """Base class for conditional inference forests.

    Warning: This class should not be used directly. Use derived classes instead.
    """

    @abstractmethod
    def __init__(
        self,
        *,
        n_estimators: int,
        selector: str | list[str],
        splitter: str,
        alpha_selector: float,
        alpha_splitter: float,
        adjust_alpha_selector: bool,
        adjust_alpha_splitter: bool,
        n_resamples_selector: str | int | None,
        n_resamples_splitter: str | int | None,
        early_stopping_selector: str | None,
        early_stopping_splitter: str | None,
        early_stopping_confidence_selector: float,
        early_stopping_confidence_splitter: float,
        feature_muting: bool,
        feature_scanning: bool,
        threshold_scanning: bool,
        threshold_method: str,
        max_thresholds: str | float | int | None,
        max_depth: int | None,
        max_features: str | float | int | None,
        min_samples_split: int,
        min_samples_leaf: int,
        min_impurity_decrease: float,
        honesty: bool,
        honesty_fraction: float,
        bootstrap_method: str | None,
        max_samples: int | float | None,
        n_jobs: int | None,
        oob_score: bool,
        random_state: int | None,
        verbose: int,
        check_for_unused_parameters: bool,
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
        self.early_stopping_confidence_selector = early_stopping_confidence_selector
        self.early_stopping_confidence_splitter = early_stopping_confidence_splitter
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
        self.honesty = honesty
        self.honesty_fraction = honesty_fraction
        self.bootstrap_method = bootstrap_method
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.oob_score = oob_score
        self.random_state = random_state
        self.verbose = verbose
        self.check_for_unused_parameters = check_for_unused_parameters

        self._validate_parameters({**self.get_params(), "estimator_type": self._estimator_type})
        if self.check_for_unused_parameters:
            self._validate_parameter_combinations()

    def _validate_parameter_combinations(self) -> None:
        """Runtime check to determine if any hyperparameters are unused due to constraints.

        Bootstrap constraints:
        1. If bootstrap_method is None =>
            - sampling_method = None
            - max_samples = None
        """
        super()._validate_parameter_combinations()

        params = self.get_params()

        flags = []
        if params["bootstrap_method"] is None:
            # Use .get() to handle regressor which doesn't have sampling_method
            flags = [key for key in ["sampling_method", "max_samples"] if params.get(key)]
        if flags:
            warnings.warn(
                "Unused hyperparameter(s) detected: When bootstrap_method=None, hyperparameter(s) "
                f"({', '.join(flags)}) should be None",
                stacklevel=2,
            )

    @property
    def _parameter_model(self) -> type[BaseModel]:
        """Model for hyperparameter validation.

        Returns
        -------
        type[BaseModel]
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
        X, y = self._validate_data_fit(X=X, y=y, estimator_type=self._estimator_type)
        n, p = X.shape

        # Private attributes for all parameters - for consistency to reference across other classes and methods
        self._random_state = (
            int(np.random.randint(1, 1_000_000)) if self.random_state is None else self.random_state
        )
        self._verbose = min(self.verbose, 3)

        if self._estimator_type == EstimatorType.CLASSIFIER:
            n_classes = len(np.unique(y))
            self._label_encoder = LabelEncoder()
            y = self._label_encoder.fit_transform(y)
            self._class_to_index = {
                cls: idx for idx, cls in enumerate(self._label_encoder.classes_)
            }

        self._max_samples = (
            calculate_max_value(n_values=n, desired_max=self.max_samples) if self.max_samples else n
        )

        max_cpus = cpu_count()
        value = 1 if self.n_jobs is None else self.n_jobs
        if value == 0:
            raise ValueError(
                "n_jobs=0 is invalid. Use n_jobs=1 for single-threaded or n_jobs=-1 for all cores."
            )
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
        if self._estimator_type == EstimatorType.CLASSIFIER:
            base_estimator = ConditionalInferenceTreeClassifier
            self.classes_ = self._label_encoder.classes_
            self.n_classes_ = n_classes
        else:
            base_estimator = ConditionalInferenceTreeRegressor

        base_estimator = base_estimator(
            **{param: getattr(self, f"_{param}") for param in base_estimator._get_param_names()}
        )
        # Set default parameters for base estimator
        base_estimator.verbose = 0
        base_estimator.check_for_unused_parameters = False

        self.estimators_ = []
        for j in range(self._n_estimators):
            base_estimator = clone(base_estimator)
            # Update random state to unique value for each tree estimator
            base_estimator.random_state = self._random_state + j
            self.estimators_.append(base_estimator)

        # Train estimators
        if self._estimator_type == EstimatorType.CLASSIFIER:
            self.estimators_ = Parallel(n_jobs=self._n_jobs, verbose=self._verbose, backend="loky")(
                delayed(_parallel_fit_classifier)(
                    estimator=estimator,
                    X=X,
                    y=y,
                    max_samples=self._max_samples,
                    estimator_idx=estimator_idx,
                    n_estimators=self._n_estimators,
                    bootstrap_method=self._bootstrap_method,
                    sampling_method=self._sampling_method,
                    verbose=self._verbose,
                )
                for estimator_idx, estimator in enumerate(self.estimators_, 1)
            )
        else:
            self.estimators_ = Parallel(n_jobs=self._n_jobs, verbose=self._verbose, backend="loky")(
                delayed(_parallel_fit_regressor)(
                    estimator=estimator,
                    X=X,
                    y=y,
                    max_samples=self._max_samples,
                    estimator_idx=estimator_idx,
                    n_estimators=self._n_estimators,
                    bootstrap_method=self._bootstrap_method,
                    verbose=self._verbose,
                )
                for estimator_idx, estimator in enumerate(self.estimators_, 1)
            )

        # Aggregate feature importances
        for estimator in self.estimators_:
            self.feature_importances_ += estimator.feature_importances_
        fi_sum = self.feature_importances_.sum()
        if fi_sum:
            self.feature_importances_ /= fi_sum

        if self.oob_score:
            if self._bootstrap_method is None:
                raise ValueError(
                    "oob_score requires bootstrap_method to be set (bootstrap enabled)."
                )
            self._compute_oob_score(X, y)

        return self

    def _align_proba(self, proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """Align probability columns to the forest's global class order."""
        if proba.shape[1] == self.n_classes_:
            return proba

        classes = np.asarray(classes)
        aligned = np.zeros((proba.shape[0], self.n_classes_), dtype=float)
        if np.issubdtype(classes.dtype, np.integer):
            cols = classes.astype(int)
        else:
            try:
                cols = np.array([self._class_to_index[cls] for cls in classes], dtype=int)
            except KeyError:
                cols = classes.astype(int)
        aligned[:, cols] = proba
        return aligned

    def _compute_oob_score(self, X: np.ndarray, y: np.ndarray) -> None:
        """Compute OOB predictions and score (sklearn-compatible)."""
        n = X.shape[0]
        n_oob = np.zeros(n, dtype=int)

        if self._estimator_type == EstimatorType.CLASSIFIER:
            oob_decision_function = np.zeros((n, self.n_classes_), dtype=float)
        else:
            oob_prediction = np.zeros(n, dtype=float)

        for j, estimator in enumerate(self.estimators_):
            if self._estimator_type == EstimatorType.CLASSIFIER:
                kwargs = {
                    "y": y,
                    "max_samples": self._max_samples,
                    "bayesian_bootstrap": self._bootstrap_method == BootstrapMethod.BAYESIAN,
                    "random_state": self._random_state + j,
                }
                if self._sampling_method == SamplingMethod.STRATIFIED:
                    boot_idx = stratified_bootstrap_sample(**kwargs)
                elif self._sampling_method == SamplingMethod.UNDERSAMPLE:
                    boot_idx = undersample_bootstrap_sample(**kwargs)
                elif self._sampling_method == SamplingMethod.OVERSAMPLE:
                    boot_idx = oversample_bootstrap_sample(**kwargs)
                else:
                    boot_idx = classic_bootstrap_sample(**kwargs)
            else:
                boot_idx = classic_bootstrap_sample(
                    y=y,
                    max_samples=self._max_samples,
                    bayesian_bootstrap=self._bootstrap_method == BootstrapMethod.BAYESIAN,
                    random_state=self._random_state + j,
                )

            oob_idx = np.setdiff1d(np.arange(n), boot_idx)
            if oob_idx.size == 0:
                continue

            if self._estimator_type == EstimatorType.CLASSIFIER:
                proba = estimator.predict_proba(X[oob_idx])  # type: ignore
                proba = self._align_proba(proba, estimator.classes_)  # type: ignore
                oob_decision_function[oob_idx] += proba
            else:
                preds = estimator.predict(X[oob_idx])  # type: ignore
                oob_prediction[oob_idx] += preds

            n_oob[oob_idx] += 1

        if np.any(n_oob == 0):
            warnings.warn(
                "Some inputs do not have OOB scores. This probably means too few trees were used "
                "to compute any reliable OOB estimates.",
                stacklevel=2,
            )

        mask = n_oob > 0
        if self._estimator_type == EstimatorType.CLASSIFIER:
            if np.any(mask):
                oob_decision_function[mask] /= n_oob[mask, None]
            self.oob_decision_function_ = oob_decision_function
            # Only score samples with OOB predictions (n_oob > 0)
            y_pred = np.argmax(oob_decision_function[mask], axis=1)
            self.oob_score_ = float((y_pred == y[mask]).mean())
        else:
            if np.any(mask):
                oob_prediction[mask] /= n_oob[mask]
            self.oob_prediction_ = oob_prediction
            # Only score samples with OOB predictions (n_oob > 0)
            self.oob_score_ = float(r2_score(y[mask], oob_prediction[mask]))

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return leaf indices for each sample and estimator (sklearn-compatible)."""
        X = self._validate_data_predict(X)

        leaf_ids = np.zeros((X.shape[0], self._n_estimators), dtype=int)
        for idx, estimator in enumerate(self.estimators_):
            leaf_ids[:, idx] = estimator.apply(X)  # type: ignore

        return leaf_ids

    def decision_path(self, X: np.ndarray) -> tuple[csr_matrix, np.ndarray]:
        """Return decision path for each sample across all estimators.

        Returns
        -------
        indicator : csr_matrix
            Sparse indicator matrix with concatenated node paths.
        n_nodes_ptr : np.ndarray
            Cumulative sum of node counts for each estimator.
        """
        X = self._validate_data_predict(X)

        indicators = []
        n_nodes_ptr = np.zeros(self._n_estimators + 1, dtype=int)
        n_nodes = 0
        for idx, estimator in enumerate(self.estimators_):
            indicator = estimator.decision_path(X)  # type: ignore
            indicators.append(indicator)
            n_nodes += indicator.shape[1]
            n_nodes_ptr[idx + 1] = n_nodes

        return hstack(indicators).tocsr(), n_nodes_ptr

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target."""
        pass


class ConditionalInferenceForestClassifier(ClassifierMixin, BaseConditionalInferenceForest):
    """Conditional inference forest classifier."""

    _estimator_type = "classifier"

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        selector: str | list[str] = "mc",
        splitter: str = "gini",
        alpha_selector: float = 0.05,
        alpha_splitter: float = 0.05,
        adjust_alpha_selector: bool = True,
        adjust_alpha_splitter: bool = True,
        n_resamples_selector: str | int | None = "auto",
        n_resamples_splitter: str | int | None = "auto",
        early_stopping_selector: str | None = "adaptive",
        early_stopping_splitter: str | None = "adaptive",
        early_stopping_confidence_selector: float = 0.95,
        early_stopping_confidence_splitter: float = 0.95,
        feature_muting: bool = True,
        feature_scanning: bool = True,
        max_features: str | float | int | None = "sqrt",
        threshold_method: str = "exact",
        threshold_scanning: bool = True,
        max_thresholds: str | float | int | None = None,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        honesty: bool = False,
        honesty_fraction: float = 0.5,
        bootstrap_method: str | None = "bayesian",
        sampling_method: str | None = "stratified",
        max_samples: int | float | None = None,
        oob_score: bool = False,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: int = 1,
        check_for_unused_parameters: bool = False,
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
        self.early_stopping_confidence_selector = early_stopping_confidence_selector
        self.early_stopping_confidence_splitter = early_stopping_confidence_splitter
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
        self.honesty = honesty
        self.honesty_fraction = honesty_fraction
        self.bootstrap_method = bootstrap_method
        self.sampling_method = sampling_method
        self.max_samples = max_samples
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.check_for_unused_parameters = check_for_unused_parameters

        self._validate_parameters({**self.get_params(), "estimator_type": self._estimator_type})
        if self.check_for_unused_parameters:
            self._validate_parameter_combinations()

    @property
    def _parameter_model(self) -> type[BaseModel]:
        """Model for hyperparameter validation.

        Returns
        -------
        type[BaseModel]
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
            proba = estimator.predict_proba(X)  # type: ignore
            y_hat += self._align_proba(proba, estimator.classes_)  # type: ignore

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
        labels = np.argmax(y_hat, axis=1)
        return self._label_encoder.inverse_transform(labels)


class ConditionalInferenceForestRegressor(RegressorMixin, BaseConditionalInferenceForest):
    """Conditional inference forest regressor.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of estimators.

    selector : {"pc", "dc", "rdc"} or list, default="pc"
        Method for feature selection.

    splitter : {"mse", "mae"}, default="mse"
        Method for split selection.

    alpha_selector : float, default=0.05
        Alpha for feature selection.

    alpha_splitter : float, default=0.05
        Alpha for split selection.

    adjust_alpha_selector : bool, default=True
        Whether to perform a Bonferroni correction during feature selection.

    adjust_alpha_splitter : bool, default=True
        Whether to perform a Bonferroni correction during split selection.

    n_resamples_selector : {"auto", "minimum", "maximum"} or int, default="auto"
        Number of resamples to use in permutation test for feature selection.

    n_resamples_splitter : {"auto", "minimum", "maximum"} or int, default="auto"
        Number of resamples to use in permutation test for split selection.

    early_stopping_selector : {"adaptive", "simple"} or None, default="adaptive"
        Early stopping method for feature selection permutation tests.

    early_stopping_splitter : {"adaptive", "simple"} or None, default="adaptive"
        Early stopping method for split selection permutation tests.

    early_stopping_confidence_selector : float, default=0.95
        Confidence threshold for adaptive stopping in feature selection.

    early_stopping_confidence_splitter : float, default=0.95
        Confidence threshold for adaptive stopping in split selection.

    feature_muting : bool, default=True
        Whether to perform feature muting.

    feature_scanning : bool, default=True
        Whether to perform feature scanning.

    max_features : {"sqrt", "log2"}, int, float, or None, default="sqrt"
        Maximum number of features to use for feature selection.

    threshold_method : {"exact", "random", "histogram", "percentile"}, default="exact"
        Method to calculate thresholds on a feature used during split selection.

    threshold_scanning : bool, default=True
        Whether to perform threshold scanning.

    max_thresholds : {"sqrt", "log2"}, int, float, or None, default=None
        Maximum number of thresholds to use for split selection.

    max_depth : int, default=None
        Maximum depth to grow tree.

    min_samples_split : int, default=2
        Minimum samples required for a valid binary split.

    min_samples_leaf : int, default=1
        Minimum number of samples in a leaf node.

    min_impurity_decrease : float, default=0.0
        Minimum impurity decrease required for a valid binary split.

    bootstrap_method : {"bayesian", "classic"} or None, default="bayesian"
        Type of bootstrap to use. Set to None to disable bootstrapping.

    max_samples : int or float, default=None
        Bootstrap sample cap (count or fraction). When set, each tree is fit on at most
        max_samples rows after bootstrap resampling.

    oob_score : bool, default=False
        Whether to compute out-of-bag score (requires bootstrap).

    n_jobs : int, default=None
        Number of jobs to run in parallel.

    random_state : int, default=None
        Random seed.

    verbose : int, default=1
        Controls verbosity when fitting and predicting.

    check_for_unused_parameters : bool, default=False
        Check for unused hyperparameters. Useful for debugging.

    Attributes
    ----------
    feature_importances_ : np.ndarray
        Feature importances for each feature.

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : List[str]
        List of feature names seen during fit.

    estimators_ : List[ConditionalInferenceTreeRegressor]
        List of fitted estimators.

    oob_score_ : float
        Out-of-bag R² score (if enabled).

    oob_prediction_ : np.ndarray
        Out-of-bag predictions (if enabled).
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        selector: str | list[str] = "pc",
        splitter: str = "mse",
        alpha_selector: float = 0.05,
        alpha_splitter: float = 0.05,
        adjust_alpha_selector: bool = True,
        adjust_alpha_splitter: bool = True,
        n_resamples_selector: str | int | None = "auto",
        n_resamples_splitter: str | int | None = "auto",
        early_stopping_selector: str | None = "adaptive",
        early_stopping_splitter: str | None = "adaptive",
        early_stopping_confidence_selector: float = 0.95,
        early_stopping_confidence_splitter: float = 0.95,
        feature_muting: bool = True,
        feature_scanning: bool = True,
        max_features: str | float | int | None = "sqrt",
        threshold_method: str = "exact",
        threshold_scanning: bool = True,
        max_thresholds: str | float | int | None = None,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        honesty: bool = False,
        honesty_fraction: float = 0.5,
        bootstrap_method: str | None = "bayesian",
        max_samples: int | float | None = None,
        oob_score: bool = False,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: int = 1,
        check_for_unused_parameters: bool = False,
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
            early_stopping_selector=early_stopping_selector,
            early_stopping_splitter=early_stopping_splitter,
            early_stopping_confidence_selector=early_stopping_confidence_selector,
            early_stopping_confidence_splitter=early_stopping_confidence_splitter,
            feature_muting=feature_muting,
            feature_scanning=feature_scanning,
            max_features=max_features,
            threshold_method=threshold_method,
            threshold_scanning=threshold_scanning,
            max_thresholds=max_thresholds,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            honesty=honesty,
            honesty_fraction=honesty_fraction,
            bootstrap_method=bootstrap_method,
            max_samples=max_samples,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            check_for_unused_parameters=check_for_unused_parameters,
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
            Predicted target values.
        """
        X = self._validate_data_predict(X)

        y_hat = np.zeros(X.shape[0])
        for estimator in self.estimators_:
            y_hat += estimator.predict(X)  # type: ignore

        y_hat /= self._n_estimators

        return y_hat
