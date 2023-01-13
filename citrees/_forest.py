from abc import ABCMeta, abstractmethod
from multiprocessing import cpu_count
from typing import Optional, Literal, Union

from joblib import delayed, Parallel
import numpy as np
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
    n_classes: int,
    tree_idx: int,
    n_estimators: int,
    bootstrap_method: Optional[str],
    sampling_method: Optional[str],
    verbose: int,
    random_state: int,
) -> ConditionalInferenceTreeClassifier:
    """Utility function for building trees in parallel.

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
        kwargs = {"bayesian_bootstrap": bootstrap_method == "bayesian", "random_state": tree.random_state}
        if sampling_method in ["balanced", "stratify"]:
            idx_classes = [np.where(y == j)[0] for j in range(n_classes)]
            if sampling_method == "balanced":
                n = np.bincount(y).min()
                idx = balanced_bootstrap_sample(idx_classes=idx_classes, n=n, **kwargs)
            else:
                idx = stratify_bootstrap_sample(idx_classes=idx_classes, **kwargs)
            idx = np.concatenate(idx)
        else:
            idx = classic_bootstrap_sample(n=len(y), **kwargs)
        tree.fit(X[idx], y[idx])
    else:
        tree.fit(X, y)

    return tree


# def _parallel_fit_regressor(tree, X, y, n, tree_idx, n_estimators, bootstrap,
#                             bayes, verbose, random_state):
#     """Utility function for building trees in parallel
#     Note: This function can't go locally in a class, because joblib complains
#           that it cannot pickle it when placed there
#     Parameters
#     ----------
#     tree : CITreeRegressor
#         Instantiated conditional inference tree
#     X : 2d array-like
#         Array of features
#     y : 1d array-like
#         Array of labels
#     n : int
#         Number of samples
#     tree_idx : int
#         Index of tree in forest
#     n_estimators : int
#         Number of total estimators
#     bootstrap : bool
#         Whether to perform bootstrap sampling
#     bayes : bool
#         If True, performs Bayesian bootstrap sampling
#     verbose : bool or int
#         Controls verbosity of training process
#     random_state : int
#         Sets seed for random number generator
#     Returns
#     -------
#     tree : CITreeRegressor
#         Fitted conditional inference tree
#     """
#     # Print status if conditions met
#     if verbose and n_estimators >= 10:
#         denom = n_estimators if verbose > 1 else 10
#         if (tree_idx+1) % int(n_estimators/denom) == 0:
#             logger("tree", "Building tree %d/%d" % (tree_idx+1, n_estimators))

#     # Bootstrap sample if specified
#     if bootstrap:
#         random_state = random_state*(tree_idx+1)
#         idx          = normal_sampled_idx(random_state, n, bayes)

#         # Train
#         tree.fit(X[idx], y[idx])
#     else:
#         tree.fit(X, y)

#     return tree


def _accumulate_prediction(predict, X, out, lock) -> None:
    """Utility function to aggregate predictions in parallel.

    Parameters
    ----------
    predict : function handle
        Alias to prediction method of class

    X : 2d array-like
        Array of features

    out : 1d or 2d array-like
        Array of labels

    lock : threading lock
        A lock that controls worker access to data structures for aggregating predictions.
    """
    prediction = predict(X)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]


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

        if self.max_samples:
            self._max_samples = calculate_max_value(n_values=n, desired_max=self.max_values)
        else:
            self._max_samples = n

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
            base_estimator.random_state += j
            self.estimators_.append(base_estimator)

        # Train estimators
        if self._estimator_type == "classifier":
            self.estimators_ = Parallel(n_jobs=self._n_jobs, verbose=self._verbose, backend="loky")(
                delayed(_parallel_fit_classifier)(
                    tree=tree,
                    X=X,
                    y=y,
                    n_classes=self.n_classes_,
                    tree_idx=tree_idx,
                    n_estimators=self._n_estimators,
                    bootstrap_method=self._bootstrap_method,
                    sampling_method=self._sampling_method,
                    verbose=self._verbose,
                    random_state=tree.random_state,
                )
                for tree_idx, tree in enumerate(self.estimators_)
            )
        else:
            pass
            # _parallel_fit_regressor

        # Normalize feature importances
        # self.feature_importances_ /= self.feature_importances_.sum()

        return self

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """ADD HERE."""
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
        max_depth: Optional[int] = None,
        max_features: Optional[Union[str, float, int]] = "sqrt",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        bootstrap_method: Optional[str] = "bayesian",
        sampling_method: Optional[str] = "stratify",
        max_samples: Optional[Union[int, float]] = None,
        n_jobs: int = None,
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
        self.max_depth = max_depth
        self.max_features = max_features
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
        """ADD HERE."""
        return ConditionalInferenceForestClassifierParameters

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
