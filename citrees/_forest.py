from abc import ABCMeta, abstractmethod
from multiprocessing import cpu_count
from typing import Optional, Literal, Union

import numpy as np
from pydantic import PositiveInt
from pydantic.main import ModelMetaclass
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder

from ._tree import (
    BaseConditionalInferenceTreeEstimator,
    BaseConditionalInferenceTreeParameters,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
    ProbabilityFloat,
)
from ._utils import calculate_max_value



# def stratify_sampled_idx(random_state, y, bayes):
#     """Indices for stratified bootstrap sampling in classification.
#     Parameters
#     ----------
#     random_state : int
#         Sets seed for random number generator
#     y : 1d array-like
#         Array of labels
#     bayes : bool
#         If True, performs Bayesian bootstrap sampling
#     Returns
#     -------
#     idx : list
#         Stratified sampled indices for each class
#     """
#     np.random.seed(random_state)
#     idx = []
#     for label in np.unique(y):

#         # Grab indices for class
#         tmp = np.where(y==label)[0]

#         # Bayesian bootstrapping if specified
#         p = bayes_boot_probs(n=len(tmp)) if bayes else None

#         idx.append(np.random.choice(tmp, size=len(tmp), replace=True, p=p))

#     return idx


# def stratify_unsampled_idx(random_state, y, bayes):
#     """Unsampled indices for stratified bootstrap sampling in classification
#     Parameters
#     ----------
#     random_state : int
#         Sets seed for random number generator
#     y : 1d array-like
#         Array of labels
#     bayes : bool
#         If True, performs Bayesian bootstrap sampling
#     Returns
#     -------
#     idx : list
#         Stratified unsampled indices for each class
#     """
#     np.random.seed(random_state)
#     sampled = stratify_sampled_idx(random_state, y, bayes)
#     idx     = []
#     for i, label in enumerate(np.unique(y)):
#         idx.append(np.setdiff1d(np.where(y==label)[0], sampled[i]))
#     return idx


# def balanced_sampled_idx(random_state, y, bayes, min_class_p):
#     """Indices for balanced bootstrap sampling in classification
#     Parameters
#     ----------
#     random_state : int
#         Sets seed for random number generator
#     y : 1d array-like
#         Array of labels
#     bayes : bool
#         If True, performs Bayesian bootstrap sampling
#     min_class_p : float
#         Minimum proportion of class labels
#     Returns
#     -------
#     idx : list
#         Balanced sampled indices for each class
#     """
#     np.random.seed(random_state)
#     idx, n = [], int(np.floor(min_class_p*len(y)))
#     for i, label in enumerate(np.unique(y)):

#         # Grab indices for class
#         tmp = np.where(y==label)[0]

#         # Bayesian bootstrapping if specified
#         p = bayes_boot_probs(n=len(tmp)) if bayes else None

#         idx.append(np.random.choice(tmp, size=n, replace=True, p=p))

#     return idx


# def balanced_unsampled_idx(random_state, y, bayes, min_class_p):
#     """Unsampled indices for balanced bootstrap sampling in classification
#     Parameters
#     ----------
#     random_state : int
#         Sets seed for random number generator
#     y : 1d array-like
#         Array of labels
#     bayes : bool
#         If True, performs Bayesian bootstrap sampling
#     min_class_p : float
#         Minimum proportion of class labels
#     Returns
#     -------
#     idx : list
#         Balanced unsampled indices for each class
#     """
#     np.random.seed(random_state)
#     sampled = balanced_sampled_idx(random_state, y, bayes, min_class_p)
#     idx     = []
#     for i, label in enumerate(np.unique(y)):
#         idx.append(np.setdiff1d(np.where(y==label)[0], sampled[i]))
#     return idx


# def normal_sampled_idx(random_state, n, bayes):
#     """Indices for bootstrap sampling
#     Parameters
#     ----------
#     random_state : int
#         Sets seed for random number generator
#     n : int
#         Sample size
#     bayes : bool
#         If True, performs Bayesian bootstrap sampling
#     Returns
#     -------
#     idx : list
#         Sampled indices
#     """
#     np.random.seed(random_state)

#     # Bayesian bootstrapping if specified
#     p = bayes_boot_probs(n=n) if bayes else None

#     return np.random.choice(np.arange(n, dtype=int), size=n, replace=True, p=p)


# def normal_unsampled_idx(random_state, n, bayes):
#     """Unsampled indices for bootstrap sampling
#     Parameters
#     ----------
#     random_state : int
#         Sets seed for random number generator
#     y : 1d array-like
#         Array of labels
#     n : int
#         Sample size
#     bayes : bool
#         If True, performs Bayesian bootstrap sampling
#     Returns
#     -------
#     idx : list
#         Unsampled indices
#     """
#     sampled = normal_sampled_idx(random_state, n, bayes)
#     counts  = np.bincount(sampled, minlength=n)
#     return np.arange(n, dtype=int)[counts==0]


# def _parallel_fit_classifier(tree, X, y, n, tree_idx, n_estimators, bootstrap,
#                              bayes, verbose, random_state, class_weight=None,
#                              min_dist_p=None):
#     """Utility function for building trees in parallel
#     Note: This function can't go locally in a class, because joblib complains
#           that it cannot pickle it when placed there
#     Parameters
#     ----------
#     tree : CITreeClassifier
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
#     class_weight : str
#         Type of sampling during bootstrap, None for regular bootstrapping,
#         'balanced' for balanced bootstrap sampling, and 'stratify' for
#         stratified bootstrap sampling
#     min_class_p : float
#         Minimum proportion of class labels
#     Returns
#     -------
#     tree : CITreeClassifier
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
#         if class_weight == 'balanced':
#             idx = np.concatenate(
#                 balanced_sampled_idx(random_state, y, bayes, min_dist_p)
#                 )
#         elif class_weight == 'stratify':
#             idx = np.concatenate(
#                 stratify_sampled_idx(random_state, y, bayes)
#                 )
#         else:
#             idx = normal_sampled_idx(random_state, n, bayes)

#         # Note: We need to pass the classes in the case of the bootstrap
#         # because not all classes may be sampled and when it comes to prediction,
#         # the tree models learns a different number of classes across different
#         # bootstrap samples
#         tree.fit(X[idx], y[idx], np.unique(y))
#     else:
#         tree.fit(X, y)
    
#     return tree


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


# def _accumulate_prediction(predict, X, out, lock):
#     """Utility function to aggregate predictions in parallel
#     Parameters
#     ----------
#     predict : function handle
#         Alias to prediction method of class
#     X : 2d array-like
#         Array of features
#     out : 1d or 2d array-like
#         Array of labels
#     lock : threading lock
#         A lock that controls worker access to data structures for aggregating
#         predictions
#     Returns
#     -------
#     None
#     """
#     prediction = predict(X)
#     with lock:
#         if len(out) == 1:
#             out[0] += prediction
#         else:
#             for i in range(len(out)): out[i] += prediction[i]





class BaseConditionalInferenceForestParameters(BaseConditionalInferenceTreeParameters):
    """Model for BaseConditionalInferenceForest parameters."""

    n_estimators: PositiveInt
    bootstrap_method: Optional[Literal["bayesian", "classic"]]
    max_samples: Optional[Union[PositiveInt, ProbabilityFloat]]
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
        value = min(self.n_jobs, max_cpus)
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
        if self._estimator_type == "classifier":
            self.base_estimator_ = ConditionalInferenceTreeClassifier
            self.classes_ = np.unique(y)
            self.n_classes_ = n_classes
        else:
            self.base_estimator_ = ConditionalInferenceTreeRegressor

        self.base_estimator_params_ = {
            param: getattr(self, f"_{param}") for param in self.base_estimator_.get_param_names()
        }

        self.feature_importances_ = np.zeros(p, dtype=float)
        self.n_features_in_ = p
        
        # Create estimators
        self.estimators_ = []
        for j in range(self._n_estimators):
            params = self.base_estimator_params_
            params["random_state"] = self._random_state + j
            self.estimators_.append(
                self.base_estimator_(**params)
            )
            
        # Train estimators
        import pdb; pdb.set_trace()

        # Normalize feature importances
        # self.feature_importances_ /= self.feature_importances_.sum()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ADD HERE.

        Parameters
        ----------

        Returns
        -------
        """
        pass
