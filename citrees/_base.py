from abc import ABCMeta, abstractmethod
from math import ceil
from multiprocessing import cpu_count
from typing import Optional, Tuple, TypedDict, Union
import warnings

import numpy as np
from numba import njit
from scipy.stats import norm
from sklearn.base import BaseEstimator

from ._selector import ClassifierSelectors, ClassifierSelectorTests, RegressorSelectors, RegressorSelectorTests
from ._splitter import ClassifierSplitters, ClassifierSplitterTests, RegressorSplitters, RegressorSplitterTests, ThresholdMethods
from ._utils import random_sample

_MIN_ALPHA = 0.001
_MAX_resamples = 5 * ceil(1 / _MIN_ALPHA)
_PVAL_PRECISION = 0.05

# TODO: Cast data types for classifiers and regressors


@njit(cache=True, fastmath=True, nogil=True)
def _calculate_max_value(*, n_values: int, desired_max: Union[str, float, int]) -> int:
    """ADD HERE.

    Parameters
    ----------

    Returns
    -------
    """
    total = None
    if desired_max == "sqrt":
        total = ceil(np.sqrt(n_values))
    elif desired_max == "log":
        total = ceil(np.log(n_values))
    elif type(desired_max) is float:
        total = ceil(n_values * desired_max)
    elif type(desired_max) is int:
        total = min(desired_max, n_values)
    return total


@njit(fastmath=True, nogil=True)
def _node_split(
    X: np.ndarray, y: np.ndarray, feature: int, threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split node based on feature and threshold.

    Parameters
    ----------
    X : np.ndarray
        Features for node.

    y : np.ndarray:
        Labels for node.

    feature : int
        Index of feature to use for splitting node.

    threshold : float
        Threshold value to use for creating binary split on node.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Features and labels in left and right children nodes with order (X_left, y_left, X_right, y_right).
    """
    idx = X[:, feature] <= threshold
    return X[idx], y[idx], X[~idx], y[~idx]


class Node(TypedDict, total=False):
    """Node in decision tree.

    Parameters
    ----------
    feature : int, optional
        Column index of feature.

    pval_feature : float, optional
        Probability value from feature selection.

    threshold : float, optional
        Best split point found in feature.

    pval_threshold : float, optional
        Probability value from split selection.

    impurity : float, optional
        Impurity measuring quality of split.

    value : Union[np.ndarray, float], optional
        Estimate of class probabilities for classification trees and estimate of central tendency for regression trees.

    left_child : Node, optional
        Left child node where feature value met the threshold.

    right_child : Node, optional
        Left child node where feature value did not meet the threshold.

    n_samples : int, optional
        Number of samples at the node.
    """

    feature: Optional[int]
    pval_feature: Optional[float]
    threshold: Optional[float]
    pval_threshold: Optional[float]
    impurity: Optional[float]
    value: Optional[Union[np.ndarray, float]]
    left_child: Optional["Node"]
    right_child: Optional["Node"]
    n_samples: Optional[int]


class BaseConditionalInferenceTree(BaseEstimator, metaclass=ABCMeta):
    """Base class for conditional inference trees.

    Warning: This class should not be used directly. Use derived classes instead.
    """

    @abstractmethod
    def __init__(
        self,
        *,
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
        n_jobs: int,
        random_state: Optional[int],
        verbose: int,
    ) -> None:
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
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    @abstractmethod
    def _node_value(self, y: np.ndarray) -> Union[float, np.ndarray]:
        """Calculate value in terminal node."""
        pass

    def __str__(self) -> str:
        """Class as string.

        Returns
        -------
        str
            Class as string.
        """
        return self.__repr__()

    def __repr__(self) -> str:
        """Class as string.

        Returns
        -------
        str
            Class as string.
        """
        params = self.get_params()
        class_name = self.__class__.__name__
        return class_name + str(params).replace(": ", "=").replace("'", "").replace("{", "(").replace("}", ")")

    def _select_best_feature(self, X: np.ndarray, y: np.ndarray, features: np.ndarray) -> Tuple[int, float, bool]:
        """TODO:

        Parameters
        ----------
        TODO:

        Returns
        -------
        TODO:
        """
        best_feature = features[0]
        best_pval_feature = np.inf

        for feature in features:
            if self._adjust_alpha_selector:
                # TODO: Implement this
                pass
            x = X[:, feature]

            # Check for constant feature and mute if necessary
            if np.all(x == x[0]):
                if self._feature_muting and len(self._available_features) > 1:
                    self._mute_feature(feature)
                continue

            pval_feature = self._selector_test(x=x, y=y, **self._selector_kwargs)

            # Check for feature muting
            if self._feature_muting and pval_feature >= 1 - self._alpha_selector and len(self._available_features) > 1:
                self._mute_feature(feature)
                continue

            # Update best feature
            if pval_feature < best_pval_feature:
                best_feature = feature
                best_pval_feature = pval_feature
                reject_H0_feature = best_pval_feature < self._alpha_selector

                # Check for early stopping
                if pval_feature == 0 or self._early_stopping_selector and reject_H0_feature:
                    break

        return best_feature, best_pval_feature, reject_H0_feature

    def _select_best_split(self, x: np.ndarray, y: np.ndarray, thresholds: np.ndarray) -> Tuple[float, float, bool]:
        """TODO:

        Parameters
        ----------
        TODO:

        Returns
        -------
        TODO:
        """
        best_threshold = thresholds[0]
        best_pval_threshold = np.inf

        for threshold in thresholds:
            if self._adjust_alpha_splitter:
                # TODO: Implement this
                pass

            # Check for constant split and mute if necessary
            if np.all(x == threshold):
                continue
            pval_threshold = self._splitter_test(x=x, y=y, threshold=threshold, **self._splitter_kwargs)

            # Update best threshold
            if pval_threshold < best_pval_threshold:
                best_threshold = threshold
                best_pval_threshold = pval_threshold
                reject_H0_threshold = best_pval_threshold < self._alpha_splitter

                # Check for early stopping
                if pval_threshold == 0 or self._early_stopping_splitter and reject_H0_threshold:
                    break

        return best_threshold, best_pval_threshold, reject_H0_threshold

    def _bonferonni_correction(self, adjust: str) -> None:
        """TODO:

        Parameters
        ----------
        TODO:
        """
        import pdb

        pdb.set_trace()

    def _mute_feature(self, feature: int) -> None:
        """Mute feature from being selected during tree building.

        Parameters
        ----------
        feature : int
            Index of feature to mute.
        """
        p = len(self._available_features)

        # Handle edge case
        if p == 1:
            warnings.warn(f"Unable to mute feature ({feature}), only (1) feature available for feature selection")
        else:
            # Mask feature and recalculate max_features
            idx = self._available_features == feature
            self._available_features = self._available_features[~idx]
            p = len(self._available_features)
            if self.max_features is not None:
                self._max_features = _calculate_max_value(
                    n_values=p,
                    desired_max=self.max_features,
                )
            else:
                self._max_features = p

    def _node_impurity(
        self, *, y: np.ndarray, idx: np.ndarray, n: int, n_left: int, n_right: int
    ) -> Tuple[float, float]:
        """Calculate node impurity.

        Parameters
        ----------
        TODO:

        Returns
        -------
        TODO:
        """
        impurity = self._splitter(y)
        children_impurity = (n_left / n) * self._splitter(y[idx]) + (n_right / n) * self._splitter(y[~idx])
        impurity_decrease = impurity - children_impurity
        return impurity, impurity_decrease

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursively build tree.

        Parameters
        ----------
        X : np.ndarray
            Training features.

        y : np.ndarray
            Training labels.

        depth : int, optional (default=0)
            Depth of tree.

        Returns
        -------
        Node
            Node in decision tree.
        """
        np.random.seed(self._random_state + depth)
        reject_H0_feature = False
        reject_H0_threshold = False
        impurity_decrease = -1
        n, p = X.shape
        if self.verbose > 2:
            print(f"Building tree at depth ({depth}) with ({n}) samples and ({self._max_features}) features")

        # Check for stopping criteria at node level
        if n >= self._min_samples_split and depth <= self._max_depth and not np.all(y == y[0]):
            # Feature selection
            features = random_sample(self._available_features, size=self._max_features, replace=False)
            best_feature, best_pval_feature, reject_H0_feature = self._select_best_feature(X, y, features)

        # Split selection
        if reject_H0_feature:
            x = X[:, best_feature]
            thresholds = self._threshold_method(x, max_thresholds=self._max_thresholds)
            best_threshold, best_pval_threshold, reject_H0_threshold = self._select_best_split(x, y, thresholds)

        # Calculate impurity decrease
        if reject_H0_threshold:
            idx = x <= best_threshold
            n_left = idx.sum()
            n_right = n - n_left
            if n_left >= self.min_samples_leaf and n_right >= self.min_samples_leaf:
                impurity, impurity_decrease = self._node_impurity(y=y, idx=idx, n=n, n_left=n_left, n_right=n_right)

        if impurity_decrease >= self.min_impurity_decrease:
            X_left, y_left, X_right, y_right = _node_split(
                X=X,
                y=y,
                feature=best_feature,
                threshold=best_threshold,
            )
            left_child = self._build_tree(X=X_left, y=y_left, depth=depth + 1)
            right_child = self._build_tree(X=X_right, y=y_right, depth=depth + 1)

            return Node(
                feature=best_feature,
                pval_feature=best_pval_feature,
                threshold=best_threshold,
                pval_threshold=best_pval_threshold,
                impurity=impurity,
                left_child=left_child,
                right_child=right_child,
                n_samples=n,
            )

        value = self._node_value(y)
        return Node(value=value, n_samples=n)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseConditionalInferenceTree":
        """Train estimator.

        Parameters
        ----------
        X : np.ndarray
            Training features.

        y : np.ndarray:
            Training labels.

        Returns
        -------
        self
            Fitted estimator.
        """
        # Validate hyperparameters
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
            if not _MIN_ALPHA <= value <= 1.0:
                raise ValueError(f"{attribute} ({value}) should be in range [{_MIN_ALPHA}, 1.0]")
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

        n, p = X.shape
        for attribute in ["max_features", "max_thresholds"]:
            value = getattr(self, attribute)
            if type(value) == str:
                supported = ["sqrt", "log"]
                if value not in supported:
                    raise ValueError(f"{attribute} ({value}) not supported, expected one of: {supported}")
            elif type(value) == float:
                if not 0 < float <= 1.0:
                    raise ValueError(f"{attribute} ({value}) should be in range (0, 1.0]")
            elif type(value) == int:
                if value < 1:
                    raise ValueError(f"{attribute} ({value}) should be >= 1")
            if attribute == "max_features":
                value = _calculate_max_value(n_values=p, desired_max=value) if value is not None else p
            setattr(self, f"_{attribute}", value)

        for attribute, dtype, lower_limit in zip(
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

        if self.n_jobs is None:
            self._n_jobs = 1
        else:
            max_cpus = cpu_count()
            value = min(self.n_jobs, max_cpus)
            if value < 0:
                cpus = np.arange(1, max_cpus + 1)
                if abs(value) > max_cpus:
                    value = max_cpus
                else:
                    value = cpus[value]
            self._n_jobs = value

        self._random_state = int(np.random.randint(1, 100_000)) if self.random_state is None else self.random_state
        self._verbose = min(abs(int(self.verbose)), 3)

        # Validate data
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

        if self.feature_names_in_ is None:
            self.feature_names_in_ = [f"f{j}" for j in range(X.shape[1])]

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

        # Private and fitted estimator attributes
        self._available_features = np.arange(X.shape[1], dtype=int)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        if self._estimator_type == "classifier":
            self._selector_test = ClassifierSelectorTests[self.selector]
            self._selector_kwargs = {
                "classes": self.classes_,
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

        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
        self.n_features_in_ = X.shape[1]

        # Start recursion
        self.tree_ = self._build_tree(X, y)

        return self
