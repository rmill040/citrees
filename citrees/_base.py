from abc import ABC, abstractmethod, abstractproperty
from math import ceil
from multiprocessing import cpu_count
from typing import Any, Dict, Literal, Optional, Tuple, TypedDict, Union

import numpy as np
from loguru import logger
from numba import njit
from pydantic import BaseModel, confloat, conint, NonNegativeFloat, NonNegativeInt, PositiveInt, validator
from pydantic.fields import ModelField
from pydantic.main import ModelMetaclass
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder

from ._selector import ClassifierSelectors
from ._splitter import ThresholdMethods
from ._utils import random_sample

# Pydantic aliases to keep consistency in names
ConstrainedFloat = confloat
ConstrainedInt = conint

_MIN_ALPHA = 0.001
_MAX_resamples = 5 * ceil(1 / _MIN_ALPHA)
_PVAL_PRECISION = 0.05
NResamples = Optional[Union[Literal["auto", "minimum"], ConstrainedInt(gt=1)]]
MaxOptions = Optional[Union[PositiveInt, ConstrainedFloat(gt=0.0, le=1.0), Literal["sqrt", "log2"]]]


# TODO: Cast data types for classifiers and regressors


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


class BaseConditionalInferenceTreeParameters(BaseModel):
    """Model for BaseConditionalInferenceTree parameters.

    Parameters
    ----------
    TODO:
    """

    alpha_selector: ConstrainedFloat(ge=_MIN_ALPHA, le=1.0) = 0.05
    alpha_splitter: ConstrainedFloat(ge=_MIN_ALPHA, le=1.0) = 0.05
    adjust_alpha_selector: bool = True
    adjust_alpha_splitter: bool = True
    n_resamples_selector: NResamples = "auto"
    n_resamples_splitter: NResamples = "auto"
    early_stopping_selector: bool = True
    early_stopping_splitter: bool = True
    feature_muting: bool = True
    threshold_method: Literal["exact", "random", "histogram", "percentile"] = "exact"
    max_thresholds: MaxOptions = None
    max_features: MaxOptions = None
    max_depth: Optional[PositiveInt] = None
    min_samples_split: ConstrainedInt(ge=2) = 2
    min_samples_leaf: PositiveInt = 1
    min_impurity_decrease: NonNegativeFloat = 0.0
    n_jobs: int = 1
    random_state: Optional[NonNegativeInt] = None
    verbose: NonNegativeInt = 1

    @validator("n_resamples_selector", "n_resamples_splitter", always=True)
    def validate_n_resamples(
        cls: ModelMetaclass, v: NResamples, field: ModelField, values: Dict[str, Any]
    ) -> NResamples:
        """Validate n_resamples_{selector,splitter}."""
        if v is None:
            _v = 0
        else:
            alpha = (
                values.get("alpha_selector") if field.name == "n_resamples_selector" else values.get("alpha_splitter")
            )
            ll = ceil(1 / alpha)
            if v == "auto":
                # Approximate upper limit
                z = norm.ppf(1 - alpha)
                ul = ceil(z * z * (alpha * (1 - alpha)) / (_PVAL_PRECISION * _PVAL_PRECISION))
                _v = max(ll, ul)
            elif v == "minimum":
                _v = ll
            else:
                # Need at least 1 / alpha number of resamples
                if v < ll:
                    _v = ll

        setattr(cls, f"_{field.name}", min(_v, _MAX_resamples))
        return v

    @validator("threshold_method")
    def validate_threshold_method(cls: ModelMetaclass, v: str, field: ModelField) -> str:
        """Validate threshold_method."""
        setattr(cls, f"_{field.name}", ThresholdMethods[v])
        return v

    @validator("max_depth", always=True)
    def validate_max_depth(cls: ModelMetaclass, v: Optional[PositiveInt], field: ModelField) -> Optional[PositiveInt]:
        """Validate max_depth."""
        setattr(cls, f"_{field.name}", np.inf if v is None else v)
        return v

    @validator("min_samples_leaf")
    def validate_min_samples_leaf(cls: ModelMetaclass, v: int, field: ModelField, values: Dict[str, Any]) -> int:
        """Validate min_samples_leaf."""
        _v = values["min_samples_split"] - 1 if v >= values["min_samples_split"] else v

        setattr(cls, f"_{field.name}", _v)
        return v

    @validator("n_jobs", always=True)
    def validate_n_jobs(cls: ModelMetaclass, v: int, field: ModelField) -> int:
        """Validate n_jobs."""
        max_cpus = cpu_count()
        v = min(v, max_cpus)
        if v < 0:
            cpus = np.arange(1, max_cpus + 1)
            if abs(v) > max_cpus:
                v = max_cpus
            else:
                v = cpus[v]

        setattr(cls, f"_{field.name}", max(1, v))
        return v

    @validator("random_state", always=True)
    def validate_random_state(
        cls: ModelMetaclass, v: Optional[NonNegativeInt], field: ModelField
    ) -> Optional[NonNegativeInt]:
        """Validate random_state."""
        setattr(cls, f"_{field.name}", int(np.random.randint(1, 100_000)) if v is None else v)
        return v


class BaseConditionalInferenceTree(ABC):
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
        hps = self._validator(
            selector=selector,
            splitter=splitter,
            alpha_selector=alpha_selector,
            alpha_splitter=alpha_splitter,
            adjust_alpha_selector=adjust_alpha_selector,
            adjust_alpha_splitter=adjust_alpha_splitter,
            threshold_method=threshold_method,
            max_thresholds=max_thresholds,
            early_stopping_selector=early_stopping_selector,
            early_stopping_splitter=early_stopping_splitter,
            feature_muting=feature_muting,
            n_resamples_selector=n_resamples_selector,
            n_resamples_splitter=n_resamples_splitter,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

        self.selector = hps.selector
        self.splitter = hps.splitter
        self.alpha_selector = hps.alpha_selector
        self.alpha_splitter = hps.alpha_splitter
        self.adjust_alpha_selector = hps.adjust_alpha_selector
        self.adjust_alpha_splitter = hps.adjust_alpha_splitter
        self.threshold_method = hps.threshold_method
        self.max_thresholds = hps.max_thresholds
        self.early_stopping_selector = hps.early_stopping_selector
        self.early_stopping_splitter = hps.early_stopping_splitter
        self.feature_muting = hps.feature_muting
        self.n_resamples_selector = hps.n_resamples_selector
        self.n_resamples_splitter = hps.n_resamples_splitter
        self.max_depth = hps.max_depth
        self.max_features = hps.max_features
        self.min_samples_split = hps.min_samples_split
        self.min_samples_leaf = hps.min_samples_leaf
        self.min_impurity_decrease = hps.min_impurity_decrease
        self.n_jobs = hps.n_jobs
        self.random_state = hps.random_state
        self.verbose = hps.verbose

        # Private attributes and methods
        self._node_split = _node_split
        self._label_encoder = LabelEncoder()

        # Create private attributes for all parameters as some are inferred at runtime and some change during training
        # This helps keep the attribute access notation consistent and simple
        for key in self.get_params().keys():
            p_key = f"_{key}"
            if hasattr(hps, p_key):
                value = getattr(hps, p_key)
                if key in ["selector", "splitter", "threshold_method"]:
                    # Get bound method fromm hps
                    value = getattr(value, "__func__")
            else:
                value = getattr(hps, key)
            setattr(self, p_key, value)

            # Add in permutation test attributes for selector and splitter
            if key in ["selector", "splitter"]:
                p_key = f"_{key}_test"
                value = getattr(getattr(hps, p_key), "__func__")
                setattr(self, p_key, value)

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

    @abstractproperty
    def _validator(self) -> ModelMetaclass:
        """Model to validate estimator's hyperparameters."""
        pass

    @abstractmethod
    def _node_value(self, y: np.ndarray) -> float:
        """Calculate value in terminal node."""
        pass

    def _calculate_max_features(self) -> None:
        """Calculate maximum features available."""
        p = len(self._available_features)

        if self.max_features is None:
            self._max_features = p
        elif self.max_features == "sqrt":
            self._max_features = ceil(np.sqrt(p))
        elif self.max_features == "log":
            self._max_features = ceil(np.log(p))
        elif type(self.max_features) is float:
            self._max_features = ceil(self.max_features * p)
        elif type(self.max_features) is int:
            self._max_features = min(self.max_features, p)

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
            logger.warning(f"Unable to mute feature ({feature}), only (1) feature available for feature selection")
        else:
            # Mask feature and recalculate max_features
            idx = self._available_features == feature
            self._available_features = self._available_features[~idx]
            self._calculate_max_features()

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
        logger.debug(f"Building tree at depth ({depth}) with ({n}) samples and ({self._max_features}) features")

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
                impurity, impurity_decrease = self._node_impurity(y=y, idx=idx, n_left=n_left, n_right=n_right)

        if impurity_decrease >= self.min_impurity_decrease:
            X_left, y_left, X_right, y_right = self._node_split(
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
        """Train conditional inference tree.

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
        self.feature_names_in_ = None

        # Check X
        if not isinstance(X, np.ndarray):
            if isinstance(X, (list, tuple)):
                X = np.array(X)
            elif hasattr(X, "values"):
                self.n_features_names_in_ = X.columns.tolist()
                X = X.values
            else:
                try:
                    raise ValueError(
                        f"Unsupported type for X, got ({type(X)}) but expected np.ndarray, list, tuple, or pandas data "
                        "structure"
                    )
                except ValueError as e:
                    logger.error(e)
                    raise

        if X.ndim == 1:
            X = X[:, None]

        if self.feature_names_in_ is None:
            self.feature_names_in_ = [f"f{j}" for j in range(X.shape[1])]

        # Check y
        if not isinstance(y, np.ndarray):
            if isinstance(y, (list, tuple)):
                y = np.array(y)
            elif hasattr(y, "values"):
                y = y.values
            else:
                try:
                    raise ValueError(
                        f"Unsupported type for y, got ({type(y)}) but expected np.ndarray, list, tuple, or pandas data "
                        "structure"
                    )
                except ValueError as e:
                    logger.error(e)
                    raise

        if y.ndim == 2:
            y = y.ravel()
        elif y.ndim > 2:
            try:
                raise ValueError(f"Multi-output labels are not supported for y, detected ({y.ndim - 1}) outputs")
            except ValueError as e:
                logger.error(e)
                raise

        y = self._label_encoder.fit_transform(y)

        # Compare X and y
        if len(X) != len(y):
            try:
                raise ValueError(f"Found inconsistent number of samples between X ({len(X)}) and y ({len(y)})")
            except ValueError as e:
                logger.error(e)
                raise

        self._available_features = np.arange(X.shape[1], dtype=int)
        self._calculate_max_features()

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        if self.selector in ClassifierSelectors.keys():
            self._selector_kwargs = {
                "classes": self.classes_,
                "n_resamples": self._n_resamples_selector,
                "early_stopping": self._early_stopping_selector,
                "alpha": self._alpha_selector,
                "random_state": self._random_state,
            }
            self._splitter_kwargs = {
                "n_resamples": self._n_resamples_splitter,
                "early_stopping": self._early_stopping_splitter,
                "alpha": self._alpha_splitter,
                "random_state": self._random_state,
            }
        else:
            TODO:
            self._selector_kwargs = {}
            self._splitter_kwargs = {}

        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
        self.n_features_in_ = X.shape[1]
        self.tree_ = self._build_tree(X, y)
