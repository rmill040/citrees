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

# Pydantic aliases to keep consistency in names
ConstrainedFloat = confloat
ConstrainedInt = conint

_MIN_ALPHA = 0.001
_MAX_PERMUTATIONS = 2_000
_PVAL_PRECISION = 0.05


class Node(TypedDict, total=False):
    """Node in decision tree.

    Parameters
    ----------
    feature : int, optional
        Column index of feature.

    feature_pval : float, optional
        Probability value from feature selection.

    threshold : float, optional
        Best split point found in feature.

    threshold_pval : float, optional
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
    feature_pval: Optional[float]
    threshold: Optional[float]
    threshold_pval: Optional[float]
    impurity: Optional[float]
    value: Optional[Union[np.ndarray, float]]
    left_child: Optional["Node"]
    right_child: Optional["Node"]
    n_samples: Optional[int]


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


class BaseConditionalInferenceTreeParameters(BaseModel):
    """Model for BaseConditionalInferenceTree parameters.

    Parameters
    ----------
    """

    alpha_feature: ConstrainedFloat(ge=_MIN_ALPHA, le=1.0) = 0.05
    alpha_split: ConstrainedFloat(ge=_MIN_ALPHA, le=1.0) = 0.05
    adjust_alpha_feature: bool = True
    adjust_alpha_split: bool = True
    early_stopping_selector: bool = True
    early_stopping_splitter: bool = True
    n_permutations_selector: Union[Literal["auto"], ConstrainedInt(gt=0, lt=_MAX_PERMUTATIONS)] = "auto"
    n_permutations_splitter: Union[Literal["auto"], ConstrainedInt(gt=0, lt=_MAX_PERMUTATIONS)] = "auto"
    feature_muting: bool = True
    threshold_method: Literal["exact", "random", "histogram", "percentile"]
    max_thresholds: Optional[Union[ConstrainedFloat(gt=0.0, le=1.0), PositiveInt]] = None
    max_features: Optional[Union[PositiveInt, ConstrainedFloat(gt=0.0, le=1.0), Literal["sqrt", "log2"]]] = None
    max_depth: Optional[PositiveInt] = None
    min_samples_split: ConstrainedInt(ge=2) = 2
    min_samples_leaf: PositiveInt = 1
    min_impurity_decrease: NonNegativeFloat = 0.0
    n_jobs: int = 1
    random_state: Optional[NonNegativeInt] = None
    verbose: NonNegativeInt = 1

    @validator("max_thresholds", always=True)
    def validate_max_thresholds(
        cls: ModelMetaclass,
        v: Optional[Union[ConstrainedFloat(gt=0.0, le=1.0), PositiveInt]],
        field: ModelField,
        values: Dict[str, Any],
    ) -> Optional[Union[ConstrainedFloat(gt=0.0, le=1.0), PositiveInt]]:
        """Validate max_thresholds."""
        if values["threshold_method"] == "exact":
            _v = None
        elif values["threshold_method"] == "random":
            _v = 0.25
        elif values["threshold_method"] == "histogram":
            _v = 64
        elif values["threshold_method"] == "percentile":
            _v = 20

        setattr(cls, f"_{field.name}", _v)

        return v

    @validator("n_permutations_selector", "n_permutations_splitter", always=True)
    def validate_n_permutations(
        cls: ModelMetaclass, v: Union[Literal["auto"], PositiveInt], field: ModelField, values: Dict[str, Any]
    ) -> Union[Literal["auto"], PositiveInt]:
        """Validate n_permutations_selector."""
        alpha = values.get("alpha_feature") if field.name == "n_permutations_selector" else values.get("alpha_split")
        ll = ceil(1 / alpha)
        if v == "auto":
            # Approximate upper limit
            z = norm.ppf(1 - alpha)
            ul = ceil(z * z * (alpha * (1 - alpha)) / (_PVAL_PRECISION * _PVAL_PRECISION))
            v = max(ll, ul)
        else:
            # Need at least 1 / alpha number of permutations
            if v < ll:
                v = ll

        setattr(cls, f"_{field.name}", min(v, _MAX_PERMUTATIONS))
        return v

    @validator("max_depth", always=True)
    def validate_max_depth(cls: ModelMetaclass, v: Optional[PositiveInt], field: ModelField) -> Optional[PositiveInt]:
        """Validate max_depth."""
        setattr(cls, f"_{field.name}", np.inf if v is None else v)
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
        alpha_feature: float,
        alpha_split: float,
        adjust_alpha_feature: bool,
        adjust_alpha_split: bool,
        threshold_method: str,
        max_thresholds: Optional[Union[str, float, int]],
        early_stopping_selector: bool,
        early_stopping_splitter: bool,
        feature_muting: bool,
        n_permutations_selector: Union[str, int],
        n_permutations_splitter: Union[str, int],
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
            alpha_feature=alpha_feature,
            alpha_split=alpha_split,
            adjust_alpha_feature=adjust_alpha_feature,
            adjust_alpha_split=adjust_alpha_split,
            threshold_method=threshold_method,
            max_thresholds=max_thresholds,
            early_stopping_selector=early_stopping_selector,
            early_stopping_splitter=early_stopping_splitter,
            feature_muting=feature_muting,
            n_permutations_selector=n_permutations_selector,
            n_permutations_splitter=n_permutations_splitter,
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
        self.alpha_feature = hps.alpha_feature
        self.alpha_split = hps.alpha_split
        self.adjust_alpha_feature = hps.adjust_alpha_feature
        self.adjust_alpha_split = hps.adjust_alpha_split
        self.threshold_method = hps.threshold_method
        self.max_thresholds = hps.max_thresholds
        self.early_stopping_selector = hps.early_stopping_selector
        self.early_stopping_splitter = hps.early_stopping_splitter
        self.feature_muting = hps.feature_muting
        self.n_permutations_selector = hps.n_permutations_selector
        self.n_permutations_splitter = hps.n_permutations_splitter
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
        for param in self.get_params().keys():
            name = f"_{param}"
            if hasattr(hps, name):
                setattr(self, name, getattr(hps, name))

    def __str__(self) -> str:
        """Class as string."""
        return self.__repr__()

    def __repr__(self) -> str:
        """Class as string."""
        params = self.get_params()
        class_name = self.__class__.__name__
        return class_name + str(params).replace(": ", "=").replace("'", "").replace("{", "(").replace("}", ")")

    @abstractproperty
    def _validator(self) -> ModelMetaclass:
        """Model to validate estimator's hyperparameters."""
        pass

    @abstractmethod
    def _selector(self, X: np.ndarray, y: np.ndarray, features: np.ndarray) -> Tuple[int, float]:
        """Find most correlated feature with label."""
        pass

    @abstractmethod
    def _splitter(self, X: np.ndarray, y: np.ndarray, thresholds: np.ndarray) -> Tuple[float, float]:
        """Find optimal threshold for binary split in node."""
        pass

    @abstractmethod
    def _node_impurity(self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Calculate node impurity."""
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
            self._max_features = self.max_features

    def _feature_muting(self, feature: int) -> None:
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
        n, p = X.shape
        feature_pval = np.inf
        threshold_pval = np.inf
        logger.debug(f"Building tree at depth ({depth}) with ({n}) samples and ({self._max_features}) features")

        # Check for stopping criteria at node level
        if n >= self.min_samples_split and depth <= self._max_depth and not np.all(y == y[0]):
            # Feature selection
            features = np.random.choice(self._available_features, size=self._max_features, replace=False)
            feature, feature_pval = self._selector(X, y, features)

        # Check for stopping criteria at feature selection level
        if feature_pval < self.alpha_feature:
            # Split selection
            thresholds = np.random.choice(np.unique(X[:, feature]), size=self.max_thresholds)
            threshold, threshold_pval = self._splitter(X[:, feature], y, thresholds)

        # Check for stopping criteria at split selection level
        if threshold_pval < self.alpha_split:

            # Node split
            X_left, y_left, X_right, y_right = self._node_split(X, y, feature, threshold)

            # Node impurity
            impurity = self._node_impurity(y, y_left, y_right)

            # Recursively build subtrees for left and right branches
            left_child = self._build_tree(X=X_left, y=y_left, depth=depth + 1)
            right_child = self._build_tree(X=X_right, y=y_right, depth=depth + 1)

            # Non-terminal node
            logger.debug(f"Non-terminal node at depth ({depth})")
            return Node(
                feature=feature,
                feature_pval=feature_pval,
                threshold=threshold,
                threshold_pval=threshold_pval,
                impurity=impurity,
                left_child=left_child,
                right_child=right_child,
                n_samples=n,
            )

        # Terminal node
        logger.debug(f"Terminal node at depth ({depth})")
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

        # Private attributes for tree fitting
        self._available_features = np.arange(X.shape[1], dtype=int)
        self._calculate_max_features()

        # Fitted attributes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
        self.n_features_in_ = X.shape[1]
        self.tree_ = self._build_tree(X, y)
