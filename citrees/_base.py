from abc import ABC, abstractmethod
from decimal import Decimal
from multiprocessing import cpu_count
from typing import Any, Dict, Optional, Literal, Tuple, TypedDict, Union

from loguru import logger
from numba import njit
import numpy as np
from pydantic import BaseModel, confloat, conint, NonNegativeFloat, NonNegativeInt, PositiveInt, validator
from pydantic.fields import ModelField
from pydantic.main import ModelMetaclass
from sklearn.preprocessing import LabelEncoder


# Pydantic aliases to keep consistency in names
ConstrainedFloat = confloat
ConstrainedInt = conint


class Node(TypedDict, total=False):
    """Node in decision tree.

    Parameters
    ----------
    feature : int, optional (default=None)
        Column index of feature.

    feature_pval : float, optional (default=None)
        Probability value from feature selection.

    threshold : float, optional (default=None)
        Best split point found in feature.

    threshold_pval : float, optional (default=None)
        Probability value from split selection.

    impurity : float, optional (default=None)
        Impurity measuring quality of split.

    value : Union[np.ndarray, float], optional (default=None)
        Estimate of class probabilities for classification trees and estimate of central tendency for regression trees.

    left_child : Node
        Left child node where feature value met the threshold.

    right_child : Node
        Left child node where feature value did not meet the threshold.

    n_samples : int, optional (default=None)
        Number of samples at the node.
    """

    feature: Optional[int] = None
    feature_pval: Optional[float] = None
    threshold: Optional[float] = None
    threshold_pval: Optional[float] = None
    impurity: Optional[float] = None
    value: Optional[Union[np.ndarray, float]] = None
    left_child: Optional["Node"] = None
    right_child: Optional["Node"] = None
    n_samples: Optional[int] = None


@njit(fastmath=True, nogil=True, parallel=True)
def _node_split(
    X: np.ndarray, y: np.ndarray, feature: int, threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split node based on feature and threshold.

    Parameters
    ----------
    X : np.ndarray
        Features in node.

    y : np.ndarray
        Labels in node.

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
    ADD HERE.
    """

    splitter: Literal["best", "random", "hist-local", "hist-global"]
    alpha_feature: ConstrainedFloat(gt=0.0, le=1.0) = 0.05
    alpha_split: ConstrainedFloat(gt=0.0, le=1.0) = 0.05
    n_bins: PositiveInt = 256
    early_stopping: bool = True
    feature_scanning: bool = True
    feature_muting: bool = True
    n_permutations_selector: Union[Literal["auto"], NonNegativeInt] = "auto"
    n_permutations_splitter: Union[Literal["auto"], NonNegativeInt] = "auto"
    max_depth: Optional[PositiveInt] = None
    max_features: Optional[Union[PositiveInt, ConstrainedFloat(gt=0.0, le=1.0), Literal["sqrt", "log2"]]] = None
    min_samples_split: ConstrainedInt(ge=2) = 2
    min_samples_leaf: PositiveInt = 1
    min_impurity_decrease: NonNegativeFloat = 0.0
    n_jobs: int = 1
    random_state: Optional[NonNegativeInt] = None
    verbose: NonNegativeInt = 1

    @validator("n_permutations_selector", "n_permutations_splitter")
    def validate_n_permutations(
        cls: ModelMetaclass, v: Union[Literal["auto"], PositiveInt], field: ModelField, values: Dict[str, Any]
    ) -> int:
        """Validate n_permutations_selector."""
        if field.name == "n_permutations_selector":
            alpha = values.get("alpha_feature")
        else:
            alpha = values.get("alpha_split")
        if v == "auto":
            exponent = abs(Decimal(str(alpha)).as_tuple().exponent)
            v = 10 ** exponent

        return v

    @validator("n_jobs", always=True)
    def validate_n_jobs(cls: ModelMetaclass, v: int) -> int:
        """Validate n_jobs."""
        max_cpus = cpu_count()
        v = min(v, max_cpus)
        if v < 0:
            cpus = np.arange(1, max_cpus + 1)
            if abs(v) > max_cpus:
                v = max_cpus
            else:
                v = cpus[v]

        return max(1, v)


class BaseConditionalInferenceTree(ABC):
    """Base class for conditional inference trees.

    Warning: This class should not be used directly, use derived classes instead.
    """

    @abstractmethod
    def __init__(
        self,
        *,
        criterion: str,
        selector: str,
        splitter: str,
        alpha_feature: float,
        alpha_split: float,
        n_bins: int,
        early_stopping: bool,
        feature_scanning: bool,
        feature_muting: bool,
        n_permutations_selector: int,
        n_permutations_splitter: int,
        max_depth: Optional[int],
        max_features: Optional[int],
        min_samples_split: int,
        min_samples_leaf: int,
        min_impurity_decrease: float,
        n_jobs: int,
        random_state: Optional[int],
        verbose: int,
    ) -> None:

        self.criterion = criterion
        self.selector = selector
        self.splitter = splitter
        self.alpha_feature = alpha_feature
        self.alpha_split = alpha_split
        self.n_bins = n_bins
        self.early_stopping = early_stopping
        self.feature_scanning = feature_scanning
        self.feature_muting = feature_muting
        self.n_permutations_selector = n_permutations_selector
        self.n_permutations_splitter = n_permutations_splitter
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        # Private attributes and methods
        self._node_split = _node_split
        self._label_encoder = LabelEncoder()
        if self.max_depth is None:
            self._max_depth = np.inf

    @abstractmethod
    def _node_impurity(self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Calculate node impurity.

        Parameters
        ----------
        y : np.ndarray
            Parent node labels.

        y_left : np.ndarray
            Left child node labels.

        y_right : np.ndarray
            Right child node labels

        Returns
        -------
        float
            Node impurity measure.
        """
        pass

    @abstractmethod
    def _node_value(self, y: np.ndarray) -> float:
        """Calculate value in terminal node.
        
        Parameters
        ----------
        y : np.ndarray
            Node labels.

        Returns
        -------
        float
            Node value estimate.            
        """
        pass

    @abstractmethod
    def _splitter(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Find optimal threshold for binary split in node.
        
        Parameters
        ----------
        X : np.ndarray
            ADD HERE.
        
        y : np.ndarray
            ADD HERE.
        
        Returns
        -------
        Tuple[float, float]
            Threshold and threshold probability value with order (threshold, threshold_pval).
        """
        pass

    @abstractmethod
    def _selector(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        """Find most correlated feature with label.
        
        Parameters
        ----------
        X : np.ndarray
            ADD HERE.
        
        y : np.ndarray
            ADD HERE.
        
        Returns
        -------
        Tuple[int, float]
            Feature index and probability value with order (feature, feature_pval).
        """
        pass

    def _calculate_max_features(self) -> None:
        """Calculate maximum features available."""
        p = sum(self._available_features)

        if self.max_features is None:
            self._max_features = p
        elif self.max_features == "sqrt":
            self._max_features = int(np.ceil(np.sqrt(p)))
        elif self.max_features == "log":
            self._max_features = int(np.ceil(np.log(p)))
        elif type(self.max_features) is float:
            self._max_features = int(np.ceil(self.max_features * p))
        elif type(self.max_features) is int:
            self._max_features = self.max_features

        # Catch cases where out of bounds values
        if self._max_features > p:
            self._max_features = p

    def _mute_feature(self, feature: int) -> None:
        """Mute feature from being selected during tree building.

        Parameters
        ----------
        feature : int
            Index of feature to mute.
        """
        p = sum(self._available_features)

        # Handle edge case here
        if p == 1:
            logger.warning("Unable to mute feature, (1) feature remains for feature selection")
        else:
            # Mask feature and recalculate max_features
            self._available_features[feature] = False
            self._calculate_max_features()

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursively builds tree.

        Parameters
        ----------
        X : np.ndarray
            2d array of features.

        y : np.ndarray
            1d array of labels.

        depth : int, optional (default=0)
            Depth of current recursive call.

        Returns
        -------
        Node
            Node in decision tree.
        """
        n, p = X.shape
        feature_pval = np.inf
        threshold_pval = np.inf
        logger.debug(f"Building tree at depth ({depth}) with ({n}) samples")

        # Check for stopping criteria at node level
        if n >= self.min_samples_split and depth <= self._max_depth and np.all(y != y[0]):
            logger.debug(f"Running feature selection with ({self._max_features}) features")
            feature, feature_pval = self._selector(X, y)

        # Check for stopping critera at feature selection level
        if feature_pval < self.alpha_feature:
            logger.debug(f"Feature ({feature}) selected, p-value ({feature_pval}) < ({self.alpha_feature})")
            # Split selection
            logger.debug("Running split selection")
            threshold, threshold_pval = self._splitter(X[:, feature], y)

        # Check for stopping criteria at split selection level
        if threshold_pval < self.alpha_split:
            logger.debug(f"Split probability value ({threshold_pval}) < ({self.alpha_split})")

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
        """Train decision tree.

        Parameters
        ----------
        X : np.ndarray
            Array of features.

        y : np.ndarray:
            Array of labels.

        Returns
        -------
        self
            Instance of BaseConditionalInferenceTree.
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

        # Define attributes for building trees
        self._available_features = np.array([True] * X.shape[1])
        self._calculate_max_features()

        # Fitted attributes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
        self.n_features_in_ = X.shape[1]
        self.tree_ = self._build_tree(X, y)
