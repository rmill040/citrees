from abc import ABC, abstractmethod
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


@njit(fastmath=True, nogil=True, parallel=True)
def _node_split(
    X: np.ndarray, y: np.ndarray, feature: int, threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split node based on feature and threshold."""
    idx = X[:, feature] <= threshold
    return X[idx], y[idx], X[~idx], y[~idx]


@njit(fastmath=True, nogil=True, parallel=True)
def _node_value_clf(y: np.ndarray) -> float:
    """Calculate class probabilities in terminal node for classification trees."""
    pass


@njit(fastmath=True, nogil=True, parallel=True)
def _node_value_reg(y: np.ndarray) -> float:
    """Calculate mean value in terminal node for regression trees."""
    np.mean(y)


@njit(fastmath=True, nogil=True, parallel=True)
def _node_impurity_gini(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Calculate impurity of a node using Gini index."""
    pass


@njit(fastmath=True, nogil=True, parallel=True)
def _node_impurity_entropy(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Calculate impurity of a node using entropy."""
    pass


@njit(fastmath=True, nogil=True, parallel=True)
def _node_impurity_mse(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Calculate impurity of a node using mean squared error."""
    pass


@njit(fastmath=True, nogil=True, parallel=True)
def _node_impurity_mae(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Calculate impurity of a node using mean absolute error."""
    pass


CRITERIA_CLF = {
    "gini": _node_impurity_gini,
    "entropy": _node_impurity_entropy,
}

CRITERIA_REG = {
    "mse": _node_impurity_mse,
    "mae": _node_impurity_mae,
}


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


class BaseConditionalInferenceTreeParameters(BaseModel):
    """Model for BaseConditionalInferenceTree parameters.

    Parameters
    ----------
    ADD HERE.
    """

    splitter: Literal["best", "random", "hist-local", "hist-global"]
    alpha_selector: ConstrainedFloat(gt=0.0, le=1.0) = 0.05
    alpha_splitter: ConstrainedFloat(gt=0.0, le=1.0) = 0.05
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
            alpha = values.get("feature_alpha")
        if v == "auto":
            import pdb

            pdb.set_trace()

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
        feature_alpha: float,
        split_alpha: float,
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
        self.feature_alpha = feature_alpha
        self.split_alpha = split_alpha
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
        
        # Aliases
        self._node_split = _node_split
        self._label_encoder = LabelEncoder()

    def _mute_feature(self, feature: int) -> None:
        """Mute feature from being selected."""
        pass

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
            Node in recursive splitting.
        """
        n, p = X.shape
        feature_pval = np.inf
        split_pval = np.inf
        logger.debug(f"Building tree at depth ({depth}) with ({n}) samples")

        # Check for stopping criteria at node level
        if n >= self.min_samples_split and depth <= self.max_depth and np.all(y != y[0]):
            logger.debug("Running feature selection")
            feature, feature_pval = self._selector(X, y)

        # Check for stopping critera at feature selection level
        if feature_pval < self.feature_alpha:
            logger.debug(f"Feature ({feature}) selected, p-value ({feature_pval}) < ({self.feature_alpha})")
            # Split selection
            logger.debug(f"Running split selection with feature ({feature})")
            threshold, split_pval = self._splitter(X[:, feature], y)

        # Check for stopping criteria at split selection level
        if split_pval < self.split_alpha:
            logger.debug(f"Split probability value ({split_pval}) < ({self.split_alpha})")
            if feature not in self._protected_features:
                self._protected_features.add(feature)
                logger.debug(f"Added feature ({feature}) to protected set, size={len(self._protected_features)}")

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
                split_pval=split_pval,
                threshold=threshold,
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
        # Check X
        if not isinstance(X, np.ndarray):
            if isinstance(X, list):
                X = np.array(X)
            elif hasattr(X, "values"):
                X = X.values
            else:
                try:
                    raise ValueError(
                        f"Unsupported type for X, got ({type(X)}) but expected np.ndarray, list, tuple, or pandas data structure"
                    )
                except ValueError as e:
                    logger.error(e)
                    raise

        if len(X.shape) == 1:
            X = X[:, None]

        # Check y
        if not isinstance(y, np.ndarray):
            if isinstance(y, list):
                y = np.array(y)
            elif hasattr(y, "values"):
                y = y.values
            else:
                try:
                    raise ValueError(
                        f"Unsupported type for y, got ({type(y)}) but expected np.ndarray, list, tuple, or pandas data structure"
                    )
                except ValueError as e:
                    logger.error(e)
                    raise

        if len(y.shape) > 1:
            y = y.ravel()

        # n, p = X.shape
        self._available_features = np.arange(X.shape[1], dtype=int)
        self._protected_features = set()
        self.tree_ = self._build_tree(X, y)
