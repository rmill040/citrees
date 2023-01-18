import warnings
from abc import ABCMeta, abstractmethod, abstractproperty
from math import ceil
from typing import Any, Dict, Literal, Optional, Tuple, TypedDict, Union

import numpy as np
from pydantic import BaseModel, confloat, conint, NonNegativeFloat, NonNegativeInt, PositiveInt, validator
from pydantic.main import ModelField, ModelMetaclass
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder

from ._selector import ClassifierSelectors, ClassifierSelectorTests, RegressorSelectors, RegressorSelectorTests
from ._splitter import ClassifierSplitters, ClassifierSplitterTests, RegressorSplitters, RegressorSplitterTests
from ._threshold_method import ThresholdMethods
from ._utils import calculate_max_value, estimate_mean, estimate_proba, random_sample, split_data

# Type aliases
ConstrainedInt = conint
ConstrainedFloat = confloat
ProbabilityFloat = ConstrainedFloat(gt=0.0, le=1.0)
NResamplesOption = Union[Literal["minimum", "auto"], NonNegativeInt]
MaxValuesOption = Optional[Union[Literal["sqrt", "log2"], ProbabilityFloat, NonNegativeInt]]  # type: ignore


class Node(TypedDict, total=False):
    """Node in decision tree.

    Parameters
    ----------
    feature : int
        Column index of feature.

    pval_feature : float
        Probability value from feature selection.

    threshold : float
        Best split point found in feature.

    pval_threshold : float
        Probability value from split selection.

    impurity : float
        Impurity measuring quality of split.

    value : Union[np.ndarray, float]
        Estimate of class probabilities for classification trees and estimate of central tendency for regression trees.

    left_child : Node
        Left child node where feature value met the threshold.

    right_child : Node
        Left child node where feature value did not meet the threshold.

    n_samples : int
        Number of samples at the node.
    """

    feature: int
    pval_feature: float
    threshold: float
    pval_threshold: float
    impurity: float
    value: Union[np.ndarray, float]
    left_child: "Node"
    right_child: "Node"
    n_samples: int


class BaseConditionalInferenceTreeParameters(BaseModel):
    """Model for BaseConditionalInferenceTree parameters."""

    estimator_type: Literal["classifier", "regressor"]
    selector: str
    splitter: str
    alpha_selector: ProbabilityFloat  # type: ignore
    alpha_splitter: ProbabilityFloat  # type: ignore
    adjust_alpha_selector: bool
    adjust_alpha_splitter: bool
    n_resamples_selector: NResamplesOption
    n_resamples_splitter: NResamplesOption
    early_stopping_selector: bool
    early_stopping_splitter: bool
    feature_muting: bool
    feature_scanning: bool
    threshold_scanning: bool
    threshold_method: Literal["exact", "random", "percentile", "histogram"]
    max_thresholds: MaxValuesOption
    max_features: MaxValuesOption
    max_depth: Optional[PositiveInt]
    min_samples_split: ConstrainedInt(ge=2)  # type: ignore
    min_samples_leaf: PositiveInt
    min_impurity_decrease: NonNegativeFloat
    random_state: Optional[NonNegativeInt]
    verbose: NonNegativeInt

    @validator("selector", "splitter", always=True)
    def validate_selector_splitter(cls: ModelMetaclass, v: str, field: ModelField, values: Dict[str, Any]) -> str:
        """Validate {selector,splitter}"""
        if field.name == "selector":
            registry = ClassifierSelectors if values["estimator_type"] == "classifier" else RegressorSelectors
        else:
            registry = ClassifierSplitters if values["estimator_type"] == "classifier" else RegressorSplitters
        supported = registry.keys()
        if v not in supported:
            raise ValueError(
                f"{field.name} ({v}) not supported for ({values['estimator_type']}) estimator, expected one of: "
                f"{supported}"
            )

        return v

    @validator("n_resamples_selector", "n_resamples_splitter", always=True)
    def validate_n_resamples(
        cls: ModelMetaclass, v: NResamplesOption, field: ModelField, values: Dict[str, Any]
    ) -> NResamplesOption:
        """Validate n_resamples_{selector,splitter}."""
        if type(v) == int:
            alpha = values["alpha_selector"] if "selector" in field.name else values["alpha_splitter"]
            min_samples = ceil(1 / alpha)
            if v < min_samples:
                raise ValueError(f"{field.name} ({v}) should be > {min_samples} with alpha={alpha}")

        return v


class BaseConditionalInferenceTreeEstimator(BaseEstimator, metaclass=ABCMeta):
    """Base conditional inference tree estimator."""

    @abstractproperty
    def _parameter_validator(self) -> ModelMetaclass:
        """Model for validating hyperparameters."""
        pass

    def __repr__(self) -> str:
        """Class as string.

        Returns
        -------
        str
            Class as string.
        """
        string = self.__class__.__name__ + "("
        params = self.get_params()
        n_params = len(params)
        for j, (param, value) in enumerate(params.items()):
            if type(value) == str:
                string += f"{param}='{value}'"
            else:
                string += f"{param}={value}"
            if j < n_params - 1:
                string += ", "

        string += ")"

        return string

    def __str__(self) -> str:
        """Class as string.

        Returns
        -------
        str
            Class as string.
        """
        return self.__repr__()

    def _validate_data_fit(self, *, X: Any, y: Any, estimator_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Validate data for training by checking types and casting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.

        y : array-like of shape (n_samples,)
            Training target.

        estimator_type : {"classifier", "regressor"}
            Type of estimator.

        Returns
        -------
        np.ndarray
            Training features.

        np.ndarray
            Training target.
        """
        feature_names_in = None
        if not isinstance(X, np.ndarray):
            if isinstance(X, (list, tuple)):
                X = np.array(X)
            elif hasattr(X, "values"):
                if hasattr(X, "columns"):
                    feature_names_in = X.columns.tolist()
                X = X.values
            else:
                raise ValueError(
                    f"Unsupported type for X ({type(X)}), expected np.ndarray, list, tuple, or pandas data structure"
                )

        if X.ndim == 1:
            X = X[:, None]
        elif X.ndim > 2:
            raise ValueError(
                f"Arrays with more than 2 dimensions are not supported for X, detected ({X.ndim}) dimensions"
            )

        if feature_names_in is None:
            feature_names_in = [f"f{j}" for j in range(1, X.shape[1] + 1)]
        self.feature_names_in_ = feature_names_in

        if not isinstance(y, np.ndarray):
            if isinstance(y, (list, tuple)):
                y = np.array(y)
            elif hasattr(y, "values"):
                y = y.values
            else:
                raise ValueError(
                    f"Unsupported type for y ({type(y)}), expected np.ndarray, list, tuple, or pandas data structure"
                )

        if y.ndim == 2:
            y = y.ravel()
        elif y.ndim > 2:
            raise ValueError(f"Multi-output labels are not supported for y, detected ({y.ndim - 1}) outputs")

        if len(X) != len(y):
            raise ValueError(f"Different number of samples between X ({len(X)}) and y ({len(y)})")

        X = X.astype(float)
        y = y.astype(int) if estimator_type == "classifier" else y.astype(float)

        return X, y

    def _validate_data_predict(self, X: Any) -> np.ndarray:
        """Validate data for inference by checking types and casting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Inference features.

        Returns
        -------
        np.ndarray
            Inference features.
        """
        if not hasattr(self, "feature_names_in_"):
            raise ValueError("Estimator not trainined, must call fit() method before predict()")

        feature_names = None
        if not isinstance(X, np.ndarray):
            if isinstance(X, (list, tuple)):
                X = np.array(X)
            elif hasattr(X, "values"):
                if hasattr(X, "columns"):
                    feature_names = X.columns.tolist()
                X = X.values
            else:
                raise ValueError(
                    f"Unsupported type for X ({type(X)}), expected np.ndarray, list, tuple, or pandas data structure"
                )

        if X.ndim == 1:
            X = X[:, None]
        elif X.ndim > 2:
            raise ValueError(
                f"Arrays with more than 2 dimensions are not supported for X, detected ({X.ndim}) dimensions"
            )

        if X.shape[1] != len(self.feature_names_in_):
            raise ValueError(f"X should have ({len(self.feature_names_in_)}) features, got ({X.shape[1]})")

        if feature_names:
            if set(feature_names) != set(self.feature_names_in_):
                diff = list(set(feature_names) - set(self.feature_names_in_))
                raise ValueError(f"Mismatch in feature names for X, missing ({len(diff)}) features: {diff}")

        X = X.astype(float)
        return X

    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Validate hyperparameters.

        Parameters
        ----------
        params : Dict[str, Any]
            Hyperparameters.
        """
        self._parameter_validator(**params)


class BaseConditionalInferenceTree(BaseConditionalInferenceTreeEstimator, metaclass=ABCMeta):
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
        feature_scanning: bool,
        threshold_scanning: bool,
        threshold_method: str,
        max_thresholds: Optional[Union[str, float, int]],
        max_depth: Optional[int],
        max_features: Optional[Union[str, float, int]],
        min_samples_split: int,
        min_samples_leaf: int,
        min_impurity_decrease: float,
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
        self.feature_scanning = feature_scanning
        self.threshold_scanning = threshold_scanning
        self.threshold_method = threshold_method
        self.max_thresholds = max_thresholds
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.verbose = verbose

        self._validate_params({**self.get_params(), "estimator_type": self._estimator_type})

    @abstractmethod
    def _node_value(self, y: np.ndarray) -> Union[float, np.ndarray]:
        """Calculate value in terminal node."""
        pass

    @property
    def _parameter_validator(self) -> ModelMetaclass:
        """Model for hyperparameter validation.

        Returns
        -------
        ModelMetaclass
            Model to validate hyperparameters.
        """
        return BaseConditionalInferenceTreeParameters

    def _select_best_feature(self, X: np.ndarray, y: np.ndarray, features: np.ndarray) -> Tuple[int, float, bool]:
        """Select best feature associated with the target.

        Parameters
        ----------
        X : np.ndarray
            Training features.

        y : np.ndarray
            Training target.

        features : np.ndarray
            Feature indices.

        Returns
        -------
        best_feature : int
            Index of best feature.

        best_pval_feature : float
            Pvalue of best feature selection test.

        reject_H0_feature : bool
            Whether to reject the null hypothesis of no significant association between features and target.
        """
        best_feature = features[0]
        best_pval_feature = np.inf

        for feature in features:
            x = X[:, feature]

            # Check for constant feature and mute if necessary
            if np.all(x == x[0]):
                if self._feature_muting and self._max_features > 1:
                    self._mute_feature(feature)
                continue

            pval_feature = self._selector_test(x=x, y=y, **self._selector_test_kwargs)

            # Update best feature
            if pval_feature < best_pval_feature:
                best_feature = feature
                best_pval_feature = pval_feature
                reject_H0_feature = best_pval_feature < self._alpha_selector

                # Check for early stopping
                if pval_feature == 0 or self._early_stopping_selector and reject_H0_feature:
                    break

            # Check for feature muting
            if (
                self._feature_muting
                and pval_feature >= max(self._alpha_selector, 1 - self._alpha_selector)
                and self._max_features > 1
            ):
                self._mute_feature(feature)

        return best_feature, best_pval_feature, reject_H0_feature

    def _select_best_split(self, x: np.ndarray, y: np.ndarray, thresholds: np.ndarray) -> Tuple[float, float, bool]:
        """Select best binary split.

        Parameters
        ----------
        x : np.ndarray
            Training features.

        y : np.ndarray
            Training target.

        thresholds : np.ndarray
            Thresholds to use for testing binary splits.

        Returns
        -------
        best_threshold : float
            Value of best threshold.

        best_pval_threshold : float
            Pvalue of best split selection test.

        reject_H0_threshold : bool
            Whether to reject the null hypothesis of significant binary split on data.
        """
        best_threshold = thresholds[0]
        best_pval_threshold = np.inf

        if self._adjust_alpha_splitter:
            self._bonferroni_correction(adjust="splitter", n_tests=len(thresholds))

        for threshold in thresholds:
            # Check for constant split and mute if necessary
            if np.all(x == threshold):
                continue
            pval_threshold = self._splitter_test(x=x, y=y, threshold=threshold, **self._splitter_test_kwargs)

            # Update best threshold
            if pval_threshold < best_pval_threshold:
                best_threshold = threshold
                best_pval_threshold = pval_threshold
                reject_H0_threshold = best_pval_threshold < self._alpha_splitter

                # Check for early stopping
                if pval_threshold == 0 or self._early_stopping_splitter and reject_H0_threshold:
                    break

        return best_threshold, best_pval_threshold, reject_H0_threshold

    def _bonferroni_correction(self, *, adjust: str, n_tests: int) -> None:
        """TODO:

        Parameters
        ----------
        TODO:
        """
        if n_tests > 1:
            alpha = getattr(self, f"alpha_{adjust}")
            n_resamples = getattr(self, f"n_resamples_{adjust}")
            _alpha = alpha / n_tests
            if type(n_resamples) == str:
                lower_limit = ceil(1 / _alpha)
                if n_resamples == "minimum":
                    _n_resamples = lower_limit
                else:
                    z = norm.ppf(1 - _alpha)
                    upper_limit = ceil(z * z * (1 - _alpha) / _alpha)
                    _n_resamples = max(lower_limit, upper_limit)
            else:
                _n_resamples = n_resamples * n_tests

            setattr(self, f"_alpha_{adjust}", _alpha)
            setattr(self, f"_n_resamples_{adjust}", _n_resamples)

    def _mute_feature(self, feature: int) -> None:
        """Mute feature from being selected during tree building.

        When feature muting is performed, a single feature gets removed from the set of available features during
        feature selection. Given the reduced number of features, the self._max_features attribute must be recalculated
        and if alpha adjustment is enabled, then the self._alpha_selector must also be recalculated to reflect the
        change in feature space.

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
            self._available_features: np.ndarray = self._available_features[~idx]  # make mypy happy with type
            p = len(self._available_features)
            if self.max_features:
                self._max_features = calculate_max_value(
                    n_values=p,
                    desired_max=self.max_features,
                )
            else:
                self._max_features = p

            # Update alpha if needed
            if self._adjust_alpha_selector:
                self._bonferroni_correction(adjust="selector", n_tests=self._max_features)

    def _scan_features(self, X: np.ndarray, y: np.ndarray, features: np.ndarray) -> np.ndarray:
        """ADD HERE.

        Parameters
        ----------

        Returns
        -------
        """
        scores = np.array([self._selector(X[:, feature], y, **self._selector_kwargs) for feature in features])
        ranks = np.argsort(scores)[::-1]
        return features[ranks]

    def _scan_thresholds(self, x: np.ndarray, y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """ADD HERE.

        Parameters
        ----------

        Returns
        -------
        """
        scores = []
        for threshold in thresholds:
            idx = x <= threshold
            scores.append(self._splitter(y[idx]) + self._splitter(y[~idx]))
        ranks = np.argsort(scores)
        return thresholds[ranks]

    def _node_impurity(
        self, *, y: np.ndarray, idx: np.ndarray, n: int, n_left: int, n_right: int
    ) -> Tuple[float, float]:
        """Calculate node impurity.

        Parameters
        ----------
        y : np.ndarray
            ADD HERE.

        idx : np.ndarray
            ADD HERE.

        n : int
            ADD HERE.

        n_left : int
            ADD HERE.

        n_right : int
            ADD HERE.

        Returns
        -------
        TODO:
        """
        impurity = self._splitter(y)
        children_impurity = (n_left / n) * self._splitter(y[idx]) + (n_right / n) * self._splitter(y[~idx])
        impurity_decrease = impurity - children_impurity
        return impurity, impurity_decrease

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Recursively build tree.

        Parameters
        ----------
        X : np.ndarray
            Training features.

        y : np.ndarray
            Training target.

        depth : int
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
            print(f"Building tree at depth ({depth}) with ({n}) samples")

        # Check for stopping criteria at node level
        if n >= self._min_samples_split and depth <= self._max_depth and not np.all(y == y[0]):
            # Feature selection

            # self._max_features is automatically updated whenever a feature is muted so no need to recalculate during
            # each iteration as we do below for self._max_thresholds
            features = random_sample(x=self._available_features, size=self._max_features, replace=False)
            if self._feature_scanning and self._early_stopping_selector:
                features = self._scan_features(X, y, features)
            best_feature, best_pval_feature, reject_H0_feature = self._select_best_feature(X, y, features)

        if reject_H0_feature:
            # Split selection
            x = X[:, best_feature]

            # Always recalculate self._max_thresholds given the sample size may change at each recursive call
            n_unique = len(np.unique(x))
            if self.max_thresholds:
                self._max_thresholds = calculate_max_value(n_values=n_unique, desired_max=self.max_thresholds)
            else:
                self._max_thresholds = n_unique

            thresholds = self._threshold_method(x, max_thresholds=self._max_thresholds)
            if self._threshold_scanning and self._early_stopping_splitter:
                thresholds = self._scan_thresholds(x, y, thresholds)
            best_threshold, best_pval_threshold, reject_H0_threshold = self._select_best_split(x, y, thresholds)

        # Calculate impurity decrease
        if reject_H0_threshold:
            idx = x <= best_threshold
            n_left = idx.sum()
            n_right = n - n_left
            if n_left >= self._min_samples_leaf and n_right >= self._min_samples_leaf:
                impurity, impurity_decrease = self._node_impurity(  # type: ignore
                    y=y, idx=idx, n=n, n_left=n_left, n_right=n_right
                )

        if impurity_decrease >= self._min_impurity_decrease:
            self.feature_importances_[best_feature] += impurity_decrease
            X_left, y_left, X_right, y_right = split_data(
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

    def fit(self, X: Any, y: Any) -> "BaseConditionalInferenceTree":
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
        self._available_features = np.arange(p, dtype=int)
        self._max_depth = self.max_depth if self.max_depth else np.inf
        self._random_state = int(np.random.randint(1, 100_000)) if self.random_state is None else self.random_state
        self._verbose = min(self.verbose, 3)
        for param in ["n_resamples_selector", "n_resamples_splitter"]:
            value = getattr(self, param)
            if type(value) == str:
                alpha = getattr(self, "alpha_selector") if "selector" in param else getattr(self, "alpha_splitter")
                lower_limit = ceil(1 / alpha)
                if value == "minimum":
                    value = lower_limit
                else:
                    # Approximate upper limit
                    z = norm.ppf(1 - alpha)
                    upper_limit = ceil(z * z * (1 - alpha) / alpha)
                    value = max(lower_limit, upper_limit)
            setattr(self, f"_{param}", value)

        # No need to adjust alphas yet if flags are enabled. The alpha adjustments happen during each call to
        # self._selector_test and self._splitter_test
        for param in ["alpha_selector", "alpha_splitter", "early_stopping_selector", "early_stopping_splitter"]:
            setattr(self, f"_{param}", getattr(self, param))

        self._selector: Any
        self._selector_test: Any
        self._splitter: Any
        self._splitter_test: Any
        if self._estimator_type == "classifier":
            n_classes = getattr(self, "n_classes_", len(np.unique(y)))
            self._selector = ClassifierSelectors[self.selector]
            self._selector_kwargs = {
                "n_classes": n_classes,
            }
            self._selector_test = ClassifierSelectorTests[self.selector]
            self._selector_test_kwargs = {
                "n_classes": n_classes,
                "n_resamples": self._n_resamples_selector,
                "early_stopping": self._early_stopping_selector,
                "alpha": self._alpha_selector,
                "random_state": self._random_state,
            }
            self._splitter = ClassifierSplitters[self.splitter]
            self._splitter_test = ClassifierSplitterTests[self.splitter]
            self._label_encoder = LabelEncoder()
            y = self._label_encoder.fit_transform(y)
        else:
            self._selector = RegressorSelectors[self.selector]
            self._selector_kwargs = {
                "standardize": False,
            }
            self._selector_test = RegressorSelectorTests[self.selector]
            self._selector_test_kwargs = {
                "standardize": False,
            }
            self._splitter = RegressorSplitters[self.splitter]
            self._splitter_test = RegressorSplitterTests[self.splitter]

        self._splitter_test_kwargs = {
            "n_resamples": self._n_resamples_splitter,
            "early_stopping": self._early_stopping_splitter,
            "alpha": self._alpha_splitter,
            "random_state": self._random_state,
        }

        self._threshold_method: Any = ThresholdMethods[self.threshold_method]

        if self.max_features:
            self._max_features = calculate_max_value(n_values=p, desired_max=self.max_features)
        else:
            self._max_features = p

        # Set rest of parameters as private attributes
        for param, value in self.get_params().items():
            p_param = f"_{param}"
            if not hasattr(self, p_param):
                setattr(self, p_param, value)

        # Fitted attributes
        if self._estimator_type == "classifier":
            self.classes_ = np.unique(y)
            if not hasattr(self, "n_classes_"):
                self.n_classes_ = n_classes

        self.feature_importances_ = np.zeros(p, dtype=float)
        self.n_features_in_ = p
        self.tree_ = self._build_tree(X, y, depth=1)

        # Normalize feature importances
        self.feature_importances_ /= self.feature_importances_.sum()

        return self

    def _predict_value(self, x: np.ndarray, tree: Optional[Node] = None) -> np.ndarray:
        """ADD HERE.

        Parameters
        ----------

        Returns
        -------
        """
        # If we have a value => return value as the prediction
        if tree is None:
            tree = self.tree_

        if "value" in tree:
            return tree["value"]

        # Determine if we will follow left or right branch
        feature_value = x[tree["feature"]]
        branch = tree["left_child"] if feature_value <= tree["threshold"] else tree["right_child"]

        # Recurse subtree
        return self._predict_value(x, branch)

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target."""
        pass


class ConditionalInferenceTreeClassifier(BaseConditionalInferenceTree, ClassifierMixin):
    """Conditional inference tree classifier.

    Parameters
    ----------
    selector : {"mc", "mi", "hybrid"}, default="mc"
        Method for feature selection.

    splitter : {"gini", "entropy"}, default="gini"
        Method for split selection.

    alpha_selector : float, default=0.05
        Alpha for feature selection.

    alpha_splitter : float, default=0.05
        Alpha for split selection.

    adjust_alpha_selector : bool, default=True
        Whether to perform a Bonferroni correction during feature selection.

    adjust_alpha_splitter : bool, default=True
        Whether to perform a Berferonni correction during split selection.

    n_resamples_selector : {"auto", "minimum"} or int, default="auto"
        Number of resamples to use in permutation test for feature selection.

    n_resamples_splitter : {"auto", "minimum"} or int, default="auto"
        Number of resamples to use in permutation test for split selection.

    early_stopping_selector : bool, default=True
        Use early stopping during feature selection.

    early_stopping_splitter : bool, default=True
        Use early stopping during split selection.

    feature_muting: bool

    feature_scanning: bool

    threshold_scanning: bool

    threshold_method : {"exact", "random", "histogram", "percentile"}, default="exact"
        Method to calculate thresholds for a feature used during split selection.

    max_thresholds : int, default=None
        Number of bins to use when using histogram splitters.

    max_features: MaxValuesOption

    max_depth: Optional[PositiveInt]

    min_samples_split: ConstrainedInt(ge=2)

    min_samples_leaf: PositiveInt

    min_impurity_decrease: NonNegativeFloat

    random_state: Optional[NonNegativeInt]

    verbose: NonNegativeInt

    Attributes
    ----------
    classes_ : np.ndarray
        Unique class labels.

    n_classes_ : int
        Number of classes.

    feature_importances_ : np.ndarray
        Feature importances for each feature.

    n_features_in_ : int
        Number of features seen during fit.

    tree_ : Node
        The underlying decision tree.
    """

    def __init__(
        self,
        *,
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
        max_features: Optional[Union[str, float, int]] = None,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: Optional[int] = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(
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
            feature_muting=feature_muting,
            feature_scanning=feature_scanning,
            threshold_scanning=threshold_scanning,
            threshold_method=threshold_method,
            max_thresholds=max_thresholds,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            verbose=verbose,
        )

    def _node_value(self, y: np.ndarray) -> Union[float, np.ndarray]:
        """Calculate value in terminal node.

        Parameters
        ----------
        y : np.ndarray
            Training target.

        Returns
        -------
        np.ndarray
            Class probabilities.
        """
        return estimate_proba(y=y, n_classes=self.n_classes_)

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

        return np.array([self._predict_value(x) for x in X])

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


class ConditionalInferenceTreeRegressor(BaseConditionalInferenceTree, RegressorMixin):
    """Conditional inference tree regressor.

    Parameters
    ----------
    selector : {"pc", "dc", "hybrid"}, optional (default="pc")
        Method for feature selection.

    splitter : {"mse", "mae"}, optional (default="mse")
        Method for split selection.

    alpha_selector : float, optional (default=0.05)
        Alpha for feature selection.

    alpha_splitter : float, optional (default=0.05)
        Alpha for split selection.

    adjust_alpha_selector : bool, optional (default=True)
        ADD HERE.

    adjust_alpha_splitter : bool, optional (default=True)
        ADD HERE.

    ...

    threshold_method : {"exact", "random", "histogram", "percentile"}, optional (default="exact")
        Method to calculate thresholds for a feature used during split selection.

    max_thresholds : int, optional (default=256)
        Number of bins to use when using histogram splitters.

    early_stopping_selector : bool, optional (default=True)
        Use early stopping during feature selection.

    early_stopping_splitter : bool, optional (default=True)
        Use early stopping during split selection.
    ...

    Attributes
    ----------
    classes_ : np.ndarray
        Unique class labels.

    n_classes_ : int
        Number of classes.

    feature_importances_ : np.ndarray
        Feature importances estimated during training.

    n_features_in_ : int
        Number of

    tree_ : Node
        ADD HERE.
    """

    def __init__(
        self,
        *,
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
        max_features: Optional[Union[str, float, int]] = None,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: Optional[int] = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(
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
            feature_muting=feature_muting,
            feature_scanning=feature_scanning,
            threshold_scanning=threshold_scanning,
            threshold_method=threshold_method,
            max_thresholds=max_thresholds,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            verbose=verbose,
        )

    def _node_value(self, y: np.ndarray) -> Union[float, np.ndarray]:
        """Calculate value in terminal node.

        Parameters
        ----------
        y : np.ndarray
            Training target.

        Returns
        -------
        float
            Mean of target in node.
        """
        return estimate_mean(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target.

        Parameters
        ----------
        X : np.ndarray
            Features.

        Returns
        -------
        np.ndarray
            Predicted target.
        """
        X = self._validate_data_predict(X)

        return np.array([self._predict_value(x) for x in X])
