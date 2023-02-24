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
from ._utils import calculate_max_value, estimate_mean, estimate_proba, split_data

# Type aliases
ConstrainedInt = conint
ConstrainedFloat = confloat
ProbabilityFloat = ConstrainedFloat(gt=0.0, le=1.0)
NResamplesOption = Optional[Union[Literal["minimum", "maximum", "auto"], NonNegativeInt]]
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
    check_for_unused_parameters: bool

    @validator("selector", "splitter", always=True)
    def validate_selector_splitter(cls: ModelMetaclass, v: str, field: ModelField, values: Dict[str, Any]) -> str:
        """Validate {selector,splitter}."""
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
            attribute = "selector" if "selector" in field.name else "splitter"
            alpha = values[f"alpha_{attribute}"]
            min_samples = ceil(1 / alpha)
            if v < min_samples:
                raise ValueError(f"{field.name} ({v}) should be >= {min_samples} with alpha_{attribute}={alpha}")

        return v


class BaseConditionalInferenceTreeEstimator(BaseEstimator, metaclass=ABCMeta):
    """Base conditional inference tree estimator."""

    @abstractproperty
    def _parameter_model(self) -> ModelMetaclass:
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
            raise ValueError("Estimator not trained, must call fit() method before predict()")

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

    def _validate_parameters(self, params: Dict[str, Any]) -> None:
        """Validate hyperparameters.

        Parameters
        ----------
        params : Dict[str, Any]
            Hyperparameters.
        """
        self._parameter_model(**params)

    def _validate_parameter_combinations(self) -> None:
        """Runtime check to determine if any hyperparameters are unused due to constraints.

        Selector constraints:
        1. n_resamples_selector is None =>
            - adjust_alpha_selector = False
            - feature_muting = False
            - early_stopping_selector = False

        2. early_stopping_selector == False =>
            - feature_scanning == False

        Splitter constraints:
        1. n_resamples_splitter is None =>
            - adjust_alpha_splitter = False
            - early_stopping_splitter = False

        2. early_stopping_splitter = False =>
            - threshold_scanning = False
        """
        params = self.get_params()

        flags = []
        if params["n_resamples_selector"] is None:
            flags = [
                key for key in ["adjust_alpha_selector", "feature_muting", "early_stopping_selector"] if params[key]
            ]
        if flags:
            warnings.warn(
                "Unused hyperparameter(s) detected: When n_resamples_selector=None, hyperparameter(s) "
                f"({', '.join(flags)}) should be False"
            )

        if not params["early_stopping_selector"] and params["feature_scanning"]:
            warnings.warn(
                "Unused hyperparameter detected: When early_stopping_selector=False, hyperparameter "
                "('feature_scanning') should be False"
            )

        flags = []
        if params["n_resamples_splitter"] is None:
            flags = [key for key in ["adjust_alpha_splitter", "early_stopping_splitter"] if params[key]]
        if flags:
            warnings.warn(
                "Unused hyperparameter(s) detected: When n_resamples_splitter=None, hyperparameter(s) "
                f"({', '.join(flags)}) should be False"
            )

        if not params["early_stopping_splitter"] and params["threshold_scanning"]:
            warnings.warn(
                "Unused hyperparameters detected: When early_stopping_splitter=False, hyperparameter "
                "('feature_scanning') should be False"
            )


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
        n_resamples_selector: Optional[Union[str, int]],
        n_resamples_splitter: Optional[Union[str, int]],
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
        check_for_unused_parameters: bool,
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
        self.check_for_unused_parameters = check_for_unused_parameters

        self._validate_parameters({**self.get_params(), "estimator_type": self._estimator_type})
        if self.check_for_unused_parameters:
            self._validate_parameter_combinations()

    @abstractmethod
    def _node_value(self, y: np.ndarray) -> Union[float, np.ndarray]:
        """Calculate value in terminal node."""
        pass

    @property
    def _parameter_model(self) -> ModelMetaclass:
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

        best_pval : float
            Pvalue of best feature selection test.

        reject_H0 : bool
            Whether to reject the null hypothesis of no significant association between features and target.
        """
        best_feature = features[0]
        best_pval = np.inf
        reject_H0 = self._n_resamples_selector is None  # Hack to force True when permutation testing is disabled
        best_metric = -np.inf

        for feature in features:
            x = X[:, feature]

            # Feature selection with permutation testing
            if self._n_resamples_selector:
                pval_feature = self._selector_test(x=x, y=y, **self._selector_test_kwargs)

                # Update best feature
                if pval_feature < best_pval:
                    best_feature = feature
                    best_pval = pval_feature
                    reject_H0 = best_pval < self._alpha_selector

                    # Check for early stopping
                    if pval_feature == 0 or (self._early_stopping_selector and reject_H0):
                        break

                # Check for feature muting
                alpha = max(self._alpha_selector, 1 - self._alpha_selector)
                if self._feature_muting and pval_feature >= alpha and len(self._available_features) > 1:
                    self._mute_feature(feature=feature, reason=f"FEATURE PVAL ({pval_feature}) >= ALPHA ({alpha})")

            # Feature selection without permutation testing
            else:
                metric = self._selector(x=x, y=y, **self._selector_kwargs)
                if metric > best_metric:
                    best_feature = feature
                    best_metric = metric

        return best_feature, best_pval, reject_H0

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

        best_pval : float
            Pvalue of best split selection test.

        reject_H0 : bool
            Whether to reject the null hypothesis of significant binary split on data.
        """
        best_threshold = thresholds[0]
        best_pval = np.inf
        reject_H0 = self._n_resamples_splitter is None  # Hack to force True when permutation testing is disabled
        best_metric = np.inf

        if self._adjust_alpha_splitter and self._n_resamples_splitter is not None:
            self._bonferroni_correction(adjust="splitter", n_tests=len(thresholds))

        for threshold in thresholds:
            # Split selection with permutation testing
            if self._n_resamples_splitter:
                pval_threshold = self._splitter_test(x=x, y=y, threshold=threshold, **self._splitter_test_kwargs)

                # Update best threshold
                if pval_threshold < best_pval:
                    best_threshold = threshold
                    best_pval = pval_threshold
                    reject_H0 = best_pval < self._alpha_splitter

                    # Check for early stopping
                    if pval_threshold == 0 or (self._early_stopping_splitter and reject_H0):
                        break

            # Split selection without permutation testing
            else:
                metric = self._split_impurity(x=x, y=y, threshold=threshold)
                if metric < best_metric:
                    best_threshold = threshold
                    best_metric = metric

        return best_threshold, best_pval, reject_H0

    def _bonferroni_correction(self, *, adjust: str, n_tests: int) -> None:
        """Implement Bonferroni correction to account for multiple hypothesis tests.

        During training, when Bonferonni correction is enabled, alpha will be adjusted based on the number of hypothesis
        tests, assuming permutation tests are used. Likewise, the number of resamples for a permutation test will be
        updated to enough resamples are run to compare the achieved significance level against alpha.

        Parameters
        ----------
        adjust : {"splitter", "selector"}
            Which attributes to use for Bonferroni correction.

        n_tests : int
            Number of hypothesis tests to perform.
        """
        if n_tests > 1:
            alpha = getattr(self, f"alpha_{adjust}")
            n_resamples = getattr(self, f"n_resamples_{adjust}")
            _alpha = alpha / n_tests
            if type(n_resamples) == str:
                lower_limit = ceil(1 / _alpha)
                if n_resamples == "minimum":
                    _n_resamples = lower_limit
                elif n_resamples == "maximum":
                    _n_resamples = ceil(1 / (4 * alpha * alpha))
                else:
                    z = norm.ppf(1 - _alpha)
                    upper_limit = ceil(z * z * (1 - _alpha) / _alpha)
                    _n_resamples = max(lower_limit, upper_limit)
            else:
                _n_resamples = n_resamples * n_tests

            setattr(self, f"_alpha_{adjust}", _alpha)
            setattr(self, f"_n_resamples_{adjust}", _n_resamples)

    def _mute_feature(self, *, feature: int, reason: str) -> None:
        """Mute feature from being selected during tree building.

        When feature muting is performed, a single feature gets removed from the set of available features during
        feature selection. Given the reduced number of features, the self._max_features attribute must be recalculated
        and if alpha adjustment is enabled, then the self._alpha_selector must also be recalculated to reflect the
        change in feature space.

        Parameters
        ----------
        feature : int
            Index of feature to mute.

        reason : str
            Reason for muting feature.
        """
        # Drop feature and recalculate maximum features available
        idx = self._available_features == feature
        self._available_features: np.ndarray = self._available_features[~idx]  # make mypy happy with type

        p = len(self._available_features)
        if self.verbose > 2:
            print(f"Muted feature ({self.feature_names_in_[feature]}) because ({reason}), ({p}) features " "available")

        if p > 0:
            self._max_features = (
                calculate_max_value(n_values=p, desired_max=self.max_features) if self.max_features else p
            )

            # Update alpha if needed
            if self._adjust_alpha_selector and self._n_resamples_selector is not None:
                self._bonferroni_correction(adjust="selector", n_tests=self._max_features)
        else:
            self._max_features = 0

    def _scan_features(self, X: np.ndarray, y: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Perform feature scanning to return the feature indices that are most associated with the target.

        Parameters
        ----------
        X : np.ndarray
            Features.

        y : np.ndarray
            Target.

        features : np.ndarray
            Feature indices

        Returns
        -------
        np.ndarray
            Feature indices sorted in descending order based on strength of association with the target.
        """
        scores = np.array([self._selector(X[:, feature], y, **self._selector_kwargs) for feature in features])
        ranks = np.argsort(scores)[::-1]

        return features[ranks]

    def _scan_thresholds(self, x: np.ndarray, y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """Perform threshold scanning to return the threshold that result in the best binary split.

        Parameters
        ----------
        x : np.ndarray
            Feature.

        y : np.ndarray
            Target.

        thresholds : np.ndarray
            Thresholds.

        Returns
        -------
        np.ndarray
            Thresholds sorted in ascending order based on the impurity in children nodes resulting from the binary
            split.
        """
        scores = []
        for threshold in thresholds:
            idx = x <= threshold
            scores.append(self._splitter(y[idx]) + self._splitter(y[~idx]))
        ranks = np.argsort(scores)

        return thresholds[ranks]

    def _split_impurity(self, *, x: np.ndarray, y: np.ndarray, threshold: float) -> float:
        """Perform binary split and calculate split impurity as weighted sum of children node impurities.

        Parameters
        ----------
        x : np.ndarray
            Feature.

        y : np.ndarray
            Target.

        threshold : float
            Threshold value to create binary split on feature.

        Returns
        -------
        float
            Weighted impurity metric.
        """
        idx = x <= threshold
        n = len(idx)
        n_left = sum(idx)
        n_right = n - n_left

        return (n_left / n) * self._splitter(y[idx]) + (n_right / n) * self._splitter(y[~idx])

    def _node_impurity(
        self, *, y: np.ndarray, idx: np.ndarray, n: int, n_left: int, n_right: int
    ) -> Tuple[float, float]:
        """Calculate node impurity.

        Parameters
        ----------
        y : np.ndarray
            Target.

        idx : np.ndarray
            Array of bool values used to create binary split.

        n : int
            Total number of samples in parent node.

        n_left : int
            Total number of samples in left child node.

        n_right : int
            Total number of samples in right child node.

        Returns
        -------
        parent_impurity : float
            Parent node impurity.

        impurity_decrease : float
            Node impurity decrease after binary split on threshold.
        """
        parent_impurity = self._splitter(y)
        children_impurity = (n_left / n) * self._splitter(y[idx]) + (n_right / n) * self._splitter(y[~idx])
        impurity_decrease = parent_impurity - children_impurity

        return parent_impurity, impurity_decrease

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
        prng = np.random.RandomState(self._random_state)
        reject_H0_feature = False
        reject_H0_threshold = False
        impurity_decrease = -1
        n, p = X.shape

        # Check for stopping criteria at node level
        if n >= self._min_samples_split and depth <= self._max_depth and not np.all(y == y[0]):
            if self.verbose > 2:
                print(f"Building tree at depth ({depth}) with ({n}) samples")

            # Check for constant features and mute if constant
            for feature in self._available_features:
                x = X[:, feature]
                if np.all(x == x[0]):
                    self._mute_feature(feature=feature, reason="FEATURE IS CONSTANT")

            # Feature selection

            # Note: self._max_features is automatically updated only when a feature is muted so no need to recalculate
            # during each iteration as we do below for self._max_thresholds

            # If early stopping, we scan features and sort based on most promising features but no need to also
            # randomly permute since a random sample is already taken from the feature set
            if len(self._available_features):
                features = (
                    prng.choice(self._available_features, size=self._max_features, replace=False)
                    if len(self._available_features) > 1
                    else self._available_features
                )
                if self._early_stopping_selector and self._scan_features and len(features) > 1:
                    features = self._scan_features(X, y, features)
                best_feature, best_pval_feature, reject_H0_feature = self._select_best_feature(X, y, features)

        if reject_H0_feature:
            # Split selection
            x = X[:, best_feature]

            # Always recalculate self._max_thresholds given the sample size will change at each recursive call
            x_unique = np.unique(x)
            n_unique = len(x_unique)
            if n_unique > 4:
                self._max_thresholds = (
                    calculate_max_value(n_values=n_unique, desired_max=self.max_thresholds)
                    if self.max_thresholds
                    else n_unique
                )

                # If early stopping, we either scan thresholds and sort based on most promising thresholds or create a
                # random permutation of the values to help randomize chance of finding a good split and early stopping
                thresholds = self._threshold_method(
                    x, max_thresholds=self._max_thresholds, random_state=self._random_state
                )
            else:
                # With smaller samples, just use all midpoints as potential split points
                thresholds = (x_unique[:-1] + x_unique[1:]) / 2
                self._max_thresholds = len(thresholds)

            if self._early_stopping_splitter:
                thresholds = (
                    self._scan_thresholds(x, y, thresholds)
                    if self._threshold_scanning and len(thresholds) > 1
                    else prng.permutation(thresholds)
                )
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
        self._random_state = int(np.random.randint(1, 1_000_000)) if self.random_state is None else self.random_state
        self._verbose = min(self.verbose, 3)
        for param in ["n_resamples_selector", "n_resamples_splitter"]:
            value = getattr(self, param)
            if type(value) == str:
                alpha = self.alpha_selector if "selector" in param else self.alpha_splitter
                lower_limit = ceil(1 / alpha)
                if value == "minimum":
                    value = lower_limit
                elif value == "maximum":
                    value = ceil(1 / (4 * alpha * alpha))
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

        common_selector_kwargs = {
            "random_state": self._random_state,
        }
        common_selector_test_kwargs = {
            "n_resamples": self._n_resamples_selector,
            "early_stopping": self._early_stopping_selector,
            "alpha": self._alpha_selector,
            "random_state": self._random_state,
        }
        if self._estimator_type == "classifier":
            self._selector = ClassifierSelectors[self.selector]
            self._selector_test = ClassifierSelectorTests[self.selector]
            self._splitter = ClassifierSplitters[self.splitter]
            self._splitter_test = ClassifierSplitterTests[self.splitter]

            n_classes = getattr(self, "n_classes_", len(np.unique(y)))
            self._selector_kwargs = {
                **common_selector_kwargs,
                **{"n_classes": n_classes},
            }
            self._selector_test_kwargs = {
                **common_selector_test_kwargs,
                **{"n_classes": n_classes},
            }

            self._label_encoder = LabelEncoder()
            y = self._label_encoder.fit_transform(y)
        else:
            self._selector = RegressorSelectors[self.selector]
            self._selector_test = RegressorSelectorTests[self.selector]
            self._splitter = RegressorSplitters[self.splitter]
            self._splitter_test = RegressorSplitterTests[self.splitter]

            self._selector_kwargs = {
                **common_selector_kwargs,
                **{"standardize": True},
            }
            self._selector_test_kwargs = {
                **common_selector_test_kwargs,
                **{"standardize": True},
            }

        self._splitter_test_kwargs = {
            "n_resamples": self._n_resamples_splitter,
            "early_stopping": self._early_stopping_splitter,
            "alpha": self._alpha_splitter,
            "random_state": self._random_state,
        }

        self._threshold_method: Any = ThresholdMethods[self.threshold_method]
        if self.threshold_method != "exact" and self.max_thresholds is None:
            warnings.warn(
                f"Using threshold_method='{self.threshold_method}' with max_thresholds=None is not recommended, "
                "consider reducing max_thresholds to speed up split selection."
            )
        self._max_features = calculate_max_value(n_values=p, desired_max=self.max_features) if self.max_features else p

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
        fi_sum = self.feature_importances_.sum()
        if fi_sum:
            self.feature_importances_ /= fi_sum

        return self

    def _predict_value(self, x: np.ndarray, tree: Optional[Node] = None) -> Union[float, np.ndarray]:
        """Predict target for single sample.

        Parameters
        ----------
        x : np.ndarray
            Feature.

        tree : Node, default=None
            Fitted decision tree.

        Returns
        -------
        Union[float, np.ndarray]
            Predicted value for sample.
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

    n_resamples_selector : {"auto", "minimum", "maximum"} or int, default="auto"
        Number of resamples to use in permutation test for feature selection.

    n_resamples_splitter : {"auto", "minimum", "maximum"} or int, default="auto"
        Number of resamples to use in permutation test for split selection.

    early_stopping_selector : bool, default=True
        Use early stopping during feature selection.

    early_stopping_splitter : bool, default=True
        Use early stopping during split selection.

    feature_muting : bool, default=True
        Whether to perform feature muting.

    feature_scanning : bool, default=True
        Whether to perform feature scanning.

    max_features : {"sqrt", "log2"}, int, or float, default=None
        Maximum number of features to use for feature selection.

    threshold_method : {"exact", "random", "histogram", "percentile"}, default="exact"
        Method to calculate thresholds on a feature used during split selection.

    threshold_scanning : bool, default=True
        Whether to perform threshold scanning.

    max_thresholds : {"sqrt", "log2"}, int, or float, default=None
        Maximum number of thresholds to use for split selection.

    max_depth : int, default=None
        Maximum depth to grow tree.

    min_samples_split : int, default=2
        Minimim samples required for a valid binary split.

    min_samples_leaf : int, default=1
        Minimum number of samples in a leaf node.

    min_impurity_decrease : float, default=0.0
        Minimum impurity decrease required for a valid binary split.

    random_state : int, default=None
        Random seed.

    verbose : int, default=1
        Controls verbosity when fitting and predicting.

    check_for_unused_parameters : bool, default=False
        Check for unused hyperparameters. Useful for debugging.

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

    feature_names_in_ : List[str]
        List of feature names seen during fit.

    tree_ : Node
        Underlying decision tree object.
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
        n_resamples_selector: Optional[Union[str, int]] = "auto",
        n_resamples_splitter: Optional[Union[str, int]] = "auto",
        early_stopping_selector: bool = True,
        early_stopping_splitter: bool = True,
        feature_muting: bool = True,
        feature_scanning: bool = True,
        max_features: Optional[Union[str, float, int]] = None,
        threshold_method: str = "exact",
        threshold_scanning: bool = True,
        max_thresholds: Optional[Union[str, float, int]] = None,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: Optional[int] = None,
        verbose: int = 1,
        check_for_unused_parameters: bool = False,
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
            max_features=max_features,
            threshold_method=threshold_method,
            threshold_scanning=threshold_scanning,
            max_thresholds=max_thresholds,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            verbose=verbose,
            check_for_unused_parameters=check_for_unused_parameters,
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
    selector : {"pc", "dc", "hybrid"}, default="pc"
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
        Whether to perform a Berferonni correction during split selection.

    n_resamples_selector : {"auto", "minimum", "maximum"} or int, default="auto"
        Number of resamples to use in permutation test for feature selection.

    n_resamples_splitter : {"auto", "minimum", "maximum"} or int, default="auto"
        Number of resamples to use in permutation test for split selection.

    early_stopping_selector : bool, default=True
        Use early stopping during feature selection.

    early_stopping_splitter : bool, default=True
        Use early stopping during split selection.

    feature_muting : bool, default=True
        Whether to perform feature muting.

    feature_scanning : bool, default=True
        Whether to perform feature scanning.

    max_features : {"sqrt", "log2"}, int, or float, default=None
        Maximum number of features to use for feature selection.

    threshold_method : {"exact", "random", "histogram", "percentile"}, default="exact"
        Method to calculate thresholds on a feature used during split selection.

    threshold_scanning : bool, default=True
        Whether to perform threshold scanning.

    max_thresholds : {"sqrt", "log2"}, int, or float, default=None
        Maximum number of thresholds to use for split selection.

    max_depth : int, default=None
        Maximum depth to grow tree.

    min_samples_split : int, default=2
        Minimim samples required for a valid binary split.

    min_samples_leaf : int, default=1
        Minimum number of samples in a leaf node.

    min_impurity_decrease : float, default=0.0
        Minimum impurity decrease required for a valid binary split.

    random_state : int, default=None
        Random seed.

    verbose : int, default=1
        Controls verbosity when fitting and predicting.

    check_for_unused_parameters : bool, default=False
        Check for unused hyperparameters. Useful for debugging.

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
        Underlying decision tree object.
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
        n_resamples_selector: Optional[Union[str, int]] = "auto",
        n_resamples_splitter: Optional[Union[str, int]] = "auto",
        early_stopping_selector: bool = True,
        early_stopping_splitter: bool = True,
        feature_muting: bool = True,
        feature_scanning: bool = True,
        max_features: Optional[Union[str, float, int]] = None,
        threshold_method: str = "exact",
        threshold_scanning: bool = True,
        max_thresholds: Optional[Union[str, float, int]] = None,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: Optional[int] = None,
        verbose: int = 1,
        check_for_unused_parameters: bool = False,
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
            max_features=max_features,
            threshold_method=threshold_method,
            threshold_scanning=threshold_scanning,
            max_thresholds=max_thresholds,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            verbose=verbose,
            check_for_unused_parameters=check_for_unused_parameters,
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
