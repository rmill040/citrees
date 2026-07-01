import warnings
from abc import ABCMeta, abstractmethod
from math import ceil
from typing import Any, TypedDict

import numpy as np
from pydantic import BaseModel, field_validator, model_validator
from scipy.sparse import csr_matrix
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from citrees._selector import (
    ClassifierSelectors,
    ClassifierSelectorTests,
    RegressorSelectors,
    RegressorSelectorTests,
    _ptest_multi,
)
from citrees._splitter import (
    ClassifierSplitters,
    ClassifierSplitterTests,
    RegressorSplitters,
    RegressorSplitterTests,
)
from citrees._threshold_method import ThresholdMethods
from citrees._types import (
    ConfidenceFloat,
    EarlyStoppingOption,
    EstimatorType,
    HonestyFraction,
    MaxValuesOption,
    MinSamplesSplit,
    NonNegativeFloat,
    NonNegativeInt,
    NResamples,
    NResamplesOption,
    PositiveInt,
    ProbabilityFloat,
    ThresholdMethod,
)
from citrees._utils import calculate_max_value, estimate_mean, estimate_proba, split_data


class Node(TypedDict, total=False):
    feature: int
    pval_feature: float
    threshold: float
    pval_threshold: float
    impurity: float
    value: np.ndarray | float
    left_child: "Node"
    right_child: "Node"
    n_samples: int
    node_id: int


class BaseConditionalInferenceTreeParameters(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    estimator_type: EstimatorType
    selector: str | list[str]
    splitter: str
    alpha_selector: ProbabilityFloat
    alpha_splitter: ProbabilityFloat
    adjust_alpha_selector: bool
    adjust_alpha_splitter: bool
    n_resamples_selector: NResamplesOption
    n_resamples_splitter: NResamplesOption
    early_stopping_selector: EarlyStoppingOption
    early_stopping_splitter: EarlyStoppingOption
    early_stopping_confidence_selector: ConfidenceFloat
    early_stopping_confidence_splitter: ConfidenceFloat
    feature_muting: bool
    feature_scanning: bool
    threshold_scanning: bool
    threshold_method: ThresholdMethod
    max_thresholds: MaxValuesOption
    max_features: MaxValuesOption
    max_depth: PositiveInt | None = None
    min_samples_split: MinSamplesSplit
    min_samples_leaf: PositiveInt
    min_impurity_decrease: NonNegativeFloat
    honesty: bool
    honesty_fraction: HonestyFraction
    random_state: NonNegativeInt | None = None
    verbose: NonNegativeInt
    check_for_unused_parameters: bool

    @field_validator("max_features", "max_thresholds", mode="before")
    @classmethod
    def _reject_bool(cls, v: Any) -> Any:
        """Reject boolean values before they get coerced to int."""
        if isinstance(v, bool):
            raise ValueError(f"Cannot be boolean, got {v!r}")
        return v

    @model_validator(mode="after")
    def validate_selector_splitter(self) -> "BaseConditionalInferenceTreeParameters":
        if self.estimator_type == EstimatorType.CLASSIFIER:
            sel_registry = ClassifierSelectors
            spl_registry = ClassifierSplitters
        else:
            sel_registry = RegressorSelectors
            spl_registry = RegressorSplitters

        # Handle selector validation - can be string or list of strings
        selectors = [self.selector] if isinstance(self.selector, str) else self.selector

        if len(selectors) == 0:
            raise ValueError("selector list cannot be empty")

        for sel in selectors:
            if sel not in sel_registry:
                raise ValueError(f"selector '{sel}' not in {sel_registry.keys()}")

        # Check for duplicate selectors
        if len(selectors) != len(set(selectors)):
            raise ValueError(
                f"selector list contains duplicates: {selectors}. Each selector must be unique."
            )

        # Validate that mi is not in a list for classification (mi is not on [0,1] scale)
        if (
            self.estimator_type == EstimatorType.CLASSIFIER
            and len(selectors) > 1
            and "mi" in selectors
        ):
            raise ValueError(
                "selector 'mi' cannot be used in a list with other selectors because mutual information "
                "is not on the same [0,1] scale as 'mc' and 'rdc'. Use selector='mi' alone instead."
            )

        if self.splitter not in spl_registry:
            raise ValueError(f"splitter '{self.splitter}' not in {spl_registry.keys()}")

        for attr in ["n_resamples_selector", "n_resamples_splitter"]:
            v = getattr(self, attr)
            if isinstance(v, int):
                alpha_attr = "alpha_selector" if "selector" in attr else "alpha_splitter"
                alpha = getattr(self, alpha_attr)
                min_samples = ceil(1 / alpha)
                if v < min_samples:
                    raise ValueError(f"{attr} ({v}) should be >= {min_samples}")

        return self

    @model_validator(mode="after")
    def validate_max_values(self) -> "BaseConditionalInferenceTreeParameters":
        for param in ["max_features", "max_thresholds"]:
            value = getattr(self, param)
            if isinstance(value, bool):
                raise ValueError(f"{param} cannot be a bool, got {value!r}")
            if isinstance(value, int) and value <= 0:
                raise ValueError(f"{param} must be >= 1 when int, got {value}")
            if isinstance(value, float) and (not np.isfinite(value) or value <= 0.0 or value > 1.0):
                raise ValueError(f"{param} must be in (0, 1] when float, got {value}")
        return self


class BaseConditionalInferenceTreeEstimator(BaseEstimator, metaclass=ABCMeta):
    @property
    @abstractmethod
    def _parameter_model(self) -> type[BaseModel]:
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
            if isinstance(value, str):
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

    def _validate_data_fit(
        self, *, X: Any, y: Any, estimator_type: str
    ) -> tuple[np.ndarray, np.ndarray]:
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
            if isinstance(X, list | tuple):
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
        self.feature_names_in_ = np.array(feature_names_in, dtype=object)

        if not isinstance(y, np.ndarray):
            if isinstance(y, list | tuple):
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
            raise ValueError(
                f"Multi-output labels are not supported for y, detected ({y.ndim - 1}) outputs"
            )

        if len(X) != len(y):
            raise ValueError(f"Different number of samples between X ({len(X)}) and y ({len(y)})")

        X = X.astype(float)
        # Keep labels as-is for classifiers; LabelEncoder handles encoding during fit.
        y = y if estimator_type == EstimatorType.CLASSIFIER else y.astype(float)

        # Reject NaN/Inf values - required for fastmath optimizations in split_data
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError(
                "Input X contains NaN or Inf values. "
                "Please handle missing/infinite values before fitting."
            )
        # Check y only for numeric dtypes (string labels are valid for classifiers)
        if np.issubdtype(y.dtype, np.number) and (np.any(np.isnan(y)) or np.any(np.isinf(y))):
            raise ValueError(
                "Input y contains NaN or Inf values. "
                "Please handle missing/infinite values before fitting."
            )

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
            if isinstance(X, list | tuple):
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
            raise ValueError(
                f"X should have ({len(self.feature_names_in_)}) features, got ({X.shape[1]})"
            )

        if feature_names and not np.array_equal(feature_names, self.feature_names_in_):
            missing = [name for name in self.feature_names_in_ if name not in feature_names]
            extra = [name for name in feature_names if name not in self.feature_names_in_]
            if missing or extra:
                raise ValueError(
                    "Mismatch in feature names for X, missing "
                    f"({len(missing)}) features: {missing}; extra ({len(extra)}) features: {extra}"
                )
            raise ValueError(
                "Feature names are out of order for X, expected "
                f"{self.feature_names_in_} but got {feature_names}"
            )

        X = X.astype(float)
        return X

    def _validate_parameters(self, params: dict[str, Any]) -> None:
        """Validate hyperparameters.

        Parameters
        ----------
        params : dict[str, Any]
            Hyperparameters.

        """
        self._parameter_model(**params)

    def _validate_parameter_combinations(self) -> None:
        """Runtime check to determine if any hyperparameters are unused due to constraints.

        Selector constraints:
        1. n_resamples_selector is None =>
            - adjust_alpha_selector = False
            - feature_muting = False
            - early_stopping_selector = None

        2. early_stopping_selector is None =>
            - feature_scanning == False

        Splitter constraints:
        1. n_resamples_splitter is None =>
            - adjust_alpha_splitter = False
            - early_stopping_splitter = None

        2. early_stopping_splitter is None =>
            - threshold_scanning = False
        """
        params = self.get_params()

        flags = []
        if params["n_resamples_selector"] is None:
            flags = [
                key
                for key in ["adjust_alpha_selector", "feature_muting", "early_stopping_selector"]
                if params[key]
            ]
        if flags:
            warnings.warn(
                "Unused hyperparameter(s) detected: When n_resamples_selector=None, hyperparameter(s) "
                f"({', '.join(flags)}) should be False",
                stacklevel=2,
            )

        if params["early_stopping_selector"] is None and params["feature_scanning"]:
            warnings.warn(
                "Unused hyperparameter detected: When early_stopping_selector=None, hyperparameter "
                "('feature_scanning') should be False",
                stacklevel=2,
            )

        flags = []
        if params["n_resamples_splitter"] is None:
            flags = [
                key for key in ["adjust_alpha_splitter", "early_stopping_splitter"] if params[key]
            ]
        if flags:
            warnings.warn(
                "Unused hyperparameter(s) detected: When n_resamples_splitter=None, hyperparameter(s) "
                f"({', '.join(flags)}) should be False",
                stacklevel=2,
            )

        if params["early_stopping_splitter"] is None and params["threshold_scanning"]:
            warnings.warn(
                "Unused hyperparameters detected: When early_stopping_splitter=None, hyperparameter "
                "('threshold_scanning') should be False",
                stacklevel=2,
            )


class BaseConditionalInferenceTree(BaseConditionalInferenceTreeEstimator, metaclass=ABCMeta):
    """Base class for conditional inference trees.

    Warning: This class should not be used directly. Use derived classes instead.
    """

    @abstractmethod
    def __init__(
        self,
        *,
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
        random_state: int | None,
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
        self.random_state = random_state
        self.verbose = verbose
        self.check_for_unused_parameters = check_for_unused_parameters

        self._validate_parameters({**self.get_params(), "estimator_type": self._estimator_type})
        if self.check_for_unused_parameters:
            self._validate_parameter_combinations()

    @abstractmethod
    def _node_value(self, y: np.ndarray) -> float | np.ndarray:
        """Calculate value in terminal node."""
        pass

    @property
    def _parameter_model(self) -> type[BaseModel]:
        """Model for hyperparameter validation.

        Returns
        -------
        type[BaseModel]
            Model to validate hyperparameters.

        """
        return BaseConditionalInferenceTreeParameters

    def _compute_selector_score(self, x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
        """Compute selector score, using max across all selectors in multi-selector mode.

        Parameters
        ----------
        x : np.ndarray
            Feature values.

        y : np.ndarray
            Target values.

        Returns
        -------
        best_score : float
            Best selector score (max across selectors in multi-selector mode).

        best_selector_name : str
            Name of the selector that produced the best score.

        """
        if self._multi_selector:
            best_score = -np.inf
            best_name = self._selector_names[0]
            for name in self._selector_names:
                score = self._selectors[name](x=x, y=y, **self._selector_kwargs)
                # For pc (Pearson correlation), take absolute value to handle negative correlations
                if name == "pc":
                    score = abs(score)
                if score > best_score:
                    best_score = score
                    best_name = name
            return best_score, best_name
        else:
            score = self._selector(x=x, y=y, **self._selector_kwargs)
            # For pc (Pearson correlation), take absolute value
            if self._selector_names[0] == "pc":
                score = abs(score)
            return score, self._selector_names[0]

    def _select_best_feature(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        features: np.ndarray,
        available_features: np.ndarray,
    ) -> tuple[int, float, bool, np.ndarray]:
        """Select best feature associated with the target.

        Parameters
        ----------
        X : np.ndarray
            Training features.

        y : np.ndarray
            Training target.

        features : np.ndarray
            Feature indices.

        available_features : np.ndarray
            Feature indices available for descendant nodes. When feature muting is enabled, this set can shrink during
            feature selection, but the updated set must only be propagated to descendants of the current node.

        Returns
        -------
        best_feature : int
            Index of best feature.

        best_pval : float
            Pvalue of best feature selection test.

        reject_H0 : bool
            Whether to reject the null hypothesis of no significant association between features and target.

        updated_available_features : np.ndarray
            Available feature indices after muting decisions at this node.

        """
        best_feature = features[0]
        best_pval = np.inf
        # When permutation testing is disabled (n_resamples=None), always accept features.
        # This provides CART-like behavior where splits are based purely on the selector
        # metric without statistical significance testing. Users who disable permutation
        # testing should rely on other regularization (max_depth, min_samples_leaf, etc.)
        # to control overfitting.
        reject_H0 = self._n_resamples_selector is None
        best_metric = -np.inf
        # Counters for reservoir sampling tie-breaking (uniform selection among ties)
        n_best_pval = 0
        n_best_metric = 0

        # Apply Bonferroni correction for multiple feature tests (matches R's partykit::ctree behavior)
        if self._adjust_alpha_selector and self._n_resamples_selector is not None:
            self._bonferroni_correction(adjust="selector", n_tests=len(features))
            # Update kwargs to reflect Bonferroni-corrected values (preserve type-specific params)
            self._selector_test_kwargs.update(
                {
                    "n_resamples": self._n_resamples_selector,
                    "early_stopping": self._early_stopping_selector,
                    "alpha": self._alpha_selector,
                    "random_state": self._random_state,
                    "confidence": self._early_stopping_confidence_selector,
                }
            )

        for feature in features:
            x = X[:, feature]

            # Feature selection with permutation testing
            if self._n_resamples_selector:
                if self._multi_selector:
                    # Max-T permutation test: compute max(selector_scores) INSIDE each
                    # permutation (fixed-B) to provide valid max-T p-values
                    type_arg = self._selector_kwargs.get(
                        "n_classes", self._selector_kwargs.get("standardize")
                    )
                    pval_feature = _ptest_multi(
                        funcs=[self._selectors[name] for name in self._selector_names],
                        func_args=[type_arg] * len(self._selector_names),
                        x=x,
                        y=y,
                        n_resamples=self._selector_test_kwargs["n_resamples"],
                        early_stopping=self._selector_test_kwargs["early_stopping"],
                        alpha=self._selector_test_kwargs["alpha"],
                        random_state=self._selector_test_kwargs["random_state"],
                        confidence=self._selector_test_kwargs["confidence"],
                    )
                else:
                    pval_feature = self._selector_test(x=x, y=y, **self._selector_test_kwargs)

                # Update best feature with reservoir sampling for ties
                if pval_feature < best_pval:
                    best_feature = int(feature)
                    best_pval = pval_feature
                    n_best_pval = 1
                    reject_H0 = best_pval < self._alpha_selector

                    # Check for early stopping
                    if self._early_stopping_selector is not None and reject_H0:
                        break
                elif pval_feature == best_pval:
                    # Reservoir sampling: probability 1/k for k-th tie
                    n_best_pval += 1
                    if self._rng.random_sample() < 1.0 / n_best_pval:
                        best_feature = int(feature)

                # Check for feature muting: mute features that are not statistically significant
                # (i.e., p-value >= alpha means we fail to reject H0 of no association)
                if (
                    self._feature_muting
                    and pval_feature >= self._alpha_selector
                    and len(available_features) > 1
                ):
                    available_features = self._mute_feature(
                        available_features=available_features,
                        feature=feature,
                        reason=f"FEATURE PVAL ({pval_feature}) >= ALPHA ({self._alpha_selector})",
                    )

            # Feature selection without permutation testing
            else:
                metric, _ = self._compute_selector_score(x, y)
                # Update best feature with reservoir sampling for ties
                if metric > best_metric:
                    best_feature = int(feature)
                    best_metric = metric
                    n_best_metric = 1
                elif metric == best_metric:
                    # Reservoir sampling: probability 1/k for k-th tie
                    n_best_metric += 1
                    if self._rng.random_sample() < 1.0 / n_best_metric:
                        best_feature = int(feature)

        return best_feature, best_pval, reject_H0, available_features

    def _select_best_split(
        self, x: np.ndarray, y: np.ndarray, thresholds: np.ndarray
    ) -> tuple[float, float, bool]:
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
        # When permutation testing is disabled (n_resamples=None), always accept splits.
        # This provides CART-like behavior where splits are based purely on the splitter
        # metric without statistical significance testing. Users who disable permutation
        # testing should rely on other regularization (max_depth, min_samples_leaf, etc.)
        # to control overfitting.
        reject_H0 = self._n_resamples_splitter is None
        best_metric = np.inf
        # Counters for reservoir sampling tie-breaking (uniform selection among ties)
        n_best_pval = 0
        n_best_metric = 0

        if self._adjust_alpha_splitter and self._n_resamples_splitter is not None:
            self._bonferroni_correction(adjust="splitter", n_tests=len(thresholds))
            # Update kwargs to reflect Bonferroni-corrected values
            self._splitter_test_kwargs = {
                "n_resamples": self._n_resamples_splitter,
                "early_stopping": self._early_stopping_splitter,
                "alpha": self._alpha_splitter,
                "random_state": self._random_state,
                "confidence": self._early_stopping_confidence_splitter,
            }

        for threshold in thresholds:
            # Split selection with permutation testing
            if self._n_resamples_splitter:
                pval_threshold = self._splitter_test(
                    x=x, y=y, threshold=threshold, **self._splitter_test_kwargs
                )

                # Update best threshold with reservoir sampling for ties
                if pval_threshold < best_pval:
                    best_threshold = float(threshold)
                    best_pval = pval_threshold
                    n_best_pval = 1
                    reject_H0 = best_pval < self._alpha_splitter

                    # Check for early stopping
                    if self._early_stopping_splitter is not None and reject_H0:
                        break
                elif pval_threshold == best_pval:
                    # Reservoir sampling: probability 1/k for k-th tie
                    n_best_pval += 1
                    if self._rng.random_sample() < 1.0 / n_best_pval:
                        best_threshold = float(threshold)

            # Split selection without permutation testing
            else:
                metric = self._split_impurity(x=x, y=y, threshold=threshold)
                # Update best threshold with reservoir sampling for ties
                if metric < best_metric:
                    best_threshold = float(threshold)
                    best_metric = metric
                    n_best_metric = 1
                elif metric == best_metric:
                    # Reservoir sampling: probability 1/k for k-th tie
                    n_best_metric += 1
                    if self._rng.random_sample() < 1.0 / n_best_metric:
                        best_threshold = float(threshold)

        return best_threshold, best_pval, reject_H0

    def _bonferroni_correction(self, *, adjust: str, n_tests: int) -> None:
        """Implement Bonferroni correction to account for multiple hypothesis tests.

        During training, when Bonferroni correction is enabled, alpha will be adjusted based on the number of hypothesis
        tests, assuming permutation tests are used. Likewise, the number of resamples for a permutation test will be
        updated to enough resamples are run to compare the achieved significance level against alpha.

        Parameters
        ----------
        adjust : {"splitter", "selector"}
            Which attributes to use for Bonferroni correction.

        n_tests : int
            Number of hypothesis tests to perform.

        """
        alpha = getattr(self, f"alpha_{adjust}")
        n_resamples = getattr(self, f"n_resamples_{adjust}")

        # Bonferroni is a per-node adjustment. Even when n_tests=1 (no adjustment), we must
        # reset the private per-node attributes to the unadjusted values to avoid leaking
        # state across nodes.
        n_tests = max(int(n_tests), 1)
        _alpha = alpha / n_tests

        if n_resamples is None:
            _n_resamples = None
        elif isinstance(n_resamples, str):
            lower_limit = ceil(1 / _alpha)
            if n_resamples == NResamples.MINIMUM:
                _n_resamples = lower_limit
            elif n_resamples == NResamples.MAXIMUM:
                _n_resamples = ceil(1 / (4 * _alpha * _alpha))
            else:
                z = norm.ppf(1 - _alpha)
                upper_limit = ceil(z * z * (1 - _alpha) / _alpha)
                _n_resamples = max(lower_limit, upper_limit)
        else:
            _n_resamples = n_resamples * n_tests

        setattr(self, f"_alpha_{adjust}", _alpha)
        setattr(self, f"_n_resamples_{adjust}", _n_resamples)

    def _mute_feature(
        self,
        *,
        available_features: np.ndarray,
        feature: int,
        reason: str,
    ) -> np.ndarray:
        """Remove a feature from the candidate set that will be propagated to descendants.

        Important: this must be traversal-order invariant. In particular, we must not mutate any shared/global
        candidate set during recursion, otherwise siblings can affect each other's available feature sets.
        """
        idx = available_features == feature
        updated = available_features[~idx]

        if self.verbose > 2:
            p = len(updated)
            print(
                f"Muted feature ({self.feature_names_in_[feature]}) because ({reason}), ({p}) features available"
            )

        return updated

    def _scan_features(self, X: np.ndarray, y: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Perform feature scanning.

        Return the feature indices that are most associated with the target.

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
        scores = np.array(
            [self._compute_selector_score(X[:, feature], y)[0] for feature in features]
        )
        ranks = np.argsort(scores)[::-1]

        return features[ranks]

    def _scan_thresholds(self, x: np.ndarray, y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """Perform threshold scanning to return thresholds ordered by weighted split impurity.

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
            Thresholds sorted in ascending order based on weighted child impurity from the binary split.

        """
        scores = np.array(
            [self._split_impurity(x=x, y=y, threshold=threshold) for threshold in thresholds]
        )
        ranks = np.argsort(scores)

        return thresholds[ranks]

    def _split_impurity(self, *, x: np.ndarray, y: np.ndarray, threshold: float) -> float:
        """Perform binary split.

        Calculate split impurity as the weighted sum of child-node impurities.

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
        n_left = idx.sum()
        n_right = n - n_left

        return (n_left / n) * self._splitter(y[idx]) + (n_right / n) * self._splitter(y[~idx])

    def _node_impurity(
        self, *, y: np.ndarray, idx: np.ndarray, n: int, n_left: int, n_right: int
    ) -> tuple[float, float]:
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
        children_impurity = (n_left / n) * self._splitter(y[idx]) + (n_right / n) * self._splitter(
            y[~idx]
        )
        impurity_decrease = parent_impurity - children_impurity

        return parent_impurity, impurity_decrease

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        depth: int,
        available_features: np.ndarray,
    ) -> Node:
        """Recursively build tree.

        Parameters
        ----------
        X : np.ndarray
            Training features.

        y : np.ndarray
            Training target.

        depth : int
            Depth of tree.

        available_features : np.ndarray
            Feature indices available to this node. This set must be treated as immutable; any muting decisions at this
            node should be applied to a local copy and the updated set propagated to descendants.

        Returns
        -------
        Node
            Node in decision tree.

        """
        reject_H0_feature = False
        reject_H0_threshold = False
        impurity_decrease = -1.0
        n, p = X.shape
        local_available = available_features

        # Keep track of maximum tree depth
        self.depth_ = max(getattr(self, "depth_", 0), depth)

        # Check for stopping criteria at node level
        if n >= self._min_samples_split and depth <= self._max_depth and not np.all(y == y[0]):
            if self.verbose > 2:
                print(f"Building tree at depth ({depth}) with ({n}) samples")

            # Check for constant features and remove them from the local candidate set. This is safe to propagate to
            # descendants because a feature that is constant on a node's samples cannot become non-constant deeper in
            # that subtree. Crucially, this must not affect siblings.
            for feature in available_features:
                x = X[:, feature]
                if np.all(x == x[0]):
                    local_available = self._mute_feature(
                        available_features=local_available,
                        feature=int(feature),
                        reason="FEATURE IS CONSTANT",
                    )

            # Feature selection

            # If early stopping, we scan features and sort based on most promising features but no need to also
            # randomly permute since a random sample is already taken from the feature set
            if len(local_available):
                max_features = (
                    calculate_max_value(
                        n_values=len(local_available), desired_max=self.max_features
                    )
                    if self.max_features
                    else len(local_available)
                )
                features = (
                    self._rng.choice(local_available, size=max_features, replace=False)
                    if len(local_available) > 1
                    else local_available
                )
                if (
                    self._early_stopping_selector is not None
                    and self._feature_scanning
                    and len(features) > 1
                ):
                    features = self._scan_features(X, y, features)
                best_feature, best_pval_feature, reject_H0_feature, local_available = (
                    self._select_best_feature(
                        X,
                        y,
                        features=features,
                        available_features=local_available,
                    )
                )

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
                threshold_seed = None
                if self.threshold_method == ThresholdMethod.RANDOM:
                    threshold_seed = int(self._rng.randint(1, 1_000_000))
                thresholds = self._threshold_method(
                    x,
                    max_thresholds=self._max_thresholds,
                    random_state=threshold_seed,
                )
            else:
                # With smaller samples, just use all midpoints as potential split points
                thresholds = (x_unique[:-1] + x_unique[1:]) / 2
                self._max_thresholds = len(thresholds)

            # Filter out thresholds that would violate min_samples_leaf. This prevents selecting an
            # "optimal" split that is invalid and then failing to consider other valid thresholds.
            if self._min_samples_leaf > 1 and len(thresholds):
                x_sorted = np.sort(x)
                n_left = np.searchsorted(x_sorted, thresholds, side="right")
                valid = (n_left >= self._min_samples_leaf) & (n - n_left >= self._min_samples_leaf)
                thresholds = thresholds[valid]
                self._max_thresholds = len(thresholds)

            if len(thresholds):
                if self._early_stopping_splitter:
                    thresholds = (
                        self._scan_thresholds(x, y, thresholds)
                        if self._threshold_scanning and len(thresholds) > 1
                        else self._rng.permutation(thresholds)
                    )
                best_threshold, best_pval_threshold, reject_H0_threshold = self._select_best_split(
                    x, y, thresholds
                )
            else:
                reject_H0_threshold = False

        # Calculate impurity decrease
        if reject_H0_threshold:
            idx = x <= best_threshold
            n_left = int(idx.sum())
            n_right = int(n - n_left)
            if n_left >= self._min_samples_leaf and n_right >= self._min_samples_leaf:
                impurity, impurity_decrease = self._node_impurity(
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
            left_child = self._build_tree(
                X=X_left, y=y_left, depth=depth + 1, available_features=local_available
            )
            right_child = self._build_tree(
                X=X_right, y=y_right, depth=depth + 1, available_features=local_available
            )

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
        self._max_depth = self.max_depth if self.max_depth else np.inf
        self._random_state = (
            int(np.random.randint(1, 1_000_000)) if self.random_state is None else self.random_state
        )
        # RNG used across nodes to avoid reseeding at each recursion level.
        self._rng = np.random.RandomState(self._random_state)
        self._verbose = min(self.verbose, 3)
        for param in ["n_resamples_selector", "n_resamples_splitter"]:
            value = getattr(self, param)
            if isinstance(value, str):
                alpha = self.alpha_selector if "selector" in param else self.alpha_splitter
                lower_limit = ceil(1 / alpha)
                if value == NResamples.MINIMUM:
                    value = lower_limit
                elif value == NResamples.MAXIMUM:
                    value = ceil(1 / (4 * alpha * alpha))
                else:
                    # Approximate upper limit
                    z = norm.ppf(1 - alpha)
                    upper_limit = ceil(z * z * (1 - alpha) / alpha)
                    value = max(lower_limit, upper_limit)
            setattr(self, f"_{param}", value)

        # No need to adjust alphas yet if flags are enabled. The alpha adjustments happen during each call to
        # self._selector_test and self._splitter_test
        for param in [
            "alpha_selector",
            "alpha_splitter",
            "early_stopping_selector",
            "early_stopping_splitter",
            "early_stopping_confidence_selector",
            "early_stopping_confidence_splitter",
        ]:
            setattr(self, f"_{param}", getattr(self, param))

        common_selector_kwargs = {
            "random_state": self._random_state,
        }
        common_selector_test_kwargs = {
            "n_resamples": self._n_resamples_selector,
            "early_stopping": self._early_stopping_selector,
            "alpha": self._alpha_selector,
            "random_state": self._random_state,
            "confidence": self._early_stopping_confidence_selector,
        }

        # Handle list-based selectors (multi-selector mode)
        selectors = [self.selector] if isinstance(self.selector, str) else self.selector
        self._multi_selector = len(selectors) > 1
        self._selector_names = selectors

        if self._estimator_type == EstimatorType.CLASSIFIER:
            self._selectors: dict[str, Any] = {
                name: ClassifierSelectors[name] for name in selectors
            }
            self._selector_tests: dict[str, Any] = {
                name: ClassifierSelectorTests[name] for name in selectors
            }
            self._splitter: Any = ClassifierSplitters[self.splitter]
            self._splitter_test: Any = ClassifierSplitterTests[self.splitter]

            n_classes = len(np.unique(y))
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
            self._selectors = {name: RegressorSelectors[name] for name in selectors}
            self._selector_tests = {name: RegressorSelectorTests[name] for name in selectors}
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

        # For backwards compatibility, set _selector and _selector_test to first selector
        # These are used by _scan_features
        self._selector = self._selectors[selectors[0]]
        self._selector_test = self._selector_tests[selectors[0]]

        self._splitter_test_kwargs = {
            "n_resamples": self._n_resamples_splitter,
            "early_stopping": self._early_stopping_splitter,
            "alpha": self._alpha_splitter,
            "random_state": self._random_state,
            "confidence": self._early_stopping_confidence_splitter,
        }

        self._threshold_method: Any = ThresholdMethods[self.threshold_method]
        if self.threshold_method != ThresholdMethod.EXACT and self.max_thresholds is None:
            warnings.warn(
                f"Using threshold_method='{self.threshold_method}' with max_thresholds=None is not recommended, "
                "consider reducing max_thresholds to speed up split selection.",
                stacklevel=2,
            )
        # Set rest of parameters as private attributes
        for param, value in self.get_params().items():
            p_param = f"_{param}"
            if not hasattr(self, p_param):
                setattr(self, p_param, value)

        # Fitted attributes
        if self._estimator_type == EstimatorType.CLASSIFIER:
            self.classes_ = self._label_encoder.classes_
            self.n_classes_ = n_classes

        self.feature_importances_ = np.zeros(p, dtype=float)
        self.n_features_in_ = p

        # Honest estimation: split data into splitting and estimation samples
        if self.honesty:
            X_split, X_est, y_split, y_est = train_test_split(
                X,
                y,
                test_size=self.honesty_fraction,
                random_state=self._random_state,
            )
            # Build tree structure using splitting sample
            self.tree_ = self._build_tree(
                X_split, y_split, depth=1, available_features=np.arange(p, dtype=int)
            )
            # Re-estimate leaf values using estimation sample
            self._reestimate_leaf_values(X_est, y_est)
        else:
            self.tree_ = self._build_tree(X, y, depth=1, available_features=np.arange(p, dtype=int))

        # Normalize feature importances
        fi_sum = self.feature_importances_.sum()
        if fi_sum:
            self.feature_importances_ /= fi_sum

        # Assign node IDs for apply/decision_path
        self._assign_node_ids()

        return self

    def _get_leaf_path(self, x: np.ndarray, tree: Node | None = None, path: tuple = ()) -> tuple:
        """Get the path to the leaf node for a sample.

        Parameters
        ----------
        x : np.ndarray
            Feature vector for single sample.

        tree : Node, default=None
            Fitted decision tree.

        path : tuple
            Current path (sequence of 'L' or 'R' directions).

        Returns
        -------
        tuple
            Path from root to leaf (e.g., ('L', 'R', 'L')).

        """
        if tree is None:
            tree = self.tree_

        if "value" in tree:
            return path

        feature_value = x[tree["feature"]]
        if feature_value <= tree["threshold"]:
            return self._get_leaf_path(x, tree["left_child"], path + ("L",))
        else:
            return self._get_leaf_path(x, tree["right_child"], path + ("R",))

    def _reestimate_leaf_values(self, X: np.ndarray, y: np.ndarray) -> None:
        """Re-estimate leaf values using estimation sample (honest estimation).

        Uses path-based leaf identification for robustness to serialization.

        Parameters
        ----------
        X : np.ndarray
            Estimation sample features.

        y : np.ndarray
            Estimation sample targets.

        """
        # Group samples by leaf path (serialization-safe)
        leaf_samples: dict[tuple, list[int]] = {}
        for i in range(len(X)):
            path = self._get_leaf_path(X[i])
            if path not in leaf_samples:
                leaf_samples[path] = []
            leaf_samples[path].append(i)

        # Re-estimate each leaf using its estimation samples
        self._reestimate_tree_by_path(self.tree_, y, leaf_samples, path=())

    def _reestimate_tree_by_path(
        self, tree: Node, y: np.ndarray, leaf_samples: dict[tuple, list[int]], path: tuple
    ) -> None:
        """Re-estimate leaf values in tree using path-based identification (iterative).

        Parameters
        ----------
        tree : Node
            Tree node to process.

        y : np.ndarray
            Estimation sample targets.

        leaf_samples : dict
            Mapping from path to sample indices.

        path : tuple
            Current path from root.

        """
        # Use stack-based iteration to avoid recursion limit issues
        stack = [(tree, path)]

        while stack:
            node, current_path = stack.pop()

            if "value" in node:
                # This is a leaf node
                if current_path in leaf_samples and len(leaf_samples[current_path]) > 0:
                    indices = leaf_samples[current_path]
                    y_leaf = y[indices]
                    node["value"] = self._node_value(y_leaf)
                    node["n_samples"] = len(indices)
                # If no estimation samples fall into this leaf, keep the original value
            else:
                # Add children to stack (right first so left is processed first)
                stack.append((node["right_child"], current_path + ("R",)))
                stack.append((node["left_child"], current_path + ("L",)))

    def _predict_value(self, x: np.ndarray, tree: Node | None = None) -> float | np.ndarray:
        """Predict target for single sample using iterative traversal.

        Parameters
        ----------
        x : np.ndarray
            Feature.

        tree : Node, default=None
            Fitted decision tree.

        Returns
        -------
        float | np.ndarray
            Predicted value for sample.

        """
        if tree is None:
            tree = self.tree_

        # Iterative traversal (avoids recursion limit issues)
        while "value" not in tree:
            feature_value = x[tree["feature"]]
            tree = tree["left_child"] if feature_value <= tree["threshold"] else tree["right_child"]

        return tree["value"]

    def _assign_node_ids(self) -> None:
        """Assign deterministic node IDs to all nodes in the tree.

        Node IDs are assigned in pre-order to match sklearn-style apply/decision_path behavior.
        """
        if "node_id" in self.tree_ and hasattr(self, "_n_nodes"):
            return

        stack = [self.tree_]
        node_id = 0
        while stack:
            node = stack.pop()
            node["node_id"] = node_id
            node_id += 1
            if "value" not in node:
                # Right first so left is processed first (pre-order)
                stack.append(node["right_child"])
                stack.append(node["left_child"])

        self._n_nodes = node_id

    def apply(self, X: Any) -> np.ndarray:
        """Return leaf indices for each sample (sklearn-compatible)."""
        X = self._validate_data_predict(X)
        self._assign_node_ids()

        leaf_ids = np.empty(X.shape[0], dtype=int)
        for i, x in enumerate(X):
            node = self.tree_
            while "value" not in node:
                node = (
                    node["left_child"]
                    if x[node["feature"]] <= node["threshold"]
                    else node["right_child"]
                )
            leaf_ids[i] = node["node_id"]

        return leaf_ids

    def decision_path(self, X: Any) -> csr_matrix:
        """Return sparse matrix indicating the decision path for each sample."""
        X = self._validate_data_predict(X)
        self._assign_node_ids()

        indices: list[int] = []
        indptr = [0]

        for x in X:
            node = self.tree_
            while True:
                indices.append(node["node_id"])
                if "value" in node:
                    break
                node = (
                    node["left_child"]
                    if x[node["feature"]] <= node["threshold"]
                    else node["right_child"]
                )
            indptr.append(len(indices))

        data = np.ones(len(indices), dtype=int)
        return csr_matrix((data, indices, indptr), shape=(X.shape[0], self._n_nodes))

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target."""
        pass


class ConditionalInferenceTreeClassifier(ClassifierMixin, BaseConditionalInferenceTree):
    """Conditional inference tree classifier.

    Uses permutation-based Stage A screening to reduce split-selection bias.
    Fixed-B p-value calibration is a fixed-node property; adaptive fitted-tree
    rankings and importances are empirical model outputs.

    Parameters
    ----------
    early_stopping_selector : {"adaptive", "simple"} or None, default="adaptive"
        Early stopping method for feature selection permutation tests:
        - "adaptive": Bayesian Beta CDF posterior-confidence stopping (speed-oriented; returns a +1 Monte Carlo estimate
          at a stopping time)
        - "simple": Futility + significance stopping (inflates Type I error)
        - None: No early stopping (fixed-B test)

    early_stopping_splitter : {"adaptive", "simple"} or None, default="adaptive"
        Early stopping method for split selection permutation tests.

    early_stopping_confidence_selector : float, default=0.95
        Confidence threshold for adaptive stopping in feature selection.

    early_stopping_confidence_splitter : float, default=0.95
        Confidence threshold for adaptive stopping in split selection.

    Notes
    -----
    P-values use the Phipson & Smyth (2010) +1 correction to ensure they are
    never exactly zero: p = (b+1)/(m+1) instead of p = b/m.

    References
    ----------
    Phipson & Smyth (2010). "Permutation P-values Should Never Be Zero."
    SAGMB 9(1):39. https://pubmed.ncbi.nlm.nih.gov/21044043/

    """

    _estimator_type = "classifier"

    def __init__(
        self,
        *,
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
        max_features: str | float | int | None = None,
        threshold_method: str = "exact",
        threshold_scanning: bool = True,
        max_thresholds: str | float | int | None = None,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        honesty: bool = False,
        honesty_fraction: float = 0.5,
        random_state: int | None = None,
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
            random_state=random_state,
            verbose=verbose,
            check_for_unused_parameters=check_for_unused_parameters,
        )

    def _node_value(self, y: np.ndarray) -> float | np.ndarray:
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
        labels = np.argmax(y_hat, axis=1)
        return self._label_encoder.inverse_transform(labels)


class ConditionalInferenceTreeRegressor(RegressorMixin, BaseConditionalInferenceTree):
    """Conditional inference tree regressor.

    Uses permutation-based Stage A screening to reduce split-selection bias.
    Fixed-B p-value calibration is a fixed-node property; adaptive fitted-tree
    rankings and importances are empirical model outputs.

    Parameters
    ----------
    early_stopping_selector : {"adaptive", "simple"} or None, default="adaptive"
        Early stopping method for feature selection permutation tests:
        - "adaptive": Bayesian Beta CDF posterior-confidence stopping (speed-oriented; returns a +1 Monte Carlo estimate
          at a stopping time)
        - "simple": Futility + significance stopping (inflates Type I error)
        - None: No early stopping (fixed-B test)

    early_stopping_splitter : {"adaptive", "simple"} or None, default="adaptive"
        Early stopping method for split selection permutation tests.

    early_stopping_confidence_selector : float, default=0.95
        Confidence threshold for adaptive stopping in feature selection.

    early_stopping_confidence_splitter : float, default=0.95
        Confidence threshold for adaptive stopping in split selection.

    Notes
    -----
    P-values use the Phipson & Smyth (2010) +1 correction to ensure they are
    never exactly zero: p = (b+1)/(m+1) instead of p = b/m.

    References
    ----------
    Phipson & Smyth (2010). "Permutation P-values Should Never Be Zero."
    SAGMB 9(1):39. https://pubmed.ncbi.nlm.nih.gov/21044043/

    """

    _estimator_type = "regressor"

    def __init__(
        self,
        *,
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
        max_features: str | float | int | None = None,
        threshold_method: str = "exact",
        threshold_scanning: bool = True,
        max_thresholds: str | float | int | None = None,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        honesty: bool = False,
        honesty_fraction: float = 0.5,
        random_state: int | None = None,
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
            random_state=random_state,
            verbose=verbose,
            check_for_unused_parameters=check_for_unused_parameters,
        )

    def _node_value(self, y: np.ndarray) -> float | np.ndarray:
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
