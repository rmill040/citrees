from typing import Literal

import numpy as np
from pydantic import validator
from pydantic.fields import ModelField
from pydantic.main import ModelMetaclass
from sklearn.base import BaseEstimator, ClassifierMixin

from ._base import BaseConditionalInferenceTree, BaseConditionalInferenceTreeParameters
from ._selector import ClassifierSelectors, ClassifierSelectorTests
from ._splitter import ClassifierSplitters, ClassifierSplitterTests


class ConditionalInferenceTreeClassifierParameters(BaseConditionalInferenceTreeParameters):
    """Model for BaseConditionalInferenceTree parameters.

    Parameters
    ----------
    selector : {"mc", "mi", "hybrid"}, optional (default="mc")
        Method for feature selection.

    splitter : {"gini", "entropy"}, optional (default="gini")
        Method for split selection.
    """

    selector: Literal["mc", "mi", "hybrid"] = "mc"
    splitter: Literal["gini", "entropy"] = "gini"

    @validator("selector")
    def validate_selector(cls: ModelMetaclass, v: str, field: ModelField) -> str:
        """Validate selector."""
        setattr(cls, f"_{field.name}", ClassifierSelectors[v])
        setattr(cls, f"_{field.name}_test", ClassifierSelectorTests[v])
        return v

    @validator("splitter")
    def validate_splitter(cls, v: str, field: ModelField) -> str:
        """Validate splitter."""
        setattr(cls, f"_{field.name}", ClassifierSplitters[v])
        setattr(cls, f"_{field.name}_test", ClassifierSplitterTests[v])
        return v


class ConditionalInferenceTreeClassifier(BaseConditionalInferenceTree, BaseEstimator, ClassifierMixin):
    """Conditional inference tree classifier.

    Parameters
    ----------
    selector : {"mc", "mi", "hybrid"}, optional (default="mc")
        Method for feature selection.

    splitter : {"gini", "entropy"}, optional (default="gini")
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
        selector="mc",
        splitter="gini",
        alpha_selector=0.05,
        alpha_splitter=0.05,
        adjust_alpha_selector=True,
        adjust_alpha_splitter=True,
        threshold_method="exact",
        max_thresholds=None,
        early_stopping_selector=True,
        early_stopping_splitter=True,
        feature_muting=True,
        n_resamples_selector="auto",
        n_resamples_splitter="auto",
        max_depth=None,
        max_features=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        n_jobs=1,
        random_state=None,
        verbose=0,
    ) -> None:
        super().__init__(
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

    @property
    def _validator(self) -> ModelMetaclass:
        """Validation model for estimator's hyperparameters."""
        return ConditionalInferenceTreeClassifierParameters

    def _node_value(self, y: np.ndarray) -> float:
        """Calculate value in terminal node.

        Parameters
        ----------
        y : np.ndarray
            Labels for node.

        Returns
        -------
        float
            Node value estimate.
        """
        return 0.0
