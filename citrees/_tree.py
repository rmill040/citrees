from typing import Literal, Tuple

import numpy as np
from pydantic.main import ModelMetaclass
from sklearn.base import BaseEstimator, ClassifierMixin

from ._base import BaseConditionalInferenceTree, BaseConditionalInferenceTreeParameters


class ConditionalInferenceTreeClassifierParameters(BaseConditionalInferenceTreeParameters):
    """Model for BaseConditionalInferenceTree parameters.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, optional (default="gini")
        Criterion for splitting.

    selector : {"mc", "mi", "hybrid"}, optional (default="mc")
        Method for feature selection.
    """

    criterion: Literal["gini", "entropy"] = "gini"
    selector: Literal["mc", "mi", "hybrid"] = "mc"


class ConditionalInferenceTreeClassifier(BaseConditionalInferenceTree, BaseEstimator, ClassifierMixin):
    """Conditional inference tree classifier.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, optional (default="gini")
        Criterion for splitting.

    selector : {"mc", "mi", "hybrid"}, optional (default="mc")
        Method for feature selection.

    splitter : {"best", "random", "hist-local", "hist-global"}
        Method for split selection.

    alpha_feature : float, optional (default=0.05)
        Alpha for feature selection.

    alpha_split : float, optional (default=0.05)
        Alpha for split selection.

    n_bins : int, optional (default=256)
        Number of bins to use when using histogram splitters.

    early_stopping_selector : bool, optional (default=True)
        Use early stopping during feature selection.

    early_stopping_splitter : bool, optional (default=True)
        Use early stopping during split selection.

    feature_scanning : bool, optional (default=True)
        ADD HERE.
    ...

    Attributes
    ----------
    classes_ : np.ndarray
        ADD HERE.

    n_classes_ : int
        ADD HERE.

    feature_importances_ : np.ndarray
        ADD HERE.

    n_features_in_ : int
        ADD HERE.

    tree_ : Node
        ADD HERE.
    """
    def __init__(
        self,
        *,
        criterion="gini",
        selector="mc",
        splitter="best",
        alpha_feature=0.05,
        alpha_split=0.05,
        n_bins=256,
        early_stopping_selector=True,
        early_stopping_splitter=True,
        feature_scanning=True,
        feature_muting=True,
        n_permutations_selector="auto",
        n_permutations_splitter="auto",
        max_depth=None,
        max_features=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        n_jobs=1,
        random_state=None,
        verbose=0,
    ) -> None:
        hps = self._model(
            criterion=criterion,
            selector=selector,
            splitter=splitter,
            alpha_feature=alpha_feature,
            alpha_split=alpha_split,
            n_bins=n_bins,
            early_stopping_selector=early_stopping_selector,
            early_stopping_splitter=early_stopping_splitter,
            feature_scanning=feature_scanning,
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

        super().__init__(**hps.dict())

    @property
    def _model(self) -> ModelMetaclass:
        """Model for estimator's hyperparameters."""
        return ConditionalInferenceTreeClassifierParameters

    def _node_impurity(self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Calculate node impurity.

        Parameters
        ----------
        y : np.ndarray
            Labels for parent node.

        y_left : np.ndarray
            Labels for left child node.

        y_right : np.ndarray
            Labels for right child node.

        Returns
        -------
        float
            Node impurity measure.
        """
        pass

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
        pass

    def _splitter(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Find optimal threshold for binary split in node.

        Parameters
        ----------
        X : np.ndarray
            Features for node.

        y : np.ndarray
            Labels for node.

        Returns
        -------
        Tuple[float, float]
            Threshold and threshold probability value with order (threshold, threshold_pval).
        """
        pass

    def _selector(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        """Find most correlated feature with label.

        Parameters
        ----------
        X : np.ndarray
            Features for node.

        y : np.ndarray
            Labels for node.

        Returns
        -------
        Tuple[int, float]
            Feature index and probability value with order (feature, feature_pval).
        """
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConditionalInferenceTreeClassifier":
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
        super().fit(X, y)

        return self
