from typing import Union

from numba import njit
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from ._base import BaseConditionalInferenceTree


@njit(cache=True, fastmath=True, nogil=True)
def _estimate_proba(*, y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """ADD HERE.
    
    Parameters
    ----------
    
    Returns
    -------
    """
    return np.array([np.mean(y == j) for j in classes])


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

    def _node_value(self, y: np.ndarray) -> Union[float, np.ndarray]:
        """Calculate class probabilities in terminal node.

        Parameters
        ----------
        y : np.ndarray
            Class labels for node.

        Returns
        -------
        np.ndarray
            Class probabilities.
        """
        return _estimate_proba(y=y, classes=self.classes_)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConditionalInferenceTreeClassifier":
        """ADD HERE.
        
        Parameters
        ----------
        
        Returns
        -------
        """
        X = X.astype(float)
        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(y)
        super().fit(X=X, y=y)

        return self
