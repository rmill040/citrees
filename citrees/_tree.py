from typing import ClassVar, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ._base import BaseConditionalInferenceTree, BaseConditionalInferenceTreeParameters


class ConditionalInferenceTreeClassifierParameters(BaseConditionalInferenceTreeParameters):
    """Model for BaseConditionalInferenceTree parameters.

    Parameters
    ----------
    selector : str
        Feature selection method.
    """
    criterion: Literal["gini", "entropy"] = "gini"
    selector: Literal["mc", "mi", "hybrid"] = "mc"


class ConditionalInferenceTreeClassifier(BaseConditionalInferenceTree, BaseEstimator, ClassifierMixin):
    """Conditional inference tree classifier.

    Parameters
    ----------
    ADD HERE.
    """
    model: ClassVar = ConditionalInferenceTreeClassifierParameters

    def __init__(
        self,
        *,
        criterion="gini",
        selector="mc",
        splitter="best",
        alpha_selector=0.05,
        alpha_splitter=0.05,
        n_bins=256,
        early_stopping=True,
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
        hps = self.__class__.model(
            criterion=criterion,
            selector=selector,
            splitter=splitter,
            alpha_selector=alpha_selector,
            alpha_splitter=alpha_splitter,
            n_bins=n_bins,
            early_stopping=early_stopping,
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

        super(BaseConditionalInferenceTree, self).__init__(**hps.dict())

    def _splitter(self):
        pass

    def _selector(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConditionalInferenceTreeClassifier":
        """ADD HERE.

        Parameters
        ----------

        Returns
        -------
        """
        super().fit(X, y)
