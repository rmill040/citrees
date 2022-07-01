from abc import ABC, abstractmethod

# from sklearn.utils.validation import check_is_fitted
from typing import Any, ClassVar, Dict, Optional, Union

# import numpy as np
from pydantic import BaseModel, confloat, conint, PositiveInt, validator

# from multiprocessing import cpu_count


_VALIDATOR_KWARGS = {"pre": False, "always": True}


########
# TREE #
########


class ModelConditionalInferenceTreeBase(BaseModel):
    """Model for ConditionalInferenceTreeBase parameters."""

    alpha: confloat(gt=0.0, lt=1.0) = 0.05  # type: ignore
    selector: Any
    splitter: Any
    early_stopping: bool = True
    muting: bool = True
    n_permutations: PositiveInt = 100
    max_depth: Optional[PositiveInt] = None
    max_features: Optional[PositiveInt] = None
    max_samples: Optional[PositiveInt] = None
    min_samples_split: Optional[conint(ge=2)] = None  # type: ignore
    min_samples_leaf: Optional[PositiveInt] = None
    class_weight: Optional[Union[Dict[float, float], str]] = None
    n_jobs: Optional[int] = None
    random_state: Optional[PositiveInt] = None
    verbose: Optional[PositiveInt] = None

    @validator("selector", **_VALIDATOR_KWARGS)
    def validate_selector(cls, v, field):  # type: ignore
        """Validate selector."""
        return v

    @validator("splitter", **_VALIDATOR_KWARGS)
    def validate_splitter(cls, v, field):  # type: ignore
        """Validate splitter."""
        return v

    @validator("n_jobs", **_VALIDATOR_KWARGS)
    def validate_n_jobs(cls, v, field):  # type: ignore
        """Validate n_jobs."""
        return v

    @validator("class_weight", **_VALIDATOR_KWARGS)
    def validate_class_weight(cls, v, field):  # type: ignore
        """Validate class_weight."""
        return v


class BaseConditionalInferenceTree(ABC):
    """Base class for conditional inference tree."""

    validator: ClassVar = ModelConditionalInferenceTreeBase

    @abstractmethod
    def __init__(
        self,
        *,
        alpha: float,
        selector: str,
        splitter: str,
        early_stopping: bool,
        muting: bool,
        n_permutations: int,
        max_depth: Optional[int],
        max_features: Optional[int],
        max_samples: Optional[int],
        min_samples_split: Optional[int],
        min_samples_leaf: Optional[int],
        class_weight: Optional[str],
        n_jobs: Optional[int],
        random_state: Optional[int],
        verbose: int,
    ) -> None:
        # Validate arguments using pydantic
        parameters = BaseConditionalInferenceTree.validator(
            alpha=alpha,
            selector=selector,
            splitter=splitter,
            early_stopping=early_stopping,
            muting=muting,
            n_permutations=n_permutations,
            max_depth=max_depth,
            max_features=max_features,
            max_samples=max_samples,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

        self.alpha = parameters.alpha
        self.selector = parameters.selector
        self.splitter = parameters.splitter
        self.early_stopping = parameters.early_stopping
        self.muting = parameters.muting
        self.n_permutations = parameters.n_permutations
        self.max_depth = parameters.max_depth
        self.max_features = parameters.max_features
        self.max_samples = parameters.max_samples
        self.min_samples_split = parameters.min_samples_split
        self.min_samples_leaf = parameters.min_samples_leaf
        self.class_weight = parameters.class_weight
        self.n_jobs = parameters.n_jobs
        self.random_state = parameters.random_state
        self.verbose = parameters.verbose

        # Additional attributes
        self.root = None
        self.splitter_counter_ = 0


##########
# FOREST #
##########


class ModelConditionalInferenceForestBase(ModelConditionalInferenceTreeBase):
    """Model for ConditionalInferenceForestBase parameters."""

    bootstrap: bool = True
    bayes: bool = True


class BaseConditionalInferenceForest(ABC):
    """Base class for conditional inference forest."""

    validator: ClassVar = ModelConditionalInferenceForestBase

    @abstractmethod
    def __init__(
        self,
        *,
        alpha: float,
        selector: str,
        splitter: str,
        early_stopping: bool,
        muting: bool,
        n_permutations: int,
        max_depth: Optional[int],
        max_features: Optional[int],
        max_samples: Optional[int],
        min_samples_split: Optional[int],
        min_samples_leaf: Optional[int],
        class_weight: Optional[str],
        bootstrap: bool,
        bayes: bool,
        n_jobs: Optional[int],
        random_state: Optional[int],
        verbose: int,
    ) -> None:
        # Validate arguments using pydantic
        parameters = BaseConditionalInferenceForest.validator(
            alpha=alpha,
            selector=selector,
            splitter=splitter,
            early_stopping=early_stopping,
            muting=muting,
            n_permutations=n_permutations,
            max_depth=max_depth,
            max_features=max_features,
            max_samples=max_samples,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            bootstrap=bootstrap,
            bayes=bayes,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

        self.alpha = parameters.alpha
        self.selector = parameters.selector
        self.splitter = parameters.splitter
        self.early_stopping = parameters.early_stopping
        self.muting = parameters.muting
        self.n_permutations = parameters.n_permutations
        self.max_depth = parameters.max_depth
        self.max_features = parameters.max_features
        self.max_samples = parameters.max_samples
        self.min_samples_split = parameters.min_samples_split
        self.min_samples_leaf = parameters.min_samples_leaf
        self.class_weight = parameters.class_weight
        self.bootstrap = parameters.bootstrap
        self.bayes = parameters.bayes
        self.n_jobs = parameters.n_jobs
        self.random_state = parameters.random_state
        self.verbose = parameters.verbose

        # Package parameters for single tree models
        self._tree_kwargs = {
            "alpha": self.alpha,
            "selector": self.selector,
            "splitter": self.splitter,
            "early_stopping": self.early_stopping,
            "muting": self.muting,
            "n_permutations": self.n_permutations,
            "max_depth": self.max_depth,
            "max_features": self.max_features,
            "max_samples": self.max_samples,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "class_weight": self.class_weight,
            "bootstrap": self.bootstrap,
            "bayes": self.bayes,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }


if __name__ == "__main__":
    import pdb

    pdb.set_trace()
