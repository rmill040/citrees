# flake8: noqa
from citrees._conformal import ConformalClassifier, ConformalRegressor, CQR
from citrees._forest import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
)
from citrees._importance import SHAPExplainer, compute_importance
from citrees._tree import ConditionalInferenceTreeClassifier, ConditionalInferenceTreeRegressor
from citrees._types import (
    BootstrapMethod,
    EarlyStopping,
    EstimatorType,
    ImportanceMethod,
    MaxValuesMethod,
    NResamples,
    SamplingMethod,
    ThresholdMethod,
)

__all__ = [
    # Estimators
    "ConditionalInferenceTreeClassifier",
    "ConditionalInferenceTreeRegressor",
    "ConditionalInferenceForestClassifier",
    "ConditionalInferenceForestRegressor",
    # Utilities
    "compute_importance",
    "SHAPExplainer",
    "ConformalClassifier",
    "ConformalRegressor",
    "CQR",
    # Enums
    "EarlyStopping",
    "NResamples",
    "MaxValuesMethod",
    "ThresholdMethod",
    "BootstrapMethod",
    "SamplingMethod",
    "ImportanceMethod",
    "EstimatorType",
]

# Increase recursion limit for deep tree building
# Note: _predict_value and _reestimate_tree_by_path use iterative traversal,
# but _build_tree is recursive. Depth is bounded by max_depth, min_samples_split,
# and min_samples_leaf parameters. 5000 is sufficient for any reasonable tree.
import sys

_current_limit = sys.getrecursionlimit()
if _current_limit < 5000:
    try:
        sys.setrecursionlimit(5000)
    except (ValueError, RecursionError):
        pass
