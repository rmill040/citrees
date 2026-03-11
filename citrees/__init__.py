# flake8: noqa
from importlib.metadata import version as _get_version

from citrees._forest import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
)
from citrees._tree import ConditionalInferenceTreeClassifier, ConditionalInferenceTreeRegressor
from citrees._types import (
    EarlyStopping,
    EstimatorType,
    MaxValuesMethod,
    NResamples,
    SamplingMethod,
    ThresholdMethod,
)

try:
    __version__ = _get_version("citrees")
except Exception:
    __version__ = "unknown"

__all__ = [
    "__version__",
    "ConditionalInferenceTreeClassifier",
    "ConditionalInferenceTreeRegressor",
    "ConditionalInferenceForestClassifier",
    "ConditionalInferenceForestRegressor",
    "EarlyStopping",
    "NResamples",
    "MaxValuesMethod",
    "ThresholdMethod",
    "SamplingMethod",
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
