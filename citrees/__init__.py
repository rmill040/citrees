# flake8: noqa
from citrees._conformal import ConformalClassifier, ConformalRegressor, CQR
from citrees._forest import ConditionalInferenceForestClassifier, ConditionalInferenceForestRegressor
from citrees._importance import SHAPExplainer, compute_importance
from citrees._tree import ConditionalInferenceTreeClassifier, ConditionalInferenceTreeRegressor

__all__ = [
    "ConditionalInferenceTreeClassifier",
    "ConditionalInferenceTreeRegressor",
    "ConditionalInferenceForestClassifier",
    "ConditionalInferenceForestRegressor",
    "compute_importance",
    "SHAPExplainer",
    "ConformalClassifier",
    "ConformalRegressor",
    "CQR",
]

# Try to update recursion limit
try:
    import sys

    sys.setrecursionlimit(100_000)
except Exception:
    pass
