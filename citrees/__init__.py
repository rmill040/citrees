# flake8: noqa
from citrees._forest import ConditionalInferenceForestClassifier, ConditionalInferenceForestRegressor
from citrees._tree import ConditionalInferenceTreeClassifier, ConditionalInferenceTreeRegressor

# Try to update recursion limit
try:
    import sys

    sys.setrecursionlimit(100_000)
except Exception:
    pass
