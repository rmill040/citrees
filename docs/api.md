# API Reference

This page documents the public `citrees` package surface. The generated sections
below are rendered from the package docstrings, so parameter defaults and method
signatures stay tied to the implementation.

## Main Estimators

Use these classes for normal model fitting and prediction.

::: citrees.ConditionalInferenceTreeClassifier

::: citrees.ConditionalInferenceTreeRegressor

::: citrees.ConditionalInferenceForestClassifier

::: citrees.ConditionalInferenceForestRegressor

## Configuration Enums

These enums are exported from `citrees` and can be passed directly instead of
their string values.

::: citrees.EarlyStopping

::: citrees.NResamples

::: citrees.MaxValuesMethod

::: citrees.ThresholdMethod

::: citrees.SamplingMethod

## Scikit-Learn Compatibility

The estimators follow scikit-learn conventions for `fit`, `predict`,
`get_params`, `set_params`, cloning, pipelines, and grid search.

```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from citrees import ConditionalInferenceForestClassifier

scores = cross_val_score(
    ConditionalInferenceForestClassifier(random_state=42),
    X,
    y,
    cv=5,
)

grid = GridSearchCV(
    ConditionalInferenceForestClassifier(random_state=42),
    {
        "alpha_selector": [0.01, 0.05, 0.10],
        "n_estimators": [50, 100],
    },
    cv=5,
)
grid.fit(X, y)

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("forest", ConditionalInferenceForestClassifier(random_state=42)),
    ]
)
pipe.fit(X_train, y_train)
```

See [Parameters](parameters.md) for tuning guidance and compatibility notes for
parameter combinations such as `bootstrap`, `oob_score`, `sampling_method`, and
`n_resamples_*`.
