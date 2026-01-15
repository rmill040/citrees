# API Reference

## Tree Classes

### ConditionalInferenceTreeClassifier

```python
from citrees import ConditionalInferenceTreeClassifier
```

A conditional inference tree for classification tasks.

**Parameters:**

See [Parameters Reference](parameters.md) for complete list.

**Attributes:**

| Attribute              | Type    | Description                        |
| ---------------------- | ------- | ---------------------------------- |
| `tree_`                | dict    | The fitted tree structure          |
| `n_features_in_`       | int     | Number of features seen during fit |
| `feature_importances_` | ndarray | Feature importance scores (MDI)    |
| `classes_`             | ndarray | Unique class labels                |
| `n_classes_`           | int     | Number of classes                  |

**Methods:**

```python
# Fit the tree
tree.fit(X, y)

# Predict class labels
y_pred = tree.predict(X)

# Predict class probabilities
y_proba = tree.predict_proba(X)

# Get leaf indices
leaf_ids = tree.apply(X)

# Get decision path
path = tree.decision_path(X)

# Export tree structure
tree.export_text()
tree.export_graphviz(filename)
```

---

### ConditionalInferenceTreeRegressor

```python
from citrees import ConditionalInferenceTreeRegressor
```

A conditional inference tree for regression tasks.

**Parameters:**

See [Parameters Reference](parameters.md) for complete list.

**Attributes:**

| Attribute              | Type    | Description                        |
| ---------------------- | ------- | ---------------------------------- |
| `tree_`                | dict    | The fitted tree structure          |
| `n_features_in_`       | int     | Number of features seen during fit |
| `feature_importances_` | ndarray | Feature importance scores (MDI)    |

**Methods:**

```python
# Fit the tree
tree.fit(X, y)

# Predict values
y_pred = tree.predict(X)

# Get leaf indices
leaf_ids = tree.apply(X)

# Get decision path
path = tree.decision_path(X)
```

---

## Forest Classes

### ConditionalInferenceForestClassifier

```python
from citrees import ConditionalInferenceForestClassifier
```

A random forest of conditional inference trees for classification.

**Parameters:**

See [Parameters Reference](parameters.md) for complete list.

**Attributes:**

| Attribute              | Type    | Description                        |
| ---------------------- | ------- | ---------------------------------- |
| `estimators_`          | list    | List of fitted trees               |
| `n_features_in_`       | int     | Number of features seen during fit |
| `feature_importances_` | ndarray | Averaged feature importance        |
| `classes_`             | ndarray | Unique class labels                |
| `n_classes_`           | int     | Number of classes                  |
| `oob_score_`           | float   | Out-of-bag score (if computed)     |

**Methods:**

```python
# Fit the forest
forest.fit(X, y)

# Predict class labels
y_pred = forest.predict(X)

# Predict class probabilities
y_proba = forest.predict_proba(X)

# Get leaf indices for each tree
leaf_ids = forest.apply(X)

# Get decision paths
paths = forest.decision_path(X)
```

---

### ConditionalInferenceForestRegressor

```python
from citrees import ConditionalInferenceForestRegressor
```

A random forest of conditional inference trees for regression.

**Parameters:**

See [Parameters Reference](parameters.md) for complete list.

**Attributes:**

| Attribute              | Type    | Description                        |
| ---------------------- | ------- | ---------------------------------- |
| `estimators_`          | list    | List of fitted trees               |
| `n_features_in_`       | int     | Number of features seen during fit |
| `feature_importances_` | ndarray | Averaged feature importance        |
| `oob_score_`           | float   | Out-of-bag score (if computed)     |

**Methods:**

```python
# Fit the forest
forest.fit(X, y)

# Predict values
y_pred = forest.predict(X)

# Get leaf indices for each tree
leaf_ids = forest.apply(X)
```

---

## Utility Functions

### Export Functions

```python
# Text representation
text = tree.export_text(feature_names=feature_names)

# GraphViz DOT format
tree.export_graphviz(
    filename="tree.dot",
    feature_names=feature_names,
    class_names=class_names,
)

# Render to image (requires graphviz)
# dot -Tpng tree.dot -o tree.png
```

---

## Scikit-learn Compatibility

citrees classes are compatible with scikit-learn utilities:

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from citrees import ConditionalInferenceForestClassifier

# Cross-validation
scores = cross_val_score(
    ConditionalInferenceForestClassifier(),
    X, y, cv=5
)

# Grid search
param_grid = {
    'alpha_selector': [0.01, 0.05, 0.10],
    'n_estimators': [50, 100, 200],
}
grid = GridSearchCV(
    ConditionalInferenceForestClassifier(),
    param_grid,
    cv=5
)
grid.fit(X, y)

# Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('forest', ConditionalInferenceForestClassifier()),
])
pipe.fit(X_train, y_train)
```
