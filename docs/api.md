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
| `feature_names_in_`    | List[str] | Feature names seen during fit (order matters for pandas inputs) |
| `feature_importances_` | ndarray | Feature importance scores (MDI)    |
| `classes_`             | ndarray | Unique class labels                |
| `n_classes_`           | int     | Number of classes                  |

**Methods:**

```python
# Fit the tree
tree.fit(X, y)

# Leaf indices for each sample
leaf_ids = tree.apply(X)

# Decision path (csr_matrix)
path = tree.decision_path(X)

# Predict class labels
y_pred = tree.predict(X)

# Predict class probabilities
y_proba = tree.predict_proba(X)

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
| `feature_names_in_`    | List[str] | Feature names seen during fit (order matters for pandas inputs) |
| `feature_importances_` | ndarray | Feature importance scores (MDI)    |

**Methods:**

```python
# Fit the tree
tree.fit(X, y)

# Leaf indices for each sample
leaf_ids = tree.apply(X)

# Decision path (csr_matrix)
path = tree.decision_path(X)

# Predict values
y_pred = tree.predict(X)

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
| `feature_names_in_`    | List[str] | Feature names seen during fit (order matters for pandas inputs) |
| `feature_importances_` | ndarray | Averaged feature importance        |
| `classes_`             | ndarray | Unique class labels                |
| `n_classes_`           | int     | Number of classes                  |
| `oob_score_`           | float   | OOB accuracy over samples with OOB predictions (if enabled) |
| `oob_decision_function_` | ndarray | OOB class probabilities (if enabled; rows with no OOB remain zero) |

**Methods:**

```python
# Fit the forest
forest.fit(X, y)

# Leaf indices per estimator (n_samples, n_estimators)
leaf_ids = forest.apply(X)

# Decision paths across all estimators
indicator, n_nodes_ptr = forest.decision_path(X)

# Predict class labels
y_pred = forest.predict(X)

# Predict class probabilities
y_proba = forest.predict_proba(X)

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
| `feature_names_in_`    | List[str] | Feature names seen during fit (order matters for pandas inputs) |
| `feature_importances_` | ndarray | Averaged feature importance        |
| `oob_score_`           | float   | OOB R² over samples with OOB predictions (if enabled) |
| `oob_prediction_`      | ndarray | OOB predictions (if enabled; entries with no OOB remain zero) |

**Methods:**

```python
# Fit the forest
forest.fit(X, y)

# Leaf indices per estimator (n_samples, n_estimators)
leaf_ids = forest.apply(X)

# Decision paths across all estimators
indicator, n_nodes_ptr = forest.decision_path(X)

# Predict values
y_pred = forest.predict(X)

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
