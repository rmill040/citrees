# Parameters Reference

Complete reference for all citrees parameters with tuning guidance.

## Tree Parameters

### Core Parameters

| Parameter        | Type        | Default                        | Description                             |
| ---------------- | ----------- | ------------------------------ | --------------------------------------- |
| `selector`       | str or list | `'mc'` (clf) / `'pc'` (reg)    | Feature selection method                |
| `splitter`       | str         | `'gini'` (clf) / `'mse'` (reg) | Split criterion                         |
| `alpha_selector` | float       | 0.05                           | P-value threshold for feature selection |
| `alpha_splitter` | float       | 0.05                           | P-value threshold for split selection   |

### Resampling Parameters

| Parameter              | Type              | Default           | Description                        |
| ---------------------- | ----------------- | ----------------- | ---------------------------------- |
| `n_resamples_selector` | NResamples or int | `NResamples.AUTO` | Permutation resamples for selector |
| `n_resamples_splitter` | NResamples or int | `NResamples.AUTO` | Permutation resamples for splitter |

Options for `n_resamples_*`:

- `NResamples.AUTO`: Adaptive based on alpha (recommended). Implemented as:
  - `lower = ceil(1/alpha)`
  - `upper = ceil(z^2 * (1 - alpha) / alpha)` where `z = Φ^{-1}(1 - alpha)`
  - `B = max(lower, upper)`
- `NResamples.MINIMUM`: `ceil(1/alpha)` resamples (minimum resolution to allow `p < alpha` with +1 correction)
- `NResamples.MAXIMUM`: `ceil(1 / (4 * alpha^2))` resamples (high precision; matches `100` when `alpha=0.05`)
- `int`: Exact number of resamples (must be `>= ceil(1/alpha)`; with Bonferroni, this scales by the number of tests)

### Alpha Adjustment

| Parameter               | Type | Default | Description                               |
| ----------------------- | ---- | ------- | ----------------------------------------- |
| `adjust_alpha_selector` | bool | True    | Bonferroni correction for feature tests   |
| `adjust_alpha_splitter` | bool | True    | Bonferroni correction for threshold tests |

### Early Stopping

| Parameter                           | Type                  | Default                    | Description |
| ----------------------------------- | --------------------- | -------------------------- | ----------- |
| `early_stopping_selector`           | EarlyStopping or None | `EarlyStopping.ADAPTIVE`   | Sequential stopping rule for selector permutation tests |
| `early_stopping_splitter`           | EarlyStopping or None | `EarlyStopping.ADAPTIVE`   | Sequential stopping rule for splitter permutation tests |
| `early_stopping_confidence_selector`| float                 | 0.95                       | Posterior-confidence threshold γ for `EarlyStopping.ADAPTIVE` (selectors) |
| `early_stopping_confidence_splitter`| float                 | 0.95                       | Posterior-confidence threshold γ for `EarlyStopping.ADAPTIVE` (splitters) |

Notes:
- `EarlyStopping.ADAPTIVE` uses a Beta posterior confidence rule to stop when confident about `p < alpha` or `p >= alpha`.
- `EarlyStopping.SIMPLE` uses a futility + significance heuristic and can inflate Type I error.
- Use `early_stopping_*=None` for fixed-B Monte Carlo p-values (recommended for publication-grade p-value claims).

### Feature Optimization

| Parameter          | Type | Default | Description                           |
| ------------------ | ---- | ------- | ------------------------------------- |
| `feature_muting`   | bool | True    | Remove clearly uninformative features |
| `feature_scanning` | bool | True    | Test promising features first         |

### Threshold Generation

| Parameter            | Type            | Default                 | Description                      |
| -------------------- | --------------- | ----------------------- | -------------------------------- |
| `threshold_method`   | ThresholdMethod | `ThresholdMethod.EXACT` | How to generate split candidates |
| `max_thresholds`     | int or float    | None                    | Maximum thresholds per feature   |
| `threshold_scanning` | bool            | True                    | Test promising thresholds first  |

Options for `threshold_method`:

- `ThresholdMethod.EXACT`: All unique values (precise, slower)
- `ThresholdMethod.RANDOM`: Random subset of values
- `ThresholdMethod.PERCENTILE`: Quantile-based thresholds
- `ThresholdMethod.HISTOGRAM`: Equal-width bins

### Tree Structure

| Parameter           | Type               | Default | Description                   |
| ------------------- | ------------------ | ------- | ----------------------------- |
| `max_depth`         | int                | None    | Maximum tree depth            |
| `min_samples_split` | int                | 2       | Minimum samples to split node |
| `min_samples_leaf`  | int                | 1       | Minimum samples in leaf       |
| `max_features`      | str, int, or float | None    | Features per split            |

Options for `max_features`:

- `None`: All features
- `MaxValuesMethod.SQRT`: Square root of total features
- `MaxValuesMethod.LOG2`: Log base 2 of total features
- `int`: Exact number of features
- `float`: Fraction of total features

### Honest Estimation

| Parameter          | Type  | Default | Description                                  |
| ------------------ | ----- | ------- | -------------------------------------------- |
| `honesty`          | bool  | False   | Enable sample splitting                      |
| `honesty_fraction` | float | 0.5     | Fraction for estimation sample (rest for structure) |

---

## Forest Parameters

All tree parameters plus:

| Parameter          | Type            | Default                    | Description                      |
| ------------------ | --------------- | -------------------------- | -------------------------------- |
| `n_estimators`     | int             | 100                        | Number of trees                  |
| `max_samples`      | float           | None                       | Bootstrap sample size (fraction) |
| `bootstrap_method` | BootstrapMethod | `BootstrapMethod.BAYESIAN` | Sampling method                  |
| `sampling_method`  | SamplingMethod  | `SamplingMethod.STRATIFIED`| How to stratify samples          |
| `n_jobs`           | int             | 1                          | Parallel jobs (-1 for all cores) |
| `oob_score`        | bool            | False                      | Compute out-of-bag score (requires bootstrap) |

Options for `bootstrap_method`:

- `BootstrapMethod.BAYESIAN`: Poisson bootstrap (recommended)
- `BootstrapMethod.CLASSIC`: Standard bootstrap with replacement

Options for `sampling_method`:

- `SamplingMethod.STRATIFIED`: Maintain class proportions
- `SamplingMethod.BALANCED`: Equal class weights

---

## Selector Options

### Classification Selectors

| Selector | Description                       | Scale  | Complexity |
| -------- | --------------------------------- | ------ | ---------- |
| `'mc'`   | Multiple correlation (ANOVA)      | [0, 1] | O(n)       |
| `'mi'`   | Mutual information                | [0, ∞) | O(n log n) |
| `'rdc'`  | Randomized dependence coefficient | [0, 1] | O(n log n) |

### Regression Selectors

| Selector | Description                       | Scale  | Complexity |
| -------- | --------------------------------- | ------ | ---------- |
| `'pc'`   | Pearson correlation               | [0, 1] | O(n)       |
| `'dc'`   | Distance correlation              | [0, 1] | O(n²)      |
| `'rdc'`  | Randomized dependence coefficient | [0, 1] | O(n log n) |

### Multi-Selector Mode

```python
# Classification: combine mc and rdc (both [0,1] scale)
tree = ConditionalInferenceTreeClassifier(selector=['mc', 'rdc'])

# Regression: combine all three
tree = ConditionalInferenceTreeRegressor(selector=['pc', 'dc', 'rdc'])
```

!!! warning "Mutual Information" `'mi'` cannot be combined with other selectors
because its scale is unbounded.

---

## Splitter Options

### Classification Splitters

| Splitter    | Description      | Range           |
| ----------- | ---------------- | --------------- |
| `'gini'`    | Gini impurity    | [0, 0.5] binary |
| `'entropy'` | Information gain | [0, log₂K]      |

### Regression Splitters

| Splitter | Description         | Range  |
| -------- | ------------------- | ------ |
| `'mse'`  | Mean squared error  | [0, ∞) |
| `'mae'`  | Mean absolute error | [0, ∞) |

---

## Tuning Guide

### Conservative (High Interpretability)

```python
tree = ConditionalInferenceTreeClassifier(
    alpha_selector=0.01,        # Strict significance
    alpha_splitter=0.01,
    n_resamples_selector=NResamples.MAXIMUM,
    adjust_alpha_selector=True,
    adjust_alpha_splitter=True,
    feature_muting=True,
)
```

### Balanced (Default)

```python
tree = ConditionalInferenceTreeClassifier(
    alpha_selector=0.05,
    alpha_splitter=0.05,
    n_resamples_selector=NResamples.AUTO,
    adjust_alpha_selector=True,
    early_stopping_selector=EarlyStopping.ADAPTIVE,
)
```

### Fast (Exploratory)

```python
tree = ConditionalInferenceTreeClassifier(
    alpha_selector=0.10,
    alpha_splitter=0.10,
    n_resamples_selector=NResamples.MINIMUM,
    adjust_alpha_selector=False,
    early_stopping_selector=EarlyStopping.ADAPTIVE,
    threshold_method=ThresholdMethod.PERCENTILE,
    max_thresholds=50,
)
```

### Large Datasets

```python
tree = ConditionalInferenceTreeClassifier(
    threshold_method=ThresholdMethod.HISTOGRAM,
    max_thresholds=256,
    max_depth=10,
    min_samples_leaf=20,
)
```

### High-Dimensional Data

```python
tree = ConditionalInferenceTreeClassifier(
    max_features=MaxValuesMethod.SQRT,
    feature_muting=True,
    feature_scanning=True,
    selector=['mc', 'rdc'],  # Multiple selectors
)
```
