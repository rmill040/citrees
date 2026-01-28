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

| Parameter              | Type                     | Default           | Description                        |
| ---------------------- | ------------------------ | ----------------- | ---------------------------------- |
| `n_resamples_selector` | NResamples, int, or None | `NResamples.AUTO` | Permutation resamples for selector |
| `n_resamples_splitter` | NResamples, int, or None | `NResamples.AUTO` | Permutation resamples for splitter |

Options for `n_resamples_*`:

- `NResamples.AUTO`: Adaptive based on alpha (recommended). Implemented as:
  - `lower = ceil(1/alpha)`
  - `upper = ceil(z^2 * (1 - alpha) / alpha)` where `z = Φ^{-1}(1 - alpha)`
  - `B = max(lower, upper)`
- `NResamples.MINIMUM`: `ceil(1/alpha)` resamples (minimum resolution to allow
  `p < alpha` with +1 correction)
- `NResamples.MAXIMUM`: `ceil(1 / (4 * alpha^2))` resamples (high precision;
  matches `100` when `alpha=0.05`)
- `int`: Exact number of resamples (must be `>= ceil(1/alpha)`; with Bonferroni,
  this scales by the number of tests)
- `None`: Disable permutation tests (selection/splitting uses raw association or
  impurity metric)

Bonferroni behavior:

- When `adjust_alpha_* = True` and multiple hypotheses are tested at a node,
  citrees internally uses the Bonferroni threshold `alpha / n_tests` and adjusts
  the effective resample budget accordingly (string presets apply to
  `alpha / n_tests`; integers are multiplied by `n_tests`).

### Alpha Adjustment

| Parameter               | Type | Default | Description                               |
| ----------------------- | ---- | ------- | ----------------------------------------- |
| `adjust_alpha_selector` | bool | True    | Bonferroni correction for feature tests   |
| `adjust_alpha_splitter` | bool | True    | Bonferroni correction for threshold tests |

### Early Stopping

| Parameter                            | Type                  | Default                  | Description                                                               |
| ------------------------------------ | --------------------- | ------------------------ | ------------------------------------------------------------------------- |
| `early_stopping_selector`            | EarlyStopping or None | `EarlyStopping.ADAPTIVE` | Sequential stopping rule for selector permutation tests                   |
| `early_stopping_splitter`            | EarlyStopping or None | `EarlyStopping.ADAPTIVE` | Sequential stopping rule for splitter permutation tests                   |
| `early_stopping_confidence_selector` | float                 | 0.95                     | Posterior-confidence threshold γ for `EarlyStopping.ADAPTIVE` (selectors) |
| `early_stopping_confidence_splitter` | float                 | 0.95                     | Posterior-confidence threshold γ for `EarlyStopping.ADAPTIVE` (splitters) |

Notes:

- `EarlyStopping.ADAPTIVE` uses a Beta posterior confidence rule to stop when
  confident about `p < alpha` or `p >= alpha`.
- `EarlyStopping.SIMPLE` uses a futility + significance heuristic and can
  inflate Type I error.
- Use `early_stopping_*=None` for fixed-B Monte Carlo p-values (recommended for
  publication-grade p-value claims).
- `adjust_alpha_*`, `feature_muting`, and `early_stopping_*` are only used when
  `n_resamples_*` is not `None`.

### Feature Optimization

| Parameter          | Type | Default | Description                           |
| ------------------ | ---- | ------- | ------------------------------------- |
| `feature_muting`   | bool | True    | Remove clearly uninformative features |
| `feature_scanning` | bool | True    | Test promising features first         |

Notes:

- `feature_scanning` only applies when `early_stopping_selector` is not `None`.

### Threshold Generation

| Parameter            | Type                                 | Default                 | Description                      |
| -------------------- | ------------------------------------ | ----------------------- | -------------------------------- |
| `threshold_method`   | ThresholdMethod                      | `ThresholdMethod.EXACT` | How to generate split candidates |
| `max_thresholds`     | MaxValuesMethod, int, float, or None | None                    | Maximum thresholds per feature   |
| `threshold_scanning` | bool                                 | True                    | Test promising thresholds first  |

Options for `threshold_method`:

- `ThresholdMethod.EXACT`: All unique **midpoints** (precise, slower)
- `ThresholdMethod.RANDOM`: Random subset of **midpoints**
- `ThresholdMethod.PERCENTILE`: Quantile-based **midpoints**
- `ThresholdMethod.HISTOGRAM`: Histogram bin edges over **midpoints**

Notes:

- `threshold_scanning` only applies when `early_stopping_splitter` is not
  `None`.
- If `threshold_method != ThresholdMethod.EXACT` and `max_thresholds=None`, all
  midpoints are tested (can be slow).

Options for `max_thresholds`:

- `None`: Use all available thresholds
- `MaxValuesMethod.SQRT`: √n thresholds (n = unique values; midpoints = n - 1)
- `MaxValuesMethod.LOG2`: log₂(n) thresholds (n = unique values; midpoints =
  n - 1)
- `int`: Exact number of thresholds
- `float` (0.0, 1.0]: Fraction of available thresholds

### Tree Structure

| Parameter               | Type                                 | Default | Description                        |
| ----------------------- | ------------------------------------ | ------- | ---------------------------------- |
| `max_depth`             | int                                  | None    | Maximum tree depth                 |
| `min_samples_split`     | int                                  | 2       | Minimum samples to split node      |
| `min_samples_leaf`      | int                                  | 1       | Minimum samples in leaf            |
| `min_impurity_decrease` | float                                | 0.0     | Minimum impurity decrease to split |
| `max_features`          | MaxValuesMethod, int, float, or None | None    | Features per split                 |

Options for `max_features`:

- `None`: All features
- `MaxValuesMethod.SQRT`: Square root of total features
- `MaxValuesMethod.LOG2`: Log base 2 of total features
- `int`: Exact number of features
- `float` (0.0, 1.0]: Fraction of total features

### Honest Estimation

| Parameter          | Type  | Default | Description                                         |
| ------------------ | ----- | ------- | --------------------------------------------------- |
| `honesty`          | bool  | False   | Enable sample splitting                             |
| `honesty_fraction` | float | 0.5     | Fraction for estimation sample (rest for structure) |

---

## Miscellaneous

| Parameter                     | Type        | Default | Description                                                                          |
| ----------------------------- | ----------- | ------- | ------------------------------------------------------------------------------------ |
| `random_state`                | int or None | None    | Random seed for permutation tests, sampling, and bootstrap; `None` uses a fresh seed |
| `verbose`                     | int         | 1       | Verbosity level (0=quiet; higher prints more progress; forests cap at 3)             |
| `check_for_unused_parameters` | bool        | False   | Warn when parameters are ineffective due to other settings                           |

---

## Forest Parameters

All tree parameters plus:

| Parameter          | Type                 | Default                     | Description                                                                             |
| ------------------ | -------------------- | --------------------------- | --------------------------------------------------------------------------------------- |
| `n_estimators`     | int                  | 100                         | Number of trees                                                                         |
| `max_samples`      | int/float/None       | None                        | Bootstrap sample cap (count or fraction)                                                |
| `bootstrap_method` | BootstrapMethod/None | `BootstrapMethod.BAYESIAN`  | Sampling method (or disable bootstrap)                                                  |
| `sampling_method`  | SamplingMethod/None  | `SamplingMethod.STRATIFIED` | How to sample classes during bootstrap (classification only)                            |
| `n_jobs`           | int or None          | None                        | Parallel jobs (-1 for all cores)                                                        |
| `oob_score`        | bool                 | False                       | Compute out-of-bag score (requires bootstrap; scores only samples with OOB predictions) |

Options for `bootstrap_method`:

- `BootstrapMethod.BAYESIAN`: Bayesian bootstrap with Dirichlet weights
  (recommended)
- `BootstrapMethod.CLASSIC`: Standard bootstrap with replacement
- `None`: Disable bootstrap (no OOB; `sampling_method` and `max_samples` must be `None`, and `oob_score` must be `False`)

Options for `sampling_method`:

- `SamplingMethod.STRATIFIED`: Sample within each class (expected proportions match training data)
- `SamplingMethod.UNDERSAMPLE`: Sample `n_min` per class (minority count), so total size is `K*n_min` (then capped by `max_samples`)
- `SamplingMethod.OVERSAMPLE`: Allocate a fixed total size (`max_samples` or `n`) as evenly as possible across classes (diff ≤ 1) and sample within class
- `None`: Unstratified bootstrap (classic or Bayesian depending on `bootstrap_method`)

Notes:

- `sampling_method` applies to classification forests only.
- Negative `n_jobs` values follow sklearn-style semantics (e.g., `-1` = all
  cores).
- `n_jobs=0` is invalid; use `None` or `1` to disable parallelism.
- `sampling_method` requires `bootstrap_method` to be set (bootstrap enabled).
- Invalid combinations (e.g., `bootstrap_method=None` with `sampling_method` set) raise a validation error.
- Forest classes default `max_features=MaxValuesMethod.SQRT` (trees default
  `None`).

Options for `max_samples`:

- `None`: No cap (the sampler chooses the per-tree base sample size)
- `int`: Maximum number of samples (must be >= 1)
- `float` (0.0, 1.0]: Fraction of samples

`max_samples` is only used when `bootstrap_method` is not `None`.

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
