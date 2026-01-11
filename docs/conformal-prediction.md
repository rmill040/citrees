# Conformal Prediction

Conformal prediction provides distribution-free uncertainty quantification with
guaranteed coverage. It transforms any point predictor into a set predictor that
contains the true value with a specified probability, without making
distributional assumptions.

## Overview

| Aspect        | Description                                                |
| ------------- | ---------------------------------------------------------- |
| Purpose       | Prediction intervals/sets with coverage guarantees         |
| Assumption    | Exchangeability (weaker than i.i.d.)                       |
| Coverage      | Finite-sample valid: $P(Y \in \hat{C}(X)) \geq 1 - \alpha$ |
| Key reference | [Vovk et al. (2005)](http://www.alrw.net/articles/01.pdf)  |

---

## The Problem: Uncertainty Quantification

Standard machine learning models produce **point predictions** without
uncertainty:

```
Standard Prediction
───────────────────

Input X ──────► Model ──────► ŷ = 42.3

Question: How confident should we be in this prediction?
Answer: Unknown! No uncertainty quantification.
```

### Why Uncertainty Matters

| Application           | Consequence of Ignoring Uncertainty     |
| --------------------- | --------------------------------------- |
| Medical diagnosis     | Overconfident predictions harm patients |
| Financial forecasting | No risk assessment                      |
| Autonomous systems    | Unsafe decision making                  |
| Scientific research   | Invalid conclusions                     |

---

## Conformal Prediction: The Solution

Conformal prediction wraps any predictor to produce **prediction sets** with
guaranteed coverage:

```
Conformal Prediction
────────────────────

Input X ──────► Model + Conformal ──────► C(X) = [38.1, 46.5]

Guarantee: P(Y ∈ C(X)) ≥ 1 - α (e.g., 90%)
           This holds for ANY distribution!
```

### Key Properties

| Property                | Description                                         |
| ----------------------- | --------------------------------------------------- |
| **Distribution-free**   | No assumptions on $P(X, Y)$                         |
| **Finite-sample valid** | Coverage holds for any $n$, not just asymptotically |
| **Model-agnostic**      | Works with any base predictor                       |
| **Adaptive**            | Intervals can vary with input uncertainty           |

---

## Mathematical Framework

### Setup

Given:

- Training data: $(X_1, Y_1), \ldots, (X_n, Y_n)$
- Calibration data: $(X_{n+1}, Y_{n+1}), \ldots, (X_{n+m}, Y_{n+m})$
- Test point: $X_{test}$
- Significance level: $\alpha \in (0, 1)$

Goal: Construct $\hat{C}(X_{test})$ such that:

$$P(Y_{test} \in \hat{C}(X_{test})) \geq 1 - \alpha$$

### Conformity Scores

A **conformity score** measures how well a prediction fits an observation:

$$s_i = s(X_i, Y_i, \hat{f})$$

Common scores:

| Task           | Score               | Formula                        |
| -------------- | ------------------- | ------------------------------ | ------------------ | -------------------- |
| Regression     | Absolute residual   | $s_i =                         | Y_i - \hat{f}(X_i) | $                    |
| Regression     | Normalized residual | $s_i =                         | Y_i - \hat{f}(X_i) | / \hat{\sigma}(X_i)$ |
| Classification | 1 - probability     | $s_i = 1 - \hat{f}_{Y_i}(X_i)$ |

### Quantile Computation

The prediction set is constructed using the $(1-\alpha)$ quantile of calibration
scores:

$$\hat{q} = \text{Quantile}_{1-\alpha}\left(\{s_i\}_{i=n+1}^{n+m} \cup \{\infty\}\right)$$

The $\infty$ ensures finite-sample validity when $m$ is small.

### Coverage Guarantee

**Theorem** (Vovk et al., 2005): If $(X_i, Y_i)$ are exchangeable, then:

$$P(Y_{test} \in \hat{C}(X_{test})) \geq 1 - \alpha$$

This holds for **any** underlying distribution and **any** sample size.

---

## Algorithms

### Split Conformal Prediction

The simplest and most common approach:

```
Algorithm: Split Conformal Prediction (Regression)
Input: Training data D_train, calibration data D_cal, test point x, level α

1. Train model:
   f̂ ← Train(D_train)

2. Compute calibration scores:
   For i in D_cal:
       s_i ← |y_i - f̂(x_i)|

3. Compute quantile:
   q̂ ← Quantile((1-α)(1 + 1/|D_cal|), {s_i})

4. Construct prediction interval:
   ŷ ← f̂(x)
   C(x) ← [ŷ - q̂, ŷ + q̂]

5. Return C(x)
```

### CV+ (Cross-Validation+)

More data-efficient than split conformal:

```
Algorithm: CV+ Conformal Prediction
Input: Data D, test point x, K folds, level α

1. K-fold cross-validation:
   For k = 1 to K:
       D_train_k ← D \ D_k
       f̂_k ← Train(D_train_k)

       For i in D_k:
           s_i ← |y_i - f̂_k(x_i)|
           ŷ_i ← f̂_k(x_i)

2. Train final model:
   f̂ ← Train(D)
   ŷ_test ← f̂(x)

3. For each candidate y:
   s_test(y) ← |y - ŷ_test|
   p(y) ← (1 + Σ_i 𝟙[s_i ≤ s_test(y)]) / (n + 1)

4. Prediction set:
   C(x) ← {y : p(y) > α}

5. Return C(x)
```

### Jackknife+-after-Bootstrap (for Random Forests)

Efficient method leveraging out-of-bag predictions:

```
Algorithm: Jackknife+-after-Bootstrap
Input: Random Forest f̂ with OOB predictions, test point x, level α

1. Compute OOB residuals:
   For i = 1 to n:
       ŷ_i^OOB ← OOB_prediction(f̂, x_i)
       s_i ← |y_i - ŷ_i^OOB|

2. Compute test predictions from each tree:
   For t = 1 to T:
       ŷ_test^t ← tree_t.predict(x)

3. For each tree t, compute leave-one-out residuals:
   r_i^t ← y_i - ŷ_i^{-t}  (prediction without tree t)

4. Construct interval:
   Lower ← Quantile_α/2({ŷ_test^t - max_i(r_i^t)})
   Upper ← Quantile_{1-α/2}({ŷ_test^t + max_i(r_i^t)})

5. Return [Lower, Upper]
```

---

## Conformal Prediction for Random Forests

### Using Out-of-Bag Predictions

Random forests provide natural calibration via **out-of-bag (OOB)** predictions:

```
OOB-based Conformal Prediction
──────────────────────────────

Bootstrap Sample 1: [1, 3, 3, 5, 7]  → Tree 1
Bootstrap Sample 2: [2, 2, 4, 6, 7]  → Tree 2
Bootstrap Sample 3: [1, 4, 5, 5, 6]  → Tree 3

For point i=2:
  - NOT in Sample 1 → Use Tree 1 for OOB prediction
  - IN Sample 2 → Skip Tree 2
  - NOT in Sample 3 → Use Tree 3 for OOB prediction

OOB_pred(2) = average(Tree 1, Tree 3)
OOB_residual(2) = |y_2 - OOB_pred(2)|
```

### Advantages for Forests

| Advantage           | Description                                     |
| ------------------- | ----------------------------------------------- |
| No data splitting   | Uses all data for both training and calibration |
| Natural calibration | OOB errors approximate leave-one-out            |
| Efficiency          | Single training pass                            |

---

## Classification: Prediction Sets

### Conformal Classification

For classification, conformal prediction produces **prediction sets** (subsets
of classes):

```
Algorithm: Conformal Classification
Input: Trained classifier f̂, calibration D_cal, test x, level α

1. Compute calibration scores:
   For i in D_cal:
       s_i ← 1 - f̂(x_i)[y_i]  # 1 minus true class probability

2. Compute threshold:
   q̂ ← Quantile((1-α)(1 + 1/|D_cal|), {s_i})

3. Prediction set:
   C(x) ← {k : f̂(x)[k] ≥ 1 - q̂}

4. Return C(x)
```

### Adaptive Prediction Sets (APS)

More sophisticated scoring for better efficiency:

```python
def adaptive_prediction_set(probs, calibration_scores, alpha):
    """
    Adaptive Prediction Sets (Romano et al., 2020)

    Uses cumulative probability mass instead of single class probability.
    Produces smaller prediction sets on average.
    """
    # Sort classes by probability
    sorted_indices = np.argsort(-probs)
    sorted_probs = probs[sorted_indices]

    # Cumulative sum
    cumsum = np.cumsum(sorted_probs)

    # Find threshold from calibration
    q = np.quantile(calibration_scores, 1 - alpha)

    # Include classes until cumsum exceeds 1 - q
    prediction_set = []
    for i, (idx, cs) in enumerate(zip(sorted_indices, cumsum)):
        prediction_set.append(idx)
        if cs >= 1 - q:
            break

    return prediction_set
```

---

## Practical Considerations

### Calibration Set Size

The calibration set size affects coverage precision:

| Calibration Size | Coverage Precision | Recommendation |
| ---------------- | ------------------ | -------------- |
| 50               | ±7%                | Minimum viable |
| 100              | ±5%                | Acceptable     |
| 500              | ±2%                | Good           |
| 1000+            | ±1%                | Excellent      |

For exact finite-sample coverage of $1 - \alpha$, you need at least
$\lceil 1/\alpha \rceil - 1$ calibration points.

### Conditional vs. Marginal Coverage

| Type            | Guarantee                     | Achievability       |
| --------------- | ----------------------------- | ------------------- | -------------------- |
| **Marginal**    | $P(Y \in C(X)) \geq 1-\alpha$ | Always achievable   |
| **Conditional** | $P(Y \in C(X)                 | X=x) \geq 1-\alpha$ | Generally impossible |

Conformal prediction guarantees **marginal** coverage. For different inputs,
actual coverage may vary.

### Handling Distribution Shift

Standard conformal assumes exchangeability. Under distribution shift:

| Method             | Approach                            |
| ------------------ | ----------------------------------- |
| Weighted conformal | Reweight scores by likelihood ratio |
| Online conformal   | Update calibration over time        |
| Robust conformal   | Use worst-case quantiles            |

---

## Implementation in citrees

### Conformal Regression

```python
from citrees import ConformalForestRegressor

# Create conformal forest
forest = ConformalForestRegressor(
    n_estimators=100,
    alpha=0.10,  # 90% coverage target
    method='jackknife+',  # or 'cv+', 'split'
)

# Fit and calibrate
forest.fit(X_train, y_train)

# Predict with intervals
predictions = forest.predict(X_test)  # Point predictions
intervals = forest.predict_interval(X_test)  # [lower, upper]

# Check coverage
coverage = np.mean(
    (y_test >= intervals[:, 0]) & (y_test <= intervals[:, 1])
)
print(f"Empirical coverage: {coverage:.2%}")  # Should be ≥ 90%
```

### Conformal Classification

```python
from citrees import ConformalForestClassifier

# Create conformal classifier
forest = ConformalForestClassifier(
    n_estimators=100,
    alpha=0.10,  # 90% coverage target
    method='aps',  # Adaptive Prediction Sets
)

# Fit and calibrate
forest.fit(X_train, y_train)

# Predict with sets
predictions = forest.predict(X_test)  # Most likely class
prediction_sets = forest.predict_set(X_test)  # Set of classes

# Example output
# prediction_sets[0] = {2, 5}  # True class is in this set with ≥90% prob
```

### Parameters

| Parameter          | Type  | Default      | Description                             |
| ------------------ | ----- | ------------ | --------------------------------------- |
| `alpha`            | float | 0.10         | Miscoverage rate (1 - coverage)         |
| `method`           | str   | 'jackknife+' | Conformal method                        |
| `calibration_size` | float | 0.2          | Fraction for calibration (split method) |

### Available Methods

| Method         | Description                               | Data Efficiency |
| -------------- | ----------------------------------------- | --------------- |
| `'split'`      | Simple split conformal                    | Low             |
| `'cv+'`        | Cross-validation+                         | High            |
| `'jackknife+'` | Jackknife+ after bootstrap                | High            |
| `'aps'`        | Adaptive prediction sets (classification) | High            |

---

## Comparison with Other Methods

### Conformal vs. Bayesian Uncertainty

| Aspect         | Conformal       | Bayesian              |
| -------------- | --------------- | --------------------- |
| Assumptions    | Exchangeability | Prior + likelihood    |
| Computation    | Fast            | Often expensive       |
| Coverage       | Guaranteed      | Approximate           |
| Interpretation | Frequentist     | Posterior probability |

### Conformal vs. Bootstrap Intervals

| Aspect      | Conformal           | Bootstrap       |
| ----------- | ------------------- | --------------- |
| Assumptions | Exchangeability     | i.i.d.          |
| Coverage    | Finite-sample valid | Asymptotic      |
| Width       | Adaptive            | Fixed quantiles |
| Computation | O(n)                | O(B × n)        |

---

## Recent Advances (2024-2025)

### coverforest Package

[coverforest](https://arxiv.org/abs/2501.14570) (2025) provides efficient
conformal prediction for random forests:

- 2-9× faster than existing implementations
- Supports split conformal, CV+, Jackknife+-after-bootstrap
- Efficient pairwise comparisons for OOB-based methods

### Adaptive Prediction Intervals

[Regression Trees for Fast and Adaptive Prediction Intervals](https://arxiv.org/pdf/2402.07357)
(2024):

- Interpolates between conformal and non-conformal
- Calibrated intervals with theoretical guarantees
- Improved scalability

### Circular Data

[Projected Random Forests and Conformal Prediction of Circular Data](https://arxiv.org/abs/2410.24145)
(2024):

- Extends conformal to circular responses
- Uses OOB dynamics to avoid separate calibration

---

## Configuration Examples

### High Coverage (Conservative)

```python
forest = ConformalForestRegressor(
    n_estimators=500,
    alpha=0.05,  # 95% coverage
    method='cv+',  # More data-efficient
)
```

### Efficient (Fast)

```python
forest = ConformalForestRegressor(
    n_estimators=100,
    alpha=0.10,  # 90% coverage
    method='split',  # Fastest
    calibration_size=0.2,
)
```

### Adaptive Width

```python
# For heteroscedastic data (varying noise)
forest = ConformalForestRegressor(
    n_estimators=200,
    alpha=0.10,
    method='jackknife+',
    normalize_scores=True,  # Divide by local std estimate
)
```

---

## References

1. **Foundational Book**:
   [Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.](http://www.alrw.net/)

2. **Tutorial**:
   [Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.](https://arxiv.org/abs/2107.07511)

3. **CV+**:
   [Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021). Predictive Inference with the Jackknife+. Annals of Statistics.](https://arxiv.org/abs/1905.02928)

4. **Random Forests**:
   [Johansson, U., Boström, H., Löfström, T., & Linusson, H. (2014). Regression Conformal Prediction with Random Forests. Machine Learning, 97(1-2), 155-176.](https://link.springer.com/article/10.1007/s10994-014-5453-0)

5. **coverforest**:
   [Brayford, J., et al. (2025). coverforest: Conformal Predictions with Random Forest in Python. Neurocomputing.](https://arxiv.org/abs/2501.14570)

6. **Survey**:
   [Fontana, M., et al. (2024). Conformal Prediction: A Unified Review of Theory and New Challenges. ACM Computing Surveys.](https://dl.acm.org/doi/10.1145/3736575)

7. **Adaptive Sets**:
   [Romano, Y., Sesia, M., & Candès, E. (2020). Classification with Valid and Adaptive Coverage. NeurIPS.](https://arxiv.org/abs/2006.02544)
