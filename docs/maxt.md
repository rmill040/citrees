# The standardized max-type split test (`threshold_test="maxt"`)

This page states the statistic, the validity argument, and the sources, so that
a statistician can check the construction without reading the kernels.

## Setting

At a node, one feature `x` has been selected and `K` candidate thresholds
`t_1, ..., t_K` are functions of `x` alone. For labels `y` and threshold `j`,
let `I_j(y)` be the weighted child impurity of the split `x <= t_j` (Gini or
entropy for classification, mean squared or mean absolute error for regression).
Candidates that leave one side empty are excluded.

The default test (`threshold_test="bonferroni"`) runs one permutation test per
candidate at level `alpha / K` with a `K`-scaled permutation budget. It is valid
by the union bound and strictly conservative, because adjacent thresholds induce
nearly identical splits; the effective number of tests is far below `K`.

## Statistic

Draw `B` independent uniformly random permutations `pi_1, ..., pi_B` of the
labels and set `pi_0` to the identity. Form the `(B + 1) x K` matrix

    S[b, j] = I_j(y_{pi_b}),   b = 0, ..., B,   j = 1, ..., K.

For each column compute the mean `mu_j` and standard deviation `sigma_j` of its
`B + 1` entries, and standardize:

    Z[b, j] = (S[b, j] - mu_j) / sigma_j.

Columns with `sigma_j = 0` carry no information and are dropped. The statistic
of row `b` is its minimum standardized impurity,

    T_b = min_j Z[b, j],

and the p-value is the inclusive rank of the observed row,

    p = (1 + #{b >= 1 : T_b <= T_0}) / (B + 1).

The chosen threshold is `argmin_j Z[0, j]`, first index on ties. The node splits
when `p < alpha`, with `alpha` unadjusted.

This is the max-T statistic of Westfall and Young (1993), applied to a minimum
because smaller impurity is more extreme. It is the same standardization that
conditional inference trees apply to their split statistics (Hothorn, Hornik and
Zeileis, 2006, Section 3), where each candidate statistic is centered by its
conditional expectation and scaled by its conditional standard deviation before
the maximum is taken.

## Why standardize

Under the null, candidates differ in variability. A threshold at the tail of `x`
isolates few observations and its impurity varies widely from permutation to
permutation; a threshold near the median varies little. The minimum of the raw
impurities is therefore dominated by the widest columns, and on a heavy tailed
target the "best" raw split isolates a handful of extreme values. This was
observed on a regression sample where the raw-minimum version produced negative
held-out R^2 while the Bonferroni default was positive. Standardizing each
column puts the candidates on one scale, so the minimum reflects how unusual a
split is relative to its own null distribution.

## Validity

Assume the labels are exchangeable given `x`: for every permutation `pi`, `y_pi`
has the same conditional distribution as `y`. This is the standard permutation
null and is what the other citrees permutation tests assume.

1. The rows of `S` are exchangeable. Row 0 is the observed labels and rows
   `1, ..., B` are independent uniform permutations of them; under the
   assumption, relabeling any row as "observed" leaves the joint law unchanged.
2. Column standardization preserves exchangeability of the rows. Each `mu_j` and
   `sigma_j` is a symmetric function of the `B + 1` entries of column `j`, so
   the map from `S` to `Z` commutes with any permutation of the rows. The
   moments must include row 0 for this to hold; using only the permuted rows
   would break the symmetry.
3. The row minimum is one fixed function applied to every row, so
   `T_0, ..., T_B` are exchangeable.
4. For exchangeable `T_0, ..., T_B`, the inclusive rank of `T_0` satisfies
   `P(p <= alpha) <= floor(alpha (B + 1)) / (B + 1) <= alpha` for every `alpha`,
   ties included (at most `r` entries can have inclusive rank at most `r`).
   Counting ties with `<=` is what makes this hold; the `+ 1` terms are the
   identity permutation and are required (Phipson and Smyth, 2010).

Under the node's global null the probability of declaring any split is at most
`alpha`, whatever the dependence among the `K` thresholds. This is weak
familywise control. The test is conservative under ties rather than exact.
Strong familywise control, per-threshold rejections when some thresholds do
carry signal, is not claimed and is not needed: the node makes one decision.

## Adaptive stopping

With `early_stopping_splitter="adaptive"` the permutations are drawn in batches
of 32 and, after each batch, the column moments and the extreme count are
recomputed from all rows drawn so far. The test stops when a
`Beta(1 + extreme, 1 + m - extreme)` posterior puts probability at least
`confidence` on either side of `alpha`. As for the other adaptive kernels in
citrees, the returned value is a stopping-time estimate rather than a
fixed-budget p-value. Exact recursion and simulation put the null rejection rate
at 0.0496 to 0.0498 for the default confidence 0.95, which is why values below
0.95 are refused.

## Cost

The Bonferroni default evaluates about `K^2 / alpha` impurities per node. The
max-type test evaluates about `K / alpha`. On a 3,597 row cohort with 256
candidate thresholds and no early stopping, one tree took 341 seconds under the
default and 4.7 seconds under the max-type test; with adaptive stopping the two
were within a factor of 1.3.

## References

- Westfall, P. H. and Young, S. S. (1993). _Resampling-Based Multiple Testing:
  Examples and Methods for p-Value Adjustment._ Wiley. Chapter 2, the max-T
  procedure.
- Hothorn, T., Hornik, K. and Zeileis, A. (2006). Unbiased recursive
  partitioning: a conditional inference framework. _Journal of Computational and
  Graphical Statistics_ 15(3), 651-674. Section 3, standardized split
  statistics.
- Dudoit, S., Shaffer, J. P. and Boldrick, J. C. (2003). Multiple hypothesis
  testing in microarray experiments. _Statistical Science_ 18(1), 71-103.
  Comparison of max-T and min-P.
- Ge, Y., Dudoit, S. and Speed, T. P. (2003). Resampling-based multiple testing
  for microarray data analysis. _Test_ 12(1), 1-77. Algorithms.
- Phipson, B. and Smyth, G. K. (2010). Permutation p-values should never be
  zero. _Statistical Applications in Genetics and Molecular Biology_ 9(1), 39.
