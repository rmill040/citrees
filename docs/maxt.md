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

This is a single-step max-T style statistic in the sense of Westfall and Young
(1993), applied to a minimum because smaller impurity is more extreme. Two
differences from the textbook procedure should be stated plainly. Only the
single-step global stage is used: the test returns one p-value for the node and
one threshold, not per-threshold adjusted p-values, and there is no step-down.
And the candidate moments are estimated from the resampled rows themselves
rather than taken from a pivotal statistic; that empirical null centering and
scaling is the device of Pollard and van der Laan (2004). The construction has
the same standardize-then-optimize shape as the split statistics of conditional
inference trees (Hothorn, Hornik and Zeileis, 2006, Section 3), with two
substitutions: their statistic is linear and is standardized by its exact
conditional moments from Strasser and Weber (1999), whereas impurity is not
linear and has no closed-form moments, so Monte Carlo moments from the `B + 1`
rows take their place.

## Why standardize

Under the null, candidates differ in variability. A threshold at the tail of `x`
isolates few observations and its impurity varies widely from permutation to
permutation; a threshold near the median varies little. The minimum of the raw
impurities is therefore dominated by the widest columns, and on a heavy tailed
target the "best" raw split isolates a handful of extreme values. Standardizing
each column puts the candidates on one scale, so the minimum reflects how
unusual a split is relative to its own null distribution.

Where the chosen threshold lands is an empirical property, not a consequence of
the validity argument below, which speaks only to the decision to split. Two
independent simulations with a planted step at zero and Student t noise with 1.5
degrees of freedom (n = 400, 64 candidate thresholds) found the standardized
test placing the split within 0.6 of the step in 193 of 200 and 77 of 79
significant replicates, against 166 of 200 and 62 of 80 for the Bonferroni
default, whose misses were tail thresholds isolating a few extreme values.

## Validity

Assume the labels are exchangeable given `x`: for every permutation `pi`, `y_pi`
has the same conditional distribution as `y`. This is the standard permutation
null and is what the other citrees permutation tests assume. The argument treats
`x` and the thresholds as fixed. In a tree the feature reaching this test was
chosen using the same labels, so the p-value is a nodewise statement given the
selected feature, not a selective-inference p-value; the same is true of the
Bonferroni default, and the tree's protection against spurious splits comes from
the feature gate that precedes it.

1. The rows of `S` are exchangeable. Row 0 is the observed labels and rows
   `1, ..., B` are independent uniform draws from the permutation group with the
   identity adjoined; under the assumption, relabeling any row as "observed"
   leaves the joint law unchanged (Hemerik and Goeman, 2018). The rows are
   exchangeable, not independent: they share one label vector. In the
   implementation the draws are seeded consecutively from a Mersenne Twister, an
   idealization of independent draws shared with every other permutation test in
   the package.
2. Column standardization preserves exchangeability of the rows. Each `mu_j` and
   `sigma_j` is a symmetric function of the `B + 1` entries of column `j`, so
   the map from `S` to `Z` commutes with any permutation of the rows. The
   moments must include row 0 for this to hold. Estimating them from the
   permuted rows only would single out row 0: its standardized values would be
   systematically more extreme than those of the permuted rows, so that variant
   is anti-conservative, not merely asymmetric. Candidates with an empty side
   are excluded by a rule that depends on `x` and the thresholds alone, so the
   excluded set is the same in every row. The standard deviation uses the
   population form, dividing by `B + 1`.
3. The row minimum is one fixed function applied to every row, so
   `T_0, ..., T_B` are exchangeable.
4. For exchangeable `T_0, ..., T_B`, the inclusive rank of `T_0` satisfies
   `P(p <= alpha) <= floor(alpha (B + 1)) / (B + 1) <= alpha` for every `alpha`,
   ties included (at most `r` entries can have inclusive rank at most `r`).
   Counting ties with `<=` is what makes this hold; the `+ 1` terms are the
   identity permutation and are required (Phipson and Smyth, 2010).

Under the node's global null the probability of declaring any split is at most
`alpha`, whatever the dependence among the `K` thresholds. This is weak
familywise control. The test is conservative under ties rather than exact, and
the node uses the strict rule `p < alpha`, which is more conservative still.
When every candidate is excluded the test returns `p = 1` and no threshold.
Strong familywise control, per-threshold rejections when some thresholds do
carry signal, is not claimed and is not needed: the node makes one decision.

## Adaptive stopping

With `early_stopping_splitter="adaptive"` (the `"simple"` setting is mapped to
the same rule for this test) the permutations are drawn in batches of 32 and,
after each batch, the column moments, the row minima and the extreme count are
recomputed from all rows drawn so far. The test stops when a
`Beta(1 + extreme, 1 + m - extreme)` posterior puts probability at least
`confidence` on either side of `alpha`. Because the moments are recomputed, the
standardized observed value and the chosen threshold can move as the budget
grows. As for the other adaptive kernels in citrees, the returned value is a
stopping-time estimate rather than a fixed-budget p-value, and no level theorem
is claimed for it. The evidence is simulation: across three independent runs of
2,000 to 10,000 nulls per cell, at 2, 16 and 64 thresholds, for Gini and mean
squared error, the adaptive rejection rate at a nominal 0.05 ranged from 0.035
to 0.054, and a 10,000-null stress run at 64 thresholds gave 0.043 and 0.047.
Confidence values below 0.95 are refused because the rule's measured null rate
exceeds the nominal level there.

## Budget and cost

In the Bonferroni default each candidate's test must resolve a p-value of
`alpha / K`, so the permutation budget is scaled to about `K / alpha` per
candidate. That scaling is a precondition for the test to be able to reject at
all, not an optimization: with an unscaled budget the smallest attainable
p-value exceeds `alpha / K` and the branch never splits. The node therefore
evaluates about `K^2 / alpha` impurities.

The max-type test only has to resolve `alpha`, so `1 / alpha` permutations
suffice for validity. A larger budget sharpens the ranking of the candidates and
the resolution of the p-value. With the minimum budget of 20 permutations at
`alpha = 0.05` the test is markedly conservative (null rate about 0.03) and
loses power relative to a budget of 999. For that reason, when the resample
setting is a named rule rather than an explicit integer, the exhaustive max-type
test uses 999 permutations, the Monte Carlo budget partykit uses for its own
maximally selected statistics. The node then evaluates `999 K` impurities, about
`K / 50` of the Bonferroni cost at `alpha = 0.05`: 20 times cheaper at `K = 64`
and 5 times at `K = 256`. An explicit integer budget is honored as given. With
adaptive stopping the budget is chosen by the stopping rule and the two tests
cost about the same.

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
- Hemerik, J. and Goeman, J. J. (2018). Exact testing with random permutations.
  _TEST_ 27(4), 811-825. Validity with the identity adjoined to random draws.
- Pollard, K. S. and van der Laan, M. J. (2004). Choice of a null distribution
  in resampling-based multiple testing. _Journal of Statistical Planning and
  Inference_ 125(1-2), 85-100. Empirical null centering and scaling.
- Strasser, H. and Weber, C. (1999). On the asymptotic theory of permutation
  statistics. _Mathematical Methods of Statistics_ 8, 220-250. Exact conditional
  moments of linear statistics.
