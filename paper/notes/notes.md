# citrees: Writing Notes (Theory-First)

This file is an internal staging document. The manuscript source of truth is
`paper/arxiv/` (LaTeX). Use these notes to develop/park ideas and then migrate
stable content into the LaTeX sections/appendices.

Project-level writing workflow:

- Claims tracker: `paper/CLAIMS_INDEX.md`
- Proof QA checklist: `paper/WRITING_CHECKLIST.md`
- Scripts → figures/tables → claims map: `paper/notes/figures_plan.md`

**Scope.** The focus here is the validity of the permutation p-values used for
_Stage A (feature screening)_ and the resulting finite-sample error-control
statements (Bonferroni/root-level). Wherever a statement is only heuristic or
requires additional selective-inference machinery, it is labeled as such. All
paper-facing p-value guarantees are for **fixed-$B$** permutation tests at
**fixed nodes** where the tested families (and the resample budget $B$) are
**label-independent functions of** $(X_t,U)$; internal nodes are adaptive, so we
do not present classical p-values there.

---

## 0. Manuscript skeleton (WIP)

This file is the evolving manuscript draft. The technical pieces live in:

- Appendix A: permutation-test theory + calibration checks
- Appendix B: optional feature-muting analysis (heuristic)
- Appendix C: open TODOs

### 0.1 Proposed paper structure

1. **Introduction**
   - Problem: feature selection is unstable/bias-prone; “high-cardinality noise”
     bias in greedy trees is a canonical failure mode.
   - Goal: rank features with controlled false discoveries + strong empirical
     performance, at scale.
   - Contributions (draft):
     1. a Python implementation of conditional inference trees/forests focused
        on permutation-test screening,
     2. clean finite-sample Stage A/root guarantees in fixed-$B$ mode (with
        explicit scope limits),
     3. an idealized lemma formalizing CART’s high-cardinality selection bias
        mechanism,
     4. a benchmark suite across synthetic ground-truth datasets + real tabular
        datasets (rank → downstream eval).

2. **Method**
   - Two-stage tree split: Stage A (feature) → Stage B (threshold).
   - Filter / permutation / embedding / wrapper methods evaluated (protocol is
     two-stage: rank → downstream eval).
   - Implementation details that matter for claims: Bonferroni in Stage A,
     fixed-$B$ vs adaptive stopping, tie handling.

3. **Theory (clean claims only)**
   - Theorem 1 (+1 Monte Carlo p-values are super-uniform under
     exchangeability).
   - Stage A Bonferroni and root-level “tree splits” guarantee.
   - Multi-selector max-statistic validity.
   - Optional appendix note: if discussing early stopping, state only what is
     controlled by the stopping _criterion_ and avoid treating returned values
     as fixed-$B$ p-values at a stopping time.

4. **Experiments**
   - Benchmarks: synthetic classification (ground truth), real classification,
     real regression.
   - Metrics: precision/recall/F1@k (synthetic), downstream performance vs $k$,
     runtime.
   - Statistical comparisons: paired tests / critical difference diagrams (if we
     keep them).

5. **Discussion + limitations**
   - Exchangeability assumptions; adaptive internal nodes; Stage B
     post-selection; sequential p-values caveat.

### 0.2 Benchmark suite (what to show)

**Synthetic (classification, ground truth; 169 datasets across families)**
Main-text candidates: bias (high-cardinality noise), nonlinear, correlated
blocks, high-dimensional scaling, weak-signal.

**Real datasets**

- Classification: focus on $p\gg n$ (e.g.,
  arcene/dexter/dorothea/gisette/madelon + gene-expression sets), plus a small
  number of “standard” tabular datasets for sanity checks.
- Regression: include all available datasets in `paper/data/regression/real/`.

**Methods compared (keep full list in appendix; highlight a core subset in main
text)** Filters (`mc/mi/rdc`, `pc/dc/rdc`, `mrmr`), permutation filters
(`ptest_*`), embeddings (`cit/cif`, `rf/et`, `xgb/lgbm/cat`), wrappers
(`boruta`, `pi/cpi`, `shap`, `rfe`).

### 0.3 Figures/tables plan (draft)

Main text (candidate set):

- Fig: selection bias under null
  (`paper/results/figures/selection_bias_demo.png`)
- Fig: synthetic performance slices
  (`paper/results/figures/signal_strength.png`,
  `paper/results/figures/high_dimensional.png`,
  `paper/results/figures/correlated_features.png`,
  `paper/results/figures/redundant_features.png`)
- Fig: controlled selection-bias sanity checks (synthetic toy data):
  `paper/results/figures/feature_selection_clf.png`,
  `paper/results/figures/regression_comparison.png`
- Fig: real-data downstream comparison (from Stage 2 metrics; generate from
  S3-backed artifacts)
- Fig: runtime (`paper/results/figures/timing_speedup.png` and/or
  `paper/results/figures/timing_bars.png`)

Appendix / supplement:

- Fixed-$B$ p-value calibration
  (`paper/results/figures/fixedB_pvalue_calibration.png`)
- Sequential stopping calibration
  (`paper/results/figures/sequential_stopping_calibration.png`)
- Any feature-muting analysis (Appendix B +
  `paper/results/figures/muting_*.png`) if we keep it at all.

## Abstract

Greedy decision trees and tree ensembles often select split variables by
optimizing impurity reduction over many candidate thresholds. This creates a
multiple-comparisons effect: under the null, features with more candidate split
points are more likely to appear optimal, producing “high-cardinality” selection
bias and unstable feature rankings. Conditional inference trees address this by
separating variable selection from split-point optimization via
permutation-based hypothesis testing.

We present `citrees`, a Python library implementing conditional inference trees
and forests with Monte Carlo permutation p-values using the Phipson–Smyth (+1)
correction. Our paper focuses on the inferentially clean part of the algorithm:
Stage A feature screening at a fixed node (especially the root). Under standard
exchangeability assumptions, the +1 Monte Carlo permutation p-value is
super-uniform, and with Bonferroni correction we obtain finite-sample bounds on
the probability of splitting under the global null and on splitting on any
particular null feature. For motivation, we also give a simple idealized lemma
showing that CART-style greedy selection chooses features in proportion to their
number of candidate splits under an exchangeable null model. We benchmark
feature rankings produced by `citrees` and baselines on synthetic ground-truth
datasets and real tabular datasets using a two-stage rank-then-evaluate
protocol. We emphasize scope: internal-node tests and Stage B split tests are
post-selection/adaptive and are treated as algorithmic stopping statistics
unless additional selective-inference machinery is used.

## 1. Introduction

Feature selection is a central primitive in modern tabular machine learning:
practitioners want rankings that are predictively useful, stable across
resamples, and interpretable. These goals are especially sharp in regimes with
many features and comparatively few samples ($p\gg n$), where naive selection
rules can overfit to noise.

Decision trees provide an appealing embedding feature selector: the split
structure yields an implicit ranking, and tree-based ensembles scale well in
practice. However, classical CART-style trees choose split variables by
maximizing a data-dependent objective over many candidate thresholds. This
creates a systematic selection bias: even under a global null where all features
are unrelated to the response, features with more candidate split points are
more likely to produce extreme impurity improvements by chance. This bias has
been documented for variable importance in random forests and related models
(e.g., Strobl et al., 2007), and it appears immediately at the root in one-split
trees.

**Terminology (high-cardinality selection bias).** At a fixed node, each feature
$j$ typically contributes $m_j$ candidate thresholds (e.g., $m_j=\ell_j-1$
midpoints for a feature with $\ell_j$ unique values). We use “high-cardinality
selection bias” to mean the phenomenon that, under a global null where all
features are noise, a greedy threshold-optimizing split rule can assign larger
split probability to features with larger $m_j$ purely because they offer more
candidate comparisons. Lemma 3b formalizes this mechanism under an
exchangeable/tie-free null idealization.

Conditional inference trees (Hothorn et al., 2006) mitigate this issue by
replacing greedy optimization with hypothesis testing: at each node, first
screen features for association with the response (Stage A), and only if a
feature passes a significance threshold does the algorithm proceed to choose a
split threshold (Stage B). The test-based Stage A step is the natural place to
attach finite-sample error-control statements.

This paper contributes:

1. A Python implementation of conditional inference trees and forests
   (`citrees`) that uses Monte Carlo permutation tests with the Phipson–Smyth
   (+1) correction as the inferential backbone for Stage A screening.
2. A clean set of finite-sample results for fixed-$B$ Stage A screening at a
   fixed node (especially the root): +1 permutation p-values are super-uniform
   under exchangeability, and Bonferroni correction yields explicit bounds on
   false splitting under the global null.
3. A simple idealized lemma formalizing the CART selection-bias mechanism: under
   an exchangeable/tie-free null model, greedy CART selects features in
   proportion to their number of candidate thresholds.
4. A benchmark suite evaluating feature rankings across synthetic ground-truth
   datasets and real datasets under a two-stage rank → downstream evaluation
   protocol.
5. A reproducibility/audit trail for paper-facing claims: Appendix F records a
   claim tracker (proved vs cited vs empirical), and
   `paper/notes/figures_plan.md` maps each figure/table to its generating script
   and artifacts.

**Scope and limitations.** Our theoretical guarantees are intentionally scoped:
we treat Stage A p-values as classical fixed-$B$ permutation p-values at a fixed
node (with root-level consequences). We do not present Stage B p-values or
internal-node p-values in an adaptively grown tree as classical post-selection
p-values without additional machinery; they are treated as algorithmic
split-validation/stopping statistics.

In particular, under a root-level global null and fixed-$B$ Stage A testing with
Bonferroni correction, we can bound the probability that the fitted tree
contains any split by $\alpha_{\text{sel}}$ (Appendix A.3).

### 1.1 Related work (brief)

**Greedy trees and selection bias.** CART (Breiman et al., 1984) selects split
variables by optimizing impurity decrease over candidate thresholds, which
induces a multiple-comparisons effect and favors high-cardinality noise features
under the null. This phenomenon is closely related to known biases in tree-based
variable importance measures (Strobl et al., 2007).

**Conditional inference trees.** The conditional inference framework of Hothorn
et al. (2006) separates variable selection from split-point selection using
permutation-based hypothesis tests, providing a principled alternative to greedy
impurity optimization.

Widely used implementations exist in R (e.g., `ctree`/`cforest` in the
`party`/`partykit` lineage). The goal of `citrees` is not to change the
underlying conditional inference idea, but to provide a Python implementation
with a clear fixed-$B$ Monte Carlo permutation-testing foundation for Stage A
screening and with an explicit audit trail of what is and is not claimed
inferentially in an adaptively grown tree.

**Permutation tests and multiplicity.** Our fixed-$B$ results rely on standard
exchangeability-based permutation test validity combined with the Phipson–Smyth
(+1) correction for Monte Carlo permutation p-values (Phipson & Smyth, 2010) and
family-wise error control via Bonferroni correction.

## 2. Methods (skeleton)

### 2.1 citrees (as an embedding feature selector)

At each node $t$:

- **Stage A (feature screening):** test each feature $j\in F_t$ via a
  permutation test; split only if at least one feature is significant after
  multiplicity correction.
- **Stage B (threshold screening):** for the selected feature, test candidate
  thresholds and choose a split.

In the benchmark suite, tree/forest models (including citrees) act as
**embedding feature selectors** by producing a feature ranking (e.g., split
counts / impurity-based importance), which is then evaluated downstream at
top-$k$.

#### 2.1.1 Node expansion (Stage A → Stage B)

We write the nodewise algorithm in a form that matches the implementation and
separates what is _tested_ from what is _optimized_:

**Stage A (screening / gatekeeper).** For each candidate feature $j\in F_t$,
test the null

$$
H^{\text{sel}}_{t,j}: X_{t,j}\perp Y_t
$$

using a selector statistic $T^{\text{sel}}_j(X_{t,j},Y_t)$ (right-tail: larger
means stronger association). Compute permutation p-values $p_{t,j}$ and select

$$
j_t^\star := \arg\min_{j\in F_t} p_{t,j},\qquad p_t^\star := \min_{j\in F_t} p_{t,j}.
$$

If Bonferroni is enabled, split only if $p_t^\star < \alpha_{\text{sel}}/|F_t|$;
otherwise require $p_t^\star < \alpha_{\text{sel}}$.

**Stage B (threshold screening).** Given $j_t^\star$, construct a finite
candidate set of thresholds $C_{t,j_t^\star}$ (midpoints of unique values,
optionally subsampled **without replacement**). Thresholds that would violate
`min_samples_leaf` are filtered out before testing. For each
$c\in C_{t,j_t^\star}$, compute a split-quality statistic

$$
T^{\text{split}}_{t,c} := I(Y^L_{t,c}) + I(Y^R_{t,c}),
$$

where $I(\cdot)$ is a node impurity functional (Gini/entropy for classification;
MSE/MAE for regression) and the left/right children are induced by
$X_{t,j_t^\star}\le c$. Smaller values indicate better splits, so Stage B uses a
left-tail permutation test. With Bonferroni enabled, accept a split only if
$\min_{c\in C_{t,j_t^\star}} p_{t,c} < \alpha_{\text{split}}/|C_{t,j_t^\star}|$.
If early stopping is enabled and threshold scanning is disabled, the
implementation randomizes the threshold order to reduce ordering bias. If no
valid thresholds remain after filtering, the node becomes a leaf.

**Stage A selector statistics (what $T^{\text{sel}}$ is).** citrees treats
$T^{\text{sel}}$ as a generic association score whose null is calibrated by
permutation. The theory in Appendix A depends only on exchangeability under the
permutation scheme; the specific selector affects power.

In this paper’s experiments we use:

- **Classification:** `mc` (correlation ratio / ANOVA-style association), `mi`
  (estimated mutual information), `rdc` (randomized dependence coefficient).
- **Regression:** `pc` (absolute Pearson correlation), `dc` (distance
  correlation), `rdc` (randomized dependence coefficient).

**Remark (links to standard test statistics).** For intuition, `mc` is the
correlation ratio $\eta=\sqrt{\mathrm{SSB}/\mathrm{SST}}$, which is a strictly
increasing transform of the one-way ANOVA $F$ statistic at fixed $(n,g)$; `pc`
is $|\rho|$, which is a strictly increasing transform of the usual correlation
$t$ statistic at fixed $n$. Since fixed-$B$ permutation p-values depend only on
the ordering of permuted statistics, strictly monotone transforms give identical
permutation p-values (see Appendix A.2.4 and Appendix A.6).

Definitions and implementation-aligned formulas are collected in Appendix H and
Appendix A.6.

**Structural constraints.** After Stage B, the implementation additionally
enforces constraints such as `min_samples_leaf` and `min_impurity_decrease`.
Importantly, `min_impurity_decrease` and feature importances use the standard
_weighted_ impurity decrease

$$
\Delta I := I(Y_t)-\Big(\tfrac{n_L}{n_t}I(Y^L)+\tfrac{n_R}{n_t}I(Y^R)\Big),
$$

which is separate from the (unweighted) Stage B permutation-test statistic.

Pseudocode for these steps is provided in Appendix G.

#### 2.1.2 Permutation p-values (+1 correction)

citrees uses Monte Carlo permutation tests with the Phipson–Smyth (+1)
correction. For a statistic $T$ and $B$ permutations, the +1 p-value takes the
form

$$
p = \frac{1 + \sum_{b=1}^B \mathbf{1}\{T_b \text{ is at least as extreme as } T_0\}}{B+1},
$$

using a right-tail convention for Stage A and a left-tail convention for Stage
B.

**Non-exchangeable data (restricted permutations required).** If observations
are clustered, longitudinal, or time-series dependent, simple label permutations
are invalid because exchangeability fails. Any inferential interpretation then
requires an appropriate **restricted permutation scheme** (e.g.,
block/cluster/stratified permutations) matched to the dependence structure.
citrees does not implement such schemes by default, so p-values should not be
interpreted without them in these settings.

Examples include: permuting labels **within clusters**, permuting **within time
blocks**, or permuting **within strata** defined by grouping variables; the
exact restriction should mirror the dependence structure.

### 2.1.3 Feature rankings from trees and forests

For trees and forests, we use mean decrease in impurity (MDI) aggregated over
splits:

$$
\mathrm{Importance}_j := \sum_{\text{nodes }v\text{ that split on }j} \Delta I_v,
$$

normalized to sum to 1 per fitted estimator. This yields a feature ranking that
can be evaluated downstream at top-$k$.

### 2.2 Methods compared (overview)

We compare four families of feature selectors (full lists and exact configs live
in `paper/docs/README.md`):

- **Filters:** correlation / MI / RDC / mRMR
- **Permutation-test filters:** `ptest_*` methods that convert filter scores
  into permutation p-values
- **Embeddings:** tree ensembles (`cit/cif`, `rf/et`, `xgb/lgbm/cat`)
- **Wrappers:** Boruta, permutation importance (PI/CPI), SHAP, RFE

### 2.3 Benchmark protocol (two-stage architecture)

The experiments follow the repo’s two-stage pipeline (`paper/docs/README.md`):

1. **Stage 1 (selection):** each method outputs a feature ranking per
   dataset/seed/fold.
2. **Stage 2 (evaluation):** train downstream models on top-$k$ features for
   $k\in\{5,10,25,50,100,\text{all}\}$.

This decouples “how we rank features” from “how selected features perform in a
standard model”.

## 3. Theory (main results and scope)

This section states the main theoretical claims in the strongest defensible
form; proofs and technical details are in Appendix A.

**Summary (the theorem chain used in the paper).**

1. (+1 p-values) At a fixed node in fixed-$B$ mode, the +1 Monte Carlo
   permutation p-value is super-uniform under exchangeability (Appendix A.2,
   Theorem 1).
2. (Stage A control) Bonferroni yields a nodewise family-wise bound on Stage A
   rejection, and a per-feature bound on splitting on any particular null
   feature that does not depend on its number of candidate thresholds (Appendix
   A.3, Proposition 3 and Proposition 3a).
3. (Root consequences) The tree can only split if the root passes Stage A,
   giving a global root-level bound on “any split at all” (Appendix A.3,
   Proposition 4).
4. (Forests) Root-level bounds lift to forests by linearity/union bounds,
   yielding bounds on how often null features appear at the root and on expected
   root importance contributions (Appendix A.3, Corollary 4a).

**Box 1 (Notation & assumptions for theory statements).**

- **Fixed node (A0.1).** Results are stated for a fixed node; in practice this
  is cleanest at the root.
- **Conditional exchangeability (A0.2).** Under the null, labels are
  exchangeable given $(X_t,U)$.
- **Label-independent tested family (A0.3).** $F_t$, candidate thresholds, and
  $B$ depend only on $(X_t,U)$.
- **Fixed-$B$ p-values (A0.4).** No optional stopping; early stopping is treated
  as a heuristic.
- **Tie handling (A0.5).** Ties are counted against the null or broken at
  random.
- **Idealized symmetry (A0.6).** Used only for CART-bias lemmas; not a
  data-model claim.
- **Additional conditions.** Bounded impurities (A0.7) and honest-split
  independence (A0.8) are stated where used.
- See Appendix A.1.1 for the formal ledger and Appendix A.1.2 for the
  result-to-assumption map.

### 3.1 Fixed-$B$ Monte Carlo permutation p-values

At a fixed node and for a fixed hypothesis, permutation tests rely on an
exchangeability assumption: under the null, the labels are invariant in law to
permutations (conditional on the covariates treated as fixed by the test). In
fixed-$B$ mode, citrees uses Monte Carlo permutation p-values with the
Phipson–Smyth (+1) correction. Under exchangeability, the resulting +1 p-value
is super-uniform (Appendix A.2, Theorem 1).

### 3.2 Stage A error control at a fixed node (especially the root)

Stage A performs multiple hypothesis tests (one per candidate feature). Under
global null at a fixed node, Bonferroni correction yields a finite-sample
family-wise error bound for Stage A rejection (Appendix A.3, Proposition 3). For
any particular null feature at that node, the probability that the algorithm
splits on that feature is bounded by $\alpha_{\text{sel}}/m_t$ (Appendix A.3,
Proposition 3a). These bounds depend on the size of the tested feature family
and do not depend on a feature’s number of candidate thresholds.

These statements assume the tested feature family $F_t$ (and the resampling
budget $B$) are chosen independently of $Y_t$ under the null, i.e., they are
measurable functions of $(X_t,U)$ at the fixed node. They are **not** guaranteed
at adaptively selected internal nodes where node membership depends on $Y$.

At the root (or any fixed node), these bounds can also be stated **conditional
on $(X_t,U)$**, where $U$ denotes any algorithm randomness used to choose
candidate families (e.g., feature subsampling), because permutation-test
super-uniformity holds conditionally under exchangeability (Appendix A.2,
“random candidate sets are safe if label-independent”).

### 3.3 Root-level global statement: probability the tree splits at all

While internal-node inference is complicated by adaptivity, one global statement
remains clean: the fitted tree can only contain any split if the root passes
Stage A. Under a global null at the root in fixed-$B$ mode with Bonferroni
correction, this yields a root-level bound on the probability the fitted tree
contains any split (Appendix A.3, Proposition 4).

### 3.4 CART selection bias mechanism (idealized)

For motivation, we include a short idealized lemma formalizing the
high-cardinality bias mechanism in greedy CART-style selection: under an
exchangeable/tie-free null idealization, greedy selection over candidate
thresholds chooses features in proportion to their number of candidate splits
(Appendix A.3, Lemma 3b). This is a clean mathematical model of the
multiple-comparisons effect that Stage A screening is designed to avoid.

#### Definition (cardinality bias under a symmetric null)

Fix a node and candidate features $j\in\{1,\dots,p\}$. For each feature $j$, let
$C_j$ be a finite candidate split set with size $m_j:=|C_j|$ (e.g.,
$m_j=\ell_j-1$ midpoints for an ordered feature with $\ell_j$ unique values).
Let an algorithm choose a split variable $\widehat j$ based on some candidate
scores computed from the data.

Under a stylized _symmetric global-null_ idealization in which all candidate
split scores are exchangeable across all $(j,c)$ pairs (so no variable is “truly
better” than another), we say:

- the algorithm is **cardinality-unbiased** if $\widehat j$ is uniform on
  $\{1,\dots,p\}$, and
- the algorithm exhibits **cardinality bias** if $\mathbb{P}(\widehat j=j)$
  increases with $m_j$.

Lemma 3b shows that greedy impurity optimization exhibits cardinality bias in
this idealization, while Lemma 3e shows that Stage A screening is
cardinality-unbiased under an analogous exchangeability idealization on
p-values. The main paper-facing guarantee we use does not require these symmetry
assumptions: Proposition 3a provides a cardinality-free Type I bound for
splitting on any particular null feature at a fixed node.

Before the symmetry lemma, it is useful to state an assumption-light version of
the “many chances” mechanism as a simple union-bound calculation.

#### Lemma 3a (threshold search amplifies null extremes; union bound)

**Assumptions (minimal).** None beyond the union bound; independence is not
required.

Fix a node and a feature $j$ with a finite candidate threshold set $C_j$ of size
$m_j:=|C_j|$. For each $c\in C_j$, let $\Delta_{j,c}$ denote an impurity
decrease (larger is better). Fix $\delta\in\mathbb{R}$ and define the
single-candidate tail probabilities
$p_{c,\delta}:=\mathbb{P}(\Delta_{j,c}\ge \delta)$ (under the global null, for a
fixed $c$). Then without any independence assumptions,

$$
\mathbb{P}\!\left(\max_{c\in C_j}\Delta_{j,c}\ge \delta\right)
=\mathbb{P}\!\left(\bigcup_{c\in C_j}\{\Delta_{j,c}\ge \delta\}\right)
\le \sum_{c\in C_j} p_{c,\delta}
\le m_j\cdot \max_{c\in C_j} p_{c,\delta}.
$$

If additionally $(\Delta_{j,c})_{c\in C_j}$ are independent and identically
distributed with $\mathbb{P}(\Delta_{j,c}\ge \delta)=p_\delta$, then

$$
\mathbb{P}\!\left(\max_{c\in C_j}\Delta_{j,c}\ge \delta\right)=1-(1-p_\delta)^{m_j}.
$$

**Proof.** The first bound is the union bound. Under independence,
$\mathbb{P}(\max_{c\in C_j}\Delta_{j,c}<\delta)=\prod_{c\in C_j}\mathbb{P}(\Delta_{j,c}<\delta)=(1-p_\delta)^{m_j}$,
so taking complements yields the identity. $\square$

**Interpretation.** If “a surprisingly good split” corresponds to a rare event
$\{\Delta_{j,c}\ge\delta\}$ for any fixed threshold $c$, then searching over
$m_j$ thresholds multiplies the chance of seeing such an extreme by a factor on
the order of $m_j$ (for small tail probabilities). This is the precise
mathematical content behind informal statements like “a feature with 1000 unique
values has 999 chances to look good.”

**Remark (categorical splits can be worse than linear in cardinality).** If an
algorithm treats a categorical feature with $L$ unordered levels as allowing all
binary partitions of the levels, then the number of candidate splits is
$2^{L-1}-1$ (non-empty proper subsets, modulo swapping left/right). Many
implementations avoid enumerating all such partitions, but this highlights that
“high-cardinality” can create very large multiple-comparisons effects in greedy
split optimization.

**Concrete categorical example (how to read the exponential count).** If a
categorical feature has $L=10$ unordered levels and all binary partitions are
allowed, then it contributes $2^{10-1}-1=511$ candidate splits at a node.
Compared to a binary feature with $m=1$ candidate split, the idealized Lemma 3b
symmetry model would assign selection probability $511/(511+1)\approx 0.998$ to
the 10-level categorical feature under a global null.

**Worked example (numbers, not a model claim).** If a single candidate threshold
has a $1\%$ chance to exceed some “looks-good” benchmark $\delta$ under the null
($p_\delta=0.01$), then with $m_j=999$ candidates and independence the chance
that at least one candidate exceeds $\delta$ is
$1-(1-0.01)^{999}\approx 1-e^{-9.99}\approx 0.99995$.

Under an additional symmetry idealization (exchangeable Stage A p-values under
the global null), Stage A’s selected feature is uniform among the tested
features (Appendix A.3, Lemma 3e).

In the same root-level global-null setting, Stage A screening yields a
cardinality-free bound: with uniform random feature subsampling (`max_features`)
and Bonferroni correction over the tested subset, the probability that the root
splits on any particular null feature is bounded by $\alpha_{\text{sel}}/p$
(Appendix A.3, Corollary 3d), independent of the feature’s number of candidate
thresholds.

### 3.5 What we do not claim

We do not present internal-node p-values in an adaptively grown tree, or Stage B
split p-values after selecting a feature using the same labels, as classical
post-selection p-values without additional selective-inference machinery
(Appendix A.4).

## 4. Experiments (skeleton)

### 4.1 Benchmarks / datasets

**Synthetic classification (ground truth).** 169 synthetic datasets spanning:
standard, bias (high-cardinality noise), nonlinear, correlated blocks,
redundant, correlated-noise confounders, toeplitz, and weak-signal regimes.
Ground truth is stored in parquet metadata and evaluated via
precision/recall/F1@k.

**Real classification.** Use datasets in `paper/data/classification/real/`,
emphasizing $p\gg n$ feature selection (e.g.,
arcene/dexter/dorothea/gisette/madelon + gene-expression datasets), with a
smaller number of “standard” tabular datasets for sanity checks.

**Real regression.** Use all available datasets in `paper/data/regression/real/`
(coepra1/2/3, comm_violence, community_crime, facebook, imports-85,
residential).

### 4.2 Metrics

**Synthetic:** precision/recall/F1@k on informative features; optionally
“informative+redundant” variants and confounder selection rates (see
`paper/scripts/analysis/synthetic_analysis.py`).

**Real:** downstream performance vs $k$:

- classification: ROC AUC, accuracy, F1 (weighted). (Log loss is not computed in
  the current pipeline.)
- regression: $R^2$/RMSE/MAE (choose a primary metric for the main text)

**Runtime:** wall-clock time for Stage 1 methods + speedups from early stopping
/ parallelism.

### 4.3 Reporting (how we summarize)

Decide what goes in the main text vs appendix:

- **Main text:** a small number of “story” figures that answer one question
  each, plus a single aggregated summary (across synthetic families and across
  real datasets).
- **Appendix:** full method list (all 19) + per-dataset tables/curves and any
  critical-difference diagrams.

### 4.4 Recommended results ordering (to manage the data volume)

1. **Selection-bias sanity check (the “why”)**
   - Fig: `paper/results/figures/selection_bias_demo.png`
   - Optional companion: `paper/results/figures/informative_ratio.png`

2. **Synthetic ground-truth performance (the “does it recover the right
   features?”)**
   - Start with easiest-to-interpret axes:
     - Fig: `paper/results/figures/signal_strength.png` (effect-size sweep)
     - Fig: `paper/results/figures/sample_size.png` (data efficiency)
   - Then show robustness regimes:
     - Fig: `paper/results/figures/high_dimensional.png` ($p$ scaling)
     - Fig: `paper/results/figures/correlated_features.png` (correlation
       blocks/confounding)
     - Fig: `paper/results/figures/redundant_features.png` (redundancy)
   - Keep imbalanced/multiclass/complexity as appendix unless they materially
     change conclusions:
     - `paper/results/figures/imbalanced.png`,
       `paper/results/figures/multiclass.png`,
       `paper/results/figures/complexity_vs_accuracy.png`

3. **Real-data downstream utility (the “does feature selection help?”)**
   - Generate from Stage 2 metrics (S3-backed artifacts); do **not** use toy
     synthetic figures here.
   - Fig names TBD (once the real-data pipeline outputs are finalized).

4. **Runtime / scalability (the “can we afford it?”)**
   - Fig: `paper/results/figures/timing_speedup.png` (headline speedup)
   - Optional: `paper/results/figures/timing_bars.png` (granular breakdown)

5. **Theory calibration checks (appendix)**
   - Fixed-$B$ p-value calibration:
     `paper/results/figures/fixedB_pvalue_calibration.png`
   - Sequential stopping calibration:
     `paper/results/figures/sequential_stopping_calibration.png`

## 5. Discussion and limitations

### 5.1 Conditional inference framing (what our p-values mean)

All p-values in citrees are computed in a **conditional inference** framework:
at a fixed node $t$, the permutation test operates conditionally on the
covariates treated as fixed by the permutation scheme. For example, for a
right-tail Stage A selector statistic $T$, the permutation p-value can be
written as

$$
p \approx \mathbb{P}\!\left(T(X_{t,j}, \pi(Y_t)) \ge T(X_{t,j}, Y_t)\;\middle|\; X_t,U\right),
$$

where $\pi$ is a uniform random permutation and $U$ denotes label-independent
algorithmic randomness (e.g., `max_features`, threshold subsampling, RNG seeds
that define the tested family). Under the corresponding null exchangeability
assumption, the +1 Monte Carlo p-value is super-uniform (Appendix A.2).

This differs from **marginal/asymptotic** p-values from parametric tests (e.g.,
ANOVA $F$ tests or correlation tests): those can be more powerful when their
assumptions hold, but they require stronger distributional conditions.

### 5.2 Scope of guarantees (what we claim)

We keep theoretical claims deliberately narrow:

1. **Fixed-node, fixed-$B$ permutation validity.** Appendix A.2 gives
   finite-sample super-uniformity of the +1 Monte Carlo permutation p-value
   under exchangeability.
2. **Stage A error control at a fixed node (especially the root).** Appendix A.3
   gives Bonferroni-based nodewise and root-level bounds, including a
   cardinality-free per-feature bound (Proposition 3a).
3. **Root-level global statement (“any split implies root rejection”).**
   Proposition 4 bounds the probability that the fitted tree has any split under
   a root-level global null.

### 5.3 What we do not claim (reviewer traps to avoid)

1. **Stage B is post-selection.** Stage B tests thresholds after selecting a
   feature using the same labels. Without additional selective-inference
   machinery or sample splitting, Stage B p-values should be treated as
   algorithmic split-validation statistics, not classical post-selection
   p-values (Appendix A.4).
2. **Internal nodes are adaptive.** In an adaptively grown tree, node membership
   depends on earlier splits that depend on the labels, so exchangeability can
   fail after conditioning on “reaching node $t$”. This means even Stage A
   p-values at internal nodes are **not** classical fixed-family p-values, even
   if $F_t$ is label-independent at that node. This is why the cleanest
   inferential statements are fixed-node or root-level.
3. **Early stopping and muting are computational heuristics.** When enabled,
   early stopping returns a +1 estimate at a data-dependent stopping time, and
   feature muting changes the tested family across nodes. We treat both as
   engineering choices and do not attach fixed-$B$ p-value guarantees to them
   (Appendix A.5 and Appendix B).

### 5.4 CART bias lemmas as motivation (not a data model)

Lemmas 3a–3c are _idealized_ symmetry calculations meant to isolate the
multiple-comparisons mechanism behind high-cardinality selection bias in greedy
impurity optimization. They do not assert that real-world data satisfy
exchangeability across candidate thresholds, only that “optimize over many
candidates” can create systematic selection effects even under null settings.
The main citrees guarantee we use is Proposition 3a, which is assumption-light
and does not depend on a feature’s number of candidate thresholds.

### 5.5 Relationship to existing work (high level)

- **ctree/cforest lineage.** citrees follows the same high-level principle as
  conditional inference trees: separate variable screening (tests) from
  split-point optimization. We emphasize Monte Carlo permutation p-values with
  the +1 correction as the foundation for Stage A screening.
- **Bias corrections for random forests.** A different literature addresses bias
  in _importance measures_ (e.g., conditional permutation importance). citrees
  targets the bias mechanism earlier in the pipeline (variable selection at
  split time), rather than correcting a biased importance score post hoc.
- **Honest estimation.** Honesty exists in the library but is not a core
  contribution of this paper; Appendix D records the clean unbiasedness
  statement under an independent sample split.

## 6. Conclusion

citrees is a Python implementation of conditional inference trees and forests
with Monte Carlo permutation tests as the inferential backbone for Stage A
feature screening. The paper’s clean theoretical statements are finite-sample
and root-/fixed-node scoped: +1 permutation p-values are super-uniform under
exchangeability, and Bonferroni yields explicit nodewise/root bounds that are
independent of a feature’s number of candidate thresholds. These results
motivate and calibrate the empirical feature-selection benchmarks in
`paper/scripts/`.

## 7. Future directions (not required for this paper)

1. **Selective inference for internal nodes and Stage B.** Develop valid
   post-selection inference for adaptively selected nodes and thresholds (likely
   requiring sample splitting or selective-inference machinery).
2. **Tree-wide multiple testing.** Extend root-/fixed-node error control to
   tree-wide error notions (FWER/FDR) using hierarchical testing or other
   sequential multiple testing designs.
3. **Computational acceleration.** Explore batching, shared permutations, and
   GPU acceleration for permutation-test workloads.
4. **More powerful multiplicity corrections.** Replace Bonferroni with step-down
   FWER control (e.g., Holm) or resampling step-down methods (e.g.,
   Westfall–Young) when computation permits; consider weighted procedures when
   defensible.
5. **Multi-selector without “same-scale” restriction.** Instead of taking a max
   over raw selector scores, combine per-selector permutation p-values within a
   feature (e.g., within-feature Bonferroni, or other p-value combination rules
   under explicit dependence assumptions). This could allow including `mi`
   alongside bounded selectors in a multi-selector feature test.
6. **Explicit “paper mode” configuration.** Codify a recommended configuration
   profile for paper-facing claims (fixed $B$, no early stopping/muting, report
   multiplicity and budgets explicitly) to make scope and reproducibility
   review-proof.

---

## Appendix A. Theory and calibration

### A.1 Setup and notation

Let $(X_i, Y_i)_{i=1}^n$ be the training data and consider a fixed node $t$ with
index set $I_t$ (size $n_t$). Write $X_t := (X_i)_{i\in I_t}$ and
$Y_t := (Y_i)_{i\in I_t}$.

At node $t$, citrees performs two tests:

- **Stage A (feature screening):** For each candidate feature $j\in F_t$ (with
  $m_t := |F_t|$), test the null $H^{\text{sel}}_{t,j}$: “$X_{t,j}$ is
  independent of $Y_t$ (in the sense required by the permutation scheme)”.
- **Stage B (threshold screening):** After a feature $j_t^\star$ is selected,
  test candidate thresholds $c\in C_{t,j_t^\star}$ (with
  $\ell_{t,j}:=|C_{t,j}|$) using a split-quality statistic.

citrees uses Monte Carlo permutation tests: rather than enumerating all
permutations, it draws $B$ i.i.d. random permutations and estimates the
permutation tail probability.

**Paper-facing inferential target.** The cleanest theoretical guarantees in this
framework are:

1. **Fixed-node statements** (treating $t$ as fixed, not adaptively selected),
   especially at the **root**.
2. **Stage A** error-control statements; Stage A is the “gatekeeper” for whether
   the tree splits at all.

---

### A.1.1 Assumptions and scope ledger (used throughout Appendix A)

We label the minimal assumptions for later reference. When a result below is
described as “fixed-node,” it means the node $t$ is treated as fixed (not
adaptively selected).

**A0.1 (Fixed node).** The node $t$ is fixed, not data-adaptively selected.

**A0.2 (Conditional exchangeability).** Under the null being tested, $Y_t$ is
exchangeable conditional on $(X_t,U)$, where $U$ denotes label-independent
algorithmic randomness (e.g., `max_features`, threshold subsampling, RNG seeds).

**A0.3 (Label-independent tested family).** The tested feature family $F_t$,
candidate thresholds, and the resample budget $B$ are measurable functions of
$(X_t,U)$ and do **not** depend on the observed labels $Y_t$ (or on permutation
outcomes).

**A0.4 (Fixed-$B$ p-values).** Permutation p-values are computed with a fixed
number of resamples $B$ (no optional stopping).

**A0.5 (Tie handling).** Ties are either (i) counted against the null
(conservative) or (ii) resolved by randomized tie-breaking for exact uniformity.

**A0.6 (Idealized symmetry model for CART lemmas).** In Lemmas 3a–3c and related
remarks, candidate split scores are assumed exchangeable across all $(j,c)$ (and
tie-free, or tie-broken at random).

**A0.7 (Impurity moment bounds).** For Proposition 3g, the root impurity is
bounded (classification). For Proposition 3h,
$\mathbb{E}[I(Y_{\text{root}})^2] < \infty$.

**A0.8 (Honest estimation).** For Appendix D, the sample split $(S,E)$ is
independent of the observed data.

### A.1.2 Result → assumption checklist (compact)

| Result                                                | Minimal assumptions                                                      |
| ----------------------------------------------------- | ------------------------------------------------------------------------ |
| Lemma 1 (exchangeability of Monte Carlo stats)        | A0.1, A0.2, A0.3                                                         |
| Theorem 1 (+1 p-value super-uniformity)               | A0.1, A0.2, A0.3, A0.4, A0.5                                             |
| Corollary 1 (conditional super-uniformity)            | A0.1, A0.2, A0.3, A0.4, A0.5                                             |
| Lemma 2 (Bonferroni)                                  | Super-uniform p-values (Theorem 1), hence A0.1–A0.5                      |
| Proposition 3 (Stage A, global null)                  | A0.1–A0.5                                                                |
| Proposition 3a (per-feature bound)                    | A0.1–A0.5                                                                |
| Corollary 3a'' (partial null)                         | A0.1–A0.5                                                                |
| Corollary 3c/3d (root bounds)                         | A0.1–A0.5; Corollary 3d additionally assumes uniform feature subsampling |
| Proposition 3g (root impurity bound, classification)  | A0.1–A0.5, A0.7 (bounded impurity)                                       |
| Proposition 3h (root impurity bound, moment form)     | A0.1–A0.5, A0.7 (finite second moment)                                   |
| Lemma 3a (union bound on threshold search)            | No distributional assumptions; union bound only                          |
| Lemma 3b / 3b' (CART proportional selection)          | A0.6 (exchangeable candidate scores; tie-free or randomized)             |
| Lemma 3c (CART root-importance bias, idealized)       | A0.6                                                                     |
| Lemma 3e / 3e' (Stage A uniform selection, idealized) | Exchangeable p-values at the node; tie-free or randomized                |
| Lemma 3f (uniformity under `max_features`, idealized) | Lemma 3e + uniform feature subsampling                                   |
| Lemma (monotone invariance, A.2.4)                    | Algebraic; no probabilistic assumptions beyond strict monotonicity       |
| Lemma A.1–A.5 (bounds/impurity nonnegativity)         | Algebraic; finite moments as stated in each lemma                        |
| Proposition 4 / Corollary 4a (root-level bounds)      | A0.1–A0.5 (root fixed)                                                   |
| Proposition D.1 / D.2 (honesty)                       | A0.8 (independent split), i.i.d. sampling                                |

In the results below, the “Assumptions” line lists the standing A0.1–A0.5
conditions unless additional conditions are stated.

### A.2 Monte Carlo permutation p-values (+1 correction)

Fix a node $t$, feature $j$, and a test statistic $T(\cdot,\cdot)$ with a tail
convention where “more extreme” means larger (right tail). Let
$\pi_1,\dots,\pi_B$ be i.i.d. uniform random permutations of $\{1,\dots,n_t\}$,
independent of the data, and set $Y_t^{(b)} := \pi_b(Y_t)$.

Define

$$
T_0 := T(X_{t,j}, Y_t),\qquad T_b := T(X_{t,j}, Y_t^{(b)})\quad (b=1,\dots,B),
$$

and the **+1 Monte Carlo p-value**

$$
p := \frac{1 + \sum_{b=1}^B \mathbf{1}\{T_b \ge T_0\}}{B+1}.
$$

This is the correction recommended by Phipson & Smyth (2010) (“Permutation
P-values Should Never Be Zero”).

**Assumptions.** A0.1–A0.3 (fixed node, conditional exchangeability,
label-independent tested family).

#### Lemma 1 (exchangeability of the Monte Carlo permutation statistics)

Let $\Pi_0,\Pi_1,\dots,\Pi_B$ be i.i.d. uniform random permutations of
$\{1,\dots,n_t\}$, independent of $(X_t,Y_t)$. Define

$$
T_b := T(X_{t,j}, \Pi_b(Y_t))\quad (b=0,1,\dots,B),
$$

where $T$ is the chosen test statistic and tail convention. Then conditional on
$(X_t,Y_t)$, the vector $(T_0,\dots,T_B)$ is exchangeable (hence exchangeable
unconditionally).

**Proof.** Conditional on $(X_t,Y_t)$, the random variables $(T_b)_{b=0}^B$ are
measurable functions of the i.i.d. permutations $(\Pi_b)_{b=0}^B$. Any
permutation of the indices $b$ leaves the joint law unchanged because
$(\Pi_0,\dots,\Pi_B)$ is i.i.d. $\square$

**Why the usual “unpermuted + $B$ permuted” computation is covered.** Lemma 1
draws $\Pi_0$ as a _random_ permutation, whereas implementations typically take
$\Pi_0$ to be the identity (i.e., $T_0=T(X_{t,j},Y_t)$). Under a
permutation-test null where $Y_t$ is exchangeable, these lead to the same
distribution for $(T_0,\dots,T_B)$, so the rank-based proof below applies to the
standard implementation. The exchangeability (and hence validity) is
**conditional on** $(X_t,U)$, not on the realized $Y_t$; conditioning on the
observed labels breaks the symmetry.

**Assumptions.** A0.1–A0.5.

#### Theorem 1 (finite-sample validity under exchangeability)

Assume that, under the null hypothesis being tested, $(T_0,T_1,\dots,T_B)$ is
exchangeable. Then for all $\alpha\in[0,1]$,

$$
\mathbb{P}(p \le \alpha)\le \alpha.
$$

If ties occur, using $\mathbf{1}\{T_b \ge T_0\}$ “counts ties against the null”
and remains conservative. If you want an exactly uniform p-value in the presence
of ties, you can use randomized tie-breaking (e.g., lexicographic tie-breaking
with i.i.d. $U_b\sim\mathrm{Unif}(0,1)$).

**Proof (rank argument).** We present the right-tail case; the left-tail case is
identical with inequalities reversed.

Define the (upper) rank of $T_0$ among $\{T_0,\dots,T_B\}$ as

$$
R := 1 + \sum_{b=1}^B \mathbf{1}\{T_b \ge T_0\},
$$

so $p = R/(B+1)$.

If $(T_0,\dots,T_B)$ is almost surely tie-free, then $R\in\{1,\dots,B+1\}$ is
exactly the rank of $T_0$ in descending order. Exchangeability implies that each
position is equally likely, hence

$$
\mathbb{P}(R=r)=\frac{1}{B+1}\quad (r=1,\dots,B+1).
$$

Therefore, for any $\alpha\in[0,1]$,

$$
\mathbb{P}(p\le \alpha)
=\mathbb{P}(R\le (B+1)\alpha)
=\frac{\lfloor (B+1)\alpha\rfloor}{B+1}
\le \alpha.
$$

If ties can occur, introduce i.i.d. $U_0,\dots,U_B\sim\mathrm{Unif}(0,1)$
independent of $(T_0,\dots,T_B)$ and define tie-broken statistics
$\widetilde{T}_b := (T_b, U_b)$ ordered lexicographically. Then
$(\widetilde{T}_0,\dots,\widetilde{T}_B)$ is exchangeable and almost surely
tie-free, so the corresponding tie-broken p-value

$$
\widetilde{p} := \frac{1 + \sum_{b=1}^B \mathbf{1}\{\widetilde{T}_b \ge \widetilde{T}_0\}}{B+1}
$$

satisfies $\mathbb{P}(\widetilde{p}\le\alpha)\le\alpha$. Moreover, for every
realization,
$\mathbf{1}\{\widetilde{T}_b \ge \widetilde{T}_0\} \le \mathbf{1}\{T_b \ge T_0\}$,
hence $\widetilde{p}\le p$ pointwise. Therefore
$\mathbb{P}(p\le\alpha)\le \mathbb{P}(\widetilde{p}\le\alpha)\le\alpha$.
$\square$

**How exchangeability arises in permutation tests.** Under a permutation-test
null (e.g., $X_{t,j}\perp Y_t$ under i.i.d. sampling), the conditional
distribution of $Y_t$ given the covariates treated as fixed by the test is
invariant to permutations. Consequently,
$T(X_{t,j},Y_t)\stackrel{d}{=}T(X_{t,j},\pi(Y_t))$ for uniform $\pi$, and the
Monte Carlo vector $(T_0,T_1,\dots,T_B)$ is exchangeable (conditional on $X_t$
and any label-independent algorithmic randomness used to form the tested
family). If the data violate exchangeability (e.g., clustered, longitudinal, or
time-series structure), these guarantees do not apply without a corresponding
restricted permutation scheme.

**Remark (random candidate sets are safe if label-independent).** At the root
(or any fixed node), the tested feature family and test configuration may be
random, e.g.:

- `max_features` draws a random subset of features to test in Stage A,
- some threshold methods subsample candidate thresholds using label-independent
  randomness,
- the resample budget $B$ may be chosen as a function of the number of tested
  hypotheses (e.g., via Bonferroni).

These choices are compatible with Theorem 1 as long as they are measurable
functions of $(X_t,U)$ where $U$ denotes algorithmic randomness independent of
$Y_t$ under the null. Formally, conditioning on $(X_t,U)$ fixes the tested
family and the statistic, so the usual permutation exchangeability argument
applies conditionally, and then unconditionally by averaging over $(X_t,U)$.
This explicitly **excludes** selecting $F_t$, thresholds, or $B$ based on the
observed $Y_t$ (or on intermediate permutation outcomes), unless one is willing
to give up fixed-$B$ p-value interpretations.

**Assumptions.** A0.1–A0.5.

#### Corollary 1 (conditional super-uniformity; fixed node)

Fix a node $t$ and any label-independent algorithmic randomness $U$ (e.g.,
random feature subsampling, subsampled threshold candidates, RNG seeds used to
define the tested family). If, under the null hypothesis being tested, $Y_t$ is
exchangeable conditional on $(X_t,U)$, then the +1 Monte Carlo permutation
p-value satisfies, for all $\alpha\in[0,1]$,

$$
\mathbb{P}(p\le \alpha\mid X_t,U)\le \alpha\quad\text{almost surely}.
$$

**Proof.** Conditional on $(X_t,U)$, the tested family and the statistic are
fixed and the exchangeability assumption reduces to the setup of Lemma 1 +
Theorem 1. $\square$

**What is (and is not) implied.**

- This theorem gives **finite-sample super-uniformity** with **no independence
  assumptions** among tests. It is the core mathematical justification for
  permutation p-values inside citrees.
- This theorem does **not** automatically justify interpreting p-values computed
  at **adaptively selected internal nodes** as classical p-values for fixed
  hypothesis families; tree growth creates data-dependent conditioning events.

---

#### A.2.1 Monte Carlo resolution (why we require a minimum number of resamples)

Let

$$
K := \sum_{b=1}^B \mathbf{1}\{T_b \ge T_0\}
$$

so that $p=(K+1)/(B+1)$. Then $p$ takes values in the grid
$\{1/(B+1),2/(B+1),\dots,1\}$, and the minimum possible reported p-value is
$p_{\min}=1/(B+1)$.

Therefore, if a procedure compares $p$ to a threshold $\alpha'$ (e.g.,
$\alpha'=\alpha$ for a single test or $\alpha'=\alpha/m$ for Bonferroni with $m$
tests), then **rejection is only possible if** $p_{\min}<\alpha'$, equivalently
$B+1>1/\alpha'$. A simple sufficient rule is

$$
B \;\ge\; \left\lceil\frac{1}{\alpha'}\right\rceil.
$$

This is why citrees enforces a minimum number of resamples when `n_resamples` is
set to an enum mode (and, under Bonferroni, scales resampling so that comparing
to the adjusted level remains meaningful). If the user supplies a numeric
`n_resamples`, the code uses that value (scaled by the number of tests under
Bonferroni) even if it is smaller than $\lceil 1/\alpha' \rceil$. In the **early
stopping** modes, the sequential routines also enforce
`min_resamples = ceil(1/alpha')` by bumping `n_resamples` up if needed.

**Fixed-$B$ requirement.** Theorem 1 assumes the permutation sample size $B$ is
fixed **a priori** or chosen as a label-independent function of $(X_t,U)$ at the
fixed node. If $B$ is chosen adaptively based on $Y_t$ (or on running
permutation outcomes), the fixed-$B$ super-uniformity result does not apply.

**Remark (exact achieved level at finite $B$).** Under a tie-free null
idealization, the +1 p-value is discrete-uniform on $\{1/(B+1),\dots,1\}$, so
comparing to a nominal threshold $u$ yields an exact null rejection probability
of $\lfloor (B+1)u\rfloor/(B+1)\le u$. This shows how finite-$B$ discreteness
can introduce an additional (usually small) “floor” conservativeness beyond the
nominal level.

#### A.2.2 Empirical check: fixed-$B$ p-value calibration

Reproduce a simple null calibration study (Theorem 1 backstop):

```bash
uv sync --group paper
UV_CACHE_DIR=$PWD/.uv-cache uv run python paper/scripts/theory/generate_fixedB_pvalue_calibration.py
```

Outputs:

- `paper/results/cache/fixedB_pvalue_calibration_data.parquet`
- `paper/results/figures/fixedB_pvalue_calibration.png`

#### A.2.3 Multi-selector mode (max-statistic / “max-T” combination)

citrees allows combining multiple selector statistics (e.g.,
`selector=['pc','dc','rdc']` for regression) by defining a single combined test
statistic as the maximum across selectors.

Formally, let $T^{(1)},\dots,T^{(K)}$ be $K$ selector statistics computed on the
same $(X_{t,j},Y_t)$ with a shared tail convention, and define the combined
statistic

$$
M_0 := \max_{k=1,\dots,K} |T^{(k)}(X_{t,j},Y_t)|.
$$

On permutation $b$, compute

$$
M_b := \max_{k=1,\dots,K} |T^{(k)}(X_{t,j},\pi_b(Y_t))|.
$$

Then compute the +1 p-value using $M_b$ in place of $T_b$:

$$
p_{\max} := \frac{1 + \sum_{b=1}^B \mathbf{1}\{M_b \ge M_0\}}{B+1}.
$$

Under the same exchangeability conditions as Theorem 1, $(M_0,\dots,M_B)$ is
exchangeable (as a measurable function of the exchangeable vector of selector
statistics), hence $p_{\max}$ is a valid super-uniform p-value. This is a
standard “max-statistic” construction (often called a max-T method; see Westfall
& Young (1993) for broader resampling-based multiple testing).

**Note (scales).** In classification, citrees does not allow mutual information
(`mi`) to be combined with bounded statistics like `mc` and `rdc` in
multi-selector mode because of scale mismatch. This is a power/robustness
choice, not a validity requirement for the max-statistic construction itself.

---

#### A.2.4 Monotone invariance (and links to ANOVA/correlation tests)

Fixed-$B$ permutation p-values depend only on the ordering of
$(T_0,T_1,\dots,T_B)$, so strictly monotone transforms of the test statistic
yield identical +1 p-values (up to tie conventions).

**Assumptions (local).** Strict monotonicity of $\varphi$.

**Lemma (monotone invariance).** For a right-tail +1 p-value

$$
p := \frac{1+\sum_{b=1}^B \mathbf{1}\{T_b\ge T_0\}}{B+1},
$$

if $\varphi$ is strictly increasing then replacing each $T_b$ by $\varphi(T_b)$
leaves $p$ unchanged (and similarly for left-tail tests with $\le$).

**Corollaries (used in citrees selectors).**

- `mc` is the correlation ratio $\eta=\sqrt{\mathrm{SSB}/\mathrm{SST}}$ and is a
  strictly increasing transform of the one-way ANOVA $F$ statistic at fixed
  $(n,g)$ (Appendix A.6.1).
- `pc` is $|\rho|$ and is a strictly increasing transform of the usual
  correlation $|t|$ statistic at fixed $n$ (Appendix A.6.2).

This provides a familiar interpretation of Stage A’s selectors while keeping
calibration purely permutation-based.

---

### A.3 Bonferroni control for Stage A (nodewise + root-level)

**Standing assumptions for A.3.** The node $t$ is treated as fixed; the tested
feature family $F_t$ and the resample budget $B$ are measurable functions of
$(X_t,U)$ and are independent of $Y_t$ under the null; and Stage A uses
fixed-$B$ permutation p-values. Under these conditions, Theorem 1 yields
super-uniform p-values for each true null tested at $t$.

**Assumptions.** A0.1–A0.5 (super-uniform p-values from Theorem 1).

#### Lemma 2 (Bonferroni with super-uniform p-values)

Let $p_1,\dots,p_m$ be p-values such that for each true null $H_j$ and all
$u\in[0,1]$, $\mathbb{P}(p_j \le u)\le u$. Under the global null (all $H_j$
true),

$$
\mathbb{P}\!\left(\min_{1\le j\le m} p_j \le \frac{\alpha}{m}\right)\le \alpha.
$$

No independence assumptions are required (union bound).

**Remark (expected number of false rejections under Bonferroni).** Under the
same super-uniformity assumption, if we form the Bonferroni rejection set
$R=\{j: p_j\le \alpha/m\}$ and let $\mathcal{H}_0$ be the set of true nulls,
then

$$
\mathbb{E}[|R\cap \mathcal{H}_0|] \le \alpha
\quad\text{and}\quad
\mathbb{P}(|R\cap \mathcal{H}_0|\ge r)\le \alpha/r\ \text{ for }r\ge 1,
$$

by linearity of expectation and Markov’s inequality:

**Proof.** Let
$V:=|R\cap\mathcal{H}_0|=\sum_{j\in\mathcal{H}_0}\mathbf{1}\{p_j\le \alpha/m\}$.
Then

$$
\mathbb{E}[V]
=\sum_{j\in\mathcal{H}_0}\mathbb{P}(p_j\le \alpha/m)
\le \sum_{j\in\mathcal{H}_0}\alpha/m
\le \alpha.
$$

For $r\ge 1$, Markov’s inequality gives
$\mathbb{P}(V\ge r)\le \mathbb{E}[V]/r \le \alpha/r$. $\square$

**Assumptions.** A0.1–A0.5.

#### Proposition 3 (Stage A, fixed node, global null over tested features)

At a fixed node $t$, suppose $H^{\text{sel}}_{t,j}$ holds for all $j\in F_t$ and
let $p_{t,j}$ be valid permutation p-values (Theorem 1). If Stage A uses
Bonferroni (threshold $\alpha_{\text{sel}}/m_t$), then

$$
\mathbb{P}\!\left(\exists j\in F_t:\; p_{t,j}\le \alpha_{\text{sel}}/m_t\right)\le \alpha_{\text{sel}}.
$$

**Proof.** Stage A rejects the global null only if
$\min_{j\in F_t} p_{t,j}\le \alpha_{\text{sel}}/m_t$. Under the global null,
Lemma 2 applies directly and yields the bound. $\square$

**Assumptions.** A0.1–A0.5.

#### Proposition 3a (per-feature false selection bound; no global-null needed)

At a fixed node $t$, fix a particular feature $j\in F_t$ whose null
$H^{\text{sel}}_{t,j}$ is true, so $p_{t,j}$ is super-uniform (Theorem 1). If
Stage A uses Bonferroni, then the event “the node splits on feature $j$” implies
$p_{t,j}\le \alpha_{\text{sel}}/m_t$, hence

$$
\mathbb{P}(\text{node }t\text{ splits on feature }j)\le \alpha_{\text{sel}}/m_t.
$$

This bound does not depend on the feature’s number of unique values, which is
one precise sense in which Stage A avoids the classic CART-style
“high-cardinality selection bias”.

**Proof.** If the node splits on $j$, then Stage A must have accepted $j$ at the
Bonferroni threshold, so $p_{t,j}\le \alpha_{\text{sel}}/m_t$. Since $p_{t,j}$
is super-uniform under the null, we have
$\mathbb{P}(p_{t,j}\le \alpha_{\text{sel}}/m_t)\le \alpha_{\text{sel}}/m_t$, and
the claim follows by set inclusion. $\square$

**Assumptions.** A0.1–A0.5.

#### Corollary 3a'' (probability Stage A selects a null feature; partial-null bound)

At a fixed node $t$, let $\mathcal{H}_{0,t}\subseteq F_t$ be the set of tested
features whose Stage A nulls are true, with
$m_{0,t}:=|\mathcal{H}_{0,t}|\le m_t$. Under the assumptions of Proposition 3a
(super-uniformity for each $j\in\mathcal{H}_{0,t}$), we have

$$
\mathbb{P}(\text{node }t\text{ splits on some }j\in\mathcal{H}_{0,t})
\le
m_{0,t}\cdot \alpha_{\text{sel}}/m_t
\le
\alpha_{\text{sel}}.
$$

**Proof.** By the union bound and Proposition 3a applied to each
$j\in\mathcal{H}_{0,t}$,

$$
\mathbb{P}(\text{node }t\text{ splits on some }j\in\mathcal{H}_{0,t})
\le \sum_{j\in\mathcal{H}_{0,t}}\mathbb{P}(\text{node }t\text{ splits on }j)
\le \sum_{j\in\mathcal{H}_{0,t}}\alpha_{\text{sel}}/m_t
= m_{0,t}\alpha_{\text{sel}}/m_t.
$$

The final inequality uses $m_{0,t}\le m_t$. $\square$

**Assumptions.** A0.1–A0.5.

#### Corollary 3c (root split bound over tested features)

Specializing Proposition 3a to the root: if Stage A tests a feature family
$F_{\text{root}}$ of size $m_{\text{root}}:=|F_{\text{root}}|$ and uses
Bonferroni correction, then under the global null at the root, for any tested
feature $j\in F_{\text{root}}$,

$$
\mathbb{P}(\text{the root splits on feature }j)\le \alpha_{\text{sel}}/m_{\text{root}}.
$$

**Assumptions.** A0.1–A0.5.

#### Corollary 3d (uniform root split bound under random feature subsampling)

Assume the global null holds at the root for all $p$ features. Suppose Stage A
tests a random subset $F_{\text{root}}\subseteq\{1,\dots,p\}$ obtained by
sampling $m$ features uniformly without replacement (as in `max_features`) and
applies Bonferroni over the tested subset. Then for any feature
$j\in\{1,\dots,p\}$,

$$
\mathbb{P}(\text{the root splits on feature }j)\le \alpha_{\text{sel}}/p.
$$

**Proof (conditioning on the tested subset).** If $j\notin F_{\text{root}}$, the
root cannot split on $j$. If $j\in F_{\text{root}}$, Corollary 3c gives
$\mathbb{P}(\text{root splits on }j\mid F_{\text{root}})\le \alpha_{\text{sel}}/|F_{\text{root}}|=\alpha_{\text{sel}}/m$.
Therefore

$$
\mathbb{P}(\text{root splits on }j)
= \mathbb{E}\big[\mathbb{P}(\text{root splits on }j\mid F_{\text{root}})\big]
\le \mathbb{P}(j\in F_{\text{root}})\cdot \alpha_{\text{sel}}/m
= (m/p)\cdot \alpha_{\text{sel}}/m
= \alpha_{\text{sel}}/p.
$$

$\square$

**Assumptions.** A0.1–A0.5 + uniform feature subsampling (`max_features`).

#### Proposition 3g (expected root impurity decrease for a null feature; classification)

Consider a classification tree at the root in fixed-$B$ mode with Stage A
Bonferroni screening at level $\alpha_{\text{sel}}$. Fix a particular feature
$j$ whose Stage A null is true at the root (so $X_{\cdot j}\perp Y$). Let
$\Delta I_{\text{root}}$ be the (weighted) impurity decrease at the root _if the
root splits_ and define the root contribution attributable to feature $j$ as

$$
Z_j := \Delta I_{\text{root}}\cdot \mathbf{1}\{\text{root splits on feature }j\}.
$$

If the splitter impurity is bounded by a deterministic constant $I_{\max}$
(e.g., for Gini with $K$ classes, $I_{\max}=1-1/K$; for entropy with $K$
classes, $I_{\max}=\log_2 K$), then

$$
\mathbb{E}[Z_j] \le I_{\max}\cdot \mathbb{P}(\text{root splits on }j).
$$

In particular:

1. If the root tests a fixed set $F_{\text{root}}$ of size $m_{\text{root}}$ and
   uses Bonferroni, then

   $$
   \mathbb{E}[Z_j] \le I_{\max}\,\alpha_{\text{sel}}/m_{\text{root}}
   \quad\text{for each tested null feature }j\in F_{\text{root}}.
   $$

2. Under uniform feature subsampling at the root (as in `max_features`) with
   Bonferroni over the tested subset, if the complete null holds for all $p$
   features then
   $$
   \mathbb{E}[Z_j] \le I_{\max}\,\alpha_{\text{sel}}/p
   \quad\text{for each feature }j\in\{1,\dots,p\}.
   $$

**Proof.** Since $\Delta I_{\text{root}} \le I(Y_{\text{root}})\le I_{\max}$
almost surely, we have $Z_j \le I_{\max}\mathbf{1}\{\text{root splits on }j\}$
pointwise and therefore
$\mathbb{E}[Z_j]\le I_{\max}\mathbb{P}(\text{root splits on }j)$.

The remaining bounds follow by substituting the corresponding root
split-probability bounds: Corollary 3c gives
$\mathbb{P}(\text{root splits on }j)\le \alpha_{\text{sel}}/m_{\text{root}}$ for
tested null features, and Corollary 3d gives
$\mathbb{P}(\text{root splits on }j)\le \alpha_{\text{sel}}/p$ under uniform
subsampling and complete null. $\square$

**Remark (scope).** Proposition 3g controls only the expected impurity decrease
attributable to a null feature at the root. It does not control impurity-based
importance accumulated at internal nodes.

**Assumptions.** A0.1–A0.5 + A0.7 (bounded classification impurity).

#### Proposition 3h (root importance bound for unbounded impurities; moment form)

The boundedness assumption in Proposition 3g is automatic for Gini/entropy, but
not for regression impurities such as MSE/MAE. A simple moment-based bound is
still available.

Let $I(\cdot)\ge 0$ be any nonnegative node impurity functional and let
$\Delta I_{\text{root}}$ be the usual weighted impurity decrease at the root:

$$
\Delta I_{\text{root}}
:=
I(Y_{\text{root}})
-
\Big(\tfrac{n_L}{n} I(Y_L) + \tfrac{n_R}{n} I(Y_R)\Big),
$$

with $n_L+n_R=n$ when the root splits and $\Delta I_{\text{root}}:=0$ otherwise.
Fix a feature $j$ and define
$Z_j:=\Delta I_{\text{root}}\cdot \mathbf{1}\{\text{root splits on feature }j\}$
as before. Then:

1. pointwise, $\Delta I_{\text{root}}\le I(Y_{\text{root}})$ (because the
   weighted child impurity is nonnegative), and
2. for any $j$,
   $$
   \mathbb{E}[Z_j]
   \le
   \Big(\mathbb{E}[I(Y_{\text{root}})^2]\Big)^{1/2}\cdot
   \Big(\mathbb{P}(\text{root splits on }j)\Big)^{1/2}.
   $$

**Proof.** Since $I(\cdot)\ge 0$, we have
$\Delta I_{\text{root}}\le I(Y_{\text{root}})$ pointwise and therefore
$Z_j\le I(Y_{\text{root}})\mathbf{1}\{\text{root splits on }j\}$. Applying
Cauchy–Schwarz yields

$$
\mathbb{E}[Z_j]
\le
\mathbb{E}\!\left[I(Y_{\text{root}})\mathbf{1}\{\text{root splits on }j\}\right]
\le
\Big(\mathbb{E}[I(Y_{\text{root}})^2]\Big)^{1/2}\cdot
\Big(\mathbb{P}(\text{root splits on }j)\Big)^{1/2}.
$$

$\square$

**Assumptions.** A0.1–A0.5 + A0.7 (finite second moment of root impurity).

**Assumptions (A0.6).** Exchangeable, tie-free candidate scores (idealized
null).

#### Lemma 3b (CART bias mechanism, idealized)

The high-cardinality bias in greedy trees can be formalized as a pure
multiple-comparisons effect.

Fix a node with candidate features $j\in\{1,\dots,p\}$ and candidate split sets
$C_j$ with sizes $|C_j|=m_j\ge 1$. Suppose each candidate split $(j,c)$ has a
score $S_{j,c}$ where smaller is better (e.g., child impurity), and under the
global null the collection $\{S_{j,c}\}$ is exchangeable and tie-free.

Let a CART-style procedure choose
$(\widehat j,\widehat c)\in\arg\min_{j,c} S_{j,c}$ and split on feature
$\widehat j$. Then

$$
\mathbb{P}(\widehat j=j)=\frac{m_j}{\sum_{k=1}^p m_k}.
$$

**Proof (symmetry).** Under exchangeability and no ties, the argmin is uniformly
distributed over the $\sum_k m_k$ candidate splits, so it belongs to feature $j$
with probability $m_j/\sum_k m_k$. $\square$

**Remark (ties).** The tie-free assumption is only to avoid specifying how a
CART implementation breaks ties between equally good candidate splits. The
proportional-selection conclusion persists under randomized tie-breaking:

**Assumptions (A0.6 + randomized ties).** I.i.d. continuous perturbations for
tie-breaking.

**Lemma 3b' (CART proportional selection with randomized tie-breaking).** Assume
the exchangeability condition of Lemma 3b, but allow ties. Let $(U_{j,c})$ be
i.i.d. $\mathrm{Unif}(0,1)$ independent of the scores, and define the tie-broken
scores $\widetilde S_{j,c}:=(S_{j,c},U_{j,c})$ ordered lexicographically. Let
CART choose $(\widehat j,\widehat c)\in\arg\min_{j,c}\widetilde S_{j,c}$ and
split on feature $\widehat j$. Then

$$
\mathbb{P}(\widehat j=j)=\frac{m_j}{\sum_{k=1}^p m_k}.
$$

**Proof.** The augmented collection $\{(S_{j,c},U_{j,c})\}$ is exchangeable and
almost surely tie-free (because the continuous $U$ breaks ties with probability
1). The proof of Lemma 3b applies verbatim. $\square$

**Remark (random feature subsampling does not remove the bias mechanism).** If a
CART-style method first samples a subset of candidate features $F$ (as in random
forests) and then chooses the best candidate split among those features by
minimizing impurity, the same symmetry argument implies that conditional on $F$,

$$
\mathbb{P}(\widehat j=j\mid F)=\frac{m_j}{\sum_{k\in F} m_k}\quad (j\in F),
$$

so high-cardinality features remain favored **within the tested subset**.

**Proof.** Conditional on $F$, apply the Lemma 3b symmetry argument to the
restricted candidate collection $\{S_{j,c}: j\in F,\ c\in C_j\}$. $\square$

**Concrete “999 chances” example (how to read Lemma 3b).** If a continuous
feature has $\ell=1000$ unique values, it contributes $m=\ell-1=999$ candidate
midpoints at a node, while a binary feature contributes $m=1$. In the idealized
setting of Lemma 3b (exchangeable, tie-free split scores), the probability CART
splits on the high-cardinality feature is

$$
\frac{999}{999+1}=0.999.
$$

This is the correct mathematical counterpart of the informal “999 chances”
intuition (it does not require treating the 999 candidate split scores as
independent).

**Same setup, citrees Stage A contrast (finite-sample bound, not an
idealization).** If Stage A tests both features (so $m_t=2$) and uses Bonferroni
at level $\alpha_{\text{sel}}$, then for either feature $j$ whose Stage A null
is true at the node,

$$
\mathbb{P}(\text{node splits on feature }j)\le \alpha_{\text{sel}}/2,
$$

independent of whether $j$ has 1 candidate threshold or 999 candidate
thresholds.

**Toy extreme-value calculation (independent scores; illustrative only).** If we
additionally idealize the candidate split scores for a fixed feature as i.i.d.
$\mathrm{Unif}(0,1)$ with smaller scores better, then the best attainable score
$S_{\min}:=\min_{c\in C_j} S_{j,c}$ satisfies

$$
\mathbb{P}(S_{\min} > s) = (1-s)^{m_j}\quad (s\in[0,1]),
$$

so

$$
\mathbb{E}[S_{\min}] = \frac{1}{m_j+1}.
$$

For $m_j=999$, this gives $\mathbb{E}[S_{\min}]=0.001$, while for $m_j=1$ it
gives $\mathbb{E}[S_{\min}]=0.5$. This calculation is purely motivational (real
impurity scores are neither independent nor uniform), but it quantifies the
“extreme-of-$m$” advantage created by threshold search.

**Root-level contrast (what this buys you).** Lemma 3b formalizes that a greedy
CART-style procedure allocates selection probability in proportion to the number
of candidate splits $m_j$ under an exchangeable null idealization. In contrast,
the citrees Stage A guarantee bounds
$\mathbb{P}(\text{root splits on a particular null feature }j)$ by
$\alpha_{\text{sel}}/m_{\text{root}}$ (tested set) or by $\alpha_{\text{sel}}/p$
under uniform `max_features` (Corollary 3d), with **no dependence on $m_j$**.

Under the same exchangeable-null idealization, even a one-node
impurity-importance score inherits the same multiplicity mechanism:

**Assumptions (A0.6).** Exchangeable candidate scores (idealized null).

#### Lemma 3c (root impurity-importance bias in CART, idealized)

In the setting of Lemma 3b, suppose the split score is an impurity decrease
$\Delta_{j,c}$ where **larger is better**, and CART chooses
$(\widehat j,\widehat c)\in\arg\max_{j,c}\Delta_{j,c}$. Let

$$
\Delta_{\max} := \max_{j\in\{1,\dots,p\},\,c\in C_j}\Delta_{j,c},
\qquad
Z_j := \Delta_{\max}\cdot \mathbf{1}\{\widehat j=j\}.
$$

Then

$$
\mathbb{E}[Z_j] = \frac{m_j}{\sum_{k=1}^p m_k}\,\mathbb{E}[\Delta_{\max}].
$$

**Proof (symmetry).** By exchangeability and tie-freeness, the argmax index is
uniformly distributed over the $\sum_k m_k$ candidate splits, so
$\mathbb{P}(\widehat j=j)=m_j/\sum_k m_k$. By symmetry, the conditional
distribution of $\Delta_{\max}$ given the argmax index does not depend on which
candidate achieves the maximum, hence
$\mathbb{E}[\Delta_{\max}\mid \widehat j=j]=\mathbb{E}[\Delta_{\max}]$.
Therefore

$$
\mathbb{E}[Z_j]
=\mathbb{E}\!\left[\Delta_{\max}\mathbf{1}\{\widehat j=j\}\right]
=\mathbb{P}(\widehat j=j)\cdot \mathbb{E}[\Delta_{\max}\mid \widehat j=j]
=\frac{m_j}{\sum_k m_k}\,\mathbb{E}[\Delta_{\max}].
$$

$\square$

**Remark (scope).** Lemma 3c is a one-node (e.g., root) symmetry calculation
under an exchangeable null idealization. It does not claim that full-tree/forest
impurity importance is proportional to $m_j$ in general; deeper nodes are
data-dependent and typically violate the exchangeability assumptions.

**Assumptions (idealized).** Exchangeable Stage-A p-values at a fixed node;
tie-free (or randomized in Lemma 3e').

#### Lemma 3e (Stage A feature selection is uniform under an exchangeable null, idealized)

Fix a node with tested feature set $F_t$ of size $m_t:=|F_t|$, and suppose that
under the global null the Stage A p-value vector $(p_{t,j})_{j\in F_t}$ is
exchangeable and tie-free. Let $j_t^\star:=\arg\min_{j\in F_t} p_{t,j}$. Then:

$$
\mathbb{P}(j_t^\star=j)=\frac{1}{m_t}\quad\text{for all }j\in F_t.
$$

Moreover, for any threshold $\tau\in[0,1]$,

$$
\mathbb{P}(j_t^\star=j \mid \min_{k\in F_t} p_{t,k}\le \tau)=\frac{1}{m_t}.
$$

**Proof.** By exchangeability and tie-freeness, the argmin index is unique and
must have the same distribution for all $j\in F_t$, hence
$\mathbb{P}(j_t^\star=j)=1/m_t$. The conditional claim follows because the event
$\{\min_{k\in F_t} p_{t,k}\le \tau\}$ is invariant to permutations of feature
indices, so $\mathbb{P}(j_t^\star=j,\ \min_k p_{t,k}\le \tau)$ is equal for all
$j$; dividing by $\mathbb{P}(\min_k p_{t,k}\le \tau)$ gives the result.
$\square$

**Remark (scope, and ties).** This is a stylized symmetry statement (e.g.,
i.i.d. noise features and a common Stage A test). In fixed-$B$ Monte Carlo
tests, p-values are discrete so ties are possible; deterministic tie-breaking
can break exact uniformity. Under randomized tie-breaking, the symmetry
conclusion remains exact:

**Assumptions (idealized).** Exchangeable Stage-A p-values at a fixed node;
randomized tie-breaking.

**Lemma 3e' (Stage A uniform selection with randomized tie-breaking).** Assume
$(p_{t,j})_{j\in F_t}$ is exchangeable, but allow ties. Let $(U_j)_{j\in F_t}$
be i.i.d. $\mathrm{Unif}(0,1)$ independent of the p-values, and define
$\widetilde p_{t,j}:=(p_{t,j},U_j)$ ordered lexicographically. Let
$j_t^\star:=\arg\min_{j\in F_t}\widetilde p_{t,j}$. Then $j_t^\star$ is uniform
on $F_t$:

$$
\mathbb{P}(j_t^\star=j)=\frac{1}{m_t}\quad\text{for all }j\in F_t,
$$

and for any $\tau\in[0,1]$,

$$
\mathbb{P}(j_t^\star=j \mid \min_{k\in F_t} p_{t,k}\le \tau)=\frac{1}{m_t}.
$$

**Proof.** The augmented vector $\{(p_{t,j},U_j)\}_{j\in F_t}$ is exchangeable
and almost surely tie-free, so the argmin index is uniformly distributed by
symmetry. For the conditional statement, the event $\{\min_k p_{t,k}\le\tau\}$
depends only on $(p_{t,k})$ and is invariant under permutations of feature
indices; apply the same symmetry argument to
$\mathbb{P}(j_t^\star=j,\ \min_k p_{t,k}\le\tau)$. $\square$

For the paper’s core claim we do not need Lemma 3e: without feature
exchangeability (and regardless of ties), Proposition 3a provides the universal
bound
$\mathbb{P}(\text{node splits on a particular null feature }j)\le \alpha_{\text{sel}}/|F_t|$,
independent of that feature’s number of candidate thresholds.

**Remark (how conservative is Bonferroni in an i.i.d. idealization?).** If,
under the global null, the $m_t$ Stage A p-values were i.i.d.
$\mathrm{Unif}(0,1)$ and tie-free, then Stage A would reject at the Bonferroni
level $\alpha_{\text{sel}}/m_t$ with probability

$$
\mathbb{P}\!\left(\min_{j\in F_t} p_{t,j} < \alpha_{\text{sel}}/m_t\right)
= 1-\left(1-\alpha_{\text{sel}}/m_t\right)^{m_t}
\le \alpha_{\text{sel}},
$$

and by Lemma 3e the selected feature would be uniform even conditional on
rejection, giving

$$
\mathbb{P}(\text{node }t\text{ splits on a particular }j\in F_t)
= \frac{1-\left(1-\alpha_{\text{sel}}/m_t\right)^{m_t}}{m_t}
\approx \alpha_{\text{sel}}/m_t.
$$

With a fixed-$B$ +1 permutation p-value (tie-free) one can make the same
calculation with the discrete-uniform grid by replacing
$\alpha_{\text{sel}}/m_t$ with
$\lfloor (B+1)\alpha_{\text{sel}}/m_t\rfloor/(B+1)$ (Appendix A.2.1).

This is only a heuristic calibration (real p-values are discrete and dependent),
but it shows the $\alpha_{\text{sel}}$ and $\alpha_{\text{sel}}/m_t$ bounds are
typically close to tight under symmetry.

**Remark (uniform `max_features` vs CART “mtry”).** In the same
exchangeable-null idealization, if Stage A tests a uniformly sampled subset $F$
of size $m$ and selects the minimum p-value within that subset, then the
selected feature is uniform on $\{1,\dots,p\}$ (an exact symmetry calculation):

**Assumptions (idealized).** Exchangeable Stage-A p-values at a fixed node;
tie-free (or randomized in Lemma 3e').

**Lemma 3f (Stage A remains uniform under random feature subsampling;
idealized).** Fix a node and a full feature set $\{1,\dots,p\}$. Let $F$ be a
random subset of size $m$ sampled uniformly without replacement, independent of
the data (as in `max_features`). Suppose that conditional on $F$, the Stage A
p-values $(p_j)_{j\in F}$ are exchangeable and almost surely tie-free. Let
$j^\star:=\arg\min_{j\in F} p_j$. Then:

1. conditional on $F$, $j^\star$ is uniform on $F$, and
2. unconditionally, $j^\star$ is uniform on $\{1,\dots,p\}$:
   $$
   \mathbb{P}(j^\star=j)=\frac{1}{p}\quad\text{for all }j\in\{1,\dots,p\}.
   $$

**Proof.** Conditional uniformity is Lemma 3e applied on $F$. For the
unconditional statement,

$$
\mathbb{P}(j^\star=j)
=\mathbb{E}\big[\mathbb{P}(j^\star=j\mid F)\big]
=\mathbb{E}\Big[\mathbf{1}\{j\in F\}\cdot\frac{1}{|F|}\Big]
=\mathbb{P}(j\in F)\cdot\frac{1}{m}
=\frac{m}{p}\cdot\frac{1}{m}
=\frac{1}{p}.
$$

$\square$

This contrasts with the CART “mtry” remark above: even after subsampling
features, greedy impurity optimization remains biased toward high-cardinality
features **within the tested subset**.

#### A.3.1 Empirical check: selection bias vs CART

To illustrate the difference between optimization-based variable selection
(CART) and Stage A’s testing-based selection, we include a small null
simulation:

- CART (`sklearn.tree.DecisionTreeClassifier`, `max_depth=1`) selects the root
  feature by maximizing impurity decrease over many possible thresholds, which
  is known to favor high-cardinality features under the global null.
- citrees Stage A selects the root feature by comparing permutation p-values
  against $\alpha/m$.

Reproduce:

```bash
uv sync --group paper
UV_CACHE_DIR=$PWD/.uv-cache uv run python paper/scripts/theory/generate_selection_bias_demo.py
```

Outputs:

- `paper/results/cache/selection_bias_demo_data.parquet`
- `paper/results/figures/selection_bias_demo.png`

**Assumptions.** A0.1–A0.5 (root fixed).

#### Proposition 4 (safe, global statement: “any split implies root rejection”)

Tree adaptivity complicates internal-node inference, but one global statement
remains clean:

> The fitted tree can only have any split if the **root** passes Stage A.

Consequently, if the global null holds for all tested features at the root (and
Stage A uses Bonferroni with fixed $B$), then

$$
\mathbb{P}(\text{the fitted tree has at least one internal split}) \le \alpha_{\text{sel}}.
$$

**Proof.** If the tree has any split, then the root must have passed Stage A;
thus the event “tree splits” is a subset of
$\{\min_{j\in F_{\text{root}}} p_{t,j}\le \alpha_{\text{sel}}/m_{\text{root}}\}$.
Under the global null at the root, Proposition 3 gives the bound. $\square$

**Assumptions.** A0.1–A0.5 per tree at root; complete null across features. (No
independence between trees is required for the union bound.)

#### Corollary 4a (forest-level bounds under complete null; root-level)

Consider a forest with $M$ trees, each of which uses the same root Stage A
Bonferroni screening at level $\alpha_{\text{sel}}$ in fixed-$B$ mode. Under the
complete null at the root (all $p$ features are independent of $Y$), we have:

1. $\mathbb{E}[N_{\text{split}}] \le M\,\alpha_{\text{sel}}$, where
   $N_{\text{split}}$ is the number of trees with at least one split.

2. $\mathbb{P}(N_{\text{split}}\ge 1) \le \min\{1,\;M\,\alpha_{\text{sel}}\}$.

If, additionally, each tree’s root tests either all $p$ features or a
uniform-without-replacement subset (as in `max_features`) and applies Bonferroni
over its tested subset, then for any fixed feature $j$ the expected number of
trees whose root splits on $j$ is bounded by $M\,\alpha_{\text{sel}}/p$.

**Remark (forest-level root importance under null; classification).** Using the
same boundedness argument as in Corollary 3d’s remark, if the forest is used as
an embedding feature selector via impurity-based importance and the complete
null holds at the root, then for any feature $j$,

$$
\mathbb{E}\!\left[\sum_{m=1}^M \Delta I_{\text{root}}^{(m)}\cdot \mathbf{1}\{\text{tree }m\text{ splits at root on }j\}\right]
\le M\,I_{\max}\cdot \frac{\alpha_{\text{sel}}}{p},
$$

where $I_{\max}=1-1/K$ for Gini and $I_{\max}=\log_2 K$ for entropy.

**Proof.** Apply Proposition 4 to each tree and then use linearity of
expectation and the union bound. Under uniform feature subsampling, use
Corollary 3d per tree and linearity. $\square$

---

### A.4 Post-selection caveats (what we do _not_ claim)

1. **Stage B is post-selection.** Stage B is performed _after selecting_
   $j_t^\star$ using the same labels $Y_t$. Without sample splitting or
   selective-inference adjustments, Stage B p-values should be treated as
   **algorithmic stopping statistics**, not classical post-selection p-values
   (e.g., Berk et al., 2013; Lee et al., 2016; Fithian et al., 2014).
2. **Internal nodes are adaptive.** In an adaptively-grown tree, a node $t$
   corresponds to a random index set $I_t$ determined by earlier splits that
   depend on the labels; conditioning on “these samples reach node $t$” can
   break exchangeability under nulls. This is why the most defensible
   inferential statements are either (i) for a fixed node, or (ii) root-level
   (see also Dwork et al., 2015; Leeb & Pötscher, 2015 for general cautions
   about adaptive inference).

---

### A.5 Adaptive sequential permutation testing (early stopping)

citrees optionally uses **adaptive early stopping** as a computational shortcut
for Monte Carlo permutation tests: stop when the evidence is overwhelming for
either “significant” or “not significant”.

**Paper-facing scope.** In adaptive mode, citrees returns the standard +1 Monte
Carlo estimate

$$
\widehat{p}_n := \frac{L_n+1}{n+1}
$$

evaluated at a **data-dependent stopping time** $n=\tau$. We do **not** treat
this returned $\widehat{p}_\tau$ as a classical fixed-$B$ permutation p-value
under optional stopping. For paper-facing p-value guarantees, use **fixed-$B$**
permutation tests (`early_stopping_*=None`) so Theorem 1 applies directly.

**Empirical backstop (optional).** If we discuss early stopping in the paper at
all, we will treat it as a computational heuristic and support it with
calibration figures rather than optional-stopping p-value theorems (see
`paper/scripts/theory/generate_sequential_stopping_calibration.py`).

---

### A.6 Concrete statistics used in citrees (definitions + basic bounds)

This appendix records the exact nodewise statistics used in the codebase and a
few basic properties that justify design decisions (e.g., which selectors can be
combined in multi-selector mode).

#### A.6.1 Classification selector: multiple correlation (`mc`)

At a node with samples $\{(x_i,y_i)\}_{i=1}^n$, where $x_i\in\mathbb{R}$ and
$y_i\in\{1,\dots,K\}$, define the overall mean $\mu := \frac1n\sum_{i=1}^n x_i$
and class means $\mu_k := \frac{1}{n_k}\sum_{i:y_i=k} x_i$ with
$n_k := |\{i:y_i=k\}|$.

Define the “total” and “between-class” sums of squares

$$
\mathrm{SST} := \sum_{i=1}^n (x_i-\mu)^2,\qquad
\mathrm{SSB} := \sum_{k=1}^K n_k(\mu_k-\mu)^2.
$$

citrees uses the multiple correlation coefficient

$$
\mathrm{mc}(x,y) := \sqrt{\mathrm{SSB}/\mathrm{SST}}
$$

when $\mathrm{SST}>0$ (constant features are muted before testing).

**Assumptions (local).** $\mathrm{SST}>0$ (non-constant feature).

**Lemma A.1 (boundedness of `mc`).** If $\mathrm{SST}>0$, then
$0\le \mathrm{mc}(x,y)\le 1$.

**Proof.** The standard ANOVA decomposition gives

$$
\sum_{i=1}^n (x_i-\mu)^2
=
\sum_{k=1}^K\sum_{i:y_i=k}(x_i-\mu_k)^2 + \sum_{k=1}^K n_k(\mu_k-\mu)^2.
$$

Both terms on the right are nonnegative, so $\mathrm{SSB}\le \mathrm{SST}$ and
hence $0\le \mathrm{SSB}/\mathrm{SST}\le 1$. Taking square roots yields the
claim. $\square$

**Assumptions (local).** $\mathrm{SST}>0$, at least two nonempty classes
($g\ge 2$), and $\mathrm{SSW}:=\mathrm{SST}-\mathrm{SSB}>0$ (equivalently
$\eta^2<1$).

**Lemma A.2 (relationship to one-way ANOVA $F$).** Let $g := |\{k : n_k>0\}|$ be
the number of nonempty classes at the node, and assume $\mathrm{SST}>0$ and
$g\ge 2$. Define $\eta^2 := \mathrm{SSB}/\mathrm{SST}\in[0,1)$. The one-way
ANOVA $F$ statistic can be written as

$$
F \;=\; \frac{\mathrm{SSB}/(g-1)}{(\mathrm{SST}-\mathrm{SSB})/(n-g)}
\;=\; \frac{n-g}{g-1}\cdot \frac{\eta^2}{1-\eta^2}.
$$

Consequently, for fixed $(n,g)$, the mappings $\eta \mapsto \eta^2 \mapsto F$
are strictly increasing and induce the same feature ordering and the same
fixed-$B$ permutation p-values (Appendix A.2.4).

**Proof.** The first equality is the standard one-way ANOVA definition with
$\mathrm{SSW}:=\mathrm{SST}-\mathrm{SSB}$. Substituting
$\eta^2=\mathrm{SSB}/\mathrm{SST}$ yields the displayed expression. Strict
monotonicity in $\eta^2\in[0,1)$ follows because $\eta^2/(1-\eta^2)$ is strictly
increasing on $[0,1)$. $\square$

#### A.6.2 Regression selector: Pearson correlation (`pc`)

For vectors $x,y\in\mathbb{R}^n$ with nonzero empirical variances, define

$$
\rho(x,y) := \frac{\sum_{i=1}^n (x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum_{i=1}^n (x_i-\bar x)^2}\sqrt{\sum_{i=1}^n (y_i-\bar y)^2}}.
$$

citrees uses the magnitude $|\rho(x,y)|$ as the association score and
permutation-test statistic.

**Assumptions (local).** Nonzero empirical variances of $x$ and $y$.

**Lemma A.3 (boundedness of `pc`).** Whenever the denominator is nonzero,
$|\rho(x,y)|\le 1$.

**Proof.** Let $x'_i := x_i-\bar x$ and $y'_i := y_i-\bar y$. By Cauchy–Schwarz,

$$
\Big|\sum_{i=1}^n x'_i y'_i\Big|
\le \sqrt{\sum_{i=1}^n (x'_i)^2}\sqrt{\sum_{i=1}^n (y'_i)^2}.
$$

Dividing both sides by the product of norms yields $|\rho(x,y)|\le 1$. $\square$

**Assumptions (local).** $n\ge 3$, nonzero empirical variances, and no perfect
linear dependence ($|\rho|<1$).

**Lemma A.4 (relationship to the correlation $t$ statistic).** Assume $n\ge 3$
and that the empirical variances of $x$ and $y$ are nonzero; if $|\rho(x,y)|=1$
then $|t|=\infty$ and the monotone relationship below holds by continuity, so we
focus on the regular case $|\rho(x,y)|<1$. Define the usual correlation test
statistic

$$
t \;:=\; \rho(x,y)\,\sqrt{\frac{n-2}{1-\rho(x,y)^2}}.
$$

Then $|t|$ is a strictly increasing function of $|\rho(x,y)|$ on $[0,1)$.
Consequently, ranking features by `pc` $=|\rho|$ is equivalent to ranking them
by $|t|$, and fixed-$B$ permutation p-values computed from $|\rho|$ agree with
those computed from $|t|$ (Appendix A.2.4).

**Proof.** Let $r:=|\rho|\in[0,1)$. Then $|t|=\sqrt{n-2}\cdot f(r)$ where
$f(r):=r/\sqrt{1-r^2}$. Since $f'(r)=(1-r^2)^{-3/2}>0$ for $r\in[0,1)$, $f$ is
strictly increasing. $\square$

#### A.6.3 Regression selector: distance correlation (`dc`)

citrees uses distance correlation via the `dcor` library (Székely–Rizzo–Bakirov,
2007). In its population definition, distance correlation takes values in
$[0,1]$ and equals $0$ if and only if the variables are independent (under mild
moment conditions). (The code uses a finite-sample estimator as the nodewise
test statistic.)

#### A.6.4 Selector: randomized dependence coefficient (`rdc`)

citrees implements the RDC of Lopez-Paz et al. (2013) in a 1D-to-1D form:

1. Apply the empirical CDF transform $x\mapsto \widehat{F}_x(x)$ (rank / $n$).
2. Add a bias coordinate, then apply $k$ random linear projections.
3. Apply sinusoidal features $\cos(\cdot),\sin(\cdot)$.
4. Return the maximum absolute correlation between standardized feature columns.

Because each correlation is bounded by 1, the returned value satisfies
$0\le \mathrm{rdc}(x,y)\le 1$.

For multiclass classification, citrees applies the 1D RDC to each one-vs-rest
class indicator and takes the maximum across classes.

#### A.6.5 Split impurities (`gini`, `entropy`, `mse`, `mae`)

For a classification node with empirical class probabilities $p_1,\dots,p_K$:

- Gini impurity: $\mathrm{gini}(p) := 1-\sum_{k=1}^K p_k^2$, hence
  $0\le \mathrm{gini}\le 1-1/K$.
- Entropy impurity: $\mathrm{ent}(p):= -\sum_{k=1}^K p_k \log_2 p_k$, hence
  $0\le \mathrm{ent}\le \log_2 K$.

For a regression node with targets $y_1,\dots,y_n$:

- MSE impurity: $\mathrm{mse}(y):=\frac1n\sum_{i=1}^n (y_i-\bar y)^2$ (empirical
  variance), so $\mathrm{mse}\ge 0$.
- MAE impurity: $\mathrm{mae}(y):=\frac1n\sum_{i=1}^n |y_i-\mathrm{median}(y)|$,
  so $\mathrm{mae}\ge 0$.

**Assumptions (local).** Finite samples; weights $w_L,w_R\ge 0$ sum to 1; MAE
defined as $L^1$ risk over constants.

**Lemma A.5 (bounds and impurity-decrease nonnegativity for common splitters).**

1. (**Bounds, classification**) For any probability vector $p\in\Delta_K$,

   $$
   0\le \mathrm{gini}(p) \le 1-\frac{1}{K}
   \quad\text{and}\quad
   0\le \mathrm{ent}(p) \le \log_2 K.
   $$

2. (**Nonnegative impurity decrease, classification**) For a node with empirical
   class distribution $p$ and children distributions $p_L,p_R$ with weights
   $w_L,w_R\ge 0$ summing to 1, we have

   $$
   \mathrm{gini}(p)\;\ge\; w_L\,\mathrm{gini}(p_L)+w_R\,\mathrm{gini}(p_R),
   \quad
   \mathrm{ent}(p)\;\ge\; w_L\,\mathrm{ent}(p_L)+w_R\,\mathrm{ent}(p_R).
   $$

   Therefore the CART-style weighted impurity decrease
   $\Delta I := I(p) - (w_L I(p_L)+w_R I(p_R))$ is always nonnegative for
   $I\in\{\mathrm{gini},\mathrm{ent}\}$.

3. (**Nonnegative impurity decrease, MSE**) For a regression node with targets
   $Y$ and child targets $Y_L,Y_R$ with weights $w_L=n_L/n$ and $w_R=n_R/n$, the
   variance decomposition gives

   $$
   \mathrm{mse}(Y)
   \;=\;
   w_L\,\mathrm{mse}(Y_L)+w_R\,\mathrm{mse}(Y_R)
   \;+\; w_L w_R(\bar Y_L-\bar Y_R)^2,
   $$

   so the MSE impurity decrease is always nonnegative.

4. (**Nonnegative impurity decrease, MAE**) Define the MAE node impurity as the
   minimal $L^1$ risk over constants:
   $$
   \mathrm{mae}(Y) := \min_{c\in\mathbb{R}} \frac{1}{n}\sum_{i=1}^n |Y_i-c|,
   $$
   which is attained by any median. For a binary split with weights $w_L,w_R$,
   we have
   $$
   \mathrm{mae}(Y)\;\ge\; w_L\,\mathrm{mae}(Y_L)+w_R\,\mathrm{mae}(Y_R),
   $$
   so the corresponding MAE impurity decrease is always nonnegative.

**Proof.**

1. For Gini, $\sum_{k=1}^K p_k^2 \ge 1/K$ by Cauchy–Schwarz, with equality at
   the uniform distribution, yielding $\mathrm{gini}(p)\le 1-1/K$. Nonnegativity
   is immediate since $\sum_k p_k^2\le \sum_k p_k = 1$. For entropy,
   nonnegativity is standard and the maximum $\log_2 K$ occurs at the uniform
   distribution (e.g., by concavity of entropy and Jensen’s inequality).

2. The parent class distribution satisfies $p = w_L p_L + w_R p_R$. Both Gini
   and entropy are concave on $\Delta_K$, so
   $I(w_L p_L + w_R p_R)\ge w_L I(p_L)+w_R I(p_R)$ for
   $I\in\{\mathrm{gini},\mathrm{ent}\}$.

3. The variance decomposition is the standard identity
   $\mathrm{Var}(Y)=\mathbb{E}[\mathrm{Var}(Y\mid G)] + \mathrm{Var}(\mathbb{E}[Y\mid G])$
   for a two-group indicator $G$, applied to the empirical distribution within
   the node.

4. For any constant $c$, we have
   $$
   \begin{aligned}
   w_L\,\mathrm{mae}(Y_L)+w_R\,\mathrm{mae}(Y_R)
   &\le w_L\cdot \frac{1}{n_L}\sum_{i\in L}|Y_i-c|
   + w_R\cdot \frac{1}{n_R}\sum_{i\in R}|Y_i-c| \\
   &= \frac{1}{n}\sum_{i=1}^n |Y_i-c|.
   \end{aligned}
   $$
   Taking the minimum over $c$ on the right-hand side yields
   $w_L\,\mathrm{mae}(Y_L)+w_R\,\mathrm{mae}(Y_R)\le \mathrm{mae}(Y)$, as
   claimed. $\square$

#### A.6.6 Why `mi` cannot be in multi-selector mode (scale incompatibility)

The multi-selector mode takes a maximum over selector scores. This only makes
sense when the scores are on a common scale. In citrees:

- `mc`, `pc` (after absolute value), `dc`, and `rdc` are bounded in $[0,1]$,
- for classification, population mutual information satisfies
  $0\le I(X;Y)\le H(Y)\le \log K$ (units depend on the log base), so it is not
  normalized to $[0,1]$ and its scale depends on $K$ and on the entropy of the
  class labels,

so including `mi` in a max-with-others selector list would change the meaning of
the maximum and would require additional normalization or theory.

---

## Appendix B. Feature muting analysis (optional)

This appendix concerns **feature muting**, which is a computational heuristic
rather than a core inferential component of citrees.

citrees implements **feature muting** as a computational heuristic: after
testing a feature at a node, if the p-value is non-significant at the nodewise
threshold (by default, `p >= alpha`), the feature is removed from the candidate
set for **descendants of that node** (subtree-local propagation; siblings are
isolated), provided there is at least one other candidate feature remaining.

**Implementation note (sibling isolation / traversal-order invariance).** Muting
(and constant-feature filtering) must be applied to candidate sets that are
local to the subtree. Otherwise, muting decisions in one child can remove
candidates from its sibling, making the fitted tree depend on traversal order
(or parallel schedule). The unit test
`tests/unit/test_tree.py::TestCandidateSetIsolation` guards this behavior.

This section explains why **aggressive global screening** (e.g., a one-shot root
decision to never consider a feature again) can be dangerous under
conditional/interaction effects, and motivates keeping muting conservative and
treating it as a speed-only heuristic (compare `feature_muting=True` vs
`False`).

### B.1 The gated effect model

Consider a simple data-generating process where feature $X_1$ has predictive
power only within a subset of the data:

$$
X_0, X_1 \sim \mathcal{N}(0, 1) \text{ independent}, \quad
Z = \mathbf{1}\{X_0 > c\} \text{ with } p := P(Z=1), \quad
Y = \begin{cases} \mathbf{1}\{X_1 > 0\} & \text{if } Z=1 \\ \text{Bernoulli}(1/2) & \text{if } Z=0 \end{cases}
$$

Feature $X_1$ is informative about $Y$ **only in the gated subset** where $Z=1$.
At the root (full sample), the signal from $X_1$ is diluted by the $(1-p)$
fraction of noise observations.

### B.2 Population correlations

**Lemma (Population Moments).** Let $\rho_{\text{root}} := \text{Corr}(X_1, Y)$
and $\rho_{\text{gate}} := \text{Corr}(X_1, Y \mid Z=1)$. Then:

$$
\rho_{\text{root}} = \frac{2p}{\sqrt{2\pi}} \approx 0.798\,p, \qquad
\rho_{\text{gate}} = \frac{2}{\sqrt{2\pi}} \approx 0.798.
$$

**Key insight:** The root correlation scales linearly with $p$, while the gate
correlation is constant (~0.798). For small $p$, the root signal becomes
undetectable while the gate signal remains strong.

### B.3 Approximate power functions (Fisher z approximation)

Using the Fisher z-transformation with bias correction (Hotelling, 1953), we
compute an **approximate** power curve for the two-sided correlation test. (The
exact finite-sample power uses a noncentral-$t$ distribution.)

**Critical value:** For sample size $n$ and significance level $\alpha$:

$$
r_\alpha(n) = \frac{t_{n-2,\,1-\alpha/2}}{\sqrt{t_{n-2,\,1-\alpha/2}^2 + (n-2)}}
$$

**Power function:** For population correlation $\rho$:

$$
\pi(\rho; n, \alpha) = 1 - \Phi\!\left(\frac{z_\alpha - \zeta}{\sigma_z}\right) + \Phi\!\left(\frac{-z_\alpha - \zeta}{\sigma_z}\right)
$$

where $z_\alpha = \tanh^{-1}(r_\alpha)$,
$\zeta = \tanh^{-1}(\rho) + \frac{\rho}{2(n-1)}$, and $\sigma_z = 1/\sqrt{n-3}$.

**Application to gated model:**

- Root power: $\pi_{\text{root}}(p; n, \alpha) = \pi(0.798p;\, n,\, \alpha)$
- Gate power:
  $\pi_{\text{gate}}(p; n, \alpha) = \pi(0.798;\, \lfloor np \rfloor,\, \alpha)$

### B.4 Gap region: where local muting succeeds but global fails

Define the **gap region** as the set of gate probabilities where global muting
fails (root power $\le \beta_L$) but local muting succeeds (gate power
$\ge \beta_H$):

$$
\mathcal{G}(n, \alpha, \beta_L, \beta_H) = \{p : \pi_{\text{root}}(p) \le \beta_L \text{ and } \pi_{\text{gate}}(p) \ge \beta_H\}
$$

**Table: Gap Region Boundaries** ($\alpha=0.05$, $\beta_L=0.2$, $\beta_H=0.8$)

|     n | $p_{\min}$ | $p_{\max}$ | Ratio | Width |
| ----: | ---------: | ---------: | ----: | ----: |
|   100 |      0.100 |      0.141 |   1.4 | 0.041 |
|   500 |      0.020 |      0.063 |   3.1 | 0.043 |
| 1,000 |      0.010 |      0.044 |   4.4 | 0.034 |
| 2,000 |      0.005 |      0.031 |   6.2 | 0.026 |
| 5,000 |      0.002 |      0.020 |   9.9 | 0.018 |

The gap exists for $n \ge 50$ and widens (in ratio) as $n$ increases.

### B.5 Tree depth propagation (heuristic)

As the tree grows deeper, sample sizes at internal nodes decrease (roughly
halving at each level for balanced splits), which affects detection power.

**Critical depth:** The maximum depth at which the gated signal remains
detectable is approximately:

$$
d_{\max} \approx 2\log_2\!\left(\frac{0.798\sqrt{np}}{z_{1-\alpha/2}}\right)
$$

For $n=2000$, $p=0.05$: $d_{\max} \approx 4.1$, meaning the signal is detectable
at depths 0–4 but not beyond. This calculation uses the Fisher‑$z$ approximation
and a balanced‑split heuristic for node sizes; it is included for intuition, not
as a formal guarantee.

### B.6 Practical implications

1. **When to use local muting:** If your application involves features with
   **conditional effects** (informative only in subsets of the data), be
   cautious with any muting heuristic. In such settings, consider disabling
   muting (`feature_muting=False`) and comparing results, since
   conditionally-informative features can look null on larger mixed samples.

2. **Diagnostic:** Compare trees with `feature_muting=True` vs
   `feature_muting=False`. Large discrepancies in which features appear may
   indicate conditional effects being missed by muting.

3. **Sample size requirements:** Even with local muting, sufficient samples are
   needed in the gated branch. If $np < 20$ (expected gate size), detection
   power may be inadequate regardless of muting scope.

### B.7 Reproducibility

The power calculations above are implemented in
`paper/scripts/theory/theoretical_predictions.py`. To reproduce the gap region
table:

```bash
UV_CACHE_DIR=$PWD/.uv-cache uv run python paper/scripts/theory/theoretical_predictions.py
```

Run status (2026-01-21): executed; console tables printed (a soft-gate
calibration step emits a benign overflow warning in `exp` for large sharpness
values).

**Optional empirical validation (Appendix B only).** The Monte Carlo check in
`paper/scripts/theory/muting_power_gap.py` writes its outputs to
`paper/results/theory/`. A quick run was executed on 2026-01-21 (`--quick`); it
completed with expected numpy warnings from correlation calculations on small
samples and produced `muting_power_gap_raw.parquet` and
`muting_power_gap_summary.parquet`.

---

## Appendix C. TODOs

1. Decide whether to mention adaptive early stopping in the main paper at all;
   if included, treat it as an engineering heuristic backed by calibration
   figures rather than optional-stopping p-value theorems.
2. (Optional) Standardize citations to use `paper/arxiv/references.bib` keys
   consistently across `paper/notes/*.md`.

---

## Appendix D. Honest estimation (optional)

citrees optionally uses sample splitting (“honesty”) to decouple structure
learning from leaf estimation. This appendix records the clean conditional
unbiasedness statement under an **independent** sample split; honesty is not a
core focus of this paper.

### D.1 Setup

Split the indices $\{1,\dots,n\}$ into disjoint sets $S$ (“splitting”) and $E$
(“estimation”) using a random split. Build the tree structure (the partition of
feature space into leaves) using only data indexed by $S$. Let $\Pi$ denote the
resulting partition into leaves (a random object measurable w.r.t. the
$\sigma$-field generated by $\{(X_i,Y_i)\}_{i\in S}$).

**Assumption (independent sample split).** Assume the random index split $(S,E)$
is independent of the observed sample $\{(X_i,Y_i)\}_{i=1}^n$. Under this
assumption, conditional on $S$ the splitting-sample data
$\{(X_i,Y_i)\}_{i\in S}$ is independent of the estimation-sample data
$\{(X_i,Y_i)\}_{i\in E}$. Since the learned partition $\Pi$ is measurable with
respect to $\sigma(S, \{(X_i,Y_i)\}_{i\in S})$, we have the conditional
independence

$$
\{(X_i,Y_i)\}_{i\in E} \perp \Pi \mid S.
$$

(We do not generally have $E \perp \Pi$ unconditionally because $\Pi$ depends on
which indices are placed in $S$.)

For a leaf (cell) $L \in \Pi$, define the estimation indices landing in that
leaf

$$
E(L) := \{ i \in E : X_i \in L \}.
$$

### D.2 Unbiasedness conditional on the learned partition

**Proposition D.1 (honest leaf mean is unbiased, regression).** Assume i.i.d.
sampling and an independent sample split $(S,E)$ as in Section D.1. Consider
regression, and define the honest leaf estimator

$$
\widehat{\mu}(L) := \frac{1}{|E(L)|}\sum_{i\in E(L)} Y_i,
$$

on the event $\{|E(L)| \ge 1\}$. Let the target parameter be

$$
\mu(L) := \mathbb{E}[Y \mid X \in L].
$$

Then

$$
\mathbb{E}\!\left[\widehat{\mu}(L)\;\middle|\;\Pi\right] = \mu(L)
\quad \text{on the event } \{|E(L)| \ge 1\}.
$$

**Proof.** By Section D.1, $\{(X_i,Y_i)\}_{i\in E} \perp \Pi \mid S$. In
particular, conditional on $(\Pi,S)$ the estimation-sample observations are
i.i.d. from $P$ and independent of the (random) leaf partition.

Fix a leaf $L\in \Pi$ and, for $i\in E$, define the indicator
$I_i := \mathbf{1}\{X_i\in L\}$ and the random count $N:=\sum_{i\in E} I_i$.

On the event $\{N\ge 1\}$ we can write the honest mean as a ratio

$$
\widehat{\mu}(L)=\frac{\sum_{i\in E} I_i Y_i}{\sum_{i\in E} I_i}.
$$

Condition on $(\Pi,S,(I_i)_{i\in E})$. Then $N$ is fixed, and for every $i$ with
$I_i=1$ we have

$$
\mathbb{E}[Y_i\mid \Pi,S,I_i=1]=\mathbb{E}[Y\mid X\in L]=\mu(L),
$$

because $I_i=1$ is the event $\{X_i\in L\}$ and $L$ is fixed given $\Pi$.
Therefore, on $\{N\ge 1\}$,

$$
\mathbb{E}\!\left[\widehat{\mu}(L)\;\middle|\;\Pi,S,(I_i)_{i\in E}\right]
=\frac{1}{N}\sum_{i\in E:I_i=1}\mathbb{E}[Y_i\mid \Pi,S,I_i=1]
=\mu(L).
$$

Taking conditional expectations first over $(I_i)_{i\in E}$ and then over $S$
yields $\mathbb{E}[\widehat{\mu}(L)\mid \Pi]=\mu(L)$ on $\{N\ge 1\}$. $\square$

**Classification analogue (requires independent split).** If the index split
$(S,E)$ is independent of the data, then for classification the honest leaf
class-probability vector

$$
\widehat{p}_k(L) := \frac{1}{|E(L)|}\sum_{i\in E(L)} \mathbf{1}\{Y_i = k\}
$$

is similarly unbiased for $p_k(L) := \mathbb{P}(Y=k \mid X\in L)$, conditional
on $\Pi$, on $\{|E(L)|\ge 1\}$.

**Important implementation note.** If a leaf receives zero estimation samples,
citrees currently retains the splitting-sample leaf value. That fallback is not
covered by Proposition D.1.

### D.3 Variance (regression)

**Proposition D.2 (variance of honest leaf estimator, regression).** Under the
assumptions of Proposition D.1, let $n_E(L) := |E(L)|$ be the number of
estimation samples in leaf $L$. Define
$\sigma^2(L) := \mathrm{Var}(Y \mid X \in L)$. Then, conditional on $\Pi$ and on
$\{n_E(L) = n\}$ for some $n \ge 1$:

$$
\mathrm{Var}(\widehat{\mu}(L) \mid \Pi, n_E(L) = n) = \frac{\sigma^2(L)}{n}.
$$

**Proof.** Use the notation $I_i=\mathbf{1}\{X_i\in L\}$ and
$N=\sum_{i\in E}I_i$ from Proposition D.1.

Condition on $(\Pi,S,(I_i)_{i\in E})$. On the event $\{N=n\ge 1\}$, the
variables $\{Y_i: i\in E, I_i=1\}$ are independent with common distribution
$(Y\mid X\in L)$, hence

$$
\mathrm{Var}\!\left(\widehat{\mu}(L)\;\middle|\;\Pi,S,(I_i)_{i\in E}\right)=\sigma^2(L)/n
\quad\text{on }\{N=n\ge 1\}.
$$

Now condition only on $(\Pi, N=n)$ and apply the law of total variance with the
refinement $(S,(I_i)_{i\in E})$:

$$
\mathrm{Var}(\widehat{\mu}(L)\mid \Pi,N=n)
=\mathbb{E}\!\left[\mathrm{Var}\!\left(\widehat{\mu}(L)\;\middle|\;\Pi,S,(I_i)_{i\in E}\right)\;\middle|\;\Pi,N=n\right]
\,+\,\mathrm{Var}\!\left(\mathbb{E}[\widehat{\mu}(L)\mid \Pi,S,(I_i)_{i\in E}]\;\middle|\;\Pi,N=n\right).
$$

The first term equals $\sigma^2(L)/n$ and the second term is $0$ by Proposition
D.1. $\square$

**Bias–variance trade-off (rigorous part + intuition).** Honesty reduces
adaptive bias in leaf _estimation_ by using an estimation sample that is
independent of the partition-learning step. Propositions D.1–D.2 make the
variance statement precise: for a fixed leaf $L$ and $n_E(L)=n\ge 1$, the honest
mean has conditional variance $\sigma^2(L)/n$.

Using fewer observations for estimation typically increases variance. If an
alternative estimator used $n_{\mathrm{all}}(L)$ i.i.d. observations from
$(Y\mid X\in L)$, its variance would be $\sigma^2(L)/n_{\mathrm{all}}(L)$, so
moving from $n_{\mathrm{all}}(L)$ to $n_E(L)$ inflates variance by the factor
$n_{\mathrm{all}}(L)/n_E(L)\ge 1$.

---

## Appendix E. Notes for writing (citations, scope, and positioning)

This appendix is not paper-facing prose. It is a working checklist to keep
claims and citations aligned with what is actually proved in Appendix A (and
what is empirical).

### E.1 Citations to include (placeholders)

Add BibTeX entries for:

- Conditional inference trees: Hothorn, Hornik, Zeileis (2006)
- Linear-statistic permutation tests used in the ctree lineage: Strasser & Weber
  (1999) (optional)
- +1 correction for Monte Carlo permutation tests: Phipson & Smyth (2010)
- Multiple testing background (optional): Holm (1979) and/or a standard
  reference
- RDC: Lopez-Paz, Hennig, Schölkopf (2013)
- Distance correlation: Székely, Rizzo, Bakirov (2007)
- Mutual information estimator (if used): Kraskov, Stögbauer, Grassberger (2004)
  (KSG)
- Honesty / honest forests (if mentioned): Wager & Athey (2018); Athey,
  Tibshirani, Wager (2019)

### E.2 Implementation-to-theory alignment checklist (paper-facing)

Before making inferential claims, decide what you will support:

1. **Permutation p-value validity.**
   - Use `early_stopping_* = None` for fixed-$B$ p-values so Theorem 1 applies
     directly.
   - If using `early_stopping_* = "adaptive"` for speed, interpret results via
     the accept/reject rule (not as a fixed-$B$ p-value at a stopping time).
   - Avoid `early_stopping_* = "simple"` for inferential claims.
   - If `n_resamples_* = None`, Stage A/B do not produce p-values; do not attach
     inferential claims.

2. **Multiplicity correction matches the tested family.**
   - If subsampling features (`max_features`) or thresholds (`max_thresholds`),
     be explicit: error control is over the tested subset, not over all $p$
     features / all thresholds.

3. **Phipson–Smyth +1 correction everywhere.**
   - All permutation tests should use the +1 convention `(1 + count)/(1 + B)`.

4. **Honesty claims match the sampling scheme.**
   - Proposition D.1 assumes the index split $(S,E)$ is independent of the
     observed data.

### E.3 Safe claims (copy/paste checklist)

The items below are intended to be safe to include without over-claiming:

1. **Permutation p-value validity (finite sample).** The +1 Monte Carlo
   permutation p-value is super-uniform under the null (Theorem 1).

2. **Stage A family-wise error at a fixed node (global null).** If all tested
   features at a node are null, Bonferroni gives
   $\mathbb{P}(\text{select any feature at that node}) \le \alpha_{\text{sel}}$
   (Proposition 3).

3. **Per-feature false selection bound (partial null).** For any particular null
   feature in the tested family,
   $\mathbb{P}(\text{node splits on that feature}) \le \alpha_{\text{sel}}/m$
   (Proposition 3a).

4. **Root-level global-null bound on any split.** Under a global null at the
   root across tested features, the probability the learned tree has any split
   is at most $\alpha_{\text{sel}}$ (Proposition 4).

5. **(Optional) Honest estimation unbiasedness.** If `honesty=True` and the
   sample split is independent of the observed data, leaf means are unbiased
   conditional on the learned partition on leaves that receive estimation
   samples (Proposition D.1).

### E.4 Relationship to conditional inference trees (ctree lineage; positioning notes)

citrees is inspired by the conditional inference tree framework of Hothorn,
Hornik, and Zeileis (2006), which selects splitting variables via hypothesis
tests derived from permutation invariance (in that lineage, often using linear
statistics; see also Strasser & Weber, 1999).

citrees follows the same high-level principle—test-based variable selection to
mitigate selection bias—but plugs in different association scores (e.g., `mc`,
`mi`, `rdc`, `dc`, `pc`) and computes Monte Carlo permutation p-values (with the
Phipson–Smyth +1 correction) for these statistics.

### E.5 Conditional inference vs. marginal inference (notes)

In citrees, the natural framework for permutation tests is conditional on the
covariates (and label-independent algorithmic randomness) treated as fixed by
the permutation scheme:

$$
p \approx \mathbb{P}\!\left(T(X_{t,j}, \pi(Y_t)) \ge T(X_{t,j}, Y_t)\;\middle|\; X_t,U\right).
$$

**Exchangeability vs. independence.** Permutation validity requires
exchangeability under the null, not necessarily i.i.d. sampling in full
generality. In the standard i.i.d. setting with $X\perp Y$, exchangeability
conditional on $X$ holds automatically.

### E.6 Computational considerations (notes; not theory claims)

**Early stopping modes.**

- `early_stopping="adaptive"`: posterior-confidence stopping for accept/reject
  decisions; treat as a computational heuristic (do not interpret returned
  $\widehat p_\tau$ as fixed-$B$ p-values).
- `early_stopping=None`: fixed-$B$ permutation tests (cleanest mode for
  paper-facing statements).
- `early_stopping="simple"`: a baseline heuristic; avoid for inferential claims.

**Choosing `n_resamples`.** For paper-facing claims tied to Theorem 1 and
Bonferroni, use fixed-$B$ (disable early stopping) and report $B$. For
speed-focused experiments, `n_resamples="auto"` with adaptive early stopping can
reduce compute (treated as an engineering choice; calibrated empirically).

**Time complexity per node (back-of-envelope).**

- Stage A: $O(m_t \cdot B \cdot n_t)$ for $m_t$ tested features and $B$
  permutations at node size $n_t$ (ignoring constant factors from the selector
  statistic).
- Stage B: $O(|C_{t,j^\star}|\cdot B \cdot n_t)$ for the tested threshold family
  on the selected feature.

**Parallelization.** citrees parallelizes across trees (forest) and can
parallelize permutation loops for fixed-$B$ tests (Numba `prange`).

### E.7 Relationship to other methods (notes; for related work sections)

**Comparison with R’s `partykit::ctree`.** citrees is inspired by Hothorn et al.
(2006) but differs in implementation and defaults:

| Aspect             | `partykit::ctree`               | `citrees`                                        |
| ------------------ | ------------------------------- | ------------------------------------------------ |
| Test statistic     | Linear statistics               | Multiple options (`mc`, `mi`, `rdc`, `pc`, `dc`) |
| P-values           | Often asymptotic approximations | Monte Carlo permutation (+1)                     |
| Multiple selectors | Typically single                | Optional max-statistic within permutations       |
| Implementation     | R                               | Python (NumPy + Numba)                           |

**Comparison with generalized random forests (GRF).** GRF targets causal
inference and asymptotic inference for treatment effects; citrees targets
prediction/selection and nodewise screening validity for permutation tests. If
we mention GRF, keep the goals distinct (and avoid implying causal inference is
a core feature of this paper).

### E.8 Future directions (expanded notes)

1. **Selective inference for internal nodes.** Develop valid post-selection
   inference for non-root nodes and Stage B.
2. **Tree-wide multiple testing.** Extend nodewise/root control to a tree-wide
   error notion (FWER/FDR).
3. **Acceleration.** GPU kernels / batching for permutation tests; shared
   permutations across features/thresholds.

---

## Appendix F. Claim tracker (internal)

This appendix is an internal audit log to keep the manuscript mathematically
honest.

**Goal:** Every claim that matters for the paper should be either:

- **PROVED** in this manuscript text (with explicit assumptions and scope), or
- **CITED** to an external reference (`paper/arxiv/references.bib`), or
- **EMPIRICAL** with a reproducible script + committed figure/table output under
  `paper/results/` (with optional parquet caches under `paper/results/cache/`).

If a claim is none of the above, mark it **TODO** (or move it to Appendix E as a
future direction).

### F.1 Fixed-$B$ permutation p-values

**Assumption note.** All A-series claims below are fixed-node, fixed-$B$
statements and rely on the assumption ledger in Appendix A.1.1 (especially
A0.1–A0.4). Where additional conditions are needed (e.g., bounded impurity),
they are stated in the corresponding proofs.

|  ID | Claim (short)                                                                                                                                                                                                                                                                                                                          | Where in paper.md             | Verification                                                                                | Status                            |
| --: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------- | ------------------- | ----------------------------- | ------------------ | ------ |
|  A1 | +1 Monte Carlo permutation p-value is super-uniform under exchangeability                                                                                                                                                                                                                                                              | Appendix A.2 (Theorem 1)      | Proof by exchangeable-rank argument; +1 convention matches Phipson & Smyth (2010)           | PROVED + CITED                    |
| A1a | Under tie-free null, exact achieved level is $\lfloor (B+1)u\rfloor/(B+1)$ when thresholding at $u$; Bonferroni inherits additional “floor” conservativeness                                                                                                                                                                           | Appendix A.2.1                | Theorem 1 discrete-uniform CDF + union bound                                                | PROVED                            |
|  A2 | Conditional-on-$(X,U)$ super-uniformity (candidate-set randomness allowed if label-independent)                                                                                                                                                                                                                                        | Appendix A.2 (Corollary 1)    | Conditioning argument + A1                                                                  | PROVED                            |
|  A3 | Bonferroni FWER control with super-uniform p-values (no independence required)                                                                                                                                                                                                                                                         | Appendix A.3 (Lemma 2)        | Union bound                                                                                 | PROVED                            |
|  A4 | “Any split implies root Stage A rejection” $\Rightarrow$ under global null at root, $\Pr(\text{tree splits}) \le \alpha_{\text{sel}}$                                                                                                                                                                                                  | Appendix A.3 (Proposition 4)  | Set inclusion + A3 at root                                                                  | PROVED                            |
|  A6 | Forest-level root-null bounds: $\mathbb{E}[\#\text{splitting trees}] \le M\alpha_{\text{sel}}$ and $\Pr(\text{any tree splits}) \le M\alpha_{\text{sel}}$ (union bound), plus per-feature root split count $\le M\alpha_{\text{sel}}/p$ under uniform `max_features`                                                                   | Appendix A.3 (Corollary 4a)   | Linearity + union bound + A4 + Corollary 3d                                                 | PROVED                            |
|  A7 | Under bounded classification impurities (Gini/entropy), expected root impurity decrease attributable to a null feature is bounded by $I_{\max}\cdot\Pr(\text{root splits on }j)$, hence $\le I_{\max}\alpha_{\text{sel}}/m_{\text{root}}$ (tested set) or $\le I_{\max}\alpha_{\text{sel}}/p$ (uniform `max_features` + complete null) | Appendix A.3 (Proposition 3g) | Bounding $\Delta I\le I_{\max}$ + root split-probability bounds                             | PROVED (under stated assumptions) |
| A7b | For any nonnegative impurity (e.g., regression MSE/MAE), expected root impurity-importance contribution of a null feature admits the moment bound $\mathbb{E}[Z_j]\le (\mathbb{E}[I(Y_{\text{root}})^2])^{1/2}\sqrt{\Pr(\text{root splits on }j)}$                                                                                     | Appendix A.3 (Proposition 3h) | Pointwise $\Delta I\le I(Y_{\text{root}})$ + Cauchy–Schwarz + root split-probability bounds | PROVED (under stated assumptions) |
|  A8 | Bonferroni also bounds expected false rejections: for $R=\{j:p_j\le \alpha/m\}$, $\mathbb{E}[                                                                                                                                                                                                                                          | R\cap\mathcal H_0             | ]\le \alpha$ and $\Pr(                                                                      | R\cap\mathcal H_0                 | \ge r)\le \alpha/r$ | Appendix A.3 (Lemma 2 remark) | Linearity + Markov | PROVED |
|  A5 | Multi-selector “max statistic inside each permutation” yields valid p-values                                                                                                                                                                                                                                                           | Appendix A.2.3                | Composite statistic preserves exchangeability; cite max-T framing (Westfall & Young, 1993)  | PROVED + CITED                    |

**Empirical backstops (optional).**

- `paper/scripts/theory/generate_fixedB_pvalue_calibration.py` →
  `paper/results/cache/fixedB_pvalue_calibration_data.parquet` and
  `paper/results/figures/fixedB_pvalue_calibration.png`.
- `paper/scripts/theory/generate_selection_bias_demo.py` →
  `paper/results/cache/selection_bias_demo_data.parquet` and
  `paper/results/figures/selection_bias_demo.png`.
- `paper/scripts/theory/generate_sequential_stopping_calibration.py` →
  `paper/results/cache/sequential_stopping_calibration_data.parquet` and
  `paper/results/figures/sequential_stopping_calibration.png`. Run status
  (2026-01-20): scripts executed under the prior defaults; defaults were
  increased on 2026-01-21, so rerun is needed to refresh the cached artifacts
  with the larger settings.

### F.2 Stage B and adaptive tree growth (what we do _not_ claim)

|  ID | Claim (short)                                                                                                                                                         | Where in paper.md | Verification                                                                                                         | Status |
| --: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------- | ------ |
|  B1 | Stage B p-values are post-selection and should not be presented as classical p-values without selective inference / splitting                                         | Appendix A.4      | Cite post-selection / selective-inference warnings (e.g., Berk et al., 2013; Lee et al., 2016; Fithian et al., 2014) | CITED  |
|  B2 | Internal-node tests in an adaptively grown tree are not classical p-values for a fixed family (conditioning on data-dependent node membership breaks exchangeability) | Appendix A.4      | Cite adaptive-data-analysis / post-selection cautions (e.g., Dwork et al., 2015; Leeb & Pötscher, 2015)              | CITED  |

### F.3 Early stopping (what we do _not_ claim)

We keep early stopping (`early_stopping_*!="None"`) as an engineering feature,
but the paper’s p-value claims are made only in fixed-$B$ mode
(`early_stopping_*=None`) via Theorem 1 and the Stage A/root consequences
derived from it.

### F.4 Critical review / risk register (be intentionally picky)

This section is written in the style of an adversarial reviewer. Each item is
either an assumption that must be stated explicitly, or a place where the
mathematics could be misread as stronger than it is.

1. **Exact scope of theorems (fixed node vs adaptive tree).**
   - Airtight: fixed-node permutation p-values conditional on $(X,U)$ (Appendix
     A.2 and Corollary 1), especially at the root.
   - Not airtight: interpreting internal-node tests as classical p-values for a
     fixed family in an adaptively grown tree (B2).
   - Paper action: keep all “valid p-value” language explicitly fixed-node/root
     scoped.

2. **Stage B is post-selection (“double dipping”).**
   - Stage B reuses the same labels after selecting $j_t^\star$ in Stage A, so
     naïve “Stage B p-value” interpretation is post-selection (B1).
   - Paper action: treat Stage B p-values as algorithmic
     split-validation/stopping statistics unless we add sample splitting or
     selective-inference machinery.

3. **Early stopping is not classical inference.**
   - Early stopping returns a +1 Monte Carlo estimate at a data-dependent time;
     we do not interpret it as a classical fixed-$B$ permutation p-value under
     optional stopping.
   - Paper action: never write “adaptive mode yields valid p-values”; make all
     paper-facing p-value claims in fixed-$B$ mode.

4. **Candidate-set randomness must be label-independent for clean validity.**
   - Root-level random subsets of features/thresholds are safe if chosen as
     functions of $(X,U)$, independent of $Y$ under the null (Appendix A.2).
   - Feature muting / scanning can induce label-dependent candidate families
     later in the tree.
   - Paper action: do not “import” fixed-node validity into adaptive-tree-wide
     claims.

5. **What “unbiased feature selection” means.**
   - “Unbiased” here means “no systematic preference for high-cardinality noise
     under a symmetric null,” not an “unbiased estimator” claim in a parametric
     sense.
   - Paper action: define terminology precisely (Section 3.4) and avoid
     overloaded language.

6. **No-permutation mode is not inferential.**
   - If `n_resamples_* = None`, Stage A/B skip permutation tests and select by
     raw scores/impurity only.
   - Paper action: do not attach p-value or error-control language to this mode.

7. **Label-dependent resampling is outside the fixed-node theory.**
   - In forests, stratified/balanced bootstraps use $Y$ to choose indices; our
     fixed-node guarantees are stated conditional on the realized sample at a
     node, and we do not add new theory for the resampling mechanism itself.

### F.5 Motivation: selection bias in CART (idealized)

|  ID | Claim (short)                                                                                                                                                  | Where in paper.md        | Verification                                                                                                             | Status   |
| --: | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------ | -------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------- |
|  F0 | Under a simple “many thresholds” idealization, $P(\max_{c\in C_j}\Delta_{j,c}\ge \delta)$ grows with $                                                         | C_j                      | $ (union bound always; exact formula under independence)                                                                 | Lemma 3a | Union bound; independence product formula                                                                   | PROVED (under stated assumptions) |
|  F1 | Under an exchangeable/tie-free null idealization, greedy CART selects feature $j$ with probability proportional to its number of candidate splits $            | C_j                      | $                                                                                                                        | Lemma 3b | Symmetry / exchangeability: argmin index uniform over all candidate splits                                  | PROVED                            |
| F1a | Same as F1, but without tie-free assumption if ties are broken by an independent random perturbation                                                           | Lemma 3b'                | Add i.i.d. $\mathrm{Unif}(0,1)$ perturbations to make argmin tie-free; symmetry argument                                 | PROVED   |
| F1b | Under the same exchangeable/tie-free null idealization, expected root impurity-importance contribution of feature $j$ is proportional to its candidate count $ | C_j                      | $                                                                                                                        | Lemma 3c | Symmetry: argmax index uniform; conditional distribution of $\Delta_{\max}$ does not depend on argmax index | PROVED (under stated assumptions) |
| F1c | Even with random feature subsampling (e.g., mtry), CART allocates selection probability in proportion to candidate split counts within the tested subset       | Lemma 3b remark (“mtry”) | Conditional symmetry on the tested subset                                                                                | PROVED   |
|  F2 | Under an exchangeable/tie-free null idealization, Stage A selects a tested feature uniformly (and remains uniform conditional on rejecting)                    | Lemma 3e                 | Symmetry / exchangeability: argmin index uniform over tested features; symmetric conditioning event preserves uniformity | PROVED   |
| F2a | Same as F2, but without tie-free assumption if ties are broken by an independent random perturbation                                                           | Lemma 3e'                | Random tie-breaking + symmetry argument                                                                                  | PROVED   |
| F2b | Under an exchangeable-null idealization, Stage A remains uniform over all $p$ features even with uniform random `max_features` subsampling                     | Lemma 3f                 | Condition on tested subset + symmetry; average over uniform subset                                                       | PROVED   |

## Appendix G. Algorithms (pseudocode aligned to implementation)

This appendix provides concise pseudocode and notation aligned to `citrees/`.
Proofs and careful scope statements are in Appendix A and Appendix F.

### G.1 Notation

Training data:

$$
\{(X_i, Y_i)\}_{i=1}^n,\quad X_i \in \mathbb{R}^p.
$$

At a node $t$, let $I_t \subseteq \{1,\dots,n\}$ be the sample indices reaching
that node and write $X_t := (X_i)_{i\in I_t}$, $Y_t := (Y_i)_{i\in I_t}$, with
$n_t := |I_t|$.

Candidate families:

- Feature candidates: $F_t \subseteq \{1,\dots,p\}$, with $m_t := |F_t|$.
- Threshold candidates (for a chosen feature $j$): $C_{t,j} \subset \mathbb{R}$,
  with $\ell_{t,j} := |C_{t,j}|$.

Significance thresholds:

- Stage A (feature screening): $\alpha_{\text{sel}}$.
- Stage B (split screening): $\alpha_{\text{split}}$.
- When Bonferroni is enabled, use $\alpha_{\text{sel}}/m_t$ and
  $\alpha_{\text{split}}/\ell_{t,j}$.

### G.2 Test statistics used in citrees

#### G.2.1 Stage A (feature–response association)

Each selector defines a nodewise association statistic
$T^{\text{sel}}_j(X_{t,j}, Y_t)$ with the convention “larger is more extreme”
(right-tail test). Examples used in code:

- `mc` (classification): correlation ratio $\eta=\sqrt{SS_B/SS_T}\in[0,1]$
- `pc` (regression): $|{\rm Corr}(X_{t,j},Y_t)|\in[0,1]$
- `dc` (regression): distance correlation $\in[0,1]$
- `rdc` (both): randomized dependence coefficient $\in[0,1]$
- `mi` (classification): mutual information $\in[0,H(Y)]$ (scale depends on
  label entropy; $\le \log K$)

#### G.2.2 Stage B (split quality for a fixed threshold)

Let $I(\cdot)$ be a node impurity functional (Gini, entropy, MSE, MAE). For a
feature $j$ and threshold $c$, define the partition

$$
I_t^L(j,c) := \{i\in I_t : X_{ij}\le c\},\qquad I_t^R(j,c) := I_t\\setminus I_t^L(j,c).
$$

citrees uses the **unweighted** child-impurity sum as the Stage B test
statistic:

$$
T^{\text{split}}_{j,c}(X_{t,j},Y_t) := I(Y_{I_t^L(j,c)}) + I(Y_{I_t^R(j,c)}),
$$

with the convention “smaller is more extreme” (left-tail test).

Note: `min_impurity_decrease` and `feature_importances_` are computed using
**weighted** impurity decrease (CART-style):

$$
\Delta I = I(Y_t) - \Big(\tfrac{n_L}{n_t}I(Y_L)+\tfrac{n_R}{n_t}I(Y_R)\Big).
$$

### G.3 Fixed-$B$ permutation p-values (+1 correction)

```
Algorithm G1: +1 Monte Carlo permutation p-value (fixed-B)
Input: data (x, y), statistic T, permutations B
Output: p-value p

1. Compute observed statistic: T0 ← T(x, y)
2. L ← 0
3. For b = 1..B:
     - y_b ← Permute(y)
     - Tb ← T(x, y_b)
     - If Tb is at least as extreme as T0 (by the chosen tail), set L ← L + 1
4. Return p ← (L + 1) / (B + 1)
```

Right-tail (selectors): “extreme” means $T_b \ge T_0$. Left-tail (splitters):
“extreme” means $T_b \le T_0$.

**Selector convention.** In citrees, Stage A permutation tests apply this
algorithm to the _magnitude_ of the association statistic (e.g., $|{\rm Corr}|$
for `pc`). For nonnegative selectors (`mc`, `dc`, `rdc`, `mi`) this has no
effect but keeps a consistent “more extreme = larger” convention.

Paper-facing validity statements for Algorithm G1 require exchangeability
(Appendix A.2, Theorem 1) and fixed-$B$ mode (`early_stopping=None`).

### G.4 Multi-selector mode (max-T across selectors)

When `selector=[s1,...,sS]`, citrees uses a max-statistic inside each
permutation (Westfall–Young max-T):

$$
T^{\max}(x,y) := \max_{s\in\mathcal S} |T_s(x,y)|.
$$

```
Algorithm G2: Max-T p-value for one feature (fixed-B)
Input: (x, y), selectors {T1,...,TS}, permutations B
Output: p-value p

1. T0 ← max_s |Ts(x, y)|
2. L ← 0
3. For b = 1..B:
     - y_b ← Permute(y)
     - Tb ← max_s |Ts(x, y_b)|
     - If Tb ≥ T0: L ← L + 1
4. Return p ← (L + 1) / (B + 1)
```

Stage A then computes this p-value for each feature $j\in F_t$ and selects the
feature with the smallest p-value.

### G.5 One node expansion (Stage A → Stage B)

```
Algorithm G3: NodeExpand(t)
Input: node data (X_t, Y_t), candidate features F_t, thresholds C_{t,j},
       levels α_sel, α_split, constraints (min_samples_leaf, min_impurity_decrease, …),
       early_stopping, feature_scanning, feature_muting, threshold_scanning
Output: either Leaf or (j*, c*) split with children

1. Stage A (feature screening):
   Remove constant features from F_t (subtree-local).
   If |F_t| = 0: return Leaf
   (Optional) If early stopping and feature_scanning: order F_t by raw association score.
   α_eff ← α_sel/|F_t| if Bonferroni enabled else α_sel
   If permutation testing is disabled:
       - Select j* by the largest raw selector score and skip the p-value gate.
   Else:
       p* ← ∞, j* ← first feature in F_t
       For each j in F_t:
           p_j ← PermTestSelector(X_{t,j}, Y_t)         # Algorithm G1 or G2
           (Optional) If feature_muting and p_j ≥ α_eff and more than one feature remains: remove j from descendant set
           If p_j < p*: set p* ← p_j and j* ← j
           (Optional) If early stopping and p* < α_eff: break
       If p_{j*} ≥ α_eff: return Leaf

2. Build threshold candidates:
   If feature has ≤ 4 unique values: use all midpoints.
   Else: C ← C_{t,j*}   # midpoints of unique values, optionally subsampled
   If |C| = 0: return Leaf
   Filter out thresholds that violate min_samples_leaf (if configured)
   If |C| = 0: return Leaf
   If early stopping is enabled and threshold_scanning is on: sort C by weighted impurity (best first)
   Else if early stopping is enabled and threshold_scanning is off: randomize C

3. Stage B (threshold screening):
   α_eff_split ← α_split/|C| if Bonferroni enabled else α_split
   If permutation testing is disabled:
       - Select c* that minimizes split impurity and skip the p-value gate.
   Else:
       p* ← ∞, c* ← first threshold in C
       For each c in C:
           p_c ← PermTestSplit(X_{t,j*}, Y_t; c)        # Algorithm G1 (left-tail)
           If p_c < p*: set p* ← p_c and c* ← c
           (Optional) If early stopping and p* < α_eff_split: break
       If p* ≥ α_eff_split: return Leaf

4. Structural constraints:
   If split violates min_samples_leaf: return Leaf
   If ΔI(j*, c*) < min_impurity_decrease: return Leaf

5. Return internal node split on (j*, c*) and recurse on children.
```

Important scope note: Stage B p-values and internal-node p-values are
post-selection/adaptive; treat them as algorithmic split-validation/stopping
statistics unless adding selective-inference machinery (Appendix A.4 and
Appendix F).

### G.6 Sequential stopping inside permutation tests (computational heuristic)

When `early_stopping ∈ {"adaptive","simple"}`, citrees may stop the permutation
loop early and returns the +1 estimate evaluated at a data-dependent stopping
time. This is **not** a fixed-$B$ p-value.

Posterior-confidence (“adaptive”) stopping uses

$$
p \mid L_n,n \sim {\rm Beta}(1+L_n,\;1+n-L_n),\qquad S_n := P(p<\alpha\mid L_n,n)=I_\alpha(1+L_n,1+n-L_n),
$$

and stops once $S_n\ge\gamma$ or $1-S_n\ge\gamma$.

For paper-facing p-value guarantees, use fixed-$B$ mode (`early_stopping=None`).

### G.7 Forest training (bagging)

citrees forests fit $M$ independent trees and aggregate their predictions. Each
tree is a conditional inference tree with the same Stage A → Stage B node
expansion (Algorithm G3), typically with:

- random feature subsampling at each node via `max_features`, and
- optional bootstrap resampling of rows.

```
Algorithm G4: FitForest
Input: training data (X, Y), number of trees M,
       tree hyperparameters θ_tree,
       bootstrap method (optional), max_samples (optional)
Output: fitted forest {T₁,...,T_M}

For m = 1..M:
    If bootstrap enabled:
        - Draw bootstrap indices I_m by the chosen method
        - If max_samples < n, subsample I_m (without replacement) down to max_samples
        - Fit tree T_m ← FitTree(X[I_m], Y[I_m]; θ_tree)
    Else:
        - Fit tree T_m ← FitTree(X, Y; θ_tree)

Return {T₁,...,T_M}
```

Prediction is standard bagging:

- classification: average class probabilities over trees,
- regression: average predictions over trees.

## Appendix H. Detailed methods (implementation-aligned)

This appendix provides comprehensive algorithmic details for the citrees
library, intended to serve as the foundation for the Methods section. Proofs and
careful scope statements are in Appendix A (fixed-node/root validity) and
Appendix F (what we do _not_ claim), with optional honest estimation details in
Appendix D.

### Table of Contents

1. [Algorithm Overview](#1-algorithm-overview)
2. [Tree Building Algorithm](#2-tree-building-algorithm)
3. [Feature Selection (Stage A)](#3-feature-selection-stage-a)
4. [Split Selection (Stage B)](#4-split-selection-stage-b)
5. [Permutation Testing](#5-permutation-testing)
6. [Sequential Permutation Testing](#6-sequential-permutation-testing)
7. [Multiple Testing Correction](#7-multiple-testing-correction)
8. [Multi-Selector Mode](#8-multi-selector-mode)
9. [Computational Heuristics](#9-computational-heuristics)
10. [Forest Ensemble](#10-forest-ensemble)
11. [Honest Estimation](#11-honest-estimation)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Implementation Details](#13-implementation-details)
14. [References](#references)

---

### 1. Algorithm Overview

citrees implements conditional inference trees and forests that use
permutation-based hypothesis testing for variable and split selection. The key
innovation over CART-style trees is replacing greedy optimization with
statistical testing, which provides:

1. **Reduced high-cardinality selection bias**: Features are screened via
   permutation-test p-values rather than by maximizing impurity improvement over
   many candidate thresholds. This avoids the pure multiple-comparisons
   mechanism that favors high-cardinality noise (with the usual fixed-node/root
   scope caveats for adaptive trees).

2. **Built-in stopping criterion (permutation-test mode)**: When permutation
   testing is enabled, the tree stops growing when no feature shows
   statistically significant association with the response, providing principled
   regularization.

3. **Interpretable screening p-values (with scope caveats)**: Stage A produces
   permutation p-values for screening feature–response association at a fixed
   node (especially at the root, and in fixed-$B$ mode). Stage B and
   internal-node p-values are post-selection/adaptive and should be treated as
   algorithmic stopping statistics unless additional selective-inference
   machinery is used.

#### High-Level Algorithm

```
Algorithm 1: Conditional Inference Tree
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Training data (X, y), significance levels α_sel, α_split
Output: Decision tree T

function BuildTree(X, y, depth):
    if StoppingCriteria(X, y, depth):
        return LeafNode(value = AggregateResponse(y))

    # Stage A: Feature Selection
    F ← SelectCandidateFeatures(X)
    (j*, p*_sel) ← TestFeatureAssociation(X, y, F)

    if p*_sel ≥ α_sel / |F|:  # Bonferroni-adjusted threshold
        return LeafNode(value = AggregateResponse(y))

    # Stage B: Split Selection
    C ← GenerateThresholdCandidates(X[:, j*])
    (c*, p*_split) ← TestSplitQuality(X[:, j*], y, C)

    if p*_split ≥ α_split / |C|:  # Bonferroni-adjusted threshold
        return LeafNode(value = AggregateResponse(y))

    # Check impurity decrease constraint
    if ImpurityDecrease(y, X[:, j*], c*) < min_impurity_decrease:
        return LeafNode(value = AggregateResponse(y))

    # Recursive split
    (X_L, y_L), (X_R, y_R) ← SplitData(X, y, j*, c*)
    left_child ← BuildTree(X_L, y_L, depth + 1)
    right_child ← BuildTree(X_R, y_R, depth + 1)

    return InternalNode(feature=j*, threshold=c*, left=left_child, right=right_child)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Note.** If `adjust_alpha_selector`/`adjust_alpha_splitter` are disabled,
compare to $\alpha_{\text{sel}}$ / $\alpha_{\text{split}}$ rather than the
Bonferroni-adjusted levels. If permutation testing is disabled
(`n_resamples_* = None`), Stage A and/or Stage B use the raw selector/splitter
scores without p-values (see Sections 3.4 and 4.5).

---

### 2. Tree Building Algorithm

#### 2.1 Stopping Criteria

The tree building process terminates at a node when any of the following
conditions are met:

1. **Sample size**: $n_t < \text{min\_samples\_split}$ (default: 2)
2. **Maximum depth**: $\text{depth} > \text{max\_depth}$
3. **Pure node**: All labels $y_t$ are identical
4. **No significant features**: Stage A fails to reject any null hypothesis
5. **No significant splits**: Stage B fails to reject any null hypothesis
6. **Insufficient impurity decrease**:
   $\Delta I < \text{min\_impurity\_decrease}$
7. **Minimum leaf size violated**: $n_{\text{left}} < \text{min\_samples\_leaf}$
   or $n_{\text{right}} < \text{min\_samples\_leaf}$

#### 2.2 Leaf Value Computation

**Classification**: The leaf value is a probability vector over classes:

$$
\hat{p}_k(L) = \frac{1}{|L|} \sum_{i \in L} \mathbf{1}\{y_i = k\}, \quad k = 1, \ldots, K
$$

**Regression**: The leaf value is the mean of responses:

$$
\hat{\mu}(L) = \frac{1}{|L|} \sum_{i \in L} y_i
$$

---

### 3. Feature Selection (Stage A)

#### 3.1 Overview

At each node $t$ with samples $(X_t, y_t)$ where $n_t = |I_t|$, we test for
association between each candidate feature $X_{t,j}$ and the response $y_t$.
Features that are constant at the node are removed from the candidate set
(subtree-local), since they cannot become non-constant deeper in that subtree.

#### 3.2 Null Hypothesis

For each feature $j$ in the candidate set $F_t$:

$$
H^{\text{sel}}_{t,j}: X_{t,j} \perp y_t
$$

i.e., feature $j$ is independent of the response at node $t$.

#### 3.3 Test Statistics (Selectors)

citrees implements the following association measures:

##### 3.3.1 Multiple Correlation (mc) - Classification

The multiple correlation coefficient measures the strength of linear association
between a feature and class membership:

$$
\text{mc}(x, y) = \sqrt{\frac{\text{SSB}}{\text{SST}}}
$$

where:

- **SST** (Total Sum of Squares): $\text{SST} = \sum_{i=1}^n (x_i - \bar{x})^2$
- **SSB** (Between-class Sum of Squares):
  $\text{SSB} = \sum_{k=1}^K n_k (\bar{x}_k - \bar{x})^2$
- $\bar{x}$ is the overall mean
- $\bar{x}_k$ is the mean of feature values for class $k$
- $n_k$ is the number of samples in class $k$

**Properties**:

- Range: $[0, 1]$
- $\text{mc} = 0$ when class means are identical (no linear association)
- $\text{mc} = 1$ only when within-class variance is zero (all variation is
  between classes)

**Relation to ANOVA.** Writing $\eta^2=\mathrm{SSB}/\mathrm{SST}$ and letting
$g$ be the number of nonempty classes at the node, we have $R^2=\eta^2$ for
regressing $x$ on class indicators, and the one-way ANOVA statistic satisfies

$$
F=\frac{\mathrm{SSB}/(g-1)}{(\mathrm{SST}-\mathrm{SSB})/(n-g)}=\frac{n-g}{g-1}\cdot\frac{\eta^2}{1-\eta^2}.
$$

For fixed $(n,g)$ this is a strictly increasing transform of $\eta$ (and hence
of `mc`), so using `mc` vs $F$ yields the same feature ordering and identical
fixed-$B$ permutation p-values (up to tie conventions).

**Implementation** (`citrees/_selector.py`):

```python
@njit(cache=True, nogil=True, fastmath=True)
def mc(x, y, n_classes, random_state=None):
    mu = x.mean()
    sst = np.sum((x - mu) ** 2)
    ssb = 0.0
    for j in range(n_classes):
        x_j = x[y == j]
        if len(x_j) > 0:
            mu_j = x_j.mean()
            ssb += len(x_j) * (mu_j - mu) ** 2
    return np.sqrt(ssb / sst) if sst > 0 else 0.0
```

##### 3.3.2 Mutual Information (mi) - Classification

Mutual information quantifies the information shared between the feature and
class labels:

$$
I(X; Y) = \sum_{k=1}^K \int p(x, y=k) \log \frac{p(x, y=k)}{p(x) p(y=k)} dx
$$

citrees uses the scikit-learn implementation based on k-nearest neighbor
estimation (Kraskov et al., 2004).

**Properties**:

- Range: $[0, H(Y)]$ (for $K$ classes, $H(Y)\le \log K$; log base sets units)
- Non-negative: $I(X; Y) \geq 0$
- $I(X; Y) = 0$ if and only if $X \perp Y$

**Important**: Because MI is not on the same $[0,1]$ scale as the bounded
selectors and its scale depends on $H(Y)$ and $K$, we do not combine it with
other selectors in max-statistic mode without explicit normalization.

##### 3.3.3 Pearson Correlation (pc) - Regression

The absolute Pearson correlation coefficient:

$$
\text{pc}(x, y) = \left| \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}} \right|
$$

**Properties**:

- Range: $[0, 1]$ (after absolute value)
- Measures linear association only
- $\text{pc} = 0$ does not imply independence (only uncorrelatedness)

**Relation to the correlation $t$ test.** For $n\ge 3$, the usual correlation
test statistic is

$$
t=\rho\sqrt{\frac{n-2}{1-\rho^2}},
$$

where $\rho$ is the (signed) Pearson correlation and `pc` uses $|\rho|$. The
mapping $|\rho|\mapsto |t|$ is strictly increasing on $[0,1)$, so `pc` ranks
features equivalently to $|t|$ and fixed-$B$ permutation p-values agree under
either statistic (up to tie conventions).

In simple linear regression of $y$ on $x$, we also have $R^2=\rho^2$, so `pc`
equals $\sqrt{R^2}$ and $t^2$ is a monotone transform of $R^2$ at fixed $n$:

$$
t^2=(n-2)\cdot\frac{R^2}{1-R^2}.
$$

**Implementation** (`citrees/_selector.py`):

```python
@njit(cache=True, nogil=True, fastmath=True)
def _correlation(x, y):
    n = len(x)
    sx, sy, sx2, sy2, sxy = 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(n):
        sx += x[i]; sy += y[i]
        sx2 += x[i]*x[i]; sy2 += y[i]*y[i]
        sxy += x[i]*y[i]
    cov = n * sxy - sx * sy
    denom = np.sqrt((n * sx2 - sx*sx) * (n * sy2 - sy*sy))
    return cov / denom if denom > 0 else 0.0
```

##### 3.3.4 Distance Correlation (dc) - Regression

Distance correlation (Székely et al., 2007) detects both linear and nonlinear
dependencies:

$$
\text{dCor}(X, Y) = \frac{\text{dCov}(X, Y)}{\sqrt{\text{dVar}(X) \cdot \text{dVar}(Y)}}
$$

where distance covariance is defined via doubly-centered distance matrices.

**Properties**:

- Range: $[0, 1]$
- $\text{dCor}(X, Y) = 0$ if and only if $X \perp Y$ under finite first-moment
  conditions (Székely et al., 2007)
- Detects nonlinear dependencies
- Complexity: $O(n^2)$

**Implementation**: citrees wraps the `dcor` library.

##### 3.3.5 Randomized Dependence Coefficient (rdc) - Both

The RDC (Lopez-Paz et al., 2013) is a computationally efficient nonlinear
dependence measure:

$$
\text{rdc}(x, y) = \max_{j,k} |\text{corr}(\phi_j(F_x(x)), \phi_k(F_y(y)))|
$$

where:

- $F_x, F_y$ are empirical CDFs (rank transform)
- $\phi_j$ are random nonlinear features:
  $\phi(u) = [\cos(w^\top u), \sin(w^\top u)]$
- $w \sim \mathcal{N}(0, s^2 I)$ with bandwidth $s = 1/6$

**Algorithm**:

```
Algorithm 2: Randomized Dependence Coefficient
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Vectors x, y ∈ ℝⁿ, number of projections k=10, bandwidth s=1/6
Output: RDC score ∈ [0, 1]

1. Apply empirical CDF transform:
   u_i ← rank(x_i) / n,  v_i ← rank(y_i) / n

2. Augment with bias term:
   U ← [u, 1] ∈ ℝⁿˣ²,  V ← [v, 1] ∈ ℝⁿˣ²

3. Generate random projections:
   W_x, W_y ~ N(0, s²) ∈ ℝ²ˣᵏ

4. Create nonlinear features:
   Φ_x ← [cos(U W_x), sin(U W_x)] ∈ ℝⁿˣ²ᵏ
   Φ_y ← [cos(V W_y), sin(V W_y)] ∈ ℝⁿˣ²ᵏ

5. Return max |corr(Φ_x[:, j], Φ_y[:, k])| over all j, k
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Properties**:

- Range: $[0, 1]$
- Complexity: $O(n \log n)$ due to sorting for rank transform
- Detects nonlinear dependencies
- Targets a copula-based dependence notion via random nonlinear projections
  (Lopez-Paz et al., 2013)

**Classification detail.** For multiclass classification, citrees computes RDC
between $x$ and each one-vs-rest class indicator and takes the maximum across
classes as the Stage A statistic (to stay in a 1D-to-1D RDC setting).

**Implementation** (`citrees/_selector.py`):

```python
@njit(cache=True, nogil=True, fastmath=True)
def _rdc(x, y, k, s, seed):
    X_feat = _rdc_features(x, k, s, seed)
    Y_feat = _rdc_features(y, k, s, seed + 1000)
    return _rdc_cancor(X_feat, Y_feat)
```

#### 3.4 Feature Selection Procedure

```
Algorithm 3: Feature Selection at Node t
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Data (X_t, y_t), candidate features F_t, α_sel, B permutations, adjust_alpha,
       max_features, early_stopping, feature_scanning, feature_muting
Output: Best feature j*, p-value p*, rejection decision

0. Remove constant features from F_t (subtree-local; constant features cannot become non-constant deeper).
   If |F_t| = 0: return (None, 1.0, False)

1. Set candidate feature order:
   F_t ← RandomSubset(F_t, max_features) if max_features is set; otherwise RandomPermutation(F_t)

2. If feature_scanning and early_stopping enabled:
   F_t ← SortByAssociationScore(F_t, X_t, y_t)  # Most promising first

3. Set effective threshold:
   α_eff ← α_sel/|F_t| if adjust_alpha else α_sel

4. Initialize: p* ← ∞, j* ← F_t[0]

5. For each feature j in F_t:
   a. Compute association score: θ_0 ← |T(X_t[:,j], y_t)|
   b. Compute p-value via permutation test:
      p_j ← PermutationTest(X_t[:,j], y_t, B, α_eff)

   c. If p_j < p*:
      p* ← p_j, j* ← j

   d. Early stopping check (if enabled):
      If p* < α_eff: break  # first significant new best

   e. Feature muting (if enabled):
      If p_j ≥ α_eff and more than one feature remains:
         Remove j from the subtree-local candidate set for descendants of this node

6. Return (j*, p*, p* < α_eff)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Note.** Feature scanning uses raw (non-permutation) association scores; in
multi-selector mode it uses the maximum across selectors to sort features.

**No-permutation mode.** If `n_resamples_selector=None`, Stage A skips
permutation testing and selects the feature with the largest raw selector score.
In this mode, `adjust_alpha_selector`, `early_stopping_selector`, and
`feature_muting` are effectively inactive (the implementation warns if they are
set).

**`max_features` semantics (implementation-aligned).** The candidate subset size
is computed by `calculate_max_value(n_values=|F_t|, desired_max=max_features)`,
with:

- integer: `min(max_features, |F_t|)`,
- float in $(0,1]$: `ceil(|F_t| * max_features)` (values $\ge 1$ are effectively
  “all features”),
- `"sqrt"`: `ceil(sqrt(|F_t|))`,
- `"log2"`: `ceil(log2(|F_t|))`,
- `None`: all features. The subset is then sampled **without replacement** from
  the local candidate set.

---

### 4. Split Selection (Stage B)

#### 4.1 Overview

Given the selected feature $j^*$, we find the best threshold $c$ to partition
the data into left and right children.

#### 4.2 Null Hypothesis

For each threshold $c$ in the candidate set $C_{t,j^*}$:

$$
H^{\text{split}}_{t,j^*,c}: y_t \text{ is exchangeable w.r.t. the partition } (I^L_t, I^R_t)
$$

where $I^L_t = \{i : X_{i,j^*} \leq c\}$ and $I^R_t = I_t \setminus I^L_t$.

**Post-selection caveat.** This test is conducted after selecting $j^*$ using
the same labels, so the resulting Stage B p-values are post-selection/adaptive
and should be treated as algorithmic stopping statistics unless
selective-inference adjustments or sample splitting are introduced.

#### 4.3 Test Statistics (Splitters)

**Important convention.** The permutation test statistic is the **unweighted**
sum of child impurities $I(y_L)+I(y_R)$ (left-tail). Weighted impurity is used
only for threshold scanning and the deterministic `min_impurity_decrease` check.

##### 4.3.1 Gini Index - Classification

$$
\text{Gini}(y) = 1 - \sum_{k=1}^K \hat{p}_k^2
$$

where $\hat{p}_k = \frac{1}{n} \sum_{i=1}^n \mathbf{1}\{y_i = k\}$.

**Split statistic** (lower is better):

$$
T^{\text{split}}(c) = \text{Gini}(y_L) + \text{Gini}(y_R)
$$

**Properties**:

- Range: $[0, 1 - 1/K]$
- $\text{Gini} = 0$ for pure nodes
- Maximum when all classes are equally represented

**Implementation** (`citrees/_splitter.py`):

```python
@njit(cache=True, fastmath=True, nogil=True)
def gini(y):
    n = len(y)
    p = np.bincount(y) / n
    return 1 - np.sum(p * p)
```

##### 4.3.2 Entropy - Classification

$$
\text{Entropy}(y) = -\sum_{k=1}^K \hat{p}_k \log_2(\hat{p}_k)
$$

**Properties**:

- Range: $[0, \log_2 K]$
- $\text{Entropy} = 0$ for pure nodes
- Maximum for uniform distribution

**Implementation** (`citrees/_splitter.py`):

```python
@njit(cache=True, fastmath=True, nogil=True)
def entropy(y):
    n = len(y)
    p = np.bincount(y) / n
    p = p[p != 0]  # Avoid log(0)
    return -np.sum(np.log2(p) * p)
```

##### 4.3.3 Mean Squared Error (MSE) - Regression

$$
\text{MSE}(y) = \frac{1}{n} \sum_{i=1}^n (y_i - \bar{y})^2
$$

This is the empirical variance of the response.

**Implementation** (`citrees/_splitter.py`):

```python
@njit(cache=True, fastmath=True, nogil=True)
def mse(y):
    dev = y - y.mean()
    return np.mean(dev * dev)
```

##### 4.3.4 Mean Absolute Error (MAE) - Regression

$$
\text{MAE}(y) = \frac{1}{n} \sum_{i=1}^n |y_i - \mathrm{median}(y)|
$$

More robust to outliers than MSE.

**Implementation** (`citrees/_splitter.py`):

```python
@njit(cache=True, fastmath=True, nogil=True)
def mae(y):
    dev = np.abs(y - np.median(y))
    return np.mean(dev)
```

#### 4.4 Threshold Generation Methods

citrees supports four methods for generating candidate thresholds:

##### 4.4.1 Exact (default)

Uses all midpoints between consecutive unique values:

$$
C = \left\{ \frac{x_{(i)} + x_{(i+1)}}{2} : i = 1, \ldots, m-1 \right\}
$$

where $x_{(1)} < x_{(2)} < \cdots < x_{(m)}$ are the unique sorted values.

**Implementation** (`citrees/_threshold_method.py`):

```python
@njit(cache=True, fastmath=True, nogil=True)
def exact(x, max_thresholds=None, random_state=None):
    values = np.unique(x)
    midpoints = (values[:-1] + values[1:]) / 2
    return midpoints
```

##### 4.4.2 Random

Random subsample of midpoints (without replacement):

$$
C = \text{RandomSample}(\text{Midpoints}, k)
$$

##### 4.4.3 Percentile

Equally spaced percentiles of midpoints:

$$
C = \{Q_p(\text{Midpoints}) : p \in \text{linspace}(0, 100, k)\}
$$

##### 4.4.4 Histogram

Histogram bin edges of midpoints.

**Small-sample fallback.** If the feature has very few unique values ($\le 4$),
the implementation uses all midpoints directly, regardless of threshold method.

**`max_thresholds` semantics (implementation-aligned).** When `max_thresholds`
is set, the count is computed by
`calculate_max_value(n_values=n_unique, desired_max=max_thresholds)` using the
same integer/float/`sqrt`/`log2` rules as `max_features` (floats $\ge 1$
effectively mean “all thresholds”), then applied to the candidate midpoints for
the selected threshold method. When `threshold_method="exact"`, all midpoints
are returned regardless of `max_thresholds` (it is accepted for API
compatibility but not used).

**Warning behavior.** If `threshold_method != "exact"` and
`max_thresholds=None`, the code emits a warning because the method will still
use all midpoints (which can be large). The run proceeds, but it is not the
recommended setting for speed.

#### 4.5 Split Selection Procedure

```
Algorithm 4: Split Selection at Node t
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Feature x = X_t[:,j*], response y_t, α_split, B permutations, adjust_alpha,
       max_thresholds, min_samples_leaf, early_stopping, threshold_scanning
Output: Best threshold c*, p-value p*, rejection decision

1. Generate threshold candidates:
   If feature has ≤ 4 unique values: use all midpoints.
   Else: C ← ThresholdMethod(x; max_thresholds)  # exact returns all midpoints
   If |C| = 0: return (None, 1.0, False)
   Filter out thresholds that violate min_samples_leaf (if configured)
   If |C| = 0: return (None, 1.0, False)

2. If threshold_scanning and early_stopping enabled:
   C ← SortByWeightedImpurity(C, x, y_t)  # Best (lowest) weighted impurity first
   Else if early_stopping enabled:
      C ← RandomPermutation(C)              # Avoid ordering bias for early stopping

3. Set effective threshold:
   α_eff ← α_split/|C| if adjust_alpha else α_split

4. Initialize: p* ← ∞, c* ← C[0]

5. For each threshold c in C:
   a. Compute split impurity:
      θ_0 ← Impurity(y[x ≤ c]) + Impurity(y[x > c])

   b. Compute p-value via permutation test (LEFT-TAIL):
      p_c ← PermutationTestSplit(x, y_t, c, B, α_eff)

   c. If p_c < p*:
      p* ← p_c, c* ← c

   d. Early stopping (if enabled):
      If p* < α_eff: break  # first significant new best

6. Check deterministic split constraints:
   If min_samples_leaf violated by c*: set reject=False

7. Return (c*, p*, p* < α_eff)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Important**: Split selection uses a **left-tail** test (smaller impurity =
better split), while feature selection uses a **right-tail** test (larger
association = stronger signal).

**No-permutation mode.** If `n_resamples_splitter=None`, Stage B skips
permutation testing and selects the threshold that minimizes the **weighted**
split impurity; `adjust_alpha_splitter` and `early_stopping_splitter` are
inactive (the implementation warns if they are set).

**Left-tail note for sequential tests.** The sequential stopping algorithms use
the same logic as in Stage A, but for Stage B the “extreme” condition is
`θ_b ≤ θ_0` (no absolute value), consistent with the left-tail convention.

---

### 5. Permutation Testing

#### 5.1 Core Principle

Permutation tests rely on an exchangeability invariance: under the null
hypothesis (in the sense used by the permutation scheme), the conditional
distribution of $Y$ given the covariates treated as fixed by the test is
invariant under permutations. A common sufficient condition is i.i.d. sampling
with $X \perp Y$. Therefore:

$$
T(X, Y) \stackrel{d}{=} T(X, \pi(Y)) \quad \text{for all permutations } \pi
$$

#### 5.2 Monte Carlo Permutation Test Algorithm

```
Algorithm 5: Monte Carlo Permutation Test (Fixed-B)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Data (x, y), test statistic T, number of permutations B
Output: P-value p

1. Compute observed statistic: θ_0 ← T(x, y)

2. Initialize count: k ← 0

3. For b = 1, ..., B:
   a. Generate random permutation π_b
   b. Compute permuted statistic: θ_b ← T(x, π_b(y))
   c. Update k using the chosen tail:
      - right-tail (selectors): if θ_b ≥ θ_0, increment k
      - left-tail (splitters): if θ_b ≤ θ_0, increment k
      - two-sided: use |θ_b| ≥ |θ_0|

4. Return p-value with +1 correction:
   p ← (k + 1) / (B + 1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### 5.3 Phipson-Smyth +1 Correction

The p-value formula $p = (k+1)/(B+1)$ instead of $p = k/B$ ensures:

1. **Non-zero p-values**: $p \geq 1/(B+1) > 0$, critical for multiple testing
   correction
2. **Conservative estimate (conditional)**: if $p^*$ is the exact permutation
   p-value, then $\mathbb{E}[p \mid p^*] = p^* + (1-p^*)/(B+1) \geq p^*$
3. **Finite-sample validity (super-uniformity)**: under the exchangeability
   null, $\mathbb{P}(p \leq \alpha) \leq \alpha$ for all $\alpha\in[0,1]$

**Reference**: Phipson & Smyth (2010). "Permutation P-values Should Never Be
Zero." SAGMB 9(1):39.

#### 5.4 Implementation Details

**Right-tail test** (feature selection - larger = more extreme):

```python
# citrees/_selector.py
theta = np.abs(func(x, y, func_arg, random_state=random_state))
rng = np.random.default_rng(random_state)
y_ = y.copy()
theta_p = np.empty(n_resamples)
for i in range(n_resamples):
    rng.shuffle(y_)
    theta_p[i] = func(x, y_, func_arg, random_state=random_state)
return (1 + np.sum(np.abs(theta_p) >= theta)) / (1 + n_resamples)
```

**Left-tail test** (split selection - smaller = more extreme):

```python
# citrees/_splitter.py
idx = x <= threshold
theta = func(y[idx]) + func(y[~idx])
rng = np.random.default_rng(random_state)
y_ = y.copy()
theta_p = np.empty(n_resamples)
for i in range(n_resamples):
    rng.shuffle(y_)
    theta_p[i] = func(y_[idx]) + func(y_[~idx])
return (1 + np.sum(theta_p <= theta)) / (1 + n_resamples)
```

**Randomized selector note.** For selectors with internal randomness (e.g., RDC,
mutual information), the same `random_state` is used across all permutations, so
the permutation p-value is conditional on that label-independent randomness.
This is consistent with the fixed-node validity assumptions (A0.2–A0.3).

---

### 6. Sequential Permutation Testing

citrees implements sequential Monte Carlo stopping rules to reduce computational
cost in permutation testing.

**Important inferential note.** When `early_stopping_*=None`, citrees reports
fixed-$B$ Monte Carlo permutation p-values with the Phipson–Smyth (+1)
correction, and standard permutation-test guarantees apply. When
`early_stopping_* ∈ {"adaptive","simple"}`, the algorithm may stop at a
data-dependent time and returns the +1 Monte Carlo estimate evaluated at that
stopping time; this number should not be treated as a classical fixed-$B$
p-value. For publication-grade p-value claims, use fixed-$B$ mode
(`early_stopping_*=None`) and report $B$ explicitly.

#### 6.1 Motivation

In a fixed-$B$ permutation test:

- Under $H_0$: Need many permutations to establish non-significance
- Under $H_1$: Few permutations often suffice to establish significance
- Problem: Fixed-$B$ is wasteful when the result is "obvious"

#### 6.2 Simple Sequential (Baseline)

Stops early under two conditions:

1. **Significance**: Current p-value $< \alpha$ (after minimum resamples)
2. **Futility**: Best possible p-value $\geq \alpha$ (cannot reject)

**Bonferroni note.** When Stage A uses a Bonferroni-adjusted level
$\alpha'=\alpha/m$, replace $\alpha$ by $\alpha'$ throughout the stopping rules
(including `min_resamples ← ⌈1/α⌉`).

```
Algorithm 6: Simple Sequential Permutation Test
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Data (x, y), statistic T, max permutations B, threshold α
Output: P-value p

1. Compute θ_0 ← |T(x, y)|
2. min_resamples ← ⌈1/α⌉
3. B ← max(B, min_resamples)
4. k ← 0  # extreme count

5. For b = 1, ..., B:
   a. Permute and compute: θ_b ← |T(x, π_b(y))|
   b. If θ_b ≥ θ_0: k ← k + 1
   c. p_current ← (k + 1) / (b + 1)

   d. If b ≥ min_resamples:
      # Early significance
      If p_current < α: return p_current

      # Futility: best possible p-value
      p_best ← (k + 1) / (B + 1)
      If p_best ≥ α and k ≥ 3: return p_current

6. Return (k + 1) / (B + 1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Warning**: This method inflates Type I error because it "peeks" at running
p-values without proper sequential adjustment.

**Left-tail variant (splitters).** For split selection, use the unweighted
impurity statistic (no absolute value) and replace the exceedance check by
`θ_b ≤ θ_0`.

#### 6.3 Adaptive Sequential (Bayesian)

Uses Bayesian posterior to make stopping decisions:

**Model**: After $n$ permutations with $k$ exceedances:

$$
k \mid p \sim \text{Binomial}(n, p)
$$

$$
p \sim \text{Beta}(1, 1) \quad \text{(uniform prior)}
$$

$$
p \mid k, n \sim \text{Beta}(1 + k, 1 + n - k)
$$

**Stopping rule**: Stop when confident about significance/non-significance:

$$
P(p < \alpha \mid k, n) \geq \gamma \quad \text{or} \quad P(p \geq \alpha \mid k, n) \geq \gamma
$$

where $\gamma = 0.95$ (default confidence).

**Bonferroni note.** When Stage A uses a Bonferroni-adjusted level
$\alpha'=\alpha/m$, replace $\alpha$ by $\alpha'$ throughout (including
`min_resamples ← ⌈1/α⌉`).

```
Algorithm 7: Adaptive Sequential Permutation Test
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Data (x, y), statistic T, max permutations B, threshold α, confidence γ
Output: P-value p

1. Compute θ_0 ← |T(x, y)|
2. min_resamples ← ⌈1/α⌉
3. B ← max(B, min_resamples)
4. k ← 0  # extreme count

5. For b = 1, ..., B:
   a. Permute and compute: θ_b ← |T(x, π_b(y))|
   b. If θ_b ≥ θ_0: k ← k + 1

   c. If b ≥ min_resamples:
      # Compute Beta CDF: P(p < α | k, b)
      prob_sig ← BetaCDF(α; 1+k, 1+b-k)

      # Confident significant
      If prob_sig ≥ γ: return (k + 1) / (b + 1)

      # Confident non-significant
      If (1 - prob_sig) ≥ γ: return (k + 1) / (b + 1)

6. Return (k + 1) / (B + 1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Left-tail variant (splitters).** For split selection, use the unweighted
impurity statistic (no absolute value) and replace the exceedance check by
`θ_b ≤ θ_0`.

#### 6.4 Beta CDF Computation

The Beta CDF $I_\alpha(a, b) = P(X \leq \alpha)$ for $X \sim \text{Beta}(a, b)$
is computed using Lentz's continued fraction algorithm:

```python
# citrees/_sequential.py
@njit(cache=True, fastmath=True)
def _beta_cdf(x, a, b):
    if x <= 0: return 0.0
    if x >= 1: return 1.0

    # Use symmetry for numerical stability
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _beta_cdf(1 - x, b, a)

    # Continued fraction expansion
    log_prefix = a * log(x) + b * log(1-x) - log(a)
    log_prefix += lgamma(a+b) - lgamma(a) - lgamma(b)

    # Lentz's algorithm for continued fraction
    # ... (see implementation)

    return exp(log_prefix) * result
```

#### 6.5 Empirical Performance

The tradeoff is best understood empirically via null/signal simulations; see
`paper/scripts/theory/` for reproducible calibration scripts.

Paper-facing outputs live under `paper/results/figures/`:

- `paper/scripts/theory/generate_sequential_stopping_calibration.py` generates
  `paper/results/figures/sequential_stopping_calibration.png` (null calibration
  under the continuous-null idealization).

Adaptive stopping can reduce computation substantially on clearly
non-significant tests, but fixed-$B$ mode remains the clean option for classical
p-value claims.

---

### 7. Multiple Testing Correction

#### 7.1 Bonferroni Correction

At each node, citrees performs $m$ hypothesis tests (one per candidate feature
or threshold). To control family-wise error rate (FWER), Bonferroni correction
is applied:

$$
\text{Reject } H_j \text{ if } p_j < \frac{\alpha}{m}
$$

**Properties**:

- FWER $\leq \alpha$ under any dependence structure
- Conservative when tests are positively dependent
- No independence assumptions required

#### 7.2 Dynamic Adjustment of Permutation Count

When Bonferroni is enabled, the effective threshold becomes $\alpha/m$. To
maintain resolution for smaller thresholds, citrees scales the number of
permutations:

**NResamples modes**:

1. **minimum**: $B = \lceil 1/(\alpha/m) \rceil = \lceil m/\alpha \rceil$
   - Minimum required for rejection possibility

2. **maximum**: $B = \lceil 1/(4(\alpha/m)^2) \rceil$
   - High precision, diminishing returns

3. **auto** (default):
   $B = \max(\lceil 1/(\alpha/m) \rceil, z_{1-\alpha/m}^2 (1-\alpha/m)/(\alpha/m))$
   - Balances precision and computation
   - Uses normal approximation for Monte Carlo error

If a numeric $B$ is provided (rather than an enum), citrees multiplies it by $m$
under Bonferroni to preserve per-test resolution.

**Implementation** (`citrees/_tree.py`):

```python
def _bonferroni_correction(self, *, adjust: str, n_tests: int):
    if n_tests > 1:
        _alpha = alpha / n_tests
        if isinstance(n_resamples, str):
            if n_resamples == NResamples.MINIMUM:
                _n_resamples = ceil(1 / _alpha)
            elif n_resamples == NResamples.MAXIMUM:
                _n_resamples = ceil(1 / (4 * _alpha * _alpha))
            else:  # AUTO
                z = norm.ppf(1 - _alpha)
                upper_limit = ceil(z * z * (1 - _alpha) / _alpha)
                _n_resamples = max(ceil(1/_alpha), upper_limit)
        else:  # numeric B
            _n_resamples = n_resamples * n_tests
```

#### 7.3 Scaling Examples

| Features | $\alpha$ | Effective $\alpha$ | Min B  | Auto B  |
| -------- | -------- | ------------------ | ------ | ------- |
| 10       | 0.05     | 0.005              | 200    | 1,321   |
| 50       | 0.05     | 0.001              | 1,000  | 9,540   |
| 100      | 0.05     | 0.0005             | 2,000  | 21,645  |
| 1,000    | 0.05     | 0.00005            | 20,000 | 302,719 |

---

### 8. Multi-Selector Mode

#### 8.1 Overview

citrees supports combining multiple selectors (e.g., `selector=['mc', 'rdc']`).
This is useful when:

- Linear and nonlinear associations may both be present
- Different selectors have complementary strengths
- Increased sensitivity is desired

#### 8.2 Max-T Method (Westfall & Young, 1993)

To maintain valid Type I error when using multiple selectors, citrees uses the
**max-T method**:

**Composite statistic**:

$$
T^{\text{max}}(x, y) = \max_{s \in \mathcal{S}} |T_s(x, y)|
$$

**Key insight**: Compute the maximum **inside** each permutation, not just on
the observed data.

```
Algorithm 8: Max-T Multi-Selector Permutation Test
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Data (x, y), selectors {T_1, ..., T_S}, permutations B, threshold α
Output: P-value p

1. Compute observed max statistic:
   θ_0 ← max_{s} |T_s(x, y)|

2. k ← 0

3. For b = 1, ..., B:
   a. Permute: y_b ← π_b(y)
   b. Compute max statistic under permutation:
      θ_b ← max_{s} |T_s(x, y_b)|  # MAX INSIDE PERMUTATION
   c. If θ_b ≥ θ_0: k ← k + 1

4. Return p ← (k + 1) / (B + 1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### 8.3 Validity

**Theorem (fixed-$B$; exchangeability).** The max-T p-value is valid
(super-uniform under $H_0$).

**Proof.** The composite statistic $T^{\text{max}}$ is a measurable function of
the data. Under $H_0$, exchangeability of $Y$ implies that the vector
$(T^{\text{max}}_0, T^{\text{max}}_1, \ldots, T^{\text{max}}_B)$ is
exchangeable, where $T^{\text{max}}_b = \max_s |T_s(x, \pi_b(y))|$. The rank
argument from Theorem 1 (Appendix A.2) then applies directly, so the +1 p-value
based on $T^{\text{max}}$ is super-uniform. $\square$

#### 8.4 Scale Compatibility

**Requirement**: All selectors in multi-selector mode must be on a comparable
scale.

| Selector | Scale                    | Can Combine? |
| -------- | ------------------------ | ------------ |
| mc       | [0, 1]                   | Yes          |
| rdc      | [0, 1]                   | Yes          |
| pc       | [0, 1] (after abs)       | Yes          |
| dc       | [0, 1]                   | Yes          |
| mi       | [0, H(Y)] ($\le \log K$) | **No**       |

Mutual information (mi) cannot be combined because its scale depends on $H(Y)$
and $K$ (not normalized to $[0,1]$), which would bias a raw max toward mi
without additional normalization.

#### 8.5 Empirical Validation

Theorem 1 already gives a finite-sample super-uniformity guarantee for fixed-$B$
permutation p-values (and therefore for max-T p-values, since the max statistic
is just a measurable function of the data). Empirical calibration plots can be
useful as a sanity check, but we avoid hard-coding rejection rates in the
manuscript unless they are generated by a committed, reproducible script under
`paper/scripts/theory/` and saved into `paper/results/`.

---

### 9. Computational Heuristics

citrees implements several heuristics to improve computational efficiency. These
can change the algorithm’s behavior and, when used inside permutation testing
(e.g., sequential stopping), can complicate classical “fixed-$B$ p-value”
interpretations. For paper-facing p-value guarantees, use fixed-$B$ mode
(`early_stopping=None`) and avoid adaptive hypothesis-family modifications
(e.g., disable `feature_muting`).

#### 9.1 Feature Muting

**Idea**: Remove features that show no association with the response from future
consideration.

**Criterion**: A feature $j$ is muted if:

$$
p_j \geq \alpha
$$

Here $\alpha$ is the current nodewise threshold (after any Bonferroni
adjustment). In addition, citrees never mutes the **last remaining** candidate
feature at a node (to avoid empty sets).

This removes features that are clearly non-significant while keeping marginal
cases.

**Effect on theory**: Feature muting adaptively changes the hypothesis family
across nodes, making global FWER statements more complex. Use
`feature_muting=False` for clean theoretical guarantees.

#### 9.2 Feature Scanning

**Idea**: When early stopping is enabled, test the most promising features
first.

**Procedure**:

1. Compute association scores for all candidate features (no permutation)
2. Sort features by decreasing score
3. Test in sorted order

This increases the probability of early stopping when a significant feature
exists. When early stopping is disabled, feature scanning is ignored (order does
not change the result).

#### 9.3 Threshold Scanning

**Idea**: Similar to feature scanning, but for thresholds.

**Procedure**:

1. Compute weighted child impurity for all candidate thresholds (no permutation)
2. Sort thresholds by increasing impurity
3. Test in sorted order

Threshold scanning only affects ordering when early stopping is enabled;
otherwise the full set is evaluated and order does not change the selected
split.

#### 9.4 Parallel Permutation Tests

For large permutation counts without early stopping, citrees uses Numba's
parallel loops:

```python
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_mc_parallel(x, y, n_classes, n_resamples, random_state):
    theta = mc(x, y, n_classes)
    theta_p = np.empty(n_resamples)
    for i in prange(n_resamples):  # Parallel loop
        np.random.seed(random_state + i)
        y_perm = y.copy()
        np.random.shuffle(y_perm)
        theta_p[i] = mc(x, y_perm, n_classes)
    return (1 + np.sum(np.abs(theta_p) >= theta)) / (1 + n_resamples)
```

Threshold: `_PARALLEL_THRESHOLD = 200` permutations.

---

### 10. Forest Ensemble

#### 10.1 Overview

`ConditionalInferenceForestClassifier` and `ConditionalInferenceForestRegressor`
build ensembles of conditional inference trees using bagging.

#### 10.2 Bootstrap Methods

##### 10.2.1 Classic Bootstrap

Standard bootstrap with replacement:

$$
\mathcal{D}^{(b)} = \{(X_{\pi_i}, Y_{\pi_i})\}_{i=1}^n, \quad \pi_i \stackrel{\text{iid}}{\sim} \text{Uniform}\{1, \ldots, n\}
$$

If `max_samples < n`, the bootstrap indices are subsampled (without replacement)
down to `max_samples` (duplicates can remain because the subsampling is
performed on the bootstrap draw).

##### 10.2.2 Bayesian Bootstrap (default)

Weights samples according to a Dirichlet prior:

$$
w \sim \text{Dirichlet}(1, \ldots, 1),
$$

equivalently draw $Z_i \sim \mathrm{Exp}(1)$ and set $w_i = Z_i / \sum_j Z_j$.

The Bayesian bootstrap samples indices with probabilities proportional to $w$.
If `max_samples < n`, the resulting bootstrap sample is subsampled down to
`max_samples`.

**Implementation** (`citrees/_utils.py`):

```python
@njit(cache=True, fastmath=True, nogil=True)
def bayesian_bootstrap_proba(*, n: int, random_state: int):
    np.random.seed(random_state)
    p = np.random.exponential(scale=1.0, size=n)
    return p / p.sum()
```

#### 10.3 Sampling Methods (Classification)

##### 10.3.1 Stratified (default)

Maintains class proportions in each bootstrap sample:

- Sample separately within each class
- Combine to form final sample If `max_samples < n`, allocate a proportional
  number of samples per class (rounded to sum to `max_samples`) and subsample
  within each class.

##### 10.3.2 Balanced

Forces equal class sizes:

- Sample $n_{\min} = \min_k n_k$ from each class
- Useful for imbalanced datasets The initial balanced bootstrap sample size is
  $K \cdot n_{\min}$ (which can be smaller than $n$). If `max_samples < n`, the
  bootstrap sample is further subsampled to size `max_samples` while keeping
  classes balanced (as equal as possible via integer allocation; per-class
  counts may differ by at most 1).

#### 10.4 Aggregation

**Classification**: Average predicted probabilities

$$
\hat{p}(y = k \mid x) = \frac{1}{T} \sum_{t=1}^T \hat{p}_t(y = k \mid x)
$$

**Regression**: Average predictions

$$
\hat{y}(x) = \frac{1}{T} \sum_{t=1}^T \hat{y}_t(x)
$$

#### 10.5 Feature Importance

Mean Decrease in Impurity (MDI) aggregated across trees:

$$
\text{Importance}_j = \frac{1}{T}\sum_{t=1}^T \text{Importance}^{(t)}_j,
\qquad
\text{Importance}^{(t)}_j \propto \sum_{\text{node } v \text{ splits on } j} \Delta I_v
$$

where each tree’s $\text{Importance}^{(t)}$ is normalized to sum to 1; the
forest then averages (and renormalizes).

#### 10.6 Out-of-bag (OOB) scoring (optional)

If `oob_score=True`, the forest computes out-of-bag predictions by
**reconstructing** each tree’s bootstrap indices using the same `random_state`
and bootstrap method, then predicting on the complement set:

- Classification: average OOB class probabilities and report accuracy on samples
  with at least one OOB prediction.
- Regression: average OOB predictions and report $R^2$ on samples with at least
  one OOB prediction.

If some samples are never OOB (too few trees or heavy resampling), the code
emits a warning and scores only those with `n_oob > 0`. OOB scoring requires
bootstrap to be enabled; otherwise an error is raised.

---

### 11. Honest Estimation

#### 11.1 Overview

Honest estimation (Wager & Athey, 2018) uses sample splitting to decouple tree
structure learning from leaf value estimation, reducing adaptive bias.

#### 11.2 Procedure

```
Algorithm 9: Honest Tree Building
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Data (X, y), honesty_fraction η (default 0.5)
Output: Honest tree T

1. Split data into:
   - Splitting sample: (X_S, y_S) with n_S = ⌊(1-η)n⌋ samples
   - Estimation sample: (X_E, y_E) with n_E = ⌈ηn⌉ samples

   Note: Use random split (not stratified) for theoretical guarantees

2. Build tree structure using splitting sample:
   T ← BuildTree(X_S, y_S)

3. Re-estimate leaf values using estimation sample:
   For each leaf L in T:
      E_L ← {i ∈ E : X_i routes to L}
      If |E_L| > 0:
         L.value ← AggregateResponse(y_{E_L})
      # Else: keep original value from splitting sample

4. Return T
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### 11.3 Theoretical Guarantee

**Proposition** (Unbiased honest estimation): Under honest estimation with
independent sample split:

$$
\mathbb{E}[\hat{\mu}(L) \mid \Pi] = \mu(L) := \mathbb{E}[Y \mid X \in L]
$$

for any leaf $L$ with $|E_L| \geq 1$, where $\Pi$ is the learned partition.

**Proof**: See Appendix D (Proposition D.1).

#### 11.4 Implementation Details

**Sample splitting** (`citrees/_tree.py`):

```python
if self.honesty:
    X_split, X_est, y_split, y_est = train_test_split(
        X, y,
        test_size=self.honesty_fraction,
        random_state=self._random_state,
        # stratify=None for theoretical guarantees
    )
    self.tree_ = self._build_tree(X_split, y_split, depth=1)
    self._reestimate_leaf_values(X_est, y_est)
```

**Path-based leaf identification** (robust to serialization):

```python
def _get_leaf_path(self, x, tree=None, path=()):
    if tree is None:
        tree = self.tree_
    if "value" in tree:
        return path
    if x[tree["feature"]] <= tree["threshold"]:
        return self._get_leaf_path(x, tree["left_child"], path + ("L",))
    else:
        return self._get_leaf_path(x, tree["right_child"], path + ("R",))
```

---

### 12. Evaluation Metrics

This section defines the metrics used to evaluate feature selection methods,
particularly for demonstrating reduced selection bias under null (e.g.,
high-cardinality noise bias in greedy trees vs Stage A screening).

#### 12.1 Precision@k, Recall@k, F1@k

For synthetic datasets with known ground truth (informative feature indices):

$$
\text{Precision}@k = \frac{|\text{top}_k \cap \text{informative}|}{k}
$$

$$
\text{Recall}@k = \frac{|\text{top}_k \cap \text{informative}|}{|\text{informative}|}
$$

$$
\text{F1}@k = 2 \cdot \frac{\text{Precision}@k \cdot \text{Recall}@k}{\text{Precision}@k + \text{Recall}@k}
$$

**Redundant features (informative+redundant).** For datasets with redundant
features (linear combinations of informative ones), we also report metrics using
the union of informative and redundant indices as ground truth:

$$
\text{Precision}^{\mathrm{IR}}@k = \frac{|\text{top}_k \cap (\text{informative} \cup \text{redundant})|}{k}
$$

$$
\text{Recall}^{\mathrm{IR}}@k = \frac{|\text{top}_k \cap (\text{informative} \cup \text{redundant})|}{|\text{informative} \cup \text{redundant}|}
$$

This avoids penalizing methods that select redundant-but-correct proxies of the
signal.

#### 12.2 Noise Selection Rate (False Positive Rate)

The noise selection rate measures how often a feature selection method
incorrectly ranks noise features in the top-k:

$$
\text{NSR}@k = \frac{|\text{top}_k \cap \text{noise}|}{k}
$$

Where:

- `top_k` = set of top k features by ranking (from feature selection method)
- `noise` = set of indices of known noise features (from synthetic ground truth)
- Range: [0, 1], lower is better
- 0.0 = no noise selected (perfect)
- 1.0 = all selected features are noise (worst case)

**Significance for Selection Bias:**

This metric is critical for evaluating selection bias. Traditional methods like
CART and Random Forest are known to favor high-cardinality features (features
with many unique values), leading to elevated NSR even when those features are
pure noise.

Conditional inference methods (citrees) use permutation-based hypothesis testing
which is invariant to feature cardinality, and should therefore reduce one
important source of high-cardinality selection bias in variable selection.
However, metrics like NSR@k are not “Type I error at level $\alpha$” and are not
directly controlled by $\alpha$; we treat NSR@k as an empirical metric.

**Confounder selection rate (correlated noise).** For confounder datasets (noise
features correlated with informative features), we report:

$$
\text{ConfounderRate}@k = \frac{|\text{top}_k \cap \text{confounders}|}{k}
$$

This measures how often a method is misled by correlated-but-noncausal features.

**Expected behavior (qualitative).** Under global-null “selection bias”
simulations, greedy impurity optimization tends to over-select high-cardinality
noise features, while testing-based screening can mitigate this. The magnitude
of NSR@k differences is benchmark-dependent and should be reported empirically
rather than asserted as a direct consequence of level-$\alpha$ testing.

#### 12.3 Nogueira Stability Index

Measures consistency of feature selection across repeated runs (e.g., CV folds,
random seeds):

$$
\text{Stability} = 1 - \frac{\frac{1}{p} \sum_{j=1}^{p} \hat{f}_j (1 - \hat{f}_j)}{\frac{\bar{k}}{p} (1 - \frac{\bar{k}}{p})}
$$

Where:

- $p$ = total number of features
- $\hat{f}_j$ = selection frequency of feature $j$ across $M$ runs
- $\bar{k}$ = average number of selected features

Range: [-1, 1], higher is better. 1.0 indicates perfect consistency.

#### 12.4 Pairwise Statistical Comparisons

After establishing overall ranking differences via Friedman test, we perform
pairwise comparisons using the **Wilcoxon signed-rank test** with
**Holm-Bonferroni correction** for family-wise error rate (FWER) control.

**Wilcoxon signed-rank test** is a non-parametric paired test that compares
matched samples without assuming normality. For each pair of methods $(i, j)$:

- $H_0$: Methods $i$ and $j$ have equal performance distributions
- $H_1$: Methods $i$ and $j$ differ in performance

**Holm-Bonferroni correction** controls FWER by a step-down procedure:

1. Sort $m$ p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. For $i = 1, \ldots, m$: reject $H_{(i)}$ if $p_{(i)} \leq \alpha/(m-i+1)$
3. Stop at first non-rejection

This is more powerful than Bonferroni while maintaining FWER $\leq \alpha$.

**Implementation detail (alignment + minimum pairs).** The code aligns pairs
using complete-case rows for the two methods/metric (dropna). It only runs
Wilcoxon when there are at least 10 paired values; otherwise the pair is
skipped.

#### 12.5 Cohen's d Effect Size

To quantify practical significance beyond statistical significance, we report
Cohen's $d$ effect size for each pairwise comparison:

$$
d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}
$$

where $s_{\text{pooled}} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$

**Interpretation:**

| $\lvert d\rvert$ range | Effect size |
| ---------------------- | ----------- |
| $< 0.2$                | Negligible  |
| $0.2 - 0.5$            | Small       |
| $0.5 - 0.8$            | Medium      |
| $\ge 0.8$              | Large       |

#### 12.6 Statistical Analysis Pipeline

All statistical analyses follow a unified pipeline applied to each dataset type
(synthetic, classification, regression):

1. **Friedman omnibus test**: Tests whether at least one method differs
2. **Pairwise Wilcoxon + Holm**: Identifies which specific pairs differ
3. **Cohen's d**: Quantifies effect magnitude for each significant pair
4. **Bootstrap CIs**: 95% confidence intervals via 2000 bootstrap resamples
5. **Critical difference diagrams**: Visualizes method rankings with CD bars
6. **Kendall’s W**: Effect size for the Friedman test (reported alongside the
   chi-square statistic)

This ensures consistent, reproducible statistical comparisons across all
experiments.

**Implementation detail (Friedman/Nemenyi).** The Friedman test uses
complete-case rows across all methods for the metric. Critical-difference
diagrams use the Nemenyi CD computed from the Friedman ranks; the script also
computes pairwise Nemenyi significance matrices for reporting/plots.

**Output granularity.** The analysis script emits an overall aggregate
(averaging across all k values and downstream models) and also generates
per‑downstream‑model and per‑model‑per‑k outputs (Appendix H /
`paper/docs/README.md`).

#### 12.7 Bootstrap Confidence Intervals

To quantify uncertainty in performance estimates, we compute bootstrap
confidence intervals using the percentile method:

1. **Resample**: Draw $B = 2000$ bootstrap samples with replacement
2. **Compute**: Calculate mean for each bootstrap sample
3. **Percentile**: Take [2.5th, 97.5th] percentiles for 95% CI

$$
\text{CI}_{95\%} = [q_{0.025}, q_{0.975}]
$$

This non-parametric approach makes no distributional assumptions and provides a
simple percentile CI; we only report it when the sample size is $\geq 5$.
Bootstrap resampling is performed across independent seeds/datasets (not across
CV folds).

**Output format**: Results are reported as `mean [CI_lo, CI_hi]`, e.g.,
`0.847 [0.823, 0.871]`.

---

### 13. Implementation Details

#### 13.1 Technology Stack

| Component            | Technology   | Purpose                                        |
| -------------------- | ------------ | ---------------------------------------------- |
| Core                 | NumPy        | Array operations                               |
| JIT compilation      | Numba        | Performance-critical functions                 |
| Validation           | Pydantic v2  | Parameter validation                           |
| API                  | scikit-learn | BaseEstimator, ClassifierMixin, RegressorMixin |
| Parallelism          | joblib       | Forest training                                |
| Distance correlation | dcor         | Specialized dCor implementation                |

#### 13.2 Registry Pattern

Selectors and splitters are registered via decorators for extensibility:

```python
from citrees._registry import ClassifierSelectors

@ClassifierSelectors.register("mc")
@njit(cache=True, nogil=True, fastmath=True)
def mc(x, y, n_classes, random_state=None):
    ...
```

Available registries:

- `ClassifierSelectors` / `ClassifierSelectorTests`
- `RegressorSelectors` / `RegressorSelectorTests`
- `ClassifierSplitters` / `ClassifierSplitterTests`
- `RegressorSplitters` / `RegressorSplitterTests`
- `ThresholdMethods`

#### 13.3 Type System

All types are centralized in `citrees/_types.py`:

```python
# Numeric constraints
ProbabilityFloat = Annotated[float, Field(gt=0.0, le=1.0)]
PositiveInt = Annotated[int, Field(gt=0)]
ConfidenceFloat = Annotated[float, Field(gt=0.5, lt=1.0)]

# String enums
class EarlyStopping(StrEnum):
    ADAPTIVE = "adaptive"
    SIMPLE = "simple"

class NResamples(StrEnum):
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    AUTO = "auto"
```

#### 13.4 Computational Complexity

| Operation                      | Complexity           | Notes                                                                              |
| ------------------------------ | -------------------- | ---------------------------------------------------------------------------------- |
| Single permutation             | O(n)                 | Shuffle + O(n) statistic (e.g., `mc`, `pc`, impurities); some selectors are higher |
| Feature selection (1 feature)  | O(B · n)             | For O(n) selectors; e.g., `mc`, `pc`                                               |
| Feature selection (m features) | O(m · B · n)         | For O(n) selectors; with early stopping: often much less                           |
| Split selection (k thresholds) | O(k · B · n)         | Split statistic is O(n) per permutation; with early stopping: often much less      |
| Tree building                  | O(d · m · B · n)     | d = depth, typical d << n                                                          |
| Forest building (T trees)      | O(T · d · m · B · n) | Embarrassingly parallel                                                            |

**Memory**: O(n · p) for data storage, O(B) for permutation statistics.

**Selector-dependent costs.** The per-permutation “statistic cost” depends on
the selector:

- `mc`, `pc`: O(n)
- `rdc`: typically O(n log n) due to rank transforms and linear algebra
- `dc`: O(n²) for the standard distance-correlation estimator
- `mi`: depends on the estimator/configuration (often superlinear)

#### 13.5 Numerical Stability

- **Log-space computation**: Beta CDF uses log-gamma and log-exp
- **Continued fraction**: Lentz's algorithm with underflow protection
- **Division by zero**: Guarded with explicit checks and fallback values

---

### References

1. Hothorn, T., Hornik, K., & Zeileis, A. (2006). Unbiased recursive
   partitioning: A conditional inference framework. _Journal of Computational
   and Graphical Statistics_, 15(3), 651-674.

2. Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be
   zero: calculating exact P-values when permutations are randomly drawn.
   _Statistical Applications in Genetics and Molecular Biology_, 9(1),
   Article 39.

3. Westfall, P. H., & Young, S. S. (1993). _Resampling-based multiple testing:
   Examples and methods for p-value adjustment_. John Wiley & Sons.

4. Lopez-Paz, D., Hennig, P., & Schölkopf, B. (2013). The randomized dependence
   coefficient. _Advances in Neural Information Processing Systems_, 26.

5. Székely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007). Measuring and testing
   dependence by correlation of distances. _The Annals of Statistics_, 35(6),
   2769-2794.

6. Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous
   treatment effects using random forests. _Journal of the American Statistical
   Association_, 113(523), 1228-1242.

7. Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual
   information. _Physical Review E_, 69(6), 066138.

8. Strobl, C., Boulesteix, A. L., Zeileis, A., & Hothorn, T. (2007). Bias in
   random forest variable importance measures: Illustrations, sources and a
   solution. _BMC Bioinformatics_, 8(1), 25.
