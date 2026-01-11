# citrees: Mathematical Notes (Draft)

This file is meant to seed a *statistical* manuscript. It focuses only on claims that can be stated and proved with
high confidence from first principles. Anything that depends on adaptive, data-driven choices (tree growth, feature
muting, multi-selector selection, early stopping, etc.) is flagged explicitly.

## 0. Scope and “rigorous mode”

Most clean finite-sample guarantees in conditional inference trees come from (i) **exchangeability-based permutation
tests** and (ii) **multiple-testing corrections** applied to a *fixed family* of hypotheses.

For theorems/proofs below, the “rigorous mode” assumptions align with:

- Fixed resamples per test: `early_stopping_selector=False`, `early_stopping_splitter=False`
- Multiplicity correction enabled: `adjust_alpha_selector=True`, `adjust_alpha_splitter=True`
- Single-selector mode (not multi-selector): `selector` is a string, not a list
- No global feature muting across nodes: `feature_muting=False` (heuristic; complicates global error control)

These restrictions are not required to *run* the algorithm; they are the conditions under which we can write clean,
defensible statements.

## 1. Notation

Let the training data be i.i.d. samples
$$
\{(X_i, Y_i)\}_{i=1}^n \sim P,\quad X_i \in \mathbb{R}^p,
$$
with

- **Classification**: $Y_i \in \{1,\dots,K\}$,
- **Regression**: $Y_i \in \mathbb{R}$.

At a node $t$ in the tree, let $I_t \subseteq \{1,\dots,n\}$ be the index set of samples reaching the node, with
$n_t := |I_t|$, and let $X_{t} \in \mathbb{R}^{n_t \times p}$ and $Y_t \in \mathbb{R}^{n_t}$ (or $\{1,\dots,K\}^{n_t}$)
denote the restricted data.

The algorithm chooses a (possibly random) *candidate feature set* $F_t \subseteq \{1,\dots,p\}$, typically with
$|F_t| = m_t$ where $m_t$ is determined by `max_features`.

## 2. Nodewise hypotheses and test statistics

citrees separates (A) feature selection from (B) threshold selection.

### 2.1 Feature-selection nulls

For each candidate feature $j \in F_t$, define the nodewise null hypothesis
$$
H^{\text{sel}}_{t,j}: \; X_{t,j} \perp Y_t,
$$
where $X_{t,j} \in \mathbb{R}^{n_t}$ is the $j$-th feature column restricted to node $t$.

Let $T^{\text{sel}}_j(X_{t,j}, Y_t)$ be an association statistic. Examples implemented in citrees include:

- Classification: multiple correlation / ANOVA-type measures, mutual information, RDC
- Regression: Pearson correlation, distance correlation, RDC

For theory, the only requirement is that $T^{\text{sel}}_j$ is a measurable real-valued function of the data at node $t$.

### 2.2 Threshold-selection nulls

Fix a feature $j$ and a candidate threshold $c \in \mathbb{R}$. Define the induced partition
$$
I^L_t(j,c) := \{ i \in I_t : X_{ij} \le c \}, \qquad I^R_t(j,c) := I_t \setminus I^L_t(j,c).
$$
Let $I(\cdot)$ be a node impurity functional (e.g., Gini, entropy, MSE, MAE). Define the split statistic
$$
T^{\text{split}}_{j,c}(X_{t,j}, Y_t) := I(Y_{I^L_t(j,c)}) + I(Y_{I^R_t(j,c)}),
$$
where smaller values indicate “better” splits (lower within-child impurity).

For each $c$ in a candidate set $C_{t,j}$ (e.g., midpoints of unique values, or a subsample thereof), the natural null is
$$
H^{\text{split}}_{t,j,c}: \; Y_t \text{ is exchangeable w.r.t. the partition } (I^L_t(j,c), I^R_t(j,c)).
$$
In particular, if $X_{t,j} \perp Y_t$, then $H^{\text{split}}_{t,j,c}$ holds for all $c$.

### 2.3 Algorithm (formal node-level description)

We describe one node expansion in a way that matches the implementation in `citrees/_tree.py`:

Let $F_t \subseteq \{1,\dots,p\}$ be the candidate feature set (potentially a random subset of currently-available
features). Let $m_t := |F_t|$.

**Stage A (feature selection / stopping rule).**

1. For each $j \in F_t$, compute a permutation p-value $p_{t,j}$ for testing $H^{\text{sel}}_{t,j}$.
2. Let $j_t^\star := \arg\min_{j\in F_t} p_{t,j}$ and $p_t^\star := \min_{j\in F_t} p_{t,j}$.
3. If `adjust_alpha_selector=True`, compare against $\alpha_{\text{sel}}/m_t$ (Bonferroni); otherwise compare against
   $\alpha_{\text{sel}}$.
4. If $p_t^\star$ is not below the relevant threshold, stop and return a leaf node.

**Stage B (threshold selection / split validation).**

1. Given the selected feature $j_t^\star$, construct a finite threshold candidate set $C_{t,j_t^\star}$ with
   $\ell_t := |C_{t,j_t^\star}|$. In citrees, these are midpoints of unique sorted values (possibly subsampled).
2. For each $c \in C_{t,j_t^\star}$, compute a permutation p-value $p_{t,c}$ for testing $H^{\text{split}}_{t,j_t^\star,c}$
   using the split statistic $T^{\text{split}}_{j_t^\star,c}$.
3. Let $c_t^\star := \arg\min_{c\in C_{t,j_t^\star}} p_{t,c}$ and $p_{t,\text{split}}^\star := \min_{c\in C_{t,j_t^\star}} p_{t,c}$.
4. If `adjust_alpha_splitter=True`, compare against $\alpha_{\text{split}}/\ell_t$; otherwise compare against
   $\alpha_{\text{split}}$.
5. If $p_{t,\text{split}}^\star$ is not below the relevant threshold, stop and return a leaf node.

If both stages reject and additional deterministic constraints are satisfied (e.g., `min_samples_leaf`,
`min_impurity_decrease`), split the node into left/right children by the rule $X_{ij_t^\star}\le c_t^\star$ and recurse.

## 3. Permutation p-values (finite-sample validity)

### 3.0 Exact vs Monte Carlo permutation tests

For a fixed node $t$ and feature $j$, an *exact* permutation test would enumerate all $n_t!$ permutations (or all
distinct labelings under the relevant group action) to compute the null distribution of $T(X_{t,j}, Y_t)$.
citrees uses *Monte Carlo* permutation tests: sample $B$ random permutations and estimate the tail probability.

All results below are stated for the Monte Carlo test with fixed $B$.

### 3.1 Exchangeability assumption

Permutation tests rely on an exchangeability invariance. A sufficient condition for $H^{\text{sel}}_{t,j}$ is that,
conditional on the observed $X_{t,j}$, the labels $Y_t$ are exchangeable:
$$
(Y_{t,1},\dots,Y_{t,n_t}) \stackrel{d}{=} (Y_{t,\pi(1)},\dots,Y_{t,\pi(n_t)}) \quad \text{for all permutations } \pi.
$$
This is standard in randomization/permutation test theory and is satisfied under i.i.d. sampling with $X_{t,j} \perp Y_t$.

### 3.2 A key lemma: exchangeability of the permutation statistics

The “rank proof” of permutation p-value validity is simplest when the vector $(T_0,\dots,T_B)$ is exchangeable. We
spell out one sufficient construction that makes this true.

Let $\Pi_0,\Pi_1,\dots,\Pi_B$ be i.i.d. uniform random permutations of $\{1,\dots,n_t\}$, independent of $(X_t, Y_t)$.
Define
$$
T_b := T(X_{t,j}, \Pi_b(Y_t)) \quad (b=0,1,\dots,B),
$$
where $T$ is the chosen test statistic and tail.

**Lemma 1 (exchangeability of $(T_0,\dots,T_B)$).**  
Conditional on $(X_t, Y_t)$, the vector $(T_0,\dots,T_B)$ is exchangeable. Consequently, it is exchangeable
unconditionally.

**Proof.** Conditional on $(X_t,Y_t)$, the random variables $T_b$ are measurable functions of the i.i.d. random
permutations $(\Pi_b)_{b=0}^B$. Any permutation of the indices $b$ leaves the joint law unchanged because
$(\Pi_0,\dots,\Pi_B)$ is i.i.d. ∎

**Why this matches the practical procedure.**  
In practice, one typically sets $T_0 := T(X_{t,j}, Y_t)$ (the “unpermuted” labels) and samples only
$\Pi_1,\dots,\Pi_B$. Under the exchangeability of $Y_t$ (Section 3.1), $Y_t \stackrel{d}{=} \Pi_0(Y_t)$ for uniform
$\Pi_0$ independent of $Y_t$. Therefore the distribution of $(T_0,\dots,T_B)$ in the “unpermuted + $B$ permuted”
procedure is the same as in the i.i.d.-permutations construction above, and the validity conclusions apply.

### 3.3 Monte Carlo permutation p-value with +1 correction

Fix a statistic $T(\cdot,\cdot)$ (either feature selection or split selection, with the appropriate tail). Let
$\pi_1,\dots,\pi_B$ be i.i.d. uniform random permutations of $\{1,\dots,n_t\}$, drawn independently of the data.
Define permuted labels $Y_t^{(b)} := \pi_b(Y_t)$.

Define the observed and permuted statistics
$$
T_0 := T(X_{t,j}, Y_t), \qquad T_b := T(X_{t,j}, Y_t^{(b)}) \;\; (b=1,\dots,B).
$$

**Right-tail form (larger is “more extreme”):**
$$
p := \frac{1 + \sum_{b=1}^B \mathbf{1}\{T_b \ge T_0\}}{B+1}.
$$

**Left-tail form (smaller is “more extreme”):**
$$
p := \frac{1 + \sum_{b=1}^B \mathbf{1}\{T_b \le T_0\}}{B+1}.
$$

This is exactly the “+1 correction” recommended by Phipson & Smyth (2010) for Monte Carlo permutation tests.

### 3.4 Theorem: super-uniformity of the +1 p-value

**Theorem 1 (Monte Carlo permutation p-values are valid).**  
Assume that under the null hypothesis of interest, the random variables $(T_0, T_1, \dots, T_B)$ are exchangeable.
Let $p$ be defined by either tail form above (using $\ge$ or $\le$ consistently). Then for all $\alpha \in [0,1]$,
$$
\mathbb{P}(p \le \alpha) \le \alpha.
$$
If $(T_0,\dots,T_B)$ is almost surely free of ties, then $p$ is discrete-uniform on $\{1/(B+1), \dots, 1\}$ and
$$
\mathbb{P}(p \le \alpha) = \frac{\lfloor (B+1)\alpha \rfloor}{B+1}.
$$

**Proof (rank argument).**  
We present the right-tail case; the left-tail is identical with inequalities reversed.

Define the (upper) rank of $T_0$ among $\{T_0,\dots,T_B\}$ as
$$
R := 1 + \sum_{b=1}^B \mathbf{1}\{T_b \ge T_0\},
$$
so $p = R/(B+1)$.

If there are no ties almost surely, then $R \in \{1,\dots,B+1\}$ is exactly the rank of $T_0$ in *descending* order.
Exchangeability implies that each position is equally likely, hence
$$
\mathbb{P}(R = r) = \frac{1}{B+1}, \quad r=1,\dots,B+1,
$$
which yields the discrete-uniform claim and the displayed CDF formula.

If ties can occur, then using $\mathbf{1}\{T_b \ge T_0\}$ yields a conservative p-value (it “counts ties against” the
null). Formally, introduce i.i.d. $U_0,\dots,U_B \sim \mathrm{Unif}(0,1)$ independent of $(T_0,\dots,T_B)$ and define the
tie-broken statistics $\widetilde{T}_b := (T_b, U_b)$ ordered lexicographically. Then $(\widetilde{T}_0,\dots,
\widetilde{T}_B)$ is exchangeable and almost surely tie-free, so the corresponding tie-broken p-value
$$
\widetilde{p} := \frac{1 + \sum_{b=1}^B \mathbf{1}\{\widetilde{T}_b \ge \widetilde{T}_0\}}{B+1}
$$
satisfies $\mathbb{P}(\widetilde{p} \le \alpha)=\lfloor (B+1)\alpha\rfloor/(B+1) \le \alpha$.
Moreover, for every realization, $\mathbf{1}\{\widetilde{T}_b \ge \widetilde{T}_0\} \le \mathbf{1}\{T_b \ge T_0\}$, hence
$\widetilde{p}\le p$ pointwise and therefore
$\mathbb{P}(p \le \alpha) \le \mathbb{P}(\widetilde{p} \le \alpha) \le \alpha$. ∎

**How exchangeability arises in permutation tests.**  
Under $H^{\text{sel}}_{t,j}$ (or $H^{\text{split}}_{t,j,c}$), the conditional distribution of $Y_t$ given the covariates
that define the null is invariant to permutations. Therefore $T(X_{t,j}, Y_t)$ has the same distribution as
$T(X_{t,j}, \pi(Y_t))$ for uniform $\pi$, and the collection $(T_0,\dots,T_B)$ is exchangeable.

### 3.5 Conditional (on-$X$) validity

The most defensible way to state permutation-test validity in a tree context is conditional on the covariates that are
treated as fixed by the permutation procedure.

**Corollary 2 (conditional super-uniformity).**  
Fix $(X_t, U)$ where $U$ denotes any algorithmic randomness used to choose the candidate family (e.g., the indices in
`max_features`, or subsampled thresholds) and assume that, under the null hypothesis being tested, $Y_t$ is
exchangeable conditional on $(X_t,U)$. Then the +1 Monte Carlo permutation p-value satisfies
$$
\mathbb{P}(p \le \alpha \mid X_t, U) \le \alpha \quad \text{for all } \alpha\in[0,1].
$$

**Proof.** Conditional on $(X_t,U)$, the candidate family and the statistic are fixed, and the exchangeability
assumption reduces to the setup of Lemma 1 + Theorem 1. ∎

### 3.6 Monte Carlo resolution and error (fixed $B$)

Even when the *test* is valid, finite $B$ limits the granularity of the p-value and introduces Monte Carlo variability.
This can be described exactly.

Let
$$
p^\star := \mathbb{P}\big(T(X_{t,j}, \Pi(Y_t)) \text{ is at least as extreme as } T(X_{t,j}, Y_t)\;\big|\;X_t,Y_t\big),
$$
where $\Pi$ is a uniform random permutation independent of everything. Under the Monte Carlo scheme with i.i.d.
permutations, the exceedance count
$$
K := \sum_{b=1}^B \mathbf{1}\{T_b \ge T_0\} \quad \text{(or } \mathbf{1}\{T_b \le T_0\}\text{)}
$$
satisfies, conditional on $(X_t,Y_t)$,
$$
K \sim \mathrm{Binomial}(B, p^\star),
$$
so the reported +1 p-value is $p=(1+K)/(B+1)$.

From this representation:

1. **Resolution:** $p \in \{1/(B+1), 2/(B+1), \dots, 1\}$. In particular, to have the possibility of rejecting at level
   $\alpha$, one needs $1/(B+1)\le \alpha$, i.e. $B \ge 1/\alpha - 1$.
2. **Concentration:** by Hoeffding’s inequality, for any $\varepsilon>0$,
   $$
   \mathbb{P}\Big(\Big|\frac{K}{B} - p^\star\Big|\ge \varepsilon \;\Big|\; X_t,Y_t\Big) \le 2e^{-2B\varepsilon^2}.
   $$
   This gives a simple “how many permutations are needed” calibration for Monte Carlo accuracy (separate from validity).

## 4. Multiplicity correction (Bonferroni)

### 4.1 Lemma: Bonferroni with super-uniform p-values

**Lemma 2 (Bonferroni control).**  
Let $p_1,\dots,p_m$ be p-values for hypotheses $H_1,\dots,H_m$ such that, for each true null $H_j$,
$\mathbb{P}(p_j \le u) \le u$ for all $u \in [0,1]$. Then under the global null (all $H_j$ true),
$$
\mathbb{P}\Big(\min_{1\le j\le m} p_j \le \frac{\alpha}{m}\Big) \le \alpha.
$$

**Proof.** By the union bound,
$$
\mathbb{P}\Big(\min_j p_j \le \frac{\alpha}{m}\Big)
=
\mathbb{P}\Big(\bigcup_{j=1}^m \{p_j \le \alpha/m\}\Big)
\le \sum_{j=1}^m \mathbb{P}(p_j \le \alpha/m)
\le m \cdot \frac{\alpha}{m} = \alpha.
$$
No independence assumptions are required. ∎

### 4.2 Proposition: nodewise false split probability (global-null bound)

This is the cleanest statement we can make without getting into adaptive inference across the entire tree.

**Proposition 3 (nodewise global-null control for feature selection).**  
At a fixed node $t$, let $F_t$ be the candidate feature set with $m_t := |F_t|$. Assume the global null holds:
$H^{\text{sel}}_{t,j}$ is true for all $j \in F_t$. Let $p_{t,j}$ be valid permutation p-values as in Theorem 1 and
apply Bonferroni by using the threshold $\alpha_{\text{sel}}/m_t$. Then
$$
\mathbb{P}\Big(\exists j\in F_t:\; p_{t,j} \le \alpha_{\text{sel}}/m_t\Big) \le \alpha_{\text{sel}}.
$$

**Proof.** Apply Lemma 2 with $m=m_t$. ∎

**Proposition 3a (per-feature false selection bound).**  
Under the assumptions of Proposition 3, fix any particular feature $j\in F_t$. If the algorithm splits at node $t$ and
chooses feature $j$ (i.e., $j_t^\star=j$), then necessarily $p_{t,j}\le \alpha_{\text{sel}}/m_t$. Consequently,
$$
\mathbb{P}(j_t^\star = j \text{ and node } t \text{ splits}) \le \alpha_{\text{sel}}/m_t.
$$

**Proof.** The event $\{j_t^\star=j \text{ and split at }t\}$ implies $p_{t,j} = \min_{k\in F_t} p_{t,k} \le
\alpha_{\text{sel}}/m_t$, hence
$$
\mathbb{P}(j_t^\star=j \text{ and split at }t) \le \mathbb{P}(p_{t,j}\le \alpha_{\text{sel}}/m_t) \le \alpha_{\text{sel}}/m_t
$$
by Corollary 2 / Theorem 1 (super-uniformity for a true null). ∎

**Proposition 3′ (nodewise global-null control for threshold selection, fixed feature).**  
Fix a node $t$ and a feature $j$, and let $C_{t,j}$ be the candidate threshold set with $|C_{t,j}| = \ell_{t,j}$.
Assume $H^{\text{sel}}_{t,j}$ holds (so $X_{t,j}\perp Y_t$), which implies $H^{\text{split}}_{t,j,c}$ holds for all
$c\in C_{t,j}$. Let $p_{t,j,c}$ be valid permutation p-values for the split statistic and apply Bonferroni by using the
threshold $\alpha_{\text{split}}/\ell_{t,j}$. Then
$$
\mathbb{P}\Big(\exists c\in C_{t,j}:\; p_{t,j,c} \le \alpha_{\text{split}}/\ell_{t,j}\Big) \le \alpha_{\text{split}}.
$$

**Proof.** Apply Lemma 2 with $m=\ell_{t,j}$. ∎

**Remarks.**

1. This bound is *nodewise* and does not claim global family-wise error control over the entire adaptively-grown tree.
2. Additional constraints (e.g., `min_samples_leaf`, `min_impurity_decrease`) can only reduce the probability of making a
   split, so they preserve the inequality.

## 4.3 Random feature/threshold subsampling (root-level validity)

citrees optionally tests only a subset of features (`max_features`) and/or a subset of thresholds (`max_thresholds`
through the threshold method). This is statistically safe at the *root*, because these candidate sets are functions of
the covariates and the algorithm RNG, not of the labels.

**Proposition 3″ (root-level validity under random candidate sets).**  
Consider the root node (so the candidate pool is the full sample). Suppose:

1. Under the null, $Y$ is exchangeable conditional on $X$ (e.g., i.i.d. with $X \perp Y$).
2. The tested feature set $F$ is (possibly random) but measurable w.r.t. $(X, U)$ where $U$ is algorithm randomness
   independent of $Y$.
3. For each tested feature $j\in F$, the threshold candidate set $C_j$ is (possibly random) but measurable w.r.t.
   $(X_{\cdot j}, U)$, independent of $Y$ given $(X,U)$.

Then any +1-corrected permutation p-values computed by permuting $Y$ (holding $X$ and the candidate sets fixed) are
super-uniform under the null, and Bonferroni correction over the tested families yields the corresponding root-level
FWER bounds.

**Proof sketch.** Conditional on $(X,U)$, the tested families are fixed and $Y$ remains exchangeable under the null.
Apply Theorem 1 to each tested hypothesis and then Lemma 2 to the family. ∎

### 4.4 Root-level “tree makes any split” guarantee (safe global statement)

Tree adaptivity makes it hard to interpret internal-node tests as classical inferential p-values. However, there is one
global statement that remains clean: **a tree can only split if the root rejects.**

**Proposition 3b (global-null split probability at the root).**  
Consider the root node and suppose the global null holds for all tested features at the root (i.e., for every
$j\in F_{\text{root}}$, $X_{\cdot j}\perp Y$ so the feature-selection null is true). In “rigorous mode” (fixed $B$ and
Bonferroni over the tested feature family), we have
$$
\mathbb{P}(\text{the fitted tree has at least one internal split}) \le \alpha_{\text{sel}}.
$$

**Proof.** The event “the fitted tree has at least one split” implies “the root splits,” which implies that at the root
we rejected at least one of the feature-selection nulls. Apply Proposition 3 at the root. ∎

**Corollary 3c (per-feature root split bound).**  
Under the assumptions of Proposition 3b with $m_{\text{root}} := |F_{\text{root}}|$, for any particular tested feature
$j\in F_{\text{root}}$,
$$
\mathbb{P}(\text{the root splits on feature }j) \le \alpha_{\text{sel}}/m_{\text{root}}.
$$

**Proof.** This is Proposition 3a specialized to the root. ∎

## 5. Honest estimation: unbiased leaf predictions (when leaves get estimation data)

citrees optionally uses sample splitting (“honesty”) to decouple structure learning from leaf estimation.

### 5.1 Setup

Split the indices $\{1,\dots,n\}$ into disjoint sets $S$ (“splitting”) and $E$ (“estimation”), using a random split
independent of the data. Build the tree structure (the partition of feature space into leaves) using only data indexed
by $S$. Let $\Pi$ denote the resulting partition into leaves (a random object measurable w.r.t. the $\sigma$-field
generated by $\{(X_i,Y_i)\}_{i\in S}$).

For a leaf (cell) $L \in \Pi$, define the estimation indices landing in that leaf
$$
E(L) := \{ i \in E : X_i \in L \}.
$$

### 5.2 Proposition: unbiasedness conditional on the learned partition

**Proposition 4 (honest leaf mean is unbiased).**  
Assume i.i.d. sampling. Consider regression, and define the honest leaf estimator
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

**Proof.** Conditional on $\Pi$, the set $E$ is independent of $\Pi$, so the samples $\{(X_i,Y_i)\}_{i\in E}$ are i.i.d.
from $P$ and can be viewed as an i.i.d. sample passed through the (random but fixed) measurable set $L$.
Conditional on $\Pi$ and on the event $\{|E(L)|\ge 1\}$, the variables $\{Y_i : i\in E(L)\}$ are i.i.d. with mean
$\mu(L)$. Therefore the sample average has conditional expectation $\mu(L)$. ∎

**Classification analogue.**  
For classification, the honest leaf class-probability vector
$$
\widehat{p}_k(L) := \frac{1}{|E(L)|}\sum_{i\in E(L)} \mathbf{1}\{Y_i = k\}
$$
is similarly unbiased for $p_k(L) := \mathbb{P}(Y=k \mid X\in L)$, conditional on $\Pi$, on $\{|E(L)|\ge 1\}$.

**Important implementation note.**  
If a leaf receives zero estimation samples, citrees currently retains the splitting-sample leaf value. That fallback is
not covered by Proposition 4.

## 6. Where proofs stop (and why)

The following aspects are important in practice but require much more care (or algorithm changes) to attach
publication-grade proofs:

1. **Early stopping inside permutation testing** (`early_stopping_* = True`).  
   Because the number of permutations becomes a stopping time depending on partial results, the returned quantity is not
   the fixed-$B$ Monte Carlo p-value of Theorem 1. For strict inferential claims, disable early stopping.

2. **Multi-selector mode** (`selector=[...]`).  
   In current code, the selector statistic used for a feature is chosen *after looking at the data* (pick the max score
   among selectors) but the permutation reference distribution is computed for that chosen statistic alone, without
   adjusting for the selection over selectors. This is a classic “selective inference” issue; a rigorous alternative is
   to define a composite statistic
   $$
   T^{\text{sel}}(X_{t,j},Y_t) := \max_{s\in \mathcal{S}} T^{\text{sel}}_s(X_{t,j},Y_t)
   $$
   and compute the permutation p-value using the *same max* inside each permutation. (That is provably valid by Theorem
   1, but it is not what the current implementation does.)

3. **Feature muting across nodes** (`feature_muting=True`).  
   Muting uses intermediate p-values to remove features globally from future consideration. This adaptively changes the
   hypothesis family across the tree and makes global error statements subtle.

4. **Global error control across the entire tree.**  
   Even with valid nodewise tests, the tree construction is adaptive; p-values at internal nodes should not be read as
   classical inferential p-values for a fixed hypothesis family without additional machinery (sample splitting, selective
   inference, or fully specified global testing procedures).

5. **Forest-level theory.**  
   Consistency, rates, and uncertainty quantification for the forest (especially with permutation-based splitting)
   require assumptions and arguments beyond what can be asserted from the current implementation alone.

## 7. Citations to include (placeholders)

Add BibTeX entries for:

- Conditional inference trees: Hothorn, Hornik, Zeileis (2006)
- Linear-statistic permutation tests used in ctree lineage: Strasser & Weber (1999) (if you want to track the original
  conditional inference machinery)
- +1 correction for Monte Carlo permutation tests: Phipson & Smyth (2010)
- General permutation test background (optional but often expected): e.g. Lehmann & Romano, *Testing Statistical
  Hypotheses* (or a standard permutation test reference you prefer)
- Multiple testing correction background (optional): Bonferroni, Holm (1979)
- RDC: Lopez-Paz, Hennig, Schölkopf (2013)
- Distance correlation: Székely, Rizzo, Bakirov (2007) (and/or subsequent dCor references)
- Mutual information (if used): Kraskov, Stögbauer, Grassberger (2004) for KSG-style estimation
- Honesty / honest forests: Wager & Athey (2018); Athey, Tibshirani, Wager (2019)

## 8. Implementation-to-theory alignment checklist (paper-facing)

Before making inferential claims in a paper, it’s worth explicitly deciding which of the following you will *support*:

1. **Fixed-$B$ permutation p-values only.**  
   Use `early_stopping_* = False` so Theorem 1 applies as stated.

2. **Single-selector p-values only.**  
   Use `selector="mc"` / `"pc"` / `"rdc"` etc. If you want multi-selector, change the statistic to a max-over-selectors
   *inside each permutation* (Section 6.2).

3. **Multiplicity correction actually matches the tested family.**  
   If you subsample features (`max_features`) or thresholds (`max_thresholds`), be explicit in the paper: you are
   controlling error over the tested subset, not over all $p$ features / all thresholds.

4. **Phipson–Smyth +1 correction everywhere.**  
   In `citrees/_splitter.py`, the parallel MAE permutation test (`_ptest_mae_parallel`) currently returns
   `np.mean(theta_p <= theta)` (no +1 correction). For strict alignment with Section 3, either (a) avoid MAE in paper
   runs, or (b) patch the implementation to use `(1 + count)/(1 + B)` consistently.
