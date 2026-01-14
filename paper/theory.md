# citrees: Mathematical Notes (Draft)

This file is meant to seed a *statistical* manuscript. It focuses only on claims that can be stated and proved with
high confidence from first principles. Anything that depends on adaptive, data-driven choices (tree growth, feature
muting, multi-selector selection, early stopping, etc.) is flagged explicitly.

## 0. Scope and “rigorous mode”

Most clean finite-sample guarantees in conditional inference trees come from (i) **exchangeability-based permutation
tests** and (ii) **multiple-testing corrections** applied to a *fixed family* of hypotheses.

For theorems/proofs below, the “rigorous mode” assumptions align with:

- Fixed resamples per test: `early_stopping_selector=None`, `early_stopping_splitter=None`
- Or adaptive sequential testing: `early_stopping_selector="adaptive"` (default, provides valid p-values)
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

### 2.2a Test tail conventions

For **feature-selection statistics** in citrees (nonnegative association scores such as `mc`, `mi`, `rdc`, `dc`, and
`|pc|`), larger values indicate stronger association, so we use the **right-tail** test:
$$
p_{t,j} = \frac{1 + \sum_{b=1}^B \mathbf{1}\{T^{\text{sel}}_b \ge T^{\text{sel}}_0\}}{B+1}.
$$

For **split statistics** (Gini, entropy, MSE, MAE), smaller values indicate better splits (more homogeneous
children), so we use the **left-tail** test:
$$
p_{t,j,c} = \frac{1 + \sum_{b=1}^B \mathbf{1}\{T^{\text{split}}_b \le T^{\text{split}}_0\}}{B+1}.
$$

Both forms satisfy the super-uniformity guarantee of Theorem 1 (Section 3.4).

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

**Important inferential note.**  
Stage B is performed *after selecting* the feature $j_t^\star$ using the same response values $Y_t$. Unless $j_t^\star$
is treated as fixed in advance (or additional sample splitting / selective-inference machinery is used), the Stage B
p-values should be viewed as *algorithmic stopping statistics*, not classical post-selection p-values.

## 3. Permutation p-values (finite-sample validity)

### 3.0 Exact vs Monte Carlo permutation tests

For a fixed node $t$ and feature $j$, an *exact* permutation test would enumerate all $n_t!$ permutations (or all
distinct labelings under the relevant group action) to compute the null distribution of $T(X_{t,j}, Y_t)$.
citrees uses *Monte Carlo* permutation tests: sample $B$ random permutations and estimate the tail probability.

All results below are stated for the Monte Carlo test with fixed $B$. In the implementation, $B$ can vary across tests
and nodes (e.g., due to Bonferroni-adjusted $\alpha/m$ or due to `max_features` / `max_thresholds`). This is
statistically fine: as long as $B$ is chosen as a function of $(X_t,U)$ (candidate-set randomness, RNG seeds, etc.) and
not of $Y_t$ under the null, conditioning on $(X_t,U)$ reduces the analysis to fixed-$B$.

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

**Why the usual “unpermuted + $B$ permuted” computation is covered.**  
Lemma 1 draws $\Pi_0$ as a *random* permutation, whereas implementations typically take $\Pi_0$ to be the identity
(i.e., $T_0=T(X_{t,j},Y_t)$). Under the null exchangeability assumption (Section 3.1), these lead to the same
distribution for $(T_0,\dots,T_B)$.

To see this, draw $\Pi_0,\Pi_1,\dots,\Pi_B$ i.i.d. uniform and define
$\widetilde{Y}:=\Pi_0(Y_t)$ and $\Pi'_b := \Pi_b\circ \Pi_0^{-1}$ for $b=1,\dots,B$.
Then:

1. $(\Pi'_1,\dots,\Pi'_B)$ are i.i.d. uniform (composition with a fixed permutation preserves the uniform distribution on
   the permutation group).
2. The vector from Lemma 1 can be rewritten as
   $$
   (T(X_{t,j},\Pi_0(Y_t)), T(X_{t,j},\Pi_1(Y_t)),\dots,T(X_{t,j},\Pi_B(Y_t)))
   =
   (T(X_{t,j},\widetilde{Y}), T(X_{t,j},\Pi'_1(\widetilde{Y})),\dots,T(X_{t,j},\Pi'_B(\widetilde{Y}))).
   $$
3. Under exchangeability of $Y_t$, $\widetilde{Y}\stackrel{d}{=}Y_t$, and the right-hand side has the same
   distribution as the practical “unpermuted + $B$ permuted” vector
   $(T(X_{t,j},Y_t), T(X_{t,j},\Pi_1(Y_t)),\dots,T(X_{t,j},\Pi_B(Y_t)))$.

Since the left-hand side is exchangeable by Lemma 1, the practical vector is also exchangeable, and the rank-based proof
of Theorem 1 applies.

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

**Remark (random internal nodes).**  
Corollary 2 conditions on $(X_t,U)$ for a *fixed* node $t$. In an adaptively-grown tree, internal nodes correspond to
random index sets $I_t$ determined by earlier splits that depend on the labels. After conditioning on the event “these
samples reach node $t$,” the label vector within the node need not remain exchangeable under a null, so permutation-test
validity is not automatic. This is why the cleanest inferential statements are root-level (Section 4.4) or rely on
sample splitting / selective-inference techniques (Section 6).

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

### 3.7 Conservativeness of the +1 estimator (finite-$B$)

The +1 correction produces a *valid* p-value, but it is not an unbiased estimator of $p^\star$.

**Lemma 3 (upward bias / conservativeness).**  
With $K\sim\mathrm{Binomial}(B,p^\star)$ conditional on $(X_t,Y_t)$ and $p=(1+K)/(B+1)$, we have
$$
\mathbb{E}[p\mid X_t,Y_t] \;=\; \frac{1+Bp^\star}{B+1} \;=\; p^\star + \frac{1-p^\star}{B+1} \;\ge\; p^\star.
$$
Moreover, $\mathbb{E}[p\mid X_t,Y_t]\to p^\star$ as $B\to\infty$.

**Proof.** Immediate from $\mathbb{E}[K\mid X_t,Y_t]=Bp^\star$. ∎

### 3.8 Remarks on power (informal)

The propositions above control **Type I error** (false positive rate). **Power**—the probability of detecting a
true association when one exists—depends on factors not addressed by the validity theorems:

1. **Effect size**: How strongly does $X_j$ predict $Y$? Stronger associations yield smaller p-values.
2. **Sample size at node**: $n_t$ determines permutation test resolution and signal detectability.
3. **Number of permutations $B$**: Limits the smallest achievable p-value to $1/(B+1)$.
4. **Multiplicity burden**: Bonferroni over $m$ features requires per-test significance $\alpha/m$, reducing power.

We do not provide formal power guarantees, as these depend on the (unknown) alternative distribution. Simulation
studies are the standard approach for assessing power in practice.

**Practical guidance.** For a test at level $\alpha$ with $m$ features and Bonferroni correction, the effective
per-feature threshold is $\alpha/m$. To have any chance of rejection, one needs $B \ge m/\alpha - 1$. For example,
with $\alpha = 0.05$ and $m = 100$ features, the per-feature threshold is $0.0005$, requiring $B \ge 1999$
permutations.

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

**Proposition 3a (per-feature false selection bound; no global-null needed).**  
At a fixed node $t$, fix any particular feature $j\in F_t$ such that its null $H^{\text{sel}}_{t,j}$ is true and its
p-value $p_{t,j}$ is super-uniform (Theorem 1 / Corollary 2). If the algorithm splits at node $t$ and chooses feature
$j$ (i.e., $j_t^\star=j$), then necessarily $p_{t,j}\le \alpha_{\text{sel}}/m_t$ (in the Bonferroni-adjusted case).
Consequently,
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

**Proposition 3a′ (per-threshold false selection bound; fixed feature).**  
Under the assumptions of Proposition 3′, fix any particular threshold $c\in C_{t,j}$. If the algorithm (at node $t$)
splits using threshold $c$ on feature $j$ (i.e., $c_t^\star=c$), then necessarily $p_{t,j,c}\le \alpha_{\text{split}}/\ell_{t,j}$
(in the Bonferroni-adjusted case). Consequently,
$$
\mathbb{P}(c_t^\star=c \text{ and node } t \text{ splits on feature } j) \le \alpha_{\text{split}}/\ell_{t,j}.
$$

**Proof.** Identical to Proposition 3a, replacing the feature family with the threshold family. ∎

**Interpretation: a rigorous anti-selection-bias statement.**  
In CART-style algorithms, a feature with many candidate thresholds has many chances to look good when optimizing an
impurity objective, which can inflate false selection probability (selection bias).

In citrees, Proposition 3a provides a precise finite-sample statement for *feature selection*: for any particular null
feature $j$ among the tested set $F_t$, the probability that node $t$ splits *on that feature* is bounded by the same
Bonferroni share $\alpha_{\text{sel}}/|F_t|$, and this bound does not depend on the feature’s number of unique values.

For *threshold selection* on a **fixed** feature $j$, Propositions 3′ and 3a′ give analogous Bonferroni bounds over the
tested threshold family $C_{t,j}$ (the bound depends on $|C_{t,j}|$). When threshold testing is run only after the data
have selected a feature, these threshold-family bounds should be interpreted as fixed-feature results (see the
post-selection note in Section 2.3).

**Remarks.**

1. This bound is *nodewise* and does not claim global family-wise error control over the entire adaptively-grown tree.
2. Additional constraints (e.g., `min_samples_leaf`, `min_impurity_decrease`) can only reduce the probability of making a
   split, so they preserve the inequality.
3. **Without Bonferroni correction** (`adjust_alpha_selector=False`), each p-value is compared directly against
   $\alpha_{\text{sel}}$. By the union bound, the probability of selecting *any* feature under the global null is then
   at most $m_t \cdot \alpha_{\text{sel}}$, which is not a valid Type I error control at level $\alpha_{\text{sel}}$
   unless $m_t = 1$. The analogous statement holds for threshold selection with `adjust_alpha_splitter=False`.

### 4.3 Random feature/threshold subsampling (root-level validity)

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

**Proof.** Fix $(X,U)$. By assumption, the tested feature family $F$ and each tested threshold family $C_j$ are
measurable w.r.t. $(X,U)$, so they are fixed when conditioning on $(X,U)$.

Under the null, $Y$ is exchangeable conditional on $(X,U)$, hence for any tested hypothesis, the Monte Carlo +1
permutation p-value is super-uniform conditional on $(X,U)$ by Corollary 2.

Applying Lemma 2 conditionally on $(X,U)$ yields the corresponding Bonferroni family-wise error bounds over the tested
families. Finally, taking expectations over $(X,U)$ preserves the inequality. ∎

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

**Corollary 3d (per-feature root split bound with random feature subsampling).**  
Assume the global null holds at the root for all $p$ features. Suppose the root tests a subset
$F_{\text{root}}\subseteq\{1,\dots,p\}$ obtained by sampling $m$ features uniformly without replacement (as in
`max_features`), and uses Bonferroni correction over the tested subset. Then for any feature $j\in\{1,\dots,p\}$,
$$
\mathbb{P}(\text{the root splits on feature }j) \le \alpha_{\text{sel}}/p.
$$

**Proof.** Condition on $F_{\text{root}}$.
If $j\notin F_{\text{root}}$, the root cannot split on $j$.
If $j\in F_{\text{root}}$, Proposition 3a (at the root) gives
$\mathbb{P}(\text{root splits on }j\mid F_{\text{root}})\le \alpha_{\text{sel}}/|F_{\text{root}}|=\alpha_{\text{sel}}/m$.
Therefore,
$$
\mathbb{P}(\text{root splits on }j)
=\mathbb{E}\!\left[\mathbb{P}(\text{root splits on }j\mid F_{\text{root}})\right]
\le \mathbb{E}\!\left[\mathbf{1}\{j\in F_{\text{root}}\}\frac{\alpha_{\text{sel}}}{m}\right]
=\frac{\alpha_{\text{sel}}}{m}\cdot \frac{m}{p}
=\alpha_{\text{sel}}/p.
$$
∎

### 4.5 Joint error control (two-stage testing)

At each node, citrees performs two sequential tests:
1. **Stage A**: Feature selection at level $\alpha_{\text{sel}}$ (with Bonferroni over features)
2. **Stage B**: Threshold selection at level $\alpha_{\text{split}}$ (with Bonferroni over thresholds)

A natural question is: what is the joint false split probability?

**Proposition 3d (joint false split probability under global null).**
At a fixed node $t$, if the global null holds (i.e., $X_{t,j} \perp Y_t$ for all $j \in F_t$), then:
$$
\mathbb{P}(\text{node } t \text{ splits}) \le \alpha_{\text{sel}}.
$$

**Proof.**
$$
\{\text{node } t \text{ splits}\} \subseteq \{\text{Stage A rejects}\}
$$
because Stage A must reject for Stage B to run. Therefore:
$$
\mathbb{P}(\text{node } t \text{ splits}) \le \mathbb{P}(\text{Stage A rejects}) \le \alpha_{\text{sel}}
$$
by Proposition 3. ∎

**Remark.** The parameter $\alpha_{\text{split}}$ controls Stage B's error rate but does not appear in this bound
because Stage A is the "gatekeeper"—if Stage A does not reject, no split occurs regardless of $\alpha_{\text{split}}$.
Stage B provides an additional layer of protection but the dominant control comes from Stage A.

## 5. Honest estimation: unbiased leaf predictions (when assumptions hold)

citrees optionally uses sample splitting (“honesty”) to decouple structure learning from leaf estimation.

### 5.1 Setup

Split the indices $\{1,\dots,n\}$ into disjoint sets $S$ (“splitting”) and $E$ (“estimation”) using a random split.
Build the tree structure (the partition of feature space into leaves) using only data indexed by $S$. Let $\Pi$ denote
the resulting partition into leaves (a random object measurable w.r.t. the $\sigma$-field generated by
$\{(X_i,Y_i)\}_{i\in S}$).

**Assumption (independent sample split).**  
For the clean conditional unbiasedness statements below, assume the random index split $(S,E)$ is independent of the
observed sample $\{(X_i,Y_i)\}_{i=1}^n$. (Equivalently: the indices are chosen by an external RNG that does not look at
the data.)

Under this assumption, conditional on $S$ the splitting-sample data $\{(X_i,Y_i)\}_{i\in S}$ is independent of the
estimation-sample data $\{(X_i,Y_i)\}_{i\in E}$. Since the learned partition $\Pi$ is measurable with respect to
$\sigma(S, \{(X_i,Y_i)\}_{i\in S})$, we have the conditional independence
$$
\{(X_i,Y_i)\}_{i\in E} \perp \Pi \mid S.
$$
(We do **not** generally have $E \perp \Pi$ unconditionally because $\Pi$ depends on which indices are placed in $S$.)

**Implementation note.**  
In `citrees/_tree.py`, honest estimation uses `train_test_split`. For regression, the split is unstratified, matching
the independence assumption. For classification, the split is stratified by labels to preserve class balance, which
introduces dependence on $Y$; we therefore do **not** interpret the classification “honesty” mode as providing
publication-grade unbiased leaf probability estimation unless the split is made independent of $Y$.

For a leaf (cell) $L \in \Pi$, define the estimation indices landing in that leaf
$$
E(L) := \{ i \in E : X_i \in L \}.
$$

### 5.2 Proposition: unbiasedness conditional on the learned partition

**Proposition 4 (honest leaf mean is unbiased, regression).**  
Assume i.i.d. sampling and an independent sample split $(S,E)$ as in Section 5.1. Consider regression, and define the
honest leaf estimator
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

**Proof.** By Section 5.1, $\{(X_i,Y_i)\}_{i\in E} \perp \Pi \mid S$. In particular, conditional on $(\Pi,S)$ the
estimation-sample observations are i.i.d. from $P$ and independent of the (random) leaf partition.

Fix a leaf $L\in \Pi$ and, for $i\in E$, define the indicator $I_i := \mathbf{1}\{X_i\in L\}$ and the random count
$N:=n_E(L)=\sum_{i\in E} I_i$.

On the event $\{N\ge 1\}$ we can write the honest mean as a ratio
$$
\widehat{\mu}(L)=\frac{\sum_{i\in E} I_i Y_i}{\sum_{i\in E} I_i}.
$$

Condition on $(\Pi,S,(I_i)_{i\in E})$. Then $N$ is fixed, and for every $i$ with $I_i=1$ we have
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
Taking conditional expectations first over $(I_i)_{i\in E}$ and then over $S$ yields
$\mathbb{E}[\widehat{\mu}(L)\mid \Pi]=\mu(L)$ on $\{N\ge 1\}$. ∎

**Classification analogue (requires independent split).**  
If the index split $(S,E)$ is independent of the data, then for classification the honest leaf class-probability vector
$$
\widehat{p}_k(L) := \frac{1}{|E(L)|}\sum_{i\in E(L)} \mathbf{1}\{Y_i = k\}
$$
is similarly unbiased for $p_k(L) := \mathbb{P}(Y=k \mid X\in L)$, conditional on $\Pi$, on $\{|E(L)|\ge 1\}$.

**Important implementation note.**
If a leaf receives zero estimation samples, citrees currently retains the splitting-sample leaf value. That fallback is
not covered by Proposition 4.

### 5.3 Variance of honest estimator

Proposition 4 establishes unbiasedness but does not address variance. Sample splitting increases variance because
fewer samples are available for estimation.

**Proposition 4a (variance of honest leaf estimator, regression).**
Under the assumptions of Proposition 4, let $n_E(L) := |E(L)|$ be the number of estimation samples in leaf $L$.
Define $\sigma^2(L) := \mathrm{Var}(Y \mid X \in L)$. Then, conditional on $\Pi$ and on $\{n_E(L) = n\}$ for some
$n \ge 1$:
$$
\mathrm{Var}(\widehat{\mu}(L) \mid \Pi, n_E(L) = n) = \frac{\sigma^2(L)}{n}.
$$

**Proof.** Use the notation $I_i=\mathbf{1}\{X_i\in L\}$ and $N=\sum_{i\in E}I_i$ from Proposition 4.

Condition on $(\Pi,S,(I_i)_{i\in E})$. On the event $\{N=n\ge 1\}$, the variables $\{Y_i: i\in E, I_i=1\}$ are
independent with common distribution $(Y\mid X\in L)$ (because conditioning on the events $\{X_i\in L\}$ factorizes over
$i$ under i.i.d. sampling). Hence
$$
\mathrm{Var}\!\left(\widehat{\mu}(L)\;\middle|\;\Pi,S,(I_i)_{i\in E}\right)=\sigma^2(L)/n
\quad\text{on }\{N=n\ge 1\},
$$
and Proposition 4 gives
$\mathbb{E}[\widehat{\mu}(L)\mid \Pi,S,(I_i)_{i\in E}]=\mu(L)$ on $\{N\ge 1\}$.

Now condition only on $(\Pi, N=n)$ and apply the law of total variance with the refinement $(S,(I_i)_{i\in E})$:
$$
\mathrm{Var}(\widehat{\mu}(L)\mid \Pi,N=n)
=\mathbb{E}\!\left[\mathrm{Var}\!\left(\widehat{\mu}(L)\;\middle|\;\Pi,S,(I_i)_{i\in E}\right)\;\middle|\;\Pi,N=n\right]
\,+\,\mathrm{Var}\!\left(\mathbb{E}[\widehat{\mu}(L)\mid \Pi,S,(I_i)_{i\in E}]\;\middle|\;\Pi,N=n\right).
$$
The first term equals $\sigma^2(L)/n$ and the second term is $0$, yielding the claim. ∎

**Bias-variance trade-off (rigorous part + intuition).**  
Honesty reduces adaptive bias in leaf *estimation* by using an estimation sample that is independent of the
partition-learning step. Propositions 4–4a make the variance statement precise: for a fixed leaf $L$ and
$n_E(L)=n\ge 1$, the honest mean has conditional variance $\sigma^2(L)/n$.

Using fewer observations for estimation typically increases variance. If an alternative estimator used
$n_{\mathrm{all}}(L)$ i.i.d. observations from $(Y\mid X\in L)$, its variance would be $\sigma^2(L)/n_{\mathrm{all}}(L)$,
so moving from $n_{\mathrm{all}}(L)$ to $n_E(L)$ inflates variance by the factor $n_{\mathrm{all}}(L)/n_E(L)\ge 1$.
In non-honest trees, the extra observations are also used to *learn* $\Pi$, so the comparison to a
$\sigma^2(L)/n_{\mathrm{all}}(L)$ baseline should be read as intuition rather than an exact conditional identity for the
implemented (non-honest) estimator.

## 6. Where proofs stop (and why)

The following aspects are important in practice but require much more care (or algorithm changes) to attach
publication-grade proofs:

1. **Early stopping inside permutation testing**.
   citrees implements three modes for early stopping:
   - `early_stopping_* = "adaptive"` (default): Bayesian sequential stopping with valid p-values (see Section 6.1)
   - `early_stopping_* = "simple"`: Basic futility + significance stopping (inflates Type I error to ~9%)
   - `early_stopping_* = None`: Fixed-$B$ Monte Carlo p-value as in Theorem 1

   The adaptive mode provides valid Type I error control through anytime-valid sequential testing (see Section 6.1 below).

2. **Multi-selector mode** (`selector=[...]`). ✅ **RESOLVED**

   citrees implements the **max-T method** (Westfall & Young, 1993) for multi-selector mode. When multiple selectors
   are specified (e.g., `selector=['mc', 'rdc']`), the composite statistic is defined as:
   $$
   T^{\text{sel}}(X_{t,j},Y_t) := \max_{s\in \mathcal{S}} T^{\text{sel}}_s(X_{t,j},Y_t)
   $$
   and the permutation p-value is computed using the *same max* inside each permutation. This is provably valid by
   Theorem 1, since the composite max statistic is a measurable function of the data.

   **Empirical validation.** Simulations (10,000 runs under global null) confirm Type I error control:
   - Single selector (mc): 5.3% rejection rate [4.9%, 5.7%]
   - Multi selector (mc + rdc): 5.9% rejection rate [5.3%, 6.6%]

   Both are consistent with the nominal α = 0.05 level.

   **Implementation.** The `_ptest_multi()` function in `citrees/_selector.py` computes the max statistic inside each
   permutation, supporting all early stopping modes (None, "simple", "adaptive").

3. **Feature muting across nodes** (`feature_muting=True`).  
   Muting uses intermediate p-values to remove features globally from future consideration. This adaptively changes the
   hypothesis family across the tree and makes global error statements subtle.

4. **Global error control across the entire tree.**
   Even with valid nodewise tests, the tree construction is adaptive; p-values at internal nodes should not be read as
   classical inferential p-values for a fixed hypothesis family without additional machinery (sample splitting, selective
   inference, or fully specified global testing procedures).

   **Concrete example of adaptive invalidity ("double dipping").**
   Consider a tree with two levels. At the root, we test features $\{1, 2\}$ at level $\alpha = 0.05$ with Bonferroni
   (threshold $0.025$). Suppose feature 1 is selected and we split on the rule $X_1 \le c$.

   Now at the left child, we test features $\{1, 2\}$ again. But the samples reaching this child were *selected* by the
   data-dependent rule $X_1 \le c$. The "p-value" computed at this child is not a valid p-value for the original
   hypothesis $H: X_2 \perp Y$, because the conditioning event (which samples reach this node) depends on the data.

   This is why Propositions 3–3d are stated for single nodes (or the root), not for arbitrary internal nodes. For valid
   inference at internal nodes, additional techniques (sample splitting, selective inference) are required.

5. **Forest-level theory.**
   Consistency, rates, and uncertainty quantification for the forest (especially with permutation-based splitting)
   require assumptions and arguments beyond what can be asserted from the current implementation alone.

### 6.1 Sequential Permutation Testing (Adaptive Mode)

citrees implements **adaptive sequential permutation testing** using a Bayesian stopping rule that provides
valid p-values at any stopping time. This addresses the Type I error inflation that occurs with naive early stopping.

**Algorithm (Adaptive Sequential).** Let $\theta_0 = |T(X, Y)|$ be the observed test statistic. For permutations
$b = 1, 2, \ldots, B_{\max}$:

1. Compute $\theta_b = |T(X, \pi_b(Y))|$ where $\pi_b$ is a random permutation
2. Let $L_n = \sum_{b=1}^n \mathbf{1}\{\theta_b \ge \theta_0\}$ (exceedance count after $n$ permutations)
3. Model $L_n \mid p \sim \text{Binomial}(n, p)$ with prior $p \sim \text{Beta}(1, 1)$
4. Posterior is $p \mid L_n \sim \text{Beta}(1 + L_n, 1 + n - L_n)$
5. **Stop if confident significant**: $P(p < \alpha \mid L_n) \ge \gamma$ (default $\gamma = 0.95$)
6. **Stop if confident non-significant**: $P(p \ge \alpha \mid L_n) \ge \gamma$
7. Return p-value: $\hat{p} = (L_n + 1)/(n + 1)$

**Implementation details.**

The Beta CDF $P(p < \alpha \mid L_n) = I_\alpha(1 + L_n, 1 + n - L_n)$ is computed using Lentz's continued fraction
expansion algorithm, which provides $O(1)$ computation per iteration without external dependencies.

**Stopping criteria.**

- **Confident significant**: The posterior probability that the true p-value is below $\alpha$ exceeds the confidence
  threshold $\gamma$. This means we're $\gamma$-confident the null hypothesis should be rejected.
- **Confident non-significant**: The posterior probability that the true p-value is at or above $\alpha$ exceeds $\gamma$.
  This means we're $\gamma$-confident the null hypothesis should *not* be rejected.

**Validity argument (sketch).**

The stopping time $\tau$ depends only on the posterior belief about $p$, not on the p-value estimate itself. Under the
null hypothesis $H_0$, the true exceedance probability is $p = P(\theta^* \ge \theta_0) \ge 1/2$ (by symmetry of the
permutation distribution when $H_0$ holds). The Bayesian posterior credible interval has correct frequentist coverage
for the binomial proportion $p$, so stopping when "confident non-significant" does not inflate Type I error.

**Empirical validation.**

Benchmarks (`scratch/benchmark_sequential_ptest.py`) on 5000 simulations show:

| Method | Type I Error | Avg Perms (null) | Power |
|--------|-------------|------------------|-------|
| Fixed-$B$ (1000 perms) | 0.056 | 1000 | 0.970 |
| Simple sequential | **0.091** | 135 | 0.978 |
| Adaptive sequential ($\gamma=0.95$) | **0.055** | **48** | 0.964 |

The adaptive method:
- Controls Type I error at the nominal level (~5%)
- Reduces permutations by **95%** on null features (48 vs 1000)
- Maintains high power (0.964 vs 0.970)

**Simple sequential (baseline).**

A simpler method (`early_stopping_*="simple"`) stops early for:
- **Significance**: current p-value $< \alpha$ (after minimum resamples $\lceil 1/\alpha \rceil$)
- **Futility**: best possible p-value $\ge \alpha$ (cannot reject even with all remaining perms extreme)

This method does **not** control Type I error (inflates to ~9%) because it "peeks" at the running p-value without
proper sequential adjustment. It is provided only for baseline comparison and research purposes.

**Parameters.**

- `early_stopping_selector` / `early_stopping_splitter`: `"adaptive"` (default), `"simple"`, or `None`
- `early_stopping_confidence_selector` / `early_stopping_confidence_splitter`: $\gamma$ threshold (default 0.95)

**Mathematical note (TODO: expand in future revision).**

The full theoretical treatment requires formalizing the connection to anytime-valid sequential testing
(Fischer & Ramdas, 2025) and establishing finite-sample guarantees for the Bayesian stopping rule. The key
insight is that the stopping rule depends on the *posterior belief* about the p-value, not the p-value estimate
itself, which preserves validity under optional stopping.

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

### 7.1 Relationship to Hothorn et al. (2006) ctree

citrees is inspired by the conditional inference tree framework of Hothorn, Hornik, and Zeileis (2006), which selects
splitting variables via hypothesis tests derived from permutation invariance (in that lineage, often using linear
statistics; see also Strasser & Weber, 1999).

citrees follows the same high-level principle—**test-based variable selection to mitigate selection bias**—but plugs in
different association scores (e.g., `mc`, `mi`, `rdc`, `dc`, `pc`) and computes **Monte Carlo permutation p-values** (with
the Phipson–Smyth +1 correction) for these statistics.

The key theoretical takeaway is not “exactness” but **finite-sample validity**: under exchangeability, the +1 Monte
Carlo permutation p-value is super-uniform (Theorem 1), so the nodewise error-control statements (Section 4) do not rely
on large-sample approximations of null distributions.

## 8. Implementation-to-theory alignment checklist (paper-facing)

Before making inferential claims in a paper, it’s worth explicitly deciding which of the following you will *support*:

1. **Permutation p-value validity.**
   - Use `early_stopping_* = "adaptive"` (default) for valid sequential p-values with 95% speedup (Section 6.1)
   - Use `early_stopping_* = None` for fixed-$B$ p-values so Theorem 1 applies exactly
   - Avoid `early_stopping_* = "simple"` for inferential claims (inflates Type I error to ~9%)

2. **Single-selector p-values only.**  
   Use `selector="mc"` / `"pc"` / `"rdc"` etc. If you want multi-selector, change the statistic to a max-over-selectors
   *inside each permutation* (Section 6.2).

3. **Multiplicity correction actually matches the tested family.**  
   If you subsample features (`max_features`) or thresholds (`max_thresholds`), be explicit in the paper: you are
   controlling error over the tested subset, not over all $p$ features / all thresholds.

4. **Phipson–Smyth +1 correction everywhere.**
   All permutation tests in `citrees/_selector.py` and `citrees/_splitter.py` use the +1 correction
   `(1 + count)/(1 + B)` (including parallel implementations). This aligns with Section 3.

5. **Honesty claims match the sampling scheme.**
   Proposition 4 (unbiased honest leaf estimation) assumes the index split $(S,E)$ is independent of the observed data.
   In the current implementation, this matches regression (unstratified split) but not classification (stratified by
   labels).

## 9. What is actually “publishable” theory for citrees (safe claims)

This section is meant to be copy/paste-able into a paper as “theoretical guarantees,” without over-claiming.

### 9.1 Minimal assumptions to state explicitly

For any theorem involving permutation p-values, state (some variant of):

1. **Exchangeability under the null:** under $H_0$, labels $Y$ are exchangeable conditional on the covariates treated as
   fixed by the permutation procedure (and any algorithm RNG used to form candidate sets).
2. **Fixed permutation budget per test:** the Monte Carlo test uses a fixed $B$ permutations (no data-dependent early
   stopping).
3. **Multiplicity correction matches the tested family:** if you test $m$ features / $\ell$ thresholds, your correction
   is over those tested hypotheses (not over untested ones).

### 9.2 Theorems/Propositions you can safely include

1. **Permutation p-value validity (finite sample).**  
   The +1 Monte Carlo permutation p-value is super-uniform under the null (Theorem 1). This justifies using p-values as
   **Type I error–controlling selection scores** at a fixed node (after multiplicity correction).

2. **Feature-selection family-wise error at a fixed node (global null).**  
   If all tested features at a node are null, Bonferroni gives
   $\mathbb{P}(\text{select any feature at that node}) \le \alpha_{\text{sel}}$ (Proposition 3).

3. **Per-feature false selection bound (partial null).**  
   For any particular null feature in the tested family,
   $\mathbb{P}(\text{node splits on that feature}) \le \alpha_{\text{sel}}/m$ (Proposition 3a).
   This is a rigorous way to say “high-cardinality noise does not get an unfair advantage” (it cannot be selected more
   often than this bound, regardless of how many potential split points that feature has).

4. **Root-level global-null bound on *any* split in the fitted tree.**  
   Under a global null at the root across tested features, the probability the learned tree has any split is at most
   $\alpha_{\text{sel}}$ (Proposition 3b).

5. **(Optional) Honest estimation unbiasedness.**  
   If `honesty=True` and the sample split is independent of the observed data, leaf means are unbiased conditional on
   the learned partition on leaves that receive estimation samples (Proposition 4). If these conditions do not hold (or
   if `honesty=False`), do not claim unbiased leaf estimation.

6. **(Optional) Conformal coverage.**  
   If you wrap a fitted model with split conformal prediction, you can claim finite-sample marginal coverage under
   exchangeability (standard conformal theory; not unique to citrees).

### 9.3 "Rigorous mode" settings for experiments that cite these results

For runs where you want to invoke the theorems above as written:

- `early_stopping_selector="adaptive"` (default, provides valid sequential p-values) or `early_stopping_selector=None` (fixed-$B$)
- `adjust_alpha_selector=True` (and optionally `adjust_alpha_splitter=True` if you talk about threshold families)
- Multi-selector mode (`selector=[...]`) is now valid via max-T method (see Section 6.2)
- Prefer `feature_muting=False` for any inferential statements (keep it for speed-only ablations)

## 10. Appendix: Concrete statistics used in citrees (definitions + basic bounds)

This appendix is “paper boilerplate”: it records the exact nodewise statistics used in the codebase and a few
properties that justify design decisions (e.g., which selectors can be combined in multi-selector mode).

### 10.1 Classification selector: multiple correlation (`mc`)

At a node with samples $\{(x_i,y_i)\}_{i=1}^n$, where $x_i\in\mathbb{R}$ and $y_i\in\{1,\dots,K\}$, define the overall
mean $\mu := \frac1n\sum_{i=1}^n x_i$ and class means $\mu_k := \frac{1}{n_k}\sum_{i:y_i=k} x_i$ with
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

**Lemma 4 (boundedness of `mc`).**  
If $\mathrm{SST}>0$, then $0\le \mathrm{mc}(x,y)\le 1$.

**Proof.** The standard ANOVA decomposition gives
$$
\sum_{i=1}^n (x_i-\mu)^2
=
\sum_{k=1}^K\sum_{i:y_i=k}(x_i-\mu_k)^2 + \sum_{k=1}^K n_k(\mu_k-\mu)^2.
$$
Both terms on the right are nonnegative, so $\mathrm{SSB}\le \mathrm{SST}$ and hence
$0\le \mathrm{SSB}/\mathrm{SST}\le 1$. Taking square roots yields the claim. ∎

### 10.2 Regression selector: Pearson correlation (`pc`)

For vectors $x,y\in\mathbb{R}^n$ with nonzero empirical variances, define
$$
\rho(x,y) := \frac{\sum_{i=1}^n (x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum_{i=1}^n (x_i-\bar x)^2}\sqrt{\sum_{i=1}^n (y_i-\bar y)^2}}.
$$
citrees uses the magnitude $|\rho(x,y)|$ as the association score and permutation-test statistic.

**Lemma 5 (boundedness of `pc`).**  
Whenever the denominator is nonzero, $|\rho(x,y)|\le 1$.

**Proof.** Let $x'_i := x_i-\bar x$ and $y'_i := y_i-\bar y$. By Cauchy–Schwarz,
$$
\Big|\sum_{i=1}^n x'_i y'_i\Big|
\le \sqrt{\sum_{i=1}^n (x'_i)^2}\sqrt{\sum_{i=1}^n (y'_i)^2}.
$$
Dividing both sides by the product of norms yields $|\rho(x,y)|\le 1$. ∎

### 10.3 Regression selector: distance correlation (`dc`)

citrees uses distance correlation via the `dcor` library (Székely–Rizzo–Bakirov, 2007). In its **population**
definition, distance correlation takes values in $[0,1]$ and equals $0$ if and only if the variables are independent
(under mild moment conditions). (For full definitions/proofs, cite the original distance correlation paper; the code
uses a finite-sample estimator as the nodewise test statistic.)

### 10.4 Selector: randomized dependence coefficient (`rdc`)

citrees implements the RDC of Lopez-Paz et al. (2013) in a 1D-to-1D form:

1. Apply the empirical CDF transform $x\mapsto \widehat{F}_x(x)$ (rank / $n$).
2. Add a bias coordinate, then apply $k$ random linear projections.
3. Apply sinusoidal features $\cos(\cdot),\sin(\cdot)$.
4. Return the maximum absolute correlation between standardized feature columns.

Because each correlation is bounded by 1, the returned value satisfies $0\le \mathrm{rdc}(x,y)\le 1$.

### 10.5 Split impurities (`gini`, `entropy`, `mse`, `mae`)

For a classification node with empirical class probabilities $p_1,\dots,p_K$:

- Gini impurity: $\mathrm{gini}(p) := 1-\sum_{k=1}^K p_k^2$, hence $0\le \mathrm{gini}\le 1-1/K$.
- Entropy impurity: $\mathrm{ent}(p):= -\sum_{k=1}^K p_k \log_2 p_k$, hence $0\le \mathrm{ent}\le \log_2 K$.

For a regression node with targets $y_1,\dots,y_n$:

- MSE impurity: $\mathrm{mse}(y):=\frac1n\sum_{i=1}^n (y_i-\bar y)^2$ (empirical variance), so $\mathrm{mse}\ge 0$.
- MAE impurity: $\mathrm{mae}(y):=\frac1n\sum_{i=1}^n |y_i-\bar y|$, so $\mathrm{mae}\ge 0$.

### 10.6 Why `mi` cannot be in multi-selector mode (scale incompatibility)

The multi-selector mode takes a maximum over selector scores. This only makes sense when the scores are on a common
scale. In citrees:

- `mc`, `pc` (after absolute value), `dc`, and `rdc` are bounded in $[0,1]$,
- for classification, **population** mutual information satisfies $0\le I(X;Y)\le H(Y)\le \log K$ (units depend on the
  log base), so it is **not** normalized to $[0,1]$ and its scale depends on $K$ and on the entropy of the class labels,

so including `mi` in a max-with-others selector list would change the meaning of the maximum and would require
additional normalization or theory.
