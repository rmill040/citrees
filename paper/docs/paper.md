# citrees: Paper Draft (Theory-First)

This file is the **paper-facing** draft: it distills the clean, defensible mathematical statements from
`paper/docs/theory.md` and ties them to reproducible simulations under `paper/scripts/`.

For a claim-by-claim verification log (what is proved vs cited vs empirical), see `paper/docs/claim_audit.md`.
For a reproducible mapping from scripts → figures/tables → claims, see `paper/docs/figures_plan.md`.

**Scope.** The focus here is the validity of the permutation p-values used for *Stage A (feature screening)* and the
resulting finite-sample error-control statements (Bonferroni/root-level). Wherever a statement is only heuristic or
requires additional selective-inference machinery, it is labeled as such.

---

## 0. Manuscript skeleton (WIP)

This file is the evolving manuscript draft. The technical pieces live in:
- Appendix A: permutation-test theory + calibration checks
- Appendix B: optional feature-muting analysis (heuristic)
- Appendix C: open TODOs

### 0.1 Proposed paper structure

1. **Introduction**
   - Problem: feature selection is unstable/bias-prone; “high-cardinality noise” bias in greedy trees is a canonical
     failure mode.
   - Goal: rank features with controlled false discoveries + strong empirical performance, at scale.
   - Contributions (draft):
     1) permutation-test p-values with +1 correction as the inferential backbone for screening,
     2) scalable sequential stopping (default) with explicit scope of what is/ isn’t proven,
     3) a benchmark suite across synthetic ground-truth datasets + real tabular datasets, spanning 19 methods.

2. **Method**
   - Two-stage tree split: Stage A (feature) → Stage B (threshold).
   - Filter / permutation / embedding / wrapper methods evaluated (protocol is two-stage: rank → downstream eval).
   - Implementation details that matter for claims: Bonferroni in Stage A, fixed-$B$ vs adaptive stopping, tie handling.

3. **Theory (clean claims only)**
   - Theorem 1 (+1 Monte Carlo p-values are super-uniform under exchangeability).
   - Stage A Bonferroni and root-level “tree splits” guarantee.
   - Multi-selector max-statistic validity.
   - Sequential stopping: what is controlled (posterior-confidence “stop_sig”), what is not (returned $\hat p_\tau$ is not
     claimed to be an anytime-valid p-value).

4. **Experiments**
   - Benchmarks: synthetic classification (ground truth), real classification, real regression.
   - Metrics: precision/recall/F1@k (synthetic), downstream performance vs $k$, runtime.
   - Statistical comparisons: paired tests / critical difference diagrams (if we keep them).

5. **Discussion + limitations**
   - Exchangeability assumptions; adaptive internal nodes; Stage B post-selection; sequential p-values caveat.

### 0.2 Benchmark suite (what to show)

**Synthetic (classification, ground truth; 169 datasets across families)**  
Main-text candidates: bias (high-cardinality noise), nonlinear, correlated blocks, high-dimensional scaling, weak-signal.

**Real datasets**  
- Classification: focus on $p\gg n$ (e.g., arcene/dexter/dorothea/gisette/madelon + gene-expression sets), plus a small
  number of “standard” tabular datasets for sanity checks.
- Regression: include all available `paper/data/reg_*.parquet` datasets.

**Methods compared (keep full list in appendix; highlight a core subset in main text)**  
Filters (`mc/mi/rdc`, `pc/dc/rdc`, `mrmr`), permutation filters (`ptest_*`), embeddings (`cit/cif`, `rf/et`, `xgb/lgbm/cat`),
wrappers (`boruta`, `pi/cpi`, `shap`, `rfe`).

### 0.3 Figures/tables plan (draft)

Main text (candidate set):
- Fig: selection bias under null (`paper/results/figures/selection_bias_demo.png`)
- Fig: synthetic performance slices (`paper/results/figures/signal_strength.png`, `paper/results/figures/high_dimensional.png`,
  `paper/results/figures/correlated_features.png`, `paper/results/figures/redundant_features.png`)
- Fig: real-data downstream comparison (classification: `paper/results/figures/feature_selection_clf.png`; regression:
  `paper/results/figures/regression_comparison.png`)
- Fig: runtime (`paper/results/figures/timing_speedup.png` and/or `paper/results/figures/timing_bars.png`)

Appendix / supplement:
- Fixed-$B$ p-value calibration (`paper/results/figures/fixedB_pvalue_calibration.png`)
- Sequential stopping calibration (`paper/results/figures/sequential_stopping_calibration.png`)
- Any feature-muting analysis (Appendix B + `paper/results/figures/muting_*.png`) if we keep it at all.

## Abstract (TODO)

*TODO: 150–200 words. State the problem (feature selection bias/instability), our approach (permutation-test screening +
sequential stopping), and headline empirical results (synthetic ground truth + real datasets + runtime).*

## 1. Introduction (TODO)

*TODO:*
- *Motivation:* selection bias in greedy trees + feature selection in $p\gg n$ regimes.
- *Core idea:* statistical screening (Stage A) before split optimization; permutation tests for unbiasedness.
- *Contributions:* what is proven (fixed-$B$ Stage A/root control) vs what is empirical (adaptive early stopping).
- *Roadmap:* methods → theory → benchmarks.

## 2. Methods (skeleton)

### 2.1 citrees (as an embedding feature selector)

At each node $t$:
- **Stage A (feature screening):** test each feature $j\in F_t$ via a permutation test; split only if at least one
  feature is significant after multiplicity correction.
- **Stage B (threshold screening):** for the selected feature, test candidate thresholds and choose a split.

In the benchmark suite, tree/forest models (including citrees) act as **embedding feature selectors** by producing a
feature ranking (e.g., split counts / impurity-based importance), which is then evaluated downstream at top-$k$.

### 2.2 Methods compared (overview)

We compare four families of feature selectors (full lists and exact configs live in `paper/docs/README.md`):
- **Filters:** correlation / MI / RDC / mRMR
- **Permutation-test filters:** `ptest_*` methods that convert filter scores into permutation p-values
- **Embeddings:** tree ensembles (`cit/cif`, `rf/et`, `xgb/lgbm/cat`)
- **Wrappers:** Boruta, permutation importance (PI/CPI), SHAP, RFE

### 2.3 Benchmark protocol (two-stage architecture)

The experiments follow the repo’s two-stage pipeline (`paper/docs/README.md`):
1. **Stage 1 (selection):** each method outputs a feature ranking per dataset/seed/fold.
2. **Stage 2 (evaluation):** train downstream models on top-$k$ features for $k\in\{5,10,25,50,100,\text{all}\}$.

This decouples “how we rank features” from “how selected features perform in a standard model”.

## 3. Experiments (skeleton)

### 3.1 Benchmarks / datasets

**Synthetic classification (ground truth).** 169 synthetic datasets spanning:
standard, bias (high-cardinality noise), nonlinear, correlated blocks, redundant, correlated-noise confounders, toeplitz,
and weak-signal regimes. Ground truth is stored in parquet metadata and evaluated via precision/recall/F1@k.

**Real classification.** Use the `paper/data/clf_*.parquet` datasets, emphasizing $p\gg n$ feature selection (e.g.,
arcene/dexter/dorothea/gisette/madelon + gene-expression datasets), with a smaller number of “standard” tabular datasets
for sanity checks.

**Real regression.** Use all available regression datasets in `paper/data/reg_*.parquet`
(coepra1/2/3, comm_violence, community_crime, facebook, imports-85, residential).

### 3.2 Metrics

**Synthetic:** precision/recall/F1@k on informative features; optionally “informative+redundant” variants and confounder
selection rates (see `paper/scripts/analysis/synthetic_analysis.py`).

**Real:** downstream performance vs $k$:
- classification: AUC/accuracy/log loss (choose a primary metric for the main text)
- regression: $R^2$/MSE/MAE (choose a primary metric for the main text)

**Runtime:** wall-clock time for Stage 1 methods + speedups from early stopping / parallelism.

### 3.3 Reporting (how we summarize)

Decide what goes in the main text vs appendix:
- **Main text:** a small number of “story” figures that answer one question each, plus a single aggregated summary
  (across synthetic families and across real datasets).
- **Appendix:** full method list (all 19) + per-dataset tables/curves and any critical-difference diagrams.

### 3.4 Recommended results ordering (to manage the data volume)

1. **Selection-bias sanity check (the “why”)**
   - Fig: `paper/results/figures/selection_bias_demo.png`
   - Optional companion: `paper/results/figures/informative_ratio.png`

2. **Synthetic ground-truth performance (the “does it recover the right features?”)**
   - Start with easiest-to-interpret axes:
     - Fig: `paper/results/figures/signal_strength.png` (effect-size sweep)
     - Fig: `paper/results/figures/sample_size.png` (data efficiency)
   - Then show robustness regimes:
     - Fig: `paper/results/figures/high_dimensional.png` ($p$ scaling)
     - Fig: `paper/results/figures/correlated_features.png` (correlation blocks/confounding)
     - Fig: `paper/results/figures/redundant_features.png` (redundancy)
   - Keep imbalanced/multiclass/complexity as appendix unless they materially change conclusions:
     - `paper/results/figures/imbalanced.png`, `paper/results/figures/multiclass.png`,
       `paper/results/figures/complexity_vs_accuracy.png`

3. **Real-data downstream utility (the “does feature selection help?”)**
   - Fig (classification): `paper/results/figures/feature_selection_clf.png`
   - Fig (regression): `paper/results/figures/regression_comparison.png`

4. **Runtime / scalability (the “can we afford it?”)**
   - Fig: `paper/results/figures/timing_speedup.png` (headline speedup)
   - Optional: `paper/results/figures/timing_bars.png` (granular breakdown)

5. **Theory calibration checks (appendix)**
   - Fixed-$B$ p-value calibration: `paper/results/figures/fixedB_pvalue_calibration.png`
   - Sequential stopping calibration: `paper/results/figures/sequential_stopping_calibration.png`

## 4. Discussion and limitations (TODO)

*TODO:* Make explicit:
- exchangeability assumptions for permutation tests,
- internal-node adaptivity and Stage B post-selection caveats,
- sequential stopping: decision-level control vs anytime-valid p-values.

## 5. Conclusion (TODO)

---

## Appendix A. Theory and calibration

### A.1 Setup and notation

Let $(X_i, Y_i)_{i=1}^n$ be the training data and consider a fixed node $t$ with index set $I_t$ (size $n_t$). Write
$X_t := (X_i)_{i\in I_t}$ and $Y_t := (Y_i)_{i\in I_t}$.

At node $t$, citrees performs two tests:

- **Stage A (feature screening):** For each candidate feature $j\in F_t$ (with $m_t := |F_t|$), test the null
  $H^{\text{sel}}_{t,j}$: “$X_{t,j}$ is independent of $Y_t$ (in the sense required by the permutation scheme)”.
- **Stage B (threshold screening):** After a feature $j_t^\star$ is selected, test candidate thresholds
  $c\in C_{t,j_t^\star}$ (with $\ell_{t,j}:=|C_{t,j}|$) using a split-quality statistic.

citrees uses Monte Carlo permutation tests: rather than enumerating all permutations, it draws $B$ i.i.d. random
permutations and estimates the permutation tail probability.

**Paper-facing inferential target.** The cleanest theoretical guarantees in this framework are:

1. **Fixed-node statements** (treating $t$ as fixed, not adaptively selected), especially at the **root**.
2. **Stage A** error-control statements; Stage A is the “gatekeeper” for whether the tree splits at all.

---

### A.2 Monte Carlo permutation p-values (+1 correction)

Fix a node $t$, feature $j$, and a test statistic $T(\cdot,\cdot)$ with a tail convention where “more extreme” means
larger (right tail). Let $\pi_1,\dots,\pi_B$ be i.i.d. uniform random permutations of $\{1,\dots,n_t\}$, independent of
the data, and set $Y_t^{(b)} := \pi_b(Y_t)$.

Define
$$
T_0 := T(X_{t,j}, Y_t),\qquad T_b := T(X_{t,j}, Y_t^{(b)})\quad (b=1,\dots,B),
$$
and the **+1 Monte Carlo p-value**
$$
p := \frac{1 + \sum_{b=1}^B \mathbf{1}\{T_b \ge T_0\}}{B+1}.
$$
This is the correction recommended by Phipson & Smyth (2010) (“Permutation P-values Should Never Be Zero”).

#### Theorem 1 (finite-sample validity under exchangeability)

Assume that, under the null hypothesis being tested, $(T_0,T_1,\dots,T_B)$ is exchangeable. Then for all
$\alpha\in[0,1]$,
$$
\mathbb{P}(p \le \alpha)\le \alpha.
$$
If ties occur, using $\mathbf{1}\{T_b \ge T_0\}$ “counts ties against the null” and remains conservative. If you want an
exactly uniform p-value in the presence of ties, you can use randomized tie-breaking (e.g., lexicographic tie-breaking
with i.i.d. $U_b\sim\mathrm{Unif}(0,1)$).

**How exchangeability arises in permutation tests.**
Under a permutation-test null (e.g., $X_{t,j}\perp Y_t$ under i.i.d. sampling), the conditional distribution of $Y_t$
given the covariates treated as fixed by the test is invariant to permutations. Consequently,
$T(X_{t,j},Y_t)\stackrel{d}{=}T(X_{t,j},\pi(Y_t))$ for uniform $\pi$, and the Monte Carlo vector
$(T_0,T_1,\dots,T_B)$ is exchangeable (conditional on $X_t$ and any label-independent algorithmic randomness used to form
the tested family).

**What is (and is not) implied.**

- This theorem gives **finite-sample super-uniformity** with **no independence assumptions** among tests. It is the core
  mathematical justification for permutation p-values inside citrees.
- This theorem does **not** automatically justify interpreting p-values computed at **adaptively selected internal
  nodes** as classical p-values for fixed hypothesis families; tree growth creates data-dependent conditioning events.

---

#### A.2.1 Monte Carlo resolution (why we require a minimum number of resamples)

Let
$$
K := \sum_{b=1}^B \mathbf{1}\{T_b \ge T_0\}
$$
so that $p=(K+1)/(B+1)$. Then $p$ takes values in the grid $\{1/(B+1),2/(B+1),\dots,1\}$, and the minimum possible
reported p-value is $p_{\min}=1/(B+1)$.

Therefore, if a procedure compares $p$ to a threshold $\alpha'$ (e.g., $\alpha'=\alpha$ for a single test or
$\alpha'=\alpha/m$ for Bonferroni with $m$ tests), then **rejection is only possible if** $p_{\min}<\alpha'$,
equivalently $B+1>1/\alpha'$. A simple sufficient rule is
$$
B \;\ge\; \left\lceil\frac{1}{\alpha'}\right\rceil.
$$
This is why citrees enforces a minimum number of resamples (and, under Bonferroni, scales resampling so that comparing to
the adjusted level remains meaningful).

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

citrees allows combining multiple selector statistics (e.g., `selector=['pc','dc','rdc']` for regression) by defining a
single combined test statistic as the maximum across selectors.

Formally, let $T^{(1)},\dots,T^{(K)}$ be $K$ selector statistics computed on the same $(X_{t,j},Y_t)$ with a shared tail
convention, and define the combined statistic
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

Under the same exchangeability conditions as Theorem 1, $(M_0,\dots,M_B)$ is exchangeable (as a measurable function of
the exchangeable vector of selector statistics), hence $p_{\max}$ is a valid super-uniform p-value. This is a standard
“max-statistic” construction (often called a max-T method; see Westfall & Young (1993) for broader resampling-based
multiple testing).

**Note (scales).** In classification, citrees does not allow mutual information (`mi`) to be combined with bounded
statistics like `mc` and `rdc` in multi-selector mode because of scale mismatch. This is a power/robustness choice, not
a validity requirement for the max-statistic construction itself.

---

### A.3 Bonferroni control for Stage A (nodewise + root-level)

#### Lemma 2 (Bonferroni with super-uniform p-values)

Let $p_1,\dots,p_m$ be p-values such that for each true null $H_j$ and all $u\in[0,1]$,
$\mathbb{P}(p_j \le u)\le u$. Under the global null (all $H_j$ true),
$$
\mathbb{P}\!\left(\min_{1\le j\le m} p_j \le \frac{\alpha}{m}\right)\le \alpha.
$$
No independence assumptions are required (union bound).

#### Proposition 3 (Stage A, fixed node, global null over tested features)

At a fixed node $t$, suppose $H^{\text{sel}}_{t,j}$ holds for all $j\in F_t$ and let $p_{t,j}$ be valid permutation
p-values (Theorem 1). If Stage A uses Bonferroni (threshold $\alpha_{\text{sel}}/m_t$), then
$$
\mathbb{P}\!\left(\exists j\in F_t:\; p_{t,j}\le \alpha_{\text{sel}}/m_t\right)\le \alpha_{\text{sel}}.
$$

#### Proposition 3a (per-feature false selection bound; no global-null needed)

At a fixed node $t$, fix a particular feature $j\in F_t$ whose null $H^{\text{sel}}_{t,j}$ is true, so $p_{t,j}$ is
super-uniform (Theorem 1). If Stage A uses Bonferroni, then the event “the node splits on feature $j$” implies
$p_{t,j}\le \alpha_{\text{sel}}/m_t$, hence
$$
\mathbb{P}(\text{node }t\text{ splits on feature }j)\le \alpha_{\text{sel}}/m_t.
$$
This bound does not depend on the feature’s number of unique values, which is one precise sense in which Stage A avoids
the classic CART-style “high-cardinality selection bias”.

#### A.3.1 Empirical check: selection bias vs CART

To illustrate the difference between optimization-based variable selection (CART) and Stage A’s testing-based selection,
we include a small null simulation:

- CART (`sklearn.tree.DecisionTreeClassifier`, `max_depth=1`) selects the root feature by maximizing impurity decrease
  over many possible thresholds, which is known to favor high-cardinality features under the global null.
- citrees Stage A selects the root feature by comparing permutation p-values against $\alpha/m$.

Reproduce:

```bash
uv sync --group paper
UV_CACHE_DIR=$PWD/.uv-cache uv run python paper/scripts/theory/generate_selection_bias_demo.py
```

Outputs:
- `paper/results/cache/selection_bias_demo_data.parquet`
- `paper/results/figures/selection_bias_demo.png`

#### Proposition 4 (safe, global statement: “any split implies root rejection”)

Tree adaptivity complicates internal-node inference, but one global statement remains clean:

> The fitted tree can only have any split if the **root** passes Stage A.

Consequently, if the global null holds for all tested features at the root (and Stage A uses Bonferroni with fixed $B$),
then
$$
\mathbb{P}(\text{the fitted tree has at least one internal split}) \le \alpha_{\text{sel}}.
$$

---

### A.4 Post-selection caveats (what we do *not* claim)

1. **Stage B is post-selection.** Stage B is performed *after selecting* $j_t^\star$ using the same labels $Y_t$.
   Without sample splitting or selective-inference adjustments, Stage B p-values should be treated as **algorithmic
   stopping statistics**, not classical post-selection p-values (e.g., Berk et al., 2013; Lee et al., 2016; Fithian
   et al., 2014).
2. **Internal nodes are adaptive.** In an adaptively-grown tree, a node $t$ corresponds to a random index set $I_t$
   determined by earlier splits that depend on the labels; conditioning on “these samples reach node $t$” can break
   exchangeability under nulls. This is why the most defensible inferential statements are either (i) for a fixed node,
   or (ii) root-level (see also Dwork et al., 2015; Leeb & Pötscher, 2015 for general cautions about adaptive
   inference).

---

### A.5 Adaptive sequential permutation testing (early stopping)

citrees optionally uses **adaptive early stopping** as a computational shortcut for Monte Carlo permutation tests:
stop when the evidence is overwhelming for either “significant” or “not significant”.

#### A.5.1 What citrees returns in adaptive mode

In adaptive mode, citrees returns the standard +1 Monte Carlo estimate
$$
\widehat{p}_n := \frac{L_n+1}{n+1}
$$
evaluated at a **data-dependent stopping time** $n=\tau$. This returned $\widehat{p}$ should **not** be presented as a
classical super-uniform p-value “under optional stopping” unless an explicit anytime-valid construction is used.
(Standard fixed-$n$ p-values are not designed to be anytime-valid under repeated looks / data-dependent stopping; this is
precisely why sequential Monte Carlo testing methods exist (Appendix A.5.3).)

If you need a paper-facing p-value guarantee, use **fixed-$B$** permutation tests
(`early_stopping_*=None`), so Theorem 1 applies directly.

#### A.5.2 A clean, provable statement: what the stopping *criterion* controls

Let $I_1,I_2,\dots$ be the exceedance indicators, where $I_b = \mathbf{1}\{T_b \ge T_0\}$, and
$L_n := \sum_{b=1}^n I_b$. Define the “posterior-confidence” score
$$
S_n := \mathbb{P}(p^\star < \alpha \mid L_n, n) = I_\alpha(1+L_n,\;1+n-L_n),
$$
where $p^\star$ is the (idealized) permutation tail probability and $I_\alpha(\cdot,\cdot)$ is the regularized
incomplete beta function.

Assume the continuous-null idealization where, under $H_0$, $p^\star\sim\mathrm{Unif}(0,1)$ and conditional on $p^\star$,
the indicators $I_b$ are i.i.d. $\mathrm{Bernoulli}(p^\star)$ (this is the standard rank/PIT argument; randomized
tie-breaking can be used to justify the continuous idealization). With ties/discrete permutation distributions,
$p^\star$ is not exactly uniform; counting ties “against the null” leads to conservative behavior (see Fischer & Ramdas,
2025, Remark 1).

**Connection to anytime-valid e-values.** Define
$$
W_n := \frac{S_n}{\alpha}.
$$
Using the binomial–beta identity, this is equivalently
$$
W_n = \frac{1 - F_{\mathrm{Binom}}(L_n;\;n+1,\alpha)}{\alpha}.
$$
This is exactly the *binomial-mixture wealth process* from Fischer & Ramdas (2025, Proposition 5.2), and $(W_n)$ is a test
martingale (an e-process) under the continuous-null idealization (and remains valid/conservative with ties when ties
are treated as losses; Fischer & Ramdas, 2025, Remark 1). In particular, thresholding $W_n$ controls Type I error under
optional stopping via Ville’s inequality.

**Note.** Fischer & Ramdas (2025, Proposition 5.2) also observe that with the special choice $c=\alpha$,
$W_n^{u_\alpha}(L_n) < 1/\alpha$ for all $n$, so the $1/\alpha$ boundary is not reachable in finite time; they recommend
choosing $c<\alpha$ for practical level-$\alpha$ sequential tests. citrees instead uses the lower boundary
$\gamma/\alpha$ for the significance-stop event.

Let $\tau$ be any (possibly data-dependent) stopping time and consider the **“significance-stop” event**
$$
\{\text{stop\_sig}\} := \{S_\tau \ge \gamma\}\qquad (\gamma\in(0,1)).
$$

**Theorem 2 (Bayesian calibration at stopping times).**
Under the conditions above,
$$
\mathbb{E}[S_\tau] = \alpha.
$$

**Corollary 2 (Type I error bound for triggering the significance stop).**
By Markov’s inequality,
$$
\mathbb{P}(\text{stop\_sig}) = \mathbb{P}(S_\tau \ge \gamma)\le \frac{\alpha}{\gamma}.
$$
For the default $(\alpha,\gamma)=(0.05,0.95)$ this yields the (loose) bound $\mathbb{P}(\text{stop\_sig})\le 0.0526$.

**Proof sketch.**
Since $S_\tau=\mathbb{E}[\mathbf{1}\{p^\star<\alpha\}\mid L_\tau,\tau]$, the tower property gives
$\mathbb{E}[S_\tau]=\mathbb{P}(p^\star<\alpha)=\alpha$. Then apply Markov to $S_\tau\in[0,1]$.

**Important:** citrees does *not* reject based on $S_\tau$. It uses $S_n$ only to decide when to stop sampling
permutations, and then returns $\widehat{p}_\tau=(L_\tau+1)/(\tau+1)$ to be compared against $\alpha$ downstream. The
bound above therefore does **not** imply that the returned $\widehat{p}_\tau$ is a classical p-value “under optional
stopping”; it only bounds how often the algorithm would stop due to “confident significance” under the continuous-null
idealization.

#### A.5.3 Relationship to the sequential Monte Carlo testing literature

Theorem 2 above is a **decision-level** guarantee for the posterior-confidence stopping rule; it is not an
“anytime-valid p-value” theorem. For sequential procedures designed explicitly for anytime validity / bounded resampling
risk, see:

- Besag & Clifford (1991), “Sequential Monte Carlo p-values” (Biometrika).
- Gandy (2009), “Sequential Implementation of Monte Carlo Tests with Uniformly Bounded Resampling Risk” (JASA).
- Fischer & Ramdas (2025), “Sequential Monte-Carlo testing by betting” (JRSS-B).

---

### A.6 Reproducibility: simulations that backstop Appendix A.5

The repository contains small scripts to empirically assess the behavior of the adaptive stopping rule under $H_0$
(Type I error and stopping-time distribution) and to compare against anytime-valid alternatives:

- `paper/scripts/theory/generate_sequential_stopping_calibration.py` (writes a parquet cache into `paper/results/cache/` and
  a figure into `paper/results/figures/`)
- `paper/scripts/theory/sequential_stopping_analysis.py`
- `paper/scripts/theory/sequential_stopping_comparison.py`

To run (after environment setup):

```bash
uv sync --group paper
UV_CACHE_DIR=$PWD/.uv-cache uv run python paper/scripts/theory/generate_sequential_stopping_calibration.py
UV_CACHE_DIR=$PWD/.uv-cache uv run python paper/scripts/theory/sequential_stopping_analysis.py
UV_CACHE_DIR=$PWD/.uv-cache uv run python paper/scripts/theory/sequential_stopping_comparison.py
```

---

## Appendix B. Feature muting analysis (optional)

This appendix concerns **feature muting**, which is a computational heuristic rather than a core inferential component of
citrees.

citrees implements **feature muting** as a computational heuristic: after testing a feature at a node, if the p-value
is *extremely* non-significant (by default, in the upper tail: `p >= max(alpha, 1 - alpha)`), the feature is removed
from the candidate set for **descendants of that node** (subtree-local propagation; siblings are isolated).

This section explains why **aggressive global screening** (e.g., a one-shot root decision to never consider a feature
again) can be dangerous under conditional/interaction effects, and motivates keeping muting conservative and treating
it as a speed-only heuristic (compare `feature_muting=True` vs `False`).

### B.1 The gated effect model

Consider a simple data-generating process where feature $X_1$ has predictive power only within a subset of the data:

$$
X_0, X_1 \sim \mathcal{N}(0, 1) \text{ independent}, \quad
Z = \mathbf{1}\{X_0 > c\} \text{ with } p := P(Z=1), \quad
Y = \begin{cases} \mathbf{1}\{X_1 > 0\} & \text{if } Z=1 \\ \text{Bernoulli}(1/2) & \text{if } Z=0 \end{cases}
$$

Feature $X_1$ is informative about $Y$ **only in the gated subset** where $Z=1$. At the root (full sample), the signal
from $X_1$ is diluted by the $(1-p)$ fraction of noise observations.

### B.2 Population correlations

**Lemma (Population Moments).**
Let $\rho_{\text{root}} := \text{Corr}(X_1, Y)$ and $\rho_{\text{gate}} := \text{Corr}(X_1, Y \mid Z=1)$. Then:

$$
\rho_{\text{root}} = \frac{2p}{\sqrt{2\pi}} \approx 0.798\,p, \qquad
\rho_{\text{gate}} = \frac{2}{\sqrt{2\pi}} \approx 0.798.
$$

**Key insight:** The root correlation scales linearly with $p$, while the gate correlation is constant (~0.798). For
small $p$, the root signal becomes undetectable while the gate signal remains strong.

### B.3 Exact power functions

Using the Fisher z-transformation with bias correction (Hotelling, 1953), we compute exact power for the two-sided
correlation test:

**Critical value:** For sample size $n$ and significance level $\alpha$:
$$
r_\alpha(n) = \frac{t_{n-2,\,1-\alpha/2}}{\sqrt{t_{n-2,\,1-\alpha/2}^2 + (n-2)}}
$$

**Power function:** For population correlation $\rho$:
$$
\pi(\rho; n, \alpha) = 1 - \Phi\!\left(\frac{z_\alpha - \zeta}{\sigma_z}\right) + \Phi\!\left(\frac{-z_\alpha - \zeta}{\sigma_z}\right)
$$
where $z_\alpha = \tanh^{-1}(r_\alpha)$, $\zeta = \tanh^{-1}(\rho) + \frac{\rho}{2(n-1)}$, and $\sigma_z = 1/\sqrt{n-3}$.

**Application to gated model:**
- Root power: $\pi_{\text{root}}(p; n, \alpha) = \pi(0.798p;\, n,\, \alpha)$
- Gate power: $\pi_{\text{gate}}(p; n, \alpha) = \pi(0.798;\, \lfloor np \rfloor,\, \alpha)$

### B.4 Gap region: where local muting succeeds but global fails

Define the **gap region** as the set of gate probabilities where global muting fails (root power $\le \beta_L$) but
local muting succeeds (gate power $\ge \beta_H$):

$$
\mathcal{G}(n, \alpha, \beta_L, \beta_H) = \{p : \pi_{\text{root}}(p) \le \beta_L \text{ and } \pi_{\text{gate}}(p) \ge \beta_H\}
$$

**Table: Gap Region Boundaries** ($\alpha=0.05$, $\beta_L=0.2$, $\beta_H=0.8$)

| n | $p_{\min}$ | $p_{\max}$ | Ratio | Width |
|---:|------:|------:|------:|------:|
| 100 | 0.100 | 0.141 | 1.4 | 0.041 |
| 500 | 0.020 | 0.063 | 3.1 | 0.043 |
| 1,000 | 0.010 | 0.044 | 4.4 | 0.034 |
| 2,000 | 0.005 | 0.031 | 6.2 | 0.026 |
| 5,000 | 0.002 | 0.020 | 9.9 | 0.018 |

The gap exists for $n \ge 50$ and widens (in ratio) as $n$ increases.

### B.5 Tree depth propagation

As the tree grows deeper, sample sizes at internal nodes decrease (roughly halving at each level for balanced splits),
which affects detection power.

**Critical depth:** The maximum depth at which the gated signal remains detectable is approximately:
$$
d_{\max} \approx 2\log_2\!\left(\frac{0.798\sqrt{np}}{z_{1-\alpha/2}}\right)
$$

For $n=2000$, $p=0.05$: $d_{\max} \approx 4.1$, meaning the signal is detectable at depths 0–4 but not beyond.

### B.6 Practical implications

1. **When to use local muting:** If your application involves features with **conditional effects** (informative only
   in subsets of the data), be cautious with any muting heuristic. In such settings, consider disabling muting
   (`feature_muting=False`) and comparing results, since conditionally-informative features can look null on larger
   mixed samples.

2. **Diagnostic:** Compare trees with `feature_muting=True` vs `feature_muting=False`. Large discrepancies in which
   features appear may indicate conditional effects being missed by muting.

3. **Sample size requirements:** Even with local muting, sufficient samples are needed in the gated branch. If $np < 20$
   (expected gate size), detection power may be inadequate regardless of muting scope.

### B.7 Reproducibility

The power calculations above are implemented in `paper/scripts/theory/theoretical_predictions.py`. To reproduce the
gap region table:

```bash
UV_CACHE_DIR=$PWD/.uv-cache uv run python paper/scripts/theory/theoretical_predictions.py
```

---

## Appendix C. TODOs

1. Decide whether the main paper should:
   - treat adaptive early stopping as an engineering heuristic with empirical calibration, or
   - adopt an anytime-valid sequential method (literature above) for a fully frequentist sequential guarantee.
2. (Optional) Standardize citations to use `paper/references.bib` keys consistently across `paper/*.md`.
