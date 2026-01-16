# citrees: Paper Draft (Theory-First)

This file is the **paper-facing** draft: it distills the clean, defensible mathematical statements from
`paper/theory.md` and ties them to reproducible simulations under `paper/scripts/`.

**Scope.** The focus here is the validity of the permutation p-values used for *Stage A (feature screening)* and the
resulting finite-sample error-control statements (Bonferroni/root-level). Wherever a statement is only heuristic or
requires additional selective-inference machinery, it is labeled as such.

---

## 1. Setup and notation

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

## 2. Monte Carlo permutation p-values (+1 correction)

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

### Theorem 1 (finite-sample validity under exchangeability)

Assume that, under the null hypothesis being tested, $(T_0,T_1,\dots,T_B)$ is exchangeable. Then for all
$\alpha\in[0,1]$,
$$
\mathbb{P}(p \le \alpha)\le \alpha.
$$
If ties occur, using $\mathbf{1}\{T_b \ge T_0\}$ “counts ties against the null” and remains conservative. If you want an
exactly uniform p-value in the presence of ties, you can use randomized tie-breaking (e.g., lexicographic tie-breaking
with i.i.d. $U_b\sim\mathrm{Unif}(0,1)$).

**What is (and is not) implied.**

- This theorem gives **finite-sample super-uniformity** with **no independence assumptions** among tests. It is the core
  mathematical justification for permutation p-values inside citrees.
- This theorem does **not** automatically justify interpreting p-values computed at **adaptively selected internal
  nodes** as classical p-values for fixed hypothesis families; tree growth creates data-dependent conditioning events.

---

## 3. Bonferroni control for Stage A (nodewise + root-level)

### Lemma 2 (Bonferroni with super-uniform p-values)

Let $p_1,\dots,p_m$ be p-values such that for each true null $H_j$ and all $u\in[0,1]$,
$\mathbb{P}(p_j \le u)\le u$. Under the global null (all $H_j$ true),
$$
\mathbb{P}\!\left(\min_{1\le j\le m} p_j \le \frac{\alpha}{m}\right)\le \alpha.
$$
No independence assumptions are required (union bound).

### Proposition 3 (Stage A, fixed node, global null over tested features)

At a fixed node $t$, suppose $H^{\text{sel}}_{t,j}$ holds for all $j\in F_t$ and let $p_{t,j}$ be valid permutation
p-values (Theorem 1). If Stage A uses Bonferroni (threshold $\alpha_{\text{sel}}/m_t$), then
$$
\mathbb{P}\!\left(\exists j\in F_t:\; p_{t,j}\le \alpha_{\text{sel}}/m_t\right)\le \alpha_{\text{sel}}.
$$

### Proposition 4 (safe, global statement: “any split implies root rejection”)

Tree adaptivity complicates internal-node inference, but one global statement remains clean:

> The fitted tree can only have any split if the **root** passes Stage A.

Consequently, if the global null holds for all tested features at the root (and Stage A uses Bonferroni with fixed $B$),
then
$$
\mathbb{P}(\text{the fitted tree has at least one internal split}) \le \alpha_{\text{sel}}.
$$

---

## 4. Post-selection caveats (what we do *not* claim)

1. **Stage B is post-selection.** Stage B is performed *after selecting* $j_t^\star$ using the same labels $Y_t$.
   Without sample splitting or selective-inference adjustments, Stage B p-values should be treated as **algorithmic
   stopping statistics**, not classical post-selection p-values.
2. **Internal nodes are adaptive.** In an adaptively-grown tree, a node $t$ corresponds to a random index set $I_t$
   determined by earlier splits that depend on the labels; conditioning on “these samples reach node $t$” can break
   exchangeability under nulls. This is why the most defensible inferential statements are either (i) for a fixed node,
   or (ii) root-level.

---

## 5. Adaptive sequential permutation testing (early stopping)

citrees optionally uses **adaptive early stopping** as a computational shortcut for Monte Carlo permutation tests:
stop when the evidence is overwhelming for either “significant” or “not significant”.

### 5.1 What citrees returns in adaptive mode

In adaptive mode, citrees returns the standard +1 Monte Carlo estimate
$$
\widehat{p}_n := \frac{L_n+1}{n+1}
$$
evaluated at a **data-dependent stopping time** $n=\tau$. This returned $\widehat{p}$ should **not** be presented as a
classical super-uniform p-value “under optional stopping” unless an explicit anytime-valid construction is used.
(Standard fixed-$n$ p-values generally do not retain super-uniformity after data-dependent stopping; see, e.g., Rouder
(2014) for a concise discussion, and Fischer & Ramdas (2025) for modern anytime-valid alternatives.)

If you need a paper-facing p-value guarantee, use **fixed-$B$** permutation tests
(`early_stopping_*=None`), so Theorem 1 applies directly.

### 5.2 A clean, provable statement: what the stopping *criterion* controls

Let $I_1,I_2,\dots$ be the exceedance indicators, where $I_b = \mathbf{1}\{T_b \ge T_0\}$, and
$L_n := \sum_{b=1}^n I_b$. Define the “posterior-confidence” score
$$
S_n := \mathbb{P}(p^\star < \alpha \mid L_n, n) = I_\alpha(1+L_n,\;1+n-L_n),
$$
where $p^\star$ is the (idealized) permutation tail probability and $I_\alpha(\cdot,\cdot)$ is the regularized
incomplete beta function.

Assume the continuous-null idealization where, under $H_0$, $p^\star\sim\mathrm{Unif}(0,1)$ and conditional on $p^\star$,
the indicators $I_b$ are i.i.d. $\mathrm{Bernoulli}(p^\star)$ (this is the standard rank/PIT argument; randomized
tie-breaking can be used to justify the continuous idealization).

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

### 5.3 Relationship to the sequential Monte Carlo testing literature

Theorem 2 above is a **decision-level** guarantee for the posterior-confidence stopping rule; it is not an
“anytime-valid p-value” theorem. For sequential procedures designed explicitly for anytime validity / bounded resampling
risk, see:

- Besag & Clifford (1991), “Sequential Monte Carlo p-values” (Biometrika).
- Gandy (2009), “Sequential Implementation of Monte Carlo Tests with Uniformly Bounded Resampling Risk” (JASA).
- Fischer & Ramdas (2025), “Anytime-valid sequential Monte Carlo testing” (JRSS-B).

---

## 6. Reproducibility: simulations that backstop Section 5

The repository contains small scripts to empirically assess the behavior of the adaptive stopping rule under $H_0$
(Type I error and stopping-time distribution) and to compare against anytime-valid alternatives:

- `paper/scripts/theory/generate_sequential_stopping_calibration.py` (writes a table/figure into `paper/results/figures/`)
- `paper/scripts/theory/sequential_stopping_analysis.py`
- `paper/scripts/theory/sequential_stopping_comparison.py`

To run (after environment setup):

```bash
uv sync
uv run python paper/scripts/theory/generate_sequential_stopping_calibration.py
uv run python paper/scripts/theory/sequential_stopping_analysis.py
uv run python paper/scripts/theory/sequential_stopping_comparison.py
```

---

## 7. TODOs to finish this paper draft

1. Decide whether the main paper should:
   - treat adaptive early stopping as an engineering heuristic with empirical calibration, or
   - adopt an anytime-valid sequential method (literature above) for a fully frequentist sequential guarantee.
2. Add BibTeX entries for Besag–Clifford (1991), Gandy (2009), Fischer–Ramdas (2025), and Westfall–Young (1993) to
   `paper/references.bib` (optional if citations remain inline).
