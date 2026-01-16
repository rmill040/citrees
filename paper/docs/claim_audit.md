# Claim Audit: p-values and sequential stopping (WIP)

This file is a running audit log to keep `paper/docs/paper.md` and `paper/docs/theory.md` mathematically honest.

**Goal:** Every claim that matters for the paper should be either:
- **PROVED** in our manuscript text, or
- **CITED** to an external reference (`paper/references.bib`), or
- **EMPIRICAL** with a reproducible script + committed output under `paper/results/`.

If a claim is none of the above, it should be marked **TODO** (or moved to an explicitly non-paper “notes / archive”
section).

---

## A. Fixed-$B$ permutation p-values

| ID | Claim (short) | Where used | Verification | Status |
|---:|---|---|---|---|
| A1 | +1 Monte Carlo permutation p-value is super-uniform under exchangeability | `paper/docs/theory.md` §3.4; `paper/docs/paper.md` Appendix A.2 | Proof by exchangeable-rank argument; +1 convention matches Phipson & Smyth (2010) | PROVED + CITED |
| A2 | Conditional-on-$(X,U)$ super-uniformity (candidate-set randomness allowed if label-independent) | `paper/docs/theory.md` §3.5–4.3; `paper/docs/paper.md` Appendix A.2 | Conditioning argument + A1 | PROVED |
| A3 | Bonferroni FWER control with super-uniform p-values (no independence required) | `paper/docs/theory.md` §4.1–4.2; `paper/docs/paper.md` Appendix A.3 | Union bound | PROVED |
| A4 | “Any split implies root Stage A rejection” ⇒ under global null at root, $\Pr(\text{tree splits}) \le \alpha_{\text{sel}}$ | `paper/docs/theory.md` §4.4; `paper/docs/paper.md` Appendix A.3 | Set inclusion + A3 at root | PROVED |
| A5 | Multi-selector “max statistic inside each permutation” yields valid p-values | `paper/docs/theory.md` §6(2); `paper/docs/paper.md` Appendix A.2.3 | Composite statistic preserves exchangeability; cite max-T framing (Westfall & Young, 1993) | PROVED + CITED |

**Empirical backstops (optional):**
- E1: `paper/scripts/theory/generate_fixedB_pvalue_calibration.py` → `paper/results/figures/fixedB_pvalue_calibration_{data,plot}`.

---

## B. Stage B and adaptive tree growth (what we do *not* claim)

| ID | Claim (short) | Where used | Verification | Status |
|---:|---|---|---|---|
| B1 | Stage B p-values are post-selection and should not be presented as classical p-values without selective inference / splitting | `paper/docs/theory.md` §2.3; `paper/docs/paper.md` Appendix A.4 | Cite post-selection / selective-inference warnings (e.g., Berk et al., 2013; Lee et al., 2016; Fithian et al., 2014) | CITED |
| B2 | Internal-node tests in an adaptively grown tree are not classical p-values for a fixed family (conditioning on data-dependent node membership breaks exchangeability) | `paper/docs/theory.md` §3.5 remark + §6; `paper/docs/paper.md` Appendix A.4 | Cite adaptive-data-analysis / post-selection cautions (e.g., Dwork et al., 2015; Leeb & Pötscher, 2015) | CITED |

---

## C. Adaptive sequential stopping (early stopping)

| ID | Claim (short) | Where used | Verification | Status |
|---:|---|---|---|---|
| C1 | Under continuous-null idealization, $S_n := P(p^\star<\alpha \mid L_n,n)=I_\alpha(1+L_n,1+n-L_n)$ is a martingale | `paper/docs/theory.md` §6.1.3.8; `paper/docs/paper.md` Appendix A.5.2 | Doob martingale: $S_n = E[1\{p^\star<\alpha\}\mid \mathcal F_n]$ | PROVED |
| C2 | $W_n := S_n/\alpha = (1-F_{\text{Binom}}(L_n; n+1,\alpha))/\alpha$ is the Fischer–Ramdas binomial-mixture e-process | `paper/docs/theory.md` §6.1.3.8–6.1.3.9; `paper/docs/paper.md` Appendix A.5.2 | Beta–binomial identity + Fischer & Ramdas (2025, Prop. 5) | PROVED + CITED |
| C3 | Ville/Markov bound for the posterior-confidence stop: $\Pr(\exists n:\; S_n\ge \gamma)\le \alpha/\gamma$ | `paper/docs/theory.md` §6.1; `paper/docs/paper.md` Appendix A.5.2 | Ville on $W_n$ (or Markov + tower property at stopping times) | PROVED |
| C4 | The returned $\hat p_\tau=(L_\tau+1)/(\tau+1)$ is *not* claimed to be an anytime-valid p-value under optional stopping | `paper/docs/theory.md` §6.1; `paper/docs/paper.md` Appendix A.5.1–A.5.2 | Clarifies scope; aligns with sequential MC testing literature | POLICY (must keep) |

**Empirical backstops:**
- E2: `paper/scripts/theory/generate_sequential_stopping_calibration.py` → `paper/results/figures/sequential_stopping_calibration_{data,plot}`.
- E3: `paper/scripts/theory/sequential_stopping_comparison.py` prints a reproducible comparison vs Fischer–Ramdas on a Pearson-correlation test.
- E4: `paper/scripts/theory/supermartingale_check.py` numerically checks the martingale identity and prints a calibration run.

---

## D. Items to resolve before “paper-ready”

- Decide whether we want to include any sequential-stopping **decision-level** theorem in the main text (or keep it as
  an appendix-only statement).

---

## E. Critical review / risk register (be intentionally picky)

This section is written in the style of an adversarial reviewer. Each item is either an assumption that must be
spelled out, or a place where the mathematics could be misread as stronger than it is.

1. **Exact scope of theorems (fixed node vs adaptive tree).**
   - Airtight: fixed-node permutation p-values conditional on $(X,U)$ (A1–A3), especially at the root.
   - Not airtight: interpreting internal-node tests as classical p-values for a fixed family in an adaptively grown tree
     (B2).
   - Paper action: keep all “valid p-value” language explicitly root/fixed-node scoped.

2. **Stage B is post-selection (“double dipping”).**
   - Stage B reuses the same labels after selecting $j_t^\star$ in Stage A, so naïve “Stage B p-value” interpretation
     is post-selection (B1).
   - Paper action: treat Stage B p-values as algorithmic split-validation/stopping statistics unless we add sample
     splitting or selective-inference machinery.

3. **Early stopping: decision guarantee vs p-value guarantee.**
   - Proven: a bound on the *posterior-confidence stop\_sig event* under the continuous-null idealization (C3).
   - Not proven (and currently false in general): that the returned $\widehat p_\tau$ is super-uniform under optional
     stopping (C4).
   - Paper action: never write “adaptive mode yields valid p-values”; write “adaptive mode yields a decision-level bound
     for the confidence stop; fixed-$B$ yields valid p-values.”

4. **Continuous-null idealization vs discrete/tied statistics.**
   - The clean martingale/e-process argument (C1–C3) uses the PIT/uniform-mixture story, which is exact when the
     permutation distribution is continuous (tie-free).
   - With ties/atoms, $p^\star$ is not exactly Uniform; the sequential bound then relies on tie-handling conventions
     from the sequential betting/e-process literature, or must be presented as “idealization + empirical calibration.”
   - Paper action: keep ties/discreteness as an explicit limitation; do not overstate the sequential theorem.

5. **Monte Carlo permutation model (with-replacement draws).**
   - The sequential analysis assumes i.i.d. Monte Carlo draws from the permutation distribution (with replacement).
     This matches the implementation (fresh random shuffle each step), but differs from exact enumeration
     (without replacement).
   - Paper action: state explicitly that the analysis is for the Monte Carlo test used in the code.

6. **Candidate-set randomness must be label-independent for clean validity.**
   - Root-level random subsets of features/thresholds are safe if chosen as functions of $(X,U)$, independent of $Y$
     under the null (A2).
   - Feature muting / scanning can induce label-dependent candidate families later in the tree.
   - Paper action: do not “import” fixed-node validity into adaptive-tree-wide claims.

7. **What “unbiased feature selection” means.**
   - “Unbiased” here means “no systematic preference for high-cardinality noise under the global null,” not “unbiased
     estimator” in a parametric sense.
   - Paper action: define the notion precisely and align language with Hothorn et al. (2006).

8. **Implementation wording vs paper wording.**
   - Some inline code comments/docstrings (outside `paper/docs/`) use informal language like “valid Type I error” for
     adaptive stopping; the paper should keep the sharper distinction above.
   - Paper action: ensure the manuscript text never claims more than C3/C4 for adaptive stopping.
