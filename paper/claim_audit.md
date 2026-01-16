# Claim Audit: p-values and sequential stopping (WIP)

This file is a running audit log to keep `paper/paper.md` and `paper/theory.md` mathematically honest.

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
| A1 | +1 Monte Carlo permutation p-value is super-uniform under exchangeability | `paper/theory.md` §3.4; `paper/paper.md` Appendix A.2 | Proof by exchangeable-rank argument; +1 convention matches Phipson & Smyth (2010) | PROVED + CITED |
| A2 | Conditional-on-$(X,U)$ super-uniformity (candidate-set randomness allowed if label-independent) | `paper/theory.md` §3.5–4.3; `paper/paper.md` Appendix A.2 | Conditioning argument + Theorem A1 | PROVED |
| A3 | Bonferroni FWER control with super-uniform p-values (no independence required) | `paper/theory.md` §4.1–4.2; `paper/paper.md` Appendix A.3 | Union bound | PROVED |
| A4 | “Any split implies root Stage A rejection” ⇒ under global null at root, $\Pr(\text{tree splits}) \le \alpha_{\text{sel}}$ | `paper/theory.md` §4.4; `paper/paper.md` Appendix A.3 | Set inclusion + A3 at root | PROVED |
| A5 | Multi-selector “max statistic inside each permutation” yields valid p-values | `paper/theory.md` §6(2); `paper/paper.md` Appendix A.2.3 | Composite statistic preserves exchangeability; cite max-T framing (Westfall & Young, 1993) | PROVED + CITED |

**Empirical backstops (optional):**
- E1: `paper/scripts/theory/generate_fixedB_pvalue_calibration.py` → `paper/results/figures/fixedB_pvalue_calibration_{data,plot}`.

---

## B. Stage B and adaptive tree growth (what we do *not* claim)

| ID | Claim (short) | Where used | Verification | Status |
|---:|---|---|---|---|
| B1 | Stage B p-values are post-selection and should not be presented as classical p-values without selective inference / splitting | `paper/theory.md` §2.3; `paper/paper.md` Appendix A.4 | Conceptual/statistical caveat (selective inference literature) | TODO (cites optional) |
| B2 | Internal-node tests in an adaptively grown tree are not classical p-values for a fixed family (conditioning on data-dependent node membership breaks exchangeability) | `paper/theory.md` §3.5 remark + §6; `paper/paper.md` Appendix A.4 | Conceptual/statistical caveat | TODO (cites optional) |

---

## C. Adaptive sequential stopping (early stopping)

| ID | Claim (short) | Where used | Verification | Status |
|---:|---|---|---|---|
| C1 | Under continuous-null idealization, $S_n := P(p^\star<\alpha \mid L_n,n)=I_\alpha(1+L_n,1+n-L_n)$ is a martingale | `paper/theory.md` §6.1.3.8; `paper/paper.md` Appendix A.5.2 | Doob martingale: $S_n = E[1\{p^\star<\alpha\}\mid \mathcal F_n]$ | PROVED |
| C2 | $W_n := S_n/\alpha = (1-F_{\text{Binom}}(L_n; n+1,\alpha))/\alpha$ is the Fischer–Ramdas binomial-mixture e-process | `paper/theory.md` §6.1.3.8–6.1.3.9; `paper/paper.md` Appendix A.5.2 | Beta–binomial identity + Fischer & Ramdas (2025, Prop. 5) | PROVED + CITED |
| C3 | Ville/Markov bound for the posterior-confidence stop: $\Pr(\exists n:\; S_n\ge \gamma)\le \alpha/\gamma$ | `paper/theory.md` §6.1; `paper/paper.md` Appendix A.5.2 | Ville on $W_n$ (or Markov + tower property at stopping times) | PROVED |
| C4 | The returned $\hat p_\tau=(L_\tau+1)/(\tau+1)$ is *not* claimed to be an anytime-valid p-value under optional stopping | `paper/theory.md` §6.1; `paper/paper.md` Appendix A.5.1–A.5.2 | Clarifies scope; aligns with sequential MC testing literature | POLICY (must keep) |

**Empirical backstops:**
- E2: `paper/scripts/theory/generate_sequential_stopping_calibration.py` → `paper/results/figures/sequential_stopping_calibration_{data,plot}`.
- E3: `paper/scripts/theory/sequential_stopping_comparison.py` prints a reproducible comparison vs Fischer–Ramdas on a Pearson-correlation test.
- E4: `paper/scripts/theory/supermartingale_check.py` numerically checks the martingale identity and prints a calibration run.

---

## D. Items to resolve before “paper-ready”

- Add citations for B1/B2 (selective inference / post-selection inference and “adaptive data analysis” warnings).
- Decide whether we want to include any sequential-stopping **decision-level** theorem in the main text (or keep it as
  an appendix-only statement).
