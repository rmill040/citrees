# Claims Index (Theory + Guarantees)

This file is the internal, session-to-session index of **every formal claim** we
make in the arXiv manuscript (`paper/arxiv/`). It exists to keep the theory
airtight:

- Every manuscript claim must have a label, explicit assumptions (A0.\*), and a
  dedicated proof location (one appendix per claim).
- If a claim is mentioned in the abstract/introduction, it must appear as a
  labeled statement somewhere in the manuscript + be proved (or be removed).

See also: `paper/docs/writing-checklist.md`.

## Status legend

- `TODO`: claim is referenced but not yet written/formalized
- `DRAFT`: statement exists, proof exists but not fully QA'd
- `QA`: proof has been checked against the Proof QA checklist

## Current claims (as of today)

| ID / label                              | Where stated                                            | Type                 | Assumptions                        | Proof location                                                                         | Status | Notes / dependencies                                                                                                                          |
| --------------------------------------- | ------------------------------------------------------- | -------------------- | ---------------------------------- | -------------------------------------------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `lem:mc-exchangeability`                | `paper/arxiv/appendices/appendix_B_exchangeability.tex` | Lemma                | (Monte Carlo construction)         | `paper/arxiv/appendices/appendix_B_exchangeability.tex`                                | QA     | With i.i.d. permutations for all $b=0,\dots,B$, $(T_0,\dots,T_B)$ is exchangeable conditional on $(X_t,Y_t)$.                                 |
| `lem:identity-vs-random`                | `paper/arxiv/appendices/appendix_B_exchangeability.tex` | Lemma                | A0.2--A0.3                         | `paper/arxiv/appendices/appendix_B_exchangeability.tex`                                | QA     | Bridges the standard implementation ($T_{\mathrm{obs}}$ + $B$ random permutations) to the i.i.d.-permutation setup used by the rank argument. |
| `them:plusone-superuniform`             | `paper/arxiv/sections/04_theory.tex`                    | Theorem              | A0.1–A0.5                          | `paper/arxiv/appendices/appendix_C_plusone_pvalue.tex` (`app:proof-plusone`)           | QA     | Rank argument for +1 Monte Carlo permutation p-values; includes explicit tie-handling and conditioning.                                       |
| `prop:stageA-global-null`               | `paper/arxiv/sections/04_theory.tex`                    | Proposition          | A0.1–A0.5 + super-uniform p-values | `paper/arxiv/appendices/appendix_D_stageA_global_null.tex` (`app:proof-stageA-global`) | QA     | Union bound (no dependence assumptions). Uses A0.3 to condition on $(X_t,U)$ so $F_t,m_t$ are fixed.                                          |
| `prop:any-split-implies-root-rejection` | `paper/arxiv/sections/04_theory.tex`                    | Proposition          | A0.1–A0.5 (at root)                | `paper/arxiv/appendices/appendix_F_root_any_split.tex` (`app:proof-root`)              | QA     | Structural implication: any split in the tree implies root Stage~A rejection.                                                                 |
| `lem:cart-proportional-selection`       | `paper/arxiv/appendices/appendix_I_cart_bias.tex`       | Lemma (motivational) | A0.6                               | `paper/arxiv/appendices/appendix_I_cart_bias.tex` (`app:cart-bias`)                    | QA     | Idealized symmetry result explaining proportional selection under an exchangeable null (not a guarantee).                                     |

## Planned near-term additions (from notes migration)

- Clear “non-claims” section for:
  - Stage B p-values,
  - internal nodes/adaptivity,
  - early stopping / sequential testing outputs.
- If included as results: CART high-cardinality selection-bias lemma(s) should
  be placed in a dedicated appendix and explicitly labeled “motivational, not a
  guarantee.”
