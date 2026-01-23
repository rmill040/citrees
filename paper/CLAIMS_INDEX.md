# Claims Index (Theory + Guarantees)

This file is the internal, session-to-session index of **every formal claim** we
make in the arXiv manuscript (`paper/arxiv/`). It exists to keep the theory
airtight:

- Every manuscript claim must have a label, explicit assumptions (A0.\*), and a
  dedicated proof location (one appendix per claim).
- If a claim is mentioned in the abstract/introduction, it must appear as a
  labeled statement somewhere in the manuscript + be proved (or be removed).

See also: `paper/WRITING_CHECKLIST.md`.

## Status legend

- `TODO`: claim is referenced but not yet written/formalized
- `DRAFT`: statement exists, proof exists but not fully QA'd
- `QA`: proof has been checked against the Proof QA checklist

## Current claims (as of today)

| ID / label                              | Where stated                                            | Type                 | Assumptions                        | Proof location                                                                         | Status | Notes / dependencies                                                                                  |
| --------------------------------------- | ------------------------------------------------------- | -------------------- | ---------------------------------- | -------------------------------------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------- |
| `lem:mc-exchangeability`                | `paper/arxiv/appendices/appendix_B_exchangeability.tex` | Lemma                | (Monte Carlo construction)         | `paper/arxiv/appendices/appendix_B_exchangeability.tex`                                | DRAFT  | Used to justify the exchangeability setup for rank arguments.                                         |
| `them:plusone-superuniform`             | `paper/arxiv/sections/04_theory.tex`                    | Theorem              | A0.1–A0.5                          | `paper/arxiv/appendices/appendix_C_plusone_pvalue.tex` (`app:proof-plusone`)           | DRAFT  | Needs final QA pass on tie handling + conditioning language.                                          |
| `prop:stageA-global-null`               | `paper/arxiv/sections/04_theory.tex`                    | Proposition          | A0.1–A0.5 + super-uniform p-values | `paper/arxiv/appendices/appendix_D_stageA_global_null.tex` (`app:proof-stageA-global`) | DRAFT  | Union bound; ensure $F_t$ and $B$ are label-independent (A0.3).                                       |
| `prop:per-feature-bound`                | `paper/arxiv/sections/04_theory.tex`                    | Proposition          | A0.1–A0.5 + super-uniform p-values | `paper/arxiv/appendices/appendix_E_per_feature_bound.tex` (`app:proof-per-feature`)    | DRAFT  | Ensure event “splits on feature j” implies Stage A rejected for j (depends on algorithm definition).  |
| `prop:any-split-implies-root-rejection` | `paper/arxiv/sections/04_theory.tex`                    | Proposition          | A0.1–A0.5 (at root)                | `paper/arxiv/appendices/appendix_F_root_any_split.tex` (`app:proof-root`)              | DRAFT  | Relies on algorithm structure: any split implies root Stage A rejection.                              |
| `prop:multiselector-validity`           | `paper/arxiv/sections/04_theory.tex`                    | Proposition          | A0.1–A0.5                          | `paper/arxiv/appendices/appendix_G_multiselector.tex` (`app:multiselector`)            | DRAFT  | Validity requires exchangeability of the joint selector vector; power depends on scale/normalization. |
| `lem:cart-proportional-selection`       | `paper/arxiv/appendices/appendix_I_cart_bias.tex`       | Lemma (motivational) | A0.6                               | `paper/arxiv/appendices/appendix_I_cart_bias.tex` (`app:cart-bias`)                    | DRAFT  | Used only as motivation for high-cardinality bias; not a p-value guarantee.                           |

## Planned near-term additions (from notes migration)

These items are expected when migrating theory/methods from
`paper/notes/notes.md`:

- Formal multi-selector max-T statement (Westfall–Young style) + label.
- Clear “non-claims” section for:
  - Stage B p-values,
  - internal nodes/adaptivity,
  - early stopping / sequential testing outputs.
- If included as results: CART high-cardinality selection-bias lemma(s) should
  be placed in a dedicated appendix and explicitly labeled “motivational, not a
  guarantee.”
