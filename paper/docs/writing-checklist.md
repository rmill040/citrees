# arXiv Paper Workplan + Proof Checklist (citrees)

This is an internal workplan for drafting the full arXiv manuscript in LaTeX. Do
**not** paste this into the manuscript; use it as a session-to-session
checklist.

## Guardrails

- Paper-only edits: limit changes to `paper/**` (no `citrees/**` source edits).
- Notes source of truth: `paper/docs/drafts.md` (messy is fine).
- Manuscript source of truth: `paper/arxiv/` (LaTeX, arXiv-ready).
- Every claim in the main paper must have:
  - a label (`\label{...}`),
  - an explicit assumption list (A0.\*),
  - a dedicated proof location (one appendix per claim).

## Key paths

- Notes: `paper/docs/drafts.md`
- Manuscript entrypoint: `paper/arxiv/main.tex`
- Main sections: `paper/arxiv/sections/`
- Proofs/technical details: `paper/arxiv/appendices/`
- References: `paper/arxiv/references.bib`
- Build:
  `cd paper/arxiv && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

## 0) Session start checklist (quick sanity)

- [ ] `latexmk` builds cleanly (no fatal errors).
- [ ] Any new claim added in `sections/**` is also added to the Claims Index.
- [ ] No “floating” symbols: new notation is defined once (and reused).

## 1) Lock the paper contract (what this paper _is_)

- [ ] Write a 2–3 sentence thesis statement: what is new + what is guaranteed +
      what is empirical.
- [ ] Write an explicit scope statement:
  - [ ] What we guarantee (fixed-node/root Stage A in fixed-B mode).
  - [ ] What we do not claim (Stage B classical p-values; internal-node
        p-values; early-stopped p-values without extra work).
- [ ] Define the paper’s “deliverables” list (theory chain + benchmarks +
      implementation notes).

## 2) Create/maintain a Claims Index (single source of truth)

Create a table enumerating _every_ theorem/proposition/lemma/corollary in the
manuscript. For each claim:

- [ ] Claim label (e.g., `them:plusone-superuniform`).
- [ ] One-line statement (for tracking).
- [ ] Assumptions used (A0.\* list).
- [ ] Proof location (Appendix file name).
- [ ] Dependencies (definitions / prior lemmas).
- [ ] “Scope note” (what this does NOT imply for the adaptive full tree).

## 3) Restructure appendices: one appendix per claim (hard requirement)

Target structure:

- [ ] One shared appendix for setup/notation/assumptions (common dependencies).
- [ ] One appendix file per claim that appears in the main paper.
- [ ] If we include “motivational only” CART-bias lemmas, put them in their own
      appendix and label as non-guarantee.

Each proof appendix must begin with:

- [ ] The exact claim statement (copied verbatim from the main paper).
- [ ] “Assumptions used: A0.x–A0.y”.
- [ ] A step-by-step proof.

## 4) Proof QA checklist (apply to every claim before calling it “done”)

Notation and conditioning:

- [ ] Every symbol is defined once and reused consistently.
- [ ] Conditioning is explicit (e.g., “conditional on $(X_t, U)$”).
- [ ] Randomness sources are enumerated (permutations, feature subsampling,
      threshold subsampling, RNG seeds).
- [ ] Anything asserted to be label-independent is proven/measurable w.r.t.
      $(X_t, U)$.

Permutation validity:

- [ ] Exchangeability assumption is stated in the form actually used by the
      proof.
- [ ] The “identity vs random permutation” subtlety is handled explicitly.
- [ ] Tie handling matches the implementation convention (conservative ties or
      randomized tie-breaking).
- [ ] Fixed-B is enforced: no optional stopping unless the claim is explicitly
      about optional stopping.

Edge cases:

- [ ] Constant features / degenerate statistics handled.
- [ ] Small sample constraints noted where needed (e.g., $n_t \ge 2$,
      $B \ge 1$).
- [ ] Classification edge cases: empty classes at node, etc.

Scope:

- [ ] Each claim includes a one-sentence scope limitation (adaptive tree !=
      fixed-node inference).

## 5) Migrate “Appendix H” (methods) into LaTeX (structured, not a dump)

Goal: turn `paper/docs/drafts.md` Appendix H into:

- Main paper methods (`paper/arxiv/sections/03_method.tex`): only what a
  reviewer must know.
- Methods appendices: full implementation-aligned algorithmic detail.

Migration plan:

- [ ] Define canonical algorithms (one per unit): FitTree, Stage A test, Stage B
      test, FitForest.
- [ ] Convert Markdown pseudocode into LaTeX `algorithm` blocks.
- [ ] Define all objects used in theory (candidate sets, tested families,
      resample budgets, tails).
- [ ] Add a dedicated appendix for early stopping mechanics and clearly mark
      which theory excludes it.
- [ ] Add/finish bootstrap appendix section: current bootstrap behavior, and
      how it interacts (or does not) with fixed-node guarantees.

## 6) Background / “state of the field” (make comparisons explicit)

- [ ] Compare to R `party` / `partykit` implementations (what differs, what
      matches).
- [ ] Compare to CART/sklearn selection mechanism and why high-cardinality bias
      arises.
- [ ] List the feature-selection baseline families used in experiments
      (filters/wrappers/embedded).
- [ ] Cite selective inference / honesty / sample splitting where relevant (as
      limitations or future work).

## 7) Experiments (turn the skeleton into paper-grade results)

Design:

- [ ] Specify datasets (synthetic + real), splits, repetitions, seeds.
- [ ] Specify compute budgets (B, number of trees, max_features rules, etc.).
- [ ] Specify fairness policy for hyperparameters (fixed vs tuned; what’s
      allowed).

Outputs:

- [ ] Ranking quality: downstream performance vs top-k (rank-then-evaluate).
- [ ] Ranking stability: Kendall tau / Jaccard@k (define exactly).
- [ ] Calibration: fixed-B permutation p-value calibration plots at the root.
- [ ] Runtime: scaling vs (n, p, B), plus wall-clock comparisons.
- [ ] Figure/caption scope QA: every “calibration” plot states the simulated
      null (e.g., complete global null) and matches the fixed-node, fixed-B
      Stage~A scope; early-stopped outputs are labeled as algorithmic statistics
      (not calibrated p-values).

Artifacts:

- [ ] Decide minimum figures/tables for arXiv.
- [ ] Ensure figures are generated reproducibly and included in the arXiv
      bundle.

## 8) Implementation + reproducibility appendix

- [ ] RNG + seeding policy (Python vs Numba constraints).
- [ ] JIT on/off testing modes (what changes, what does not).
- [ ] Glossary mapping paper notation to library parameters (short and direct).

## 9) arXiv packaging checklist

- [ ] No external build steps required beyond LaTeX/BibTeX.
- [ ] All figures are local under `paper/arxiv/` (no absolute paths).
- [ ] No shell-escape dependencies.
- [ ] No undefined references/citations in a clean build.

## 10) Final polish (only after structure + proofs are stable)

- [ ] Add a short “paper roadmap” paragraph in the introduction.
- [ ] Add a notation table/glossary (main or appendix).
- [ ] Tighten language to avoid overclaiming (especially around p-values and
      adaptivity).
