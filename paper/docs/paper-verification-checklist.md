# Paper Verification Checklist

Reusable checklist for reviewing each manuscript section and keeping the paper
coherent end to end.

Use this alongside:

- `paper/docs/paper-verification-log.md` for the running session log
- `paper/docs/claims-index.md` for formal claims
- `paper/docs/writing-checklist.md` for proof-specific QA

## How to use it

For each section we review:

1. Check scope and claim type first.
2. Check notation, terminology, and references.
3. Check code/manuscript alignment where applicable.
4. Check evidence for every empirical or numerical statement.
5. Only then polish wording.

---

## 1) Scope and claim type

- [ ] Every substantive statement is clearly one of:
  - proved mathematical claim,
  - empirically supported claim,
  - implementation description,
  - limitation / non-claim.
- [ ] The section does not blur theorem-backed claims with heuristic behavior.
- [ ] The section does not imply stronger inference than the paper actually provides.
- [ ] If a limitation exists elsewhere in the paper, this section does not contradict it.

## 2) Notation

- [ ] Every symbol used in the section is defined once before use.
- [ ] Symbols match the manuscript-wide notation.
- [ ] The same object is not renamed casually across sections.
- [ ] Displayed equations use only symbols that are locally or globally defined.
- [ ] Notation density is justified; unnecessary symbols or helper variables are removed.

## 3) Terminology

- [ ] Key terms are used consistently:
  - Stage~A / Stage~B
  - fixed node
  - fixed-$B$
  - complete null / permutation null
  - p-value vs algorithmic statistic
  - screening vs ranking / embedding
- [ ] Terms used in prose match the code-facing meaning where relevant.
- [ ] Terminology in captions, tables, and appendices matches the main text.

## 4) Mathematical content

- [ ] Main-text math includes only nontrivial statements that carry real load.
- [ ] Trivial corollaries or bookkeeping consequences are moved to notes or appendices.
- [ ] Assumptions are explicit and match the statement actually being made.
- [ ] Conditioning and randomness sources are clear.
- [ ] The result says exactly what it proves, and no more.
- [ ] Any “non-claims” are stated explicitly if the topic invites overreading.

## 5) Empirical claims

- [ ] Every empirical claim maps to a figure, table, or analysis output.
- [ ] The unit of aggregation is clear and consistent.
- [ ] The comparison uses a clearly defined common dataset set when pairwise or rank claims are made.
- [ ] Primary endpoints are distinguished from robustness checks.
- [ ] Claims do not mix incomparable settings without saying so.
- [ ] Headline claims are not based on exploratory summaries unless labeled as such.

## 6) Numerical statements

- [ ] Dataset counts are correct.
- [ ] Dataset-type filters are correct (for example, real-only vs synthetic-only).
- [ ] Method counts are correct.
- [ ] Fold / seed / split counts are correct.
- [ ] Attrition or missingness does not silently change the effective sample size across comparisons.
- [ ] Percentages, rank differences, effect sizes, and runtimes are traced to outputs.
- [ ] Counts in captions, text, and tables agree with each other.
- [ ] Rounded values remain faithful to the underlying results.

## 7) Figures and tables

- [ ] Every figure/table answers one clear question.
- [ ] Captions state what is being compared and under what setting.
- [ ] Null conditions are stated explicitly for calibration figures.
- [ ] Labels do not imply inference unless inference is actually justified.
- [ ] Table/figure references in text describe the correct takeaway.
- [ ] Main-text figures are necessary; appendix-only material stays out of the way.

## 8) Cross-references

- [ ] All section references are correct and useful.
- [ ] All theorem / proposition / lemma / corollary references resolve.
- [ ] All figure / table / equation references resolve.
- [ ] Appendix references match the current organization.
- [ ] The roadmap and contribution bullets match the actual paper structure.

## 9) Citations and bibliography

- [ ] Every literature claim has an appropriate citation.
- [ ] Citations support the exact claim being made.
- [ ] Canonical sources are cited where appropriate.
- [ ] No citation is doing work it does not actually support.
- [ ] Reference entries compile cleanly and consistently.

## 10) Code-manuscript alignment

- [ ] The described algorithm matches the actual implementation order.
- [ ] Hyperparameter semantics match the code.
- [ ] Scope notes about heuristics match what the code really does.
- [ ] Runtime / memory discussion matches the actual code path.
- [ ] Benchmark protocol text matches the actual analysis pipeline.

## 11) Writing quality

- [ ] The section has one main job and sticks to it.
- [ ] Repetition across nearby sections is minimized.
- [ ] Overclaiming words are removed unless fully justified.
- [ ] Dense technical material is explained plainly where needed.
- [ ] The section still reads cleanly after the correctness pass.

## 12) Final build and packaging

- [ ] The paper builds cleanly.
- [ ] No broken references remain.
- [ ] No stale section names or labels remain after edits.
- [ ] Figures used in the paper are available locally for arXiv packaging.
- [ ] Supporting docs/logs are updated if claims or structure changed.
