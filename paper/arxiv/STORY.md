# Story of the arXiv paper

The paper tells one story. It is organized as four contributions, each answering
questions a skeptical reader asks, and every part of the paper takes the
contributions in the same order. Every section, table, and figure serves one
contribution; a study that serves none belongs in the appendix as backup or out
of the paper.

## Pitch

Conditional inference forests match random forests as feature rankers for
downstream prediction, and they rank features better than every other
conditional inference method. Their ranking rests on the forest: one tree loses
on every dataset, while the unbiased split selection they are known for removes
the random forest's preference for many-valued features without improving
recovery or prediction. The split tests set the cost instead, and the max-type
threshold test keeps their fixed-node guarantee at a fraction of it. CIF matches
random forests as a feature ranker and splits only when a permutation test
rejects; the paper proves those tests valid at a fixed node for trees grown on
subsamples or the full sample, and the max-type test makes them affordable. The
benchmarked CIF (bootstrap samples, honesty in regression) is not claimed to
have valid split tests.

## The four contributions

### 1. Ranking quality

Results, ranking quality (Section 5.1); appendix on benchmark robustness.

1. **Does CIF rank features well for prediction?** It ties random forests in
   classification (mean rank 5.86 each, tied 3rd of 17), is 2nd of 18 in
   regression, and beats every other conditional inference method (cforest 17–5
   and ctree 19–2 over classification datasets). Evidence: the real-data
   benchmark of 17 and 18 rankers on 31 datasets and the direct comparisons with
   ctree, cforest, CIT, and the single trees (Tables 2 and 3).
2. **Is that result an artifact of the setup?** Not in classification; the
   regression ordering is descriptive, and CIF is tied 6th and 4th on the
   complete-case panels (13 and 6 datasets). Evidence: the complete-case panels
   and Friedman tests, leave-one-dataset-out configuration reselection,
   breakdowns by learner, k, and seed, and paired intervals of CIF against every
   method.
3. **Is the margin over cforest only a different importance measure?** No:
   ranked by cforest's own out-of-bag permutation importance, CIF keeps its
   classification margin. Evidence: the mechanism-matched check.
4. **Is the random forest good only because of its biased importance?** No: the
   standard remedies for that bias rank below it. In classification the random
   forest ranked by out-of-bag permutation importance is 8th of 18, and cforest,
   permutation importance and conditional permutation importance are 9th, 12th
   and 17th of 17 (10th, 13th and 18th of 18 with RF permutation present).
   Evidence: the random forest ranked by out-of-bag permutation importance and
   the benchmark positions of cforest, PI, and CPI.

### 2. Ranker behavior

Results, how the CIF ranker behaves (Section 5.2); appendix on where CIF is
weaker.

5. **Does CIF find the truly informative features?** No better than the middle
   of the field: 8th to 10.5th of 15 in classification and 8th to 11.5th of 16
   in regression at the six reported list sizes. On the redundant-feature
   designs the measure saturates at k <= 10 and k >= 50; at k = 25 it separates
   the methods, with CIF 1st of 16 in regression and tied 7.5th of 15 in
   classification. Evidence: synthetic recovery at k = 1, 5, 10, 25, 50, and 100
   (Table 4).
6. **Where is it weaker, and what does the ranking rest on?** It is weakest at
   small k in classification and peaks before the full feature set on wide data.
   The forest carries the ranking: one tree loses on every dataset, while the
   importance type, bootstrap sampling, and feature muting do not matter in
   classification; in regression split-count ranking and disabling bootstrap
   lower the mean R^2 with intervals excluding zero, driven by the coepra
   datasets, while the medians stay near zero. A single tree's list is mostly
   zero-importance filler. Feature sampling keeps informative features out of
   most splits in sparse wide data (33% of splits against 92% when every feature
   is tested). Evidence: the rank by k, the high-p analysis, the CIF ablation,
   the support measurement, and the feature-use study.

### 3. The statistical machinery

Results, what the statistical machinery changes and costs (Section 5.3);
appendices on runtime and memory and on the cardinality designs.

7. **Does unbiased split selection remove the cardinality bias, and does
   removing it help?** It removes the random forest's preference for many-valued
   noise, with a mild lean toward binary noise in regression, at every list
   size. The biased random forest still recovers at least as many informative
   features at every signal and list size. Removing the two corrections deepens
   the trees and moves CIF's downstream score by at most 0.005; for CIT it
   lowers the precision of the first few picks in classification. Evidence: the
   weak-signal cardinality designs at top 5, 10, and 25 and the runtime ablation
   without the corrections.
8. **What does the statistical machinery cost?** The two corrections cost of
   order m²/α permutations per node in the feature test and K²/α in the
   threshold test, and the threshold test is where the time goes. Evidence: the
   cost accounting, the runtime ablations, the benchmark-size check, and the
   scaling measurements.

### 4. A cheaper test with guarantees

Fixed-node control (Section 3); Results, the max-type threshold test (Section
5.4); proofs appendix and appendix on the threshold test comparison.

9. **Can the guarantee be had more cheaply?** Yes. The max-type threshold test
   is valid at a fixed node, keeps its size flat as the candidate count grows,
   loses 0.006 in power at equal realized size, fits forests 10 to 36 times
   faster on the synthetic head-to-head cells (1.1 to 8 times on the real
   head-to-head datasets, 1.6 to 7 on the runtime panel), and ranks slightly
   lower (5th of 17 and 4th of 18), so it is opt-in. Without the threshold
   correction the per-threshold test has size 3 to 8 times nominal. Evidence:
   Corollary 3, the threshold-only size and matched-power study with the
   no-correction size, the scaling, real-data, and head-to-head timing studies,
   and the max-type benchmark rerun.
10. **Is the default stopping rule valid?** Yes: its null size is at most 0.0488
    at 999 permutations, and at the benchmark budget it never activates, so
    every benchmark P-value is a fixed-budget one. Evidence: Proposition 1.

## How each part of the paper tells the story

- **Abstract**: the problem, the benchmark, then contributions 1 to 4 in order,
  ending on the pitch.
- **Introduction**: the question (how the rankings compare, and what the
  corrections cost and change), then the four contributions as a numbered list.
- **Method and fixed-node control**: define the two tests, the corrections, the
  budgets and their cost terms, and the max-type rule; state the guarantees
  behind contribution 4 before the Results use them.
- **Experimental design**: the benchmark, then the other studies in the order of
  the Results.
- **Results**: one subsection per contribution, 5.1 to 5.4, answering questions
  1 to 9 in order.
- **Discussion**: interprets without re-reporting tables, in contribution order:
  an unbiased split rule and a better ranking are different properties;
  prediction and recovery ask different things of a ranking; the corrections set
  the cost and the shape of the trees, and the max-type rule is the cheaper
  opt-in. Then one guidance paragraph and one limitations paragraph.
- **Conclusion**: two paragraphs (what CIF is as a ranker: quality, recovery,
  what the ranking rests on, the cardinality result; what the split tests are
  for: validity, cost, the max-type rule), ending on the pitch.
- **Appendices**: A and B back contribution 4 (setup, assumptions, proofs); C
  and D back the benchmark (configurations, exclusions, datasets); E backs
  contribution 1 (benchmark robustness); F backs contribution 2 (where CIF is
  weaker); G backs contribution 3 (runtime and memory); H backs contributions 3
  and 4 (threshold test comparison and cardinality designs).

## Caveats that travel with the story

The limitations paragraph is the single place where the full set is stated; the
regression-descriptive caveat also travels with the headline (abstract, Q2,
results, conclusion).

- The regression ordering, on 8 datasets, is descriptive.
- The comparison of biased and unbiased importance rests on different methods
  and one ablation of the corrections, not on one switch that removes the split
  test alone; removing the corrections also deepens the trees.
- The feature-use study and the cardinality designs use the linear selectors,
  while three of the four selected configurations use RDC.
- The guarantees hold at a fixed node and exclude the bootstrap samples CIF
  uses; fitted trees and forests are assessed through the benchmark.

## Backup studies

These support an answer without carrying any part of the story, so they stay in
the appendix.

- The cost of the max-type test is measured three ways (one tree, forests, and
  the runtime datasets); the forest timing carries question 9.
- The full-node calibration cannot isolate the threshold test, because the
  feature test gates the node first; the threshold-only study answers
  question 9.
- The RDC projection-count check belongs to the companion software paper and
  appears as a pointer.

## Checking the paper against the story

- The abstract, the four contributions, the discussion, and the conclusion state
  the pitch in the same order, each in its own words; no sentence is repeated
  across sections, and results are reported once, in the Results.
- The Results take questions 1 to 9 in order, one subsection per contribution.
- Top-k results are reported as trends over list sizes or averages over them,
  never at one list size unless the design targets it.
- The discussion interprets the answers without re-reporting tables, and the
  limitations are the single place where the full set of caveats is stated.
- Every table and figure in the body answers one question; the rest sits in the
  appendix under the contribution it backs.
