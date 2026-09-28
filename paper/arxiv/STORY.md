# Story of the arXiv paper

The paper tells one story. The contributions paragraph states the answers to the
ten questions below, and the paper then takes each question in the same order.
Every section, table, and figure serves one of them; a study that serves none
belongs in the appendix as backup or out of the paper.

## Pitch

Conditional inference forests give random-forest-quality feature rankings in
which every split is a valid statistical test, and the max-type threshold test
makes that affordable. They are the best conditional inference method for
feature ranking. They are no better at finding the truly informative features,
and the unbiased split selection they are known for is not why they work: the
forest is.

## The ten questions, in order

Each question is one a skeptical reader asks. Each item gives the answer, the
evidence, and where the paper answers it.

1. **Does CIF rank features well for prediction?** It ties random forests in
   classification (mean rank 5.86 each, tied 3rd of 17), is 2nd of 18 in
   regression, and beats every other conditional inference method (cforest 17–5
   and ctree 19–2 over classification datasets). Evidence: the real-data
   benchmark of 17 and 18 rankers on 31 datasets and the direct comparisons with
   ctree, cforest, CIT, and the single trees. Paper: Results, real-data
   benchmark (Tables 2 and 3).
2. **Is that result an artifact of the setup?** Not in classification; the
   regression ordering is descriptive. Evidence: the complete-case panels and
   Friedman tests, leave-one-dataset-out configuration reselection, breakdowns
   by learner, k, and seed, and paired intervals of CIF against every method.
   Paper: Results, real-data benchmark; appendix on benchmark robustness.
3. **Is the margin over cforest only a different importance measure?** No:
   ranked by cforest's own out-of-bag permutation importance, CIF keeps its
   classification margin. Evidence: the mechanism-matched check. Paper: Results,
   real-data benchmark; appendix on benchmark robustness.
4. **Is the random forest good only because of its biased importance?** No: the
   standard remedies for that bias rank lower. The same forest ranked by
   out-of-bag permutation importance is 8th, cforest 9th, permutation importance
   12th, and conditional permutation importance last. Evidence: the random
   forest ranked by out-of-bag permutation importance and the benchmark
   positions of cforest, PI, and CPI. Paper: Results, real-data benchmark;
   appendix on benchmark robustness.
5. **Does CIF find the truly informative features?** Only as well as the middle
   of the field: 8th to 10.5th of 15 in classification and 8th to 11.5th of 16
   in regression at every list size from 1 to 100. Evidence: synthetic recovery
   at k = 1, 5, 10, 25, 50, and 100. Paper: Results, where CIF is weaker (Table
   4); appendix on where CIF is weaker.
6. **Why not?** The forest carries the ranking: one tree loses on every dataset,
   while the importance type, bootstrap sampling, and feature muting do not
   matter. A single tree's list is mostly zero-importance filler. Feature
   sampling keeps informative features out of most splits in sparse wide data
   (33% of splits against 92% when every feature is tested). CIF is weakest at
   small k in classification and peaks before the full feature set on wide data.
   Evidence: the CIF ablation, the support measurement, the feature-use study,
   the rank by k, and the high-p analysis. Paper: Results, where CIF is weaker
   and what the CIF ranking rests on; appendix on where CIF is weaker.
7. **Does the unbiased splitting remove the cardinality bias, and does removing
   it help?** It removes the random forest's preference for many-valued noise,
   with a mild lean toward binary noise in regression, but the biased random
   forest recovers at least as many informative features at every signal and
   list size. Removing the two corrections deepens the trees and moves CIF's
   downstream score by at most 0.004. Evidence: the weak-signal cardinality
   designs and the runtime ablation without the corrections. Paper: Results,
   what the corrections change; appendix on the cardinality designs.
8. **What does the statistical machinery cost?** The two corrections cost of
   order m²/α permutations per node in the feature test and K²/α in the
   threshold test, and the threshold test is where the time goes. Evidence: the
   cost accounting, the runtime ablations, the benchmark-size check, and the
   scaling measurements. Paper: Method (permutation budgets); Results, what the
   corrections cost; appendix on runtime and memory.
9. **Can the guarantee be had more cheaply?** Yes. The max-type threshold test
   is valid at a fixed node, keeps its size flat as the candidate count grows,
   loses 0.006 in power at equal realized size, fits forests 10 to 36 times
   faster, and ranks slightly lower (5th of 17 and 4th of 18), so it is opt-in.
   Evidence: Corollary 3, the threshold-only size and matched-power study, the
   no-correction size, the scaling, real-data, and head-to-head timing studies,
   and the max-type benchmark rerun. Paper: Fixed-node control; Results, the
   max-type threshold test; appendix on the threshold test comparison.
10. **Is the default stopping rule valid?** Yes: its exact null size is 0.0488
    at 999 permutations, and at the benchmark budget it never activates, so
    every benchmark P-value is a fixed-budget one. Evidence: Proposition 1.
    Paper: Fixed-node control (size of the adaptive stopping rule); proofs
    appendix.

Section 3 states the fixed-node results before the Results, so questions 9 and
10 draw on it there; the Results answer questions 1 to 9 in the order above.

## Results in story order

1. Real-data benchmark: questions 1 to 4.
2. Where CIF is weaker: question 5.
3. What the CIF ranking rests on: question 6.
4. What the corrections change: question 7.
5. What the corrections cost: question 8.
6. The max-type threshold test: question 9.

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

- The abstract, the contributions paragraph, and the conclusion state the pitch.
- The contributions paragraph answers the ten questions in order.
- The Results take questions 1 to 9 in order, one subsection per step of the
  list above.
- The discussion interprets the answers without re-reporting tables, and the
  limitations are the single home for caveats.
- Every table and figure in the body answers one question; the rest sits in the
  appendix under the question it backs.
