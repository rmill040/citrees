# Story Memo

This is the short narrative memo for the paper. It is not a numerical source of
truth.

## Core Motivation

Conditional inference got an important idea right: separate feature selection
from threshold optimization to reduce split-selection bias.

The practical problem was that the classical implementation could be too
expensive to use comfortably, especially when large permutation budgets and
exhaustive search were applied too uniformly.

The paper is therefore about a practical design question:

- which parts of conditional inference are essential,
- which computations are overbuilt,
- and which practical controls preserve the useful behavior while making the
  method usable.

In the implemented learner, adaptive stopping is the main runtime lever.
Bounded histogram thresholding is different: it is a separate Stage~B
approximation that shrinks the search set itself. Feature scanning matters only
inside the early-stopping regime and shows a smaller supporting gain there,
while threshold scanning plays the same architectural role on Stage~B
candidates. Threshold scanning, muting, and bootstrap are part of the executed
control surface, but they are background controls rather than coequal primary
discoveries.

## Main Story

The most cohesive paper story is:

1. a narrow fixed-node Stage~A feature-selection object gives the method a principled core,
2. adaptive stopping is the main practical win, bounded histogram thresholding is
   a separate second lever, and feature scanning gives a smaller supporting
   speed gain once early stopping is enabled; the other practical controls are
   background rather than primary results,
3. real classification benchmarks validate that the accelerated CIF package
   remains worth using as a feature ranker,
4. its broadest clean wins are against the historical conditional-inference
   baselines and the single-tree CIT baseline rather than the strongest generic
   ensembles,
5. high-`p`, synthetic, and mechanism studies explain the boundary in a
   classification-led way:
   CIF often improves beyond tiny budgets without usually needing the full
   feature set, is better at disciplined subset construction than at exact
   top-of-ranking discovery, and weakens when sparse candidate exposure limits
   forest growth.

This is an optimization-first paper, not a package bakeoff and not a synthetic
leaderboard paper. The benchmark is not incidental, but it is a validation
layer for the optimized learner rather than the paper's primary object.

Two boundary rules should stay explicit throughout the manuscript:

- inferential boundary:
  the theorem-backed object is only the exhaustive fixed-node Stage~A
  feature-selection rule under the nodewise complete permutation null; the
  practical-control and benchmark layers validate useful ranking behavior, not
  calibrated p-values for the full adaptive learner,
- runtime boundary:
  speed claims are only matched within-CIF comparisons under the collected
  setup; the distributed paper pipeline should not be used for cross-method
  wall-clock claims or hardware-independent speed comparisons.

## Main-Text Package

The main text should center on:

- real classification benchmark validation:
  the fixed-panel classification surface as the credibility anchor,
  the changing-support `k` trajectory as the descriptive companion,
  and regression as a smaller directional mirror,
- practical controls:
  adaptive stopping first, bounded histogram thresholding second, with feature
  scanning reported as a smaller supporting gain under early stopping and the
  remaining controls treated as background,
- a compact robustness layer:
  breadth / heterogeneity,
- a compact high-`p` saturation check,
- one synthetic figure that distinguishes subset construction from exact
  head-of-list recovery,
- one mechanism figure that explains the candidate-set-width failure mode in
  sparse classification forests.

Recommended main-text artifacts:

- `paper/results/figures/k_trajectory.png`
- `paper/results/tables/paper_presentation_benchmark_summary.csv`
- `paper/results/tables/paper_presentation_practical_controls_summary.csv`
- compact robustness summary drawn from:
  `paper/results/tables/paper_heterogeneity_cif_pairwise_breadth.csv` and
  `paper/results/tables/paper_heterogeneity_method_summary.csv`
- compact high-`p` table built from:
  `paper/results/tables/paper_high_p_cif_endpoint_summary.csv` and
  `paper/results/tables/paper_high_p_cif_best_observed_k_summary.csv`
- `paper/results/figures/synthetic_topk_focus_curves.png`
- compact head-of-list support drawn from:
  `paper/results/tables/top_ranking_summary.csv`
- `paper/results/figures/paper_mechanism_grid_forest_classification_feature_counts_p1000_i2_1000trees.png`

Calibration belongs in the main text only if the saved calibration artifacts
are rerun and re-locked.

## Supporting / Supplemental Package

These layers are important, but they should support the main story rather than
compete with it:

- fixed-node null diagnostics / calibration,
- extended high-`p` support tables and endpoint checks,
- synthetic top-ranking tables and detailed composition trends,
- extended mechanism tables and regression mechanism mirrors,
- regression benchmark mirror,
- CIF-vs-R historical comparison.

## What The Paper Should Not Claim

Do not claim:

- that CIF is the universal best method,
- that any single `k` is the main operating point,
- that the paper is mainly a Python-versus-R library comparison,
- that the adaptive learner has end-to-end inferential calibration,
- that every practical control is a central empirical finding,
- that synthetic top-of-ranking recovery is the same object as useful subset
  construction,
- that the mechanism layer fully replaces the real-data benchmark.
