# Story Motivation

This is the short narrative memo for the paper. It is not a numerical source of
truth.

## Core Motivation

Conditional inference got an important idea right: separate feature selection
from threshold optimization to reduce split-selection bias.

The practical problem was that the classical approach could be too
expensive to use comfortably, especially when many permutations and exhaustive
search were applied too uniformly.

The new benchmark results sharpen the validation story, not the motivation
itself. The motivation is to improve conditional inference trees and forests as
practical feature selection methods: reduce runtime, keep the rankings useful,
and understand where the runtime hyperparameters change the method.
The broad benchmark then checks how the improved methods rank against common
feature selection methods. In that validation layer, CIF ranks
`4th/17` on classification and `3rd/18` on regression, while staying positive
against `ctree`, `cforest`, CIT, and DT on datasets where both methods were run.
Against RT, CIF is
clearly positive on classification and nearly tied on regression. CIT provides
the single tree conditional inference comparison and a separate timing contrast.

The paper is therefore about a practical design question:

- which parts of conditional inference are essential,
- which computations are overbuilt,
- and which practical controls preserve the useful behavior while making the
  method usable.

In the benchmarked tree-growing algorithm, adaptive stopping is the largest CIF
runtime hyperparameter measured here: disabling it makes CIF `4.0--8.4x` slower
with only small changes in downstream score and feature recovery on synthetic
datasets. Bounded histogram thresholding is different: it is a separate Stage~B
search rule that shrinks the search set itself, and exact threshold search is
`1.9--10.8x` slower than histogram-256 in the CIF timing study. Feature scanning
has a smaller and less stable timing effect. For
CIT, the timing results are mixed rather than a clean speed story: disabling
adaptive stopping can be faster on synthetic runs but is not consistently faster
on real datasets, and the CIT timing run measures fit/ranking time plus
synthetic top-10 recovery rather than downstream model performance.

## Main Results To Carry Into The Motivation

- The core result is practical improvement of conditional inference trees and
  forests, not just a benchmark ranking.
- CIF timing supports the main runtime story: adaptive stopping and histogram
  thresholding provide the largest runtime reductions with small score or
  recovery changes.
- CIT timing belongs in its own lane: its runtime effects are mixed, and the
  CIT timing run measures fit/ranking time plus feature recovery on synthetic
  datasets rather than downstream model performance.
- The benchmark validation is a joined all-vs-all comparison with DT and RT
  included: `17` classification methods and `18` regression methods.
- DT and RT are supporting single tree checks. They help show that the broader
  benchmark is not only a conditional inference reference comparison, but they
  should not be abstract-level framing.
- CIF ranks `4th/17` on classification and `3rd/18` on the smaller regression
  benchmark after the runtime focused changes.
- CIF's cleanest direct wins are against the historical conditional inference
  references and the single tree conditional inference family: `22/22` wins
  versus `ctree`, `19/22` versus `cforest`, `22/23` versus CIT, `14/23` versus
  DT, and `21/23` versus RT on classification. On regression the corresponding
  wins are `7/8`, `7/8`, `6/8`, `6/8`, and `5/8`.
- CIT improves on `ctree` and `cforest` in several direct comparisons, but it
  does not cleanly beat DT on classification. That supports the decision to
  separate CIT timing evidence from CIF benchmark evidence.

## Main Story

The most cohesive paper story is:

1. a narrow fixed node Stage~A feature selection object gives the method a principled core,
2. adaptive stopping and bounded histogram thresholding reduce runtime while
   preserving ranking quality well enough to keep the method useful,
3. CIT and CIF timing should be reported separately because they measure
   different evidence,
4. real classification benchmarks validate that the improved CIF remains worth
   using as a feature ranker among modern filters, wrappers, trees, forests, and
   boosted models,
5. CIF's broadest clean wins are against the historical conditional inference
   references and single tree references, while broader method comparisons act
   as validation,
6. high-`p`, synthetic, and mechanism studies explain the boundary in a
   classification first way:
   CIF often improves beyond very small feature sets without usually needing the
   full feature set, is better at disciplined subset construction than at exact
   first feature discovery, and weakens when sparse candidate exposure limits
   forest growth.

This is an optimization first paper, not a package bakeoff and not a synthetic
benchmark paper. The benchmark is not incidental, but it is a validation layer
for the optimized learner rather than the paper's primary object.

Two boundary rules should stay explicit throughout the manuscript:

- inferential boundary:
  the theorem supports only the exhaustive fixed node Stage~A
  feature selection rule under the nodewise complete permutation null; the
  practical control and benchmark layers validate useful ranking behavior, not
  calibrated p-values for the full adaptive learner,
- runtime boundary:
  speed claims are only within-method CIT or CIF comparisons under the
  collected setup; the distributed paper pipeline should not be used for
  cross-method wall-clock claims or hardware independent speed comparisons.

## Main-Text Package

The main text should center on:

- real benchmark validation:
  the 14-dataset classification benchmark as the credibility anchor,
  classification and regression `k` trajectories with changing dataset
  support as descriptive companions, and regression as a smaller directional
  mirror,
- practical controls:
  separate CIT timing from CIF timing and quality; for CIF, adaptive stopping
  first and bounded histogram thresholding second, with feature scanning
  reported as smaller and less stable,
- a compact robustness layer:
  breadth / heterogeneity,
- a compact high-`p` saturation check,
- one synthetic figure that distinguishes subset construction from exact
  first feature recovery,
- one mechanism figure that explains the candidate feature coverage failure mode in
  sparse classification forests.

Recommended main-text artifacts:

- `paper/results/figures/k_trajectory.png`
- `paper/results/figures/regression_k_trajectory.png`
- `paper/results/tables/paper_presentation_benchmark_summary.csv`
- `paper/results/tables/paper_presentation_practical_controls_summary.csv`
- `paper/results/tables/paper_cit_runtime_ablation_summary.csv`
- compact robustness summary drawn from:
  `paper/results/tables/paper_heterogeneity_cif_pairwise_breadth.csv` and
  `paper/results/tables/paper_heterogeneity_method_summary.csv`
- compact high-`p` table built from:
  `paper/results/tables/paper_high_p_cif_endpoint_summary.csv` and
  `paper/results/tables/paper_high_p_cif_best_observed_k_summary.csv`
- `paper/results/figures/synthetic_topk_focus_curves.png`
- compact first feature support drawn from:
  `paper/results/tables/top_ranking_summary.csv`
- `paper/results/figures/paper_mechanism_grid_forest_classification_feature_counts_p1000_i2_1000trees.png`

Calibration belongs in the main text only if the saved calibration artifacts
are rerun and re-locked.

## Supporting / Supplemental Package

These layers are important, but they should support the main story rather than
compete with it:

- fixed node null diagnostics / calibration,
- extended high-`p` support tables and endpoint checks,
- synthetic top-ranking tables and detailed composition trends,
- extended mechanism tables and regression mechanism mirrors,
- CIF-vs-R historical comparison.

## What The Paper Should Not Claim

Do not claim:

- that CIF is the universal best method,
- that any single `k` is the main operating point,
- that the paper is mainly a Python-versus-R library comparison,
- that the full adaptive learner has inferential calibration,
- that every practical control is a central empirical finding,
- that exact first feature recovery is the same object as useful subset
  construction,
- that the mechanism layer fully replaces the real data benchmark.
