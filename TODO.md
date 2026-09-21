# TODO

Open work before the arXiv v3 and JSS submissions. Items are deleted when their
completion is verified. History lives in git (the 2026-09-13 log has the rerun,
kernel-fix, and cleanup record).

## Standing decisions

- EVERY EC2 launch targets the loaner droplets
  (`scratch/droplets/launch_on_droplets.py`, at most six instances per droplet).
  Never launch untargeted on-demand or spot instances, for anything. The JSS
  cloud tool is spot-only and is not used until it targets droplets. Reboot risk
  is covered by per-dataset checkpoints. As of 2026-09-13 only droplet
  29.81.7.184 accepts launches; the other three return
  InsufficientInstanceCapacity (raised with the EC2 team).
- Timing numbers in either paper come only from c6a.8xlarge runs inside the
  pinned container built from d6c0696 or later (the adaptive splitter kernel
  fix); laptop times never enter a manuscript.
- Timing and performance work uses the linear selectors (`mc`, `pc`) only. The
  accuracy benchmark keeps its protocol-selected configurations, including RDC.
- Never launch the 28 cells in `paper/benchmark/config/locked_cells.csv`
  (RDC-selector CIT/CIF on isolet, gisette, letter). They terminate the EC2
  c6a.8xlarge host; AWS reproduced the failure on 2026-09-13 and owns the root
  cause. Both manuscripts report them as host-fault exclusions.
- `threshold_test="maxt"` stays opt-in for this release; `bonferroni` remains
  the default and the benchmark rankings are Bonferroni rankings.
- Every experiment has a 48-hour wall-clock budget from launch; boxes
  self-terminate and whatever is incomplete is censored, not waited for.
- Commit before launching any replication run; the JSS suite aborts if HEAD
  changes mid-run.
- Campaign roles allow create-only S3 writes (`PutObject` with `If-None-Match`);
  uploads from boxes use `aws s3api put-object --if-none-match '*'`.

## Compute in flight (2026-09-13 evening)

## Review round 6 (2026-09-21): Codex gpt-6-astra, Fable 5.1, Opus 5

Reviews in `scratch/reviews/2026-09-21/`. Votes: arXiv RED / RED / YELLOW; JSS
RED / YELLOW / YELLOW. Fixed the same day:

- [x] "within 0.03 in every cell" was false (per-cell range -0.054 to +0.034);
      replaced everywhere by the pooled loss with its interval and the range.
- [x] "OOB permutation removes the high-cardinality preference" contradicted by
      the weak-signal table (RF permutation 500-level share 0.47 vs 0.38 for
      impurity): both papers now say it holds on the NHANES cohort and not on
      the weak-signal designs.
- [x] "smaller low-cardinality one" unsupported (larger in regression, and CIF
      keeps a 0.22-0.26 high-cardinality share): "of comparable size".
- [x] "behind boosting on the complete-case panel" hid XGBoost ahead of CIF on
      the main panel; abstract, contributions, conclusion corrected; regression
      ordering marked descriptive in the abstract.
- [x] Real-data Stage B count 9 of 12 (8 higher, 1 equal); deltas from the
      table; residential slowdown by stopping mode; ratio rounding (2.8-13,
      1.4-2.0, 340 permutations at 17 candidates); appendix G rounding; appendix
      E 0.88-0.99 and unrounded 68/110; max-type saving reconciled across the
      mixed runtime panel, the head-to-head grid, and the JSS cells.
- [x] Proposition: futility clause, batch-boundary rule, discrete-response
      remark, and the verified range stated; appendix D stopping paragraph made
      consistent with the remark; max-type threshold placement stated as the
      mechanism for the rank move; "valid at a fixed node for a predictor fixed
      without the response".
- [x] JSS: power wording matches the arXiv; root-level calibration budget (999)
      stated; "throughout this article" removed; letter 15 midpoints / 300
      permutations; balanced accuracy in the design paragraph; "not significant
      after Holm"; upper-tailed trend tests reconciled with the companion's
      reverse preference; NHANES screen-disabled claim narrowed to the permuted
      weight with the moving real predictors named; permutation control scoped
      to the cohort; n_jobs guidance in the computation section; decomposition
      run cross-referenced; NHANES table carries the two control columns.
- [ ] Interleaved citrees and partykit re-timing in one pass (Opus B4, Codex
      12): launched 2026-09-21 (r6-h2h-both-\*, 14 boxes); replace JSS Tables 6
      and 7 and the arXiv head-to-head sentence when in.
- [ ] Open, author decisions: tie-order share of top-k on wide datasets (Fable
      17); sample-weighted importance arm for NHANES (Opus B3); JSS shipped
      defaults; abstract length; lemma restatement with A0 hypotheses (Opus 14,
      Fable 28); per-threshold Bonferroni corollary (Opus 13); pooled-power
      interval unit (Codex 4, Opus 9: cells share draws; interval is
      between-cell); tutorial output (Fable J14).

## Review round 5 (2026-09-19): Codex gpt-6-astra, Fable 5.1, Opus 5

Reviews in `scratch/reviews/2026-09-19/`. Votes: arXiv RED / YELLOW / YELLOW
(was RED / RED / RED); JSS RED / YELLOW / YELLOW (was RED / YELLOW / YELLOW).
Convergent findings and their status:

- [x] Framing (all three): abstract, contributions, and conclusion now state the
      measured trade (level with random forests, behind boosting on the
      complete-case panel, smaller low-cardinality preference in place of CART
      bias, RF out-of-bag permutation importance as the cheap alternative,
      max-type ranks 5th and 4th, synthetic recovery in the lower half).
- [x] Power claim (Opus 25, Codex 4, Fable 24): pooled paired analysis, loss of
      0.006 with 95 percent interval -0.009 to -0.004 over cells; stated in
      abstract, results, and conclusion in place of "no detectable difference".
- [x] Proposition monotonicity (Fable 8, Opus 20): argument written; the fixed
      theta recursion in `paper/theory/adaptive_stopping_size.py` confirms g is
      nonincreasing and every significance stop reports below alpha at the
      shipped settings and the adjusted levels.
- [x] Numeric consistency: Stage B real deltas, weak-signal shares, support
      shares (13 of 16 designs at 90 to 100 percent), by-k positions, LODO drop,
      max-type run composition (1,880 less 88 exclusions), "bounded" wording in
      appendix G, appendix J timing defects, PI permutes a held-out split, 15
      clf / 16 reg selectors, 999-permutation budget for 0.0488.
- [x] Stage B study budget (Fable 1, Codex 5): the study runs at the `auto`
      budget, so adaptive stopping acts there; stated in the design paragraph,
      the fixed-budget caption, and the JSS.
- [x] 988 vs 667 s (Fable 3, Opus 8): unbounded depth in the arXiv scaling run
      against depth 3 in the JSS head-to-head; caption corrected.
- [x] JSS: abstract quotes Stage-B-only sizes and the screen-disabled control;
      behavior section leads with the scanning-on run; exhaustive row labelled
      Stage A only; no-adjustment column marked as not a valid test; tie-free
      condition restored in 2.4 and limitations; `scale_resamples_with_tests`
      documented; cross-run caveat in the body; 23.4 in the finding sentence;
      secondary contrasts labelled descriptive; largest calibration rate placed
      against the attainable 0.049.
- [x] Stage B scaling table vs fixed-budget arm (Fable 2, Opus 7): one-pass
      rerun of all three arms at five seeds (2026-09-19); both tables and the
      prose in both papers regenerated from it.
- [x] Runtime ablation tables re-measured with the second-fit protocol
      (2026-09-19/20, 48 boxes): CIT (7 variants), CIF knob, threshold search,
      auto-budget, single worker, and benchmark-size datasets; every range in
      `tab:cit-runtime-hyperparams`, `tab:cif-runtime-hyperparams`, and the
      benchmark-size sentences follows the new data (most moved by hundredths;
      the tree adaptive-off range narrowed from 0.58-1.04 to 0.80-1.01). Outputs
      archived under `reference/v3/ablations/review5/`.
- [x] RF ranked by out-of-bag permutation importance added as a benchmark
      comparator (author approval 2026-09-20; run 2026-09-20/21): 8th of 18 and
      6th of 19, below CIF; CIF becomes 4th of 18 in classification with it
      present; permutation step 43 min per fold on gisette.
      `tab:rf-oobperm-benchmark`.
- [x] Stage B real table in balanced accuracy with depth (author approval
      2026-09-20; rerun at five seeds).
- [ ] Author decisions still open: JSS shipped defaults (auto/exact/bonferroni)
      versus the recommended and the announced max-type default (Opus J6, Fable
      B2); the forest's default parallel split is slower than serial at the
      reference condition (Fable B1: 18.9 vs 13.7 s) and needs a library
      decision; add an unstandardized max-type arm or mark the standardization
      sentence as rationale (Opus 23); abstract length (Fable 31).

## Review round 4 (2026-09-17 evening): Codex gpt-6-astra, Fable 5.1, Opus 5

Reviews in `scratch/reviews/2026-09-17/`. Votes: arXiv RED / RED / RED (was RED
/ RED / YELLOW); JSS RED / YELLOW / YELLOW (was RED / YELLOW / RED). The six
control experiments were accepted as answering their questions; the remaining
consensus is framing plus a few new items.

Fixed at once (text): stale single-tree timing sentence; NHANES rank ordering
("between", not "below") and run identification; tie explanation restated as
several candidates exceeding all permutations; LODO rank movement 0.05 / 0.125
and "sensitivity" not "bound"; abstract adaptive ratios by estimator; bootstrap
standard errors for size-matched power (0.017 centred, 0.038 off-centre; none of
162 beyond 2 SE; detectable difference 0.03 / 0.08).

Consensus items still open:

- [x] arXiv thesis rewritten 2026-09-17 (abstract, contributions, conclusion
      around one question; ranking result stated positively) (all three): one
      primary contribution, abstract as motivation + result + guidance,
      conclusion answering the same questions; the headline is a
      panel-dependent, null-to-negative result and should be written as one.
- [x] Importance formula stated in Section 2 (2026-09-17); ablation still an
      author decision (Fable A23/J21, Opus B1): citrees sums node-level impurity
      decreases without the n_t/n weight scikit-learn uses; state and justify in
      Section 2 of both papers; a sample-weighted ablation needs refits (author
      decision; EC2).
- [ ] Zero-importance tie padding of top-k (Opus A1, Fable): quantify per
      dataset and method the support size (features with nonzero importance) and
      the share of the top-k prefix that is tie filler, or break ties by the
      Stage A statistic (library change) and rerun (author decision).
- [x] Adaptive-stopping overhead explained in Appendix E as kernel overhead
      (2026-09-17) (Opus A2): explain why single trees pay up to 1.56x on the
      scaling grid when the first posterior check coincides with exhaustion at
      the floor (batched kernel per-batch overhead?); reconcile with the
      0.87-0.98x ablation ratio; state the number of posterior evaluations per
      test.
- [x] Power study framing (2026-09-18) (Opus A3, Codex 5): tab:stageB-power
      relabelled as the split rate at nominal level 0.05; the size-matched table
      (tab:stageB-only) moved into the results section. The design remains
      narrow (depth-one, independent Gaussians, step effect).
- [x] Max-type positioned against maximally selected statistics with citations
      (2026-09-17); ctree arm still an author decision (Opus A4): cite Hothorn
      and Zeileis (2008) explicitly, state what is new (Monte Carlo column
      moments for a non-linear impurity statistic; BK vs K^2/alpha), and
      consider ctree's split rule as a third arm (author decision; EC2).
- [x] Adaptive-size proposition wording (2026-09-18) (Codex 3): three statements
      (continuous mixing exact; finite orbit within 1/n_t! by the monotone grid
      average; ties at most), one strict rejection convention with the
      non-strict values as a remark; JSS restatements aligned.
- [x] Cost accounting under scanning stated (2026-09-17) (Fable A10, Codex 9):
      K^2/alpha is the unscanned exhaustive work; give the scanned expression
      and drop or qualify "the two rules cost the same at 16 bins".
- [x] Mechanism-matched check restricted to classification (2026-09-17) (Codex
      7): restrict the conclusion to classification; report regression by
      dataset.
- [ ] PI/CPI baselines (Fable A24): PI on training rows and CPI on a 20 percent
      split are handicapped by the paper's own Appendix D argument; rerun with
      out-of-bag rows or drop with a statement (author decision; EC2).
- [x] Regression means (2026-09-18) (Fable A25, Opus A9): direct-comparison
      table gains a median column and the prose leads with medians in
      regression.
- [x] Weak-signal mixed-cardinality designs and support sizes (2026-09-18,
      author request): RF impurity fills 38 percent of its top ten with
      500-level noise at the weakest signal, CIF 31-40 percent with binary noise
      (reverse bias), max-type cuts that to 26-30; RF recovers more informative
      features at weak signal (0.42 vs 0.26-0.34); downstream within 0.01. CIF
      support excludes 20-93 percent of features while keeping 92-100 percent of
      planted ones on p <= 120, but misses signal on hard designs. Reported as a
      bias trade in 06_boundary (`tab:weak-signal-cardinality`); abstract
      unchanged.
- [x] Threshold-location precision conditioning stated (2026-09-17) (Codex 6):
      condition on jointly detected replicates or label as detection summaries.
- [ ] JSS NHANES framing (Opus B1, Fable J1, Codex 11): organize around the five
      rankings; say what remains specific to citrees given that the gate
      contributes nothing to the ranking and OOB permutation importance fixes
      the CART forest; factor-typed partykit rerun (Opus B2; cheap, author
      decision); complete-case cohort limitation (Fable J22).
- [ ] JSS defaults vs recommended vs announced (Opus B3, Fable): author decision
      before v1.0.
- [x] JSS self-containment and length (2026-09-19) (Fable J3/J4, Opus J1): the
      three threshold-test paragraphs that restated the companion's Appendix I
      are one summary paragraph that keeps every number; Editorial Workbench
      review of both papers applied (16 arXiv and 5 JSS edits).
- [x] JSS recommended forest 21 s vs exhaustive 6.3 s (2026-09-18) (Fable J15):
      measured decomposition at the reference condition (EC2,
      `performance_decomposition.py`, `tab:performance-decomposition`): the
      threshold test is 98 percent of the forest's time (18.9 vs 0.38 s), the
      parallel forest gives each tree one thread (serial 13.7 s), and forest
      trees test weaker sampled predictors.
- [x] Equal-work rows never split (found and fixed 2026-09-18): the 2026-09-17
      run gave citrees 999 permutations against alpha/50 = 0.001, and the +1
      p-value cannot fall strictly below 1/1000, so the exhaustive citrees rows
      of JSS Table 5 timed a root test only. Budget now partykit + 1; rerun on
      the second-fit protocol: tree 0.17 s (partykit 2.9), forest 14.6 s on 32
      cores (partykit 286 one core, 13 on 32). Table 5 and text rewritten.
- [x] JSS "no adjustment" speed option without its validity cost (2026-09-18)
      (Fable J16): Stage-B-only size at nominal 0.05 is 0.15-0.31 (clf) and
      0.21-0.38 (reg), rising with candidates; complete-node rate 0.036-0.049,
      held by the Stage A gate (`paper_noadjust_stageb.csv`,
      `paper_noadjust_calibration.csv`). Stated in the runtime section.
- [ ] Timing protocol paid JIT compilation inside the timed region (found
      2026-09-18): the JSS performance study's tiny warm-up and the ablation
      `warmup_jit` do not reach every kernel the measured fit uses, so the first
      cell per kernel per host included compilation (same reference tree 4.2,
      2.6, or 0.19 s by shard position; scaling-curve first repeats 5.1 vs 0.7
      s). Fixed in `performance.py` (identical full fit before the timed fit)
      and `scaling_curves.py`. Done: citrees equal-work and recommended rows
      (JSS Table 5), arXiv appendix E `tab:scaling-curves` and its paragraph
      (tree about linear in n, adaptive overhead 1.02-1.28); partykit exhaustive
      rows (unchanged); JSS Tables 6 and 7 from `head_to_head_timing.py` on the
      second-fit protocol 2026-09-19 (`paper_h2h_secondfit_citrees.csv`): cells
      under about 5 s were compile-dominated (no-adjustment 2.5 to 0.17 s,
      max-type 2.7 to 0.43 s at n = 500), recommended rows moved at most 20
      percent; prose ratios rewritten in both papers. Still to check: the
      CIT/CIF runtime ablation tables share `warmup_jit`; their ratios are
      medians over many datasets fit in one process, so only the first cells per
      kernel are exposed.

## Review round 3 (2026-09-17): Codex gpt-6-astra, Fable 5.1, Opus 5

Reviews in `scratch/reviews/2026-09-15/`. Votes: arXiv RED / RED / YELLOW (was
RED / RED / RED); JSS RED / YELLOW / RED (was RED / YELLOW / YELLOW). Items
raised by two or more models first; each verified against the sources where it
is checkable. Author decisions are marked.

Verified contradictions and errors (text fixes):

- [ ] Max-type rerun paragraph says "complete-case panels" while its caption
      says the 21/8 all-method panels (Fable, Opus). Text is wrong.
- [ ] Appendix E scaling table: the Adaptive column exceeds Full for single
      trees in most cells (2.5 vs 1.6 s) while the prose says "0.64-1.06x" and
      the abstract says inert; the ratio is inverted and the tree overhead
      (posterior checks at the minimum budget) is real (Fable, Opus). Recompute
      as Adaptive/Full, state the overhead, reconcile with the ablation tables.
- [ ] Abstract attributes the Friedman p = 0.07 to the 8-dataset regression
      panel; the test is on the 6-dataset complete-case panel and covers all 18
      methods, not "the top half" (Codex, Fable).
- [ ] Panel table calls the benchmark-size panel "at least 500 observations, 9
      clf / 5 reg"; the inventory has 10 clf / 3 reg above 500. The panel is the
      JSS performance dataset list (9 clf without madelon; 5 reg including
      imports-85 and residential below 500). Rename the panel (Fable).
- [ ] "Every single-configuration family keeps its fixed rank" is false: Boruta,
      CatBoost, ExtraTrees, cforest move by 0.05 (Codex). Say scores are fixed
      and ranks move only through CIF's reselection.
- [x] JSS "at identical exhaustive work partykit is faster": fixed by the
      equal-work rerun (see below). Was false: With adjust_alpha_selector=True
      and an integer budget the tree multiplies the budget by the predictor
      count (`_bonferroni_correction`: n_resamples \* n_tests), so citrees ran
      999 x p permutations per predictor against partykit's 999 (Codex). Rewrite
      as a procedure comparison with configured and realized work, or rerun the
      exhaustive arm with the adjustment off (author decision; the 880-cell
      campaign).
- [ ] JSS NHANES premise "citrees and scikit-learn use the same
      impurity-decrease importance" is false: citrees sums node-level impurity
      decreases unweighted by node size (`_tree.py` `_node_impurity` and
      `feature_importances_[best_feature] += impurity_decrease`); scikit-learn
      weights by node sample mass (Codex). The split-search attribution needs a
      common importance formula or the control arm below.
- [ ] Cross-paper timing: CIF 32,000 x 20 is 1,024 s in arXiv Appendix E and 672
      s in the JSS scaling table (Opus). State the configuration difference
      (planted-signal data and run) or use one number.
- [ ] JSS "adaptive stopping removes that axis" contradicts the minimum-budget
      inertness; attribute the saving to the budget rule and scanning (Codex).
- [ ] Lemma/theorem/"proves" wording inconsistent across intro, theory,
      appendices A and C, conclusion; label `thm:plusone-superuniform` on a
      lemma (Fable, Opus).
- [ ] Post-hoc pairwise intervals precede the omnibus statistics and carry no
      multiplicity adjustment; regression omnibus does not reject so no post-hoc
      is licensed there (Codex, Fable, Opus).
- [ ] Regression means on the R^2 scale are dominated by the three coepra sets
      (n < 140, p > 5,000, negative R^2); make medians primary and demote "CIF
      2nd" in the abstract (Fable, Opus, Codex).

Framing (text, larger):

- [ ] Contribution hierarchy: benchmark first, max-type as an implementation
      improvement, theory as standard and cited; abstract as motivation +
      result + guidance rather than eleven numbers; the headline rank is a
      property of the 21/8 panel and reverses on the complete-case panel where
      boosting beats CIF; the abstract's "boosting pulls ahead on
      high-dimensional data" has no supporting analysis (complete-case panel is
      the p > 100 panel by construction; state that) (all three).
- [ ] Minimum-budget degeneracy must be confronted, not mentioned once: under
      `minimum` every passing p-value is 1/(B+1), so with scanning Stage A
      selects by raw statistic order and without scanning by a uniform draw; the
      benchmarked procedure is a gated max-statistic tree, which also explains
      the JSS 8-12 percent root agreement and the arbitrary top-k beyond the
      split set (Fable, Opus). Section 2 and the discussion.
- [ ] Adaptive-size proposition: "exact" only under the continuous-mixing
      idealization; the finite permutation orbit is discrete (Codex gives an
      enumeration where the true size is 0.0044 against the urn's 0.0476); state
      exact-under-idealization plus the tie bound, keep one rejection
      convention, and label the B-range as a numerical statement (all three).
- [ ] Stage B claims exceed the corollary: "valid at the node" for adaptive
      fitted decisions, "equal power at matched size" without Monte Carlo error
      (SE about 0.02 per power estimate) and with five cells matched at
      0.031-0.046; report size-power curves or descriptive size-adjusted
      comparisons with intervals and drop equivalence language (all three).
- [ ] Cost accounting: K^2/alpha is the `minimum` budget rule's cost, not
      Bonferroni's; scanning changes the operating cost; B = 999 for max-type is
      a choice. Restate contribution 2 accordingly (Codex, Fable, Opus).
- [ ] Mechanism-matched check: restrict the conclusion to classification; the
      regression mean moved from +0.021 to -0.115 and naming coepra2 does not
      erase it (Codex).
- [ ] Feature-sampling section: CIF-all effect is the mtry combinatorial floor
      (P(informative in 32 of 1,000) about 6 percent) and a total configuration
      effect; split-share denominators confounded by tree size (Fable, Opus,
      Codex).
- [ ] JSS duplication of the companion's Stage B section (Fable, Opus).
- [ ] JSS defaults: shipped `auto` + exact + bonferroni vs recommended
      `minimum` + histogram-256 vs announced future `maxt`; announced default
      contradicts the rerun evidence; `simple` stopping still shipped (Fable,
      Opus). Author decision on defaults before v1.0.
- [ ] JSS root agreement 8/12 percent under a "matched behavior" heading; report
      honestly and explain the tie mechanism (Fable, Opus, Codex).
- [ ] JSS calibration 0.0550 exceeds the attainable 0.049; compare intervals to
      0.049 and say it is Monte Carlo error over 28 conditions (Fable, Opus).
- [ ] JSS NHANES secondary contrasts use the corrected-CV t the paper calls a
      heuristic; use the repeat-level bootstrap for all three (Codex, Opus).
- [ ] "RDC" is the max-projection variant, not the canonical-correlation RDC;
      rename or say so once in each paper (Opus).

Experiments the reviewers ask for (author decision; each is cheap on EC2):

- [x] Per-threshold Bonferroni at fixed B = 999 (2026-09-17): matches the scaled
      rule at 16 bins; at 64 and 256 bins it never splits (size 0, power 0,
      depth 0) while costing 3.7 s vs 90 s (scaled) and 6.5 s (max-type) at n =
      16,000; the adaptive path raises any budget to the floor so only the
      exhaustive column is a fixed budget. Reported in Section 5, Appendix I
      (`tab:stageB-fixed-budget`), contribution 2, and the JSS threshold-test
      subsection.
- [x] NHANES controls (2026-09-17): the same RF ranked by out-of-bag permutation
      importance drops the shuffled weight from rank 12.1 to 25.5 (below cif
      23.0, cforest 26.4); cif with the screen off is unchanged (23.1); redraw
      per repeat changes nothing (primary contrast -9.8 vs -11.0). The effect is
      impurity importance under greedy split search, and the conditional forests
      avoid it by choosing the split variable by the association statistic, not
      by the alpha gate. JSS design, finding, and abstract rewritten;
      `paper_nhanes_controls.csv`.
- [ ] JSS root-agreement positive control with feature scanning on (Fable, Opus
      J4).
- [x] CIF-all top-k recovery: launched 2026-09-17 and terminated by author
      decision before completion. CIF-all at p = 1,000 evaluates every feature
      at every node and is not a configuration anyone runs; the mechanism
      paragraph states the mtry combinatorial floor and the total configuration
      effect instead. Module `cif_all_recovery.py` kept.
- [x] Power study (2026-09-17): fourth seed at 1,500/1,000 replicates gives a
      standard error of about 0.015 per size-matched difference (mean -0.005, 1
      of 54 beyond 2 SE so far; n = 200/500 classification cells still running);
      85th-percentile step: mean -0.004, SE 0.034, none beyond 2 SE. 'Equal
      power' replaced by 'no detectable difference' in both papers.
- [x] JSS exhaustive arm at equal work (2026-09-17, 88 shards): at 999
      permutations per predictor in both libraries citrees is faster (tree 1.6
      vs 2.9 s; forest 6.3 s on 32 cores vs 286 s one-core and 13 s 32-core
      partykit); the old 706 s charged citrees 999 x 50. Table rows, caption,
      and paragraph rewritten.

## Review round 2 (2026-09-14): Codex gpt-6-astra, Fable 5.1, Opus 5

Reviews in `scratch/reviews/2026-09-14/`. Votes: arXiv RED / RED / RED; JSS RED
/ YELLOW / YELLOW. Consensus blockers, verified where checkable:

- [x] Stage B calibration isolated (2026-09-15): the Stage-B-only study (one
      predictor fixed without Y, Stage A gate off, 3 seeds, n x bins grid,
      nominal levels 0.02-0.50) gives size 0.000-0.027 (Bonferroni) and
      0.023-0.044 (max-type) at nominal 0.05; both papers say the complete-node
      rate is Stage A dominated and report these. Table `tab:stageB-only`.
- [x] Power size-matched (2026-09-15): max-type minus Bonferroni power at
      matched realized size is -0.054 to +0.034, mean -0.006, no bin trend; the
      nominal-level gap (about 0.10 in isolation) is a size gap. "More powerful"
      removed from both papers; the claim is now equal power at matched size and
      calibration closer to the nominal level. Open from the same reviews:
      correct-feature power alongside any-split power and localization on paired
      detections are still the five-seed numbers (87 vs 90 percent on the
      planted predictor).
- [x] Adaptive-size proposition fixed (28c9eff): module uses the tree's strict
      rejection; size 0.0488 at B = 999, at most 0.0496 for every B in 20-3,000
      (non-strict fails at B = 79); continuous-mixing note. Open: tabulate rho
      at the adjusted alphas the tree uses (Fable).
- [x] Narrative: abstract states the bunched ranking; +1 result a cited lemma;
      version note in Appendix J; discussion trimmed; contributions rewritten as
      three (benchmark, Stage B cost account + max-type, exact stopping size).
      Section 2 now frames the max-type rule as the cost fix (2026-09-15).
- [x] Configuration selection (2026-09-15): LODO run on the all-method panels
      (21/8) with `--panel all-method`; single-configuration families keep their
      fixed ranks exactly; CIF 5.86 (tied 3rd) to 6.10 (5th) in classification
      (20 of 21 reselect, balanced accuracy -0.002) and 6.25 to 6.50 (2nd) in
      regression (7 of 8, R^2 -0.008). Reported in Appendix G
      (`tab:lodo-config-sensitivity-allmethod`), Section 5, the discussion. The
      App. D caption already states the real-data selection rule.
- [x] cforest importance (closed 2026-09-17): the "nperm = 1 handicap" was
      rejected (partykit default; JSS uses ten only where importance variance is
      itself measured). The real confound, importance mechanism, was tested: CIF
      refit per Stage 1 fold and ranked by cforest's own out-of-bag permutation
      importance scores +0.012 mean / +0.006 median balanced accuracy over
      cforest (15 of 20) against +0.011 for split-ranked CIF; the two CIF
      rankings differ by +0.001; regression medians agree, means are a coepra2
      artifact. Appendix D `tab:cif-perm-check`, Section 4 and discussion
      sentences updated. Training-row permutation importance (sklearn default)
      is the wrong comparator (-0.10 to -0.12); kept as a secondary row.
- [x] Numeric inconsistencies (checked 2026-09-15): the arXiv items were either
      already fixed or misread by the reviewers (the abstract's 0.009 is the
      exact-search row, not the bootstrap row; the CIF-ablation prose matches
      the table row by row; Appendix E says adaptive stopping changes neither
      curve; the cardinality-table caption defines the 2,790 training-fold count
      against the 3,464 cohort count). Real errors fixed: JSS scaling ratios
      recomputed from the table (6.6-17x slower than 32-core partykit, 2.9-8.8x
      faster without the adjustment, max-type 4.4-32x faster than recommended).
- [x] JSS duplication and application (2026-09-15): three duplicated Stage B
      tables removed, one calibration table kept with a citation; NHANES
      reframed to split search with the repeat-level bootstrap as the primary
      contrast; chance baseline for root agreement; parameter table added;
      timing table caption states all columns come from one run; simple stopping
      warns at construction and the parameter table says so.
- [x] Minimum-budget degeneracy: stated in Section 2 (every rejecting candidate
      returns 1/(B+1), so the smallest-p rule is a uniform draw among passers
      without scanning) with the reverse cardinality bias as the max-type
      motivation.
- [x] Missing baselines and checks: stated as limitations in 06_discussion
      (Lasso and mutual-information filters, a tree-based downstream learner,
      the CPI caveat, RF-RFE protocol, post-hoc pairwise intervals after the
      omnibus, CIF-all recovery columns). No new comparators for v3.
- Verified non-issue: Fable's "identical power triplets" (clf delta 0.10 vs reg
  delta 0.4 at n = 100) differ at the fourth decimal and across seeds; a
  rounding coincidence under shared seeds, worth a footnote at most.

## Manuscript work

Done 2026-09-13: Stage B section (calibration, cost, real subset, power) in both
papers from the five-seed post-fix studies; arXiv runtime section, abstract,
experiments paragraph, and discussion from the decoupled post-fix ablations; JSS
performance real and scaling tables from the head-to-head rerun; discussion
validity scope; speed-mechanism wording; same-procedure wording; 1,200 s cap;
locked cells as host-fault exclusions.

Remaining Codex review items (`scratch/reviews/codex_review.md`, verified before
editing):

- [x] JSS `tab:performance-real` prose states where the max-type test is slower
      (residential, about 10 percent) and equal (imports-85).

Other manuscript items:

- [ ] Cite the JSS article for the full Stage B tables once it has an
      identifier; until then Appendix I carries them.
- [ ] Move the arXiv version note out of the introduction body before any
      journal submission.
- [ ] JSS author list (three on JSS, four on arXiv).
- [ ] Bump the package version and tag the release the JSS article describes;
      the changelog has an Unreleased section.
- [ ] When AWS reports the host-fault root cause, record it and decide whether
      the locked cells are rerun or stay excluded.

## Reference release and remaining cleanup

Done 2026-09-13: local `../data/merged-v2` and 14 unreferenced ablation CSVs
deleted; working bucket `debug/` junk, smoke/sev2/gate prefixes, 20
non-canonical benchmark campaigns, `jss/replication/dgrp` and four pre-fix JSS
campaigns deleted (bucket now ~2 GB); `reference/v3/` populated with datasets,
the canonical benchmark campaign d805868f, the max-type extension, manifests and
receipts, the post-fix ablations, the head-to-head and Stage B studies, the
locked-cell list, and a `superseded/` folder with the pre-fix Stage B and
runtime runs.

- [ ] `reference/v3/README.md` uploaded; extend it when the JSS performance
      campaign and the in-flight ablations land, and copy those in.
- [ ] Delete the superseded working prefixes once the manuscripts are final:
      `repairs/runtime-ablation-rerun/source-d3c0edf6…` (pre-fix timings),
      `repairs/h2h-maxt`, `debug/maxt-timing`.
- [ ] Other accounts: `citrees-891377167619` (1.8 GB) and `citrees-619322353947`
      (2.2 GB) hold the pre-rename benchmark layout; delete after confirming the
      canonical campaign reproduces every cell they hold. `citrees-837116549485`
      (23 GB personal: deliveries, backups, snapshots) is the author's;
      `scratch/personal-data-backup` holds the manifests of its `backups/`
      uploads and stays until that bucket is decided.
- [ ] `scratch/maxt-campaign` (59 MB attempt files): delete after the alias
      fold-back migration.
- [ ] `../data/rankings`, `metrics`, `data` (400 MB): the local mirror of the
      canonical benchmark; record the campaign it mirrors in
      `paper/results/README.md`.

## Code debt (small; after the in-flight runs land)

- [ ] `paper/analysis/README.md` maps every builder to its output and consumer;
      delete `build_cit_runtime_ablation_summary_tables.py` after v3 (superseded
      by the experiment module's `summarize`).
- [ ] Parameter tables triplicated (README, docs/parameters.md, AGENTS.md); keep
      docs/parameters.md as the source and shorten the others.
- [ ] Make forest importances independent of `n_jobs` at the tie level, or
      document that they are not.

## Deferred migrations

- [ ] Fold `cit_maxt`/`cif_maxt` back into `cit`/`cif` under a single
      `threshold_test` grid axis, only after both submissions. Steps: add
      `threshold_test=["bonferroni"]` to `_CIT_CIF_BASE`; a maintenance rename
      script mapping each old label to its new label for ranking and metric
      files with a dry-run count diff; remap `method_id` in the evaluation
      parquets and summary CSVs; regenerate `../data/grid_truth.json`; reconcile
      S3; merge the alias artifacts and delete the alias names,
      `BASE_METHOD_ALIASES`, and `maxt_extension_exclusions.csv`; cell-count
      equality check.
