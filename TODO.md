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
- [ ] JSS "at identical exhaustive work partykit is faster": false. With
      adjust_alpha_selector=True and an integer budget the tree multiplies the
      budget by the predictor count (`_bonferroni_correction`: n_resamples \*
      n_tests), so citrees ran 999 x p permutations per predictor against
      partykit's 999 (Codex). Rewrite as a procedure comparison with configured
      and realized work, or rerun the exhaustive arm with the adjustment off
      (author decision; the 880-cell campaign).
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

- [ ] Per-threshold Bonferroni at fixed B = 999 as a third arm of the Stage B
      scaling and head-to-head timing (Opus A5): isolates the budget rule from
      the test.
- [ ] NHANES: scikit-learn RF ranked by out-of-bag permutation importance and
      citrees with alpha_selector = 1 (screen off) as within-package control
      (Opus J8); redraw the noise controls per repeat (Fable).
- [ ] JSS root-agreement positive control with feature scanning on (Fable, Opus
      J4).
- [ ] CIF-all top-k recovery on the sparse high-p grid (Opus A10).
- [ ] Power study: more replicates for the size-matched cells and an off-center
      step (Fable, Opus).
- [ ] JSS exhaustive arm at equal work (adjustment off), or text only (Codex).

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
