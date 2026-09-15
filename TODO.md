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

## Review round 2 (2026-09-14): Codex gpt-6-astra, Fable 5.1, Opus 5

Reviews in `scratch/reviews/2026-09-14/`. Votes: arXiv RED / RED / RED; JSS RED
/ YELLOW / YELLOW. Consensus blockers, verified where checkable:

- [ ] Stage B calibration measures the complete node (Stage A gate x Stage B),
      so it never isolates the Stage B test the corollary is about. Add a
      Stage-B-only design (feature fixed without Y, no Stage A gate) at the same
      n x bins grid; report Stage B size directly. (All three.)
- [ ] Power is not size-matched (Stage-B-only study running on the droplet, 18
      shards; first shards, seed 0 at n = 100, show both tests conservative in
      isolation at nominal 0.05 (Bonferroni 0.006-0.016, max-type 0.018-0.042)
      and the max-type power advantage shrinking to about +0.02 in
      classification and about zero in regression at a matched realized size;
      nominal grid extended to 0.35 and 0.50 for the remaining shards so
      Bonferroni reaches size 0.05): max-type's null rate is 0.028-0.038 against
      Bonferroni's 0.007-0.029, and the power gap tracks the size gap. Report
      size-adjusted power (or size/power pairs) or drop "more powerful". Also
      report correct-feature power alongside any-split power, state every
      aggregation denominator, and compare localization on paired detections.
      (All three.)
- [x] Adaptive-size proposition fixed (28c9eff): module uses the tree's strict
      rejection; size 0.0488 at B = 999, at most 0.0496 for every B in 20-3,000
      (non-strict fails at B = 79); continuous-mixing note. Open: tabulate rho
      at the adjusted alphas the tree uses (Fable).
- [x] Narrative: abstract states the bunched ranking; +1 result a cited lemma;
      version note in Appendix J; discussion trimmed; contributions rewritten as
      three (benchmark, Stage B cost account + max-type, exact stopping size).
      Open: Section 2 max-type paragraph still says the rule was added after the
      benchmark; keep, but frame it as the cost fix (done in the contributions).
- [ ] Configuration selection: rule stated inconsistently (real-data score in
      4.6 vs synthetic recovery in App. D caption); unequal grid sizes (CIF 4,
      XGBoost 5 vs 1 for RF/ET/CatBoost/RF-RFE) selected on the reporting data;
      LODO only on the 13/6 panels. Fix the caption, quantify selection
      optimism, run LODO on the 21/8 panels.
- [ ] cforest importance uses nperm = 1 (verified in config.py) while the JSS
      uses 10; the CIF-vs-cforest margin may be a handicap. Rerun cforest at
      nperm >= 10 or state the caveat.
- [ ] Numeric inconsistencies: abstract 0.009 vs table 0.012 (bootstrap row);
      "8-29x" vs 7.9-29x; "0.15 of exhaustive / 85 percent" mixes endpoints of
      different groups (report within-group ratios); CIF-ablation prose vs table
      (0.043 vs 0.053; 0.16-0.25 vs 0.157); high-p counts 7 vs 9; caption "23
      datasets (15+8; 9 real)" inconsistent; "complete-case" used for both 21/8
      and 13/6 panels; "1,792 Stage A cells" undefined; scaling table "Adaptive
      slower than Full" vs "inert"; plain accuracy in the Stage B real table vs
      balanced accuracy elsewhere; JSS "at most 1.4 s" vs pendigits 1.9; JSS "7
      to 18x" vs 6.6-17; JSS 3,464 vs 2,790 distinct values; JSS "22 and 115
      times" not derivable; AUC 0.814/0.795 aggregate undefined.
- [ ] JSS duplication: four Stage B tables and prose identical to the arXiv;
      keep one summary table and cite. NHANES contrast is split search, not the
      importance statistic (citrees also ranks by impurity decrease);
      corrected-CV t-test misapplied to training-fold ranks; nominal predictors
      passed as numeric codes; matched-behavior root agreement 8-12 percent is
      near chance. Timing table mixes two runs (needs a within-run note or a
      single campaign). Parameter reference table missing. Simple stopping ships
      invalid at 0.087 with only an "exploratory" label.
- [ ] Minimum-budget degeneracy (Fable): under `minimum`, every passing
      candidate ties at p = 1/(B+1), so "select the smallest p" is random among
      passers unless scanning orders them; state this, and note the reverse
      cardinality bias of per-threshold Bonferroni as the motivation for
      max-type.
- [ ] Missing baselines/checks flagged: Lasso or mutual-information filter; a
      tree-based downstream learner; CPI sanity check (last by a wide margin
      after a bug fix); RF-RFE protocol details; post-hoc test after the
      Friedman omnibus; CIF-all recovery columns.
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

- [ ] JSS `tab:performance-real` prose: state explicitly where the max-type test
      is slower (residential and imports-85 by under 10 percent in the Stage B
      real subset) and use one runtime-ratio definition throughout.

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
- [ ] `tools/hooks/check_no_aws_configs.py` guards a path that no longer exists;
      point it at `paper/benchmark/infra/config.yaml` or drop it.
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
