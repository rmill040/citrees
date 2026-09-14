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

- [ ] Wide-dataset runtime ablation (`paper_*` benchmark datasets, CIT and CIF,
      six knobs; image built from 4e60fbe): shards 0-2 done or running, shards
      3-6 queued in `scratch/droplets/wide-runtime/`. Outputs
      `repairs/runtime-ablation-rerun/source-4e60fbe7…/cit_cif_runtime_ablation/     wide_shard{i}_raw.csv`.
      Summarize into `paper/results/tables/wide_runtime_ablation_summary.csv`;
      then add the large-dataset rows or a sentence to the arXiv runtime
      section.
- [ ] Scaling curves (`paper.benchmark.experiments.scaling_curves`, 20 cells,
      CIT and CIF, adaptive vs full budget; queued in
      `scratch/droplets/scaling-curves/`). Assemble with `--assemble` into
      `paper/results/tables/scaling_curves.csv`; then a figure or table for
      Appendix E.
- [ ] JSS performance campaign (full profile, 88 shards, spot, launched before
      the droplets-only rule): 86 of 88 shards archived; shards 77 and 87 were
      lost when the last spot instances ended without archiving, and the cloud
      tool refuses to materialize an incomplete campaign. The tool is spot-only
      at every layer (campaign, launch record, worker runtime), so the two
      shards cannot be finished on the droplets without changing it. The JSS
      reference-condition table (`tab:performance-reference`) therefore still
      carries PRE-FIX single-tree timings. Options for the author: (a) permit
      two spot instances to finish shards 77 and 87 and materialize; (b) run the
      reference-condition cells on a droplet through a scratch protocol in the
      head-to-head style and refill the table from that run, marking it as such;
      (c) teach the cloud tool an on-demand droplet market (code change to the
      provenance contract). Until one is chosen the table's caption must say the
      single-tree rows predate the kernel fix.

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
- [ ] arXiv rank/k consistency: report mean rank and ordinal position together;
      k-trajectory on a constant dataset panel.
- [ ] arXiv presentation: enlarge the 8-point ranking table and the
      recovery-figure legends; aggregation unit and uncertainty in captions.

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
- [ ] Promote the head-to-head timing protocol (`scratch/h2h/h2h_maxt.py`) into
      `paper/jss/replication` as a receipted component, or document it as a
      scratch protocol in the JSS replication README.
- [ ] Promote or delete `scratch/jss_batch_driver.py`.
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
