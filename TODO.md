# TODO

Open work before the arXiv v3 and JSS submissions. Items are deleted when their
completion is verified. History lives in git (the 2026-09-13 log has the rerun,
kernel-fix, and cleanup record).

## Standing decisions

- EVERY EC2 launch targets the loaner droplets (`launch_on_droplets.py`,
  archived under `reference/v3/provenance/launch-scripts/droplets/` in S3, at
  most six instances per droplet). Never launch untargeted on-demand or spot
  instances, for anything. The JSS cloud tool is spot-only and is not used until
  it targets droplets. Reboot risk is covered by per-dataset checkpoints. As of
  2026-09-13 only droplet 29.81.7.184 accepts launches; the other three return
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

## Open before submission

Review rounds 2 to 6 (reviews under `reference/v3/provenance/reviews/` in S3)
were closed on 2026-09-22 after each item was checked against the current
manuscripts; the record is in git. What remains:

- [ ] JSS tutorial forest and search numbers (100 trees, 5 folds: 0.9850,
      0.9985, OOB 0.9507, CV 0.9402) have no archived source; the reference
      release holds only the quick profile. Run the full-profile replication on
      EC2 after the last text commit and check them.
- [ ] JSS NHANES partykit rerun with factor-typed categorical predictors
      (reviewer request). The `cforest_factor` arm is in the replication script
      (`CITREES_NHANES_FACTORS=1`); run the full profile on EC2 and report the
      arm next to the numeric-coded partykit ranks.
- [ ] JSS author list: three authors on JSS, four on arXiv.
- [ ] Cite the JSS article for the full Stage B tables once it has an
      identifier; until then Appendix I carries them.
- [ ] Bump the package version and tag the release the JSS article describes;
      the changelog has an Unreleased section.
- [ ] When AWS reports the host-fault root cause, record it and decide whether
      the locked cells are rerun or stay excluded.

## Reference release and remaining cleanup

- [ ] Check that `reference/v3/README.md` lists the round-5 and round-6 runs
      (JSS performance, head-to-head one-pass, Stage B replicates) and copy in
      any that are missing.
- [ ] Delete the superseded working prefixes once the manuscripts are final:
      `repairs/runtime-ablation-rerun/` (240 objects, pre-fix timings),
      `repairs/h2h-maxt/` (7), `debug/maxt-timing/` (35).
- [ ] Other accounts: `citrees-891377167619` (1.8 GB) and `citrees-619322353947`
      (2.2 GB) hold the pre-rename benchmark layout; delete after confirming the
      canonical campaign reproduces every cell they hold. `citrees-837116549485`
      (23 GB personal: deliveries, backups, snapshots) is the author's;
      `reference/v3/provenance/personal-data-backup/` (S3) holds the manifests
      of its `backups/` uploads and stays until that bucket is decided.
- [ ] `../data/rankings`, `metrics`, `data` (400 MB): the local mirror of the
      canonical benchmark; record the campaign it mirrors (d805868f) in
      `paper/results/README.md`.

## Code debt

- [ ] Delete `build_cit_runtime_ablation_summary_tables.py` after v3 (superseded
      by the experiment module's `summarize`).
- [ ] Parameter tables in README and docs/parameters.md; keep docs/parameters.md
      as the source and shorten the README table.
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
