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

- [ ] JSS author list: three authors on JSS, four on arXiv.
- [ ] Cite the JSS article for the full Stage B tables once it has an
      identifier; until then Appendix I carries them.
- [ ] Bump the package version and tag the release the JSS article describes;
      the changelog has an Unreleased section.
- [ ] When AWS reports the host-fault root cause, record it and decide whether
      the locked cells are rerun or stay excluded.

## Data layout

All citrees data lives in one bucket, `citrees-837116549485`: `reference/v3/` is
the canonical release and `archive/<source bucket>/` holds verified dumps of
every earlier citrees bucket, including this bucket's own pre-2026-09-23 root
prefixes. `archive/local-data-mirror-2026-09-23/` is a copy of `../data` before
it was pruned to the on-grid configurations and the inputs the builders read.

The three source buckets were deleted on 2026-09-23 after their dumps (and the
arc bucket's noncurrent versions, under
`archive/citrees-856480643277-noncurrent/`) were verified. The
`citrees-<account>` ECR repositories in those three accounts were deleted the
same day.

- [ ] The data bucket holds 23 GB of noncurrent versions of its own
      pre-consolidation root prefixes, duplicates of
      `archive/citrees-837116549485/`; purge them once the layout is settled.

## Code debt

- [ ] Delete `build_cit_runtime_ablation_summary_tables.py` after v3 (superseded
      by the experiment module's `summarize`).

## Deferred migrations

- [ ] Fold `cit_maxt`/`cif_maxt` back into `cit`/`cif` under a single
      `threshold_test` grid axis, only after both submissions. Steps: add
      `threshold_test=["bonferroni"]` to `_CIT_CIF_BASE`; a maintenance rename
      script mapping each old label to its new label for ranking and metric
      files with a dry-run count diff; remap `method_id` in the evaluation
      parquets and summary CSVs; reconcile S3; merge the alias artifacts and
      delete the alias names, `BASE_METHOD_ALIASES`, and
      `maxt_extension_exclusions.csv`; cell-count equality check.
