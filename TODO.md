# TODO

Open work before the arXiv v3 and JSS submissions. Items are deleted when their
completion is verified. History lives in git.

## Standing decisions

- Timing numbers in either paper come only from c6a.8xlarge runs inside the
  pinned container; laptop times never enter a manuscript.
- Timing and performance work uses the linear selectors (`mc`, `pc`) only. The
  accuracy benchmark keeps its protocol-selected configurations, including RDC.
- Never launch isolet or gisette RDC cells; they hard-lock 32-worker hosts and
  are recorded as operational exclusions.
- `threshold_test="maxt"` stays opt-in for this release; `bonferroni` remains
  the default and the benchmark rankings are Bonferroni rankings.
- Every experiment has a 48-hour wall-clock budget from launch; boxes
  self-terminate and whatever is incomplete is censored, not waited for.
- Commit before launching any replication run; the JSS suite aborts if HEAD
  changes mid-run.

## Compute reruns (loaner account, targeted droplets)

The four droplets are 30.99.51.144, 29.81.7.184, 30.185.247.177, and
17.123.20.180, at most six instances each. Launch recipe:
`aws ec2 run-instances --additional-info target-droplet=<IP>` with the current
ECR image digest, instance profile
`citrees-campaign-2815f0756bc4d66f30fa794d53b9b854`, and output prefix
`repairs/runtime-ablation-rerun/source-<HEAD>`. Templates live in
`scratch/rerun-2026-09-11/`.

- [ ] Rerun `paper.benchmark.experiments.mirrored_knob_ablation` (one box). The
      recorded CIF rows predate the forest thread-pool fix and are invalid.
- [ ] Rerun `paper.benchmark.experiments.threshold_search_ablation` (one box).
      Same reason.
- [ ] Rebuild `paper_presentation_practical_controls_summary.csv` with
      `paper/analysis/build_manuscript_summary_tables.py`, then rewrite the CIF
      runtime table, its caption, the Section 5.2 paragraph, and the three
      abstract figures in the arXiv manuscript from the new numbers.
- [ ] Head-to-head forest timing: promote `scratch/h2h/h2h_maxt.py` into
      `paper/jss/replication` as a receipted component, then run it to fill the
      isolet, gisette, and 32,000-observation cells that the JSS tables mark
      n/r.
- [ ] Decide whether to rerun the ranking benchmark under `maxt` for the CIF and
      CIT `mc`/`pc` configurations (640 Stage 1 jobs, roughly 1,000 to 3,500
      box-hours, wall-clock floor about two days). If yes: add a
      `threshold_test` axis to `CLF_CIF_GRID`, `CLF_CIT_GRID`, `REG_CIF_GRID`,
      and `REG_CIT_GRID`, rebuild the image, launch the API server plus twenty
      workers, and add a max-type rankings subsection to both papers. If no:
      both papers keep the sentence that the effect on rankings is unmeasured.

## arXiv manuscript

- [ ] Replace the CIF runtime numbers (blocked on the reruns above).
- [ ] Cite the JSS article for the full Stage B tables once it has an
      identifier; until then the appendix carries them.
- [ ] Move the version note out of the introduction body before any journal
      submission of this manuscript.

## JSS manuscript

- [ ] Confirm the author list (three on JSS, four on arXiv).
- [ ] Multi-seed rerun of the threshold-test calibration study if the reviewers
      ask for tighter intervals; one seed at present.
- [ ] Bump the package version and tag the release the article describes; the
      changelog has an Unreleased section for the max-type test and the
      threading fixes.

## Repository

- [ ] Promote or delete `scratch/jss_batch_driver.py`; it drove the JSS
      performance shards and is not tracked.
- [ ] Decide the fate of the two completed-migration scripts in
      `paper/maintenance/` and their test in `tests/paper/test_grid.py`.
- [ ] Decide whether `../data/merged-v2` (2 GB, unreferenced) and the stale
      Bayesian-bootstrap CIF hashes in `../data` are deleted.
- [ ] Open `scratch/personal-data-backup` and decide whether it is kept.
- [ ] Make forest importances independent of `n_jobs` at the tie level, or
      document that they are not.
