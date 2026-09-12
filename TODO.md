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
17.123.20.180, at most six instances each, all in us-east-1f. Campaign roles
allow create-only S3 writes (PutObject with If-None-Match), so every upload from
a box must use `aws s3api put-object --if-none-match '*'` or boto3 with
`IfNoneMatch="*"`; `aws s3 sync` and `aws s3 cp` are denied. Templates live in
`scratch/rerun-2026-09-11/`.

- [x] `threshold_search_ablation` running on i-0a920be7c2c2fb135 since
      2026-09-11 18:03 UTC (image 50e2e6d2). Its uploads were failing until an
      inline write policy was added to its role at 18:49 UTC; the periodic and
      final uploads now succeed.
- [x] `mirrored_knob_ablation` (with `cif_maxt`) relaunched 2026-09-11 ~19:50
      UTC on droplet 29.81.7.184 as i-0a0803712c76cb728 from image
      sha256:5609d400 (built at c9855d0); output prefix
      `repairs/runtime-ablation-rerun/source-6444421c…` names the HEAD at
      launch, whose experiment code is identical to c9855d0 (only gate tooling
      differed).
- [ ] Max-type ranking extension campaign RUNNING since 2026-09-11 ~20:20 UTC.
      Image sha256:5609d400 (c9855d0); runtime contract 79153d56…; manifest
      7a466b00…; campaign 40f9761d…; GO receipt 66e1b1b9…; artifact prefix
      `repairs/maxt-extension/source-c9855d09…/campaign-40f9761d…`. API
      i-05048dcdb225d67cf (http://32.197.212.208:8000/status). Workers: launch
      maxt-ext-workers-001 (4 on droplet 29.81.7.184; the other droplets had no
      capacity) and maxt-ext-workers-002 (12 untargeted on-demand in 1b/1c/1d).
      Stage 1 queue at launch: 1,152 classification + 640 regression cells.
      Incident 2026-09-12 00:4x UTC: worker i-0ab21726a67f5a77e died without a
      traceback on classification/letter cif_maxt (rdc, honesty) seed0 during
      fold 2, likely memory pressure from 32 parallel RDC forests on 20,000
      rows; replaced by maxt-ext-workers-003. The monitor now replaces dead
      workers automatically; the API's three-attempt cap censors any cell that
      keeps killing workers. 29 letter/isolet RDC cells remain in the queue.
      Incident 2026-09-12 ~04:00 UTC: all four droplet-hosted workers (launch
      001, droplet 29.81.7.184, us-east-1f) terminated together mid-cell on
      unrelated cells; the loaner host itself went away. Replaced by
      maxt-ext-workers-004 (untargeted, 1b/1c/1d). The fleet is now entirely
      untargeted on-demand; droplets are not used for replacements. Next: when
      rankings drain, `terminate-workers`, then relaunch the API and workers
      with `--stage metrics` using the same attempt files; then
      `manifest reconcile`. Attempt files:
      `scratch/maxt-campaign/attempt-c9855d0/`; step script
      `scratch/maxt-campaign/run_campaign.sh`.
- [ ] Rebuild `paper_presentation_practical_controls_summary.csv` with
      `paper/analysis/build_manuscript_summary_tables.py`, then rewrite the CIF
      runtime table, its caption, the Section 5.2 paragraph, and the three
      abstract figures in the arXiv manuscript from the new numbers.
- [ ] Head-to-head forest timing: promote `scratch/h2h/h2h_maxt.py` into
      `paper/jss/replication` as a receipted component, then run it to fill the
      isolet, gisette, and 32,000-observation cells that the JSS tables mark
      n/r.
- [ ] After the extension campaign lands: aggregate `cit_maxt`/`cif_maxt`
      through the benchmark analysis (display names in `paper/analysis/*.py`
      currently know only `cit`/`cif`), add a max-type ranking subsection to the
      arXiv results, cite it from JSS, and delete the "effect on rankings is
      unmeasured" sentence from both papers.

## Deferred migrations

- [ ] Fold `cit_maxt`/`cif_maxt` back into `cit`/`cif` under a single
      `threshold_test` grid axis. Only after both submissions, when the
      benchmark is frozen. Steps: add `threshold_test=["bonferroni"]` to
      `_CIT_CIF_BASE`; write a maintenance rename script in the style of the
      March 2026 bootstrap rename mapping each old label to its new label for
      the ranking and metric files, with a dry-run count diff; remap `method_id`
      in the evaluation parquets and every summary CSV; regenerate
      `../data/grid_truth.json`; reconcile S3 to the new labels, which also
      retires the CIF bootstrap-era hashes still in S3; merge the alias
      artifacts under `threshold_test="maxt"` and delete the alias names,
      `BASE_METHOD_ALIASES`, and `maxt_extension_exclusions.csv`; finish with a
      cell-count equality check. The aliases exist because adding a key to the
      shared grid rehashes every completed CIT/CIF identity.

## Codex review findings (gpt-6-astra, 2026-09-11)

Full review in `scratch/reviews/codex_review.md`. Each fix is proposed to the
author one at a time before editing; none is applied from the review alone
without independent verification. Verified so far: JSS default-vs-recommended
config (constructor defaults are auto/exact/none, timing uses
minimum/histogram/256) and the Table 9 "0.9x shorter" self-contradiction.

Top blockers:

- [ ] Scope Stage B validity correctly in both papers (arXiv
      `06_discussion.tex:30`, `03_theory.tex:68`; JSS `article.tex:166`): the
      theorem needs a response-independent feature and fixed budget; distinguish
      the fixed-feature theorem, complete-node empirical rejection rates, and
      adaptive fitted-tree behavior wherever "valid" appears.
- [ ] JSS timing claims (`article.tex:657`, `:653`): call the timed setting the
      recommended benchmark configuration, not "default"; drop "identical
      statistical work" (citrees multiplies the selector budget by predictor
      count under Bonferroni) and interpret timings as procedure costs.
- [ ] JSS Table 9 / arXiv Stage B (`article.tex:915`): report the residential
      slowdown (5.46 vs 4.78 s) explicitly; use a consistently defined runtime
      ratio instead of "0.9 to 30 times shorter".
- [ ] NHANES over-claims (`article.tex:1117`, `:35`, `:983`): remove causal
      attribution to "the importance statistic" (the forests are not the same
      procedure) and the predictive-parity claim (predictions are logistic
      regression on selected columns, not the forests).
- [ ] Disclose the modified RDC statistic in JSS (`article.tex:131`): max
      pairwise projected correlation, not canonical correlation, matching the
      arXiv disclosure.
- [ ] arXiv reproducibility: add a configuration table (selected selectors,
      honesty split, forest sizes, feature sampling, comparator grids)
      (`04_experiments.tex:144`, `appendix_D_methods.tex:70`).
- [ ] arXiv rank/k consistency (`05_results.tex:163`, `:214`): report both mean
      rank and ordinal position (classification moves tied-6th to 8th); show the
      k-trajectory on a constant dataset panel (currently 21 datasets at k=5 vs
      13 at k=100).
- [ ] arXiv synthetic recovery (`06_boundary.tex:40`): remove the "more reliable
      within top-k than at top one" sentence; regression CIF ranks 11th/10th.
- [ ] arXiv ablation narrative (`05_results.tex:111`): restrict the
      negligible-effect statement to classification; report regression's
      heterogeneous single-tree effects (-0.053, -0.064, intervals excluding
      zero).
- [ ] arXiv presentation (`05_results.tex:18`, `:223`): enlarge the 8-point
      ranking table and the recovery-figure legends; specify aggregation unit
      and uncertainty in captions.

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
