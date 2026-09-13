# TODO

Open work before the arXiv v3 and JSS submissions. Items are deleted when their
completion is verified. History lives in git.

## Standing decisions

- Timing numbers in either paper come only from c6a.8xlarge runs inside the
  pinned container; laptop times never enter a manuscript.
- Timing and performance work uses the linear selectors (`mc`, `pc`) only. The
  accuracy benchmark keeps its protocol-selected configurations, including RDC.
- Never launch the 28 cells in `paper/benchmark/config/locked_cells.csv`
  (RDC-selector CIT/CIF on isolet, gisette, letter). They terminate the EC2
  c6a.8xlarge host; AWS reproduced the failure on 2026-09-13 and owns the root
  cause. Both manuscripts report them as host-fault exclusions; revisit only
  when AWS reports a fix.
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

- [x] `threshold_search_ablation` COMPLETE 2026-09-12 15:03 UTC (9.7 h on
      i-0535d285d978cb032):
      `repairs/runtime-ablation-rerun/source-6444421c…/     threshold_search_ablation/threshold_search_ablation.csv`
      (+ full stdout).
- [ ] Both ablation reruns RESTARTED 2026-09-12 ~05:20 UTC off-droplet from
      image sha256:5609d400 (create-only upload templates, profile for prefix
      `repairs/runtime-ablation-rerun/source-6444421c…`): `threshold_search`
      i-0535d285d978cb032 (1b), `mirrored_knob` (with `cif_maxt`) in 1c. The
      first runs (18:03 and 19:50 UTC on droplet 29.81.7.184) died when that
      loaner host rebooted at ~04:00 UTC; containers do not survive a reboot and
      the CSV is written only at the end, so eleven and eight hours were lost.
      Partial stdout logs are in S3 under the old prefixes with suffix
      `_droplet-reboot.stdout`. Never place multi-hour experiments on droplets.
      Since bc78401 both ablations checkpoint per dataset
      (`DATA_DIR/<experiment>.partials/`) and resume from S3 on relaunch; the
      templates take `__IMG__` and `__SHA__`. The two boxes running now predate
      that commit; any relaunch must use an image built from bc78401 or later.
- [ ] Max-type ranking extension campaign: STAGE 1 COMPLETE 2026-09-13 00:10 UTC
      (27.8 h): 1,785 of 1,792 ranking cells; 7 censored after three attempts,
      all cif_maxt with the rdc selector and honesty off (label 19b65c92…):
      gisette seed 3, isolet seeds 0, 1, 3, 4, letter seeds 2, 3. These join the
      operational exclusions. Rankings API i-05048dcdb225d67cf terminated; Stage
      2 (metrics) needs a masked manifest (metrics only where a ranking exists;
      the first metrics API refused the unmasked one): Stage 2 manifest
      `manifest-stage2/` campaign b0144d26…, manifest b1ea9063…, 1,785 required.
      Gate run for it launched 2026-09-13 00:50 UTC (arc-a i-0d03897098b5e0605,
      arc-b i-0084db401c490229f); then gate-complete,
      `manifest     materialize-rankings` from the Stage 1 prefix into
      `repairs/maxt-extension/source-c9855d09…/campaign-b0144d26…`, then
      launch-api/launch-workers with `--stage metrics`. DONE 2026-09-13 ~01:45
      UTC: Stage 2 receipt 99d6e19b…, 1,785 rankings materialized (receipt under
      `…/campaign-b0144d26…/materialization/`), metrics API i-0c5bef7bdb64a6db8
      (http://3.88.20.106:8000), 32 workers launch maxt-ext-workers-101.
      Original launch record follows. STAGE 2 COMPLETE 2026-09-13 02:30 UTC
      (about 55 min on 32 workers): 1,785 metrics artifacts, zero failures;
      metrics API terminated. Campaign compute is finished. Next is analysis:
      aggregate cit_maxt/cif_maxt from
      `…/campaign-b0144d26…/{rankings,metrics}/` through
      `paper/analysis/aggregate_pipeline_artifacts.py` and the benchmark table
      builders (which currently know only cit/cif display names), add the
      max-type ranking subsection to the arXiv results, cite it from JSS, and
      delete the "unmeasured" sentence from both papers. RUNNING since
      2026-09-11 ~20:20 UTC. Image sha256:5609d400 (c9855d0); runtime contract
      79153d56…; manifest 7a466b00…; campaign 40f9761d…; GO receipt 66e1b1b9…;
      artifact prefix
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
      untargeted on-demand; droplets are not used for replacements. 2026-09-12
      06:00 UTC: progress stalled in the heavy tail (ORL, gisette, TOX_171,
      CLL_SUB_111, ALLAML, arcene, warp\* cit_maxt rdc; coepra cit_maxt;
      letter/isolet/gisette cif_maxt rdc, which die after ~2 h per attempt);
      fleet doubled to 32 workers (maxt-ext-workers-007), projected finish about
      two days from launch. 2026-09-12 15:55 UTC: first cell censored after
      three attempts; the failing family is cif_maxt with the rdc selector and
      honesty off (label 19b65c92…) on isolet and gisette, whose containers die
      after ~2 h (memory: full-data forests, 32 parallel RDC tests); the
      honesty-on twin (e807f31a…) completes. Censored cells are listed by
      `manifest     reconcile` after Stage 1 and are reported as operational
      exclusions alongside the isolet/gisette RDC cells of the main benchmark.
      Next: when rankings drain, `terminate-workers`, then relaunch the API and
      workers with `--stage metrics` using the same attempt files; then
      `manifest reconcile`. Attempt files:
      `scratch/maxt-campaign/attempt-c9855d0/`; step script
      `scratch/maxt-campaign/run_campaign.sh`.
- [ ] Rebuild `paper_presentation_practical_controls_summary.csv` with
      `paper/analysis/build_manuscript_summary_tables.py`, then rewrite the CIF
      runtime table, its caption, the Section 5.2 paragraph, and the three
      abstract figures in the arXiv manuscript from the new numbers.
- [x] Head-to-head max-type cells DONE 2026-09-12 19:20 UTC: isolet 13.9 s,
      gisette 34.6 s, synthetic 32,000x20 32.1/33.0 s, 32,000x100 24.1/24.3 s
      (image sha256:5ffc09f5 at d3c0edf, c6a.8xlarge, scratch protocol). Filled
      into the JSS real and scaling tables and the dependent sentences;
      re-derivable from `paper/results/tables/paper_h2h_maxt_fill_cells.csv` and
      the JSONL under `repairs/h2h-maxt/source-d3c0edf6…/`. Promoting the
      protocol into `paper/jss/replication` as a receipted component remains
      open.
- [x] Max-type rankings analysed and written up 2026-09-13
      (`paper/analysis/     build_maxt_extension_tables.py`; tables
      `paper_maxt_extension_*.csv`; arXiv Table `tab:maxt-benchmark-rankings`,
      discussion, abstract; JSS threshold-test section cites the companion
      article). The "unmeasured" sentence is gone from both papers.

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

## Censored-cell diagnosis (2026-09-13)

CLOSED. The AWS EC2 team reproduced the failure (containers running the largest
RDC forests terminate after ~2 h on c6a.8xlarge, AMD EPYC 7R13, without a Python
exception) and is investigating the root cause. The AMD-vs-Intel repro boxes
(i-0a17494c2f13820ed, i-00aa9c8e4dfb9db8e) were terminated before finishing as
redundant. The 28 affected cells are listed in
`paper/benchmark/config/locked_cells.csv` (test
`tests/paper/test_locked_cells.py`) and worded as host-fault exclusions in the
arXiv manuscript (Appendix D text and table, Section 5 max-type paragraph); the
JSS article runs no RDC cells on these datasets and needs no change.

- [ ] When AWS reports the root cause, record it here and decide whether the
      locked cells are rerun (on the fixed hosts or on Intel c6i) or stay
      excluded for the submissions.

## Post-fix timing reruns (2026-09-13, image sha256:a4fc734d at e78d84a / sha256:8dba68ec at d6c0696)

Every timing table in both papers was measured before the adaptive splitter
kernel fix (d6c0696), so each is being re-measured from the fixed image. Running
since ~14:05-15:10 UTC on c6a.8xlarge:

- [ ] CIF `mirrored_knob_ablation`, 8 shards (launch_knob_shards.sh), prefix
      `repairs/runtime-ablation-rerun/source-d6c0696f…/mirrored_knob_ablation/`;
      assemble with `--assemble` from the partials as before. Replaces the 0.93x
      CIF adaptive row and every other CIF knob ratio.
- [ ] `threshold_search_ablation`, one box (i-0db774f65def12b33), same source
      prefix. Replaces the "0.9 to 30 times shorter with adaptive stopping"
      figures in the arXiv Stage B results.
- [ ] Head-to-head forest timing, 10 shards
      (`scratch/rerun-2026-09-11/     h2h-e78d84a/`, driver = h2h*maxt.py with
      isolet/gisette single-repeat), configs citrees, no-adjustment, max-type,
      partykit 1 and 32 cores, sklearn RF; 30 cells (16+4 synthetic, 10 real);
      prefix
      `repairs/h2h-rerun/source-e78d84a4…/results*{i}.json`. Replaces the JSS     performance-real and performance-scaling tables and the arXiv     head-to-head sentences (`paper_h2h_maxt_fill_cells.csv`
      becomes obsolete).
- [ ] JSS performance component, full profile (880 cells, 10 repeats), spot
      campaign of 88 performance shards, 8 instances at a time, role
      citrees-campaign-e479ceeaad5d4ac680ffaf884917e8ad; status via
      `python -m paper.jss.replication.cloud status` with the same launch
      arguments; `materialize` into a new `paper/jss/results/` directory when
      complete. Replaces the JSS reference-condition table (single trees and
      forests at 1,000 x 50, 999 permutations).

When all four land: rewrite the arXiv runtime rows/abstract figures and the JSS
performance tables and prose, then rerun the pinned-bounds test. The JSS prose
"stops each predictor's test as soon as the evidence is decisive" is wrong under
`minimum` budgets (the rule cannot stop before the floor) and is rewritten with
the tables.

## Adaptive stopping theory and scanning decoupling (2026-09-12)

Done in c0c69f4: `paper/theory/adaptive_stopping_size.py` plus test; the arXiv
proposition, remark, discussion, and method wording; the JSS scope and
discussion sentences; and the library change that makes feature and threshold
scanning independent of early stopping (default behavior unchanged).

- [ ] Decoupled `mirrored_knob_ablation` running as 8 dataset shards since
      2026-09-12 ~17:05 UTC (image sha256:5ffc09f5 at d3c0edf; 1b/1c/1d;
      i-0bbbb4140e5e9766f i-02d1db2870b022727 i-0e259ec036307515b
      i-09427eff1fea24931 i-0a80a617fcb0544b0 i-0e3ff5235dc4567e2
      i-05594e4abe871e342 i-087affa21ebf73c5d). Checkpoints land under
      `repairs/runtime-ablation-rerun/source-7a4f0122…/mirrored_knob_ablation/     mirrored_knob_ablation.partials/`
      (three datasets were already there from the single box, which was
      terminated along with the coupled run). When all 23 checkpoints exist:
      `aws s3 cp --recursive` them into
      `../data/ablation/mirrored_knob_ablation.partials/` and run
      `python -m paper.benchmark.experiments.mirrored_knob_ablation --assemble`.
      EC2 on-demand vCPU quota increase 1,152 -> 2,304 requested (pending).
- [x] Decoupled `mirrored_knob_ablation` COMPLETE 2026-09-13 ~03:10 UTC (8
      shards, 23 datasets, 276 rows; assembled from checkpoints; summary tables
      regenerated in 35027ad). Real-data median runtime ratios vs default:
      no_adaptive 0.93 (0.60-1.25), no_scan 1.31, no_threshold_scan 16.9
      (4.5-35), no_bonferroni 0.16, maxt 0.28 (0.12-1.42), all_off 0.91; by task
      group no_adaptive 0.93-1.04, feature scanning 1.08-4.18, threshold
      scanning 8.5-30, exact thresholds 1.11-21; downstream score changes at
      most 0.005. The pinned bounds in
      `tests/paper/test_paper_package.py::     test_adaptive_stopping_summary_matches_reported_bounds`
      still encode the contaminated 6.1-717x figures and move with the
      manuscript rewrite.
- [x] CIT runtime ablation rerun (decoupled knobs, image sha256:5ffc09f5 at
      d3c0edf) COMPLETE 2026-09-13 ~08:05 local: 6 dataset shards, 805 fits, 23
      datasets, all boxes terminated. Raw:
      `../data/ablation/cit_runtime_ablation_decoupled_raw.csv`; summary:
      `paper/results/tables/cit_runtime_ablation_decoupled_summary.csv`. Median
      runtime ratios vs cit_default: no_adaptive 0.12 (clf) / 0.08 (reg),
      no_feature_scan 2.1, no_threshold_scan 15, exact_thresholds 1.0,
      no_feature_mute 1.0, no_bonferroni 0.10. The no_adaptive result reverses
      the contaminated table (3-8x slower) and is an implementation effect, not
      a statistical one: under `minimum` budgets the rule cannot stop before the
      budget, and the per-threshold splitter test with adaptive stopping runs a
      pure-Python shuffle loop (`_splitter.py` around line 110) while
      `early_stopping=None` uses the Numba parallel kernel once the budget is >=
      200 (Bonferroni over 256 thresholds gives 5,120). The selector and the
      max-type splitter already have parallel batched adaptive kernels; the
      Bonferroni splitter does not. In CIF the trees saturate the cores so the
      gap vanishes (0.93x).
- [x] Fixed in d6c0696: parallel batched adaptive kernels for the four
      per-threshold splitter tests; all batched adaptive kernels (selector and
      splitter) now draw growing parallel chunks and replay the 32-permutation
      checkpoints, so stopping times, p-values, and the exact null size are
      unchanged (one chunk when the budget equals the floor). Laptop check
      (1,000 x 20, minimum budgets): adaptive 8.8 s -> 3.2 s vs 2.5 s
      exhaustive; the selector test is at parity.
- [x] CIT runtime ablation RERUN from d6c0696 (image sha256:8dba68ec…) COMPLETE
      2026-09-13 13:51 UTC: 6 shards, 805 fits, 23 datasets, ~12 min wall-clock
      per shard (the pre-fix run took ~4 h). Raw
      `../data/ablation/cit_runtime_ablation_decoupled_raw.csv` (pre-fix raw
      kept as `cit_cif_runtime_ablation_rerun_raw.csv`); summary
      `paper/results/tables/cit_runtime_ablation_decoupled_summary.csv`. Median
      ratios vs cit_default (clf / reg): no_adaptive 0.95 / 0.88 (range
      0.73-1.00), no_feature_scan 1.53 / 0.94, no_threshold_scan 9.9 / 11.3
      (1.0-57), exact_thresholds 1.02 / 0.96, no_feature_mute 0.99 / 0.97,
      no_bonferroni 0.63 / 1.68. Default CIT fits are 6-14x faster than before
      the fix (e.g. madelon 96 s -> 15 s, waveform 33 s -> 2.4 s). This is the
      table the arXiv runtime rows are rewritten from.
- [ ] Rewrite the arXiv runtime ablation rows and captions from the decoupled
      run: under `minimum` budgets the stopping row must be ~1x by the
      proposition's remark; whatever the coupled knob showed was scanning.

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
