# Active TODO

## JSS performance benchmark — partykit parallel supplement (MUST NOT DROP)

**Decision (2026-08-27, discussed with author):** Campaign 3
(`source-0e52014…/campaign-a848f8eb…`, 960 cells) measures every library at its
shipped defaults. Verified against the installed package:
`formals(partykit::cforest)$cores` is `NULL` and `applyfun` defaults to `NULL`,
so cforest fits trees with a serial `lapply` by default — the campaign's
`cores=1` cells are genuine default behavior, and campaign 3 remains valid as
the "default configuration" comparison.

**Outstanding supplement:** run the partykit _forest_ cells one more time with
`cores=32` (partykit's opt-in parallelism) and report partykit both ways in the
performance section. This pre-empts the "you handicapped the baseline" reviewer
objection.

- [x] Code change landed (env-driven `PARTYKIT_CORES`, `CITREES_PERF_VARIANT`)
      (`cores=32` in `paper/jss/replication/performance.py:559` + docstring;
      tests pass). Decide whether to land it as a separate supplement profile
      instead of changing the main harness, since campaign 3's identity is bound
      to the `cores=1` source.
- [x] Scope: only cells with `method == "partykit"` and forest model family
      (~100–200 of 960); these run 10–20× faster parallel.
- [x] New (small) campaign under the changed source; batch drivers in
      `scratch/jss_batch_driver.py` are reusable (update PREFIX/campaign/profile
      constants).
- [x] Performance section written (2026-09-02): exhaustive procedure,
      recommended configuration, partykit 1 core and 32 cores, sklearn CART; two
      tables.
- [x] **citrees shipped-defaults supplement (same supplementary campaign):** the
      main harness measures citrees matched-procedure only (all optimizations
      OFF — early_stopping=None, muting/scanning off, exact thresholds), which
      is citrees' worst case by design. Add citrees-only cells at the shipped
      defaults (adaptive stopping etc.) so the section reports (1) matched
      procedure, (2) shipped defaults, (3) partykit cores=32. Cheap: defaults
      are 4–8× faster per the ablation. NEVER present matched-procedure numbers
      as "citrees performance" without the defaults row.

## Grid endgame (2026-09-02, author directive: censor, do not wait)

Campaign 3 is closed at 947/960 measured cells (932 spot-campaign shards plus 15
cells re-measured on dedicated on-demand boxes with market provenance recorded
truthfully). The 13 unmeasured cells are all citrees forests in the
matched-procedure (exhaustive) arm with the mi/dc/rdc selectors or p=200: serial
trees under the campaign's frozen `n_jobs=1` image, each cell still running
after >48 h of wall clock. Report them as **censored at a 48-hour wall-clock
budget** in the performance table (standard timeout practice), not as
measurements. Never narrate per-cell durations in the manuscript. All straggler
and spot grid fleets are terminated; the relauncher is stopped.

**Section framing (author, 2026-09-02):** exhaustive conditional inference at
the reference procedure is computationally impractical (matched-arm cells
censored at the 48 h budget); the same statistical procedure with citrees'
optimizations (minimum resamples, adaptive stopping, muting/scanning, histogram
thresholds, parallel forests) runs in seconds to minutes. Censored cells are the
motivation for the optimizations, not a caveat.

**Selector scope:** report ONLY the linear selectors — `mc` (classification) and
`pc` (regression). Drop the selector-sweep axis (mi/rdc/dc) from the performance
section entirely; it is where nearly all censored cells live. Remaining censored
cells (p=200 matched-arm forests) are reported as censored.

## Library follow-up: forest parallel efficiency (found 2026-09-02)

`import citrees` cost ~6.5 s, half of it `dcor` imported eagerly for the dc
selector. Forest fits use loky _processes_, so every worker paid that import
under contention; 12 trees in parallel was slower than 12 trees serial.
Threading backend is far worse (GIL-bound tree building, ~300 s vs 150 s
serial), so loky stays. **Fix landed:** dcor is now imported lazily inside the
dc selector (import 6.5 s → 3.2 s; 100-tree forest 46–57 s → 33–54 s locally).
**EC2 verification (2026-09-03, c6a.8xlarge, campaign image):** the harness
reference cell reproduces 134 s for the recommended citrees forest under every
thread setting, with or without the lazy-dcor fix (the harness warm-up fit
already spawns the worker pool, so worker import cost is outside the timed
region). On strong-signal data the same forest takes 27 s (19 s with lazy dcor);
the harness's weak decaying signals (coefficients 1.0 to 0.25, unit noise) keep
p-values near the threshold where adaptive stopping cannot stop early. The 130 s
in the paper is therefore a fair hard-case measurement, and partykit on the same
data and box took 289 s (1 core) and 14.7 s (32 cores). Raw results:
s3://citrees-856480643277/debug/forest-timing-matrix/{run1,run2}. Remaining
parallel inefficiency (per-worker interpreter startup) is a release-engineering
item, not a paper blocker.

## Compute budget rule (author, 2026-09-03)

Every experiment has a hard 48-hour wall-clock budget from its (re)launch. Boxes
carry a 48 h self-destruct (sync, then terminate). Whatever is incomplete at the
cutoff is **censored, not waited for**: the table builders run in non-strict
mode, the completeness table names the excluded cells, and the manuscript states
the count of excluded datasets/cells and the budget in the caption. Never
narrate per-cell durations. Applies now to the CIF mechanism ablation
(relaunched 2026-09-03 ~15:00 UTC, expected ~6 h) and the knob/threshold runner.

**Censoring mechanics:** the CIF mechanism table is built with
`--complete-datasets-only`: a dataset enters a variant-vs-default comparison
only when both have complete seed x fold support for every downstream learner
and k; excluded datasets are listed in
`cif_mechanism_ablation_censored_datasets.csv`, and the manuscript caption
states the per-row dataset count and the 48 h budget. The knob runner's finished
CSV was rescued to S3 (`_control/ablation-rerun/`), with a watcher copying the
threshold-search CSV out every 10 min once it exists.

## Remaining pipeline gates (running autonomously)

- [x] JSS grid: 880 non-selector cells complete (0 censored after dropping the
      selector axis); performance section written and visually inspected.
- [ ] CIF mechanism ablation rerun (8 boxes, 40-way) → rebuild
      `tab:cif-ranking-ablation` (arXiv V1) from corrected surface.
- [ ] EC2 knob/threshold ablation → `tab:cit-runtime-hyperparams` runtime ratios
      (arXiv V2). **No laptop timings in the paper** — fixed-hardware EC2
      numbers only; laptop run is a score-only cross-check.
- [ ] Independent factual review of arXiv v2 (fresh-eyes subagent) after V1/V2
      land; then rebuild + visually inspect both PDFs.
- [ ] JSS top-level replication run last (validates the whole chain).
