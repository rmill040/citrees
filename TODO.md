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

- [ ] Code change exists as an UNCOMMITTED working-tree edit (`cores=32` in
      `paper/jss/replication/performance.py:559` + docstring; tests pass).
      Decide whether to land it as a separate supplement profile instead of
      changing the main harness, since campaign 3's identity is bound to the
      `cores=1` source.
- [ ] Scope: only cells with `method == "partykit"` and forest model family
      (~100–200 of 960); these run 10–20× faster parallel.
- [ ] New (small) campaign under the changed source; batch drivers in
      `scratch/jss_batch_driver.py` are reusable (update PREFIX/campaign/profile
      constants).
- [ ] Performance section: report partykit default (serial) and `cores=32` side
      by side; state both libraries' parallelism provenance explicitly.
- [ ] **citrees shipped-defaults supplement (same supplementary campaign):** the
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

## Remaining pipeline gates (running autonomously)

- [ ] JSS grid: assemble 947 measured cells + 13 censored → write JSS
      performance section once supplement completes.
- [ ] CIF mechanism ablation rerun (8 boxes, 40-way) → rebuild
      `tab:cif-ranking-ablation` (arXiv V1) from corrected surface.
- [ ] EC2 knob/threshold ablation → `tab:cit-runtime-hyperparams` runtime ratios
      (arXiv V2). **No laptop timings in the paper** — fixed-hardware EC2
      numbers only; laptop run is a score-only cross-check.
- [ ] Independent factual review of arXiv v2 (fresh-eyes subagent) after V1/V2
      land; then rebuild + visually inspect both PDFs.
- [ ] JSS top-level replication run last (validates the whole chain).
