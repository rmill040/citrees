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

## Remaining pipeline gates (running autonomously)

- [ ] JSS grid campaign 3 drain (~155 cells left) → materialize/merge with
      receipt validation → write JSS performance section.
- [ ] CIF mechanism ablation rerun (8 boxes, 40-way) → rebuild
      `tab:cif-ranking-ablation` (arXiv V1) from corrected surface.
- [ ] EC2 knob/threshold ablation → `tab:cit-runtime-hyperparams` runtime ratios
      (arXiv V2). **No laptop timings in the paper** — fixed-hardware EC2
      numbers only; laptop run is a score-only cross-check.
- [ ] Independent factual review of arXiv v2 (fresh-eyes subagent) after V1/V2
      land; then rebuild + visually inspect both PDFs.
- [ ] JSS top-level replication run last (validates the whole chain).
