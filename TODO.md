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

## JSS application section: drop DGRP, find a replacement (author, 2026-09-07)

**Decision:** the _Drosophila_ Genetic Reference Panel (DGRP) cardiac analysis
is being **removed** as the JSS application section. The author rejected it as
the paper's showcase use case. Do not invest further effort in it and do not
reinstate it.

Why it is being dropped: n is only 157-166 lines per trait across seven cardiac
traits, which is too thin to carry an application section, and framing fruit-fly
heart measurements as a "biomedical application" overclaims to any reader
expecting human clinical relevance.

What was deleted (2026-09-07): `paper/jss/data/dgrp/` in full, 8.6 GB
(`input/Phenosnip.201612.sqlite` 6.7 GB, `dgrp2.bed`/`bim` 362 MB, the two
source tarballs 1.3 GB, `derived-genotypes/` 236 MB). The directory was
gitignored and nothing under it was tracked. It is fully re-downloadable with
pinned checksums from the eLife CDN and Zenodo via
`paper/jss/replication/dgrp.py` if this decision is ever reversed.

The DGRP results were deleted at the author's direction (2026-09-07):
`paper/jss/results/dgrp-full`, `dgrp-cloud-execution`, and
`dgrp-manuscript-summary`, 18 MB, all untracked and gitignored. Together with
the raw data this means **no DGRP number in the current `article.tex` can be
regenerated** -- the text is now unsupported and must come out, not be revised.
`paper/jss/results/replication-quick/dgrp` (2.7 MB) was left in place: it is a
child of a receipt-verified whole-suite run and removing one child would
invalidate that tree's receipt for no meaningful space. It goes stale on its own
once the suite drops from six analyses to five.

**TCGA-BRCA tried and dropped (2026-09-07/08):** a preregistered receptor-status
analysis ran quick and showed a predictive null (cif, RF and a marginal screen
all at held-out log loss 0.175-0.181) with cif 40x slower than RF;
all-continuous expression is the one setting where RF importance is not biased,
so it cannot show what the library is for. The author killed the full run. Code,
spec, tests and 105 MB of data removed. Also tried and rejected: UCI Adult
(1994, stale) and ACS PUMS (census microdata, not a biomedical audience).

**Replacement landed (2026-09-08): NHANES 2021-2023 diabetes.** n=3,597 adults,
27 mixed-cardinality predictors (2 to ~n distinct values) plus three shuffled
noise controls; RF vs citrees CIF vs partykit::cforest on identical numeric
input. Preregistration
`paper/jss/replication/nhanes_diabetes-specification.json`, module
`nhanes_diabetes.py`, 25 tests. Quick profile (exploratory): RF ranks the
shuffled survey weight 12.3 of 30, citrees 21.0, cforest 22.4; all three
preregistered hypotheses in the predicted direction; positive controls pass.
Design lesson recorded in the spec: the first cohort count (2,975) was wrong
because 7/9 were recoded as missing in columns where they are valid answers;
fixed from the codebooks and re-pinned to 3,597 before any fold ran.

Still to do, in order:

- [x] Choose the replacement dataset.
- [x] Retire dgrp.py, then tcga_brca.py, and their registrations.
- [x] Full NHANES profile run 2026-09-08 on clean `609d6f7`, 12 min locally,
      output `paper/jss/results/nhanes-full` (ignored, receipt-verified).
      Primary: shuffled survey weight mean rank RF 12.1 vs citrees 23.9 (cforest
      25.2), corrected t = -4.97, p = 4e-6. Secondaries: cardinality Spearman
      RF-CIF +0.52 (p = 1e-20); low-cardinality clinical block mean rank RF 22.1
      vs CIF 15.7 (p = 5e-16). Diagnosed hypertension RF 18.8 / CIF 5.8 /
      cforest 3.8; diagnosed high cholesterol RF 20.5 / CIF 5.3 / cforest 2.3.
      CIF-cforest rank agreement 0.76 vs RF-cforest 0.45. Top-10 held-out AUC:
      cforest 0.837, CIF 0.835, RF 0.812. Median fit per fold: RF 0.3 s, cforest
      5.8 s, CIF 6.9 s. Positive controls pass for all three methods.
- [x] DGRP removed from `paper/jss/article.tex` and `refs.bib` (2026-09-08);
      NHANES design subsection, results section with the 30-row rank table,
      discussion paragraph, limitations and availability written from the
      full-run artifacts. Builds at 19 pages, 0 overfull.
- [x] Whole-suite quick replication on clean `350e94b` (2026-09-08): all six
      analyses published, every child receipt clean at the same SHA, output
      `paper/jss/results/replication-quick-350e94b` (ignored). Older
      `replication-quick` and `tcga-brca-quick` trees deleted.
- [x] Numeric claim trace of the NHANES section against the full-run artifacts:
      30 checks, 2 prose defects found and fixed (permuted age is ahead of 13
      real predictors, not 14; the discussion now quotes mean ranks 18.8 and
      20.5 rather than table positions 19 and 21).
- [x] Independent two-lane review (Claude and a second model, read-only,
      2026-09-09). Confirmed and fixed: partykit cforest had sampled all 30
      predictors per node (wrapper maps an omitted mtry to all) instead of its
      default 6; its importance used 1 permutation instead of the 10 used in the
      synthetic sections; the random forest was timed on one core against two
      all-core forests. Code fixed, amendment logged in the specification, full
      profile rerun on clean `b560ab6`: RF and citrees numbers identical, all
      partykit numbers regenerated (shuffled weight 26.4, agreement with citrees
      0.84, 20.9 s per fold). Prose: cohort cardinality 3,464 vs fold median
      2,790 separated; prediction and discussion claims restated as descriptive;
      denominators, standardization, Spearman-vs-Kendall clause, and the
      Bouckaert-Frank approximation note added; en dash removed. Both lanes
      verified every other number and found no DGRP residue and a balanced
      bibliography. Declined (structure, author's prior layout): folding the
      results section into the design subsection; the pre-existing one-line
      paragraph in the performance section is left for the author.
- [x] **NHANES manuscript artifacts now come from EC2 (2026-09-09).** The author
      pointed out that timing on the laptop is not comparable to the performance
      section, which used 32-core cloud instances. Full profile rerun on one
      c6a.8xlarge (32 vCPU) inside the pinned reference container with the repo
      at `e9abf5b` mounted; box self-terminated, IAM role
      `citrees-nhanes-timing-20260909` deleted, zero project instances left.
      Artifacts under s3 debug/nhanes-timing/out/e9abf5b and installed at
      `paper/jss/results/nhanes-full`. All-core medians per fold: RF 0.18 s,
      citrees 4.15 s, partykit 21.0 s (23x and 114x; the laptop had said 65x and
      209x). Ranks moved by at most 0.12 (RF), 0.48 (citrees), 0.10 (partykit);
      primary t(49) = -5.37, all conclusions unchanged; every number in the
      section regenerated from the EC2 artifacts. Two honest caveats recorded:
      the receipt says git_dirty=true because the Docker bind mount changed file
      modes (the same shallow clone is clean on the host; source hashes in the
      receipt match `e9abf5b`), and citrees forest importances vary slightly
      with the number of worker processes (library follow-up: make forest
      importances independent of n_jobs).
- [x] Whole-suite quick replication on clean `c851a9f` (2026-09-09): all six
      children published, every receipt clean at the same SHA; output
      `paper/jss/results/replication-quick-c851a9f` (ignored).

## Resolved after the 2026-09-09 restart (items 1-3)

1. **Fresh-clone dry run: PASS.** A clone of `2171ce1` from GitHub,
   `uv sync --group paper`, then the quick suite: all six analyses published and
   receipt-verified with no manual step (R + partykit must already be on the
   machine; NHANES data downloads itself). Clone deleted afterwards.
2. **Thread-count / n_jobs invariance fix landed (`6c20a2c`), and it moves
   results at the tie level.** Quick-suite diff, `c851a9f` (old kernels) vs
   `0e3ab51` (new): rdc sensitivity identical except timings; calibration
   differs in one cell (regression splitter, exhaustive, histogram16: 2 -> 3
   rejections of 100); behavior differs only in the WDBC forest rows (Kendall
   tau 0.398 -> 0.393, prediction score 0.941 -> 0.939, agreement 0.960 ->
   0.961); NHANES quick ranks move by up to 5 in single folds and 0.67 in a mean
   rank. Cause: the old observed statistic was reduced in auto-parallel fastmath
   order and exact ties flipped; the new value is the serial one. **Consequence
   for the paper:** the full-profile calibration, behavior and performance
   numbers were produced by the distributed cloud campaigns on pre-fix code and
   are pinned to their receipt SHAs; a reviewer running the current code will
   reproduce them up to tie flips of this magnitude, not bit-for-bit. Author
   decision needed: rerun those campaigns (cost) or state the pinned revision in
   the availability section. **The NHANES full profile was regenerated on EC2
   with the new code (`0e3ab51`, c6a.8xlarge, 2026-09-09)** and the article
   section rewritten from it: RF and partykit mean ranks unchanged to one
   decimal, citrees moved by at most 0.18; primary t(49) -5.38; timing 0.2 / 4.0
   / 20.7 s (22x, 115x). Both boxes and the IAM role are deleted. The receipt
   still reports git_dirty=true inside the container even with core.fileMode
   off; cause not identified (source hashes match).
3. **Max-type threshold test**: implemented, benchmarked, recommendation in the
   performance section notes above; default unchanged.

## Max-type split test: two-model mathematical review (2026-09-09)

Author asked for Codex (GPT-5.6, max reasoning) and Claude Opus to prove or
refute the `threshold_test="maxt"` math. Both lanes independent, both ran their
own 2,000-null simulations with the real kernels (scripts kept under
`scratch/maxt_*`).

Agreed by both: the fixed-budget max-type p-value is **valid** (inclusive rank
of the observed minimum among B+1 exchangeable values, so P(p <= alpha) <=
floor(alpha(B+1))/(B+1)); the "<=" tie rule and both +1 terms are load-bearing;
it is _weak_ familywise control (level alpha under the node's global null) and
conservative under ties, not "exact"; Bonferroni is valid and strictly
conservative because adjacent thresholds correlate at about 0.71; the
implementation matches the math line by line. Null rates in every simulated cell
0.036-0.055 (Opus, B=1,999) and 0.040-0.049 (Codex, B=100); power gain over
Bonferroni +6.0 points (paired SE 1.1) at K=16.

Fixed 2026-09-09: comments no longer say "exactly"; all-invalid threshold sets
return a NaN threshold instead of thresholds[0]; empty threshold arrays raise.
Left as is: ties in the observed argmin go to the first index (Bonferroni branch
samples uniformly); cosmetic.

**Open, author decision (affects the whole library, not just maxt):** the
Beta-posterior adaptive stopping rule has no level theorem. Codex's exact
beta-binomial recursion: at the default confidence 0.95 the level is 0.0494
(fine); at confidence 0.80, which validation permits, it is **0.0527 > 0.05**.
Options: (a) require confidence >= 0.95, (b) allow early stopping only for
futility and run the full budget before any rejection (theorem-backed, slower on
clear rejections), (c) leave as is and document. Also unverified by either lane:
post-selection validity of the split test after the feature was chosen using y
(applies equally to the Bonferroni default), strong FWER, entropy/MAE
calibration.

Hygiene: `paper/theory/study_batched_adaptive_stopping.py` counts only explicit
"reject" outcomes and ignores capped runs whose p < alpha, so its type I figure
understates production; it is cited in neither paper (the JSS calibration
numbers come from real tree fits in `calibration.py`), so no published number is
affected.

## rdc permutation kernels: buffered rewrite (2026-09-06)

The four parallel rdc kernels (`_ptest_rdc_*_parallel*` in
`citrees/_selector.py`) allocated fresh matrices and copied X inside every
permutation; on 26-class data that was 235,301 Numba allocations per
1,000-permutation test. They now standardize X once, keep one label buffer and
one projected-Y buffer per thread slot, and build one-vs-all indicator ECDFs
without a sort. The thread count is a kernel argument so Numba can still cache
them.

- Identity: 42/42 recorded outputs (p-values, permutation counts, statistics;
  multi-class, binary, regression; full and adaptive) and 200/200 null p-values
  identical to the old kernels on real isolet at 32 and 8 threads on x86
  (c6a.8xlarge); 90/90 on glass locally; results independent of thread count (1,
  2, 3, 7). Unit tests pin this
  (`tests/unit/test_selector.py::TestRdcKernelIdentity`, fixture
  `tests/data/rdc_kernel_identity_glass.json`). The indicator ECDF differs from
  the old path by one ulp in the non-member value (fast-math division compiled
  as a reciprocal multiply in some contexts); no output changed.
- Speed: 1.2-1.4x at 32 threads, 1.0-1.1x at 8 threads; allocations 235,301 to
  69 per call; peak RSS 631 to 587 MB. The change is a memory-churn fix, not a
  speedup.
- End-to-end identity: one rdc CIT fit at the ablation settings on a 3,000-row
  isolet subsample gives the same 215 nodes and the same sha256 of importances
  and node structure before and after (`scratch/rdc_tree_identity.py`).
- Host lock-ups are NOT explained by allocation churn: the new kernels hung a
  c6a.8xlarge four minutes into a single-process 32-thread run (silent console,
  both reachability checks failed), while the old kernels ran 18 minutes on the
  same box. A rerun of the identical workload on the recovered box completed. A
  second AMD box hung at the same point and was terminated; an Intel c6i.8xlarge
  ran the whole sequence (32-thread, 8-thread, both tree fits) without a single
  status-check failure. **The lock-up cause is still unidentified**; the
  evidence now points at the AMD host/hypervisor rather than at citrees, so do
  not treat this rewrite as a fix for it. Raw data under
  s3://citrees-856480643277/debug/rdc-bench/out/; all boxes terminated and the
  IAM role/profile citrees-rdc-bench-20260906 deleted.
- No benchmark or timing result needs rerunning: outputs are identical and all
  timing sections use the mc/pc selectors.

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

## Performance section: final state (2026-09-05)

Rewritten and pushed (`5da547d`). Three harness bugs were found and fixed on the
way: recommended-configuration forests ran serial (no `n_jobs`); 32 loky workers
x 32 Numba threads oversubscribed the permutation kernel (now the forest divides
the Numba pool across workers, in the library); partykit was forced to
`mtry = all` while citrees used its `sqrt` default (now both defaults are timed,
plus the all-predictor exhaustive control). The exhaustive campaign numbers were
correct all along.

Head-to-head (real datasets n >= 500, both libraries at defaults, 32 cores
available): citrees recommended is 2.7-15x faster than 32-core partykit on the
seven largest datasets and 22-260x faster than 1-core partykit; slower than
32-core partykit on gamma, madelon, vowel. Mechanism, measured: the
Bonferroni-adjusted threshold permutation test costs ~candidates/alpha
permutations per node; continuous predictors give 256 candidates (~5,000
permutations of O(n)), discrete ones few. Without that adjustment citrees beats
32-core partykit on every real dataset and on the synthetic grid to n=32,000.
Raw data: s3://citrees-856480643277/debug/head-to-head/ (results\*, threshold
diag) and `_control/perf-citrees-rerun/` (harness cells).

**Library design question for the author, now measured (2026-09-09):** the
opt-in `threshold_test="maxt"` (one max-type permutation test on the minimum
impurity over the K candidate thresholds) is implemented and tested. Benchmark
(`scratch/bench_threshold_test.py`, `scratch/bench_threshold_test.json`):

- Null familywise rejection at alpha 0.05, 300 nulls, K=32: maxt 0.050 (full)
  and 0.050 (adaptive); Bonferroni 0.020 and 0.010. Max-type is exact,
  Bonferroni is conservative by 2.5-5x, so switching the default would make
  every tree split more often and would change every published result.
- Power on a weak planted signal: maxt 0.973, Bonferroni 0.960.
- Cost per node test (K=32, n=200): maxt 1.3-1.9 ms, Bonferroni 111-184 ms
  (about 90x).
- Whole NHANES tree (n=3,597, p=27, 9 nodes): with adaptive stopping the two are
  within 1.3x (3.6-5.0 s) at K=32 and K=256; **without early stopping K=256
  Bonferroni takes 341 s and maxt 4.7 s (72x)**. The max-type test removes the
  exhaustive-mode cliff that the performance section reports as censored cells;
  it does not change the adaptive default's speed much.
- Fitted trees on NHANES are the same size and accuracy under both tests;
  importance Spearman 1.000 (K=32) and 0.999 (K=256).

Recommendation: keep `bonferroni` as the default for this release so the paper
and the receipts stay consistent, document `maxt` as the recommended setting for
exhaustive (no early stopping) runs, and consider making it the default in the
next major version after a null-calibration section is added for it.

## Remaining pipeline gates (running autonomously)

- [x] JSS performance section rewritten on corrected timings + head-to-head (see
      "Performance section: final state").
- [x] CIF ranking ablation (arXiv V1, finished 2026-09-06): the full ablation
      (selected CIF vs one tree / no bootstrap / no muting / split-count
      ranking; 5 seeds x 5 folds) is complete on 22 classification and 8
      regression real-data datasets, 3,034 metric files, zero failures. **isolet
      is excluded by author decision** (its 617-feature rdc forests hard-locked
      32-worker hosts; never run it again). `tab:cif-ranking-ablation` restored
      as a full table (mean, median, 95% CI, W-L) from
      `cif_mechanism_ablation_paired_foldseed_vs_default.csv` (replicate-paired
      at seed x fold, standard k only, builder `--exclude-datasets isolet`). One
      tree: -0.071 [-0.101, -0.045], 0-22 (clf); -0.658 [-1.692, -0.054], 0-8
      (reg). The other three changes are within 0.001 in classification; their
      regression means are pulled by the three coepra datasets (medians within
      0.007 of zero). Interim 8-dataset figures (-0.115) and the earlier -0.075
      are superseded; do not reuse. Fleet notes: the first two fleets (35
      on-demand + 35 spot, then 70 hardened boxes) hard-locked (journal ends
      abruptly, no OOM record, reboot ~20 min later, EC2 status impaired) on
      isolet at 32 workers; gisette peaked at 48-54 GB RSS with 32 workers;
      letter needed 16 workers. Fleet 3 used a systemd-resumed per-item runner
      (docker --memory=56g, per-dataset worker counts, S3 skip-if-done,
      terminate on done) and finished with two sweep boxes. All boxes and the
      fleet IAM role/profile `citrees-mech-finish-20260905` are deleted.
- [x] arXiv V2 done (2026-09-05): CIF runtime table from the EC2 knob/threshold
      rerun; CIT runtime table from the EC2 `cit_cif_runtime_ablation` rerun
      (460 fits); abstract sentence updated. Adaptive stopping: 6.1-6.5x
      synthetic / 483-717x real (CIF), 3.2-8.3x (CIT, with deeper trees and
      top-10 recovery changes up to 0.40).
- [x] Independent factual review of arXiv v2 (2026-09-05): 5 rounding /
      aggregation discrepancies found and fixed (two CI bounds, two
      rounded-upward table cells, CIT depth sentence). PDF rebuilt (39 pages, 0
      overfull), page 10 inspected.
- [x] Independent factual review of the JSS performance section (2026-09-05): 13
      prose-level discrepancies (ranges stated too favorably, repeat counts,
      memory bound) rewritten from the head-to-head tables; all 119 table cells
      verified.
- [x] JSS top-level replication run (`--profile quick`, 2026-09-05): all six
      analyses published on a clean tree at `5d79d40` in 44 minutes (calibration
      336 s, behavior 158 s, performance 1,147 s, tutorial 26 s, DGRP 923 s, RDC
      sensitivity 54 s); output `paper/jss/results/replication-quick` (ignored).
      Two fixes on the way: the performance harness dropped the unreported
      selector-sweep axis (one `mi` cell alone ran >28 min), and a stale local
      DGRP derived-genotype cache from 2026-08-10 (pre-17e51eb `schema` field)
      had to be deleted so the current writer rebuilt it. Paper tests 1,154
      passed; library tests 698 passed.
- [x] AWS teardown done 2026-09-05: 22 citrees IAM roles and instance profiles
      deleted (including the AdministratorAccess supervisor role); no project
      instances remain; S3 debug prefixes kept. The two non-project instances in
      the account were left alone.
