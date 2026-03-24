# Skipped Experiments

Experiment configurations that consistently fail after multiple retries across
different EC2 instance types. These are excluded from the experiment grid via
`_SKIPPED` in `paper/scripts/pipeline/grid.py`.

**Total skipped**: 3 configs (3 rankings + 3 metrics = 6 artifacts)

---

## r_ctree MonteCarlo on high-dimensional classification datasets

R's `partykit::ctree` with `testtype="MonteCarlo"` performs 9,999 permutation
resamples per node split. On high-dimensional datasets, certain CV fold / seed
combinations cause the R process to hang indefinitely or OOM.

The Bonferroni config (`r_ctree__9d1ca9c27dfc7f5e`, `testtype="Bonferroni"`)
uses an asymptotic approximation and completes successfully for all datasets.

### Config: `r_ctree__b6e09ceb0eb26367` (testtype=MonteCarlo)

**Params**: `alpha=0.05, minbucket=7, minsplit=20, nresample=9999, teststat='quadratic', testtype='MonteCarlo'`

| Dataset | Features | Seeds Failed | Seeds Succeeded |
|---------|----------|--------------|-----------------|
| gisette | 5,000 | 3 | 0, 1, 2, 4 |
| isolet | 617 | 2, 3 | 0, 1, 4 |

### Retry history

These specific (config, dataset, seed) combos were retried across:
- **c6a.8xlarge** (AMD EPYC, 64 GB RAM) — 2026-03-12: failed
- **r5.4xlarge** (Intel Xeon, 128 GB RAM) — 2026-03-13 to 2026-03-15: failed

Other seeds for the same config + dataset completed successfully on both
instance types, confirming the issue is specific to the data split.

---

## Previously excluded (hardcoded in `_EXCLUDED`)

These are excluded at the (method, dataset) level — all seeds, all configs:

| Method | Dataset | Features | Reason |
|--------|---------|----------|--------|
| r_ctree | dexter | 20,000 | `protect(): protection stack overflow` |
| r_cforest | dexter | 20,000 | Same as above |

---

## Pending: CIT-RDC on high-dimensional datasets (43 configs)

These configs are NOT skipped — they are slow but completing. CIT with
`selector='rdc'` on datasets with 600-11,000 features takes hours per job
(confirmed: isolet with 617 features took 20+ min per fold locally and was
still running). These need long-running EC2 instances to complete.

### Config: `cit__2f00ba06d3fd6444` (selector=rdc, honesty=False) — 27 pending

Datasets: CLL_SUB_111, ORL, arcene, dexter, gisette, isolet, orlraws10P,
pixraw10P, warpPIE10P

### Config: `cit__f79ebf6949fbb266` (selector=rdc, honesty=True) — 16 pending

Datasets: CLL_SUB_111, gisette, isolet, orlraws10P, pixraw10P
