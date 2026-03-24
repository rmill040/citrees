# Session Status — 2026-03-24

## What's Done

### Paper Section 5 (Experiments) — DRAFTED
All 5 figure placeholders replaced with real figures and data-backed text:
- **5.5.1 Selection bias** — `selection_bias_demo.png` inserted, text written
- **5.5.2 P-value calibration** — `calibration_pvalue_ecdf.png` + `ablation_power.png`, calibration table with exact rejection rates
- **5.5.3 Synthetic ranking** — `synthetic_cd_precision_at_10_clean.png` (NEW clean CD diagram), CIF rank 7.0
- **5.5.4 Runtime** — `ablation_scaling.png`, 200-1000x gap documented with exact numbers
- **5.5.5 Real-data** — `clf_cd_balanced_accuracy_clean.png` + `reg_cd_r2_clean.png` (NEW clean CD diagrams)
- **5.5.6 Ablation** — NEW section with 5 figures: alpha sensitivity, n_estimators, n_resamples, bootstrap, noise robustness

### Figures Generated
New publication-quality figures (all in `paper/results/figures/`):
- `ablation_alpha_depth.png` — Precision vs alpha, two-panel (clf/reg)
- `ablation_n_estimators.png` — Saturation curves
- `ablation_scaling.png` — CIF vs RF vs ET runtime (log scale)
- `ablation_bootstrap.png` — Bootstrap × feature subsampling grouped bars
- `ablation_power.png` — Power curve (Type I error control)
- `ablation_nresamples.png` — Diminishing returns beyond B=99
- `clf_cd_balanced_accuracy_clean.png` — Clean CD diagram (replaces ugly old one)
- `reg_cd_r2_clean.png` — Clean CD diagram
- `synthetic_cd_precision_at_10_clean.png` — Clean CD diagram

### R Baselines
- **Block 10 R baselines: COMPLETE** — 48 rows saved to `ablation_block10_r_baselines.csv`
  - Covers all 8 CLF synthetic + 4 REG synthetic datasets
  - 4 R methods: r_ctree_bonf, r_ctree_mc, r_cforest_bonf, r_cforest_mc
- **Block 11 R baselines: RUNNING** — On madelon (5/9 datasets done)

### Key R Baseline Findings (Block 10)
On clf_toeplitz (CIF's sweet spot):
- r_ctree_bonf: **P@10=0.94** (matches CIF strict_default 0.84... wait, r_ctree_bonf BEATS CIF here!)
- r_ctree_mc: P@10=0.58 (MonteCarlo much worse)
- r_cforest_bonf: P@10=0.62
- CIF strict_default: P@10=0.84 (from Block 10 log)
- CIF no_bonf_a10: P@10=0.88

On clf_weak_signal:
- r_ctree_bonf: P@10=0.12 (same as CIF strict_default)
- r_cforest_bonf: P@10=0.08
- CIF no_bonf_a10: P@10=0.26 (CIF wins when Bonferroni removed)

On clf_confounder:
- r_ctree_bonf: P@10=0.14
- r_cforest_bonf: P@10=0.18
- CIF strict_default: P@10=0.42 (from empirical findings)

### Infrastructure
- `r_methods.py` fixed for Amazon Linux (`/usr/lib64/R` path)
- R + rpy2 + partykit installed on both EC2 instances
- `r_baselines_augment.py` created and deployed
- 2 idle instances terminated (i-0b848661bdd60ddec, i-071969d2faa0b1f6e)

### Cleanup Done
- 237 corrupted s3_sync files deleted
- 11 `__pycache__` dirs removed under paper/scripts/
- 2 empty dirs removed (paper/tests/, paper/results/tables/ablation/)

## What's Still Running on EC2

### i-0af0c2b4730372675 (34.229.62.253)
- **Block 10 CIF** — ~5/12 datasets done, several hours remaining
- **Block 10 R baselines** — DONE (CSV saved)

### i-0a98ffe40b3a3befc (100.28.2.96)
- **Block 11 CIF** — 6/7 CLF done (waveform in progress), then 2 REG remaining
- **Block 11 R baselines** — ~5/9 done (madelon in progress)

## What's Needed Before Paper Submission

### Must-have (blocking)
1. **Pull Block 10 CIF CSV when done** — Will have the full strictness continuum data (CIF default through wide-open) for all synthetic datasets
2. **Pull Block 11 CIF + R baselines CSVs** — Real dataset ablation with R comparisons
3. **Merge R baselines into ablation narrative** — Update Section 5.5.6 with r_ctree/r_cforest numbers
4. **Generate strictness continuum figure** — From Block 10: depth/precision vs. strictness level, CIF configs + R baselines on same plot. This is THE key ablation figure showing CIF's position on the strictness→power spectrum relative to r_ctree/r_cforest.

### Should-have
5. **Regenerate CD diagrams with better fonts** — Current clean versions are good but could use LaTeX-rendered math fonts
6. **LaTeX compilation check** — Verify the new Section 5 compiles cleanly
7. **Dataset summary table** — Currently missing: a table with dataset name, n, p, task for all 31 datasets
8. **Hyperparameter grid table** — For appendix: full method roster with param grids

### Nice-to-have
9. **Real-data ablation figures** — From Block 11: CIF variants vs RF/ET/CIT on real datasets (bar chart)
10. **Cross-reference empirical findings doc** — Ensure Section 5 claims match the detailed analysis in empirical-findings.md
