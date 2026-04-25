# CIT Runtime Ablation Analysis

First-pass analysis of the CIT-only runtime ablation downloaded from:

- `s3://citrees-837116549485/runtime-ablation-cit-only-sharded/`

Canonical local outputs:

- `paper/results/tables/cit_runtime_ablation_raw.csv`
- `paper/results/tables/cit_runtime_ablation_dataset_summary.csv`
- `paper/results/tables/paper_cit_runtime_ablation_summary.csv`

## Run Integrity

- Complete run: `805` successful fits.
- Design: `23` dataset shards x `5` seeds x `7` CIT variants.
- Tasks: `15` classification/synthetic-or-real datasets and `8`
  regression/synthetic-or-real datasets.
- Hardware: all rows report `c7i.24xlarge` in `us-east-1`.
- Scope: CIT timing only. The script records fit/ranking runtime and synthetic
  top-`k` recovery, but it does not run downstream classifiers/regressors.

Use paired runtime ratios against `cit_default` within the same
`(task, dataset_source, dataset_type, seed)` cell. Do not use absolute times to
compare against older CIF timing studies unless the hardware and run contract
are explicitly aligned.

Paper-facing reporting should use the ratio columns in
`paper_cit_runtime_ablation_summary.csv`. Raw seconds remain provenance only in
`cit_runtime_ablation_raw.csv` and `cit_runtime_ablation_dataset_summary.csv`.

## Main Findings

`cit_no_bonferroni` is not a legitimate speed knob for the paper story. It is
fast in every group, with median runtime ratios of `0.030` for real
classification, `0.013` for synthetic classification, `0.012` for real
regression, and `0.061` for synthetic regression. But it changes the
multiple-testing rule, grows much broader trees, and should be described only
as a boundary check.

`cit_no_feature_mute` is nearly neutral for single-tree CIT. Median runtime
ratios are `1.000` for real classification, `0.998` for synthetic
classification, `0.998` for real regression, and `1.001` for synthetic
regression. It also leaves the synthetic top-`10` recovery unchanged in both
tasks. This is not strong evidence for a meaningful single-tree runtime lever.

`cit_exact_thresholds` is not a runtime improvement. It is slower in synthetic
classification (`1.336` median ratio), real regression (`2.509`), and
synthetic regression (`1.056`). Real classification is mixed (`0.982` median,
`1.370` mean), but the mean is pulled upward by slow cases. The safe
interpretation is that exact thresholds add cost without a consistent recovery
gain.

`cit_no_feature_scan` is mixed. Median ratios are `1.047` for real
classification, `0.927` for synthetic classification, `1.463` for real
regression, and `0.693` for synthetic regression. Synthetic classification
top-`10` recovery drops by about `0.015` precision and `0.017` F1. This is not
a clean general-purpose speed result.

`cit_no_threshold_scan` is also mixed. It is faster on synthetic
classification (`0.659` median ratio) with essentially unchanged mean F1, but
it is slower on synthetic regression (`1.193`) and loses about `0.016` F1. Real
data are unstable across datasets. This should be framed as a task-dependent
runtime/selection interaction, not a universal knob.

`cit_no_adaptive` is the surprising result and needs careful wording. It is
much faster on synthetic datasets (`0.180` median ratio for classification and
`0.182` for regression), with small synthetic recovery changes. Real data are
mixed: real classification has a median ratio near `0.988` but a mean ratio of
`4.008` due to large slowdowns on small/easy datasets; real regression has a
median ratio of `1.137`. Do not claim simply that adaptive stopping is always
faster for CIT.

## Outliers To Inspect Before Manuscript Claims

- `real_iris`, seed `1718`: `cit_no_adaptive` is `83.97x` slower than default.
- `real_digits`: `cit_no_feature_scan`, `cit_no_adaptive`, and
  `cit_no_threshold_scan` show several large slowdowns and broader trees.
- `real_diabetes`: `cit_no_threshold_scan` has `5.72x` to `6.61x` slowdowns on
  several seeds.
- `real_openml_madelon`: several variants are much faster than default, so it
  heavily affects mean runtime ratios.

## Paper-Safe Interpretation

The CIT runtime study supports a narrower point than the CIF runtime ablation:
CIT runtime controls interact with tree growth and selection behavior, so the
paper should report paired ratios and avoid implying that any single switch is
uniformly better. The most defensible statement is that the practical default
keeps the intended statistical rule, while exact-threshold and scan-removal
variants do not give a uniformly better runtime/quality tradeoff. The
Bonferroni-off variant is a boundary diagnostic, not a recommended setting.
