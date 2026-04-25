# CIF Runtime Ratio Analysis

First-pass analysis of the mirrored CIF knob ablation.

Canonical local outputs:

- `paper/results/tables/mirrored_knob_ablation.csv`
- `paper/results/tables/paper_mirrored_knob_ablation_summary.csv`
- `paper/results/tables/paper_presentation_practical_controls_summary.csv`

S3 provenance:

- `s3://citrees-837116549485/adhoc-batch/speed-rerun-20260413/mirrored/i-02bcac3fb76227ca1/paper/results/tables/mirrored_knob_ablation.csv`

The local table matches the S3 copy exactly by SHA-256:
`64b0ae58894b2f4161577469b0fad9cf10582ec2bc571aa9c039306fe4c43d2d`.

## Evidence Boundary

The mirrored CIF source table is aggregated across seeds. We do not have
seed-level raw CIF rows from this run. Therefore the paper-facing runtime ratio
is:

`variant dataset-level mean runtime / cif_default dataset-level mean runtime`

computed within each `(task, dataset_type)` cell, then summarized over the
real or synthetic dataset group.

This is still a ratio-based timing analysis, but it is not the same evidence
granularity as the new CIT table, which has paired seed-level ratios.

## Main Findings

`cif_no_adaptive` is consistently slower. Median runtime ratios are `5.140`
for real classification, `5.518` for synthetic classification, `8.449` for
real regression, and `3.978` for synthetic regression. This is the cleanest CIF
runtime result: adaptive stopping is a major practical control for CIF.

`cif_no_bonferroni` is fast but changes the statistical rule. Median runtime
ratios are `0.166`, `0.017`, `0.027`, and `0.022` across real classification,
synthetic classification, real regression, and synthetic regression. Treat it
as a boundary diagnostic, not a recommended setting.

`cif_no_mute` is nearly neutral. Median runtime ratios are `0.991`, `1.000`,
`0.979`, and `1.010`, with negligible downstream changes. This is not a strong
standalone speed lever.

`cif_no_scan` is mixed. It is modestly faster on real classification (`0.894`)
but slower on synthetic classification (`1.033`), real regression (`1.825`),
and synthetic regression (`1.196`). Do not present feature scan removal as a
general speedup.

`cif_no_threshold_scan` is mixed. It is near default on classification
(`0.977` real, `1.106` synthetic), faster on real regression (`0.607`), and
near default on synthetic regression (`0.940`). The downstream deltas are small
in the aggregate table, so the safe statement is that this knob is not a
uniform runtime improvement.

`cif_no_bootstrap` changes the sampling scheme, so keep it separate from pure
runtime-control claims. It is mixed around default, with median ratios of
`1.150`, `0.971`, `0.945`, and `0.944`.

`cif_all_off` is fast but not a meaningful recommended setting. It combines
several changes, including removal of Bonferroni correction, and should remain
a boundary check.

## Paper-Safe Interpretation

For CIF, the ratio story is straightforward for adaptive stopping: the default
adaptive stopping controls materially reduce runtime relative to full
permutation tests. Other implementation controls are mixed or near neutral.
The Bonferroni-off and all-off rows should not be used as practical
recommendations because they change the statistical rule.
