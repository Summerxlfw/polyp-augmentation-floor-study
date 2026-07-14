# Release 0.1.1

This release aligns the reproducibility package with the revised BSPC
manuscript.

## Changes

- Records the complete 17-configuration, 102-run training inventory.
- Adds a synchronized video-cluster bootstrap for the IIa-minus-Ip and
  large-minus-small response contrasts.
- Labels those contrast families as multiplicity-controlled exploratory
  analyses rather than training-time preregistered outcomes.
- Distinguishes six repeated trainings nested within two fixed source settings
  from six independent source domains.
- Adds the video-cluster interval to the phenotype table and harmonizes public
  provenance terminology across plots, tables, and derived CSVs.
- Uses neutral public labels (`SLAug`, `Spatial warp`, and `Paris IIa`) while
  retaining historical run identifiers only for traceability.
- Documents the SLAug shared-floor integration and study-selected SBF grid size
  of 8 for 352 × 352 inputs.
- Expands tests for the contrast bootstrap, analysis role, and 102-run
  inventory.

## Verification

The package contains no raw endoscopy images, model checkpoints, private
clinical data, server credentials, or user-specific logs. Run `pytest -q`,
`python figures/build_main_figures.py`, and
`python tables/build_main_tables.py` before publishing the archive or tag.
