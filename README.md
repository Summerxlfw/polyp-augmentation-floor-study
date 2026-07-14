# Polyp augmentation floor study

This repository contains the code and derived evidence for a harmonized
single-source cross-center study of polyp-segmentation training configurations.
It accompanies the manuscript *What remains above a strong augmentation
floor? Phenotype-conditioned analysis of cross-center polyp segmentation*.

The repository is an empirical-analysis release, not a claim that the included
coverage implementations reproduce every cited method under its native recipe.
Rows marked as coverage configurations must not be interpreted as rankings of
the corresponding published systems.

## What is included

- `src/training/augmentations.py`: standalone, tested implementations of the
  four spatial-formulation cells.
- `src/training/`: the final formal training driver snapshot, with only
  machine-specific paths changed to environment variables.
- `src/evaluation/`: boundary metrics, paired statistics, and the project-level
  evaluation adapter retained for provenance.
- `analysis/`: six-cell paired statistics, multiplicity families, factorial
  contrasts, and registry builders.
- `data/case_level/`: derived per-case metrics; no raw patient images.
- `data/derived/`: manuscript-facing result tables and plotting inputs.
- `figures/` and `tables/`: deterministic figure and LaTeX-table builders.

## Quick verification

```bash
conda env create -f environment.yml
conda activate polyp-augmentation-floor
pytest -q
python figures/build_main_figures.py
python tables/build_main_tables.py
```

The figure command writes PDF/SVG/PNG files to `figures/generated/`. The table
command writes editable LaTeX fragments to `tables/generated_tables/`.

## Recompute the main statistical families

Set `PYTHONPATH=analysis` so that all analysis modules use the same paired-unit
implementation.

```bash
export PYTHONPATH="$PWD/analysis"

python analysis/f1_family_stats.py \
  --primary-case-csv data/case_level/main_flat_cases.csv \
  --external-case-csv data/case_level/external_boundary_cases.csv \
  --output-dir reproduced/f1

python analysis/f4_family_stats.py \
  --case-csv data/case_level/main_flat_cases.csv \
  --output-dir reproduced/f4

python analysis/f5_boundary_stats.py \
  --case-csv data/case_level/external_boundary_cases.csv \
  --metric-code src/evaluation/l1_boundary_metrics.py \
  --output-dir reproduced/f5

python analysis/factorial_local_reanalysis.py \
  --flat data/case_level/factorial_4cell_flat_case.csv \
  --boundary data/case_level/factorial_4cell_boundary_case.csv \
  --out-dir reproduced/factorial
```

The primary inferential unit is the source-center-by-seed cell (two source
centers times three seeds). Case-level rows are not treated as independent
replicates. Video-cluster resampling is supporting analysis.

## Training configuration

Formal runs used Python 3.10.19, PyTorch 2.1.0, CUDA 12.1, batch size 4,
20 epochs, patience 5, two source centers, and seeds 0, 1, and 2. The full
pipeline also expects the authors' Polyp-PVT adapter and the official SLAug and
CCSDG repositories. Those third-party sources and pretrained weights are not
redistributed here; obtain them from their official repositories and follow
their licenses.

Before using the project-level driver, set at least:

```bash
export POLYP_PROJECT_ROOT=/path/to/checkout
export POLYPGEN_ROOT=/path/to/PolypGen
export SUNSEG_ROOT=/path/to/SUN-SEG
export S1_REPRO_REPOS=/path/to/official/baseline/repos
export S1_CHECKPOINT_ROOT=/path/to/checkpoints
```

The standalone augmentation module and all statistical/figure code do not
require model checkpoints.

## Data and privacy

PolypGen and SUN-SEG are public datasets and should be obtained from their
official sources. This repository redistributes only derived numerical metrics.
It contains no raw endoscopy frames, annotations, private clinical data, or
trained checkpoints. See `PROVENANCE.md` for hashes and release boundaries.

## Citation and license

Citation metadata are provided in `CITATION.cff`. Original code in this
repository is MIT licensed. Third-party datasets, repositories, and model
weights remain under their own terms.
