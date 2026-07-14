# Polyp augmentation floor study

This repository contains the code and derived evidence for a controlled
single-source cross-center study of polyp-segmentation training configurations.
It accompanies the manuscript *Beyond aggregate Dice: Phenotype-conditioned
responses under a strong augmentation floor for cross-center polyp
segmentation*.

The repository is an empirical-analysis release. Protocol-aligned implementations
test intervention families under the shared study protocol and do not claim to
reproduce every cited system under its original publication recipe.

## What is included

- `src/training/augmentations.py`: standalone, tested implementations of the
  four spatial-formulation cells.
- `src/training/`: the final formal training driver snapshot, with only
  machine-specific paths changed to environment variables.
- `src/evaluation/`: boundary metrics, paired statistics, and the project-level
  evaluation adapter retained for provenance.
- `analysis/`: paired statistics across six repeated training units,
  multiplicity families, synchronized video-cluster contrasts, factorial
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

Paired summaries use six repeated training units, with three seeds nested
within each of two fixed source settings. Their intervals describe training
repetitions under those fixed settings rather than a population of possible
source centers. Case-level rows are not treated as independent replicates;
synchronized video-cluster resampling separately quantifies test-video
sampling. The IIa-minus-Ip and large-minus-small families are explicitly
recorded as multiplicity-controlled exploratory analyses.

## Training configuration

Study runs used the `polyp_base_v5` environment with Python 3.10.19, PyTorch
2.1.0, CUDA 12.1, cuDNN 8902, NVIDIA RTX 4090 GPUs, batch size 4, 20 epochs,
patience 5, two source centers, and seeds 0, 1, and 2. The full
pipeline also expects the authors' Polyp-PVT adapter and the public SLAug and
CCSDG repositories. Those third-party sources and pretrained weights are not
redistributed here; obtain them from their source repositories and follow
their licenses.

The SLAug adaptation retained its two-pass update and used the shared study
floor. The source implementation did not define a polyp-specific
saliency-balancing-fusion grid size; this study used `S1_SBF_GRID_SIZE=8` for
352 × 352 inputs.

Before using the project-level driver, set at least:

```bash
export POLYP_PROJECT_ROOT=/path/to/checkout
export POLYPGEN_ROOT=/path/to/PolypGen
export SUNSEG_ROOT=/path/to/SUN-SEG
export S1_REPRO_REPOS=/path/to/public/baseline/repos
export S1_CHECKPOINT_ROOT=/path/to/checkpoints
```

The standalone augmentation module and all statistical/figure code do not
require model checkpoints.

## Data and privacy

PolypGen and SUN-SEG are public datasets and should be obtained from their
source repositories. This repository redistributes only derived numerical metrics.
It contains no raw endoscopy frames, annotations, private clinical data, or
trained checkpoints. See `PROVENANCE.md` for hashes and release boundaries.

Historical identifiers such as `official_SLAug` are retained only to preserve
traceability to completed runs. They do not imply execution by the original
authors or a publisher-designated reference implementation; public tables and
figures use the method name `SLAug`.

## Citation and license

Citation metadata are provided in `CITATION.cff`. Original code in this
repository is MIT licensed. Third-party datasets, repositories, and model
weights remain under their own terms.
