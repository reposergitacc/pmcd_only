# PMCD Pain Recognition

Research code for pain-intensity recognition on the PMCD dataset using single-stage and hierarchical Gradient Fusion Transformer models.

## Overview

The repository contains training and evaluation pipelines for three PMCD pain classes:

- `0`: no pain
- `1`: moderate pain
- `2`: severe pain

The models use five physiological channels: BVP, EDA-E4, respiration, EDA-RB, and EMG. Experiments support five-fold cross-validation, fixed folds, and leave-one-subject-out (LOSO) evaluation.

## Requirements

- Python 3.10+
- NumPy
- pandas
- SciPy
- scikit-learn
- matplotlib
- PyTorch
- tqdm (for the data-preparation pipeline)

Example installation:

```bash
pip install numpy pandas scipy scikit-learn matplotlib torch tqdm
```

## Data

The PMCD dataset is not included in this repository. The training scripts expect a directory containing:

```text
X.npy
y.npy
subjects.npy
```

The input is converted to the five common physiological channels used by the models. On Kaggle, the scripts can automatically search `/kaggle/input`; otherwise, pass the dataset directory explicitly with `--pmcd-dir`.

## Usage

Run commands from the directory that contains the cloned `pmcd_only` folder.

Quick pipeline check:

```bash
python -m pmcd_only.train_single_kfold --pmcd-dir /path/to/pmcd_np --output-dir results/single_kfold --quick
```

Single-model five-fold evaluation:

```bash
python -m pmcd_only.train_single_kfold --pmcd-dir /path/to/pmcd_np --output-dir results/single_kfold
```

Hierarchical five-fold evaluation:

```bash
python -m pmcd_only.train_hierarchical_kfold --pmcd-dir /path/to/pmcd_np --output-dir results/hierarchical_kfold
```

LOSO evaluation:

```bash
python -m pmcd_only.train_single_loso --pmcd-dir /path/to/pmcd_np --output-dir results/single_loso --seeds 42

python -m pmcd_only.train_hierarchical_loso --pmcd-dir /path/to/pmcd_np --output-dir results/hierarchical_loso --seeds 42
```

Use `--help` on any training module to see all options, including `--fixed-folds`, `--fold-seed`, `--epochs`, and `--batch-size`.

## Repository structure

- `config.py` — shared constants, folds, paths, and hyperparameters
- `data_utils.py` — PMCD loading, preprocessing, and fold construction
- `model.py` — model architecture, losses, and signal augmentation
- `train_utils.py` — datasets, loaders, training helpers, calibration, and metrics
- `eval_utils.py` — reports, confusion matrices, plots, and result summaries
- `train_single_kfold.py` — single three-class model with K-fold evaluation
- `train_single_loso.py` — single three-class model with LOSO evaluation
- `train_hierarchical_kfold.py` — hierarchical two-stage model with K-fold evaluation
- `train_hierarchical_loso.py` — hierarchical two-stage model with LOSO evaluation
- `train_hierarchical_loso_noval.py` — LOSO experiment without a validation split
- `count_classes_per_subject.py` — per-subject class-distribution utility
- `data pipeline.py` — Kaggle/notebook-oriented data-preparation and analysis cells

## Reproducibility

Default seeds and reference fixed-fold results are defined in `config.py`. Use `--fixed-folds` to reproduce the stored patient assignments. Generated datasets, trained weights, and result directories should not be committed to the repository.

## Notes

`data pipeline.py` combines multiple notebook/Kaggle cells and is intended to be run section by section. It is not a standalone Python module in its current form.

