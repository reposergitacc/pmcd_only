"""Data loading, preprocessing, and fold utilities for PMCD-only training."""

from __future__ import annotations

import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.model_selection import StratifiedGroupKFold

from .config import (
    CHANNELS,
    EXPECTED_PMCD_COUNTS,
    FIXED_TEST_SUBJECTS,
    KAGGLE_INPUT_ROOT,
    PMCD_CHANNEL_INDICES,
    RATE,
    WINDOW_POINTS,
)


# -----------------------------------------------------------------------------
# Core data structures
# -----------------------------------------------------------------------------

class PainData:
    """Container for preprocessed PMCD data."""
    
    def __init__(self, name: str, x: np.ndarray, y: np.ndarray, subjects: np.ndarray):
        self.name = name
        self.x = x
        self.y = np.asarray(y, dtype=np.int64)
        self.subjects = np.asarray(subjects).astype(str)

    @property
    def classes(self) -> int:
        return int(self.y.max()) + 1

    def __len__(self) -> int:
        return len(self.y)


# -----------------------------------------------------------------------------
# File I/O utilities
# -----------------------------------------------------------------------------

def find_kaggle_input_dir(required_files: list[str], search_root: Path = KAGGLE_INPUT_ROOT) -> Path | None:
    """Find first directory under search_root containing all required_files."""
    if not search_root.is_dir():
        return None
    for candidate in sorted(search_root.rglob("*")):
        if candidate.is_dir() and all((candidate / name).is_file() for name in required_files):
            return candidate
    return None


def require_files(directory: Path, names: list[str]) -> None:
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing in {directory}: {missing}")


def integer_subjects(path: Path) -> np.ndarray:
    return np.load(path).astype(np.int64).astype(str)


def labels_from_one_hot(path: Path) -> np.ndarray:
    labels = np.load(path)
    return np.argmax(labels, axis=1).astype(np.int64) if labels.ndim == 2 else labels.astype(np.int64)


# -----------------------------------------------------------------------------
# Signal preprocessing
# -----------------------------------------------------------------------------

def resample_250_to_64(window: np.ndarray) -> np.ndarray:
    """Resample 250 Hz → 64 Hz (256 points for 4s window)."""
    result = resample_poly(np.asarray(window, dtype=np.float32), up=32, down=125, axis=0)
    if result.shape != (WINDOW_POINTS, len(CHANNELS)):
        raise ValueError(f"Unexpected resampled shape: {result.shape}")
    return result.astype(np.float32, copy=False)


def build_pmcd_common5(pmcd_dir: Path, cache_dir: Path, overwrite: bool = False) -> PainData:
    """Load official PMCD arrays and extract 5 common channels at 64 Hz."""
    destination = cache_dir / "pmcd_common5.npz"
    if destination.exists() and not overwrite:
        payload = np.load(destination)
        return PainData("PMCD", payload["x"], payload["y"], payload["subjects"].astype(str))

    require_files(pmcd_dir, ["X.npy", "y.npy", "subjects.npy"])
    x = np.load(pmcd_dir / "X.npy", mmap_mode="r")
    labels = labels_from_one_hot(pmcd_dir / "y.npy")
    subjects = integer_subjects(pmcd_dir / "subjects.npy")

    if x.shape != (3455, 1000, 7, 1):
        raise ValueError(f"Unexpected official PMCD X shape: {x.shape}")
    if np.bincount(labels, minlength=3).tolist() != EXPECTED_PMCD_COUNTS:
        raise ValueError("PMCD label counts do not match official release")
    if len(np.unique(subjects)) != 49:
        raise ValueError("Expected 49 PMCD patients")

    # Select 5 common channels and resample
    windows = np.empty((len(x), WINDOW_POINTS, len(CHANNELS)), dtype=np.float32)
    for i in range(len(x)):
        full = np.asarray(x[i, :, :, 0], dtype=np.float32)
        windows[i] = resample_250_to_64(full[:, PMCD_CHANNEL_INDICES])

    np.savez(destination, x=windows, y=labels, subjects=subjects)
    return PainData("PMCD", windows, labels, subjects)


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------

def channel_stats(data: PainData, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std on given indices."""
    selected = np.asarray(data.x[indices], dtype=np.float32)
    mean = selected.mean(axis=(0, 1))
    std = selected.std(axis=(0, 1))
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


# -----------------------------------------------------------------------------
# Fold generation
# -----------------------------------------------------------------------------

def build_random_outer_folds(
    target: PainData, n_splits: int = 5, fold_seed: int | None = None
) -> tuple[dict[int, list[str]], int]:
    """Generate random patient-disjoint outer folds using StratifiedGroupKFold."""
    if fold_seed is None:
        fold_seed = int(random.SystemRandom().randint(0, 2**31 - 1))
        print(f"No --fold-seed given: generated fold_seed={fold_seed} for this run.", flush=True)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=fold_seed)
    folds: dict[int, list[str]] = {}
    for fold_number, (_, test_idx) in enumerate(
        splitter.split(np.zeros(len(target.y)), target.y, target.subjects), start=1
    ):
        fold_subjects = sorted(set(target.subjects[test_idx].tolist()), key=lambda s: int(s))
        folds[fold_number] = fold_subjects
    return folds, fold_seed


def build_loso_folds(target: PainData) -> dict[int, list[str]]:
    """Leave-One-Subject-Out: one fold per PMCD patient."""
    unique_subjects = sorted({int(s) for s in np.unique(target.subjects)})
    return {subject_id: [str(subject_id)] for subject_id in unique_subjects}


def validate_outer_folds(target: PainData, test_subject_folds: dict[int, list]) -> None:
    all_expected = {str(v) for values in test_subject_folds.values() for v in values}
    actual = set(np.unique(target.subjects))
    if all_expected != actual:
        raise ValueError(f"Folds don't cover PMCD subjects: missing={actual-all_expected}, extra={all_expected-actual}")
    flattened = [v for values in test_subject_folds.values() for v in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("A PMCD patient appears in multiple outer test folds")


def split_target(target: PainData, seed: int, fold: int, test_subject_folds: dict[int, list]):
    """Split PMCD data into train/val/test for given seed and outer fold."""
    test_subjects = np.asarray(test_subject_folds[fold]).astype(str)
    test = np.flatnonzero(np.isin(target.subjects, test_subjects))
    outer_train = np.flatnonzero(~np.isin(target.subjects, test_subjects))

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + fold)
    train_relative, val_relative = next(
        splitter.split(np.zeros(len(outer_train)), target.y[outer_train], target.subjects[outer_train])
    )
    train, val = outer_train[train_relative], outer_train[val_relative]

    train_subjects = set(target.subjects[train])
    val_subjects = set(target.subjects[val])
    test_subjects_actual = set(target.subjects[test])
    if train_subjects & val_subjects or train_subjects & test_subjects_actual or val_subjects & test_subjects_actual:
        raise RuntimeError("PMCD participant leakage detected")
    return train, val, test


# -----------------------------------------------------------------------------
# Output serialization
# -----------------------------------------------------------------------------

def save_fold_assignment(output_dir: Path, fixed_folds: bool, fold_seed: int | None, test_subject_folds: dict) -> None:
    (output_dir / "fold_assignment.json").write_text(
        json.dumps(
            {
                "fixed_folds": bool(fixed_folds),
                "fold_seed": fold_seed,
                "test_subject_folds": {str(k): v for k, v in test_subject_folds.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )