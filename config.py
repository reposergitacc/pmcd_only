"""Configuration constants shared across all PMCD-only training scripts."""

from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------------
# Reproducibility constants
# -----------------------------------------------------------------------------

CHANNELS = ["bvp", "eda_e4", "resp", "eda_rb", "emg"]
SEEDS = [42, 2026, 3407]
RATE = 64
WINDOW_POINTS = 4 * RATE  # 256 points at 64 Hz = 4 seconds

# Expected PMCD label distribution (from official release)
EXPECTED_PMCD_COUNTS = [1443, 1527, 485]  # no_pain, moderate, severe

# Fixed outer folds from original verified experiment
# Kept as opt-in (--fixed-folds) for exact reproduction
FIXED_TEST_SUBJECTS = {
    1: [9, 19, 22, 26, 34, 37, 42, 43],
    2: [2, 5, 12, 20, 24, 28, 29, 40, 47, 49],
    3: [3, 7, 11, 15, 21, 30, 32, 33, 35, 41],
    4: [1, 4, 10, 14, 17, 23, 25, 31, 38, 45, 46],
    5: [6, 8, 13, 16, 18, 27, 36, 39, 44, 48],
}

# Reference results from original fixed-fold run (single 3-class model)
REFERENCE_RESULT = {
    "accuracy_mean": 0.63419630935395,
    "balanced_accuracy_mean": 0.553225460940709,
    "macro_f1_mean": 0.5419899039726612,
}

# Kaggle paths
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_DIR = Path("/kaggle/working")

# PMCD channel mapping: official PMCD has 7 channels
# [BVP, EDA-E4, Temperature, Respiration, EDA-RB, BVP-RB, EMG]
# We use 5 common channels: indices [0, 1, 3, 4, 6]
PMCD_CHANNEL_INDICES = [0, 1, 3, 4, 6]

# Model hyperparameters (can be overridden via CLI)
DEFAULT_EPOCHS = 40
DEFAULT_PRETRAIN_EPOCHS = 25
DEFAULT_SUPCON_WARMUP = 10
DEFAULT_PATIENCE = 8
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 7e-4
DEFAULT_WEIGHT_DECAY = 2e-4
DEFAULT_LOADER_WORKERS = 0