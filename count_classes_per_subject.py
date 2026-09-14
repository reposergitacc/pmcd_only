#!/usr/bin/env python3
"""
Count class distribution per subject in PMCD dataset.
Run in Kaggle cell after data pipeline creates /kaggle/working/pmcd_np/
"""

import numpy as np
from pathlib import Path

# Path to PMCD processed data
PMCD_DIR = Path("/kaggle/working/pmcd_np")

# Load data
X = np.load(PMCD_DIR / "X.npy", mmap_mode="r")
y = np.load(PMCD_DIR / "y.npy")
subjects = np.load(PMCD_DIR / "subjects.npy")

# Convert one-hot labels to class indices if needed
if y.ndim == 2:
    y = np.argmax(y, axis=1)

# Ensure subjects are strings
subjects = subjects.astype(str)

# Class names
CLASS_NAMES = ["No pain", "Moderate", "Severe"]

print(f"Total windows: {len(y)}")
print(f"Total subjects: {len(np.unique(subjects))}")
print(f"Overall distribution: {np.bincount(y)}")
print()

# Per-subject counts
print("Per-subject class counts:")
print("-" * 60)
print(f"{'Subject':>8}  {'No pain':>8}  {'Moderate':>8}  {'Severe':>8}  {'Total':>8}")
print("-" * 60)

for subj in sorted(np.unique(subjects), key=lambda x: int(float(x))):
    mask = subjects == subj
    counts = np.bincount(y[mask], minlength=3)
    total = counts.sum()
    print(f"{subj:>8}  {counts[0]:>8}  {counts[1]:>8}  {counts[2]:>8}  {total:>8}")

print("-" * 60)

# Summary stats
print("\nSummary:")
for i, name in enumerate(CLASS_NAMES):
    per_subj = [np.sum(y[subjects == s] == i) for s in np.unique(subjects)]
    print(f"  {name}: min={min(per_subj)}, max={max(per_subj)}, mean={np.mean(per_subj):.1f}, median={np.median(per_subj):.1f}")

# Subjects missing any class
print("\nSubjects missing classes:")
for i, name in enumerate(CLASS_NAMES):
    missing = [s for s in np.unique(subjects) if np.sum(y[subjects == s] == i) == 0]
    if missing:
        print(f"  {name}: {missing}")
    else:
        print(f"  {name}: none")