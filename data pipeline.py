# -*- coding: utf-8 -*-
"""The final conference paper model
"""

"""
KAGGLE CELL 1 -- Prepare PMCD + PMED np files from raw CSVs.

Same computations as the original PMCD/create_np_files.py and
PMED/create_np_files.py, merged into a single cell. The only change from
the originals is that the raw-data directory is passed in directly instead
of being hardcoded to a local "dataset/raw-data" relative path -- since
this now runs as one process instead of two separate script files, the
PMCD and PMED functions/constants below are just prefixed (pmcd_*, pmed_*)
so nothing collides. Windowing, label mapping, filtering, and array shapes
are unchanged.

After running this cell, PMCD_NP_DIR and PMED_NP_DIR point at folders
containing X.npy/y.npy/subjects.npy and X.npy/y_heater.npy/subjects.npy.
Cell 2 (the training script) automatically picks these up because this
cell sets sys.argv before Cell 2 runs.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Raw data locations -- override these two if your dataset is mounted
# under a different path.
# -----------------------------------------------------------------------------
PMCD_RAW_ROOT = Path(
    "/kaggle/input/datasets/mrsyuk1/pain-signals-and-resting-states/pain_signals_and_resting_states"
)
PMED_RAW_ROOT = Path("/kaggle/input/datasets/mrsyuk1/pmedataset")

PMCD_NP_DIR = Path("/kaggle/working/pmcd_np")
PMED_NP_DIR = Path("/kaggle/working/pmed_np")
OVERWRITE = False  # set True to force regeneration even if np files already exist

# -----------------------------------------------------------------------------
# Auto-detect fallback, in case the above paths don't match your mount
# -----------------------------------------------------------------------------
_PMCD_SUBJECT_PATTERN = re.compile(r"^P\d{2}_\d$")
_PMED_FILE_PATTERN = re.compile(r"^S_\d{2}-synchronised-data\.csv$")


def _find_raw_root(search_root: Path, is_match) -> Path | None:
    if not search_root.is_dir():
        return None
    for candidate in sorted(search_root.rglob("*")):
        if not candidate.is_dir():
            continue
        try:
            children = list(candidate.iterdir())
        except PermissionError:
            continue
        if any(is_match(child) for child in children):
            return candidate
    return None


if not PMCD_RAW_ROOT.is_dir():
    detected = _find_raw_root(
        Path("/kaggle/input"), lambda c: c.is_dir() and _PMCD_SUBJECT_PATTERN.match(c.name)
    )
    if detected is None:
        raise FileNotFoundError(
            f"PMCD_RAW_ROOT not found ({PMCD_RAW_ROOT}) and nothing matching P##_# folders "
            "was found under /kaggle/input. Set PMCD_RAW_ROOT explicitly."
        )
    print(f"PMCD_RAW_ROOT not found at default; auto-detected: {detected}", flush=True)
    PMCD_RAW_ROOT = detected

if not PMED_RAW_ROOT.is_dir():
    detected = _find_raw_root(
        Path("/kaggle/input"), lambda c: c.is_file() and _PMED_FILE_PATTERN.match(c.name)
    )
    if detected is None:
        raise FileNotFoundError(
            f"PMED_RAW_ROOT not found ({PMED_RAW_ROOT}) and nothing matching "
            "S_##-synchronised-data.csv was found under /kaggle/input. Set PMED_RAW_ROOT explicitly."
        )
    print(f"PMED_RAW_ROOT not found at default; auto-detected: {detected}", flush=True)
    PMED_RAW_ROOT = detected

print(f"PMCD_RAW_ROOT = {PMCD_RAW_ROOT}", flush=True)
print(f"PMED_RAW_ROOT = {PMED_RAW_ROOT}", flush=True)


def _to_categorical(y, num_classes=None, dtype="float32"):
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.reshape(-1)
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    return np.reshape(categorical, output_shape)


# =============================================================================
# PMCD (PainMonit Clinical Dataset)  -- from PMCD/config.py, read_data.py,
# create_np_files.py, unchanged logic.
# =============================================================================

PMCD_SAMPLING_RATE = 250
PMCD_SENSOR_NAMES = ["Bvp", "Eda_E4", "Tmp", "Resp", "Eda_RB", "Bvp_RB", "Emg"]


def _pmcd_set_index(df):
    df = df.set_index("Seconds")
    df.index = pd.to_timedelta(df.index, unit="s")
    df.index.name = "Secs"
    return df


def _pmcd_read_txt(file_path):
    if not file_path.exists():
        raise FileExistsError(f"File '{file_path}' does not exists.")
    with open(file_path, "r") as f:
        return f.read()


def _pmcd_read_raw_data(subject_id, raw_root: Path):
    data = []
    for i in range(2):
        name = f"P{str(subject_id).zfill(2)}_{i + 1}"
        file_dir = raw_root / name
        file_path = file_dir / f"{name}.csv"

        if not file_path.exists():
            data.append(None)
            continue

        df = pd.read_csv(file_path, sep=";", decimal=",")
        df = _pmcd_set_index(df)

        df_baseline = pd.read_csv(file_dir / f"{name}_runUp.csv", sep=";", decimal=",")
        df_baseline = _pmcd_set_index(df_baseline)

        no_pain_threshold = int(_pmcd_read_txt(file_dir / "noPainThreshold.txt"))
        severe_pain_threshold = int(_pmcd_read_txt(file_dir / "severePainThreshold.txt"))

        data.append({
            "data": df, "baseline": df_baseline,
            "noPainThreshold": no_pain_threshold, "severePainThreshold": severe_pain_threshold,
        })
    return data


def _pmcd_segment(df, window_secs=4, step_size=2, sampling_rate=PMCD_SAMPLING_RATE):
    window_size = int(window_secs * sampling_rate)
    distance = step_size * sampling_rate
    input_data_length = df.shape[0]

    X = []
    for start in range(0, input_data_length, distance):
        end = start + window_size
        if end > input_data_length:
            continue
        X.append(df.values[start:end])
    return np.array(X)


def _pmcd_process_segments(x, columns, selected_sensors=PMCD_SENSOR_NAMES):
    data, labels = [], []
    for i in x:
        pain_labels = i[:, columns.index("Pain labels")]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_label = np.nanmax(pain_labels)
        if np.isnan(max_label):
            continue
        labels.append(int(max_label))
        data.append(i)

    labels = np.array(labels)
    data = np.array(data)
    data = np.stack([data[:, :, columns.index(i)] for i in selected_sensors], axis=-1)
    return data, labels


def create_np_pmcd(raw_root: Path, np_dir: Path, overwrite: bool = False):
    if np_dir.is_dir() and all((np_dir / n).is_file() for n in ("X.npy", "y.npy", "subjects.npy")) and not overwrite:
        print(f"PMCD np-dataset already present at {np_dir}, skipping.", flush=True)
        return

    data_list, labels_list, subjects_list = [], [], []
    print("Create PMCD np files...", flush=True)

    for i in tqdm(range(1, 50)):
        subject_data = _pmcd_read_raw_data(subject_id=i, raw_root=raw_root)

        for repetition in range(2):
            if subject_data[repetition] is None:
                continue

            x = subject_data[repetition]["data"]
            columns = list(x.columns)
            x = _pmcd_segment(x)

            data, labels = _pmcd_process_segments(x, columns=columns)

            mask = labels != 0
            data = data[mask]
            labels = labels[mask]

            x_baseline = _pmcd_segment(subject_data[repetition]["baseline"])
            data_baseline = np.stack(
                [x_baseline[:, :, columns.index(i)] for i in PMCD_SENSOR_NAMES], axis=-1
            )
            labels_baseline = [0] * len(data_baseline)
            subjects_baseline = [i] * len(data_baseline)

            data_list.append(data)
            labels_list.append(labels)
            subjects_list.append([i] * len(data))

            data_list.append(data_baseline)
            labels_list.append(labels_baseline)
            subjects_list.append(subjects_baseline)

    data = np.concatenate(data_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    subjects = np.concatenate(subjects_list, axis=0)
    assert len(data) == len(labels) == len(subjects)

    data = np.nan_to_num(data)
    labels = np.nan_to_num(labels)
    data = data[..., np.newaxis]
    labels = _to_categorical(labels)

    np_dir.mkdir(parents=True, exist_ok=True)
    np.save(np_dir / "X", data)
    np.save(np_dir / "y", labels)
    np.save(np_dir / "subjects", subjects)

    print("PMCD data shape:", data.shape, flush=True)
    print("PMCD labels shape:", labels.shape, flush=True)
    print("PMCD subjects shape:", subjects.shape, flush=True)
    print(f"PMCD np dataset saved under '{np_dir}'.", flush=True)


# =============================================================================
# PMED (PainMonit Experimental Dataset)  -- from PMED/config.py, read_data.py,
# create_np_files.py, unchanged logic.
# =============================================================================

PMED_SAMPLING_RATE = 250
PMED_WINDOW_SECS = 10
PMED_NUM_REPETITIONS = 8
PMED_BASELINE_TEMP = 32
PMED_SENSOR_NAMES = ["Bvp", "Eda_E4", "Resp", "Eda_RB", "Ecg", "Emg"]


def _pmed_crossings_nonzero_neg2pos(data):
    npos = data < 0
    return (npos[:-1] & ~npos[1:]).nonzero()[0]


def _pmed_segment(df, baseline_shift=5):
    X, y_heater, y_covas = [], [], []

    stim = (df["Heater_cleaned"] != PMED_BASELINE_TEMP).astype("int")
    stim[stim == False] = -1
    stim_starts = _pmed_crossings_nonzero_neg2pos(stim.values)
    num_baseline_windows = 0

    window = int(PMED_WINDOW_SECS * PMED_SAMPLING_RATE)
    for start in stim_starts:
        baseline_start = start - (baseline_shift * PMED_SAMPLING_RATE)
        if (
            (num_baseline_windows < PMED_NUM_REPETITIONS)
            and (baseline_start > window)
            and (df["Heater_cleaned"].values[baseline_start - window: baseline_start] == PMED_BASELINE_TEMP).all()
        ):
            X.append(df[PMED_SENSOR_NAMES].values[baseline_start - window: baseline_start])
            y_covas.append(0)
            y_heater.append(0)
            num_baseline_windows += 1

        start += 1
        temp = df["Heater_cleaned"].values[start]
        end = int(start + window)
        if (df["Heater_cleaned"].values[start:end] == temp).all():
            X.append(df[PMED_SENSOR_NAMES].values[start:end])
            y_covas.append(sum(df["COVAS"].values[start:end]))
            y_heater.append(temp)

    temps = np.unique(y_heater)
    conversion = {x: i for i, x in enumerate(temps)}
    y_heater = np.vectorize(conversion.get)(y_heater)

    y_covas = np.array(y_covas)
    y_covas = y_covas / y_covas.max()
    y_covas *= 100
    y_covas = np.array([int(x // 25) + 1 if x > 0 else 0 for x in y_covas])
    y_covas[y_covas == 5] = 4

    X = np.array(X)
    return X, y_heater, y_covas


def create_np_pmed(raw_root: Path, np_dir: Path, overwrite: bool = False):
    if (
        np_dir.is_dir()
        and all((np_dir / n).is_file() for n in ("X.npy", "y_heater.npy", "y_covas.npy", "subjects.npy"))
        and not overwrite
    ):
        print(f"PMED np-dataset already present at {np_dir}, skipping.", flush=True)
        return

    data_list, heater_list, covas_list, subjects_list = [], [], [], []
    print("Create PMED np files...", flush=True)

    file_names = sorted(str(p) for p in raw_root.glob("*.csv"))

    for index, filename in enumerate(tqdm(file_names)):
        subject_data = pd.read_csv(filename, sep=";", decimal=",")
        X, y_heater, y_covas = _pmed_segment(subject_data)

        data_list.append(X)
        heater_list.append(y_heater)
        covas_list.append(y_covas)
        subjects_list.append([index] * X.shape[0])

    data = np.concatenate(data_list, axis=0)
    heater = np.concatenate(heater_list, axis=0)
    covas = np.concatenate(covas_list, axis=0)
    subjects = np.concatenate(subjects_list, axis=0)
    assert len(data) == len(heater) == len(covas) == len(subjects)

    data = np.nan_to_num(data)
    data = data[..., np.newaxis]
    heater = _to_categorical(heater)
    covas = _to_categorical(covas)

    np_dir.mkdir(parents=True, exist_ok=True)
    np.save(np_dir / "X", data)
    np.save(np_dir / "y_heater", heater)
    np.save(np_dir / "y_covas", covas)
    np.save(np_dir / "subjects", subjects)

    print("PMED data shape:", data.shape, flush=True)
    print("PMED heater shape:", heater.shape, flush=True)
    print("PMED covas shape:", covas.shape, flush=True)
    print("PMED subjects shape:", subjects.shape, flush=True)
    print(f"PMED np dataset saved under '{np_dir}'.", flush=True)


# =============================================================================
# Run both, then hand off to Cell 2 via sys.argv
# =============================================================================

create_np_pmcd(PMCD_RAW_ROOT, PMCD_NP_DIR, overwrite=OVERWRITE)
create_np_pmed(PMED_RAW_ROOT, PMED_NP_DIR, overwrite=OVERWRITE)

# Cell 2 (the training script) is unmodified and reads --pmcd-dir/--pmed-dir
# via argparse; setting sys.argv here means Cell 2 needs no edits at all.
sys.argv = [sys.argv[0], "--pmcd-dir", str(PMCD_NP_DIR), "--pmed-dir", str(PMED_NP_DIR)]
print(f"\nReady. Cell 2 will use --pmcd-dir {PMCD_NP_DIR} --pmed-dir {PMED_NP_DIR}", flush=True)
print("Add extra flags manually if needed, e.g.:", flush=True)
print('  sys.argv += ["--quick"]   # before running Cell 2, for a fast pipeline test', flush=True)

"""## Confusion matrix code"""

"""
Confusion matrices + classification metrics for both
evaluation modes: the randomized-fold run and the LOSO run.

Prerequisite: you've already run cell2_train_model.py twice, once per mode,
each pointed at its OWN --output-dir so the two runs don't overwrite each
other, e.g.:

    !python cell2_train_model.py --pmcd-dir ... --pmed-dir ... \
        --output-dir /kaggle/working/results_random_folds

    !python cell2_train_model.py --pmcd-dir ... --pmed-dir ... --loso \
        --output-dir /kaggle/working/results_loso

Each run writes oof_predictions.csv: pooled out-of-fold test predictions --
every test window from every fold/seed of that run, each one predicted by a
model that never saw that window (or that window's patient, for train/val)
during training. This cell reads that file for each run's output directory
and computes, on the full pooled set:
  - A confusion matrix (raw counts + row-normalized), plotted and saved as
    a PNG in that same output directory.
  - accuracy, balanced accuracy, macro F1, subject-level macro F1 (mean/std)
  - per-class precision/recall/F1/support (sklearn classification_report)
  - severe-pain recall, severe-pain AUPRC, expected calibration error (ECE)
    -- the same metrics the training script's calculate_metrics() reports
    per fold, computed here on the pooled predictions instead.

Full metrics are also saved as JSON in each output directory. Adjust
RANDOM_FOLDS_OUTPUT_DIR / LOSO_OUTPUT_DIR below to match wherever you
pointed --output-dir for each run. Section 1 and Section 2 below are
independent -- either can be run/re-run on its own once its input
directory has an oof_predictions.csv, and you can split them into two
separate cells at the marked boundary if you'd rather run them apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

CLASS_NAMES = ("No pain", "Moderate", "Severe")


def _expected_calibration_error(labels, probabilities, bins=10):
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            result += selected.mean() * abs(
                (predicted[selected] == labels[selected]).mean() - confidence[selected].mean()
            )
    return float(result)


def build_confusion_matrix_report(output_dir: Path, run_label: str) -> dict:
    """Load an oof_predictions.csv written by cell2_train_model.py and produce
    a confusion matrix (plot + saved PNG) plus a full metrics report (saved
    as JSON next to it). Returns the metrics dict.
    """
    output_dir = Path(output_dir)
    oof_path = output_dir / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(
            f"{oof_path} not found. Run cell2_train_model.py with --output-dir {output_dir} "
            f"first (for the '{run_label}' evaluation mode)."
        )
    oof = pd.read_csv(oof_path)

    y_true = oof["actual"].to_numpy()
    y_pred = oof["predicted"].to_numpy()
    probabilities = oof[["prob_no_pain", "prob_moderate", "prob_severe"]].to_numpy()
    subjects = oof["subject"].to_numpy()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_normalized = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

    subject_scores = [
        f1_score(y_true[subjects == s], y_pred[subjects == s], average="macro", zero_division=0)
        for s in np.unique(subjects)
    ]
    severe = (y_true == 2).astype(np.int64)

    metrics = {
        "run_label": run_label,
        "n_predictions": int(len(oof)),
        "n_subjects": int(oof["subject"].nunique()),
        "n_folds": int(oof["fold"].nunique()),
        "n_seeds": int(oof["seed"].nunique()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "subject_macro_f1_mean": float(np.mean(subject_scores)),
        "subject_macro_f1_std": float(np.std(subject_scores, ddof=1)) if len(subject_scores) > 1 else None,
        "severe_recall": float(recall_score(severe, y_pred == 2, zero_division=0)),
        "severe_auprc": float(average_precision_score(severe, probabilities[:, 2])),
        "ece": _expected_calibration_error(y_true, probabilities),
        "confusion_matrix_counts": cm.tolist(),
        "confusion_matrix_row_normalized": cm_normalized.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
    }

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    im = ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right")
    ax.set_yticks(range(3))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{run_label}\npooled out-of-fold test predictions, n={len(oof)}", fontsize=10)
    for i in range(3):
        for j in range(3):
            color = "white" if cm_normalized[i, j] > 0.5 else "black"
            ax.text(
                j, i, f"{cm[i, j]}\n({cm_normalized[i, j] * 100:.1f}%)",
                ha="center", va="center", color=color, fontsize=9,
            )
    fig.colorbar(im, ax=ax, label="Row-normalized fraction")
    fig.tight_layout()

    slug = run_label.lower().replace(" ", "_").replace("-", "_")
    plot_path = output_dir / f"confusion_matrix_{slug}.png"
    fig.savefig(plot_path, dpi=150)
    plt.show()
    plt.close(fig)

    report_path = output_dir / f"confusion_matrix_report_{slug}.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        f"[{run_label}] n={metrics['n_predictions']} subjects={metrics['n_subjects']} "
        f"folds={metrics['n_folds']} seeds={metrics['n_seeds']}",
        flush=True,
    )
    print(
        f"[{run_label}] accuracy={metrics['accuracy']:.4f}  "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}",
        flush=True,
    )
    print(
        f"[{run_label}] severe_recall={metrics['severe_recall']:.4f}  "
        f"severe_auprc={metrics['severe_auprc']:.4f}  ece={metrics['ece']:.4f}",
        flush=True,
    )
    print(f"[{run_label}] saved: {plot_path.name}, {report_path.name}", flush=True)
    return metrics

# To ignore warinings
import warnings
warnings.filterwarnings('ignore')
