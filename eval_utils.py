"""Evaluation utilities: confusion matrices, summary generation, result saving."""

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

from .config import CHANNELS, REFERENCE_RESULT


CLASS_NAMES = ("No pain", "Moderate", "Severe")


def expected_calibration_error(labels, probabilities, bins=10):
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
    """Load oof_predictions.csv and produce confusion matrix plot + metrics JSON."""
    output_dir = Path(output_dir)
    oof_path = output_dir / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"{oof_path} not found. Run training first.")

    oof = pd.read_csv(oof_path)
    y_true = oof["actual"].to_numpy()
    y_pred = oof["predicted"].to_numpy()
    probs = oof[["prob_no_pain", "prob_moderate", "prob_severe"]].to_numpy()
    subjects = oof["subject"].to_numpy()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

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
        "severe_auprc": float(average_precision_score(severe, probs[:, 2])),
        "ece": expected_calibration_error(y_true, probs),
        "confusion_matrix_counts": cm.tolist(),
        "confusion_matrix_row_normalized": cm_norm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
    }

    # Plot
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right")
    ax.set_yticks(range(3))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{run_label}\npooled out-of-fold test predictions, n={len(oof)}", fontsize=10)
    for i in range(3):
        for j in range(3):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)",
                    ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, label="Row-normalized fraction")
    fig.tight_layout()

    slug = run_label.lower().replace(" ", "_").replace("-", "_")
    plot_path = output_dir / f"confusion_matrix_{slug}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    report_path = output_dir / f"confusion_matrix_report_{slug}.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[{run_label}] n={metrics['n_predictions']} subjects={metrics['n_subjects']} "
          f"folds={metrics['n_folds']} seeds={metrics['n_seeds']}", flush=True)
    print(f"[{run_label}] accuracy={metrics['accuracy']:.4f}  "
          f"balanced_accuracy={metrics['balanced_accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}", flush=True)
    print(f"[{run_label}] severe_recall={metrics['severe_recall']:.4f}  "
          f"severe_auprc={metrics['severe_auprc']:.4f}  ece={metrics['ece']:.4f}", flush=True)
    print(f"[{run_label}] saved: {plot_path.name}, {report_path.name}", flush=True)
    return metrics


def summarize(
    fold_frame: pd.DataFrame, output_dir: Path, quick: bool,
    eval_mode: str, model_description: str
) -> dict:
    """Generate summary.json and summary.csv from fold results."""
    numeric = [
        "accuracy", "balanced_accuracy", "macro_f1", "subject_macro_f1_mean",
        "severe_recall", "severe_auprc", "ece",
    ]
    summary = {
        "model": model_description,
        "source_dataset": "None (PMCD-only training)",
        "target_dataset": "PMCD three-class (train/validation/test)",
        "channels": CHANNELS,
        "eval_mode": eval_mode,
        "seeds": sorted(fold_frame.seed.unique().astype(int).tolist()),
        "folds": sorted(fold_frame.fold.unique().astype(int).tolist()),
        "runs": int(len(fold_frame)),
        "quick": quick,
    }
    for metric in numeric:
        summary[f"{metric}_mean"] = float(fold_frame[metric].mean())
        summary[f"{metric}_std"] = float(fold_frame[metric].std(ddof=1)) if len(fold_frame) > 1 else None

    if not quick and eval_mode == "fixed" and len(fold_frame) == 15:
        summary["reference_result"] = REFERENCE_RESULT
        summary["difference_from_reference"] = {
            key: summary[key] - value for key, value in REFERENCE_RESULT.items()
        }
    elif not quick:
        mode_desc = {
            "random": "Outer folds were randomized (see fold_assignment.json)",
            "loso": "Leave-One-Subject-Out evaluation (see fold_assignment.json)"
        }.get(eval_mode, eval_mode)
        summary["note"] = f"This run used {model_description}. {mode_desc}. Not directly comparable to REFERENCE_RESULT."

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    return summary


def save_results(
    output_dir: Path, rows: list, seeds: list, folds: list,
    prediction_dir: Path, model_name: str
):
    """Save fold_results.csv, oof_predictions.csv, and run_manifest.json."""
    result_path = output_dir / "fold_results.csv"
    fold_frame = pd.DataFrame(rows).sort_values(["seed", "fold"])
    requested = fold_frame[fold_frame.seed.astype(int).isin(seeds) & fold_frame.fold.astype(int).isin(folds)]
    requested.to_csv(result_path, index=False)

    pred_frames = [
        pd.read_csv(prediction_dir / f"seed{seed}_fold{fold}.csv")
        for seed in seeds for fold in folds
        if (prediction_dir / f"seed{seed}_fold{fold}.csv").exists()
    ]
    if pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(output_dir / "oof_predictions.csv", index=False)

    # manifest would be added by main script