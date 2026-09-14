#!/usr/bin/env python3
"""
PMCD-Only Single-Model Training (K-Fold)
=========================================
Trains a 3-class GradientFusionTransformer on PMCD only (no PMED pretraining).

Evaluation modes:
- Default: 5 randomized outer folds (StratifiedGroupKFold), 3 seeds = 15 runs
- --fixed-folds: Original fixed patient assignment (reproduces REFERENCE_RESULT)
- --fold-seed: Reproducible random fold assignment
- --loso: Leave-One-Subject-Out (49 folds x 3 seeds = 147 runs)
- --quick: Single seed/fold, <=3 epochs for pipeline testing

Example:
    python train_single_kfold.py --pmcd-dir /path/to/pmcd_np --output-dir results
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW

# Local modules
sys.path.insert(0, str(Path(__file__).parent))
from pmcd_only.config import (
    CHANNELS, DEFAULT_EPOCHS, DEFAULT_PATIENCE, DEFAULT_BATCH_SIZE,
    DEFAULT_LR, DEFAULT_WEIGHT_DECAY, DEFAULT_LOADER_WORKERS, SEEDS,
    FIXED_TEST_SUBJECTS, REFERENCE_RESULT, KAGGLE_INPUT_ROOT, KAGGLE_WORKING_DIR,
)
from pmcd_only.data_utils import (
    PainData, find_kaggle_input_dir, build_pmcd_common5, validate_outer_folds,
    split_target, build_random_outer_folds, build_loso_folds, save_fold_assignment,
)
from pmcd_only.model import (
    GradientFusionTransformer, BalancedFocalLoss, autocast_context,
    augment_signal,
)
from pmcd_only.train_utils import (
    WindowDataset, make_loader, predict_probabilities,
    select_validation_calibration_simple, apply_calibration,
    calculate_metrics,
)
from pmcd_only.data_utils import channel_stats
from pmcd_only.eval_utils import summarize, build_confusion_matrix_report


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmcd-dir", type=Path, default=None,
                        help="Directory with PMCD X.npy, y.npy, subjects.npy (auto-detected on Kaggle)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: /kaggle/working/pmcd_single_kfold)")
    parser.add_argument("--seeds", default="42,2026,3407")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--loader-workers", type=int, default=DEFAULT_LOADER_WORKERS)
    parser.add_argument("--quick", action="store_true", help="Pipeline test: 1 seed/fold, <=3 epochs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fixed-folds", action="store_true",
                        help="Use original fixed patient assignment (reproduces REFERENCE_RESULT)")
    parser.add_argument("--fold-seed", type=int, default=None,
                        help="Random seed for outer fold assignment (ignored with --fixed-folds/--loso)")
    parser.add_argument("--loso", action="store_true", default=False,
                        help="Leave-One-Subject-Out: 49 folds instead of 5")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unrecognized argv: {unknown}", flush=True)

    # Default output dir
    if args.output_dir is None:
        args.output_dir = KAGGLE_WORKING_DIR / "pmcd_single_kfold" if KAGGLE_WORKING_DIR.is_dir() else Path("pmcd_single_kfold")

    # Auto-detect PMCD dir on Kaggle
    if args.pmcd_dir is None:
        detected = find_kaggle_input_dir(["X.npy", "y.npy", "subjects.npy"])
        if detected is None:
            raise FileNotFoundError("Could not find --pmcd-dir and no PMCD data under /kaggle/input")
        print(f"Auto-detected --pmcd-dir = {detected}", flush=True)
        args.pmcd_dir = detected

    return args


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Main training logic
# -----------------------------------------------------------------------------

def fine_tune_target(target: PainData, seed: int, fold: int,
                     train_idx, val_idx, test_idx, args, device):
    """Train from scratch on PMCD train; select epoch/thresholds on val; test once.
    No pretraining — no data leakage."""
    set_seed(seed + fold * 101)
    mean, std = channel_stats(target, train_idx)
    loaders = {
        "train": make_loader(WindowDataset(target, train_idx, mean, std, augment=True), args.batch_size, True, device, args.loader_workers),
        "val": make_loader(WindowDataset(target, val_idx, mean, std, return_index=True), args.batch_size, False, device, args.loader_workers),
        "test": make_loader(WindowDataset(target, test_idx, mean, std, return_index=True), args.batch_size, False, device, args.loader_workers),
    }
    model = GradientFusionTransformer().to(device)
    criterion = BalancedFocalLoss(target.y[train_idx]).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_state, best_score, best_epoch, stale = None, -np.inf, 0, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for signal, labels in loaders["train"]:
            signal, labels = signal.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                logits = model(signal)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        val_labels, val_probs, _ = predict_probabilities(model, loaders["val"], device)
        val_score = f1_score(val_labels, val_probs.argmax(1), average="macro", zero_division=0)
        if val_score > best_score + 1e-4:
            best_state = copy.deepcopy(model.state_dict())
            best_score, best_epoch, stale = float(val_score), epoch, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No checkpoint created")
    model.load_state_dict(best_state)

    # Calibration on validation only
    val_labels, val_tta_probs, _ = predict_probabilities(model, loaders["val"], device, tta=True)
    calibration = select_validation_calibration_simple(val_labels, val_tta_probs)

    # Test evaluation
    test_labels, test_tta_probs, test_ids = predict_probabilities(model, loaders["test"], device, tta=True)
    test_preds = apply_calibration(test_tta_probs, calibration)

    result = {
        "seed": seed, "fold": fold, "approach": "pmcd_only_tta_calibrated",
        "best_epoch": best_epoch, "val_macro_f1_for_early_stopping": best_score,
        "test_subjects": int(len(np.unique(target.subjects[test_idx]))),
        "test_windows": int(len(test_idx)), "seconds": time.perf_counter() - started,
        "calibration": json.dumps(calibration),
        **calculate_metrics(test_labels, test_preds, test_tta_probs, target.subjects[test_ids]),
    }
    predictions = pd.DataFrame({
        "seed": seed, "fold": fold, "sample_index": test_ids,
        "subject": target.subjects[test_ids], "actual": test_labels, "predicted": test_preds,
        "prob_no_pain": test_tta_probs[:, 0], "prob_moderate": test_tta_probs[:, 1],
        "prob_severe": test_tta_probs[:, 2],
    })
    cpu_state = {k: v.cpu() for k, v in best_state.items()}
    return result, predictions, cpu_state, mean, std


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    print(f"ENVIRONMENT: cuda_available={torch.cuda.is_available()}, "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)

    args.pmcd_dir = args.pmcd_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.output_dir / "cache"
    model_dir = args.output_dir / "models"
    pred_dir = args.output_dir / "predictions"
    for d in (cache_dir, model_dir, pred_dir):
        d.mkdir(exist_ok=True)

    if not torch.cuda.is_available() and not args.quick:
        raise RuntimeError("CUDA required for full run. Use --quick for CPU test.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.quick:
        args.seeds = "42"
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 2)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("DATA: PMCD only (no PMED pretraining)", flush=True)
    print(f"  DEVICE: {device}", flush=True)

    target = build_pmcd_common5(args.pmcd_dir, cache_dir, args.overwrite)
    print(f"PMCD: {target.x.shape}, labels={np.bincount(target.y).tolist()}, subjects=49", flush=True)

    # Determine fold assignment
    if args.loso and args.fixed_folds:
        raise ValueError("--loso and --fixed-folds are mutually exclusive")
    if args.loso:
        test_folds = build_loso_folds(target)
        fold_seed = None
        eval_mode = "loso"
        print(f"Mode: LOSO ({len(test_folds)} folds)", flush=True)
    elif args.fixed_folds:
        test_folds = FIXED_TEST_SUBJECTS
        fold_seed = None
        eval_mode = "fixed"
        print("Mode: Fixed folds", flush=True)
    else:
        test_folds, fold_seed = build_random_outer_folds(target, n_splits=5, fold_seed=args.fold_seed)
        eval_mode = "random"
        print(f"Mode: Random folds (fold_seed={fold_seed})", flush=True)

    save_fold_assignment(args.output_dir, args.fixed_folds, fold_seed, test_folds)
    validate_outer_folds(target, test_folds)

    fold_keys = list(test_folds)
    folds = [fold_keys[0]] if args.quick else fold_keys

    # Resume support
    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() and not args.overwrite else []
    completed = {(int(r["seed"]), int(r["fold"])) for r in rows}

    for seed in seeds:
        for fold in folds:
            pred_path = pred_dir / f"seed{seed}_fold{fold}.csv"
            if (seed, fold) in completed and pred_path.exists() and not args.overwrite:
                print(f"RESUME seed={seed} fold={fold}", flush=True)
                continue
            train, val, test = split_target(target, seed, fold, test_folds)
            print(f"SPLIT seed={seed} fold={fold}: train_subj={len(np.unique(target.subjects[train]))} "
                  f"val_subj={len(np.unique(target.subjects[val]))} test_subj={len(np.unique(target.subjects[test]))}", flush=True)
            result, preds, state, mean, std = fine_tune_target(
                target, seed, fold, train, val, test, args, device
            )
            torch.save({
                "state_dict": state, "channel_mean": mean, "channel_std": std,
                "channels": CHANNELS, "classes": ["no_pain", "moderate", "severe"],
                "seed": seed, "fold": fold, "source": "PMCD-only",
            }, model_dir / f"pmcd_single_seed{seed}_fold{fold}.pt")
            preds.to_csv(pred_path, index=False)
            rows = [r for r in rows if not (int(r["seed"]) == seed and int(r["fold"]) == fold)]
            rows.append(result)
            pd.DataFrame(rows).sort_values(["seed", "fold"]).to_csv(result_path, index=False)
            cm = json.loads(result["confusion_matrix"])
            per_class_recall = json.loads(result["per_class_recall"])
            cm_str = " | ".join(
                " ".join(f"{cm[i][j]:4d}({cm[i][j]/sum(cm[i])*100:5.1f}%)" if sum(cm[i])>0 else f"{cm[i][j]:4d}(  0.0%)" for j in range(3))
                for i in range(3)
            )
            print(f"RESULT seed={seed} fold={fold}: acc={result['accuracy']:.4f} "
                  f"bal_acc={result['balanced_accuracy']:.4f} macro_f1={result['macro_f1']:.4f}", flush=True)
            print(f"  CM (Actual->Pred): No_pain  Moderate  Severe", flush=True)
            for i, name in enumerate(["No_pain", "Moderate", "Severe"]):
                row = " ".join(f"{cm[i][j]:4d}({cm[i][j]/sum(cm[i])*100:5.1f}%)" if sum(cm[i])>0 else f"{cm[i][j]:4d}(  0.0%)" for j in range(3))
                print(f"  {name:>8}: {row}", flush=True)
            print(f"  Per-class recall: {per_class_recall}", flush=True)

    fold_frame = pd.DataFrame(rows).sort_values(["seed", "fold"])
    requested = fold_frame[fold_frame.seed.astype(int).isin(seeds) & fold_frame.fold.astype(int).isin(folds)]
    model_desc = "GradientFusionTransformer (3-class) trained on PMCD only"
    summarize(requested, args.output_dir, args.quick, eval_mode, model_desc)

    pred_frames = [
        pd.read_csv(pred_dir / f"seed{s}_fold{f}.csv")
        for s in seeds for f in folds
        if (pred_dir / f"seed{s}_fold{f}.csv").exists()
    ]
    if pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(args.output_dir / "oof_predictions.csv", index=False)

    manifest = {
        "model": model_desc, "source": "None (PMCD-only)", "target": "PMCD",
        "channels": CHANNELS, "pmcd_channel_indices": [0, 1, 3, 4, 6],
        "patient_leakage_control": "PMCD train/val/test subject sets are pairwise disjoint",
        "test_time_adaptation": "4 deterministic views; thresholds on PMCD val only",
        "seeds": seeds, "folds": folds, "eval_mode": eval_mode, "fold_seed": fold_seed,
        "test_subject_folds": {str(k): v for k, v in test_folds.items()},
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Build detailed confusion matrix report (like original script)
    run_label = f"PMCD-only {eval_mode}"
    build_confusion_matrix_report(args.output_dir, run_label)

    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())