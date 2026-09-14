#!/usr/bin/env python3
"""
PMCD-Only Hierarchical Two-Stage Training (LOSO, No Validation Split)
=====================================================================
Leave-One-Subject-Out with ONLY train/test splits per fold.
- No inner validation split
- No early stopping (fixed epochs)
- No validation-based calibration (uses fixed thresholds)
- Trains both stages from scratch per fold on 48 subjects, tests on 1
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from sklearn.metrics import f1_score, recall_score

sys.path.insert(0, str(Path(__file__).parent))
from pmcd_only.config import (
    CHANNELS, DEFAULT_EPOCHS, DEFAULT_PATIENCE, DEFAULT_BATCH_SIZE,
    DEFAULT_LR, DEFAULT_WEIGHT_DECAY, DEFAULT_LOADER_WORKERS, SEEDS,
    FIXED_TEST_SUBJECTS, REFERENCE_RESULT, KAGGLE_INPUT_ROOT, KAGGLE_WORKING_DIR,
)
from pmcd_only.data_utils import (
    PainData, find_kaggle_input_dir, build_pmcd_common5, validate_outer_folds,
    split_target, build_loso_folds, save_fold_assignment,
)
from pmcd_only.model import (
    GradientFusionTransformer, BalancedFocalLoss, autocast_context,
    augment_signal,
)
from pmcd_only.train_utils import (
    WindowDataset, make_loader, predict_probabilities, predict_hierarchical_probabilities,
    apply_calibration, calculate_metrics,
)
from pmcd_only.data_utils import channel_stats
from pmcd_only.eval_utils import summarize, build_confusion_matrix_report


# Fixed calibration thresholds (no validation-based selection)
FIXED_CALIBRATION = {"pain_threshold": 0.5, "severe_threshold": 0.35}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmcd-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--loader-workers", type=int, default=DEFAULT_LOADER_WORKERS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fixed-folds", action="store_true",
                        help="Use 5 fixed folds instead of LOSO")
    parser.add_argument("--fold-seed", type=int, default=None)
    parser.add_argument("--loso", action="store_true", default=True)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown argv: {unknown}", flush=True)

    if args.output_dir is None:
        args.output_dir = KAGGLE_WORKING_DIR / "pmcd_hierarchical_loso_noval" if KAGGLE_WORKING_DIR.is_dir() else Path("pmcd_hierarchical_loso_noval")

    if args.pmcd_dir is None:
        detected = find_kaggle_input_dir(["X.npy", "y.npy", "subjects.npy"])
        if detected is None:
            raise FileNotFoundError("No PMCD data found")
        print(f"Auto-detected --pmcd-dir = {detected}", flush=True)
        args.pmcd_dir = detected

    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_binary_stage(model, loader, criterion, args, device, epochs: int):
    """Train one binary stage for fixed epochs (no early stopping)."""
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for signal, labels in loader:
            signal, labels = signal.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                logits = model(signal)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        print(f"  Epoch {epoch}/{epochs} loss={np.mean(losses):.5f}", flush=True)
    return {k: v.cpu() for k, v in model.state_dict().items()}


def fine_tune_hierarchical_no_val(target: PainData, seed: int, fold: int,
                                   train_idx, test_idx, args, device):
    """Hierarchical train from scratch: two independent binary stages.
    No validation split — fixed epochs, fixed calibration thresholds."""
    set_seed(seed + fold * 101)
    started = time.perf_counter()

    # ---- Stage 1: no_pain vs pain (all windows) ----
    s1_labels = (target.y > 0).astype(np.int64)
    s1_data = PainData("PMCD-s1", target.x, s1_labels, target.subjects)
    mean1, std1 = channel_stats(s1_data, train_idx)
    s1_loader = make_loader(
        WindowDataset(s1_data, train_idx, mean1, std1, augment=True),
        args.batch_size, True, device, args.loader_workers
    )
    s1_model = GradientFusionTransformer(classes=2).to(device)
    s1_criterion = BalancedFocalLoss(s1_labels[train_idx], classes=2).to(device)
    s1_state = train_binary_stage(s1_model, s1_loader, s1_criterion, args, device, args.epochs)
    s1_model.load_state_dict(s1_state)

    # ---- Stage 2: moderate vs severe (pain-present only) ----
    pain_train = train_idx[np.isin(target.y[train_idx], [1, 2])]
    if len(pain_train) == 0:
        raise RuntimeError(f"seed={seed} fold={fold}: no pain windows in train for stage 2")
    s2_labels = np.where(target.y == 2, 1, 0).astype(np.int64)
    s2_data = PainData("PMCD-s2", target.x, s2_labels, target.subjects)
    mean2, std2 = channel_stats(s2_data, pain_train)
    s2_loader = make_loader(
        WindowDataset(s2_data, pain_train, mean2, std2, augment=True),
        args.batch_size, True, device, args.loader_workers
    )
    s2_model = GradientFusionTransformer(classes=2).to(device)
    s2_criterion = BalancedFocalLoss(s2_labels[pain_train], classes=2).to(device)
    s2_state = train_binary_stage(s2_model, s2_loader, s2_criterion, args, device, args.epochs)
    s2_model.load_state_dict(s2_state)

    # ---- Test evaluation (with TTA) ----
    test_eval = (
        make_loader(WindowDataset(target, test_idx, mean1, std1, return_index=True), args.batch_size, False, device, args.loader_workers),
        make_loader(WindowDataset(target, test_idx, mean2, std2, return_index=True), args.batch_size, False, device, args.loader_workers),
    )
    test_labels, test_probs, test_ids = predict_hierarchical_probabilities(s1_model, s2_model, *test_eval, device, tta=True)
    test_preds = apply_calibration(test_probs, FIXED_CALIBRATION)

    result = {
        "seed": seed, "fold": fold, "approach": "hierarchical_pmcd_only_noval_fixed_calib",
        "stage2_train_windows": int(len(pain_train)),
        "test_subjects": int(len(np.unique(target.subjects[test_idx]))),
        "test_windows": int(len(test_idx)), "seconds": time.perf_counter() - started,
        "calibration": json.dumps(FIXED_CALIBRATION),
        **calculate_metrics(test_labels, test_preds, test_probs, target.subjects[test_ids]),
    }
    predictions = pd.DataFrame({
        "seed": seed, "fold": fold, "sample_index": test_ids,
        "subject": target.subjects[test_ids], "actual": test_labels, "predicted": test_preds,
        "prob_no_pain": test_probs[:, 0], "prob_moderate": test_probs[:, 1], "prob_severe": test_probs[:, 2],
    })
    checkpoint = {
        "stage1_state_dict": {k: v.cpu() for k, v in s1_state.items()},
        "stage2_state_dict": {k: v.cpu() for k, v in s2_state.items()},
    }
    normalization = {"stage1": (mean1, std1), "stage2": (mean2, std2)}
    return result, predictions, checkpoint, normalization


def main() -> int:
    args = parse_args()
    print(f"ENVIRONMENT: cuda={torch.cuda.is_available()}, "
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
        raise RuntimeError("CUDA required. Use --quick for CPU test.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.quick:
        args.seeds = "42"
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 2)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("DATA: PMCD only (hierarchical two-stage, no validation split)", flush=True)
    print(f"  DEVICE: {device}", flush=True)

    target = build_pmcd_common5(args.pmcd_dir, cache_dir, args.overwrite)
    print(f"PMCD: {target.x.shape}, labels={np.bincount(target.y).tolist()}, subjects=49", flush=True)

    if args.loso and args.fixed_folds:
        raise ValueError("--loso and --fixed-folds are mutually exclusive")
    if args.loso:
        test_folds = build_loso_folds(target)
        fold_seed = None
        eval_mode = "loso"
        print(f"Mode: LOSO ({len(test_folds)} folds x {len(seeds)} seeds = {len(test_folds)*len(seeds)} runs)", flush=True)
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

    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() and not args.overwrite else []
    completed = {(int(r["seed"]), int(r["fold"])) for r in rows}

    for seed in seeds:
        for fold in folds:
            pred_path = pred_dir / f"seed{seed}_fold{fold}.csv"
            if (seed, fold) in completed and pred_path.exists() and not args.overwrite:
                print(f"RESUME seed={seed} fold={fold}", flush=True)
                continue
            # Simple train/test split: test = fold's subject(s), train = all others
            test_subjects = np.asarray(test_folds[fold]).astype(str)
            test = np.flatnonzero(np.isin(target.subjects, test_subjects))
            train = np.flatnonzero(~np.isin(target.subjects, test_subjects))
            print(f"SPLIT seed={seed} fold={fold}: train_subj={len(np.unique(target.subjects[train]))} "
                  f"test_subj={len(np.unique(target.subjects[test]))}", flush=True)
            result, preds, checkpoint, norm = fine_tune_hierarchical_no_val(
                target, seed, fold, train, test, args, device
            )
            torch.save({
                **checkpoint, "normalization": norm,
                "channels": CHANNELS, "classes": ["no_pain", "moderate", "severe"],
                "seed": seed, "fold": fold, "source": "PMCD-only",
            }, model_dir / f"pmcd_hier_noval_seed{seed}_fold{fold}.pt")
            preds.to_csv(pred_path, index=False)
            rows = [r for r in rows if not (int(r["seed"]) == seed and int(r["fold"]) == fold)]
            rows.append(result)
            pd.DataFrame(rows).sort_values(["seed", "fold"]).to_csv(result_path, index=False)
            cm = json.loads(result["confusion_matrix"])
            per_class_recall = json.loads(result["per_class_recall"])
            print(f"RESULT seed={seed} fold={fold}: acc={result['accuracy']:.4f} "
                  f"bal_acc={result['balanced_accuracy']:.4f} macro_f1={result['macro_f1']:.4f}", flush=True)
            print(f"  CM (Actual->Pred): No_pain  Moderate  Severe", flush=True)
            for i, name in enumerate(["No_pain", "Moderate", "Severe"]):
                row = " ".join(f"{cm[i][j]:4d}({cm[i][j]/sum(cm[i])*100:5.1f}%)" if sum(cm[i])>0 else f"{cm[i][j]:4d}(  0.0%)" for j in range(3))
                print(f"  {name:>8}: {row}", flush=True)
            print(f"  Per-class recall: {per_class_recall}", flush=True)

    fold_frame = pd.DataFrame(rows).sort_values(["seed", "fold"])
    requested = fold_frame[fold_frame.seed.astype(int).isin(seeds) & fold_frame.fold.astype(int).isin(folds)]
    model_desc = ("Hierarchical two-stage GradientFusionTransformer (no val split, fixed calibration) "
                  "trained on PMCD only")
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
        "patient_leakage_control": "Test subject completely held out; train on remaining 48",
        "calibration": "Fixed thresholds (pain=0.5, severe=0.35) — no validation",
        "seeds": seeds, "folds": folds, "eval_mode": eval_mode, "fold_seed": fold_seed,
        "test_subject_folds": {str(k): v for k, v in test_folds.items()},
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Build confusion matrix report
    run_label = f"PMCD-only hierarchical no-val {eval_mode}"
    build_confusion_matrix_report(args.output_dir, run_label)

    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())