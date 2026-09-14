"""Training loop utilities: datasets, loaders, prediction, calibration, metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset

from .config import DEFAULT_BATCH_SIZE, DEFAULT_LOADER_WORKERS
from .model import autocast_context, augment_signal


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class WindowDataset(Dataset):
    def __init__(
        self, data, indices: np.ndarray, mean: np.ndarray, std: np.ndarray,
        return_index: bool = False, augment: bool = False
    ):
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = mean[np.newaxis, :]
        self.std = std[np.newaxis, :]
        self.return_index = return_index
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        signal = np.asarray(self.data.x[index], dtype=np.float32)
        signal = np.nan_to_num((signal - self.mean) / self.std).T  # (C, T)
        tensor = torch.from_numpy(signal)
        if self.augment:
            tensor = augment_signal(tensor)
        label = torch.tensor(self.data.y[index], dtype=torch.long)
        if self.return_index:
            return tensor, label, index
        return tensor, label


def make_loader(dataset, batch_size: int, shuffle: bool, device: torch.device, num_workers: int = 0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


# -----------------------------------------------------------------------------
# Prediction with TTA
# -----------------------------------------------------------------------------

@torch.inference_mode()
def predict_probabilities(model, loader, device, tta: bool = False):
    """Standard prediction with optional 4-view TTA."""
    model.eval()
    labels, indices, probabilities = [], [], []
    for signal, target, index in loader:
        signal = signal.to(device)
        views = [signal]
        if tta:
            left = torch.cat([signal[:, :, :1].expand(-1, -1, 2), signal[:, :, :-2]], dim=2)
            right = torch.cat([signal[:, :, 2:], signal[:, :, -1:].expand(-1, -1, 2)], dim=2)
            smooth = F.avg_pool1d(F.pad(signal, (1, 1), mode="replicate"), 3, stride=1)
            views.extend([left, right, smooth])
        view_probs = []
        for view in views:
            with autocast_context(device):
                view_probs.append(torch.softmax(model(view), dim=1))
        probabilities.append(torch.stack(view_probs).mean(dim=0).cpu().numpy())
        labels.append(target.numpy())
        indices.append(index.numpy())
    return np.concatenate(labels), np.concatenate(probabilities), np.concatenate(indices)


@torch.inference_mode()
def predict_hierarchical_probabilities(stage1_model, stage2_model, stage1_loader, stage2_loader, device, tta: bool = False):
    """Combine two binary-stage models into 3-class probabilities via chain rule."""
    stage1_model.eval()
    stage2_model.eval()

    def _tta_prob(model, signal):
        views = [signal]
        if tta:
            left = torch.cat([signal[:, :, :1].expand(-1, -1, 2), signal[:, :, :-2]], dim=2)
            right = torch.cat([signal[:, :, 2:], signal[:, :, -1:].expand(-1, -1, 2)], dim=2)
            smooth = F.avg_pool1d(F.pad(signal, (1, 1), mode="replicate"), 3, stride=1)
            views.extend([left, right, smooth])
        view_probs = []
        for view in views:
            with autocast_context(device):
                view_probs.append(torch.softmax(model(view), dim=1))
        return torch.stack(view_probs).mean(dim=0)

    labels_out, indices_out, combined = [], [], []
    for (s1, t1, i1), (s2, t2, i2) in zip(stage1_loader, stage2_loader):
        if not torch.equal(i1, i2) or not torch.equal(t1, t2):
            raise RuntimeError("Stage1/Stage2 eval loaders out of sync")
        s1, s2 = s1.to(device), s2.to(device)
        p_pain = _tta_prob(stage1_model, s1)[:, 1]
        p_sev_given_pain = _tta_prob(stage2_model, s2)[:, 1]

        p_no_pain = 1.0 - p_pain
        p_moderate = p_pain * (1.0 - p_sev_given_pain)
        p_severe = p_pain * p_sev_given_pain
        combined.append(torch.stack([p_no_pain, p_moderate, p_severe], dim=1).cpu().numpy())
        labels_out.append(t1.numpy())
        indices_out.append(i1.numpy())
    return np.concatenate(labels_out), np.concatenate(combined), np.concatenate(indices_out)


# -----------------------------------------------------------------------------
# Calibration threshold selection (validation-only)
# -----------------------------------------------------------------------------

def select_validation_calibration(labels, probabilities, severe_recall_floor_ratio: float = 0.95):
    """Choose hierarchical thresholds on validation data.
    
    Among threshold pairs within `severe_recall_floor_ratio` of best macro-F1,
    pick the one with highest severe recall.
    """
    pain_prob = probabilities[:, 1] + probabilities[:, 2]
    severe_ratio = probabilities[:, 2] / pain_prob.clip(1e-6)

    candidates = []
    for pain_thresh in np.linspace(0.30, 0.70, 17):
        for severe_thresh in np.linspace(0.15, 0.70, 23):
            predicted = np.where(
                pain_prob < pain_thresh, 0,
                np.where(severe_ratio >= severe_thresh, 2, 1)
            )
            macro_f1 = f1_score(labels, predicted, average="macro", zero_division=0)
            sev_recall = recall_score((labels == 2).astype(int), predicted == 2, zero_division=0)
            candidates.append((macro_f1, sev_recall, float(pain_thresh), float(severe_thresh)))

    best_macro = max(c[0] for c in candidates)
    floor = best_macro * severe_recall_floor_ratio
    eligible = [c for c in candidates if c[0] >= floor]
    best = max(eligible, key=lambda c: c[1])
    return {"pain_threshold": best[2], "severe_threshold": best[3]}


# Original simpler calibration (used in single-model versions)
def select_validation_calibration_simple(labels, probabilities):
    best = (-np.inf, 0.5, 0.35)
    pain_prob = probabilities[:, 1] + probabilities[:, 2]
    severe_ratio = probabilities[:, 2] / pain_prob.clip(1e-6)
    for pain_thresh in np.linspace(0.30, 0.70, 17):
        for severe_thresh in np.linspace(0.15, 0.70, 23):
            predicted = np.where(
                pain_prob < pain_thresh, 0,
                np.where(severe_ratio >= severe_thresh, 2, 1)
            )
            score = f1_score(labels, predicted, average="macro", zero_division=0)
            if score > best[0]:
                best = (score, float(pain_thresh), float(severe_thresh))
    return {"pain_threshold": best[1], "severe_threshold": best[2]}


def apply_calibration(probabilities, calibration):
    pain_prob = probabilities[:, 1] + probabilities[:, 2]
    severe_ratio = probabilities[:, 2] / pain_prob.clip(1e-6)
    return np.where(
        pain_prob < calibration["pain_threshold"], 0,
        np.where(severe_ratio >= calibration["severe_threshold"], 2, 1)
    ).astype(np.int64)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def expected_calibration_error(labels, probabilities, bins: int = 10) -> float:
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


def calculate_metrics(labels, predicted, probabilities, subjects):
    subject_scores = [
        f1_score(labels[subjects == s], predicted[subjects == s], average="macro", zero_division=0)
        for s in np.unique(subjects)
    ]
    severe = (labels == 2).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "subject_macro_f1_mean": float(np.mean(subject_scores)),
        "severe_recall": float(recall_score(severe, predicted == 2, zero_division=0)),
        "severe_auprc": float(average_precision_score(severe, probabilities[:, 2])),
        "ece": expected_calibration_error(labels, probabilities),
        "per_class_recall": json.dumps(
            recall_score(labels, predicted, labels=[0, 1, 2], average=None, zero_division=0).tolist()
        ),
        "confusion_matrix": json.dumps(confusion_matrix(labels, predicted, labels=[0, 1, 2]).tolist()),
    }


# -----------------------------------------------------------------------------
# Training loop helpers
# -----------------------------------------------------------------------------

def fit_binary_stage(model, loaders, criterion, args, device):
    """Train one binary stage: early-stop on val macro-F1."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state, best_score, best_epoch, stale = None, -np.inf, 0, 0
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
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_score, best_epoch, stale = float(val_score), epoch, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No stage checkpoint was created")
    return best_state, best_score, best_epoch