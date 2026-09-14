"""Model definitions: ModalityEncoder, GradientFusionTransformer, loss functions."""

from __future__ import annotations

import warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import CHANNELS

# Suppress PyTorch UserWarning about enable_nested_tensor with norm_first=True
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

class ModalityEncoder(nn.Module):
    """Encodes a single modality (BVP, EDA, etc.) into a feature vector."""

    def __init__(self, dimension: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, 9, stride=2, padding=4), nn.BatchNorm1d(32), nn.GELU()
        )
        self.branches = nn.ModuleList(
            [nn.Conv1d(32, 24, kernel, padding=kernel // 2) for kernel in (3, 7, 15)]
        )
        self.fuse = nn.Sequential(
            nn.BatchNorm1d(72), nn.GELU(), nn.Conv1d(72, dimension, 1), nn.GELU(),
            nn.Conv1d(dimension, dimension, 5, padding=4, dilation=2),
            nn.BatchNorm1d(dimension), nn.GELU(), nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        # signal: (B, 1, T) -> compute gradient and concatenate
        gradient = F.pad(signal[:, :, 1:] - signal[:, :, :-1], (1, 0))
        hidden = self.stem(torch.cat([signal, gradient], dim=1))  # (B, 32, T/2)
        hidden = torch.cat([branch(hidden) for branch in self.branches], dim=1)  # (B, 72, T/2)
        return self.fuse(hidden).squeeze(-1)  # (B, dimension)


class GradientFusionTransformer(nn.Module):
    """Fuses 5 modality encodings via Transformer + classifies."""

    def __init__(self, modalities: int = 5, classes: int = 3, dimension: int = 64):
        super().__init__()
        self.encoders = nn.ModuleList([ModalityEncoder(dimension) for _ in range(modalities)])
        self.cls = nn.Parameter(torch.zeros(1, 1, dimension))
        self.position = nn.Parameter(torch.zeros(1, modalities + 1, dimension))
        layer = nn.TransformerEncoderLayer(
            d_model=dimension, nhead=4, dim_feedforward=dimension * 2,
            dropout=0.15, activation="gelu", batch_first=True, norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(dimension)
        self.head = nn.Sequential(nn.Dropout(0.25), nn.Linear(dimension, classes))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, signal: torch.Tensor, return_features: bool = False):
        # signal: (B, 5, T) -> encode each modality
        tokens = torch.stack(
            [encoder(signal[:, i:i+1]) for i, encoder in enumerate(self.encoders)], dim=1
        )  # (B, 5, dimension)
        cls = self.cls.expand(signal.shape[0], -1, -1)
        features = self.norm(self.fusion(torch.cat([cls, tokens], dim=1) + self.position)[:, 0])
        logits = self.head(features)
        return (logits, features) if return_features else logits


# -----------------------------------------------------------------------------
# Loss functions
# -----------------------------------------------------------------------------

class BalancedFocalLoss(nn.Module):
    """Class-balanced focal loss."""

    def __init__(self, labels: np.ndarray, classes: int = 3, gamma: float = 1.5):
        super().__init__()
        counts = np.bincount(labels, minlength=classes)
        weights = len(labels) / (classes * np.maximum(counts, 1))
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets[:, None]).squeeze(1)
        pt = log_pt.exp()
        return (-self.weights[targets] * (1.0 - pt).pow(self.gamma) * log_pt).mean()


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.12) -> torch.Tensor:
    """Supervised contrastive loss (SupCon)."""
    features = F.normalize(features, dim=1)
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    identity = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positives = labels[:, None].eq(labels[None, :]) & ~identity
    denominator = torch.logsumexp(logits.masked_fill(identity, -torch.inf), dim=1)
    log_prob = logits - denominator[:, None]
    counts = positives.sum(dim=1).clamp_min(1)
    selected = torch.where(positives, log_prob, torch.zeros_like(log_prob))
    return -(selected.sum(dim=1) / counts).mean()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def autocast_context(device: torch.device):
    return torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda")


def augment_signal(signal: torch.Tensor) -> torch.Tensor:
    """Per-sample augmentation: scaling + noise + random zero-masking."""
    scale = torch.empty(signal.shape[0], 1).uniform_(0.92, 1.08)
    augmented = signal * scale + 0.025 * torch.randn_like(signal)
    if torch.rand(()) < 0.5:
        width = int(torch.randint(3, 13, ()).item())
        start = int(torch.randint(0, signal.shape[1] - width + 1, ()).item())
        augmented[:, start:start + width] = 0.0
    return augmented


def augment_batch(signal: torch.Tensor) -> torch.Tensor:
    """Batch augmentation for SupCon: scaling + noise + random masking."""
    scale = torch.empty(signal.shape[0], signal.shape[1], 1, device=signal.device).uniform_(0.92, 1.08)
    augmented = signal * scale + 0.025 * torch.randn_like(signal)
    mask = torch.rand(signal.shape[0], 1, signal.shape[2], device=signal.device) < 0.04
    return augmented.masked_fill(mask, 0.0)