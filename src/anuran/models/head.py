"""Trainable head on frozen encoder embeddings (the encoder never updates).

embedding -> dropout -> Linear(d_in, d_proj) -> GELU -> dropout -> z
z -> Linear(d_proj, n_classes)   multi-label logits (sigmoid; background = all zeros)
z -> Linear(d_proj, n_attr)      optional attribute head (text supervision, step 4)

Loss: masked BCE (mask=0 for secondary species of mixed choruses) with per-class
pos_weight, plus optional extra terms passed in as callables.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn


class Head(nn.Module):
    def __init__(self, d_in: int, n_classes: int, d_proj: int = 256, dropout: float = 0.2, n_attr: int = 0):
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, d_proj), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Linear(d_proj, n_classes)
        self.attr = nn.Linear(d_proj, n_attr) if n_attr else None

    def forward(self, x: torch.Tensor):
        z = self.proj(x)
        return self.cls(z), z, (self.attr(z) if self.attr is not None else None)


def masked_bce(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def pos_weights(Y: np.ndarray, M: np.ndarray, cap: float = 30.0) -> np.ndarray:
    """sqrt(neg/pos) per class, so rare classes are not drowned out by common ones."""
    pos = (Y * M).sum(0)
    neg = ((1 - Y) * M).sum(0)
    return np.clip(np.sqrt(neg / np.maximum(pos, 1)), 1.0, cap).astype(np.float32)


def fit(
    X: np.ndarray, Y: np.ndarray, M: np.ndarray, *, seed: int, epochs: int = 30, lr: float = 1e-3,
    weight_decay: float = 1e-4, batch_size: int = 256, d_proj: int = 256, dropout: float = 0.2,
    extra_losses: list[Callable] | None = None, n_attr: int = 0,
) -> Head:
    """extra_losses: callables (model_outputs, batch_index) -> scalar loss, added to BCE."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Head(X.shape[1], Y.shape[1], d_proj, dropout, n_attr)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt, Yt, Mt = (torch.from_numpy(a.astype(np.float32)) for a in (X, Y, M))
    pw = torch.from_numpy(pos_weights(Y, M))
    model.train()
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(X)), max(1, len(X) // batch_size)):
            idx_t = torch.from_numpy(idx)
            out = model(Xt[idx_t])
            loss = masked_bce(out[0], Yt[idx_t], Mt[idx_t], pw)
            for extra in extra_losses or []:
                loss = loss + extra(out, idx_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model: Head, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    out = [torch.sigmoid(model(torch.from_numpy(X[i: i + batch_size].astype(np.float32)))[0]).numpy()
           for i in range(0, len(X), batch_size)]
    return np.concatenate(out)
