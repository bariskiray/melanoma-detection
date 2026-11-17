"""Training and inference loops."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from melanoma.model import set_train_mode


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    desc: str = "train",
) -> float:
    set_train_mode(model)
    total, n = 0.0, 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x).squeeze(1), y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(1, n)


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device, tta: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, melanoma probabilities). TTA averages logits over the 4 flip variants."""
    model.eval()
    probs, labels = [], []
    for x, y in tqdm(loader, desc="predict", leave=False):
        x = x.to(device, non_blocking=True)
        if tta:
            views = [x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2)]
            logits = torch.stack([model(v).squeeze(1) for v in views]).mean(0)
        else:
            logits = model(x).squeeze(1)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)
