"""Model construction and checkpoint I/O."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import timm
import torch
from torch import nn

# HAM10000 melanoma prevalence; used to initialise the output bias.
PRIOR_PROB = 0.11


def init_head(head: nn.Linear, prior_prob: float = PRIOR_PROB) -> None:
    """Small weights + prior-matched bias.

    timm's default init for a 1-unit classifier draws weights from U(-1, 1) (fan_out = 1), which
    yields huge initial logits and an initial BCE of ~2. Starting at the class prior instead
    gives a sensible initial loss (~0.35) and fast, stable convergence.
    """
    nn.init.normal_(head.weight, std=0.01)
    nn.init.constant_(head.bias, math.log(prior_prob / (1 - prior_prob)))


def create_model(
    name: str = "efficientnet_b0",
    pretrained: bool = True,
    drop_rate: float = 0.3,
    freeze_backbone: bool = False,
) -> nn.Module:
    """timm backbone with a single-logit head (melanoma vs. rest)."""
    model = timm.create_model(name, pretrained=pretrained, num_classes=1, drop_rate=drop_rate)
    init_head(model.get_classifier())
    model.frozen_backbone = freeze_backbone
    if freeze_backbone:
        head = model.get_classifier()
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    return model


def set_train_mode(model: nn.Module) -> None:
    """``model.train()``, except that a frozen backbone keeps BatchNorm in inference mode
    (running statistics are not updated), matching standard linear-probe practice."""
    model.train()
    if getattr(model, "frozen_backbone", False):
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(path: str | Path, model: nn.Module, meta: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild the model described in the checkpoint's ``meta`` and load its weights."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    meta = ckpt["meta"]
    model = create_model(meta["model_name"], pretrained=False, drop_rate=meta.get("drop_rate", 0.0))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, meta
