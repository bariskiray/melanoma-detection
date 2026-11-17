"""Grad-CAM for timm CNNs (uses ``forward_features`` / ``forward_head``)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image
from torch import nn


def grad_cam(model: nn.Module, x: torch.Tensor) -> tuple[np.ndarray, float]:
    """Return (heatmap in [0, 1] at input resolution, melanoma probability) for one image.

    ``x`` has shape (1, 3, H, W). The map highlights regions that increase the melanoma logit.
    """
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError("grad_cam expects a single image batch of shape (1, 3, H, W)")
    model.eval()
    with torch.enable_grad():
        feats = model.forward_features(x)
        feats.retain_grad()
        logit = model.forward_head(feats).squeeze()
        model.zero_grad(set_to_none=True)
        logit.backward()
        weights = feats.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feats).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam, float(torch.sigmoid(logit).item())


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a [0, 1] heatmap onto an RGB image of the same size."""
    heat = (colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat).resize(image.size, Image.Resampling.BILINEAR)
    return Image.blend(image.convert("RGB"), heat_img, alpha)
