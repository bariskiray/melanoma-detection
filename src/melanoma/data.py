"""Metadata handling, leakage-free splitting, datasets and transforms."""

from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MELANOMA_DX = "mel"


# --------------------------------------------------------------------------- metadata
def load_ham_metadata(path: str | Path) -> pd.DataFrame:
    """HAM10000 metadata with a binary ``label`` column (1 = melanoma)."""
    df = pd.read_csv(path)
    required = {"lesion_id", "image_id", "dx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    df["label"] = (df["dx"] == MELANOMA_DX).astype(int)
    return df


def load_isic2018_labels(path: str | Path) -> pd.DataFrame:
    """ISIC 2018 Task 3 test ground truth -> image_id, dx, label (+ lesion_id when available).

    Accepts both the HAM10000-style metadata layout distributed on Harvard Dataverse
    (``lesion_id, image_id, dx, ...``) and the one-hot layout from the ISIC challenge site
    (``image, MEL, NV, ...``).
    """
    gt = pd.read_csv(path)
    if "dx" in gt.columns:
        out = gt[[c for c in ("lesion_id", "image_id", "dx") if c in gt.columns]].copy()
    else:
        classes = [c for c in gt.columns if c != "image"]
        out = pd.DataFrame({"image_id": gt["image"], "dx": gt[classes].idxmax(axis=1).str.lower()})
    out["label"] = (out["dx"] == MELANOMA_DX).astype(int)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- splitting
def make_splits(
    df: pd.DataFrame,
    mode: str = "lesion",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign every image to train/val/test, stratified by label.

    ``mode="lesion"`` splits unique lesions so that all images of a lesion end up in the
    same partition (HAM10000 has ~1.3 images per lesion). ``mode="image"`` splits images
    independently, which leaks near-duplicate photos of the same lesion across partitions;
    it exists only to quantify that leakage.
    """
    if mode not in {"lesion", "image"}:
        raise ValueError(f"unknown split mode {mode!r}")
    df = df.copy()
    if mode == "lesion":
        unit = "lesion_id"
        units = df.groupby(unit)["label"].max().reset_index()
    else:
        # Keep metadata order so this reproduces the original (v1) image-level split exactly.
        unit = "image_id"
        units = df[[unit, "label"]]

    holdout = val_frac + test_frac
    train_u, rest_u = train_test_split(units, test_size=holdout, stratify=units["label"], random_state=seed)
    val_u, test_u = train_test_split(rest_u, test_size=test_frac / holdout, stratify=rest_u["label"], random_state=seed)
    assignment = {
        **dict.fromkeys(train_u[unit], "train"),
        **dict.fromkeys(val_u[unit], "val"),
        **dict.fromkeys(test_u[unit], "test"),
    }
    df["split"] = df[unit].map(assignment)
    return df


def lesion_overlap(df: pd.DataFrame, a: str = "train", b: str = "test") -> float:
    """Fraction of images in split ``b`` whose lesion also appears in split ``a``."""
    seen = set(df.loc[df["split"] == a, "lesion_id"])
    target = df[df["split"] == b]
    return float(target["lesion_id"].isin(seen).mean()) if len(target) else 0.0


# --------------------------------------------------------------------------- transforms
class RandomDownscale:
    """Downscale so the shorter side is in ``[min_side, max_side]`` (later transforms upsample again).

    Simulates lower-resolution captures; without it the model relies on fine texture that
    disappears as soon as an image is smaller than the training resolution.
    """

    def __init__(self, min_side: int = 96, max_side: int = 256):
        self.min_side, self.max_side = min_side, max_side

    def __call__(self, img: Image.Image) -> Image.Image:
        side = random.randint(self.min_side, self.max_side)
        w, h = img.size
        if min(w, h) <= side:
            return img
        s = side / min(w, h)
        return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.BILINEAR)


class RandomJPEG:
    """Re-encode as JPEG with a random quality in ``[min_q, max_q]``."""

    def __init__(self, min_q: int = 20, max_q: int = 90):
        self.min_q, self.max_q = min_q, max_q

    def __call__(self, img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(self.min_q, self.max_q))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def build_transforms(img_size: int = 224, train: bool = False, degrade: bool = False) -> T.Compose:
    """Evaluation: resize + center crop. Training: geometric + mild colour augmentation, and with
    ``degrade=True`` random resolution loss / JPEG compression (each with p=0.5)."""
    resize = int(round(img_size * 256 / 224))
    if train:
        degradation = (
            [T.RandomApply([RandomDownscale()], p=0.5), T.RandomApply([RandomJPEG()], p=0.5)] if degrade else []
        )
        return T.Compose(
            [
                *degradation,
                T.Resize(resize),
                # Dermoscopy has no canonical orientation, so full rotations and flips are safe.
                T.RandomRotation(180),
                T.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                # Mild photometric jitter for robustness to device/lighting differences while
                # keeping the lesion's colour pattern (a key diagnostic cue) essentially intact.
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize(resize),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# --------------------------------------------------------------------------- dataset
class LesionDataset(Dataset):
    """Returns ``(image_tensor, label_float)`` for rows of a metadata frame."""

    def __init__(self, df: pd.DataFrame, image_dir: str | Path, transform=None):
        self.image_ids = df["image_id"].tolist()
        self.labels = df["label"].astype(np.float32).to_numpy()
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_dir / f"{self.image_ids[idx]}.jpg").convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx])
