"""Stress-test a checkpoint on the external set under resolution loss and JPEG compression.

Each image is degraded (downscaled so its shorter side is ``s`` px and/or re-encoded as JPEG)
and then goes through the normal evaluation transform, which upsamples it back to the model
input size. The decision threshold stays the one chosen on validation.

    python scripts/robustness.py --run runs/effnet_b0_lesion
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from melanoma.config import get_device  # noqa: E402
from melanoma.data import build_transforms, load_isic2018_labels  # noqa: E402
from melanoma.engine import predict  # noqa: E402
from melanoma.metrics import binary_metrics  # noqa: E402
from melanoma.model import load_checkpoint  # noqa: E402


class Degraded(Dataset):
    def __init__(self, ids, labels, image_dir, short_side=None, jpeg_quality=None, img_size=224):
        self.ids, self.labels, self.dir = list(ids), np.asarray(labels, np.float32), Path(image_dir)
        self.short_side, self.q = short_side, jpeg_quality
        self.tf = build_transforms(img_size, train=False)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(self.dir / f"{self.ids[i]}.jpg").convert("RGB")
        if self.short_side:
            w, h = img.size
            s = self.short_side / min(w, h)
            img = img.resize((round(w * s), round(h * s)), Image.Resampling.BILINEAR)
        if self.q:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=self.q)
            img = Image.open(buf).convert("RGB")
        return self.tf(img), torch.tensor(self.labels[i])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs/effnet_b0_lesion"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args()

    cfg = yaml.safe_load((args.run / "config.yaml").read_text())
    device = get_device()
    model, meta = load_checkpoint(args.run / "best.pt", device)
    ext = load_isic2018_labels(cfg["data"]["external_labels"])
    image_dir = Path(cfg["data"]["external_image_dir"])
    ext = ext[ext["image_id"].map(lambda i: (image_dir / f"{i}.jpg").exists())]

    conditions = [("original (256 px)", None, None)]
    conditions += [(f"{s} px", s, None) for s in (192, 160, 128, 96)]
    conditions += [("JPEG q=50", None, 50), ("JPEG q=20", None, 20)]
    results = {}
    for name, side, q in conditions:
        ds = Degraded(ext["image_id"], ext["label"], image_dir, side, q, meta["img_size"])
        y, p = predict(model, DataLoader(ds, batch_size=64, num_workers=4), device, tta=True)
        m = binary_metrics(y, p, meta["threshold"])
        results[name] = {k: m[k] for k in ("auroc", "sensitivity", "specificity")}
        print(f"{name:18s} AUROC {m['auroc']:.3f}  sens {m['sensitivity']:.1%}  spec {m['specificity']:.1%}")

    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    (args.out / "robustness.json").write_text(json.dumps(results, indent=2))

    res = [k for k in results if "JPEG" not in k]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for key, color in (("auroc", "#2563eb"), ("sensitivity", "#ea580c"), ("specificity", "#16a34a")):
        ax.plot(res, [results[r][key] for r in res], marker="o", color=color, label=key.capitalize())
    ax.set(
        ylim=(0, 1.02),
        xlabel="Shorter image side before upsampling to 224 px",
        title="Resolution robustness (ISIC 2018)",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "figures" / "robustness.png", dpi=160)


if __name__ == "__main__":
    main()
