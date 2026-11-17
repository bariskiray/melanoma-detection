"""Collect finished runs into ``reports/``: results tables, figures, Grad-CAM gallery, app checkpoint.

python scripts/make_report.py --main effnet_b0_lesion_robust --baseline effnet_b0_lesion
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from melanoma.data import build_transforms, load_isic2018_labels  # noqa: E402
from melanoma.gradcam import grad_cam, overlay  # noqa: E402
from melanoma.model import load_checkpoint  # noqa: E402

LABELS = {
    "effnet_b0_lesion_robust": "EfficientNet-B0 + degradation aug.",
    "effnet_b0_lesion": "EfficientNet-B0",
    "mobilenetv2_finetune_lesion": "MobileNetV2, full fine-tune",
    "mobilenetv2_frozen_lesion": "MobileNetV2, frozen backbone (≈ v1 setup)",
    "effnet_b0_image_split": "EfficientNet-B0, image-level split (leaky)",
}


def fmt(m: dict, ci: dict | None, key: str) -> str:
    val = f"{m[key]:.3f}"
    if ci and key in ci:
        lo, hi = ci[key]
        val += f" ({lo:.3f}–{hi:.3f})"
    return val


def results_table(runs: Path, v1_path: Path, main: str) -> str:
    rows = [
        "| Model | Split | Internal test AUROC | External AUROC | External sens. | External spec. | External PPV |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, label in LABELS.items():
        f = runs / name / "metrics.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text())["results"]
        t, e = res["test"], res["external_isic2018"]
        split = "image (leaky)" if "image_split" in name else "lesion"
        label = f"**{label} (main)**" if name == main else label
        m = e["metrics"]
        rows.append(
            f"| {label} | {split} | {fmt(t['metrics'], t['ci95'], 'auroc')} | {fmt(m, e['ci95'], 'auroc')} | "
            f"{m['sensitivity']:.1%} | {m['specificity']:.1%} | {m['ppv']:.1%} |"
        )
    if v1_path.exists():
        v1 = json.loads(v1_path.read_text())["threshold_0.3"]
        m = v1["metrics"]
        rows.append(
            f"| v1 Keras MobileNetV2 (original repo, threshold 0.3) | image (leaky) | 0.827 *(as reported)* | "
            f"{fmt(m, v1['ci95'], 'auroc')} | {m['sensitivity']:.1%} | {m['specificity']:.1%} | {m['ppv']:.1%} |"
        )
    return "\n".join(rows)


def detail_table(metrics: dict) -> str:
    keys = [
        ("auroc", "AUROC", False),
        ("auprc", "AUPRC", False),
        ("sensitivity", "Sensitivity", True),
        ("specificity", "Specificity", True),
        ("ppv", "PPV (precision)", True),
        ("npv", "NPV", True),
        ("balanced_accuracy", "Balanced accuracy", True),
    ]
    sets = [("test", "HAM10000 test (lesion-level)"), ("external_isic2018", "ISIC 2018 (external)")]
    rows = ["| Metric | " + " | ".join(s[1] for s in sets) + " |", "|---|---|---|"]
    for key, name, pct in keys:
        cells = []
        for s, _ in sets:
            m, ci = metrics[s]["metrics"], metrics[s]["ci95"]
            lo, hi = ci[key]
            cells.append(f"{m[key]:.1%} ({lo:.1%}–{hi:.1%})" if pct else f"{m[key]:.3f} ({lo:.3f}–{hi:.3f})")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    for key, name in [("tp", "TP / FP / TN / FN"), ("n", "Images (melanoma)")]:
        cells = []
        for s, _ in sets:
            m = metrics[s]["metrics"]
            cells.append(
                f"{m['tp']} / {m['fp']} / {m['tn']} / {m['fn']}" if key == "tp" else f"{m['n']} ({m['tp'] + m['fn']})"
            )
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def robustness_figure(runs: Path, names: dict[str, str], out: Path) -> str:
    data = {
        label: json.loads((runs / n / "robustness" / "robustness.json").read_text())
        for n, label in names.items()
        if (runs / n / "robustness" / "robustness.json").exists()
    }
    if not data:
        return ""
    conds = list(next(iter(data.values())))
    res = [c for c in conds if "JPEG" not in c]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, key in zip(axes, ("auroc", "sensitivity"), strict=True):
        for (label, r), color in zip(data.items(), ("#9ca3af", "#2563eb"), strict=False):
            ax.plot(res, [r[c][key] for c in res], marker="o", color=color, lw=2, label=label)
        ax.set(title=f"External {key.upper() if key == 'auroc' else key}", ylim=(0, 1.02))
        ax.set_xlabel("Shorter image side before upsampling")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    rows = ["| Condition | " + " | ".join(f"{lab} AUROC / sens." for lab in data) + " |"]
    rows.append("|---|" + "---|" * len(data))
    for c in conds:
        rows.append(
            f"| {c} | " + " | ".join(f"{r[c]['auroc']:.3f} / {r[c]['sensitivity']:.0%}" for r in data.values()) + " |"
        )
    return "\n".join(rows)


def gradcam_gallery(run: Path, out: Path, n: int = 4) -> None:
    """Most confident true positives and false positives on the external set."""
    model, meta = load_checkpoint(run / "best.pt", "cpu")
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    preds = pd.read_csv(run / "predictions.csv")
    ext = preds[preds["set"] == "external_isic2018"]
    lesions = load_isic2018_labels(cfg["data"]["external_labels"])
    if "lesion_id" in lesions:  # one image per lesion so the gallery does not repeat a lesion
        ext = ext.merge(lesions[["image_id", "lesion_id"]], on="image_id", how="left")
        ext = ext.sort_values("prob", ascending=False).drop_duplicates("lesion_id")
    thr = meta["threshold"]
    tp = ext[(ext.label == 1) & (ext.prob >= thr)].nlargest(n, "prob")
    fp = ext[(ext.label == 0) & (ext.prob >= thr)].nlargest(n, "prob")
    size = meta["img_size"]
    tf = build_transforms(size, train=False)
    crop = T.Compose([T.Resize(int(round(size * 256 / 224))), T.CenterCrop(size)])
    image_dir = Path(cfg["data"]["external_image_dir"])

    fig, axes = plt.subplots(4, n, figsize=(2.4 * n, 10))
    for row, (df, title) in enumerate([(tp, "Melanoma, flagged"), (fp, "Benign, flagged (FP)")]):
        for col, (_, r) in enumerate(df.iterrows()):
            img = Image.open(image_dir / f"{r.image_id}.jpg").convert("RGB")
            cam, _ = grad_cam(model, tf(img).unsqueeze(0))
            view = crop(img)
            axes[2 * row, col].imshow(view)
            axes[2 * row, col].set_title(f"{title}\np = {r.prob:.2f}", fontsize=8)
            axes[2 * row + 1, col].imshow(overlay(view, cam))
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Grad-CAM on the external ISIC 2018 test set", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--main", default="effnet_b0_lesion_robust")
    ap.add_argument("--baseline", default="effnet_b0_lesion", help="non-robust run for the robustness figure")
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    args = ap.parse_args()

    figs = args.reports / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    main_run = args.runs / args.main
    for name in (
        "roc_pr.png",
        "calibration.png",
        "training_history.png",
        "confusion_test.png",
        "confusion_external_isic2018.png",
    ):
        shutil.copy(main_run / name, figs / name)
    shutil.copy(main_run / "metrics.json", args.reports / "metrics_main.json")
    metrics = json.loads((main_run / "metrics.json").read_text())

    sections = {
        "Comparison": results_table(args.runs, args.reports / "v1_external.json", args.main),
        "Main model": detail_table(metrics["results"]),
        "Robustness": robustness_figure(
            args.runs,
            {args.baseline: LABELS[args.baseline], args.main: LABELS[args.main]},
            figs / "robustness.png",
        ),
    }
    text = "\n\n".join(f"## {k}\n\n{v}" for k, v in sections.items() if v)
    (args.reports / "results.md").write_text(f"# Results\n\nThreshold: {metrics['model']['threshold']:.4f}\n\n{text}\n")
    print(text)

    gradcam_gallery(main_run, figs / "gradcam_examples.png")

    ckpt = torch.load(main_run / "best.pt", map_location="cpu", weights_only=True)
    Path("models").mkdir(exist_ok=True)
    torch.save(ckpt, "models/melanoma_effnet_b0.pt")
    print("wrote reports/ and models/melanoma_effnet_b0.pt")


if __name__ == "__main__":
    main()
