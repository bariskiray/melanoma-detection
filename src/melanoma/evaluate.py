"""Evaluate a trained run on the internal (lesion-level) test set and the external ISIC 2018 test set.

The decision threshold is chosen on the validation set only and then frozen.

    python -m melanoma.evaluate --run runs/effnet_b0_lesion
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from melanoma import plots
from melanoma.config import get_device, load_config
from melanoma.data import LesionDataset, build_transforms, load_isic2018_labels
from melanoma.engine import predict
from melanoma.metrics import binary_metrics, bootstrap_ci, choose_threshold
from melanoma.model import load_checkpoint


def _loader(df: pd.DataFrame, image_dir: str, cfg: dict) -> DataLoader:
    ds = LesionDataset(df, image_dir, build_transforms(cfg["data"]["img_size"], train=False))
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=cfg["train"]["num_workers"])


def evaluate_run(run_dir: str | Path, cfg: dict, device: torch.device | None = None) -> dict:
    run_dir = Path(run_dir)
    device = device or get_device()
    ecfg, dcfg = cfg["eval"], cfg["data"]
    model, meta = load_checkpoint(run_dir / "best.pt", device)
    df = pd.read_csv(run_dir / "splits.csv")

    preds = []
    sets: dict[str, tuple[pd.DataFrame, str]] = {
        "val": (df[df.split == "val"], dcfg["image_dir"]),
        "test": (df[df.split == "test"], dcfg["image_dir"]),
    }
    ext_labels = Path(dcfg.get("external_labels") or "")
    if ext_labels.is_file():
        ext = load_isic2018_labels(ext_labels)
        exists = ext["image_id"].map(lambda i: (Path(dcfg["external_image_dir"]) / f"{i}.jpg").exists())
        if (~exists).any():
            print(f"external set: skipping {(~exists).sum()} labelled image(s) missing on disk")
        sets["external_isic2018"] = (ext[exists].reset_index(drop=True), dcfg["external_image_dir"])

    probs: dict[str, tuple] = {}
    for name, (part, image_dir) in sets.items():
        y, p = predict(model, _loader(part, image_dir, cfg), device, tta=ecfg["tta"])
        probs[name] = (y, p)
        preds.append(pd.DataFrame({"set": name, "image_id": part["image_id"].values, "label": y, "prob": p}))

    threshold = choose_threshold(*probs["val"], ecfg["threshold_strategy"], ecfg["target_sensitivity"])
    meta = {**meta, "threshold": threshold, "threshold_strategy": ecfg["threshold_strategy"]}
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    torch.save({"state_dict": ckpt["state_dict"], "meta": meta}, run_dir / "best.pt")

    results = {}
    for name, (y, p) in probs.items():
        if name == "val":
            continue
        groups = sets[name][0]["lesion_id"].values if "lesion_id" in sets[name][0] else None
        results[name] = {
            "metrics": binary_metrics(y, p, threshold),
            "ci95": bootstrap_ci(y, p, threshold, n_boot=ecfg["bootstrap"], groups=groups),
        }
        plots.confusion(
            results[name]["metrics"], f"{name} (threshold {threshold:.3f})", run_dir / f"confusion_{name}.png"
        )
    results["val"] = {"metrics": binary_metrics(*probs["val"], threshold)}

    curves = {
        {"test": "HAM10000 test", "external_isic2018": "ISIC 2018 (external)"}.get(k, k): v
        for k, v in probs.items()
        if k != "val"
    }
    plots.roc_pr_curves(curves, run_dir / "roc_pr.png")
    plots.calibration(curves, run_dir / "calibration.png")
    if (run_dir / "history.csv").exists():
        plots.training_history(pd.read_csv(run_dir / "history.csv"), run_dir / "training_history.png")

    pd.concat(preds).to_csv(run_dir / "predictions.csv", index=False)
    out = {"experiment": cfg["experiment"], "model": meta, "results": results}
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory containing best.pt, splits.csv, config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.run) / "config.yaml")
    results = evaluate_run(args.run, cfg)
    print(json.dumps({k: v["metrics"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
