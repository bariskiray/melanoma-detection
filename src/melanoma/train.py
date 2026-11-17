"""Train a melanoma classifier.

python -m melanoma.train --config configs/default.yaml
python -m melanoma.train --config configs/default.yaml --set data.split_mode=image experiment=leaky
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from melanoma.config import get_device, load_config, seed_everything
from melanoma.data import LesionDataset, build_transforms, lesion_overlap, load_ham_metadata, make_splits
from melanoma.engine import predict, train_one_epoch, warmup_cosine
from melanoma.evaluate import evaluate_run
from melanoma.metrics import binary_metrics
from melanoma.model import count_parameters, create_model, save_checkpoint


def resolve_splits(cfg: dict) -> pd.DataFrame:
    """Use the committed lesion split when available so results are exactly reproducible."""
    d = cfg["data"]
    meta = load_ham_metadata(d["metadata"])
    split_file = Path(d["split_file"])
    if d["split_mode"] == "lesion" and split_file.exists():
        saved = pd.read_csv(split_file)[["image_id", "split"]]
        df = meta.merge(saved, on="image_id", how="left", validate="one_to_one")
        if df["split"].isna().any():
            raise RuntimeError(f"{split_file} does not cover every image in the metadata")
        return df
    df = make_splits(meta, mode=d["split_mode"], seed=cfg["train"]["seed"])
    if d["split_mode"] == "lesion":
        split_file.parent.mkdir(parents=True, exist_ok=True)
        df[["image_id", "lesion_id", "dx", "label", "split"]].to_csv(split_file, index=False)
    return df


def make_loader(df: pd.DataFrame, cfg: dict, train: bool) -> DataLoader:
    ds = LesionDataset(
        df,
        cfg["data"]["image_dir"],
        build_transforms(cfg["data"]["img_size"], train=train, degrade=cfg["data"].get("degradation_aug", False)),
    )
    workers = cfg["train"]["num_workers"]
    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=train,
        drop_last=train,
        num_workers=workers,
        persistent_workers=workers > 0,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=5")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    tcfg, mcfg = cfg["train"], cfg["model"]
    seed_everything(tcfg["seed"])
    device = get_device()
    run_dir = Path(args.runs_dir) / cfg["experiment"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    df = resolve_splits(cfg)
    df.to_csv(run_dir / "splits.csv", index=False)
    summary = df.groupby("split")["label"].agg(images="size", melanoma="sum")
    summary["lesions"] = df.groupby("split")["lesion_id"].nunique()
    print(summary.to_string())
    print(f"test images whose lesion is also in train: {lesion_overlap(df):.1%}")

    train_loader = make_loader(df[df.split == "train"], cfg, train=True)
    val_loader = make_loader(df[df.split == "val"], cfg, train=False)

    model = create_model(mcfg["name"], mcfg["pretrained"], mcfg["drop_rate"], mcfg["freeze_backbone"]).to(device)
    total, trainable = count_parameters(model)
    print(f"{mcfg['name']}: {total / 1e6:.2f}M params, {trainable / 1e6:.2f}M trainable | device={device}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    steps = len(train_loader)
    scheduler = warmup_cosine(optimizer, tcfg["warmup_epochs"] * steps, tcfg["epochs"] * steps)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(tcfg["pos_weight"]), device=device))

    meta = {
        "model_name": mcfg["name"],
        "drop_rate": mcfg["drop_rate"],
        "img_size": cfg["data"]["img_size"],
        "experiment": cfg["experiment"],
    }
    history, best_auc, best_epoch, bad = [], -1.0, 0, 0
    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, desc=f"epoch {epoch}"
        )
        y_val, p_val = predict(model, val_loader, device)
        val_loss = float(nn.functional.binary_cross_entropy(torch.tensor(p_val), torch.tensor(y_val)))
        val_auc = binary_metrics(y_val, p_val, 0.5)["auroc"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auroc": val_auc,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.time() - t0,
            }
        )
        print(
            f"epoch {epoch:2d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
            f"| val_auroc {val_auc:.4f} | {time.time() - t0:.0f}s",
            flush=True,
        )
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if val_auc > best_auc:
            best_auc, best_epoch, bad = val_auc, epoch, 0
            save_checkpoint(run_dir / "best.pt", model, {**meta, "epoch": epoch, "val_auroc": val_auc})
        else:
            bad += 1
            if bad >= tcfg["early_stopping_patience"]:
                print(f"early stopping (no val AUROC gain for {bad} epochs)")
                break

    print(f"best epoch {best_epoch} (val AUROC {best_auc:.4f})")
    results = evaluate_run(run_dir, cfg, device)
    print(json.dumps({k: v["metrics"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
