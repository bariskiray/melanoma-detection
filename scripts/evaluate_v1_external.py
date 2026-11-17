"""Evaluate the original v1 Keras model (git tag v1.0) on the external ISIC 2018 test set.

v1 was trained on an image-level split, so it cannot be scored fairly on the new lesion-level
HAM10000 test set (many of those lesions were in its training data). ISIC 2018 is disjoint from
both, which makes it the only fair head-to-head comparison.

Requires TensorFlow (not a project dependency):
    git show v1.0:codes/models/skin_cancer_model.h5 > /tmp/v1.h5
    python scripts/evaluate_v1_external.py --model /tmp/v1.h5 --out reports/v1_external.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", default="data/isic2018_test/ISIC2018_Task3_Test_GroundTruth.csv")
    ap.add_argument("--images", default="data/isic2018_test/images")
    ap.add_argument("--out", default="reports/v1_external.json")
    args = ap.parse_args()

    import pandas as pd
    import tensorflow as tf
    from PIL import Image

    # melanoma.metrics only needs numpy/sklearn, so this runs in a TensorFlow env without PyTorch.
    from melanoma.metrics import binary_metrics, bootstrap_ci

    df = pd.read_csv(args.labels)
    df["label"] = (df["dx"] == "mel").astype(int)
    df = df[df["image_id"].map(lambda i: (Path(args.images) / f"{i}.jpg").exists())].reset_index(drop=True)
    model = tf.keras.models.load_model(args.model, compile=False)

    probs = []
    for start in range(0, len(df), 64):
        batch = [
            # v1 preprocessing: squash-resize to 224x224, scale to [0, 1]
            np.asarray(Image.open(Path(args.images) / f"{i}.jpg").convert("RGB").resize((224, 224)), np.float32) / 255.0
            for i in df["image_id"].iloc[start : start + 64]
        ]
        probs.append(model.predict(np.stack(batch), verbose=0).ravel())
    p, y = np.concatenate(probs), df["label"].to_numpy()

    out = {}
    for thr in (0.3, 0.5):  # 0.3 = v1 app threshold, 0.5 = Keras default used for v1's headline numbers
        out[f"threshold_{thr}"] = {
            "metrics": binary_metrics(y, p, thr),
            "ci95": bootstrap_ci(y, p, thr, groups=df["lesion_id"].to_numpy()),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            {
                k: {m: round(v["metrics"][m], 4) for m in ("auroc", "sensitivity", "specificity", "ppv")}
                for k, v in out.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
