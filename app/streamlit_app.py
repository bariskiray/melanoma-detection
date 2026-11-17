"""Streamlit demo: upload a dermoscopic image, get a melanoma probability and a Grad-CAM map.

streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms as T

from melanoma.data import build_transforms
from melanoma.gradcam import grad_cam, overlay
from melanoma.model import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CKPT = Path(os.environ.get("MELANOMA_CKPT", ROOT / "models" / "melanoma_effnet_b0.pt"))
METRICS = ROOT / "reports" / "metrics_main.json"

st.set_page_config(page_title="Melanoma Detection", page_icon="🔬", layout="wide")


@st.cache_resource
def get_model():
    model, meta = load_checkpoint(CKPT, "cpu")
    return model, meta


@st.cache_data
def get_metrics() -> dict | None:
    return json.loads(METRICS.read_text()) if METRICS.exists() else None


def tta_probability(model, x: torch.Tensor) -> float:
    with torch.no_grad():
        views = [x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2)]
        logit = torch.stack([model(v).squeeze() for v in views]).mean()
    return float(torch.sigmoid(logit))


def sidebar(meta: dict) -> None:
    st.sidebar.header("Model")
    st.sidebar.markdown(
        f"**Backbone:** `{meta['model_name']}`  \n"
        f"**Input:** {meta['img_size']}×{meta['img_size']}  \n"
        f"**Decision threshold:** {meta['threshold']:.3f} (chosen on validation for ≥85% sensitivity)"
    )
    metrics = get_metrics()
    if metrics:
        st.sidebar.header("Held-out performance")
        for key, label in [("test", "HAM10000 test (lesion-level)"), ("external_isic2018", "ISIC 2018 (external)")]:
            if key not in metrics["results"]:
                continue
            m = metrics["results"][key]["metrics"]
            st.sidebar.markdown(f"**{label}**")
            c1, c2 = st.sidebar.columns(2)
            c1.metric("AUROC", f"{m['auroc']:.3f}")
            c2.metric("Sensitivity", f"{m['sensitivity']:.1%}")
            c1.metric("Specificity", f"{m['specificity']:.1%}")
            c2.metric("PPV", f"{m['ppv']:.1%}")
    st.sidebar.caption("Trained on HAM10000 (CC BY-NC 4.0). Research/education use only.")


def main() -> None:
    st.title("🔬 Melanoma Detection from Dermoscopy")
    st.warning(
        "Research and educational demo — **not a medical device**. It must not be used for diagnosis. "
        "See a dermatologist about any skin lesion you are worried about."
    )
    if not CKPT.exists():
        st.error(
            f"Model checkpoint not found at `{CKPT}`. Download it from the GitHub release "
            "(see README) or train one with `make train`."
        )
        st.stop()

    model, meta = get_model()
    sidebar(meta)

    upload = st.file_uploader("Dermoscopic image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if upload is None:
        st.info("Upload a dermoscopic image to get a prediction.")
        return

    image = Image.open(upload).convert("RGB")
    size = meta["img_size"]
    x = build_transforms(size, train=False)(image).unsqueeze(0)
    view = T.Compose([T.Resize(int(round(size * 256 / 224))), T.CenterCrop(size)])(image)

    prob = tta_probability(model, x)
    cam, _ = grad_cam(model, x)
    flagged = prob >= meta["threshold"]

    col1, col2, col3 = st.columns([1, 1, 1.1])
    col1.image(view, caption="Model input (center crop)", use_container_width=True)
    col2.image(overlay(view, cam), caption="Grad-CAM: regions pushing towards melanoma", use_container_width=True)
    with col3:
        st.metric("Melanoma probability", f"{prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))
        if flagged:
            st.error(f"**Suspicious for melanoma** (probability ≥ threshold {meta['threshold']:.3f}).")
        else:
            st.success(f"**Below the melanoma threshold** ({meta['threshold']:.3f}).")
        st.caption(
            "The threshold favours sensitivity (screening setting), so many flagged lesions will be benign. "
            "Probabilities are calibrated on HAM10000-like images and may not transfer to phone photos "
            "or other devices."
        )


if __name__ == "__main__":
    main()
