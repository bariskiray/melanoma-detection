import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from melanoma.engine import predict, warmup_cosine
from melanoma.gradcam import grad_cam, overlay
from melanoma.model import count_parameters, create_model, load_checkpoint, save_checkpoint

NAME = "efficientnet_b0"


def test_forward_shape():
    model = create_model(NAME, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 224, 224)).shape == (2, 1)


def test_freeze_backbone_only_trains_head():
    model = create_model(NAME, pretrained=False, freeze_backbone=True)
    total, trainable = count_parameters(model)
    head = sum(p.numel() for p in model.get_classifier().parameters())
    assert trainable == head < total


def test_checkpoint_roundtrip(tmp_path):
    model = create_model(NAME, pretrained=False, drop_rate=0.0).eval()
    save_checkpoint(tmp_path / "m.pt", model, {"model_name": NAME, "drop_rate": 0.0, "threshold": 0.3})
    loaded, meta = load_checkpoint(tmp_path / "m.pt")
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-5)
    assert meta["threshold"] == 0.3


def test_predict_with_tta_returns_probabilities():
    model = create_model(NAME, pretrained=False)
    loader = DataLoader(TensorDataset(torch.randn(5, 3, 224, 224), torch.tensor([0.0, 1, 0, 1, 0])), batch_size=2)
    y, p = predict(model, loader, torch.device("cpu"), tta=True)
    assert y.shape == p.shape == (5,)
    assert ((p >= 0) & (p <= 1)).all()


def test_warmup_cosine_schedule():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = warmup_cosine(opt, warmup_steps=10, total_steps=100)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[9] <= 1.0
    assert lrs[-1] < 0.01


def test_grad_cam_output():
    model = create_model(NAME, pretrained=False)
    cam, prob = grad_cam(model, torch.randn(1, 3, 224, 224))
    assert cam.shape == (224, 224)
    assert cam.min() >= 0 and cam.max() <= 1
    assert 0 <= prob <= 1
    img = Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8))
    assert overlay(img, cam).size == (300, 300)


def test_split_file_is_leak_free():
    """The committed split must keep every lesion in a single partition."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "splits" / "ham10000_lesion_split.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    assert df.groupby("lesion_id")["split"].nunique().max() == 1
    assert set(df["split"]) == {"train", "val", "test"}


def test_head_init_starts_at_prior():
    import math

    from melanoma.model import PRIOR_PROB

    model = create_model(NAME, pretrained=False).eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.randn(4, 3, 224, 224))).squeeze(1)
    assert torch.allclose(p, torch.full_like(p, PRIOR_PROB), atol=0.05)
    assert model.get_classifier().bias.item() == pytest.approx(math.log(PRIOR_PROB / (1 - PRIOR_PROB)))


def test_frozen_backbone_keeps_batchnorm_in_eval():
    from melanoma.model import set_train_mode

    model = create_model(NAME, pretrained=False, freeze_backbone=True)
    set_train_mode(model)
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns and not any(m.training for m in bns)
    assert model.get_classifier().training
