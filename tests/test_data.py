import pandas as pd
import pytest
import torch

from melanoma.data import (
    LesionDataset,
    build_transforms,
    lesion_overlap,
    load_ham_metadata,
    load_isic2018_labels,
    make_splits,
)


def test_lesion_split_has_no_leakage(fake_metadata):
    df = make_splits(fake_metadata, mode="lesion")
    assert df["split"].notna().all()
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert lesion_overlap(df, a, b) == 0.0
    assert df.groupby("lesion_id")["split"].nunique().max() == 1


def test_image_split_leaks_lesions(fake_metadata):
    df = make_splits(fake_metadata, mode="image")
    assert lesion_overlap(df, "train", "test") > 0.0


def test_split_proportions_and_stratification(fake_metadata):
    df = make_splits(fake_metadata, mode="lesion")
    lesions = df.groupby("split")["lesion_id"].nunique() / df["lesion_id"].nunique()
    assert lesions["train"] == pytest.approx(0.70, abs=0.02)
    rates = df.groupby("split")["label"].mean()
    assert rates.max() - rates.min() < 0.05


def test_split_is_deterministic(fake_metadata):
    a = make_splits(fake_metadata, seed=7)["split"]
    b = make_splits(fake_metadata, seed=7)["split"]
    assert a.equals(b)


def test_unknown_split_mode(fake_metadata):
    with pytest.raises(ValueError):
        make_splits(fake_metadata, mode="patient")


def test_load_ham_metadata(tmp_path, fake_metadata):
    path = tmp_path / "meta.csv"
    fake_metadata.drop(columns="label").to_csv(path, index=False)
    df = load_ham_metadata(path)
    assert df["label"].sum() == (fake_metadata["dx"] == "mel").sum()


def test_load_isic2018_labels(tmp_path):
    gt = pd.DataFrame({"image": ["a", "b"], "MEL": [1.0, 0.0], "NV": [0.0, 1.0], "BCC": [0.0, 0.0]})
    gt.to_csv(tmp_path / "gt.csv", index=False)
    out = load_isic2018_labels(tmp_path / "gt.csv")
    assert out["label"].tolist() == [1, 0]
    assert out["dx"].tolist() == ["mel", "nv"]


@pytest.mark.parametrize("train", [True, False])
def test_dataset_and_transforms(fake_metadata, image_dir, train):
    ds = LesionDataset(fake_metadata.head(8), image_dir, build_transforms(224, train=train))
    x, y = ds[0]
    assert x.shape == (3, 224, 224)
    assert x.dtype == torch.float32
    assert y.item() in (0.0, 1.0)


def test_load_isic2018_labels_metadata_layout(tmp_path):
    pd.DataFrame({"lesion_id": ["L1", "L2"], "image_id": ["a", "b"], "dx": ["nv", "mel"], "age": [40, 60]}).to_csv(
        tmp_path / "gt.csv", index=False
    )
    out = load_isic2018_labels(tmp_path / "gt.csv")
    assert out["label"].tolist() == [0, 1]
    assert "lesion_id" in out.columns


def test_degradation_transforms():
    from PIL import Image

    from melanoma.data import RandomDownscale, RandomJPEG

    img = Image.new("RGB", (400, 300), (120, 60, 30))
    small = RandomDownscale(96, 128)(img)
    assert 96 <= min(small.size) <= 128
    assert RandomJPEG(20, 30)(img).size == img.size
    x = build_transforms(224, train=True, degrade=True)(img)
    assert x.shape == (3, 224, 224)
