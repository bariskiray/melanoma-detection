import numpy as np
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture
def fake_metadata() -> pd.DataFrame:
    """~600 images over 400 lesions, ~11% melanoma, some lesions photographed several times."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        dx = "mel" if rng.random() < 0.11 else rng.choice(["nv", "bkl", "bcc"])
        for j in range(int(rng.integers(1, 4))):
            rows.append({"lesion_id": f"HAM_{i:05d}", "image_id": f"ISIC_{i:05d}_{j}", "dx": dx})
    df = pd.DataFrame(rows)
    df["label"] = (df["dx"] == "mel").astype(int)
    return df


@pytest.fixture
def image_dir(tmp_path, fake_metadata):
    rng = np.random.default_rng(1)
    for image_id in fake_metadata["image_id"].head(8):
        arr = rng.integers(0, 255, size=(300, 400, 3), dtype=np.uint8)
        Image.fromarray(arr).save(tmp_path / f"{image_id}.jpg")
    return tmp_path
