# Results

Threshold: 0.0719

## Comparison

| Model | Split | Internal test AUROC | External AUROC | External sens. | External spec. | External PPV |
|---|---|---|---|---|---|---|
| **EfficientNet-B0 + degradation aug. (main)** | lesion | 0.881 (0.833–0.919) | 0.910 (0.879–0.937) | 82.5% | 81.6% | 36.3% |
| EfficientNet-B0 | lesion | 0.898 (0.863–0.930) | 0.903 (0.875–0.930) | 81.9% | 80.0% | 34.3% |
| MobileNetV2, full fine-tune | lesion | 0.877 (0.836–0.914) | 0.913 (0.886–0.936) | 83.6% | 81.1% | 36.1% |
| MobileNetV2, frozen backbone (≈ v1 setup) | lesion | 0.807 (0.761–0.852) | 0.855 (0.817–0.889) | 87.1% | 63.4% | 23.3% |
| EfficientNet-B0, image-level split (leaky) | image (leaky) | 0.930 (0.910–0.948) | 0.911 (0.880–0.937) | 77.8% | 84.6% | 39.2% |
| v1 Keras MobileNetV2 (original repo, threshold 0.3) | image (leaky) | 0.827 *(as reported)* | 0.792 (0.751–0.832) | 58.5% | 81.9% | 29.2% |

## Main model

| Metric | HAM10000 test (lesion-level) | ISIC 2018 (external) |
|---|---|---|
| AUROC | 0.881 (0.833–0.919) | 0.910 (0.879–0.937) |
| AUPRC | 0.599 (0.492–0.685) | 0.668 (0.572–0.745) |
| Sensitivity | 80.2% (71.1%–88.3%) | 82.5% (75.6%–88.8%) |
| Specificity | 80.9% (78.2%–83.6%) | 81.6% (79.3%–83.9%) |
| PPV (precision) | 34.3% (27.9%–40.3%) | 36.3% (30.2%–42.5%) |
| NPV | 97.1% (95.6%–98.3%) | 97.3% (96.1%–98.4%) |
| Balanced accuracy | 80.6% (76.0%–84.7%) | 82.0% (78.3%–85.4%) |
| TP / FP / TN / FN | 134 / 257 / 1086 / 33 | 141 / 247 / 1093 / 30 |
| Images (melanoma) | 1510 (167) | 1511 (171) |

## Robustness

| Condition | EfficientNet-B0 AUROC / sens. | EfficientNet-B0 + degradation aug. AUROC / sens. |
|---|---|---|
| original (256 px) | 0.903 / 82% | 0.910 / 82% |
| 192 px | 0.862 / 26% | 0.898 / 79% |
| 160 px | 0.830 / 13% | 0.895 / 81% |
| 128 px | 0.690 / 2% | 0.893 / 81% |
| 96 px | 0.581 / 0% | 0.892 / 78% |
| JPEG q=50 | 0.874 / 42% | 0.901 / 83% |
| JPEG q=20 | 0.630 / 5% | 0.885 / 80% |
