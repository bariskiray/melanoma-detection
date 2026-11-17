# Model card — melanoma classifier (v2)

## Overview

| | |
|---|---|
| Task | Binary classification of a single dermoscopic image: melanoma vs. any other HAM10000 diagnosis |
| Architecture | EfficientNet-B0 (ImageNet-pretrained, `timm`), single-logit head, full fine-tuning, trained with random downscaling / JPEG augmentation |
| Input | RGB dermoscopic image, resized so the shorter side is 256 px, center-cropped to 224×224 |
| Output | Melanoma probability; a decision threshold chosen on the validation set (target sensitivity ≥ 85%) |
| Weights | `melanoma_effnet_b0.pt` (GitHub release asset, ~16 MB) |
| Code license | MIT |
| Weights license | CC BY-NC 4.0 (inherited from HAM10000): non-commercial use only |

## Intended use

- Education and research on medical image classification, evaluation methodology and model auditing.
- **Not** intended for diagnosis, triage, screening of real patients or any clinical decision. It is not a
  medical device and has not been clinically validated.

## Training data

- [HAM10000](https://doi.org/10.7910/DVN/DBW86T): 10,015 dermoscopic images of 7,470 lesions, 7 diagnoses;
  1,113 melanoma images (11.1%). Labels are histopathology for melanoma and a mix of histopathology,
  follow-up, expert consensus and confocal microscopy for other classes.
- Lesion-level stratified split (70/15/15 of lesions): no lesion appears in more than one partition.
  The exact split is versioned in `splits/ham10000_lesion_split.csv`.

## Evaluation data

- **Internal test:** 1,510 HAM10000 images (1,121 lesions) never seen during training or model selection.
- **External test:** ISIC 2018 Task 3 official test set (1,511 images with available files, 171 melanoma),
  acquired at partly different sites and never used for training, model selection or threshold choice.
- 95% confidence intervals come from a lesion-cluster bootstrap (1,000 resamples).

See the README for the numbers.

## Known limitations and risks

- **Population bias.** HAM10000 comes mostly from fair-skinned patients in Austria and Australia.
  Performance on darker skin types is unknown and likely worse.
- **Acquisition shift.** Trained only on dermoscopy. Smartphone photos, clinical (non-dermoscopic) images and
  other dermatoscopes are out of distribution. Degradation augmentation keeps external AUROC ≥ 0.885 down to
  96 px inputs or JPEG quality 20 (see the README robustness section), but other shifts were not tested.
- **Low PPV at screening thresholds.** At ~82% external sensitivity, roughly two out of three flagged lesions are benign.
- **Spurious cues.** Grad-CAM shows that some false positives are driven by image corners or ruler marks rather
  than by the lesion, which points to possible acquisition artefacts in the data.
- **Calibration.** Probabilities are slightly conservative (observed melanoma rate a little above the predicted
  probability) and reflect the HAM10000 class mix (11% melanoma). A different prevalence shifts what a score means.
- **Binary task.** Other malignancies (BCC, AKIEC) are part of the negative class, so a "not melanoma" output
  does not mean "benign".

## Ethical considerations

A tool like this could falsely reassure a patient with melanoma, or cause anxiety and extra referrals
through false positives. Any real-world use would need prospective clinical validation, regulatory
approval and human oversight.
