PY ?= python

.PHONY: install data train ablations robustness report evaluate app test lint

install:            ## install the package with app + dev extras
	$(PY) -m pip install -e ".[app,dev]"

data:               ## download HAM10000 + ISIC 2018 test set (~250 MB on disk after resizing)
	$(PY) scripts/download_data.py --out data --isic2018

ABL = $(PY) -m melanoma.train --config configs/default.yaml --set data.degradation_aug=false

train:              ## main model: EfficientNet-B0, lesion-level split, degradation augmentation
	$(PY) -m melanoma.train --config configs/default.yaml

ablations:          ## baselines used in the README comparison table (no degradation augmentation)
	$(ABL) experiment=effnet_b0_lesion
	$(ABL) experiment=effnet_b0_image_split data.split_mode=image
	$(ABL) experiment=mobilenetv2_frozen_lesion model.name=mobilenetv2_100 model.freeze_backbone=true train.lr=1.0e-3
	$(ABL) experiment=mobilenetv2_finetune_lesion model.name=mobilenetv2_100

robustness:         ## resolution / JPEG stress test for the main model and the non-augmented baseline
	$(PY) scripts/robustness.py --run runs/effnet_b0_lesion_robust --out runs/effnet_b0_lesion_robust/robustness
	$(PY) scripts/robustness.py --run runs/effnet_b0_lesion --out runs/effnet_b0_lesion/robustness

report:             ## tables, figures, Grad-CAM gallery and app checkpoint -> reports/, models/
	$(PY) scripts/make_report.py --main effnet_b0_lesion_robust --baseline effnet_b0_lesion

evaluate:           ## re-evaluate a run: make evaluate RUN=runs/effnet_b0_lesion_robust
	$(PY) -m melanoma.evaluate --run $(RUN)

app:
	streamlit run app/streamlit_app.py

test:
	pytest

lint:
	ruff check src tests scripts app && ruff format --check src tests scripts app
