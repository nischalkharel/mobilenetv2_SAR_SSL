Dataset: https://huggingface.co/datasets/waterdisappear/ATRNet-STAR/tree/main
# Using Ground_Range


# Environment
# Python: Python 3.13.4


# Steps:

### STEP 0:
Edit `paths.py` so `ATRNET_GROUND_RANGE` points to the Ground_Range folder on this PC. All scripts read paths from there.

### STEP 1: 
Download ImageNet1k_v2 weight using MobileNetV2/download_mobileNetV2.py

### STEP 2: Fixed SOC-40 subsets (already generated; do not regenerate between experiments)
From the repository root:

    python preprocessing_dataset/preprocessing_dataset.py

Writes `validation.csv` (~10% of SOC-40 train, chosen by whole pose groups), `ssl_training_data.csv` (rest of train), `labeled_set.csv` (10% of the SSL pool, per class), and `class_mapping.csv` into `preprocessing_dataset/`. The official SOC-40 `test/` is never in any CSV. The script refuses to overwrite existing CSVs unless given `--overwrite`.
