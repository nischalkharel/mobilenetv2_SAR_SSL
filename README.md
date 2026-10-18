Dataset: https://huggingface.co/datasets/waterdisappear/ATRNet-STAR/tree/main
# Using Ground_Range


# Environment
# Python: Python 3.13.4
# PyTorch (GPU build): pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
# (needs an NVIDIA driver supporting CUDA 13.0+; no separate CUDA toolkit needed. For a CPU-only PC use .../whl/cpu)
# Other packages: pip install numpy pillow matplotlib   (tested: numpy 2.5.2, pillow 12.3.0, matplotlib 3.11.2)


# Steps:

### STEP 0:
Edit `paths.py` so `ATRNET_GROUND_RANGE` points to the Ground_Range folder on this PC. All scripts read paths from there.

### STEP 1: 
Download ImageNet1k_v2 weight using MobileNetV2/download_mobileNetV2.py

### STEP 2: Fixed SOC-40 subsets (already generated; do not regenerate between experiments)
From the repository root:

    python preprocessing_dataset/preprocessing_dataset.py

Writes into `preprocessing_dataset/`: `validation.csv` (800 images: 20 per class from official SOC-40 train, each from a different pose), `excluded_near_duplicates.csv` (1,904 near-copies of validation images: same vehicle, angle and scene, differing only by polarization or repeat pass; used nowhere, each row records its validation image and their similarity), `ssl_training_data.csv` (the other 65,387 train images), `labeled_set.csv` (3% of the SSL pool, per class: 1,961 images, a small labeled budget) and `class_mapping.csv`. It also writes `SUBSETS.md` and `preprocessing_dataset/charts/` (tables and charts of images per subset and per class, and the similarity of the excluded images); `python preprocessing_dataset/subset_report.py` redraws those alone. The official SOC-40 `test/` is never in any CSV. The script refuses to overwrite existing CSVs unless given `--overwrite`.

### STEP 3: A0 supervised baseline
Full instructions, the screen output and the output files are described in `Experiments/Baseline/README.md`. In order, from the repository root:

    source .venv/Scripts/activate
    python Experiments/Baseline/smoke_test.py      # optional pipeline check, about 1 min
    python Experiments/Baseline/train.py           # trains; creates Baseline_Output/<date-time>_A0_imagenet_mobilenetv2_seed0/
    python Experiments/Baseline/evaluate.py --run "Experiments/Baseline/Baseline_Output/<that run folder>"   # creates eval_test/ inside it
