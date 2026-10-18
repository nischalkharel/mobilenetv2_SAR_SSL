"""All filesystem paths for the MobileNetV2 + DINO experiment, in one place.

This file is local to the experiment on purpose: the experiment must run even if the
Baseline folder (or the repository-level paths.py) is removed. It is a separate copy of
the same settings, so when moving to another PC, edit ATRNET_GROUND_RANGE here as well.

External dependencies, configured here and never copied:
    - the ATRNet-STAR image files (ATRNET_GROUND_RANGE)
    - the subset CSVs in <repository>/preprocessing_dataset/ (SPLITS_DIR)
    - the saved ImageNet MobileNetV2 weights (MOBILENETV2_IMAGENET_WEIGHTS)
"""
import os
from pathlib import Path

# ---- Edit this per PC ------------------------------------------------------------
# Folder that contains Amplitude_8bit/ and annotation_coco/ from the Hugging Face download.
# On another machine (e.g. Pitt CRC) set the environment variable ATRNET_GROUND_RANGE instead of
# editing this line; the job scripts in crc/ do that.
ATRNET_GROUND_RANGE = Path(os.environ.get("ATRNET_GROUND_RANGE",
                                          r"C:\Users\nisch\Documents\SHREC\2026_2027\ATRNet-STAR\Ground_Range"))

# ---- Derived dataset paths (normally no need to edit) ----------------------------
# The extracted folder name is literally URL-encoded on disk; keep it exactly as is.
ATRNET_AMPLITUDE_8BIT = (
    ATRNET_GROUND_RANGE
    / "Amplitude_8bit"
    / "Amplitude%208-bit%20data_%E5%9C%B0%E8%B7%9D%E5%B9%85%E5%BA%A68%E4%BD%8D%E6%95%B0%E6%8D%AE"
)
SOC40_DIR = ATRNET_AMPLITUDE_8BIT / "SOC_40classes"  # contains train/ and test/
SOC40_TRAIN_DIR = SOC40_DIR / "train"
SOC40_TEST_DIR = SOC40_DIR / "test"

# ---- Paths found relative to this file ---------------------------------------------
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
SPLITS_DIR = REPO_ROOT / "preprocessing_dataset"  # validation.csv, ssl_training_data.csv, labeled_set.csv, ...
MOBILENETV2_IMAGENET_WEIGHTS = (
    REPO_ROOT / "MobileNetV2" / "checkpoints" / "pretrained" / "mobilenet_v2_imagenet1k_v2.pth"
)

# ---- Outputs of this experiment (one subfolder per run, never overwritten) --------------
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
SSL_OUTPUT_DIR = OUTPUT_DIR / "ssl"            # DINO runs: checkpoint.pth, exported backbone, history
FINETUNE_OUTPUT_DIR = OUTPUT_DIR / "finetune"  # supervised runs started from an exported backbone (or ImageNet)
SMOKE_OUTPUT_DIR = OUTPUT_DIR / "smoke"        # smoke-test runs only; not results
CACHE_DIR = OUTPUT_DIR / "cache"               # decoded SSL pixels (one .npy per image list); safe to delete
