"""All filesystem paths for this project, in one place.

Paths inside the repository are found relative to this file, so
they work on any machine without editing.
"""
import os
from pathlib import Path

# ---- Edit these per PC ----------------------------------------------------
# Folder that contains Amplitude_8bit/ and annotation_coco/ from the Hugging Face download.
# On another machine (e.g. Pitt CRC) set the environment variable ATRNET_GROUND_RANGE instead of
# editing this line; the job scripts in crc/ do that.
ATRNET_GROUND_RANGE = Path(os.environ.get("ATRNET_GROUND_RANGE",
                                          r"C:\Users\nisch\Documents\SHREC\2026_2027\ATRNet-STAR\Ground_Range"))

# ---- Derived dataset paths (normally no need to edit) ----------------------
# The extracted folder name is literally URL-encoded on disk; keep it exactly as is.
ATRNET_AMPLITUDE_8BIT = (
    ATRNET_GROUND_RANGE
    / "Amplitude_8bit"
    / "Amplitude%208-bit%20data_%E5%9C%B0%E8%B7%9D%E5%B9%85%E5%BA%A68%E4%BD%8D%E6%95%B0%E6%8D%AE"
)
SOC40_DIR = ATRNET_AMPLITUDE_8BIT / "SOC_40classes"  # contains train/ and test/
SOC40_TRAIN_DIR = SOC40_DIR / "train"
SOC40_TEST_DIR = SOC40_DIR / "test"

# ---- Paths inside the repository ------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
SPLITS_DIR = REPO_ROOT / "preprocessing_dataset"  # validation.csv, ssl_training_data.csv, labeled_set.csv
BASELINE_OUTPUT_DIR = REPO_ROOT / "Experiments" / "Baseline" / "Baseline_Output"  # one subfolder per run
# Override with the environment variable MOBILENETV2_IMAGENET_WEIGHTS on another machine (crc/env.sh does).
MOBILENETV2_IMAGENET_WEIGHTS = Path(os.environ.get(
    "MOBILENETV2_IMAGENET_WEIGHTS",
    REPO_ROOT / "MobileNetV2" / "checkpoints" / "pretrained" / "mobilenet_v2_imagenet1k_v2.pth"))
