"""Every filesystem location the suite uses, in one place.

Machine-specific values are read from environment variables with the PC values as defaults, so the code is
never edited when moving machines: on the PC nothing is set; on Pitt CRC `crc/env.sh` exports the variables
before every job. Everything inside the suite is found relative to this file.

Environment variables (all optional on the PC):
    SARDET100K_ROOT               folder containing JPEGImages/, Annotations/, test.json, mapping.json
    MOBILENETV2_IMAGENET_WEIGHTS  torchvision IMAGENET1K_V2 MobileNetV2 state dict (.pth)
    SAR_SPLIT                     which data split to use: 1, 2 or 3 (default 1)
    SARDET_OUTPUT_ROOT            optional: put every experiment's outputs/ under this folder instead of the
                                  experiment folder (e.g. a scratch disk); default = inside each experiment folder
"""
import os
from pathlib import Path

# ---- Machine-specific (environment variable, PC default) -------------------------------------------------
SARDET100K_ROOT = Path(os.environ.get(
    "SARDET100K_ROOT", r"C:\Users\nisch\Documents\SHREC\2026_2027\SARDet-work\Dataset_SARDet\SARDet_100K"))
MOBILENETV2_IMAGENET_WEIGHTS = Path(os.environ.get(
    "MOBILENETV2_IMAGENET_WEIGHTS",
    r"C:\Users\nisch\Documents\SHREC\2026_2027\mobilenetv2_SAR_SSL\MobileNetV2\checkpoints\pretrained"
    r"\mobilenet_v2_imagenet1k_v2.pth"))
OUTPUT_ROOT = Path(os.environ["SARDET_OUTPUT_ROOT"]) if os.environ.get("SARDET_OUTPUT_ROOT") else None

# ---- Dataset layout (as released; normally no need to edit) ----------------------------------------------
IMAGES_DIR = SARDET100K_ROOT / "JPEGImages"            # train/, val/, test/


def _first_existing(*candidates):
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


ANNOTATION_FILES = {
    "train": SARDET100K_ROOT / "Annotations" / "train.json",
    "val": SARDET100K_ROOT / "Annotations" / "val.json",
    # the archive ships test.json next to Annotations/, not inside it; accept either place
    "test": _first_existing(SARDET100K_ROOT / "test.json", SARDET100K_ROOT / "Annotations" / "test.json"),
}
MAPPING_JSON = SARDET100K_ROOT / "mapping.json"        # original file name -> anonymised file name
CLASS_NAMES = ["ship", "aircraft", "car", "tank", "bridge", "harbor"]   # category_id 0..5 in the JSON

# ---- Inside the suite --------------------------------------------------------------------------------------
SUITE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = SUITE_ROOT / "data"                         # metadata and split manifests (committed)
METADATA_CSV = DATA_DIR / "image_metadata.csv"
SPLIT = int(os.environ.get("SAR_SPLIT", "1"))
SPLIT_DIR = DATA_DIR / f"split{SPLIT}"                 # labeled_1pct.csv, labeled_3pct.csv, labeled_10pct.csv, ssl_pool.csv
VAL_SELECT_CSV = DATA_DIR / "val_select.csv"           # fixed subset of official val used for per-epoch model selection
EVAL_SUBSETS_JSON = DATA_DIR / "eval_subsets.json"     # official val/test image ids and the *_clean subsets
SHARED_FINETUNE_CONFIG = SUITE_ROOT / "shared" / "finetune_config.json"
SHARED_SSL_CONFIG = SUITE_ROOT / "shared" / "ssl_common.json"
RUN_TAG = f"_split{SPLIT}"


def image_path(official_split: str, file_name: str) -> Path:
    """Absolute path of one image. official_split is the release folder: train, val or test."""
    return IMAGES_DIR / official_split / file_name


def experiment_output_dir(experiment_dir) -> Path:
    """outputs/ of an experiment folder (or the same folder name under SARDET_OUTPUT_ROOT when set)."""
    experiment_dir = Path(experiment_dir)
    if OUTPUT_ROOT is not None:
        return OUTPUT_ROOT / experiment_dir.name / "outputs"
    return experiment_dir / "outputs"


def check(verbose=True) -> bool:
    """Report which configured locations exist; used by every smoke test."""
    items = [("SARDET100K_ROOT", SARDET100K_ROOT), ("IMAGES_DIR", IMAGES_DIR),
             ("train.json", ANNOTATION_FILES["train"]), ("val.json", ANNOTATION_FILES["val"]),
             ("test.json", ANNOTATION_FILES["test"]), ("mapping.json", MAPPING_JSON),
             ("MOBILENETV2_IMAGENET_WEIGHTS", MOBILENETV2_IMAGENET_WEIGHTS), ("METADATA_CSV", METADATA_CSV),
             ("SPLIT_DIR", SPLIT_DIR), ("VAL_SELECT_CSV", VAL_SELECT_CSV)]
    ok = True
    for name, p in items:
        exists = p.exists()
        ok &= exists
        if verbose:
            print(f"  {'ok     ' if exists else 'MISSING'}  {name:30s} {p}")
    return ok


if __name__ == "__main__":
    print(f"split {SPLIT}; output root {OUTPUT_ROOT or '(inside each experiment folder)'}")
    check()
