"""Where the chip study reads and writes (named chip_paths so it never shadows the reused Experiments paths module). Machine-specific values come from environment variables (PC defaults).

    SARDET_CHIPS_ROOT   folder holding chips_u8.npy, chips.csv, chips_meta.json, previews/ (made by make_chips.py)
                        PC default: next to the SARDet-100K dataset folder; CRC: /ix/ageorge/nik96/data/SARDet_chips
                        (set in crc/env.sh; a job may point it at node-local disk after staging)
Everything about the SARDet-100K dataset itself (images, annotations, split manifests, class names) comes from the
detection suite's SARDet/shared/paths.py, so there is exactly one place that knows where the dataset is.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SUITE_ROOT = REPO_ROOT / "SARDet"
DINO_DIR = REPO_ROOT / "Experiments" / "mobileNetv2_DINO"       # model.py, engine.py, utils.py, plots.py are reused from here
for p in (str(SUITE_ROOT), str(DINO_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
from shared import paths as suite_paths  # noqa: E402

SARDET100K_ROOT = suite_paths.SARDET100K_ROOT
IMAGES_DIR = suite_paths.IMAGES_DIR                 # <root>/JPEGImages/{train,val,test}/
ANNOTATION_FILES = suite_paths.ANNOTATION_FILES     # official COCO json per split
METADATA_CSV = suite_paths.METADATA_CSV             # SARDet/data/image_metadata.csv (source, group per image)
SUITE_DATA_DIR = suite_paths.DATA_DIR               # SARDet/data: ssl_pool.csv, val_select.csv, split{k}/labeled_*.csv
EVAL_SUBSETS_JSON = suite_paths.EVAL_SUBSETS_JSON
CLASS_NAMES = list(suite_paths.CLASS_NAMES)         # ship, aircraft, car, tank, bridge, harbor (label = index)
MOBILENETV2_IMAGENET_WEIGHTS = suite_paths.MOBILENETV2_IMAGENET_WEIGHTS

CHIPS_ROOT = Path(os.environ.get("SARDET_CHIPS_ROOT", SARDET100K_ROOT.parent / "SARDet_chips"))
CHIPS_NPY = CHIPS_ROOT / "chips_u8.npy"
CHIPS_CSV = CHIPS_ROOT / "chips.csv"
CHIPS_META = CHIPS_ROOT / "chips_meta.json"
PREVIEW_DIR = CHIPS_ROOT / "previews"

OUTPUT_ROOT = HERE / "outputs"
FINETUNE_OUTPUT_DIR = OUTPUT_ROOT / "finetune"
COMPARE_OUTPUT_DIR = OUTPUT_ROOT / "compare"

# chip rule (DESIGN.md section 2); recorded in chips_meta.json and in every run's config
CHIP_SIZE = 128          # stored chip side, px (the classifier resizes to its input size, as for ATRNet chips)
WINDOW_FLOOR = 64        # smallest window side in native pixels
WINDOW_MARGIN = 1.5      # window side = max(WINDOW_FLOOR, WINDOW_MARGIN x longer box side)
