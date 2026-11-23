"""Where the FAIR-CSAR chip study reads and writes. Machine-specific values come from environment variables.

    FAIR_CSAR_ROOT    folder holding trainval/ and test/, each with PNGImages/, Annotations/, METAXmls/, KeyPoints/
                      (the extracted SL archives without the .mat files). PC default: the sample folder next to the
                      repository (annotations complete, only a few PNGs); CRC: /ix/ageorge/nik96/data/FAIR_CSAR/extracted
                      (set in crc/env.sh; a job may point it at node-local disk after staging).
    FAIR_CHIPS_ROOT   folder for chips_u8.npy, chips.csv, chips_meta.json, previews/ (made by make_chips.py).
                      PC default: next to FAIR_CSAR_ROOT; CRC: /ix/ageorge/nik96/data/FAIR_chips.
Everything else is relative to this file. The ImageNet start weights and the chip rule come from
SARDet_chips/chip_paths.py (one place that knows them); this module does not change sys.path.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SARDET_CHIPS_DIR = REPO_ROOT / "SARDet_chips"


def _chip_paths():
    """SARDet_chips/chip_paths.py for its values only. Importing it pushes SARDet/ and Experiments/mobileNetv2_DINO/ to the
    front of sys.path; this module is also imported inside the SSL process (through RegionSSL/rssl/pools.py), so sys.path
    is restored afterwards. Scripts that need those folders (finetune_chips.py, make_chips.py) add them themselves."""
    saved = list(sys.path)
    sys.path.insert(0, str(SARDET_CHIPS_DIR))
    import chip_paths  # noqa: E402
    sys.path[:] = saved
    return chip_paths


_cp = _chip_paths()
MOBILENETV2_IMAGENET_WEIGHTS = _cp.MOBILENETV2_IMAGENET_WEIGHTS

FAIR_CSAR_ROOT = Path(os.environ.get("FAIR_CSAR_ROOT", REPO_ROOT.parent / "FAIR_CSAR_sample"))
CHIPS_ROOT = Path(os.environ.get("FAIR_CHIPS_ROOT", FAIR_CSAR_ROOT.parent / "FAIR_chips"))
CHIPS_NPY, CHIPS_CSV, CHIPS_META = CHIPS_ROOT / "chips_u8.npy", CHIPS_ROOT / "chips.csv", CHIPS_ROOT / "chips_meta.json"
PREVIEW_DIR = CHIPS_ROOT / "previews"

SPLITS = ("trainval", "test")                       # the dataset's own split folders
SLICE_SIZE = 1024                                    # every SL slice is 1024 x 1024 px at 1 m (inventory 2026-11-22)


def png_path(split: str, slice_name: str) -> Path:
    return FAIR_CSAR_ROOT / split / "PNGImages" / f"{slice_name}.png"


def annotation_path(split: str, slice_name: str) -> Path:
    return FAIR_CSAR_ROOT / split / "Annotations" / f"{slice_name}.txt"


# ---- committed manifests (made by make_manifests.py) ------------------------------------------------------
DATA_DIR = HERE / "data"
SLICES_CSV = DATA_DIR / "slices.csv"                 # every slice: split, scene, objects, role
SSL_POOL_CSV = DATA_DIR / "ssl_pool.csv"             # trainval slices minus the selection slices (same for every split)
SELECT_CSV = DATA_DIR / "select.csv"                 # fixed trainval slices held out for per-epoch model selection
SSL_TILES_CSV = DATA_DIR / "ssl_tiles.csv"           # the pool as unique 512-px tiles (slice, tile): the fair_tiles pool rows
OBJECTS_CSV = DATA_DIR / "objects.csv"               # one row per physical object (slices overlap 50 %): owner slice + obj_id
HASHES_JSON = DATA_DIR / "hashes.json"
BUDGETS = {"1pct": 0.01, "3pct": 0.03, "10pct": 0.10}
LABEL_FLOOR = 5                                      # smallest number of chips per class in any labeled set


def labeled_csv(split: int, budget: str) -> Path:
    return DATA_DIR / f"split{split}" / f"labeled_{budget}.csv"


# ---- classes (fixed order = label index; the 22 subCategory names of FAIR-CSAR-V1.0 SL) -------------------
CLASS_NAMES = ["Airbus_A220", "Airbus_A320", "Airbus_A330", "Airfreighter", "Boeing737", "Boeing747", "Boeing767",
               "Boeing777", "Fokker-50", "Gulfstream", "Helicopter", "Other_Aircraft",
               "Bulk_Carrier", "Container_Ship", "General_Cargo_Ship", "Motion_Defocusing_Ship", "Oil_Tanker",
               "Other_Ship", "Warship", "Bridge", "Tank", "Tower_Crane"]
MAJOR = {**{c: "Aircraft" for c in CLASS_NAMES[:12]}, **{c: "Ship" for c in CLASS_NAMES[12:19]},
         "Bridge": "Bridge", "Tank": "Tank", "Tower_Crane": "Tower_Crane"}

# ---- chip rule: the SARDet_chips rule unchanged (DESIGN.md section 3) -------------------------------------
CHIP_SIZE, WINDOW_FLOOR, WINDOW_MARGIN = _cp.CHIP_SIZE, _cp.WINDOW_FLOOR, _cp.WINDOW_MARGIN   # 128, 64, 1.5

# ---- fine-tune method names (here, not in finetune_chips.py, so compare.py never imports a module that
# SARDet_chips/finetune_chips.py would shadow on sys.path) ----
FAIR_METHODS = [f"fair_{m}_{arm}" for arm in ("tiles", "down") for m in ("dino", "ro", "ror")]
SARDET_METHODS = {"sardet_dino": "dino", "sardet_regiondino_o": "regiondino_o", "sardet_regiondino_o_random": "regiondino_o_random"}
METHODS = ["imagenet"] + FAIR_METHODS + list(SARDET_METHODS)

OUTPUT_ROOT = HERE / "outputs"
FINETUNE_OUTPUT_DIR = OUTPUT_ROOT / "finetune"
COMPARE_OUTPUT_DIR = OUTPUT_ROOT / "compare"


def check(verbose=True) -> bool:
    items = [("FAIR_CSAR_ROOT", FAIR_CSAR_ROOT), ("trainval annotations", FAIR_CSAR_ROOT / "trainval" / "Annotations"),
             ("test annotations", FAIR_CSAR_ROOT / "test" / "Annotations"), ("trainval PNGs", FAIR_CSAR_ROOT / "trainval" / "PNGImages"),
             ("test PNGs", FAIR_CSAR_ROOT / "test" / "PNGImages"), ("CHIPS_ROOT", CHIPS_ROOT), ("SSL_POOL_CSV", SSL_POOL_CSV),
             ("MOBILENETV2_IMAGENET_WEIGHTS", MOBILENETV2_IMAGENET_WEIGHTS)]
    ok = True
    for name, p in items:
        ok &= p.exists()
        if verbose:
            print(f"  {'ok     ' if p.exists() else 'MISSING'}  {name:30s} {p}")
    return ok


if __name__ == "__main__":
    check()
