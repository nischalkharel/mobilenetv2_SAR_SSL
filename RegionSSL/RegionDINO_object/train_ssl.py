"""RegionDINO (object only) self-supervised training of MobileNetV2 on the SARDet-100K pool.

    python RegionSSL/RegionDINO_object/train_ssl.py                        # full run (config_ssl.json)
    python RegionSSL/RegionDINO_object/train_ssl.py --config RegionSSL/RegionDINO_object/config_ssl_random.json
    python RegionSSL/RegionDINO_object/train_ssl.py --smoke                # 48 images, 2 epochs
    python RegionSSL/RegionDINO_object/train_ssl.py --resume <run folder>

Needs the region file (RegionSSL/make_regions.py, or --smoke's regions_smoke.npz). Loss = DINO on object views ONLY; the
network never sees a whole-image view (rssl/regiondino.py). Exports the teacher backbone in the shared
format, so SARDet/shared/finetune.py and SARDet_to_ATR/finetune_atr.py load it unchanged.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rssl import region_engine  # noqa: E402
from rssl.regiondino import RegionDINOMethod  # noqa: E402


class ObjectOnlyMethod(RegionDINOMethod):
    name = "regiondino_object"
    mode = "object"


if __name__ == "__main__":
    region_engine.cli(ObjectOnlyMethod, HERE)
