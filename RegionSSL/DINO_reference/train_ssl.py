"""Plain DINO with the RegionSSL engine's scene recipe (no object views): the paired reference for RegionDINO.

    python RegionSSL/DINO_reference/train_ssl.py --config RegionSSL/DINO_reference/config_ssl_atr.json   # ATRNet pool
    python RegionSSL/DINO_reference/train_ssl.py --config ... --smoke
    python RegionSSL/DINO_reference/train_ssl.py --resume <run folder>

On the SARDet pool this is DINO_SARDet (same recipe, same engine lineage), so no SARDet config lives here. On the
ATRNet pool it differs from A1 only in how the crops are drawn (window-and-zoom of the suite vs area-scale crops), so
RegionDINO_*_atr minus this row isolates the object views, and this row minus A1 measures the recipe difference.
Exports the teacher backbone in the shared format.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rssl import region_engine  # noqa: E402
from rssl.regiondino import RegionDINOMethod  # noqa: E402


class SceneOnlyMethod(RegionDINOMethod):
    name = "regiondino_scene"
    mode = "scene"


if __name__ == "__main__":
    region_engine.cli(SceneOnlyMethod, HERE)
