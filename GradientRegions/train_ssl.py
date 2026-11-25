"""RegionDINO object-only with regions from DINOSAR's ratio-gradient content map, SARDet-100K pool. See DESIGN.md.

    python GradientRegions/train_ssl.py                       # full run, config_ssl.json, region file outputs/regions_sardet/regions.npz
    python GradientRegions/train_ssl.py --config GradientRegions/config_ssl_random.json   # the gradient-size random control
    python GradientRegions/train_ssl.py --smoke               # 48 images, 2 epochs (needs outputs/regions_sardet/regions_smoke.npz: make_regions.py --smoke)
    python GradientRegions/train_ssl.py --resume <run folder>
    CRC: cd SARDet && sbatch --job-name=rog_s1 crc/ssl.sbatch ../GradientRegions 1 config_ssl.json

The only difference from RegionSSL/RegionDINO_object/train_ssl.py is where the region file comes from: this script
points REGIONSSL_REGIONS_DIR at GradientRegions/outputs/regions_sardet unless the variable is already set, then runs
the same object-only method. Run folders: GradientRegions/outputs/ssl/<stamp>_regiondino_o_gradient[_random]_ssl_mobilenetv2_split<s>_seed<s-1>/
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("REGIONSSL_REGIONS_DIR", str(HERE / "outputs" / "regions_sardet"))
sys.path.insert(0, str(HERE.parent / "RegionSSL"))
from rssl import region_engine  # noqa: E402
from rssl.regiondino import RegionDINOMethod  # noqa: E402


class GradientRegionMethod(RegionDINOMethod):
    name = "regiondino_object"
    mode = "object"


if __name__ == "__main__":
    print(f"region file folder: {os.environ['REGIONSSL_REGIONS_DIR']}", flush=True)
    region_engine.cli(GradientRegionMethod, HERE)
