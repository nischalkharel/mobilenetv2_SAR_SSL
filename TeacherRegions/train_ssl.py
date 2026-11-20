"""RegionDINO object-only with TEACHER-proposed regions (SARMAE saliency), SARDet-100K pool. See DESIGN.md.

    python TeacherRegions/train_ssl.py                       # full run, config_ssl.json, region file outputs/regions_cls/regions.npz
    python TeacherRegions/train_ssl.py --smoke               # 48 images, 2 epochs (needs outputs/regions_cls/regions_smoke.npz: teacher_saliency.py --smoke)
    python TeacherRegions/train_ssl.py --resume <run folder>
    REGIONSSL_REGIONS_DIR=TeacherRegions/outputs/regions_norm python TeacherRegions/train_ssl.py --config ... (the token-norm file)

The only difference from RegionSSL/RegionDINO_object/train_ssl.py is where the region file comes from: this script
points REGIONSSL_REGIONS_DIR at TeacherRegions/outputs/regions_cls unless the variable is already set, then runs the
same object-only method. Run folders: TeacherRegions/outputs/ssl/<stamp>_regiondino_o_teacher_ssl_mobilenetv2_split<s>_seed<s-1>/
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("REGIONSSL_REGIONS_DIR", str(HERE / "outputs" / "regions_cls"))
sys.path.insert(0, str(HERE.parent / "RegionSSL"))
from rssl import region_engine  # noqa: E402
from rssl.regiondino import RegionDINOMethod  # noqa: E402


class TeacherRegionMethod(RegionDINOMethod):
    name = "regiondino_object"
    mode = "object"


if __name__ == "__main__":
    print(f"region file folder: {os.environ['REGIONSSL_REGIONS_DIR']}", flush=True)
    region_engine.cli(TeacherRegionMethod, HERE)
