"""ViewScale: DINO on random square windows of ONE fixed size, SARDet-100K pool (the view-size sweep; see DESIGN.md).

    python ViewScale/train_ssl.py --config ViewScale/config_ssl_e64.json             # one sweep point, full run
    python ViewScale/train_ssl.py --config ViewScale/config_ssl_e64.json --smoke     # 48 images, 2 epochs
    python ViewScale/train_ssl.py --config ViewScale/config_ssl_e64.json --resume <run folder>

Same method class as RegionSSL/RegionDINO_object (loss on the small views only, no whole-scene view); the configs
set region_source "fixed" and fixed_extent_px. Exports the teacher backbone in the shared format, so the ATRNet
transfer (SARDet_to_ATR/finetune_atr.py) and the detection fine-tune (SARDet/shared/finetune.py) load it unchanged.
Run folders: ViewScale/outputs/ssl/<stamp>_viewscale_e<k>_ssl_mobilenetv2_split<s>_seed<s-1>/
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "RegionSSL"))
from rssl import region_engine  # noqa: E402
from rssl.regiondino import RegionDINOMethod  # noqa: E402


class FixedWindowMethod(RegionDINOMethod):
    name = "viewscale"
    mode = "object"


if __name__ == "__main__":
    region_engine.cli(FixedWindowMethod, HERE)
