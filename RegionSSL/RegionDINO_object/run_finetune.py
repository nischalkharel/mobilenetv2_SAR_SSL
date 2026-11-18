"""Stage 2 on SARDet-100K detection: fine-tune the shared detector from an exported RegionDINO backbone.

    python RegionSSL/RegionDINO_object/run_finetune.py --backbone <outputs/ssl/<run>/backbone_epoch100.pth> --budget 1pct|3pct|10pct

Identical protocol to the suite's baseline (SARDet/shared/finetune.py, read-only); only the backbone differs.
For the ATRNet-STAR fine-tune use SARDet_to_ATR/finetune_atr.py --method regiondino_o.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import rssl  # noqa: E402, F401  (puts SARDet on sys.path)
from shared import finetune  # noqa: E402

if __name__ == "__main__":
    cfg = json.load(open(HERE / "config_finetune.json", encoding="utf-8"))
    finetune.main(["--experiment-dir", str(HERE), "--run-name", cfg["run_name"], *sys.argv[1:]])
