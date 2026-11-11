"""SparK_SARDet stage 2: fine-tune the shared detector from an exported SparK encoder on one labeled budget.

    python SparK_SARDet/run_finetune.py --backbone <outputs/ssl/<run>/backbone_epoch100.pth> --budget 1pct|3pct|10pct

Identical protocol to the baseline (shared/finetune.py); only the backbone file differs.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared import finetune  # noqa: E402

if __name__ == "__main__":
    cfg = json.load(open(HERE / "config_finetune.json", encoding="utf-8"))
    finetune.main(["--experiment-dir", str(HERE), "--run-name", cfg["run_name"], *sys.argv[1:]])
