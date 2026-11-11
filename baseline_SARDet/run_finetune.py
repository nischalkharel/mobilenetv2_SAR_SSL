"""baseline_SARDet: fine-tune the ImageNet MobileNetV2 detector on one labeled budget of the current split.

    python baseline_SARDet/run_finetune.py --budget 1pct|3pct|10pct [--seed 0] [--smoke N --epochs 1]

Everything else (schedule, augmentation, evaluation) is the shared protocol in shared/finetune.py.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared import finetune  # noqa: E402

if __name__ == "__main__":
    cfg = json.load(open(HERE / "config.json", encoding="utf-8"))
    finetune.main(["--experiment-dir", str(HERE), "--run-name", cfg["run_name"], "--backbone", cfg["backbone"], *sys.argv[1:]])
