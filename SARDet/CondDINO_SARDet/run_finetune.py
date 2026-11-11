"""CondDINO_SARDet stage 2: fine-tune the shared detector from an exported CondDINO backbone on one labeled budget.

    python CondDINO_SARDet/run_finetune.py --backbone <outputs/ssl/<run>/backbone_epoch100.pth> --budget 1pct|3pct|10pct
    python CondDINO_SARDet/run_finetune.py --run-name condcenter_perm_mobilenetv2 --backbone ... --budget ...

Identical protocol to the baseline (shared/finetune.py); only the backbone file differs. The default run name is
config_finetune.json's; give --run-name <variant>_mobilenetv2 for the perm / oracle_class / radiometric backbones
so their fine-tune folders are told apart (the backbone file itself records the variant and its label use).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared import finetune  # noqa: E402

if __name__ == "__main__":
    cfg = json.load(open(HERE / "config_finetune.json", encoding="utf-8"))
    argv = sys.argv[1:]
    run_name = cfg["run_name"]
    if "--run-name" in argv:
        i = argv.index("--run-name")
        run_name = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    finetune.main(["--experiment-dir", str(HERE), "--run-name", run_name, *argv])
