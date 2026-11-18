"""Arm A of the cross-dataset study: fine-tune a SARDet-100K-pretrained backbone on an ATRNet-STAR labeled set.

    python SARDet_to_ATR/finetune_atr.py --method dino --backbone <SARDet SSL run>/backbone_epoch100.pth [--train-csv labeled_set_1pct.csv] [--smoke]

This is a thin wrapper: the training itself is Experiments/mobileNetv2_DINO/train_supervised.py (the A1/A2/A3
fine-tune code, byte-identical across those folders) with SARDet_to_ATR/config_finetune.json, which equals the
A1 config except for the run name. Nothing in Experiments/ is modified. Run folders land in
SARDet_to_ATR/outputs/finetune/<stamp>_XA_<method>_sardetssl_mobilenetv2[_<budget>][_split<s>]_seed0/ and are
scored with Experiments/mobileNetv2_DINO/evaluate.py --run <folder> --split validation|test.
The split (which ATRNet labeled draw) comes from the environment variable SAR_SPLIT, as for every ATRNet run.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DINO_DIR = REPO / "Experiments" / "mobileNetv2_DINO"
sys.path.insert(0, str(DINO_DIR))          # train_supervised imports its neighbours (paths, data, model, engine)
import train_supervised  # noqa: E402

METHODS = ("dino", "lejepa", "spark", "regiondino_so", "regiondino_so_random", "regiondino_o", "regiondino_o_random")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=METHODS, help="which SARDet SSL method produced the backbone")
    ap.add_argument("--backbone", required=True, type=Path, help="exported SARDet backbone (backbone_epoch100.pth)")
    ap.add_argument("--train-csv", help="labeled_set_1pct.csv / labeled_set_10pct.csv (default labeled_set.csv = 3 %%)")
    ap.add_argument("--smoke", action="store_true", help="a few images per class, 2 epochs; pipeline check only")
    ap.add_argument("--device", help='override train.device: "cuda", "cpu" or "auto"')
    args = ap.parse_args()
    if not args.backbone.exists():
        raise SystemExit(f"backbone file not found: {args.backbone}")
    cfg = json.loads((HERE / "config_finetune.json").read_text(encoding="utf-8"))
    cfg["run_name"] = cfg["run_name"].replace("METHOD", args.method)
    cfg["init"] = {"type": "backbone_checkpoint", "backbone_checkpoint": str(args.backbone.resolve())}
    cfg["cross_dataset"] = {"ssl_dataset": "SARDet-100K", "downstream_dataset": "ATRNet-STAR SOC-40", "arm": "A (SARDet SSL as run)",
                            "ssl_method": args.method}
    if args.device:
        cfg["train"]["device"] = args.device
    if args.train_csv:
        cfg["data"]["train_csv"] = args.train_csv
        cfg["run_name"] += "_" + Path(args.train_csv).stem
    out_root = HERE / "outputs" / ("smoke" if args.smoke else "finetune")
    run_dir = train_supervised.run(cfg, smoke=args.smoke, output_root=out_root)
    print(f"\nscore it with:\n  python {DINO_DIR.relative_to(REPO) / 'evaluate.py'} --run \"{run_dir}\" --split validation\n"
          f"  python {DINO_DIR.relative_to(REPO) / 'evaluate.py'} --run \"{run_dir}\" --split test")


if __name__ == "__main__":
    main()
