"""Evaluate a saved best.pt on the official SOC-40 test set (or on validation).

Copied from Experiments/Baseline/evaluate.py. Difference: it reads the class mapping and the
validation list from the copies saved inside the run folder (manifests/), and refuses to run if
those copies no longer match the hashes recorded at training time.

Run from the repository root:
    python Experiments/mobileNetv2_SparK/evaluate.py --run Experiments/mobileNetv2_SparK/outputs/finetune/<run folder> --split validation
    python Experiments/mobileNetv2_SparK/evaluate.py --run <run folder>        # official test set, final only

Writes into <run folder>/eval_<split>/:
    metrics.json              overall accuracy, mean per-class accuracy, loss, checkpoint info
    per_class_accuracy.csv    accuracy and image count for each class
    confusion_matrix.csv      rows = true class, columns = predicted class (image counts)
    predictions.csv           one row per image: path, true, predicted, confidence

The checkpoint was chosen with validation only. Test results must not be used to pick
checkpoints or change settings.
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import SARDataset, build_transform, load_class_names, manifest_hash, read_manifest, test_manifest
from engine import predict
from model import load_finetuned
from plots import plot_eval
from train_supervised import SMOKE_PER_CLASS, first_per_class
from utils import environment_info, resolve_device, save_json, sha256_file


def recorded_manifests(run_dir: Path) -> Path:
    """The run's own copies of its CSVs, after checking they match what training recorded."""
    folder = run_dir / "manifests"
    rec = json.loads((run_dir / "manifests.json").read_text(encoding="utf-8"))
    expected = {"class_mapping.csv": rec["class_mapping.csv"]["sha256"],
                rec["train"]["csv"]: rec["train"]["csv_sha256"], rec["val"]["csv"]: rec["val"]["csv_sha256"]}
    for name, sha in expected.items():
        if sha256_file(folder / name) != sha:
            raise ValueError(f"{folder / name} does not match the hash recorded at training time")
    return folder


def evaluate(run_dir: Path, split: str = "test", device_name: str = "auto") -> dict:
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    manifests = recorded_manifests(run_dir)
    class_names = load_class_names(manifests)
    model, ckpt = load_finetuned(run_dir / "best.pt", num_classes=len(class_names))
    if ckpt["class_names"] != class_names:
        raise ValueError("checkpoint class order differs from class_mapping.csv")
    device = resolve_device(device_name)
    model.to(device)

    records = (test_manifest(class_names) if split == "test"
               else read_manifest(cfg["data"]["val_csv"], class_names, manifests))
    if split == "validation":
        if cfg.get("smoke"):  # a smoke run validated on the first images of each class only
            records = first_per_class(records, SMOKE_PER_CLASS)
        rec = json.loads((run_dir / "manifests.json").read_text(encoding="utf-8"))["val"]
        if manifest_hash(records) != rec["used_content_hash"]:
            raise ValueError("validation images differ from those used during training")
    gpu = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
    print(f"Evaluating {run_dir / 'best.pt'}\n"
          f"  checkpoint : epoch {ckpt['epoch']}, val acc at training {ckpt['val_acc']:.4f}\n"
          f"  split      : {split}, {len(records)} images (read from disk)\n"
          f"  device     : {device}{gpu}", flush=True)
    loader = DataLoader(SARDataset(records, build_transform(cfg["data"], train=False), cache_in_memory=False),
                        batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])
    out = predict(model, loader, device, log_every=cfg["train"].get("log_every_batches", 25), tag=split)
    labels, preds, probs = out["labels"], out["preds"], out["probs"]

    n = len(class_names)
    confusion = torch.zeros(n, n, dtype=torch.long)
    for t, p in zip(labels.tolist(), preds.tolist()):
        confusion[t, p] += 1
    per_class = (confusion.diag().float() / confusion.sum(1).clamp(min=1).float()).tolist()

    eval_dir = run_dir / f"eval_{split}"
    eval_dir.mkdir(exist_ok=True)
    metrics = {
        "split": split,
        "num_images": len(records),
        "overall_accuracy": out["acc"],
        "mean_per_class_accuracy": sum(per_class) / n,
        "loss": out["loss"],
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_val_acc_at_training": ckpt["val_acc"],
        "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        "manifest_content_hash": manifest_hash(records),
        "preprocessing": "deterministic: resize -> gray to 3 channels -> [0,1] -> normalize (from run config)",
        "evaluated": datetime.now().isoformat(timespec="seconds"),
        "environment": environment_info(device),
    }
    save_json(metrics, eval_dir / "metrics.json")
    with open(eval_dir / "per_class_accuracy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "class_name", "num_images", "accuracy"])
        for i, c in enumerate(class_names):
            w.writerow([i, c, int(confusion[i].sum()), f"{per_class[i]:.5f}"])
    with open(eval_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted"] + class_names)
        for i, c in enumerate(class_names):
            w.writerow([c] + confusion[i].tolist())
    with open(eval_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relative_path", "true_label", "true_class", "pred_label", "pred_class", "confidence", "correct"])
        for (path, _), t, p, pr in zip(records, labels.tolist(), preds.tolist(), probs.max(1).values.tolist()):
            w.writerow([path, t, class_names[t], p, class_names[p], f"{pr:.5f}", int(t == p)])

    print(f"{split}: {len(records)} images  overall accuracy {out['acc']:.4f}  "
          f"mean per-class accuracy {metrics['mean_per_class_accuracy']:.4f}  (checkpoint epoch {ckpt['epoch']})")
    plot_eval(eval_dir)
    print(f"Outputs in {eval_dir}:\n"
          f"  metrics.json            overall and mean per-class accuracy, loss, checkpoint info\n"
          f"  per_class_accuracy.csv  accuracy for each of the {n} classes\n"
          f"  confusion_matrix.csv    rows = true class, columns = predicted class\n"
          f"  predictions.csv         one row per image: true, predicted, confidence\n"
          f"  charts/                 per_class_accuracy.pdf, confusion_matrix.pdf, top_mistakes.pdf,\n"
          f"                          confidence.pdf, calibration.pdf (one chart per file)")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained run's best.pt")
    parser.add_argument("--run", type=Path, required=True, help="run folder containing best.pt and config.json")
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate(args.run, args.split, args.device)


if __name__ == "__main__":
    main()
