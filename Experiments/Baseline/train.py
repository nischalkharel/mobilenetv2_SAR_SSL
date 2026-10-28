"""Supervised fine-tuning of MobileNetV2 on a labeled SOC-40 subset.

Run from the repository root:
    python Experiments/Baseline/train.py                                 # uses Experiments/Baseline/config.json
    python Experiments/Baseline/train.py --config path/to/other.json     # e.g. a DINO-initialized backbone
    python Experiments/Baseline/train.py --device cpu                    # override config train.device (CPU timing runs)

The whole model (backbone + new 40-class head) is trained on data.train_csv. After every
epoch the model is evaluated on data.val_csv, and the epoch with the highest validation
accuracy is saved as best.pt. The test set is never touched here.
"""
import argparse
import csv
import json
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, for "import paths"
import paths  # noqa: E402
from data import SARDataset, build_transform, load_class_names, manifest_hash, read_manifest  # noqa: E402
from engine import predict, train_one_epoch  # noqa: E402
from model import build_model  # noqa: E402
from plots import plot_history  # noqa: E402
from utils import environment_info, git_info, resolve_device, save_json, set_seed, sha256_file  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.json"
SMOKE_PER_CLASS = 8  # images per class in train and validation for a smoke run
SMOKE_EPOCHS = 2


def first_per_class(records, k):
    seen, out = {}, []
    for path, label in records:
        if seen.get(label, 0) < k:
            out.append((path, label))
            seen[label] = seen.get(label, 0) + 1
    return out


def make_optimizer(model, train_cfg):
    if train_cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    if train_cfg["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=train_cfg["lr"], momentum=0.9,
                               weight_decay=train_cfg["weight_decay"])
    raise ValueError(f"unknown optimizer {train_cfg['optimizer']}")


def run(cfg: dict, smoke: bool = False, output_root: Path = paths.BASELINE_OUTPUT_DIR) -> Path:
    """Train one model and return its run folder."""
    data_cfg, train_cfg = cfg["data"], cfg["train"]
    if smoke:
        train_cfg = {**train_cfg, "epochs": SMOKE_EPOCHS}
        cfg = {**cfg, "train": train_cfg}
    set_seed(train_cfg["seed"])
    device = resolve_device(train_cfg["device"])

    # ---- Data and leakage guards -------------------------------------------------
    class_names = load_class_names()
    train_records = read_manifest(data_cfg["train_csv"], class_names)
    val_records = read_manifest(data_cfg["val_csv"], class_names)
    if smoke:
        train_records = first_per_class(train_records, SMOKE_PER_CLASS)
        val_records = first_per_class(val_records, SMOKE_PER_CLASS)
    train_paths, val_paths = {p for p, _ in train_records}, {p for p, _ in val_records}
    if train_paths & val_paths:
        raise ValueError(f"{len(train_paths & val_paths)} images are in both training and validation")
    if not all(p.startswith("train/") for p in train_paths | val_paths):
        raise ValueError("training/validation manifests must only contain official train/ images")

    # ---- Run folder and records --------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_root) / f"{stamp}_{cfg['run_name']}{paths.RUN_TAG}_seed{train_cfg['seed']}{'_SMOKE' if smoke else ''}"
    run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier run
    save_json({**cfg, "smoke": smoke}, run_dir / "config.json")
    shutil.copy(paths.SPLITS_DIR / "class_mapping.csv", run_dir / "class_mapping.csv")
    save_json({
        "class_mapping.csv": {"sha256": sha256_file(paths.SPLITS_DIR / "class_mapping.csv")},
        "train": {"csv": data_cfg["train_csv"], "csv_sha256": sha256_file(paths.SPLITS_DIR / data_cfg["train_csv"]),
                  "used_content_hash": manifest_hash(train_records), "num_images": len(train_records)},
        "val": {"csv": data_cfg["val_csv"], "csv_sha256": sha256_file(paths.SPLITS_DIR / data_cfg["val_csv"]),
                "used_content_hash": manifest_hash(val_records), "num_images": len(val_records)},
        "note": "used_content_hash covers the images actually used (a smoke run uses a subset)",
    }, run_dir / "manifests.json")
    run_info = {"status": "running", "started": datetime.now().isoformat(timespec="seconds"),
                "environment": environment_info(device), "git": git_info(paths.REPO_ROOT),
                "model_selection": "epoch with highest validation accuracy (earliest on ties)"}
    save_json(run_info, run_dir / "run_info.json")
    gpu = f" ({run_info['environment']['gpu']})" if device.type == "cuda" else ""
    backbone_file = cfg["init"].get("backbone_checkpoint")
    print(f"{'SMOKE RUN (pipeline check only)' if smoke else 'Training run'}: {cfg['run_name']}\n"
          f"  output folder : {run_dir}\n"
          f"  device        : {device}{gpu}\n"
          f"  init          : {cfg['init']['type']}{'  ' + str(backbone_file) if backbone_file else ''}\n"
          f"  train images  : {len(train_records)} from {data_cfg['train_csv']}\n"
          f"  val images    : {len(val_records)} from {data_cfg['val_csv']} (picks best.pt)\n"
          f"  settings      : {train_cfg['epochs']} epochs, batch {train_cfg['batch_size']}, "
          f"{train_cfg['optimizer']} lr {train_cfg['lr']}, {train_cfg['scheduler']} schedule, seed {train_cfg['seed']}",
          flush=True)

    try:
        workers = data_cfg["num_workers"]
        loader_args = dict(batch_size=train_cfg["batch_size"], num_workers=workers,
                           persistent_workers=workers > 0, pin_memory=device.type == "cuda")
        t_load = time.time()
        if data_cfg["cache_in_memory"]:
            print("Reading all train and val images into memory (once per run)...", flush=True)
        train_ds = SARDataset(train_records, build_transform(data_cfg, train=True), data_cfg["cache_in_memory"])
        val_ds = SARDataset(val_records, build_transform(data_cfg, train=False), data_cfg["cache_in_memory"])
        print(f"  done in {time.time() - t_load:.0f}s", flush=True)
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=False,
                                  generator=torch.Generator().manual_seed(train_cfg["seed"]), **loader_args)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

        model = build_model(cfg["init"], num_classes=len(class_names)).to(device)
        optimizer = make_optimizer(model, train_cfg)
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
                     if train_cfg["scheduler"] == "cosine" else None)
        run_info["num_parameters"] = sum(p.numel() for p in model.parameters())

        history_path = run_dir / "history.csv"
        with open(history_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])
        best_acc, best_epoch, t_start = -1.0, None, time.time()
        log_every = train_cfg.get("log_every_batches", 0)  # progress lines within an epoch (0 = off)

        for epoch in range(1, train_cfg["epochs"] + 1):
            t0 = time.time()
            lr = optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch}/{train_cfg['epochs']}  (lr {lr:.2e})", flush=True)
            tr = train_one_epoch(model, train_loader, optimizer, device, log_every, tag="train")
            va = predict(model, val_loader, device, log_every, tag="val  ")
            if scheduler is not None:
                scheduler.step()
            secs = time.time() - t0
            with open(history_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{lr:.6g}", f"{tr['loss']:.5f}", f"{tr['acc']:.5f}",
                                        f"{va['loss']:.5f}", f"{va['acc']:.5f}", f"{secs:.1f}"])
            improved = va["acc"] > best_acc
            if improved:
                best_acc, best_epoch = va["acc"], epoch
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                            "val_acc": va["acc"], "val_loss": va["loss"],
                            "class_names": class_names, "config": cfg}, run_dir / "best.pt")
            elapsed = time.time() - t_start
            remaining = elapsed / epoch * (train_cfg["epochs"] - epoch)
            print(f"  epoch {epoch:3d}/{train_cfg['epochs']} summary  "
                  f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f}  "
                  f"val loss {va['loss']:.4f} acc {va['acc']:.4f}  {secs:.0f}s"
                  f"{'  *new best, saved best.pt' if improved else ''}\n"
                  f"  best so far: val acc {best_acc:.4f} (epoch {best_epoch})  "
                  f"elapsed {elapsed / 60:.1f} min, about {remaining / 60:.1f} min left", flush=True)
            try:  # redrawn every epoch so the charts can be opened while training runs
                plot_history(run_dir)
            except Exception as e:  # a chart problem should not stop training; history.csv has the numbers
                print(f"  WARNING: could not draw the training charts: {e}", flush=True)

        run_info.update(status="completed", finished=datetime.now().isoformat(timespec="seconds"),
                        train_minutes=round((time.time() - t_start) / 60, 2),
                        best_epoch=best_epoch, best_val_acc=best_acc)
        save_json(run_info, run_dir / "run_info.json")
        print(f"\nDone in {run_info['train_minutes']} min. Best validation accuracy {best_acc:.4f} at epoch {best_epoch}.\n"
              f"Outputs in {run_dir}:\n"
              f"  config.json        settings used\n"
              f"  class_mapping.csv  label number -> class name\n"
              f"  manifests.json     hashes of the train/val image lists used\n"
              f"  run_info.json      status, device, versions, timing, best epoch\n"
              f"  history.csv        per-epoch loss and accuracy (train and val)\n"
              f"  best.pt            model weights from the best validation epoch\n"
              f"  charts/            loss.pdf, accuracy.pdf, overfitting_gap.pdf, learning_rate.pdf (one chart per file)\n"
              f"Next: python Experiments/Baseline/evaluate.py --run \"{run_dir}\"", flush=True)
        return run_dir
    except BaseException:
        run_info.update(status="failed", error=traceback.format_exc(limit=3))
        save_json(run_info, run_dir / "run_info.json")
        raise


def main():
    parser = argparse.ArgumentParser(description="Supervised fine-tuning on SOC-40")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true",
                        help=f"{SMOKE_PER_CLASS} images per class, {SMOKE_EPOCHS} epochs; pipeline check only")
    parser.add_argument("--device", help='override train.device in the config: "cuda", "cpu" or "auto"')
    parser.add_argument("--train-csv", help="labeled CSV in preprocessing_dataset/ to train on instead of "
                        "data.train_csv, e.g. labeled_set_1pct.csv (label-budget runs; added to the run name)")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.device:
        cfg["train"]["device"] = args.device  # saved into the run's config.json, so the run records it
    if args.train_csv:
        cfg["data"]["train_csv"] = args.train_csv  # saved into the run's config.json and manifests
        cfg["run_name"] += "_" + Path(args.train_csv).stem
    run(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
