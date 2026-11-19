"""Fine-tune one backbone on SARDet-100K chips for one label budget and split, then score the two test sets.

    python SARDet_chips/finetune_chips.py --method imagenet --budget 1pct --split 1
    python SARDet_chips/finetune_chips.py --method dino --budget 3pct --split 2            # backbone found automatically
    python SARDet_chips/finetune_chips.py --method regiondino_o --budget 10pct --split 3 --backbone <file>
    python SARDet_chips/finetune_chips.py --method dino --budget 1pct --split 1 --smoke --device cpu   # pipeline check

Methods: imagenet (no SSL), dino, lejepa, spark, regiondino_so, regiondino_so_random, regiondino_o, regiondino_o_random.
The SSL backbone of method m for split s is the newest completed scene SSL run of split s (SARDet_to_ATR/find_backbone.py).
Model, training loop, transforms, run records and charts are the ATRNet fine-tune code (Experiments/mobileNetv2_DINO);
only the data (chips from one array file) and the head size (6 classes) differ. DESIGN.md section 4.
Run folder: SARDet_chips/outputs/finetune/<stamp>_S1_<method>_mobilenetv2_<budget>_split<s>_seed0/
    config.json, manifests.json, run_info.json, history.csv, best.pt, charts/, eval_test/, eval_test_clean/
"""
import argparse
import csv
import json
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chip_paths as cp  # noqa: E402  (also puts SARDet/ and Experiments/mobileNetv2_DINO/ on sys.path)
from chip_data import LABELED_CSV, ChipDataset, class_counts, content_hash, load_chip_rows, records  # noqa: E402
from data import build_transform  # noqa: E402  Experiments/mobileNetv2_DINO/data.py (128-px chips, same augmentation as ATRNet)
from engine import predict, train_one_epoch  # noqa: E402
from model import build_model  # noqa: E402
from plots import plot_history  # noqa: E402
from utils import environment_info, git_info, resolve_device, save_json, set_seed, sha256_file  # noqa: E402

sys.path.insert(0, str(cp.REPO_ROOT / "SARDet_to_ATR"))
import find_backbone  # noqa: E402

METHODS = ["imagenet", "dino", "lejepa", "spark", "regiondino_so", "regiondino_so_random", "regiondino_o", "regiondino_o_random"]
SMOKE_PER_CLASS, SMOKE_EPOCHS = 12, 2


def per_class_metrics(labels, preds, n):
    confusion = torch.zeros(n, n, dtype=torch.long)
    for t, p in zip(labels.tolist(), preds.tolist()):
        confusion[t, p] += 1
    per_class = (confusion.diag().float() / confusion.sum(1).clamp(min=1).float()).tolist()
    present = [i for i in range(n) if confusion[i].sum() > 0]
    return confusion, per_class, sum(per_class[i] for i in present) / max(1, len(present))


def score(model, sel, name, run_dir, data_cfg, device, batch_size, arr_root):
    """Score one test set once with the selected checkpoint; write eval_<name>/ like evaluate.py does, plus per source."""
    ds = ChipDataset(sel, build_transform(data_cfg, train=False), root=arr_root)
    out = predict(model, DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), device, log_every=100, tag=name)
    n = len(cp.CLASS_NAMES)
    confusion, per_class, mean_pc = per_class_metrics(out["labels"], out["preds"], n)
    eval_dir = run_dir / f"eval_{name}"
    eval_dir.mkdir(exist_ok=True)
    by_source = {}
    for r, ok in zip(sel, (out["preds"] == out["labels"]).tolist()):
        d = by_source.setdefault(r["source"], Counter())
        d["n"] += 1; d["correct"] += ok
    metrics = {"split": name, "num_chips": len(sel), "num_images": len({r["file_name"] for r in sel}),
               "overall_accuracy": out["acc"], "mean_per_class_accuracy": mean_pc, "loss": out["loss"],
               "per_class_accuracy": {c: per_class[i] for i, c in enumerate(cp.CLASS_NAMES)},
               "per_class_count": {c: int(confusion[i].sum()) for i, c in enumerate(cp.CLASS_NAMES)},
               "per_source_accuracy": {s: d["correct"] / d["n"] for s, d in sorted(by_source.items())},
               "per_source_count": {s: d["n"] for s, d in sorted(by_source.items())},
               "chips_content_hash": content_hash(sel)}
    save_json(metrics, eval_dir / "metrics.json")
    with open(eval_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["true \\ predicted"] + cp.CLASS_NAMES)
        for i, c in enumerate(cp.CLASS_NAMES):
            w.writerow([c] + confusion[i].tolist())
    with open(eval_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["idx", "file_name", "source", "true", "predicted", "correct"])
        for r, p in zip(sel, out["preds"].tolist()):
            w.writerow([r["idx"], r["file_name"], r["source"], r["class_name"], cp.CLASS_NAMES[p], int(p == r["label"])])
    print(f"  {name}: {len(sel)} chips  overall acc {out['acc']:.4f}  mean per-class acc {mean_pc:.4f}  "
          + "  ".join(f"{c} {per_class[i]:.2f}" for i, c in enumerate(cp.CLASS_NAMES)), flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--budget", required=True, choices=["1pct", "3pct", "10pct"])
    ap.add_argument("--split", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--backbone", type=Path, help="SSL backbone file; default = newest completed scene SSL run of this method and split")
    ap.add_argument("--smoke", action="store_true", help=f"{SMOKE_PER_CLASS} chips per class, {SMOKE_EPOCHS} epochs; pipeline check only")
    ap.add_argument("--device", help='override train.device: "cuda", "cpu" or "auto"')
    ap.add_argument("--chips-root", type=Path, help=f"folder with chips_u8.npy and chips.csv (default {cp.CHIPS_ROOT})")
    a = ap.parse_args()
    cfg = json.loads((cp.HERE / "config_finetune.json").read_text(encoding="utf-8"))
    cfg["run_name"] = cfg["run_name"].replace("METHOD", a.method)
    if a.method == "imagenet":
        cfg["init"] = {"type": "imagenet", "backbone_checkpoint": None}
    else:
        bb = a.backbone or find_backbone.find(a.method, a.split)
        if not bb.exists():
            raise SystemExit(f"backbone file not found: {bb}")
        cfg["init"] = {"type": "backbone_checkpoint", "backbone_checkpoint": str(bb.resolve())}
    cfg.update(method=a.method, budget=a.budget, split=a.split, chip_rule={"chip_size": cp.CHIP_SIZE, "window_floor": cp.WINDOW_FLOOR, "window_margin": cp.WINDOW_MARGIN})
    data_cfg, train_cfg = cfg["data"], cfg["train"]
    data_cfg["train_csv"], data_cfg["val_csv"] = f"split{a.split}/{LABELED_CSV[a.budget]} (chips)", "val_select.csv (chips)"   # also read by the shared chart code
    if a.device:
        train_cfg["device"] = a.device
    if a.smoke:
        train_cfg["epochs"] = SMOKE_EPOCHS
    set_seed(train_cfg["seed"])
    device = resolve_device(train_cfg["device"])
    root = a.chips_root or cp.CHIPS_ROOT

    # ---- chips -------------------------------------------------------------------------------------
    rows, chips_meta = load_chip_rows(root)
    sets = {k: records(k, rows, a.split, a.budget, root) for k in ("train", "select", "test", "test_clean")}
    if a.smoke:
        for k in sets:
            keep, seen = [], Counter()
            for r in sets[k]["rows"]:
                if seen[r["label"]] < SMOKE_PER_CLASS:
                    keep.append(r); seen[r["label"]] += 1
            sets[k]["rows"] = keep
    train_imgs = {r["file_name"] for r in sets["train"]["rows"]}
    for k in ("select", "test", "test_clean"):
        if train_imgs & {r["file_name"] for r in sets[k]["rows"]}:
            raise ValueError(f"image overlap between train and {k}")
        if any(r["official_split"] == "train" for r in sets[k]["rows"]):
            raise ValueError(f"{k} contains train-split chips")
    for k in ("select", "test", "test_clean"):
        if not sets[k]["rows"]:
            raise ValueError(f"{k} is empty")

    # ---- run folder and records --------------------------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = cp.FINETUNE_OUTPUT_DIR / f"{stamp}_{cfg['run_name']}_{a.budget}_split{a.split}_seed{train_cfg['seed']}{'_SMOKE' if a.smoke else ''}"
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json({**cfg, "smoke": a.smoke}, run_dir / "config.json")
    save_json({"chips": {"root": str(root), "chips_csv_sha256": chips_meta["chips_csv_sha256"], "chips_npy_sha256": chips_meta["chips_npy_sha256"],
                         "n_chips": chips_meta["n_chips"], "rule": cfg["chip_rule"]},
               **{k: {"source": v["source"], "csv_sha256": v["csv_sha256"], "images": v["images"], "chips": len(v["rows"]),
                      "per_class": class_counts(v["rows"]), "used_content_hash": content_hash(v["rows"])} for k, v in sets.items()},
               "note": "a chip belongs to the official split of its image; image lists are the detection suite's manifests (SARDet/data)"},
              run_dir / "manifests.json")
    init_rec = {"type": cfg["init"]["type"]}
    if cfg["init"]["backbone_checkpoint"]:
        init_rec.update(file=cfg["init"]["backbone_checkpoint"], sha256=sha256_file(Path(cfg["init"]["backbone_checkpoint"])))
        ssl_info = Path(cfg["init"]["backbone_checkpoint"]).parent / "run_info.json"
        if ssl_info.exists():
            init_rec["ssl_run_dir"] = str(ssl_info.parent)
    run_info = {"status": "running", "started": datetime.now().isoformat(timespec="seconds"), "kind": "finetune_chips",
                "method": a.method, "budget": a.budget, "split": a.split, "smoke": a.smoke,
                "environment": environment_info(device), "git": git_info(cp.REPO_ROOT),
                "model_selection": train_cfg["selection"], "initialization": init_rec}
    save_json(run_info, run_dir / "run_info.json")
    print(f"{'SMOKE RUN (pipeline check only)' if a.smoke else 'Training run'}: {run_dir.name}\n"
          f"  device  : {device}{' (' + run_info['environment'].get('gpu', '') + ')' if device.type == 'cuda' else ''}\n"
          f"  init    : {init_rec.get('file', 'ImageNet weights')}\n"
          + "".join(f"  {k:10}: {len(v['rows']):6d} chips from {v['images']} images  {class_counts(v['rows'])}\n" for k, v in sets.items())
          + f"  settings: {train_cfg['epochs']} epochs, batch {train_cfg['batch_size']}, {train_cfg['optimizer']} lr {train_cfg['lr']}, "
          f"{train_cfg['scheduler']}, seed {train_cfg['seed']}; selection by mean per-class accuracy on 'select'", flush=True)

    try:
        t_load = time.time()
        train_ds = ChipDataset(sets["train"]["rows"], build_transform(data_cfg, train=True), root=root)
        select_ds = ChipDataset(sets["select"]["rows"], build_transform(data_cfg, train=False), root=root)
        print(f"  chips in memory in {time.time() - t_load:.1f} s", flush=True)
        bs = train_cfg["batch_size"]
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False, num_workers=data_cfg["num_workers"],
                                  generator=torch.Generator().manual_seed(train_cfg["seed"]), pin_memory=device.type == "cuda")
        select_loader = DataLoader(select_ds, batch_size=bs, shuffle=False, num_workers=0)

        set_seed(train_cfg["seed"])
        model = build_model(cfg["init"], num_classes=len(cp.CLASS_NAMES)).to(device)
        decay, no_decay = [], []
        for n_, p in model.named_parameters():
            (no_decay if p.ndim <= 1 else decay).append(p)
        optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": train_cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}], lr=train_cfg["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"]) if train_cfg["scheduler"] == "cosine" else None
        run_info["num_parameters"] = sum(p.numel() for p in model.parameters())

        hist = run_dir / "history.csv"
        with open(hist, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "val_mean_class_acc", "seconds"])
        best, best_epoch, t_start = -1.0, None, time.time()
        for epoch in range(1, train_cfg["epochs"] + 1):
            t0 = time.time(); lr = optimizer.param_groups[0]["lr"]
            tr = train_one_epoch(model, train_loader, optimizer, device, train_cfg.get("log_every_batches", 0), tag="train")
            va = predict(model, select_loader, device, 0, tag="select")
            _, _, va_mean = per_class_metrics(va["labels"], va["preds"], len(cp.CLASS_NAMES))
            if scheduler:
                scheduler.step()
            secs = time.time() - t0
            with open(hist, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{lr:.6g}", f"{tr['loss']:.5f}", f"{tr['acc']:.5f}", f"{va['loss']:.5f}", f"{va['acc']:.5f}", f"{va_mean:.5f}", f"{secs:.1f}"])
            improved = va_mean > best
            if improved:
                best, best_epoch = va_mean, epoch
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_acc": va["acc"], "val_mean_class_acc": va_mean,
                            "class_names": cp.CLASS_NAMES, "config": cfg}, run_dir / "best.pt")
            elapsed = time.time() - t_start
            print(f"  epoch {epoch:3d}/{train_cfg['epochs']}  train loss {tr['loss']:.4f} acc {tr['acc']:.4f}  select acc {va['acc']:.4f} "
                  f"mean-class {va_mean:.4f}  {secs:.0f}s{'  *best' if improved else ''}   elapsed {elapsed / 60:.1f} min, "
                  f"about {elapsed / epoch * (train_cfg['epochs'] - epoch) / 60:.1f} min left", flush=True)
            try:
                plot_history(run_dir)
            except Exception as e:
                print(f"  WARNING: charts not drawn: {e}", flush=True)

        run_info.update(train_minutes=round((time.time() - t_start) / 60, 2), best_epoch=best_epoch, best_val_mean_class_acc=best)
        ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"]); model.to(device)
        print(f"\nScoring the selected epoch ({best_epoch}) once on the test sets:", flush=True)
        run_info["test"] = {k: {kk: m[kk] for kk in ("num_chips", "overall_accuracy", "mean_per_class_accuracy")}
                            for k, m in ((k, score(model, sets[k]["rows"], k, run_dir, data_cfg, device, bs, root)) for k in ("test", "test_clean"))}
        run_info.update(status="completed", finished=datetime.now().isoformat(timespec="seconds"))
        save_json(run_info, run_dir / "run_info.json")
        print(f"\nDone in {run_info['train_minutes']} min. Outputs in {run_dir}")
    except BaseException:
        run_info.update(status="failed", error=traceback.format_exc(limit=3))
        save_json(run_info, run_dir / "run_info.json")
        raise


if __name__ == "__main__":
    main()
