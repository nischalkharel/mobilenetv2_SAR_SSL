"""THE fine-tuning and scoring protocol, shared by every method (DESIGN.md section 3).

    python -m shared.finetune --experiment-dir <folder> --run-name <name> --budget 1pct|3pct|10pct \
        [--backbone <exported .pth> | imagenet] [--seed 0] [--epochs N] [--lr X] [--smoke N] [--no-final-eval]

Run folder: <experiment>/outputs/finetune/<timestamp>_<run-name>_<budget>_split<s>_seed<k>/ with config.json,
run_info.json, manifests/, history.csv, charts/, best.pt, last.pt, eval_*/ (metrics.json, per_class.csv,
pr_curves.png, detections.npz). --epochs / --lr / --smoke exist for smoke tests and the lr check only; a run
that uses them is marked "protocol_deviation" in its record so compare.py can exclude it.
"""
import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import manifests, paths
from .backbone import imagenet_weights_record, load_backbone_file
from .coco_data import DetectionDataset, build_loader
from .detector import build_detector, count_parameters, head_state_fingerprint, load_finetune_config
from .evaluate import evaluate_subset, run_detector, sources_from_metadata, summary_line
from .plots import finetune_curves
from .run_record import RunRecord, make_run_dir, sha256_of_file, write_json

LOSS_KEYS = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def param_groups(model, weight_decay):
    """No weight decay on biases and 1-D tensors (BatchNorm), the usual AdamW rule."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step, total_steps, warmup_steps, base_lr, min_factor):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * t)))


def train_one_epoch(model, loader, optimizer, device, step, total_steps, warmup_steps, base_lr, min_factor, clip):
    model.train()
    sums = {k: 0.0 for k in ["loss"] + LOSS_KEYS}
    n = 0
    for images, targets in loader:
        lr = lr_at(step, total_steps, warmup_steps, base_lr, min_factor)
        for g in optimizer.param_groups:
            g["lr"] = lr
        images = [im.to(device, non_blocking=True) for im in images]
        targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]
        losses = model(images, targets)
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: { {k: float(v) for k, v in losses.items()} }")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        sums["loss"] += float(loss.detach())
        for k in LOSS_KEYS:
            sums[k] += float(losses[k].detach())
        n += 1
        step += 1
    return {k: v / max(n, 1) for k, v in sums.items()}, step


@torch.no_grad()
def selection_score(model, dataset, det_cfg, batch_size, num_workers, device):
    from .coco_data import coco_ground_truth
    from .evaluate import coco_metrics
    results = run_detector(model, dataset, batch_size, num_workers, device, log_every=0)
    m = coco_metrics(coco_ground_truth(dataset.official_split, dataset.file_names), results)
    return m["map"], m["map50"]


def save_checkpoint(path, model, epoch, score, cfg, backbone_record):
    torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_select_map": score,
                "config": cfg, "backbone": backbone_record, "class_names": paths.CLASS_NAMES}, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-dir", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--budget", required=True, choices=list(manifests.LABELED_CSV))
    ap.add_argument("--backbone", default="imagenet", help="exported backbone .pth, or 'imagenet'")
    ap.add_argument("--seed", type=int, default=None, help="fine-tune seed (default: shared config, 0)")
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--lr", type=float, default=None, help="override (lr check only)")
    ap.add_argument("--smoke", type=int, default=0, help="use only N train images and N eval images, 1 epoch, CPU ok")
    ap.add_argument("--no-final-eval", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    cfg = load_finetune_config()
    det_cfg, tr = cfg["detector"], cfg["train"]
    seed = tr["seed"] if args.seed is None else args.seed
    epochs = tr["epochs_by_budget"][args.budget] if args.epochs is None else args.epochs
    lr = tr["lr"] if args.lr is None else args.lr
    deviation = []
    if args.epochs is not None:
        deviation.append(f"epochs={args.epochs}")
    if args.lr is not None:
        deviation.append(f"lr={args.lr}")
    if args.smoke:
        deviation.append(f"smoke={args.smoke}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    num_workers = 0 if args.smoke else tr["num_workers"]

    # ---- data (audited) --------------------------------------------------------------------------------------
    audit = manifests.quick_audit()
    train_rows = manifests.read(manifests.labeled_csv(args.budget))
    val_select_rows = manifests.read(paths.VAL_SELECT_CSV)
    if args.smoke:
        train_rows, val_select_rows = train_rows[:args.smoke], val_select_rows[:args.smoke]
    set_seed(seed)
    train_set = DetectionDataset("train", [r["file_name"] for r in train_rows], train=True, hflip_p=tr["hflip_p"], seed=seed)
    select_set = DetectionDataset("val", [r["file_name"] for r in val_select_rows], train=False)

    # ---- model -----------------------------------------------------------------------------------------------
    if args.backbone == "imagenet":
        backbone_state, backbone_record = None, {"kind": "imagenet", **imagenet_weights_record()}
    else:
        backbone_state, backbone_record = load_backbone_file(args.backbone)
        backbone_record = {"kind": "ssl", **backbone_record}
    model = build_detector(det_cfg, backbone_state, head_seed=seed).to(device)
    head_fp = head_state_fingerprint(model)

    # ---- run folder and record --------------------------------------------------------------------------------
    exp_dir = Path(args.experiment_dir).resolve()
    out_root = paths.experiment_output_dir(exp_dir) / ("smoke" if args.smoke else "finetune")
    run_dir = make_run_dir(out_root, f"{args.run_name}_{args.budget}{paths.RUN_TAG}_seed{seed}")
    (run_dir / "manifests").mkdir()
    for p in [manifests.labeled_csv(args.budget), paths.VAL_SELECT_CSV]:
        shutil.copy(p, run_dir / "manifests" / p.name)
    full_cfg = {"shared_finetune_config": cfg, "budget": args.budget, "epochs": epochs, "lr": lr, "seed": seed,
                "backbone_arg": args.backbone, "run_name": args.run_name, "experiment_dir": str(exp_dir)}
    write_json(run_dir / "config.json", full_cfg)
    record = RunRecord(run_dir, "finetune", full_cfg, extra={
        "audit": audit, "backbone": backbone_record, "head_fingerprint_sha256": head_fp,
        "parameters": count_parameters(model), "protocol_deviation": deviation or None,
        "precision": {"cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
        "pycocotools": __import__("pycocotools").__version__ if hasattr(__import__("pycocotools"), "__version__") else "unknown",
        "n_train_images": len(train_set), "n_val_select": len(select_set)}).start()
    print(f"run folder {run_dir}\n{len(train_set)} train images, {len(select_set)} val_select images, {epochs} epochs, "
          f"lr {lr}, seed {seed}, device {device}, backbone {backbone_record['kind']}", flush=True)

    # ---- training --------------------------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(param_groups(model, tr["weight_decay"]), lr=lr)
    steps_per_epoch = math.ceil(len(train_set) / tr["batch_size"])
    total_steps = steps_per_epoch * epochs
    warmup_steps = min(total_steps, steps_per_epoch * tr["warmup_epochs"])
    history, best, step = [], {"map": -1.0, "epoch": None}, 0
    t_start = time.time()
    try:
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_set.set_epoch(epoch)
            set_seed(seed + epoch)
            loader = build_loader(train_set, tr["batch_size"], shuffle=True, num_workers=num_workers, seed=seed + epoch)
            losses, step = train_one_epoch(model, loader, optimizer, device, step, total_steps, warmup_steps, lr,
                                           tr["min_lr_factor"], tr["grad_clip_norm"])
            row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **losses, "val_select_map": "", "val_select_map50": ""}
            if epoch % tr["eval_every"] == 0 or epoch == epochs:
                m, m50 = selection_score(model, select_set, det_cfg, tr["batch_size"], num_workers, device)
                row.update(val_select_map=m, val_select_map50=m50)
                if m > best["map"]:
                    best = {"map": m, "epoch": epoch}
                    save_checkpoint(run_dir / "best.pt", model, epoch, m, full_cfg, backbone_record)
            row["seconds"] = round(time.time() - t0, 1)
            history.append(row)
            with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                w.writeheader()
                w.writerows(history)
            star = "  *best" if best["epoch"] == epoch else ""
            sel = f"  val_select mAP {row['val_select_map']:.4f}" if row["val_select_map"] != "" else ""
            print(f"epoch {epoch:3d}/{epochs}  loss {losses['loss']:.3f}  lr {row['lr']:.2e}{sel}{star}  ({row['seconds']} s)", flush=True)
        save_checkpoint(run_dir / "last.pt", model, epochs, history[-1]["val_select_map"], full_cfg, backbone_record)
        (run_dir / "charts").mkdir(exist_ok=True)
        finetune_curves(run_dir / "history.csv", run_dir / "charts" / "training_curves.png")
        record.update(train_minutes=round((time.time() - t_start) / 60, 2), best_epoch=best["epoch"], best_val_select_map=best["map"],
                      checkpoints={"best.pt": sha256_of_file(run_dir / "best.pt"), "last.pt": sha256_of_file(run_dir / "last.pt")})

        # ---- final scoring: best and last checkpoints on full val, test, test_clean ------------------------------
        final = {}
        if not args.no_final_eval:
            subsets = manifests.eval_subsets()
            sources_of = sources_from_metadata()
            plan = [("best", "val", "val", subsets["val"]), ("best", "test", "test", subsets["test"]),
                    ("best", "test_clean", "test", subsets["test_clean"]), ("last", "test", "test", subsets["test"])]
            for ckpt, name, split, names in plan:
                if args.smoke:
                    names = names[:args.smoke]
                state = torch.load(run_dir / f"{ckpt}.pt", map_location="cpu", weights_only=False)["model_state_dict"]
                model.load_state_dict(state)
                m = evaluate_subset(model, split, names, run_dir / f"eval_{ckpt}_{name}", det_cfg, tr["batch_size"],
                                    num_workers, device, f"{ckpt}_{name}", sources_of)
                final[f"{ckpt}_{name}"] = {k: m[k] for k in ["map", "map50", "map75", "map_small", "map_medium", "map_large", "n_images"]}
                print(f"  {ckpt} on {name}: {summary_line(m)}", flush=True)
        record.finish(final_metrics=final)
        print(f"done: best val_select mAP {best['map']:.4f} at epoch {best['epoch']}; saved in {run_dir}")
        return run_dir
    except BaseException as e:
        record.fail(e)
        raise


if __name__ == "__main__":
    main()
