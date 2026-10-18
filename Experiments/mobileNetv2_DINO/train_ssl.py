"""DINO self-supervised training of MobileNetV2 on the unlabeled SAR pool (stage 1 of route A1).

Run from the repository root:
    python Experiments/mobileNetv2_DINO/train_ssl.py                          # full run, config_ssl.json
    python Experiments/mobileNetv2_DINO/train_ssl.py --stop-after-epoch 2     # pilot: first 2 epochs of the same schedule
    python Experiments/mobileNetv2_DINO/train_ssl.py --resume <run folder>    # continue an interrupted/stopped run
    python Experiments/mobileNetv2_DINO/train_ssl.py --smoke                  # tiny pipeline check, not a result

What happens:
    student = MultiCropWrapper(MobileNetV2 backbone from IMAGENET1K_V2, new DINO head)
    teacher = exact copy of the student (parameters AND BatchNorm buffers); never receives gradients
    per step: 2 global views -> teacher, all views -> student, DINO loss, student optimizer step,
              teacher parameters <- EMA of student parameters
    per epoch: checkpoint.pth (everything needed to resume), history.csv, charts
    after the last epoch: the TEACHER backbone is exported to backbone_teacher_epoch<N>.pth
Only ssl_training_data.csv images are used and no labels are read for training.
"""
import argparse
import contextlib
import csv
import json
import math
import os
import random
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import paths
from dino import (DataAugmentationDINO, DINOHead, DINOLoss, MobileNetV2Backbone, MultiCropWrapper,
                  cancel_gradients_last_layer, clip_gradients, collapse_diagnostics, cosine_scheduler,
                  get_params_groups)
from export_backbone import export
from plots import plot_ssl_history
from preview_views import save_view_preview
from ssl_data import SSLDataset, audit_splits, build_pixel_cache, flat_args, paths_hash, read_ssl_paths
from utils import environment_info, git_info, resolve_device, save_json, set_seed, sha256_file, sha256_text

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_ssl.json"
SMOKE_IMAGES = 48     # images in a smoke run, spread evenly over the SSL CSV (no labels involved)
SMOKE_OVERRIDES = {"optim": {"batch_size_per_gpu": 8, "epochs": 2, "warmup_epochs": 1}, "setup": {"log_every_steps": 1}}
RESUME_MAY_DIFFER = {"log_every_steps"}  # screen output only; anything else must match exactly on resume
TIMING_FROM, TIMING_TO = 5, 25           # steps used for the speed estimate (skips start-up steps)


def prepare_config(cfg: dict, smoke: bool) -> dict:
    cfg = json.loads(json.dumps(cfg))  # deep copy
    if smoke:
        for section, values in SMOKE_OVERRIDES.items():
            cfg[section].update(values)
    cfg["smoke"] = smoke
    return cfg


def validate(args, n_images: int) -> None:
    """Reject settings that would silently produce a different schedule than the one written down."""
    errors = []
    if args.epochs < 1:
        errors.append("epochs must be >= 1")
    if not 0 <= args.warmup_epochs < args.epochs:
        errors.append(f"warmup_epochs ({args.warmup_epochs}) must be >= 0 and < epochs ({args.epochs})")
    if not 0 <= args.warmup_teacher_temp_epochs <= args.epochs:
        errors.append("warmup_teacher_temp_epochs must be between 0 and epochs")
    if not 0 <= args.freeze_last_layer <= args.epochs:
        errors.append("freeze_last_layer must be between 0 and epochs")
    if args.batch_size_per_gpu > n_images:
        errors.append(f"batch_size_per_gpu ({args.batch_size_per_gpu}) is larger than the {n_images} images")
    if args.optimizer not in ("adamw", "sgd"):
        errors.append("optimizer must be adamw or sgd")
    if args.init != "imagenet":
        errors.append('init must be "imagenet" (route A1 starts from the same weights as A0)')
    if len(args.blur_p) != 3:
        errors.append("blur_p needs 3 values: global view 1, global view 2, local views")
    if errors:
        raise ValueError("invalid SSL config:\n  " + "\n  ".join(errors))


@contextlib.contextmanager
def preserved_rng():
    """Run something (the view preview) without changing the random streams used by training."""
    state = rng_state()
    yield
    restore_rng(state)


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def feature_dims(backbone, sizes) -> dict:
    """Output shape for each crop size. eval() so the probe does not touch BatchNorm running stats."""
    backbone.eval()
    with torch.no_grad():
        dims = {s: tuple(backbone(torch.zeros(2, 3, s, s)).shape) for s in sizes}
    backbone.train()
    return dims


def build_networks(args, device):
    student = MobileNetV2Backbone(imagenet_init=True)
    teacher = MobileNetV2Backbone(imagenet_init=False)  # overwritten by the student's state below
    embed_dim = student.feature_dim
    dims = feature_dims(student, [args.global_crop_size, args.local_crop_size])
    if any(d != (2, embed_dim) for d in dims.values()):
        raise ValueError(f"unexpected backbone output shapes {dims}, expected (2, {embed_dim})")

    # >>> DINO extract E11: main_dino.py L182-L212 (build student and teacher), modified
    # multi-crop wrapper handles forward with inputs of different resolutions
    student = MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.to(device), teacher.to(device)
    # one process, one GPU: ordinary BatchNorm, no SyncBatchNorm and no DistributedDataParallel
    # teacher and student start with the same weights
    teacher.load_state_dict(student.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both MobileNetV2 networks.")
    # <<< end E11
    return student, teacher, embed_dim, dims


def build_training_parts(args, student, niter_per_ep, device):
    # >>> DINO extract E12: main_dino.py L214-L251 (loss, optimizer, schedules), modified
    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        student_temp=args.student_temp,
        center_momentum=args.center_momentum,
    ).to(device)

    # ============ preparing optimizer ... ============
    params_groups = get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16 and device.type == "cuda":
        fp16_scaler = torch.amp.GradScaler("cuda")

    # ============ init schedulers ... ============
    lr_schedule = cosine_scheduler(
        args.lr * args.batch_size_per_gpu / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, niter_per_ep,
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, niter_per_ep,
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                         args.epochs, niter_per_ep)
    print(f"Loss, optimizer and schedulers ready.")
    # <<< end E12
    schedule_info = {
        "niter_per_ep": niter_per_ep, "total_iters": len(lr_schedule),
        "effective_peak_lr": args.lr * args.batch_size_per_gpu / 256.,
        "fingerprint": sha256_text(json.dumps([lr_schedule.tolist(), wd_schedule.tolist(),
                                               momentum_schedule.tolist(),
                                               dino_loss.teacher_temp_schedule.tolist()])),
    }
    return dino_loss, optimizer, fp16_scaler, (lr_schedule, wd_schedule, momentum_schedule), schedule_info


# >>> DINO extract E13: main_dino.py L301-L360 (train_one_epoch), modified
def train_one_epoch(student, teacher, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args, device, timing=None):
    sums, t0 = {}, time.time()
    for it, images in enumerate(data_loader):
        step = it + 1
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.to(device, non_blocking=True) for im in images]
        center = dino_loss.center.clone()  # the center this step uses, for the diagnostics
        # teacher and student forward passes + compute dino loss
        with torch.autocast(device.type, dtype=torch.float16, enabled=fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            raise RuntimeError("Loss is {}, stopping training".format(loss.item()))

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = clip_gradients(student, args.clip_grad)
            cancel_gradients_last_layer(epoch, student,
                                        args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = clip_gradients(student, args.clip_grad)
            cancel_gradients_last_layer(epoch, student,
                                        args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        diag = collapse_diagnostics(teacher_output, center, dino_loss.teacher_temp_schedule[epoch],
                                    teacher.last_features, len(images[0]))
        for k, v in {"loss": loss.item(), **diag}.items():
            sums[k] = sums.get(k, 0.0) + v
        if timing is not None:
            timing(step, len(data_loader))
        if args.log_every_steps and (step % args.log_every_steps == 0 or step == len(data_loader)):
            print(f"    step {step:5d}/{len(data_loader)}  loss {sums['loss'] / step:.4f}  "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}  teacher entropy {sums['teacher_entropy'] / step:.3f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    return {k: v / len(data_loader) for k, v in sums.items()} | {
        "lr": optimizer.param_groups[0]["lr"], "wd": optimizer.param_groups[0]["weight_decay"],
        "momentum": m, "teacher_temp": float(dino_loss.teacher_temp_schedule[epoch])}
# <<< end E13


HISTORY_COLUMNS = ["epoch", "loss", "teacher_entropy", "teacher_marginal_entropy", "feature_cosine", "feature_std",
                   "lr", "wd", "momentum", "teacher_temp", "center_norm", "seconds"]


def run(cfg: dict | None, smoke: bool = False, output_root: Path | None = None,
        resume_dir: Path | None = None, stop_after_epoch: int | None = None) -> Path:
    """Train (or resume) one DINO run and return its folder."""
    # ---- Config -------------------------------------------------------------------
    if resume_dir is not None:
        run_dir = Path(resume_dir)
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if cfg is not None:  # an explicitly given config must describe the same experiment
            given = prepare_config(cfg, saved["smoke"])
            diff = {k for k in vars(flat_args(given)) if getattr(flat_args(given), k) != getattr(flat_args(saved), k, None)}
            if diff - RESUME_MAY_DIFFER or given.keys() != saved.keys():
                raise ValueError(f"--config differs from the run being resumed in: {sorted(diff - RESUME_MAY_DIFFER)}")
        cfg = saved
    else:
        cfg = prepare_config(cfg, smoke)
    smoke = cfg["smoke"]
    args = flat_args(cfg)
    device = resolve_device(args.device)

    # ---- Data (no labels) and leakage guards ----------------------------------------------------
    audit = audit_splits(paths.SPLITS_DIR)
    ssl_paths = read_ssl_paths(paths.SPLITS_DIR / args.ssl_csv)
    if smoke:
        ssl_paths = ssl_paths[:: len(ssl_paths) // SMOKE_IMAGES][:SMOKE_IMAGES]
    validate(args, len(ssl_paths))
    manifest = {
        "ssl_csv": args.ssl_csv,
        "csv_sha256": sha256_file(paths.SPLITS_DIR / args.ssl_csv),
        "class_mapping_sha256": sha256_file(paths.SPLITS_DIR / "class_mapping.csv"),
        "used_paths_hash": paths_hash(ssl_paths),
        "num_images": len(ssl_paths),
        "split_audit": audit,
        "note": "used_paths_hash identifies the image list actually used (a manifest hash of sorted paths; "
                "not a hash of image contents). A smoke run uses a subset.",
    }

    # ---- Run folder and records ----------------------------------------------------------------
    if resume_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        root = Path(output_root) if output_root else (paths.SMOKE_OUTPUT_DIR if smoke else paths.SSL_OUTPUT_DIR)
        run_dir = root / f"{stamp}_{cfg['run_name']}_seed{args.seed}{'_SMOKE' if smoke else ''}"
        run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier run
        save_json(cfg, run_dir / "config.json")
        (run_dir / "manifests").mkdir()
        for f in [args.ssl_csv, "class_mapping.csv"]:
            shutil.copy(paths.SPLITS_DIR / f, run_dir / "manifests" / f)
        save_json(manifest, run_dir / "manifests.json")
        run_info = {"status": "running", "started": datetime.now().isoformat(timespec="seconds"),
                    "smoke": smoke, "environment": environment_info(device), "git": git_info(paths.REPO_ROOT),
                    "initialization": {"type": "imagenet", "file": paths.MOBILENETV2_IMAGENET_WEIGHTS.name,
                                       "sha256": sha256_file(paths.MOBILENETV2_IMAGENET_WEIGHTS),
                                       "weights": "torchvision MobileNet_V2_Weights.IMAGENET1K_V2 (same file as A0)"},
                    "checkpoint_rule": "final epoch, chosen in advance; DINO loss is not used to pick a checkpoint",
                    "export_rule": "teacher backbone, chosen in advance", "cost": {}}
        save_json(run_info, run_dir / "run_info.json")
        with open(run_dir / "history.csv", "w", newline="") as f:
            csv.writer(f).writerow(HISTORY_COLUMNS)
    else:
        recorded = json.loads((run_dir / "manifests.json").read_text(encoding="utf-8"))
        for k in ["csv_sha256", "used_paths_hash", "num_images"]:
            if recorded[k] != manifest[k]:
                raise ValueError(f"cannot resume: the SSL manifest changed since the run started ({k})")
        run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
        run_info.update(status="running", resumed=run_info.get("resumed", []) + [datetime.now().isoformat(timespec="seconds")])

    try:
        set_seed(args.seed)
        if device.type == "cuda":
            cudnn.benchmark = True  # as in main_dino.py: fastest conv algorithms (not bit-reproducible)

        # ---- Data loading ----------------------------------------------------------------------
        cache_file = None
        if args.cache_pixels:
            print(f"Preparing the pixel cache for {len(ssl_paths)} SSL images (slow only the first time)...", flush=True)
            t_load = time.time()
            cache_file = build_pixel_cache(ssl_paths, paths.CACHE_DIR)
            print(f"  {cache_file.name} ready in {time.time() - t_load:.0f}s", flush=True)
            # Image-content identity: hash of the decoded pixels themselves (unlike the manifest hash,
            # which only identifies the list of file paths).
            run_info["pixel_cache"] = {"file": cache_file.name, "sha256_of_pixels": sha256_file(cache_file)}
        dataset = SSLDataset(ssl_paths, DataAugmentationDINO(args), cache_file)
        niter_per_ep = len(dataset) // args.batch_size_per_gpu  # drop_last=True, as in main_dino.py

        if args.num_workers > 0:
            # Worker processes start fresh on Windows and inherit these variables. Each worker only
            # augments images, so one math thread is enough; with the default (one per CPU core) every
            # worker reserved large OpenBLAS buffers and ran out of memory on this PC.
            for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
                os.environ[var] = "1"

        def make_loader(epoch):
            # A fresh shuffle per epoch from seed + epoch, like DistributedSampler.set_epoch in the original.
            # So the image order of any epoch does not depend on whether the run was resumed.
            return DataLoader(dataset, batch_size=args.batch_size_per_gpu, shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(args.seed + epoch),
                              num_workers=args.num_workers, pin_memory=False,
                              persistent_workers=False)  # workers restart each epoch with that epoch's seed

        # ---- Networks, loss, optimizer, schedules -----------------------------------------------------
        student, teacher, embed_dim, dims = build_networks(args, device)
        dino_loss, optimizer, fp16_scaler, schedules, schedule_info = build_training_parts(
            args, student, niter_per_ep, device)
        lr_schedule, wd_schedule, momentum_schedule = schedules

        start_epoch = 0
        if resume_dir is not None:
            ckpt = torch.load(run_dir / "checkpoint.pth", map_location="cpu", weights_only=False)  # our own file
            diff = {k for k in set(ckpt["args"]) | set(vars(args))
                    if ckpt["args"].get(k) != getattr(args, k, None)} - RESUME_MAY_DIFFER
            if diff:
                raise ValueError(f"cannot resume: checkpoint settings differ in {sorted(diff)}")
            if ckpt["schedule_info"] != schedule_info:
                raise ValueError("cannot resume: the rebuilt schedules differ from the saved ones")
            if (ckpt["fp16_scaler"] is None) != (fp16_scaler is None):
                raise ValueError("cannot resume: mixed precision on/off differs (device changed?)")
            student.load_state_dict(ckpt["student"], strict=True)
            teacher.load_state_dict(ckpt["teacher"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            dino_loss.load_state_dict(ckpt["dino_loss"], strict=True)
            if fp16_scaler is not None:
                fp16_scaler.load_state_dict(ckpt["fp16_scaler"])
            start_epoch = ckpt["epoch"]
            if ckpt["global_step"] != start_epoch * niter_per_ep:
                raise ValueError("cannot resume: saved global step does not match the epoch boundary")
            restore_rng(ckpt["rng_state"])
            if start_epoch >= args.epochs:
                print(f"Run already finished all {args.epochs} epochs; nothing to resume.")
                run_info["status"] = "completed"
                save_json(run_info, run_dir / "run_info.json")
                return run_dir
        else:
            with preserved_rng():
                save_view_preview(ssl_paths, args, run_dir / "view_preview.png")

        n_views = 2 + args.local_crops_number
        n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        gpu = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
        print(f"\n{'SMOKE RUN (pipeline check only)' if smoke else 'DINO SSL run'}: {cfg['run_name']}"
              f"{'  (RESUMED at epoch ' + str(start_epoch + 1) + ')' if start_epoch else ''}\n"
              f"  output folder      : {run_dir}\n"
              f"  device             : {device}{gpu}, mixed precision {'on' if fp16_scaler else 'off'}\n"
              f"  init               : ImageNet IMAGENET1K_V2 (student and teacher identical at start)\n"
              f"  SSL images         : {len(ssl_paths)} from {args.ssl_csv} (no labels used)\n"
              f"  batch              : {args.batch_size_per_gpu} images x {n_views} views = "
              f"2 x {args.global_crop_size}px global + {args.local_crops_number} x {args.local_crop_size}px local\n"
              f"  dimensions         : MobileNetV2 feature {embed_dim} (checked: {dims}) -> DINO head output "
              f"{args.out_dim} (not the 40 classes)\n"
              f"  trainable params   : {n_trainable:,} (student backbone + head; teacher has 0)\n"
              f"  steps              : {niter_per_ep} per epoch, {args.epochs} epochs, {schedule_info['total_iters']} total\n"
              f"  learning rate      : reference {args.lr} per 256 images -> effective peak "
              f"{schedule_info['effective_peak_lr']:.3g} at batch {args.batch_size_per_gpu} after "
              f"{args.warmup_epochs} warm-up epochs, cosine to {args.min_lr}\n"
              f"  teacher momentum   : {args.momentum_teacher} -> 1, teacher temperature {args.teacher_temp}, "
              f"out_dim {args.out_dim}, seed {args.seed}", flush=True)
        run_info.update(num_trainable_parameters=n_trainable, feature_dim=embed_dim, out_dim=args.out_dim,
                        views_per_image=n_views, schedule=schedule_info)
        save_json(run_info, run_dir / "run_info.json")

        # ---- Training ---------------------------------------------------------------------------
        cost = run_info["cost"]
        last_epoch = args.epochs if stop_after_epoch is None else min(args.epochs, stop_after_epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(start_epoch, last_epoch):
            print(f"\nSSL epoch {epoch + 1}/{args.epochs}", flush=True)
            t0 = time.time()
            timing_marks = {}

            def timing(step, n_steps):  # speed estimate early in the first epoch of this process
                if epoch != start_epoch or step not in (TIMING_FROM, TIMING_TO):
                    return
                timing_marks[step] = time.time()
                if step == TIMING_TO:
                    s = (timing_marks[TIMING_TO] - timing_marks[TIMING_FROM]) / (TIMING_TO - TIMING_FROM)
                    left = s * (n_steps * (last_epoch - epoch) - step)
                    mem = (f", peak GPU memory {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GB"
                           if device.type == "cuda" else "")
                    print(f"  measured speed: {s:.3f} s/step -> about {s * n_steps / 60:.1f} min per epoch, "
                          f"about {left / 3600:.2f} h until epoch {last_epoch}{mem}", flush=True)
                    run_info["measured_seconds_per_step"] = round(s, 4)

            student.train()  # both stay in training mode, as in the original (the teacher is never .eval()):
            teacher.train()  # BatchNorm uses batch statistics and updates its own running stats
            stats = train_one_epoch(student, teacher, dino_loss, make_loader(epoch), optimizer,
                                    lr_schedule, wd_schedule, momentum_schedule, epoch, fp16_scaler, args,
                                    device, timing)
            secs = time.time() - t0

            # >>> DINO extract E14: main_dino.py L278-L290 (checkpoint contents), modified
            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': vars(args),
                'dino_loss': dino_loss.state_dict(),
                'fp16_scaler': fp16_scaler.state_dict() if fp16_scaler is not None else None,
                'global_step': (epoch + 1) * niter_per_ep,
                'schedule_info': schedule_info,
                'rng_state': rng_state(),
            }
            tmp = run_dir / "checkpoint.pth.tmp"
            torch.save(save_dict, tmp)
            os.replace(tmp, run_dir / "checkpoint.pth")  # only replaced once the new file is complete
            # <<< end E14

            stats["center_norm"] = dino_loss.center.norm().item()
            with open(run_dir / "history.csv", "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1] + [f"{stats[k]:.6g}" for k in HISTORY_COLUMNS[1:-1]] + [f"{secs:.1f}"])
            images_this_epoch = niter_per_ep * args.batch_size_per_gpu
            cost.update(epochs_done=epoch + 1, optimizer_steps=(epoch + 1) * niter_per_ep,
                        images_seen=(epoch + 1) * images_this_epoch,
                        views_seen_student=(epoch + 1) * images_this_epoch * n_views,
                        views_seen_teacher=(epoch + 1) * images_this_epoch * 2,
                        train_minutes=round(cost.get("train_minutes", 0) + secs / 60, 2),
                        device=str(device) + gpu,
                        peak_gpu_memory_gb=(round(max(cost.get("peak_gpu_memory_gb", 0),
                                                      torch.cuda.max_memory_allocated(device) / 2**30), 2)
                                            if device.type == "cuda" else None))
            save_json(run_info, run_dir / "run_info.json")
            print(f"  epoch {epoch + 1:3d}/{args.epochs} summary  loss {stats['loss']:.4f}  "
                  f"teacher entropy {stats['teacher_entropy']:.3f} (marginal {stats['teacher_marginal_entropy']:.3f}, "
                  f"max {math.log(args.out_dim):.2f})  feature cosine {stats['feature_cosine']:.3f}  "
                  f"{secs / 60:.1f} min, checkpoint saved", flush=True)
            try:
                plot_ssl_history(run_dir)
            except Exception as e:  # a chart problem should not stop training; history.csv has the numbers
                print(f"  WARNING: could not draw the SSL charts: {e}", flush=True)

        done = cost.get("epochs_done", 0) >= args.epochs
        run_info.update(status="completed" if done else "stopped (resumable)",
                        finished=datetime.now().isoformat(timespec="seconds"))
        save_json(run_info, run_dir / "run_info.json")
        if done:
            out = export(run_dir, "teacher")
            run_info["exported_backbone"] = {"file": out.name, "sha256": sha256_file(out)}
            save_json(run_info, run_dir / "run_info.json")
            print(f"\nDone. SSL cost: {cost['train_minutes']} min, {cost['images_seen']:,} images seen, "
                  f"{cost['optimizer_steps']:,} steps.\nOutputs in {run_dir}:\n"
                  f"  config.json, manifests.json, manifests/   settings and the image list used (with hashes)\n"
                  f"  run_info.json        status, environment, cost (time, images, steps, GPU memory)\n"
                  f"  history.csv, charts/ per-epoch loss and collapse diagnostics\n"
                  f"  view_preview.png     original chips beside the views used in training\n"
                  f"  checkpoint.pth       full training state (student, teacher, optimizer, center, RNG)\n"
                  f"  {out.name:<20} teacher backbone for fine-tuning\n"
                  f"Next: python Experiments/mobileNetv2_DINO/train_supervised.py --backbone \"{out}\"", flush=True)
        else:
            print(f"\nStopped after epoch {cost.get('epochs_done')} of {args.epochs}. Continue with:\n"
                  f"  python Experiments/mobileNetv2_DINO/train_ssl.py --resume \"{run_dir}\"", flush=True)
        return run_dir
    except BaseException:
        run_info.update(status="failed", error=traceback.format_exc(limit=3))
        save_json(run_info, run_dir / "run_info.json")
        raise


def main():
    parser = argparse.ArgumentParser(description="DINO SSL training of MobileNetV2 on the SAR SSL pool")
    parser.add_argument("--config", type=Path, help=f"default: {DEFAULT_CONFIG.name}")
    parser.add_argument("--resume", type=Path, help="run folder to continue from its checkpoint.pth")
    parser.add_argument("--stop-after-epoch", type=int,
                        help="stop (resumably) after this epoch; the schedule still assumes the configured epochs")
    parser.add_argument("--smoke", action="store_true",
                        help=f"{SMOKE_IMAGES} images, batch 8, 2 epochs; pipeline check only")
    parser.add_argument("--device", help='override setup.device: "cuda", "cpu" or "auto" (new runs only)')
    a = parser.parse_args()
    if a.resume:
        if a.smoke or a.device:
            raise SystemExit("--smoke and --device cannot be combined with --resume (the saved config is used)")
        cfg = json.loads(a.config.read_text(encoding="utf-8")) if a.config else None
        run(cfg, resume_dir=a.resume, stop_after_epoch=a.stop_after_epoch)
        return
    cfg = json.loads((a.config or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    if a.device:
        cfg["setup"]["device"] = a.device  # saved into the run's config.json
    run(cfg, smoke=a.smoke, stop_after_epoch=a.stop_after_epoch)


if __name__ == "__main__":
    main()
