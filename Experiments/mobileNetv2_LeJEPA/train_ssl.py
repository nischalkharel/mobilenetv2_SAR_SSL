"""LeJEPA self-supervised training of MobileNetV2 on the unlabeled SAR pool (stage 1 of route A2).

Run from the repository root:
    python Experiments/mobileNetv2_LeJEPA/train_ssl.py                          # full run, config_ssl.json
    python Experiments/mobileNetv2_LeJEPA/train_ssl.py --stop-after-epoch 1     # pilot: first epoch of the same schedule
    python Experiments/mobileNetv2_LeJEPA/train_ssl.py --resume <run folder>    # continue an interrupted/stopped run
    python Experiments/mobileNetv2_LeJEPA/train_ssl.py --smoke                  # tiny pipeline check, not a result

What happens:
    net = LeJEPANet(MobileNetV2 backbone from IMAGENET1K_V2, new MLP projector); ONE network, no teacher
    per step: V views of each image -> net -> projections (V, N, K)
              loss = (1 - lamb) * agreement(views vs. their mean) + lamb * SIGReg(projections)
              one AdamW step on everything (backbone + projector); no stop-gradient anywhere
    per epoch: checkpoint.pth (everything needed to resume), history.csv, charts;
               every setup.export_every epochs and after the last one: backbone_epoch<NNN>.pth
The backbone of the FINAL epoch is the one used for A2 (fixed in advance; no loss or label is used to
choose). Only ssl_training_data.csv images are used and no labels are read for training.
"""
import argparse
import contextlib
import csv
import json
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
from export_backbone import export
from lejepa_model import (DataAugmentationLeJEPA, LeJEPANet, build_sigreg, collapse_diagnostics, lejepa_loss, lr_schedule,
                          to_network_input)
from plots import plot_ssl_history
from preview_views import save_view_preview
from ssl_data import SSLDataset, audit_splits, build_pixel_cache, flat_args, paths_hash, read_ssl_paths
from utils import environment_info, git_info, resolve_device, save_json, set_seed, sha256_file, sha256_text

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_ssl.json"
SMOKE_IMAGES = 48     # images in a smoke run, spread evenly over the SSL CSV (no labels involved)
SMOKE_OVERRIDES = {"optim": {"batch_size": 8, "epochs": 2, "warmup_epochs": 1},
                   "setup": {"log_every_steps": 1, "export_every": 1}}
RESUME_MAY_DIFFER = {"log_every_steps"}  # screen output only; anything else must match exactly on resume
TIMING_FROM, TIMING_TO = 5, 25           # steps used for the speed estimate (skips start-up steps)
GPU_MEMORY_CAP_GB = 11.0                 # see run(): fail with "out of memory" instead of spilling into system RAM
HISTORY_COLUMNS = ["epoch", "loss", "agreement", "sigreg", "proj_std", "proj_mean_abs", "feature_cosine",
                   "feature_std", "lr", "seconds"]


def prepare_config(cfg: dict, smoke: bool) -> dict:
    cfg = json.loads(json.dumps(cfg))  # deep copy
    if smoke:
        for section, values in SMOKE_OVERRIDES.items():
            cfg[section].update(values)
    cfg["smoke"] = smoke
    return cfg


def validate(args, n_images: int) -> None:
    """Reject settings that would silently produce a different run than the one written down."""
    errors = []
    if args.epochs < 1:
        errors.append("epochs must be >= 1")
    if not 0 <= args.warmup_epochs < args.epochs:
        errors.append(f"warmup_epochs ({args.warmup_epochs}) must be >= 0 and < epochs ({args.epochs})")
    if args.batch_size > n_images:
        errors.append(f"batch_size ({args.batch_size}) is larger than the {n_images} images")
    if args.optimizer != "adamw":
        errors.append("optimizer must be adamw (README.md of LeJEPA: AdamW for all models)")
    if args.init != "imagenet":
        errors.append('init must be "imagenet" (route A2 starts from the same weights as A0 and A1)')
    if args.n_global_views < 2:
        errors.append("n_global_views must be >= 2 (the agreement term needs at least two views)")
    if args.n_local_views < 0:
        errors.append("n_local_views must be >= 0")
    if args.precision not in ("bf16", "fp32"):
        errors.append('precision must be "bf16" or "fp32"')
    if args.n_points % 2 != 1:
        errors.append("n_points must be odd (the library's EppsPulley asserts it)")
    if args.export_every < 0:
        errors.append("export_every must be >= 0 (0 = only after the last epoch)")
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


def train_one_epoch(net, sigreg, data_loader, optimizer, lrs, epoch, args, device, autocast_dtype, timing=None):
    """One pass over the SSL pool. Returns the per-epoch means of the logged values."""
    sums, t0 = {}, time.time()
    n_steps = len(data_loader)
    for it, views in enumerate(data_loader):
        step = it + 1
        global_it = n_steps * epoch + it
        for group in optimizer.param_groups:
            group["lr"] = lrs[global_it]
        views = to_network_input(views, args.normalize_mean, args.normalize_std, device)  # uint8 gray -> normalized RGB
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            emb, proj = net(views)
        loss, agreement, sig = lejepa_loss(proj, args.n_global_views, sigreg, args.lamb)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss is {loss.item()}, stopping training")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        diag = collapse_diagnostics(emb, proj)
        for k, v in {"loss": loss.item(), "agreement": agreement.item(), "sigreg": sig.item(), **diag}.items():
            sums[k] = sums.get(k, 0.0) + v
        if timing is not None:
            timing(step, n_steps)
        if args.log_every_steps and (step % args.log_every_steps == 0 or step == n_steps):
            print(f"    step {step:5d}/{n_steps}  loss {sums['loss'] / step:.4f}  agreement "
                  f"{sums['agreement'] / step:.4f}  sigreg {sums['sigreg'] / step:.3f}  proj std "
                  f"{sums['proj_std'] / step:.3f}  lr {optimizer.param_groups[0]['lr']:.2e}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    return {k: v / n_steps for k, v in sums.items()} | {"lr": optimizer.param_groups[0]["lr"]}


def run(cfg: dict | None, smoke: bool = False, output_root: Path | None = None,
        resume_dir: Path | None = None, stop_after_epoch: int | None = None) -> Path:
    """Train (or resume) one LeJEPA run and return its folder."""
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
        run_dir = root / f"{stamp}_{cfg['run_name']}{paths.RUN_TAG}_seed{args.seed}{'_SMOKE' if smoke else ''}"
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
                                       "weights": "torchvision MobileNet_V2_Weights.IMAGENET1K_V2 (same file as A0 and A1)"},
                    "checkpoint_rule": "final epoch, chosen in advance; the LeJEPA loss is not used to pick a checkpoint",
                    "export_rule": "single network (no teacher); intermediate exports are for analysis only",
                    "cost": {}}
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
            cudnn.benchmark = True  # fastest conv algorithms (not bit-reproducible)
            # Windows can silently move GPU memory into system RAM when the card is full, which made a test
            # run about 100 times slower instead of failing. A cap on PyTorch's allocator turns that into a
            # clear "out of memory" error. It does not stop Windows from spilling when OTHER programs also hold GPU
            # memory (README.md, "Time and memory"). Windows only: Linux (e.g. Pitt CRC) does not spill, and an
            # 11 GB cap would block the batch-64 runs on 40 GB cards.
            if os.name == "nt":
                gpu_index = device.index if device.index is not None else torch.cuda.current_device()
                torch.cuda.set_per_process_memory_fraction(
                    min(1.0, GPU_MEMORY_CAP_GB * 2**30 / torch.cuda.get_device_properties(gpu_index).total_memory),
                    gpu_index)

        # ---- Data loading ----------------------------------------------------------------------
        cache_file = None
        if args.cache_pixels:
            print(f"Preparing the pixel cache for {len(ssl_paths)} SSL images (slow only the first time)...", flush=True)
            t_load = time.time()
            cache_file = build_pixel_cache(ssl_paths, paths.CACHE_DIR)
            print(f"  {cache_file.name} ready in {time.time() - t_load:.0f}s", flush=True)
            run_info["pixel_cache"] = {"file": cache_file.name, "sha256_of_pixels": sha256_file(cache_file)}
        dataset = SSLDataset(ssl_paths, DataAugmentationLeJEPA(args), cache_file)
        niter_per_ep = len(dataset) // args.batch_size  # drop_last=True, as in MINIMAL.md

        if args.num_workers > 0:
            # Worker processes start fresh on Windows and inherit these variables. Each worker only
            # augments images, so one math thread is enough (as in the A1 DINO runs on this PC).
            for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
                os.environ[var] = "1"

        def make_loader(epoch):
            # A fresh shuffle per epoch from seed + epoch, so the image order of any epoch does not
            # depend on whether the run was resumed.
            return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(args.seed + epoch),
                              num_workers=args.num_workers, pin_memory=args.pin_memory and device.type == "cuda",
                              persistent_workers=False)  # workers restart each epoch with that epoch's seed

        # ---- Network, loss, optimizer, schedule ---------------------------------------------------
        net = LeJEPANet(imagenet_init=True, proj_hidden_dim=args.proj_hidden_dim, proj_dim=args.proj_dim)
        sizes = [args.global_crop_size] + ([args.local_crop_size] if args.n_local_views else [])
        dims = feature_dims(net.backbone, sizes)
        if any(d != (2, net.backbone.feature_dim) for d in dims.values()):
            raise ValueError(f"unexpected backbone output shapes {dims}")
        net = net.to(device)
        sigreg = build_sigreg(args).to(device)
        # README.md / MINIMAL.md: AdamW, one parameter group, weight decay on all parameters, no gradient clipping
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        autocast_dtype = torch.bfloat16 if (args.precision == "bf16" and device.type == "cuda") else None
        total_steps = args.epochs * niter_per_ep
        lrs = lr_schedule(args.lr, total_steps, args.warmup_epochs * niter_per_ep,
                          args.warmup_start_factor, args.final_lr_ratio)
        schedule_info = {"niter_per_ep": niter_per_ep, "total_iters": total_steps,
                         "warmup_iters": args.warmup_epochs * niter_per_ep,
                         "peak_lr": args.lr, "final_lr": lrs[-1], "fingerprint": sha256_text(json.dumps(lrs))}

        start_epoch = 0
        if resume_dir is not None:
            ckpt = torch.load(run_dir / "checkpoint.pth", map_location="cpu", weights_only=False)  # our own file
            diff = {k for k in set(ckpt["args"]) | set(vars(args))
                    if ckpt["args"].get(k) != getattr(args, k, None)} - RESUME_MAY_DIFFER
            if diff:
                raise ValueError(f"cannot resume: checkpoint settings differ in {sorted(diff)}")
            if ckpt["schedule_info"] != schedule_info:
                raise ValueError("cannot resume: the rebuilt schedule differs from the saved one")
            if ckpt["autocast"] != str(autocast_dtype):
                raise ValueError("cannot resume: mixed precision on/off differs (device changed?)")
            net.load_state_dict(ckpt["model"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            sigreg.load_state_dict(ckpt["sigreg"], strict=True)  # its global_step seeds the random directions
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

        n_views = args.n_global_views + args.n_local_views
        n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        gpu = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
        local = f" + {args.n_local_views} x {args.local_crop_size}px local" if args.n_local_views else " (all global)"
        print(f"\n{'SMOKE RUN (pipeline check only)' if smoke else 'LeJEPA SSL run'}: {cfg['run_name']}"
              f"{'  (RESUMED at epoch ' + str(start_epoch + 1) + ')' if start_epoch else ''}\n"
              f"  output folder      : {run_dir}\n"
              f"  device             : {device}{gpu}, precision {'bf16 autocast' if autocast_dtype else 'fp32'}\n"
              f"  init               : ImageNet IMAGENET1K_V2 (one network; no teacher)\n"
              f"  SSL images         : {len(ssl_paths)} from {args.ssl_csv} (no labels used)\n"
              f"  batch              : {args.batch_size} images x {n_views} views = {args.n_global_views} x "
              f"{args.global_crop_size}px{local}\n"
              f"  dimensions         : MobileNetV2 feature {net.backbone.feature_dim} (checked: {dims}) -> projector "
              f"{args.proj_hidden_dim}-{args.proj_hidden_dim}-{args.proj_dim}\n"
              f"  loss               : (1 - {args.lamb}) x agreement + {args.lamb} x SIGReg "
              f"({args.num_slices} directions, Epps-Pulley {args.n_points} points on [0, {args.t_max}])\n"
              f"  trainable params   : {n_trainable:,} (backbone + projector)\n"
              f"  steps              : {niter_per_ep} per epoch, {args.epochs} epochs, {total_steps} total\n"
              f"  learning rate      : AdamW peak {args.lr} after {args.warmup_epochs} warm-up epoch(s) from "
              f"{args.warmup_start_factor} x lr, cosine to {lrs[-1]:.2e}; weight decay {args.weight_decay}; seed {args.seed}",
              flush=True)
        run_info.update(num_trainable_parameters=n_trainable, feature_dim=net.backbone.feature_dim,
                        proj_dim=args.proj_dim, views_per_image=n_views, schedule=schedule_info)
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

            net.train()  # BatchNorm uses batch statistics and updates its running stats (exported with the backbone)
            stats = train_one_epoch(net, sigreg, make_loader(epoch), optimizer, lrs, epoch, args, device,
                                    autocast_dtype, timing)
            secs = time.time() - t0

            save_dict = {
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "sigreg": sigreg.state_dict(),
                "epoch": epoch + 1,
                "args": vars(args),
                "autocast": str(autocast_dtype),
                "global_step": (epoch + 1) * niter_per_ep,
                "schedule_info": schedule_info,
                "rng_state": rng_state(),
            }
            tmp = run_dir / "checkpoint.pth.tmp"
            torch.save(save_dict, tmp)
            os.replace(tmp, run_dir / "checkpoint.pth")  # only replaced once the new file is complete

            with open(run_dir / "history.csv", "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1] + [f"{stats[k]:.6g}" for k in HISTORY_COLUMNS[1:-1]] + [f"{secs:.1f}"])
            images_this_epoch = niter_per_ep * args.batch_size
            cost.update(epochs_done=epoch + 1, optimizer_steps=(epoch + 1) * niter_per_ep,
                        images_seen=(epoch + 1) * images_this_epoch,
                        views_seen=(epoch + 1) * images_this_epoch * n_views,
                        train_minutes=round(cost.get("train_minutes", 0) + secs / 60, 2),
                        device=str(device) + gpu,
                        peak_gpu_memory_gb=(round(max(cost.get("peak_gpu_memory_gb", 0),
                                                      torch.cuda.max_memory_allocated(device) / 2**30), 2)
                                            if device.type == "cuda" else None))
            save_json(run_info, run_dir / "run_info.json")
            print(f"  epoch {epoch + 1:3d}/{args.epochs} summary  loss {stats['loss']:.4f}  agreement "
                  f"{stats['agreement']:.4f}  sigreg {stats['sigreg']:.3f}  proj std {stats['proj_std']:.3f}  "
                  f"feature cosine {stats['feature_cosine']:.3f}  {secs / 60:.1f} min, checkpoint saved", flush=True)
            if (epoch + 1) == args.epochs or (args.export_every and (epoch + 1) % args.export_every == 0):
                out = export(run_dir)
                run_info.setdefault("exported_backbones", {})[out.name] = sha256_file(out)
                save_json(run_info, run_dir / "run_info.json")
            try:
                plot_ssl_history(run_dir)
            except Exception as e:  # a chart problem should not stop training; history.csv has the numbers
                print(f"  WARNING: could not draw the SSL charts: {e}", flush=True)

        done = cost.get("epochs_done", 0) >= args.epochs
        run_info.update(status="completed" if done else "stopped (resumable)",
                        finished=datetime.now().isoformat(timespec="seconds"))
        save_json(run_info, run_dir / "run_info.json")
        if done:
            final = run_dir / f"backbone_epoch{args.epochs:03d}.pth"
            print(f"\nDone. SSL cost: {cost['train_minutes']} min, {cost['images_seen']:,} images seen, "
                  f"{cost['optimizer_steps']:,} steps.\nOutputs in {run_dir}:\n"
                  f"  config.json, manifests.json, manifests/   settings and the image list used (with hashes)\n"
                  f"  run_info.json        status, environment, cost (time, images, steps, GPU memory)\n"
                  f"  history.csv, charts/ per-epoch loss terms and collapse diagnostics\n"
                  f"  view_preview.png     original chips beside the views used in training\n"
                  f"  checkpoint.pth       full training state (network, optimizer, SIGReg step, RNG)\n"
                  f"  {final.name:<20} FINAL backbone for fine-tuning (route A2)\n"
                  f"Next: python Experiments/mobileNetv2_LeJEPA/train_supervised.py --backbone \"{final}\"", flush=True)
        else:
            print(f"\nStopped after epoch {cost.get('epochs_done')} of {args.epochs}. Continue with:\n"
                  f"  python Experiments/mobileNetv2_LeJEPA/train_ssl.py --resume \"{run_dir}\"", flush=True)
        return run_dir
    except BaseException:
        run_info.update(status="failed", error=traceback.format_exc(limit=3))
        save_json(run_info, run_dir / "run_info.json")
        raise


def main():
    parser = argparse.ArgumentParser(description="LeJEPA SSL training of MobileNetV2 on the SAR SSL pool")
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
