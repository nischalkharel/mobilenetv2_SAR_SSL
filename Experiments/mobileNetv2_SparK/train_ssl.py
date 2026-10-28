"""SparK self-supervised training of MobileNetV2 on the unlabeled SAR pool (stage 1 of route A3).

Run from the repository root:
    python Experiments/mobileNetv2_SparK/train_ssl.py                          # full run, config_ssl.json
    python Experiments/mobileNetv2_SparK/train_ssl.py --resume <run folder>    # continue an interrupted/stopped run
    python Experiments/mobileNetv2_SparK/train_ssl.py --smoke                  # tiny pipeline check, not a result

What happens:
    model = SparK(sparse MobileNetV2 encoder from IMAGENET1K_V2, densify layers + mask tokens, LightDecoder)
    per step: one view of each image; a random 7 x 7 patch mask (60% hidden); the encoder sees only the
              visible patches (sparse conv/BN); the decoder reconstructs the hidden patches;
              loss = mean squared error on the hidden patches, pixels normalized per patch
              one LAMB step on everything (encoder + densify layers + mask tokens + decoder)
    per epoch: checkpoint.pth (everything needed to resume), history.csv, charts;
               every setup.export_every epochs and after the last one: backbone_epoch<NNN>.pth and
               charts/reconstruction_epoch<NNN>.png
The encoder of the FINAL epoch is the backbone used for A3 (fixed in advance; no loss or label is used to
choose). Only ssl_training_data.csv images are used and no labels are read for training.
"""
import argparse
import contextlib
import csv
import io
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
from plots import plot_ssl_history
from preview_views import save_reconstruction_preview, save_view_preview
from spark_lib.lamb import TheSameAsTimmLAMB
from spark_lib.lr_control import get_param_groups, lr_wd_annealing
from spark_model import (DOWNSAMPLE_RATIO, DataAugmentationSparK, build_spark, count_parameters,
                         feature_diagnostics, sparse, to_network_input)
from ssl_data import SSLDataset, audit_splits, build_pixel_cache, flat_args, paths_hash, read_ssl_paths
from utils import environment_info, git_info, resolve_device, save_json, set_seed, sha256_file

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_ssl.json"
SMOKE_IMAGES = 48     # images in a smoke run, spread evenly over the SSL CSV (no labels involved)
SMOKE_OVERRIDES = {"optim": {"batch_size": 8, "epochs": 2, "warmup_epochs": 1},
                   "setup": {"log_every_steps": 1, "export_every": 1}}
RESUME_MAY_DIFFER = {"log_every_steps"}  # screen output only; anything else must match exactly on resume
TIMING_FROM, TIMING_TO = 5, 25           # steps used for the speed estimate (skips start-up steps)
GPU_MEMORY_CAP_GB = 11.0                 # see run(): fail with "out of memory" instead of spilling into system RAM
HISTORY_COLUMNS = ["epoch", "loss", "feature_cosine", "feature_std", "grad_norm", "lr", "wd", "seconds"]
NOWD_KEYS = {"cls_token", "pos_embed", "mask_token", "gamma"}  # pretrain/main.py L75 (only mask_token exists here)


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
    if args.optimizer != "lamb":
        errors.append("optimizer must be lamb (the authors' pre-training optimizer)")
    if args.init != "imagenet":
        errors.append('init must be "imagenet" (route A3 starts from the same weights as A0, A1 and A2)')
    if args.precision != "fp32":
        errors.append('precision must be "fp32" (the authors train in full float32, without mixed precision)')
    if args.densify_norm != "bn":
        errors.append('densify_norm must be "bn" (the authors\' setting for BatchNorm CNNs; arg_util.py L123-L125)')
    if args.recon_channels not in (1, 3):
        errors.append("recon_channels must be 1 (SAR channel) or 3 (the authors' literal target)")
    if not 0 < args.mask_ratio < 1:
        errors.append("mask_ratio must be in (0, 1)")
    if args.input_size % 32:
        errors.append("input_size must be a multiple of 32 (MobileNetV2's total downsampling = the mask patch size)")
    if args.decoder_width < 32 or args.decoder_width % 32:
        errors.append("decoder_width must be a multiple of 32 (the decoder halves its width 5 times)")
    lo, hi = args.crop_scale
    if not 0 < lo <= hi <= 1:
        errors.append("crop_scale must satisfy 0 < min <= max <= 1")
    if args.max_grad_norm <= 0:
        errors.append("max_grad_norm must be > 0")
    if args.export_every < 0:
        errors.append("export_every must be >= 0 (0 = only after the last epoch)")
    if errors:
        raise ValueError("invalid SSL config:\n  " + "\n  ".join(errors))


def use_true_fp32() -> None:
    """Full float32 on the GPU: switch off TF32, which PyTorch uses by default for float32 convolutions on
    Ampere and newer GPUs (A100, RTX 30/40). TF32 keeps only 10 mantissa bits. Measured at the start of
    SparK training on SSL chips (README.md, "Precision"): against a float64 reference, the gradient error
    is about 100% with TF32 (the encoder's gradient is mostly rounding noise) and about 11% without it
    (CPU float32: 3%). The authors' code has no mixed precision, and the paper mentions V100 GPUs, which
    have no TF32."""
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def write_json(obj, path: Path) -> None:
    """save_json through a temporary file, so a crash can never leave a half-written file behind."""
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    save_json(obj, tmp)
    os.replace(tmp, path)


def epoch_outputs(run_dir: Path, epoch: int, args, model, ssl_paths, device, run_info: dict) -> None:
    """After the checkpoint of `epoch` (counted from 1): the backbone export (every export_every epochs and
    after the last), the reconstruction preview and the charts. Runs with the random state restored
    afterwards: the masks come from the global torch generator, and export() builds a fresh MobileNetV2
    (random initial weights), so without this a resumed run would draw different masks than an
    uninterrupted one (found by the smoke test)."""
    with preserved_rng():
        if epoch == args.epochs or (args.export_every and epoch % args.export_every == 0):
            out = export(run_dir)
            run_info.setdefault("exported_backbones", {})[out.name] = sha256_file(out)
            write_json(run_info, run_dir / "run_info.json")
            try:
                save_reconstruction_preview(model, ssl_paths, args, device,
                                            run_dir / "charts" / f"reconstruction_epoch{epoch:03d}.png")
            except Exception as e:  # a picture problem should not stop training
                print(f"  WARNING: could not draw the reconstruction preview: {e}", flush=True)
        try:
            plot_ssl_history(run_dir)
        except Exception as e:  # a chart problem should not stop training; history.csv has the numbers
            print(f"  WARNING: could not draw the SSL charts: {e}", flush=True)


def repair_after_stop(run_dir: Path, ckpt: dict, args, model, ssl_paths, device, run_info: dict) -> None:
    """A run can stop between writing checkpoint.pth and finishing that epoch's other outputs. Rebuild them
    from the checkpoint: its history row and cost, and a backbone export that is due but missing."""
    epoch = ckpt["epoch"]
    rows = (run_dir / "history.csv").read_text().splitlines()[1:]
    if len(rows) == epoch - 1:  # the checkpoint's epoch never reached history.csv
        with open(run_dir / "history.csv", "a", newline="") as f:
            csv.writer(f).writerow(ckpt["history_row"])
        print(f"  repaired: history row of epoch {epoch} restored from the checkpoint", flush=True)
    elif len(rows) != epoch:
        raise ValueError(f"cannot resume: history.csv has {len(rows)} epochs, checkpoint.pth is at epoch {epoch}")
    run_info["cost"] = ckpt["cost"]  # run_info.json may lag the checkpoint by one epoch
    due = epoch == args.epochs or (args.export_every and epoch % args.export_every == 0)
    if due and not (run_dir / f"backbone_epoch{epoch:03d}.pth").exists():
        print(f"  repaired: writing the epoch-{epoch} export that the stopped run did not write", flush=True)
        epoch_outputs(run_dir, epoch, args, model, ssl_paths, device, run_info)


@contextlib.contextmanager
def preserved_rng():
    """Run something (a preview) without changing the random streams used by training."""
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


def feature_shapes(model, input_size: int) -> list:
    """Shapes of the 4 encoder maps for an all-visible input; eval() so BatchNorm running stats are untouched."""
    model.eval()
    f = input_size // DOWNSAMPLE_RATIO
    with torch.no_grad():
        sparse._cur_active = torch.ones(2, 1, f, f, dtype=torch.bool)  # the mask every sparse layer reads
        shapes = [tuple(m.shape) for m in model.sparse_encoder(torch.zeros(2, 3, input_size, input_size))]
    model.train()
    return shapes


def train_one_epoch(model, data_loader, optimizer, epoch, args, device, schedule, timing=None):
    """One pass over the SSL pool. Returns the per-epoch means of the logged values."""
    sums, t0 = {}, time.time()
    n_steps = len(data_loader)
    for it, views in enumerate(data_loader):
        step = it + 1
        global_it = n_steps * epoch + it
        # The authors' per-iteration schedule (lr_control.py): warm-up then cosine for lr, cosine for weight decay.
        _, max_lr, _, max_wd = lr_wd_annealing(optimizer, schedule["peak_lr"], args.weight_decay, args.weight_decay_end,
                                               global_it, schedule["warmup_iters"], schedule["total_iters"])
        x = to_network_input(views, args.normalize_mean, args.normalize_std, device)  # uint8 gray -> normalized RGB
        loss, smallest, active = model(x)  # a new random mask for every image
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss is {loss.item()}, stopping training")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()  # LAMB clips the global gradient norm to max_grad_norm inside step() (main.py L79)

        diag = feature_diagnostics(smallest, active)
        for k, v in {"loss": loss.item(), "grad_norm": optimizer.global_grad_norm, **diag}.items():
            sums[k] = sums.get(k, 0.0) + v
        if timing is not None:
            timing(step, n_steps)
        if args.log_every_steps and (step % args.log_every_steps == 0 or step == n_steps):
            print(f"    step {step:5d}/{n_steps}  loss {sums['loss'] / step:.4f}  grad norm "
                  f"{sums['grad_norm'] / step:.3f}  feature cosine {sums['feature_cosine'] / step:.3f}  "
                  f"lr {max_lr:.2e}  wd {max_wd:.3f}  {time.time() - t0:.0f}s", flush=True)
    return {k: v / n_steps for k, v in sums.items()} | {"lr": max_lr, "wd": max_wd}


def run(cfg: dict | None, smoke: bool = False, output_root: Path | None = None,
        resume_dir: Path | None = None, stop_after_epoch: int | None = None) -> Path:
    """Train (or resume) one SparK run and return its folder."""
    # ---- Config -------------------------------------------------------------------
    if resume_dir is not None:
        run_dir = Path(resume_dir)
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if cfg is not None:  # an explicitly given config must describe the same experiment
            given = prepare_config(cfg, saved["smoke"])
            diff = {k for k in vars(flat_args(given)) if getattr(flat_args(given), k) != getattr(flat_args(saved), k, None)}
            if diff - RESUME_MAY_DIFFER or given.keys() != saved.keys():
                raise ValueError(f"--config differs from the run being resumed in: {sorted(diff - RESUME_MAY_DIFFER)}")
            saved["setup"]["log_every_steps"] = given["setup"]["log_every_steps"]  # the one setting allowed to change
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
        write_json(cfg, run_dir / "config.json")
        (run_dir / "manifests").mkdir()
        for f in [args.ssl_csv, "class_mapping.csv"]:
            shutil.copy(paths.SPLITS_DIR / f, run_dir / "manifests" / f)
        write_json(manifest, run_dir / "manifests.json")
        run_info = {"status": "running", "started": datetime.now().isoformat(timespec="seconds"),
                    "smoke": smoke, "environment": environment_info(device), "git": git_info(paths.REPO_ROOT),
                    "initialization": {"type": "imagenet", "file": paths.MOBILENETV2_IMAGENET_WEIGHTS.name,
                                       "sha256": sha256_file(paths.MOBILENETV2_IMAGENET_WEIGHTS),
                                       "weights": "torchvision MobileNet_V2_Weights.IMAGENET1K_V2 (same file as A0, A1 and A2)",
                                       "decoder_side": "randomly initialized (densify layers, mask tokens, LightDecoder)"},
                    "checkpoint_rule": "final epoch, chosen in advance; the SparK loss is not used to pick a checkpoint",
                    "export_rule": "encoder only (MobileNetV2 features); decoder, densify layers and mask tokens are "
                                   "discarded; intermediate exports are for analysis only",
                    "cost": {}}
        write_json(run_info, run_dir / "run_info.json")
        with open(run_dir / "history.csv", "w", newline="") as f:
            csv.writer(f).writerow(HISTORY_COLUMNS)
    else:
        recorded = json.loads((run_dir / "manifests.json").read_text(encoding="utf-8"))
        for k in ["csv_sha256", "used_paths_hash", "num_images"]:
            if recorded[k] != manifest[k]:
                raise ValueError(f"cannot resume: the SSL manifest changed since the run started ({k})")
        run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
        run_info.update(status="running", resumed=run_info.get("resumed", []) + [
            {"time": datetime.now().isoformat(timespec="seconds"), "git": git_info(paths.REPO_ROOT),
             "environment": environment_info(device)}])

    try:
        set_seed(args.seed)
        if device.type == "cuda":
            cudnn.benchmark = True  # fastest conv algorithms (not bit-reproducible); the authors set it too
            use_true_fp32()
            # Windows can silently move GPU memory into system RAM when the card is full, which made a LeJEPA
            # test run about 100 times slower instead of failing. A cap on PyTorch's allocator turns that into
            # a clear "out of memory" error. Windows only: Linux (e.g. Pitt CRC) does not spill, and an 11 GB
            # cap would block the batch-64 runs on 40 GB cards.
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
        dataset = SSLDataset(ssl_paths, DataAugmentationSparK(args), cache_file)
        niter_per_ep = len(dataset) // args.batch_size  # drop_last=True, as in A1 and A2

        if args.num_workers > 0:
            # Worker processes start fresh on Windows and inherit these variables. Each worker only
            # crops and flips images, so one math thread is enough (as in A1 and A2).
            for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
                os.environ[var] = "1"

        def make_loader(epoch):
            # A fresh shuffle per epoch from seed + epoch, so the image order of any epoch does not
            # depend on whether the run was resumed.
            return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(args.seed + epoch),
                              num_workers=args.num_workers, pin_memory=args.pin_memory and device.type == "cuda",
                              persistent_workers=False)  # workers restart each epoch with that epoch's seed

        # ---- Model, optimizer, schedule -----------------------------------------------------------
        model = build_spark(args, imagenet_init=True)
        shapes = feature_shapes(model, args.input_size)
        fmap = args.input_size // DOWNSAMPLE_RATIO  # 7 for 224 px: the mask grid
        expected = [(2, 24, 8 * fmap, 8 * fmap), (2, 32, 4 * fmap, 4 * fmap), (2, 96, 2 * fmap, 2 * fmap), (2, 1280, fmap, fmap)]
        if shapes != expected:
            raise ValueError(f"unexpected encoder feature maps {shapes}, expected {expected}")
        model = model.to(device)
        # pretrain/main.py L75-L81: no weight decay on 1-D tensors, biases and mask tokens; LAMB with
        # betas (0.9, beta2) and the global gradient norm clipped inside the optimizer; created with lr=peak
        # and weight_decay=0 as there. lr_wd_annealing() sets every group's lr and weight decay before each step.
        peak_lr = args.base_lr * args.batch_size / 256  # arg_util.py L132
        with contextlib.redirect_stdout(io.StringIO()) as listing:  # it prints every parameter name; keep that in a file
            param_groups = get_param_groups(model, nowd_keys=NOWD_KEYS)
        (run_dir / "param_groups.txt").write_text(listing.getvalue(), encoding="utf-8")
        optimizer = TheSameAsTimmLAMB(params=param_groups, lr=peak_lr, weight_decay=0.0, betas=(0.9, args.beta2),
                                      max_grad_norm=args.max_grad_norm)
        total_iters = args.epochs * niter_per_ep
        warmup_iters = args.warmup_epochs * niter_per_ep
        if round(warmup_iters) >= total_iters - 1:
            raise ValueError("warm-up must end at least 2 steps before the last step (lr_wd_annealing divides by the rest)")
        schedule = {"niter_per_ep": niter_per_ep, "total_iters": total_iters, "warmup_iters": warmup_iters,
                    "peak_lr": peak_lr, "final_lr": 0.001 * peak_lr, "warmup_start_lr": 0.005 * peak_lr,
                    "weight_decay_start": args.weight_decay, "weight_decay_end": args.weight_decay_end}

        start_epoch = 0
        if resume_dir is not None:
            ckpt = torch.load(run_dir / "checkpoint.pth", map_location="cpu", weights_only=False)  # our own file
            diff = {k for k in set(ckpt["args"]) | set(vars(args))
                    if ckpt["args"].get(k) != getattr(args, k, None)} - RESUME_MAY_DIFFER
            if diff:
                raise ValueError(f"cannot resume: checkpoint settings differ in {sorted(diff)}")
            if ckpt["schedule"] != schedule:
                raise ValueError("cannot resume: the rebuilt schedule differs from the saved one")
            model.load_state_dict(ckpt["model"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"]
            if ckpt["global_step"] != start_epoch * niter_per_ep:
                raise ValueError("cannot resume: saved global step does not match the epoch boundary")
            restore_rng(ckpt["rng_state"])
            repair_after_stop(run_dir, ckpt, args, model, ssl_paths, device, run_info)
            if start_epoch >= args.epochs:
                print(f"Run already finished all {args.epochs} epochs; nothing to resume.")
                run_info["status"] = "completed"
                write_json(run_info, run_dir / "run_info.json")
                return run_dir
        else:
            with preserved_rng():
                save_view_preview(ssl_paths, args, run_dir / "view_preview.png")

        n_params = count_parameters(model)
        gpu = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
        print(f"\n{'SMOKE RUN (pipeline check only)' if smoke else 'SparK SSL run'}: {cfg['run_name']}"
              f"{'  (RESUMED at epoch ' + str(start_epoch + 1) + ')' if start_epoch else ''}\n"
              f"  output folder      : {run_dir}\n"
              f"  device             : {device}{gpu}, precision float32"
              f"{' (TF32 off)' if device.type == 'cuda' and not cudnn.allow_tf32 else ''}\n"
              f"  init               : ImageNet IMAGENET1K_V2 encoder; decoder side random\n"
              f"  SSL images         : {len(ssl_paths)} from {args.ssl_csv} (no labels used)\n"
              f"  view               : 1 per image, {args.input_size}px, crop scale {args.crop_scale}, flip p {args.hflip_p}\n"
              f"  mask               : {fmap} x {fmap} patches of 32px, {model.len_keep} visible per image "
              f"(mask ratio {args.mask_ratio} -> {1 - model.len_keep / fmap ** 2:.3f} hidden)\n"
              f"  encoder maps       : {shapes}\n"
              f"  target             : {'the SAR channel' if args.recon_channels == 1 else 'all 3 input channels'}, "
              f"normalized per patch; loss on hidden patches only\n"
              f"  parameters         : {', '.join(f'{k} {v:,}' for k, v in n_params.items())}\n"
              f"  steps              : {niter_per_ep} per epoch, {args.epochs} epochs, {total_iters} total\n"
              f"  optimizer          : LAMB betas (0.9, {args.beta2}), grad norm clipped at {args.max_grad_norm}; "
              f"peak lr {peak_lr:.2e} (= {args.base_lr} x {args.batch_size} / 256) after {args.warmup_epochs} "
              f"warm-up epochs from {schedule['warmup_start_lr']:.2e}, cosine to {schedule['final_lr']:.2e}; "
              f"weight decay {args.weight_decay} -> {args.weight_decay_end} (cosine); seed {args.seed}",
              flush=True)
        run_info.update(parameters=n_params, encoder_feature_maps=shapes, visible_patches_per_image=model.len_keep,
                        patches_per_image=fmap ** 2, schedule=schedule,
                        precision={"autocast": None,
                                   "cudnn_allow_tf32": cudnn.allow_tf32 if device.type == "cuda" else None,
                                   "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else None})
        write_json(run_info, run_dir / "run_info.json")

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

            model.train()  # BatchNorm uses batch statistics (visible positions only) and updates its running stats
            stats = train_one_epoch(model, make_loader(epoch), optimizer, epoch, args, device, schedule, timing)
            secs = time.time() - t0
            row = [epoch + 1] + [f"{stats[k]:.6g}" for k in HISTORY_COLUMNS[1:-1]] + [f"{secs:.1f}"]
            images_this_epoch = niter_per_ep * args.batch_size
            cost.update(epochs_done=epoch + 1, optimizer_steps=(epoch + 1) * niter_per_ep,
                        images_seen=(epoch + 1) * images_this_epoch,
                        train_minutes=round(cost.get("train_minutes", 0) + secs / 60, 2),
                        device=str(device) + gpu,
                        peak_gpu_memory_gb=(round(max(cost.get("peak_gpu_memory_gb", 0),
                                                      torch.cuda.max_memory_allocated(device) / 2**30), 2)
                                            if device.type == "cuda" else None))

            # The checkpoint also carries this epoch's history row and cost, so a stop before history.csv,
            # run_info.json or the export are written is repaired on resume (repair_after_stop).
            save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "args": vars(args),
                "global_step": (epoch + 1) * niter_per_ep,
                "schedule": schedule,
                "rng_state": rng_state(),
                "history_row": row,
                "cost": dict(cost),
            }
            tmp = run_dir / "checkpoint.pth.tmp"
            torch.save(save_dict, tmp)
            os.replace(tmp, run_dir / "checkpoint.pth")  # only replaced once the new file is complete

            with open(run_dir / "history.csv", "a", newline="") as f:
                csv.writer(f).writerow(row)
            write_json(run_info, run_dir / "run_info.json")
            print(f"  epoch {epoch + 1:3d}/{args.epochs} summary  loss {stats['loss']:.4f}  grad norm "
                  f"{stats['grad_norm']:.3f}  feature cosine {stats['feature_cosine']:.3f}  feature std "
                  f"{stats['feature_std']:.4f}  {secs / 60:.1f} min, checkpoint saved", flush=True)
            epoch_outputs(run_dir, epoch + 1, args, model, ssl_paths, device, run_info)

        done = cost.get("epochs_done", 0) >= args.epochs
        run_info.update(status="completed" if done else "stopped (resumable)",
                        finished=datetime.now().isoformat(timespec="seconds"))
        write_json(run_info, run_dir / "run_info.json")
        if done:
            final = run_dir / f"backbone_epoch{args.epochs:03d}.pth"
            print(f"\nDone. SSL cost: {cost['train_minutes']} min, {cost['images_seen']:,} images seen, "
                  f"{cost['optimizer_steps']:,} steps.\nOutputs in {run_dir}:\n"
                  f"  config.json, manifests.json, manifests/   settings and the image list used (with hashes)\n"
                  f"  run_info.json        status, environment, cost (time, images, steps, GPU memory)\n"
                  f"  history.csv, charts/ per-epoch loss, gradient norm, feature diagnostics, lr, wd;\n"
                  f"                       reconstruction_epoch<NNN>.png at every export\n"
                  f"  view_preview.png     original chips beside the views and masks used in training\n"
                  f"  param_groups.txt     which parameters get weight decay (the authors' grouping)\n"
                  f"  checkpoint.pth       full training state (model, optimizer, RNG)\n"
                  f"  {final.name:<20} FINAL backbone for fine-tuning (route A3)\n"
                  f"Next: python Experiments/mobileNetv2_SparK/train_supervised.py --backbone \"{final}\"", flush=True)
        else:
            print(f"\nStopped after epoch {cost.get('epochs_done')} of {args.epochs}. Continue with:\n"
                  f"  python Experiments/mobileNetv2_SparK/train_ssl.py --resume \"{run_dir}\"", flush=True)
        return run_dir
    except BaseException:
        run_info.update(status="failed", error=traceback.format_exc(limit=3))
        write_json(run_info, run_dir / "run_info.json")
        raise


def main():
    parser = argparse.ArgumentParser(description="SparK SSL training of MobileNetV2 on the SAR SSL pool")
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
