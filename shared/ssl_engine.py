"""The parts of an SSL run that are the same for every method (DESIGN.md section 4).

A method (DINO, LeJEPA, SparK, CondDINO) implements the small Method interface below; the engine does the
rest identically for all of them: config merging with shared/ssl_common.json, the audited SSL pool, the
shared view loader, run folder + records + manifests, per-epoch checkpoint and exact resume, history.csv and
charts, cost accounting (images, views, GPU-hours, peak memory), the box-coverage diagnostic of the views,
and the export of the final-epoch backbone in the shared format.

Method interface (duck-typed; see DINO_SARDet/train_ssl.py for the reference implementation):
    name                         str, e.g. "dino"
    history_columns              list of per-epoch stat names written to history.csv (after "epoch")
    loss_keys / diag_keys        which columns go on the loss chart / the diagnostics chart
    view_plan(args)              -> [(size_key, count, blur_p), ...]   (sizes come from the shared config)
    smoke_overrides              dict of flat-arg overrides applied in smoke mode (e.g. warmup epochs)
    build(args, rows, n_steps_per_epoch, device)   networks, loss, optimizer, schedules
    schedule_fingerprint()       str; checked on resume
    train_epoch(loader, epoch, device, log_every, timing)  -> dict of stats (must include every history column);
                                 call timing(step) once per step (the engine measures speed from it)
    state_dict() / load_state_dict(d)              everything needed to resume exactly
    export_state()               -> (features state dict with torchvision keys, extra metadata dict)
    describe(args)               -> str printed at the start
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
from pathlib import Path

import numpy as np
import torch

from . import manifests, paths
from .backbone import export_backbone, imagenet_weights_record
from .plots import ssl_curves
from .run_record import RunRecord, make_run_dir, read_json, sha256_of_file, sha256_of_text, write_json
from .ssl_data import build_ssl_loader, load_common_config, load_pool_rows, SSLPoolDataset, view_box_coverage

SMOKE_IMAGES = 48
SMOKE_COMMON = {"epochs": 2, "batch_size": 8, "num_workers": 0}
TIMING_FROM, TIMING_TO = 5, 25
RESUME_MAY_DIFFER = {"log_every_steps", "device"}


def flat_args(cfg: dict) -> argparse.Namespace:
    """Merge every section of a config into one flat namespace; a key in two sections is an error."""
    flat = {}
    for section, values in cfg.items():
        if section.startswith("_") or section in ("run_name", "method"):
            continue
        if not isinstance(values, dict):
            if section in flat:
                raise KeyError(f"config key {section!r} appears twice")
            flat[section] = values
            continue
        for k, v in values.items():
            if k.startswith("_"):
                continue
            if k in flat:
                raise KeyError(f"config key {k!r} appears twice (in {section!r} and earlier)")
            flat[k] = v
    return argparse.Namespace(**flat)


def merged_config(method_cfg: dict, smoke: bool, smoke_overrides: dict) -> dict:
    """shared/ssl_common.json + the method's config (which may not redefine a shared key)."""
    common = load_common_config()
    cfg = {"run_name": method_cfg["run_name"], "method": method_cfg.get("method"),
           "common": {k: v for k, v in common.items() if k != "views"}, "views": dict(common["views"])}
    shared_keys = set(cfg["common"]) | set(cfg["views"])
    for section, values in method_cfg.items():
        if section in ("run_name", "method") or section.startswith("_"):
            continue
        clash = shared_keys & (set(values) if isinstance(values, dict) else {section})
        if clash:
            raise KeyError(f"method config may not override shared SSL keys: {sorted(clash)}")
        cfg[section] = values
    cfg["common"]["seed"] = paths.SPLIT - 1
    if smoke:
        cfg["common"].update(SMOKE_COMMON)
        for k, v in smoke_overrides.items():
            for section in cfg.values():
                if isinstance(section, dict) and k in section:
                    section[k] = v
    cfg["smoke"] = smoke
    return cfg


def rng_state():
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def restore_rng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


@contextlib.contextmanager
def preserved_rng():
    s = rng_state()
    yield
    restore_rng(s)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(method_cls, experiment_dir, config_path=None, resume_dir=None, smoke=False, stop_after_epoch=None,
        device=None, log_every_steps=100):
    """Train (or resume) one SSL run of `method_cls`; returns the run folder."""
    experiment_dir = Path(experiment_dir).resolve()
    if resume_dir is not None:
        run_dir = Path(resume_dir)
        cfg = read_json(run_dir / "config.json")
        smoke = cfg["smoke"]
    else:
        method_cfg = read_json(config_path or experiment_dir / "config_ssl.json")
        cfg = merged_config(method_cfg, smoke, getattr(method_cls, "smoke_overrides", {}))
    args = flat_args(cfg)
    args.log_every_steps = log_every_steps
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    method = method_cls(args)

    # ---- data (audited, no labels) --------------------------------------------------------------------------
    audit = manifests.quick_audit()
    rows = load_pool_rows()
    if smoke:
        rows = rows[:: max(1, len(rows) // SMOKE_IMAGES)][:SMOKE_IMAGES]
    used_hash = sha256_of_text("\n".join(r["file_name"] for r in rows))
    manifest = {"ssl_pool_csv": str(manifests.ssl_pool_csv()), "csv_sha256": audit["manifest_sha256"]["ssl_pool.csv"],
                "used_file_list_sha256": used_hash, "num_images": len(rows), "split_audit": audit,
                "columns_available_to_the_method": manifests.MANIFEST_COLUMNS, "labels_used": "none",
                "labels_read_for_diagnostic_only": "view_box_coverage: GT boxes of up to 300 pool images, logged in run_info, never seen by training"}
    plan = method.view_plan(args)
    dataset = SSLPoolDataset(rows, plan, seed=args.seed)
    n_steps = len(dataset) // args.batch_size
    if n_steps < 1:
        raise ValueError(f"batch {args.batch_size} larger than the {len(dataset)} images")
    n_views = sum(c for _, c, _ in plan)

    # ---- run folder and records ----------------------------------------------------------------------------
    if resume_dir is None:
        root = paths.experiment_output_dir(experiment_dir) / ("smoke" if smoke else "ssl")
        run_dir = make_run_dir(root, f"{cfg['run_name']}{paths.RUN_TAG}_seed{args.seed}{'_SMOKE' if smoke else ''}")
        write_json(run_dir / "config.json", cfg)
        (run_dir / "manifests").mkdir()
        shutil.copy(manifests.ssl_pool_csv(), run_dir / "manifests" / "ssl_pool.csv")
        write_json(run_dir / "manifests.json", manifest)
        record = RunRecord(run_dir, "ssl", cfg, extra={
            "method": method.name, "initialization": {"kind": "imagenet", **imagenet_weights_record()},
            "checkpoint_rule": "final epoch, fixed in advance; the SSL loss never selects a checkpoint",
            "export_rule": getattr(method, "export_rule", "final epoch"), "views_per_image": n_views,
            "view_plan": plan, "cost": {}}).start()
        with preserved_rng():
            record.update(view_box_coverage=view_box_coverage(rows, n_images=min(300, len(rows))))
        with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch"] + method.history_columns + ["seconds"])
    else:
        recorded = read_json(run_dir / "manifests.json")
        for k in ("csv_sha256", "used_file_list_sha256", "num_images"):
            if recorded[k] != manifest[k]:
                raise ValueError(f"cannot resume: the SSL manifest changed since the run started ({k})")
        record = RunRecord(run_dir, "ssl", cfg)
        record.info = read_json(run_dir / "run_info.json")
        record.info["resumed"] = record.info.get("resumed", []) + [time.strftime("%Y-%m-%dT%H:%M:%S")]
        record.update(status="running")

    method.run_dir = run_dir          # methods may write per-epoch files (previews, cell stats) here
    try:
        set_seed(args.seed)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        if args.num_workers > 0:
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ[var] = "1"
        method.build(args, rows, n_steps, device)
        fingerprint = method.schedule_fingerprint()
        start_epoch = 0
        if resume_dir is not None:
            ckpt = torch.load(run_dir / "checkpoint.pth", map_location="cpu", weights_only=False)
            diff = {k for k in set(ckpt["args"]) | set(vars(args)) if ckpt["args"].get(k) != getattr(args, k, None)} - RESUME_MAY_DIFFER
            if diff:
                raise ValueError(f"cannot resume: settings differ in {sorted(diff)}")
            if ckpt["schedule_fingerprint"] != fingerprint:
                raise ValueError("cannot resume: the rebuilt schedules differ from the saved ones")
            method.load_state_dict(ckpt["method"])
            start_epoch = ckpt["epoch"]
            restore_rng(ckpt["rng_state"])
            # a kill between the checkpoint and the history append loses that epoch's row: repair it
            with open(run_dir / "history.csv", newline="", encoding="utf-8") as f:
                last = list(csv.reader(f))[-1]
            if last[0] != str(start_epoch) and "history_row" in ckpt:
                with open(run_dir / "history.csv", "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(ckpt["history_row"])
                print(f"  repaired history.csv row for epoch {start_epoch}")
            if start_epoch >= args.epochs:
                out = run_dir / f"backbone_epoch{args.epochs}.pth"
                try:
                    torch.load(out, map_location="cpu", weights_only=True)
                    print("run already complete; nothing to resume")
                    record.update(status="completed")
                    return run_dir
                except Exception as e:  # noqa: BLE001 - killed between the last checkpoint and the export
                    print(f"run complete but the exported backbone is missing or unreadable ({e}); exporting now")
                    return _export_and_finish(method, run_dir, args, cfg, manifest, record.info.get("cost", {}), record)
        record.update(schedule_fingerprint=fingerprint, steps_per_epoch=n_steps, total_steps=n_steps * args.epochs,
                      num_trainable_parameters=method.num_trainable_parameters())
        print(f"\n{'SMOKE RUN (pipeline check only)' if smoke else method.name.upper() + ' SSL run'}: {cfg['run_name']}"
              f"{'  (resumed at epoch ' + str(start_epoch + 1) + ')' if start_epoch else ''}\n"
              f"  output folder : {run_dir}\n  device        : {device}\n  SSL images    : {len(rows)} (no labels)\n"
              f"  batch         : {args.batch_size} images x {n_views} views ({plan})\n"
              f"  steps         : {n_steps} per epoch, {args.epochs} epochs\n  seed          : {args.seed} (split {paths.SPLIT})\n"
              f"{method.describe(args)}", flush=True)

        cost = record.info["cost"]
        last_epoch = args.epochs if stop_after_epoch is None else min(args.epochs, stop_after_epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(start_epoch, last_epoch):
            print(f"\nSSL epoch {epoch + 1}/{args.epochs}", flush=True)
            t0 = time.time()
            marks = {}

            t_from, t_to = min(TIMING_FROM, max(1, n_steps - 1)), min(TIMING_TO, n_steps)   # smoke runs have few steps

            def timing(step):
                if epoch != start_epoch or step not in (t_from, t_to) or t_to <= t_from:
                    return
                marks[step] = time.time()
                if step == t_to:
                    s = (marks[t_to] - marks[t_from]) / (t_to - t_from)
                    print(f"  measured speed: {s:.3f} s/step -> {s * n_steps / 60:.1f} min per epoch, "
                          f"about {s * n_steps * (last_epoch - epoch) / 3600:.2f} h until epoch {last_epoch}", flush=True)
                    record.update(measured_seconds_per_step=round(s, 4),
                                  estimated_hours_for_all_epochs=round(s * n_steps * args.epochs / 3600, 2))

            loader = build_ssl_loader(dataset, args.batch_size, args.num_workers, args.seed, epoch)
            stats = method.train_epoch(loader, epoch, device, args.log_every_steps, timing)
            secs = time.time() - t0
            history_row = [epoch + 1] + [f"{stats[k]:.6g}" for k in method.history_columns] + [f"{secs:.1f}"]
            ckpt = {"method": method.state_dict(), "epoch": epoch + 1, "args": vars(args), "global_step": (epoch + 1) * n_steps,
                    "schedule_fingerprint": fingerprint, "rng_state": rng_state(), "history_row": history_row}
            tmp = run_dir / "checkpoint.pth.tmp"
            torch.save(ckpt, tmp)
            os.replace(tmp, run_dir / "checkpoint.pth")
            with open(run_dir / "history.csv", "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(history_row)
            images_epoch = n_steps * args.batch_size
            cost.update(epochs_done=epoch + 1, optimizer_steps=(epoch + 1) * n_steps, images_seen=(epoch + 1) * images_epoch,
                        view_forwards=(epoch + 1) * images_epoch * n_views,
                        train_minutes=round(cost.get("train_minutes", 0) + secs / 60, 2),
                        gpu_hours=round((cost.get("train_minutes", 0) + secs / 60) / 60, 3),
                        peak_gpu_memory_gb=(round(max(cost.get("peak_gpu_memory_gb") or 0, torch.cuda.max_memory_allocated(device) / 2**30), 2)
                                            if device.type == "cuda" else None))
            record.update(cost=cost)
            print(f"  epoch {epoch + 1:3d}/{args.epochs}  " + "  ".join(f"{k} {stats[k]:.4g}" for k in method.history_columns[:4])
                  + f"  {secs / 60:.1f} min, checkpoint saved", flush=True)
            try:
                (run_dir / "charts").mkdir(exist_ok=True)
                ssl_curves(run_dir / "history.csv", run_dir / "charts" / "ssl_curves.png", method.loss_keys, method.diag_keys)
            except Exception as e:  # noqa: BLE001 - charts must never stop training
                print(f"  WARNING: chart failed: {e}", flush=True)

        done = cost.get("epochs_done", 0) >= args.epochs
        if not done:
            record.update(status="stopped (resumable)")
            print(f"\nstopped after epoch {cost.get('epochs_done')} of {args.epochs}; continue with --resume \"{run_dir}\"")
            return run_dir
        return _export_and_finish(method, run_dir, args, cfg, manifest, cost, record)
    except BaseException as e:
        record.fail(e)
        raise


def _export_and_finish(method, run_dir, args, cfg, manifest, cost, record):
    features_state, extra = method.export_state()
    out = run_dir / f"backbone_epoch{args.epochs}.pth"
    meta = export_backbone(features_state, out, {
        "ssl_method": method.name, "ssl_run_dir": str(run_dir), "ssl_epoch": args.epochs, "ssl_epochs_planned": args.epochs,
        "ssl_seed": args.seed, "split": paths.SPLIT, "ssl_config": cfg, "ssl_manifest": {k: v for k, v in manifest.items() if k != "split_audit"},
        "ssl_cost": cost, "initialization_before_ssl": record.info["initialization"], "exported": extra})
    record.finish(exported_backbone={"file": out.name, **{k: meta[k] for k in ("state_sha256", "file_sha256", "num_tensors")}})
    print(f"\ndone: {cost.get('train_minutes')} min, {cost.get('images_seen', 0):,} images, {cost.get('optimizer_steps', 0):,} steps; "
          f"exported {out}")
    return run_dir


def cli(method_cls, experiment_dir, argv=None):
    """Standard command line for every method's train_ssl.py."""
    p = argparse.ArgumentParser(description=f"{method_cls.name} SSL on the SARDet-100K pool")
    p.add_argument("--config", type=Path, help="method config (default: <experiment>/config_ssl.json)")
    p.add_argument("--resume", type=Path, help="run folder to continue from its checkpoint.pth")
    p.add_argument("--stop-after-epoch", type=int)
    p.add_argument("--smoke", action="store_true", help=f"{SMOKE_IMAGES} images, batch 8, 2 epochs; pipeline check only")
    p.add_argument("--device")
    p.add_argument("--log-every-steps", type=int, default=100)
    a = p.parse_args(argv)
    return run(method_cls, experiment_dir, a.config, a.resume, a.smoke, a.stop_after_epoch, a.device, a.log_every_steps)
