"""RegionDINO: DINO applied to scenes AND to the things in them (or to the things only).

Two methods share this class (DESIGN.md):
    mode "scene_object"  loss = scene DINO loss (unchanged: 2 global + n local crops of the whole image)
                                + object_weight x object DINO loss
    mode "object"        loss = object DINO loss only (no whole-image views at all)
The object DINO loss treats every proposed region as a sample of its own: its 2 object-global views are the
teacher views, its 2 object-global + m object-local views are the student views, same head, same DINO
cross-entropy, its own centering buffer. Nothing else changes: same student / teacher backbones (MobileNetV2
from ImageNet), same head, same optimizer and schedules, same EMA, same export (teacher backbone).

All views of a step go through ONE student call and ONE teacher call (MultiCropWrapper batches same-size
views together), then the outputs are split back into the scene part and the object part.
"""
import json
import math
import time

import torch

from shared.run_record import sha256_of_text                      # SARDet suite, read-only
from shared.ssl_data import load_pool_rows, to_network_input       # noqa: F401
from dino import (DINOHead, DINOLoss, MobileNetV2Backbone, MultiCropWrapper, cancel_gradients_last_layer,  # read-only
                  clip_gradients, collapse_diagnostics, cosine_scheduler, get_params_groups)

from . import regions as R
from .region_data import RegionViewDataset, region_box_diagnostic

OBJECT_KEYS = ("objects_per_image", "obj_global_size", "obj_local_size", "obj_local_number", "obj_global_zoom", "obj_local_zoom",
               "min_object_px", "max_magnification", "region_source", "region_sampling", "obj_blur_p")


class RegionDINOMethod:
    name = "regiondino"
    mode = "scene_object"                      # overridden by the two train_ssl.py files
    loss_keys = ["loss", "scene_loss", "object_loss"]
    diag_keys = ["teacher_entropy", "teacher_marginal_entropy", "feature_cosine", "feature_std"]
    smoke_overrides = {"warmup_epochs": 1}
    export_rule = "teacher backbone, final epoch"

    def __init__(self, args):
        if self.mode not in ("scene_object", "object", "scene"):
            raise ValueError(self.mode)
        if self.mode in ("scene_object", "scene") and len(args.blur_p) != 3:
            raise ValueError("blur_p needs 3 values: global view 1, global view 2, local views")
        if self.mode != "scene":
            if len(args.obj_blur_p) != 3:
                raise ValueError("obj_blur_p needs 3 values: object-global 1, object-global 2, object-local views")
            if args.region_source not in ("felz", "random", "fixed"):
                raise ValueError("region_source must be felz (the proposer), random (the control) or fixed (the view-size sweep)")
            if args.region_source == "fixed" and not getattr(args, "fixed_extent_px", None):
                raise ValueError("region_source fixed needs fixed_extent_px (window extent in pixels of the working image)")
        self.pool_name = getattr(args, "pool", "sardet")
        self.history_columns = ["loss", "scene_loss", "object_loss", "teacher_entropy", "teacher_marginal_entropy", "feature_cosine",
                                "feature_std", "fallback_rate", "lr", "wd", "momentum", "teacher_temp", "center_norm", "object_center_norm"]

    # ---- data ----------------------------------------------------------------------------------------------
    def scene_plan(self, args):
        if self.mode == "object":
            return []
        return [("global", 1, args.blur_p[0]), ("global", 1, args.blur_p[1]), ("local", args.local_crops_number, args.blur_p[2])]

    def make_dataset(self, rows, args, smoke):
        from . import pools as P
        pool = P.get(self.pool_name)
        if self.mode == "scene":                   # plain DINO of this engine: scene views only, no region file
            self.store = None
            ds = RegionViewDataset(rows, self.scene_plan(args), None, {"objects_per_image": 0}, seed=args.seed, pool=pool)
            self.n_scene, self.n_obj_slots, self.K = sum(c for _, c, _ in ds.scene_plan), 0, 0
            ds.manifest_info = {"pool": pool.name, "sha256": None, "region_source": "none", "object_views": None}
            return ds
        self.store = R.RegionStore(pool.regions_file(smoke), scale=pool.region_scale)
        opts = {k: getattr(args, k) for k in OBJECT_KEYS}
        opts["fixed_extent_px"] = getattr(args, "fixed_extent_px", None)      # only the ViewScale configs set it
        ds = RegionViewDataset(rows, self.scene_plan(args), self.store, opts, seed=args.seed, pool=pool)
        self.n_scene = len(ds.scene_plan) and sum(c for _, c, _ in ds.scene_plan)
        self.n_obj_slots = sum(c for _, c, _ in ds.obj_slots)
        self.K = ds.K
        coverage = self.store.coverage(rows)
        if coverage["pool_images_missing"] and not smoke:
            raise FileNotFoundError(f"{coverage['pool_images_missing']} of {len(rows)} pool images have no entry in {self.store.path}; "
                                    "run RegionSSL/make_regions.py over the whole pool (a partial file would silently turn this run into the random control)")
        diag = region_box_diagnostic(rows, self.store, args.min_object_px, n_images=min(300, len(rows)), pool=pool)
        # plain data on the dataset (no closures: the dataset is pickled to the loader workers)
        ds.manifest_info = {**coverage, "pool": pool.name, "box_scale_from_proposal_frame": pool.region_scale,
                            "summary_file": str(self.store.path.with_name(self.store.path.stem + "_summary.json")),
                            "region_source": args.region_source, "diagnostic_with_labels": diag, "object_views": opts}
        return ds

    def describe(self, args):
        if self.mode == "scene":
            region = f"  RegionDINO    : mode scene (plain DINO of this engine, no object views); pool {self.pool_name}\n"
        else:
            region = (f"  RegionDINO    : mode {self.mode}; pool {self.pool_name}; {self.K} objects per image, each 2 object-global ({args.obj_global_size}px, zoom "
                      f"{args.obj_global_zoom}) + {args.obj_local_number} object-local ({args.obj_local_size}px, zoom {args.obj_local_zoom}); "
                      f"regions: {args.region_source}, min {args.min_object_px}px, sampling {args.region_sampling}; object loss weight "
                      f"{args.object_weight if self.mode == 'scene_object' else 'only term'}\n")
        return (region +
                f"  DINO          : out_dim {args.out_dim}, teacher momentum {args.momentum_teacher} -> 1, teacher temp {args.teacher_temp}, "
                f"student temp {args.student_temp}, center momentum {args.center_momentum}\n"
                f"  optimizer     : AdamW peak lr {self.peak_lr:.3g} (= {args.lr} x {args.batch_size}/256), warm-up {args.warmup_epochs} epochs, "
                f"cosine to {args.min_lr}, wd {args.weight_decay}, clip {args.clip_grad}, fp16 {'on' if self.scaler else 'off'}")

    # ---- networks, losses, optimizer (the DINO_SARDet build, plus a second loss instance for objects) -------
    def build(self, args, rows, n_steps, device):
        self.args, self.device, self.n_steps = args, device, n_steps
        student = MobileNetV2Backbone(imagenet_init=True)
        teacher = MobileNetV2Backbone(imagenet_init=False)
        embed_dim = student.feature_dim
        self.student = MultiCropWrapper(student, DINOHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head,
                                                          norm_last_layer=args.norm_last_layer)).to(device)
        self.teacher = MultiCropWrapper(teacher, DINOHead(embed_dim, args.out_dim, args.use_bn_in_head)).to(device)
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False
        common = dict(warmup_teacher_temp=args.warmup_teacher_temp, teacher_temp=args.teacher_temp,
                      warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs, nepochs=args.epochs,
                      student_temp=args.student_temp, center_momentum=args.center_momentum)
        self.scene_loss = DINOLoss(args.out_dim, self.n_scene, **common).to(device) if self.mode in ("scene_object", "scene") else None
        self.object_loss = DINOLoss(args.out_dim, self.n_obj_slots, **common).to(device) if self.mode != "scene" else None
        self.any_loss = self.object_loss if self.object_loss is not None else self.scene_loss
        groups = get_params_groups(self.student)
        self.optimizer = torch.optim.AdamW(groups) if args.optimizer == "adamw" else torch.optim.SGD(groups, lr=0, momentum=0.9)
        self.scaler = torch.amp.GradScaler("cuda") if (args.use_fp16 and device.type == "cuda") else None
        self.peak_lr = args.lr * args.batch_size / 256.0
        self.lr_schedule = cosine_scheduler(self.peak_lr, args.min_lr, args.epochs, n_steps, warmup_epochs=args.warmup_epochs)
        self.wd_schedule = cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, n_steps)
        self.momentum_schedule = cosine_scheduler(args.momentum_teacher, 1, args.epochs, n_steps)

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.student.parameters() if p.requires_grad)

    def schedule_fingerprint(self):
        return sha256_of_text(json.dumps([self.lr_schedule.tolist(), self.wd_schedule.tolist(), self.momentum_schedule.tolist(),
                                          self.any_loss.teacher_temp_schedule.tolist()]))

    # ---- one epoch -----------------------------------------------------------------------------------------
    def train_epoch(self, loader, epoch, device, log_every, timing):
        args = self.args
        self.student.train()
        self.teacher.train()
        sums, t0 = {}, time.time()
        n = len(loader)
        for it, ((scene, objects, meta), _idx) in enumerate(loader):
            step = it + 1
            git = n * epoch + it
            for i, g in enumerate(self.optimizer.param_groups):
                g["lr"] = self.lr_schedule[git]
                if i == 0:
                    g["weight_decay"] = self.wd_schedule[git]
            scene_in = [to_network_input(v, device) for v in scene]
            obj_in = [to_network_input(v, device) for v in objects]
            B = scene_in[0].shape[0] if scene_in else obj_in[0].shape[0] // self.K
            n_scene_rows = self.n_scene * B
            center_s = self.scene_loss.center.clone() if self.scene_loss is not None else None
            center_o = self.object_loss.center.clone() if self.object_loss is not None else None
            with torch.autocast(device.type, dtype=torch.float16, enabled=self.scaler is not None):
                # teacher: scene globals (2 x B rows) then object globals (2 x B*K rows); student: everything
                t_out = self.teacher(scene_in[:2] + obj_in[:2])
                s_out = self.student(scene_in + obj_in)
                n_teacher_scene = 2 * B if scene_in else 0
                t_scene, t_obj = t_out[:n_teacher_scene], t_out[n_teacher_scene:]
                s_scene, s_obj = s_out[:n_scene_rows], s_out[n_scene_rows:]
                obj_loss = self.object_loss(s_obj, t_obj, epoch) if self.object_loss is not None else torch.zeros((), device=device)
                if self.scene_loss is not None:
                    sc_loss = self.scene_loss(s_scene, t_scene, epoch)
                    loss = sc_loss + args.object_weight * obj_loss if self.object_loss is not None else sc_loss
                else:
                    sc_loss = torch.zeros((), device=device)
                    loss = obj_loss
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"loss is {loss.item()}, stopping")
            self.optimizer.zero_grad()
            if self.scaler is None:
                loss.backward()
                if args.clip_grad:
                    clip_gradients(self.student, args.clip_grad)
                cancel_gradients_last_layer(epoch, self.student, args.freeze_last_layer)
                self.optimizer.step()
            else:
                self.scaler.scale(loss).backward()
                if args.clip_grad:
                    self.scaler.unscale_(self.optimizer)
                    clip_gradients(self.student, args.clip_grad)
                cancel_gradients_last_layer(epoch, self.student, args.freeze_last_layer)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            with torch.no_grad():
                m = self.momentum_schedule[git]
                for pq, pk in zip(self.student.parameters(), self.teacher.parameters()):
                    pk.data.mul_(m).add_((1 - m) * pq.detach().data)
            # collapse diagnostics on the teacher's scene views (scene_object) or object-global views (object)
            feats = self.teacher.last_features
            if self.scene_loss is not None:
                diag = collapse_diagnostics(t_scene, center_s, self.scene_loss.teacher_temp_schedule[epoch], feats[:2 * B], B)
            else:
                diag = collapse_diagnostics(t_obj, center_o, self.object_loss.teacher_temp_schedule[epoch], feats, B * self.K)
            stats = {"loss": loss.item(), "scene_loss": sc_loss.item(), "object_loss": obj_loss.item(), **diag,
                     "fallback_rate": meta[:, 1].float().mean().item()}
            for k, v in stats.items():
                sums[k] = sums.get(k, 0.0) + v
            timing(step)
            if log_every and (step % log_every == 0 or step == n):
                print(f"    step {step:5d}/{n}  loss {sums['loss'] / step:.4f} (scene {sums['scene_loss'] / step:.4f}, object "
                      f"{sums['object_loss'] / step:.4f})  lr {self.optimizer.param_groups[0]['lr']:.2e}  teacher entropy "
                      f"{sums['teacher_entropy'] / step:.3f}  {time.time() - t0:.0f}s", flush=True)
        return {k: v / n for k, v in sums.items()} | {
            "lr": self.optimizer.param_groups[0]["lr"], "wd": self.optimizer.param_groups[0]["weight_decay"], "momentum": float(m),
            "teacher_temp": float(self.any_loss.teacher_temp_schedule[epoch]),
            "center_norm": self.scene_loss.center.norm().item() if self.scene_loss is not None else 0.0,
            "object_center_norm": self.object_loss.center.norm().item() if self.object_loss is not None else 0.0}

    # ---- resume / export -----------------------------------------------------------------------------------
    def state_dict(self):
        return {"student": self.student.state_dict(), "teacher": self.teacher.state_dict(), "optimizer": self.optimizer.state_dict(),
                "scene_loss": self.scene_loss.state_dict() if self.scene_loss is not None else None,
                "object_loss": self.object_loss.state_dict() if self.object_loss is not None else None,
                "scaler": self.scaler.state_dict() if self.scaler else None}

    def load_state_dict(self, d):
        self.student.load_state_dict(d["student"], strict=True)
        self.teacher.load_state_dict(d["teacher"], strict=True)
        self.optimizer.load_state_dict(d["optimizer"])
        if (d["scene_loss"] is None) != (self.scene_loss is None) or (d["object_loss"] is None) != (self.object_loss is None):
            raise ValueError("cannot resume: the checkpoint is from another RegionDINO mode")
        if self.scene_loss is not None:
            self.scene_loss.load_state_dict(d["scene_loss"], strict=True)
        if self.object_loss is not None:
            self.object_loss.load_state_dict(d["object_loss"], strict=True)
        if (d["scaler"] is None) != (self.scaler is None):
            raise ValueError("cannot resume: mixed precision on/off differs (device changed?)")
        if self.scaler:
            self.scaler.load_state_dict(d["scaler"])

    def export_state(self):
        prefix = "backbone.features."
        state = {k[len(prefix):]: v for k, v in self.teacher.state_dict().items() if k.startswith(prefix)}
        return state, {"which": "teacher", "mode": self.mode, "excluded_from_export": "DINO head, student, optimizer, centers"}
