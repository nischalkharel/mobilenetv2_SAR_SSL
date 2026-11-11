"""DINO self-supervised training of MobileNetV2 on the SARDet-100K unlabeled pool (DINO_SARDet, stage 1).

    python DINO_SARDet/train_ssl.py                       # full run (config_ssl.json + shared/ssl_common.json)
    python DINO_SARDet/train_ssl.py --smoke               # 48 images, 2 epochs: pipeline check only
    python DINO_SARDet/train_ssl.py --resume <run folder> # continue after a stop
    python DINO_SARDet/train_ssl.py --stop-after-epoch N  # stop resumably (CRC time limits)

student = MultiCropWrapper(MobileNetV2 features from the ImageNet file, DINO head); teacher = EMA copy.
Per step: 2 global views -> teacher, all views -> student, DINO loss, student step, teacher EMA.
After the last epoch the TEACHER backbone is exported (shared format). No labels are read.
The loop below is the ATRNet-STAR A1 loop (train_ssl.py, extracts E11-E13) on top of shared/ssl_engine.py.
"""
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared import ssl_engine  # noqa: E402
from shared.run_record import sha256_of_text  # noqa: E402
from shared.ssl_data import to_network_input  # noqa: E402

sys.path.insert(0, str(HERE))
from dino import (DINOHead, DINOLoss, MobileNetV2Backbone, MultiCropWrapper, cancel_gradients_last_layer,  # noqa: E402
                  clip_gradients, collapse_diagnostics, cosine_scheduler, get_params_groups)


class DINOMethod:
    name = "dino"
    history_columns = ["loss", "teacher_entropy", "teacher_marginal_entropy", "feature_cosine", "feature_std",
                       "lr", "wd", "momentum", "teacher_temp", "center_norm"]
    loss_keys = ["loss"]
    diag_keys = ["teacher_entropy", "teacher_marginal_entropy", "feature_cosine", "feature_std"]
    smoke_overrides = {"warmup_epochs": 1}
    export_rule = "teacher backbone, final epoch"

    def __init__(self, args):
        if len(args.blur_p) != 3:
            raise ValueError("blur_p needs 3 values: global view 1, global view 2, local views")

    def view_plan(self, args):
        return [("global", 1, args.blur_p[0]), ("global", 1, args.blur_p[1]), ("local", args.local_crops_number, args.blur_p[2])]

    def describe(self, args):
        return (f"  DINO          : out_dim {args.out_dim}, teacher momentum {args.momentum_teacher} -> 1, teacher temp "
                f"{args.teacher_temp}, student temp {args.student_temp}, center momentum {args.center_momentum}\n"
                f"  optimizer     : AdamW peak lr {self.peak_lr:.3g} (= {args.lr} x {args.batch_size}/256), warm-up "
                f"{args.warmup_epochs} epochs, cosine to {args.min_lr}, wd {args.weight_decay}, clip {args.clip_grad}, "
                f"fp16 {'on' if self.scaler else 'off'}")

    def build(self, args, rows, n_steps, device):
        self.args, self.device, self.n_steps = args, device, n_steps
        student = MobileNetV2Backbone(imagenet_init=True)
        teacher = MobileNetV2Backbone(imagenet_init=False)
        embed_dim = student.feature_dim
        # >>> as in the ATRNet-STAR A1 train_ssl.py (E11, E12)
        self.student = MultiCropWrapper(student, DINOHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head,
                                                          norm_last_layer=args.norm_last_layer)).to(device)
        self.teacher = MultiCropWrapper(teacher, DINOHead(embed_dim, args.out_dim, args.use_bn_in_head)).to(device)
        self.teacher.load_state_dict(self.student.state_dict())        # parameters AND BatchNorm buffers
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.loss = DINOLoss(args.out_dim, args.local_crops_number + 2, args.warmup_teacher_temp, args.teacher_temp,
                             args.warmup_teacher_temp_epochs, args.epochs, student_temp=args.student_temp,
                             center_momentum=args.center_momentum).to(device)
        groups = get_params_groups(self.student)
        self.optimizer = torch.optim.AdamW(groups) if args.optimizer == "adamw" else torch.optim.SGD(groups, lr=0, momentum=0.9)
        self.scaler = torch.amp.GradScaler("cuda") if (args.use_fp16 and device.type == "cuda") else None
        self.peak_lr = args.lr * args.batch_size / 256.0
        self.lr_schedule = cosine_scheduler(self.peak_lr, args.min_lr, args.epochs, n_steps, warmup_epochs=args.warmup_epochs)
        self.wd_schedule = cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, n_steps)
        self.momentum_schedule = cosine_scheduler(args.momentum_teacher, 1, args.epochs, n_steps)
        # <<<

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.student.parameters() if p.requires_grad)

    def schedule_fingerprint(self):
        return sha256_of_text(json.dumps([self.lr_schedule.tolist(), self.wd_schedule.tolist(), self.momentum_schedule.tolist(),
                                          self.loss.teacher_temp_schedule.tolist()]))

    def train_epoch(self, loader, epoch, device, log_every, timing):
        # >>> ATRNet-STAR A1 train_one_epoch (DINO extract E13), with views arriving as uint8 from the shared loader
        args = self.args
        self.student.train()
        self.teacher.train()       # the teacher is never .eval(): BN uses batch statistics, as in the original
        sums, t0 = {}, time.time()
        n = len(loader)
        for it, (views, _idx) in enumerate(loader):
            step = it + 1
            git = n * epoch + it
            for i, g in enumerate(self.optimizer.param_groups):
                g["lr"] = self.lr_schedule[git]
                if i == 0:
                    g["weight_decay"] = self.wd_schedule[git]
            images = [to_network_input(v, device) for v in views]
            center = self.loss.center.clone()
            with torch.autocast(device.type, dtype=torch.float16, enabled=self.scaler is not None):
                teacher_output = self.teacher(images[:2])
                student_output = self.student(images)
                loss = self.loss(student_output, teacher_output, epoch)
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
            diag = collapse_diagnostics(teacher_output, center, self.loss.teacher_temp_schedule[epoch],
                                        self.teacher.last_features, len(images[0]))
            for k, v in {"loss": loss.item(), **diag}.items():
                sums[k] = sums.get(k, 0.0) + v
            timing(step)
            if log_every and (step % log_every == 0 or step == n):
                print(f"    step {step:5d}/{n}  loss {sums['loss'] / step:.4f}  lr {self.optimizer.param_groups[0]['lr']:.2e}  "
                      f"teacher entropy {sums['teacher_entropy'] / step:.3f}  {time.time() - t0:.0f}s", flush=True)
        # <<<
        return {k: v / n for k, v in sums.items()} | {
            "lr": self.optimizer.param_groups[0]["lr"], "wd": self.optimizer.param_groups[0]["weight_decay"], "momentum": float(m),
            "teacher_temp": float(self.loss.teacher_temp_schedule[epoch]), "center_norm": self.loss.center.norm().item()}

    def state_dict(self):
        return {"student": self.student.state_dict(), "teacher": self.teacher.state_dict(), "optimizer": self.optimizer.state_dict(),
                "loss": self.loss.state_dict(), "scaler": self.scaler.state_dict() if self.scaler else None}

    def load_state_dict(self, d):
        self.student.load_state_dict(d["student"], strict=True)
        self.teacher.load_state_dict(d["teacher"], strict=True)
        self.optimizer.load_state_dict(d["optimizer"])
        self.loss.load_state_dict(d["loss"], strict=True)
        if (d["scaler"] is None) != (self.scaler is None):
            raise ValueError("cannot resume: mixed precision on/off differs (device changed?)")
        if self.scaler:
            self.scaler.load_state_dict(d["scaler"])

    def export_state(self):
        prefix = "backbone.features."
        state = {k[len(prefix):]: v for k, v in self.teacher.state_dict().items() if k.startswith(prefix)}
        return state, {"which": "teacher", "excluded_from_export": "DINO head, student, optimizer, center"}


if __name__ == "__main__":
    ssl_engine.cli(DINOMethod, HERE)
