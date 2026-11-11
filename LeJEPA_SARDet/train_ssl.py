"""LeJEPA self-supervised training of MobileNetV2 on the SARDet-100K unlabeled pool (LeJEPA_SARDet, stage 1).

    python LeJEPA_SARDet/train_ssl.py                       # full run (config_ssl.json + shared/ssl_common.json)
    python LeJEPA_SARDet/train_ssl.py --smoke               # 48 images, 2 epochs: pipeline check only
    python LeJEPA_SARDet/train_ssl.py --resume <run folder> # continue after a stop
    python LeJEPA_SARDet/train_ssl.py --stop-after-epoch N  # stop resumably (CRC time limits)

net = LeJEPANet(MobileNetV2 features from the ImageNet file, MLP projector); ONE network, no teacher.
Per step: 8 global views of each image -> net -> projections (V, N, K);
          loss = (1 - lamb) * agreement(views vs. their mean) + lamb * SIGReg(projections);
          one AdamW step on everything (backbone + projector); no stop-gradient, no EMA, no centering.
After the last epoch the ENCODER backbone is exported (shared format). No labels are read.
The loop below is the ATRNet-STAR A2 loop (mobileNetv2_LeJEPA/train_ssl.py, train_one_epoch) on top of
shared/ssl_engine.py; the views come from the shared recipe instead of the old 128 px chips (CHANGES.md).
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
from shared.backbone import strip_prefix  # noqa: E402
from shared.run_record import sha256_of_text  # noqa: E402
from shared.ssl_data import to_network_input  # noqa: E402

sys.path.insert(0, str(HERE))
from lejepa_model import LeJEPANet, build_sigreg, collapse_diagnostics, lejepa_loss, lr_schedule  # noqa: E402


class LeJEPAMethod:
    name = "lejepa"
    history_columns = ["loss", "agreement", "sigreg", "proj_std", "proj_mean_abs", "feature_cosine", "feature_std", "lr"]
    loss_keys = ["loss", "agreement", "sigreg"]
    diag_keys = ["proj_std", "proj_mean_abs", "feature_cosine", "feature_std"]
    smoke_overrides = {"warmup_epochs": 1}
    export_rule = "encoder backbone (the single network; no teacher), final epoch"

    def __init__(self, args):
        errors = []
        if args.n_global_views < 2:
            errors.append("n_global_views must be >= 2 (the agreement term needs at least two views)")
        if not 0 <= args.warmup_epochs < args.epochs:
            errors.append(f"warmup_epochs ({args.warmup_epochs}) must be >= 0 and < epochs ({args.epochs})")
        if args.optimizer != "adamw":
            errors.append("optimizer must be adamw (LeJEPA README: AdamW for all models)")
        if args.precision not in ("bf16", "fp32"):
            errors.append('precision must be "bf16" or "fp32"')
        if args.n_points % 2 != 1:
            errors.append("n_points must be odd (the library's EppsPulley asserts it)")
        if errors:
            raise ValueError("invalid LeJEPA config:\n  " + "\n  ".join(errors))

    def view_plan(self, args):
        # All views are global (paper Section 5.1: for non-ViT backbones "set global_views = all_views").
        return [("global", args.n_global_views, args.blur_p)]

    def describe(self, args):
        return (f"  LeJEPA        : {args.n_global_views} global views (all global), projector {args.proj_hidden_dim}-"
                f"{args.proj_hidden_dim}-{args.proj_dim}, loss (1 - {args.lamb}) x agreement + {args.lamb} x SIGReg "
                f"({args.num_slices} directions, Epps-Pulley {args.n_points} points on [0, {args.t_max}])\n"
                f"  optimizer     : AdamW peak lr {args.lr} (not batch-scaled), warm-up {args.warmup_epochs} epoch(s) from "
                f"{args.warmup_start_factor} x lr, cosine to {self.lrs[-1]:.2e}, wd {args.weight_decay} on all params, "
                f"no clipping, {'bf16 autocast' if self.autocast_dtype else 'fp32'}")

    def build(self, args, rows, n_steps, device):
        self.args, self.device, self.n_steps = args, device, n_steps
        # >>> as in the ATRNet-STAR A2 train_ssl.py (run(): "Network, loss, optimizer, schedule")
        self.net = LeJEPANet(imagenet_init=True, proj_hidden_dim=args.proj_hidden_dim, proj_dim=args.proj_dim).to(device)
        self.sigreg = build_sigreg(args).to(device)
        # README.md / MINIMAL.md: AdamW, one parameter group, weight decay on all parameters, no gradient clipping
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.autocast_dtype = torch.bfloat16 if (args.precision == "bf16" and device.type == "cuda") else None
        self.lrs = lr_schedule(args.lr, args.epochs * n_steps, args.warmup_epochs * n_steps,
                               args.warmup_start_factor, args.final_lr_ratio)
        # <<<

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def schedule_fingerprint(self):
        return sha256_of_text(json.dumps(self.lrs))

    def train_epoch(self, loader, epoch, device, log_every, timing):
        # >>> ATRNet-STAR A2 train_one_epoch, with views arriving as uint8 from the shared loader
        args = self.args
        self.net.train()  # BatchNorm uses batch statistics and updates its running stats (exported with the backbone)
        sums, t0 = {}, time.time()
        n = len(loader)
        for it, (views, _idx) in enumerate(loader):
            step = it + 1
            git = n * epoch + it
            for group in self.optimizer.param_groups:
                group["lr"] = self.lrs[git]
            images = [to_network_input(v, device) for v in views]  # uint8 gray -> normalised RGB, on the GPU
            with torch.autocast(device.type, dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None):
                emb, proj = self.net(images)
            loss, agreement, sig = lejepa_loss(proj, args.n_global_views, self.sigreg, args.lamb)  # float32
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"loss is {loss.item()}, stopping")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            diag = collapse_diagnostics(emb, proj)
            for k, v in {"loss": loss.item(), "agreement": agreement.item(), "sigreg": sig.item(), **diag}.items():
                sums[k] = sums.get(k, 0.0) + v
            timing(step)
            if log_every and (step % log_every == 0 or step == n):
                print(f"    step {step:5d}/{n}  loss {sums['loss'] / step:.4f}  agreement {sums['agreement'] / step:.4f}  "
                      f"sigreg {sums['sigreg'] / step:.3f}  proj std {sums['proj_std'] / step:.3f}  "
                      f"lr {self.optimizer.param_groups[0]['lr']:.2e}  {time.time() - t0:.0f}s", flush=True)
        # <<<
        return {k: v / n for k, v in sums.items()} | {"lr": self.optimizer.param_groups[0]["lr"]}

    def state_dict(self):
        return {"model": self.net.state_dict(), "optimizer": self.optimizer.state_dict(),
                "sigreg": self.sigreg.state_dict(),            # its global_step seeds the random directions
                "autocast": str(self.autocast_dtype)}

    def load_state_dict(self, d):
        if d["autocast"] != str(self.autocast_dtype):
            raise ValueError("cannot resume: mixed precision on/off differs (device changed?)")
        self.net.load_state_dict(d["model"], strict=True)
        self.optimizer.load_state_dict(d["optimizer"])
        self.sigreg.load_state_dict(d["sigreg"], strict=True)

    def export_state(self):
        state = strip_prefix(self.net.state_dict(), "backbone.features.")
        return state, {"which": "encoder (single network)", "excluded_from_export": "projector, optimizer, SIGReg step"}


if __name__ == "__main__":
    ssl_engine.cli(LeJEPAMethod, HERE)
