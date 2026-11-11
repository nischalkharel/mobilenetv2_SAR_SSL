"""SparK self-supervised training of MobileNetV2 on the SARDet-100K unlabeled pool (SparK_SARDet, stage 1).

    python SparK_SARDet/train_ssl.py                       # full run (config_ssl.json + shared/ssl_common.json)
    python SparK_SARDet/train_ssl.py --smoke               # 48 images, 2 epochs: pipeline check only
    python SparK_SARDet/train_ssl.py --resume <run folder> # continue after a stop
    python SparK_SARDet/train_ssl.py --stop-after-epoch N  # stop resumably (CRC time limits)

model = SparK(sparse MobileNetV2 encoder from the ImageNet file, densify layers + mask tokens, LightDecoder).
Per step: ONE 224 px view of each image (the shared recipe, blur probability 0); a random 7 x 7 patch mask
(60 % hidden); the encoder sees only the visible patches (sparse conv / BN); the decoder reconstructs the
hidden patches; loss = mean squared error on the hidden patches of the single SAR channel, pixels normalised
per patch; one LAMB step on everything (encoder + densify layers + mask tokens + decoder), full float32 with
TF32 off. After the last epoch the ENCODER (MobileNetV2 features) is exported in the shared format.
No labels are read. The loop below is the ATRNet-STAR A3 loop (mobileNetv2_SparK/train_ssl.py,
train_one_epoch) on top of shared/ssl_engine.py; CHANGES.md lists every difference.
"""
import contextlib
import io
import json
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
from spark_lib.lamb import TheSameAsTimmLAMB  # noqa: E402
from spark_lib.lr_control import get_param_groups, lr_wd_annealing  # noqa: E402
from spark_model import (DOWNSAMPLE_RATIO, build_spark, count_parameters, encoder_features_state,  # noqa: E402
                         feature_diagnostics, feature_shapes, preview_batch, save_reconstruction_preview)

NOWD_KEYS = {"cls_token", "pos_embed", "mask_token", "gamma"}  # pretrain/main.py L75 (only mask_token exists here)


def use_true_fp32() -> dict:
    """Full float32 on the GPU: switch off TF32, which PyTorch uses by default for float32 convolutions on
    Ampere and newer GPUs (A100, RTX 30/40; 10 mantissa bits). Measured on ATRNet-STAR at the start of SparK
    training (CHANGES.md, "Precision"): against a float64 reference the gradient error is about 100 % with TF32
    (the encoder's gradient is mostly rounding noise) and about 11 % without it (CPU float32: 3 %). The authors'
    code has no mixed precision and the paper mentions V100 GPUs, which have no TF32. Returns the flags."""
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return precision_flags()


def precision_flags() -> dict:
    return {"autocast": None, "dtype": "float32",
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}


class SparKMethod:
    name = "spark"
    history_columns = ["loss", "feature_cosine", "feature_std", "grad_norm", "lr", "wd"]
    loss_keys = ["loss"]
    diag_keys = ["feature_cosine", "feature_std", "grad_norm"]
    smoke_overrides = {"warmup_epochs": 0.5}   # 2.5 of 100 epochs would exceed the 2-epoch smoke run
    export_rule = "encoder (MobileNetV2 features), final epoch"

    def __init__(self, args):
        errors = []
        if args.densify_norm != "bn":
            errors.append('densify_norm must be "bn" (the authors\' setting for BatchNorm CNNs; arg_util.py L123-L125)')
        if args.recon_channels not in (1, 3):
            errors.append("recon_channels must be 1 (SAR channel) or 3 (the authors' literal target)")
        if not 0 < args.mask_ratio < 1:
            errors.append("mask_ratio must be in (0, 1)")
        if args.global_size % DOWNSAMPLE_RATIO:
            errors.append("global_size must be a multiple of 32 (MobileNetV2's total downsampling = the mask patch size)")
        if args.decoder_width < 32 or args.decoder_width % 32:
            errors.append("decoder_width must be a multiple of 32 (the decoder halves its width 5 times)")
        if args.optimizer != "lamb":
            errors.append("optimizer must be lamb (the authors' pre-training optimizer)")
        if args.precision != "fp32":
            errors.append('precision must be "fp32" (the authors train in full float32; TF32 is switched off here)')
        if args.max_grad_norm <= 0:
            errors.append("max_grad_norm must be > 0")
        if not 0 <= args.warmup_epochs < args.epochs:
            errors.append(f"warmup_epochs ({args.warmup_epochs}) must be >= 0 and < epochs ({args.epochs})")
        if errors:
            raise ValueError("invalid SparK config:\n  " + "\n  ".join(errors))
        self.run_dir = None            # set by the engine if it ever passes it; otherwise found from `timing`
        self._listing = ""

    def view_plan(self, args):
        return [("global", 1, args.blur_p)]   # one 224 px view per image; the authors use crop + flip only

    def describe(self, args):
        fmap = args.global_size // DOWNSAMPLE_RATIO
        m = self.model
        return (f"  SparK         : {fmap} x {fmap} patches of {DOWNSAMPLE_RATIO} px, {m.len_keep} visible per image (mask ratio "
                f"{args.mask_ratio} -> {1 - m.len_keep / fmap ** 2:.3f} hidden); densify norm {args.densify_norm}; "
                f"decoder width {args.decoder_width}; target {'the SAR channel' if args.recon_channels == 1 else 'all 3 input channels'}, "
                f"normalised per patch, loss on hidden patches only\n"
                f"  encoder maps  : {self.shapes}\n"
                f"  parameters    : {', '.join(f'{k} {v:,}' for k, v in self.n_params.items())}\n"
                f"  optimizer     : LAMB betas (0.9, {args.beta2}), global grad norm clipped at {args.max_grad_norm}; peak lr "
                f"{self.peak_lr:.3g} (= {args.base_lr} x {args.batch_size}/256) after {args.warmup_epochs} warm-up epochs from "
                f"{0.005 * self.peak_lr:.2e}, cosine to {0.001 * self.peak_lr:.2e}; weight decay {args.weight_decay} -> "
                f"{args.weight_decay_end} (cosine); no decay on biases, 1-D tensors and mask tokens\n"
                f"  precision     : {self.precision}")

    def build(self, args, rows, n_steps, device):
        self.args, self.device, self.n_steps = args, device, n_steps
        self.precision = use_true_fp32() if device.type == "cuda" else precision_flags()
        # SparK.__init__, get_param_groups and LAMB print the densify layers, every parameter name and the clip
        # value; that text goes to param_groups.txt in the run folder instead of the log.
        with contextlib.redirect_stdout(io.StringIO()) as listing:
            self.model = build_spark(args, imagenet_init=True).to(device)
            # >>> as in pretrain/main.py L75-L81: no weight decay on 1-D tensors, biases and mask tokens; LAMB with
            # betas (0.9, beta2) and the global gradient norm clipped inside step(); created with lr = peak and
            # weight_decay = 0 as there; lr_wd_annealing() sets every group's lr and weight decay before each step.
            self.peak_lr = args.base_lr * args.batch_size / 256     # arg_util.py L132
            groups = get_param_groups(self.model, nowd_keys=NOWD_KEYS)
            self.optimizer = TheSameAsTimmLAMB(params=groups, lr=self.peak_lr, weight_decay=0.0, betas=(0.9, args.beta2),
                                               max_grad_norm=args.max_grad_norm)
            # <<<
        self._listing = listing.getvalue()
        self.shapes = feature_shapes(self.model, args.global_size)
        fmap = args.global_size // DOWNSAMPLE_RATIO
        expected = [(2, 24, 8 * fmap, 8 * fmap), (2, 32, 4 * fmap, 4 * fmap), (2, 96, 2 * fmap, 2 * fmap), (2, 1280, fmap, fmap)]
        if self.shapes != expected:
            raise ValueError(f"unexpected encoder feature maps {self.shapes}, expected {expected}")
        self.n_params = count_parameters(self.model)
        self.total_iters = args.epochs * n_steps
        self.warmup_iters = args.warmup_epochs * n_steps
        if round(self.warmup_iters) >= self.total_iters - 1:
            raise ValueError("warm-up must end at least 2 steps before the last step (lr_wd_annealing divides by the rest)")
        self.schedule = {"steps_per_epoch": n_steps, "total_iters": self.total_iters, "warmup_iters": self.warmup_iters,
                         "peak_lr": self.peak_lr, "warmup_start_lr": 0.005 * self.peak_lr, "final_lr": 0.001 * self.peak_lr,
                         "weight_decay_start": args.weight_decay, "weight_decay_end": args.weight_decay_end}
        self.preview_views = preview_batch(rows, args.global_size, n=6, seed=0)   # fixed images, no RNG stream touched

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def schedule_fingerprint(self):
        """Hash of the per-step lr and weight decay that lr_wd_annealing will produce (a dummy optimizer is driven
        through every step), so a resume with a different step count or warm-up is refused by the engine."""
        dummy = torch.optim.SGD([torch.zeros(1)], lr=0.0)
        lrs, wds = [], []
        for it in range(self.total_iters):
            _, lr, _, wd = lr_wd_annealing(dummy, self.peak_lr, self.args.weight_decay, self.args.weight_decay_end,
                                           it, self.warmup_iters, self.total_iters)
            lrs.append(lr)
            wds.append(wd)
        return sha256_of_text(json.dumps([self.schedule, lrs, wds]))

    def train_epoch(self, loader, epoch, device, log_every, timing):
        # >>> ATRNet-STAR A3 train_one_epoch, with the view arriving as uint8 from the shared loader
        args = self.args
        self.model.train()   # BatchNorm uses batch statistics (visible positions only) and updates its running stats
        sums, t0 = {}, time.time()
        n = len(loader)
        max_lr = max_wd = float("nan")
        for it, (views, _idx) in enumerate(loader):
            step = it + 1
            git = n * epoch + it
            # the authors' per-iteration schedule (lr_control.py): warm-up then cosine for lr, cosine for weight decay
            _, max_lr, _, max_wd = lr_wd_annealing(self.optimizer, self.peak_lr, args.weight_decay, args.weight_decay_end,
                                                   git, self.warmup_iters, self.total_iters)
            x = to_network_input(views[0], device)          # the one global view: uint8 gray -> normalised RGB
            loss, smallest, active = self.model(x)          # a new random mask for every image
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss is {loss.item()}, stopping")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()   # LAMB clips the global gradient norm to max_grad_norm inside step() (main.py L79)
            diag = feature_diagnostics(smallest, active)
            for k, v in {"loss": loss.item(), "grad_norm": self.optimizer.global_grad_norm, **diag}.items():
                sums[k] = sums.get(k, 0.0) + v
            timing(step)
            if log_every and (step % log_every == 0 or step == n):
                print(f"    step {step:5d}/{n}  loss {sums['loss'] / step:.4f}  grad norm {sums['grad_norm'] / step:.3f}  "
                      f"feature cosine {sums['feature_cosine'] / step:.3f}  lr {max_lr:.2e}  wd {max_wd:.3f}  "
                      f"{time.time() - t0:.0f}s", flush=True)
        # <<<
        self._epoch_outputs(epoch, device, timing)
        return {k: v / n for k, v in sums.items()} | {"lr": max_lr, "wd": max_wd}

    # ---- per-epoch extras written into the run folder ------------------------------------------------------
    def _find_run_dir(self, timing):
        """The engine does not hand the run folder to the method. If a future engine sets method.run_dir, that is
        used; otherwise the folder is read from the RunRecord inside the engine's `timing` closure (it calls
        record.update(...)). None if neither works, in which case the preview is skipped with a warning."""
        if self.run_dir:
            return Path(self.run_dir)
        for cell in getattr(timing, "__closure__", None) or ():
            try:
                v = cell.cell_contents
            except ValueError:   # empty cell
                continue
            if hasattr(v, "run_dir") and hasattr(v, "info") and Path(v.run_dir).is_dir():
                return Path(v.run_dir)
        return None

    def _epoch_outputs(self, epoch, device, timing):
        run_dir = self._find_run_dir(timing)
        if run_dir is None:
            print("  WARNING: run folder unknown; reconstruction preview skipped", flush=True)
            return
        if self._listing and not (run_dir / "param_groups.txt").exists():
            (run_dir / "param_groups.txt").write_text(self._listing, encoding="utf-8")
        # The masks come from the global torch generator; the preview's eval forward must not shift the training
        # streams, or a resumed run would draw different masks than an uninterrupted one (ATRNet-STAR pitfall).
        with ssl_engine.preserved_rng():
            try:
                save_reconstruction_preview(self.model, self.preview_views, device, run_dir / f"reconstruction_epoch{epoch + 1:03d}.png")
            except Exception as e:  # noqa: BLE001 - a picture problem must never stop training
                print(f"  WARNING: could not draw the reconstruction preview: {e}", flush=True)

    # ---- resume and export ----------------------------------------------------------------------------------
    def state_dict(self):
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "precision": self.precision}

    def load_state_dict(self, d):
        self.model.load_state_dict(d["model"], strict=True)
        self.optimizer.load_state_dict(d["optimizer"])
        if d.get("precision") != self.precision:
            raise ValueError(f"cannot resume: precision flags differ ({d.get('precision')} vs {self.precision}; device changed?)")

    def export_state(self):
        state = encoder_features_state(self.model)
        dropped = sorted({k.split(".")[0] for k in self.model.state_dict() if not k.startswith("sparse_encoder.")})
        return state, {"which": "encoder (sparse_encoder.sp_cnn.features, prefix stripped)",
                       "excluded_from_export": dropped, "precision": self.precision, "parameters": self.n_params,
                       "visible_patches_per_image": self.model.len_keep, "schedule": self.schedule}


if __name__ == "__main__":
    ssl_engine.cli(SparKMethod, HERE)
