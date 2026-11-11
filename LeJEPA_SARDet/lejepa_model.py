"""LeJEPA for MobileNetV2 on single-channel SAR images (SARDet-100K suite): network, loss and diagnostics.

COPY of mobilenetv2_SAR_SSL/Experiments/mobileNetv2_LeJEPA/lejepa_model.py (the ATRNet-STAR A2 code), with two
changes: the data-augmentation classes are removed (views come from shared/ssl_data.py, one recipe for all
methods) and the ImageNet weights are loaded through shared/backbone.py. The network, the loss, the SIGReg
builder, the lr schedule and the collapse diagnostics are the ATRNet-STAR code, so the method is the same.

Sources (untouched copies in mobilenetv2_SAR_SSL/Experiments/mobileNetv2_LeJEPA/reference/lejepa-main_original/):
    SIGReg       lejepa_lib/ = the authors' library files, unchanged (EppsPulley + SlicingUnivariateTest)
    loss         MINIMAL.md L172-L174 and the paper's Section 5.1 listing (arXiv 2511.08544v3)
    projector    MINIMAL.md L88: torchvision MLP with BatchNorm, two hidden layers of 2048
What is NOT here, on purpose: no teacher network, no EMA, no stop-gradient, no centering, no predictor.
LeJEPA prevents collapse with SIGReg alone (paper, Section 5.1 and 6).

Three different sizes appear here; do not confuse them:
    feature_dim  1280  MobileNetV2 backbone output per image (after global average pooling); exported
    proj_dim     config projector output; the loss works on these; thrown away after SSL
    6                  number of SARDet-100K classes; only used later, in detection fine-tuning
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import mobilenet_v2
from torchvision.ops import MLP

from lejepa_lib.multivariate import SlicingUnivariateTest
from lejepa_lib.univariate import EppsPulley
from shared.backbone import load_imagenet_mobilenet_v2


# ---- Backbone: same adapter as the DINO experiment, so DINO and LeJEPA export the same thing ----------------
class MobileNetV2Backbone(nn.Module):
    """MobileNetV2 as an SSL backbone: one pooled feature vector per image.

    forward: (batch, 3, H, W) -> features (batch, 1280, H/32, W/32) -> global average pool
             -> flatten -> (batch, 1280). Works for any H, W.

    The torchvision classifier (Dropout 0.2 + Linear 1280->1000) is not kept. The submodule is named
    `features` so that `self.features.state_dict()` has exactly the keys of torchvision's
    `model.features`, which is the backbone checkpoint contract of shared/backbone.py.
    """

    def __init__(self, imagenet_init: bool):
        super().__init__()
        # Same ImageNet file as the baseline (shared/paths.py MOBILENETV2_IMAGENET_WEIGHTS), strict load.
        net = load_imagenet_mobilenet_v2() if imagenet_init else mobilenet_v2(weights=None)
        self.features = net.features
        self.feature_dim = net.classifier[1].in_features  # 1280 for the width-1.0 model

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)  # same pooling as torchvision's MobileNetV2.forward
        return torch.flatten(x, 1)


class LeJEPANet(nn.Module):
    """Backbone + projector. One network: LeJEPA has no teacher.

    forward(views): views is a list of V tensors, each (N, 3, H, W); all views of one size are run
    through the backbone together (a CNN needs one resolution per forward call), sizes in the order
    given. Returns
        emb   (V, N, 1280)      backbone features (what is exported and fine-tuned later)
        proj  (V, N, proj_dim)  projector outputs (what the loss works on)
    The (V, N, K) layout is the one used by MINIMAL.md (`.reshape(N, V, -1).transpose(0, 1)`).
    """

    def __init__(self, imagenet_init: bool, proj_hidden_dim: int, proj_dim: int):
        super().__init__()
        self.backbone = MobileNetV2Backbone(imagenet_init)
        # MINIMAL.md L88: MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d). The input size is
        # MobileNetV2's 1280 pooled features instead of the ViT's 512-value output (CHANGES.md).
        self.proj = MLP(self.backbone.feature_dim, [proj_hidden_dim, proj_hidden_dim, proj_dim],
                        norm_layer=nn.BatchNorm1d)

    def forward(self, views: list[torch.Tensor]):
        n = views[0].shape[0]
        groups, start = [], 0
        while start < len(views):  # consecutive views of the same size form one group
            end = start
            while end < len(views) and views[end].shape[-2:] == views[start].shape[-2:]:
                end += 1
            groups.append(torch.cat(views[start:end]))
            start = end
        emb = torch.cat([self.backbone(g) for g in groups])  # (V * N, 1280), view-major
        proj = self.proj(emb)
        return emb.reshape(len(views), n, -1), proj.reshape(len(views), n, -1)


def build_sigreg(args) -> SlicingUnivariateTest:
    """SIGReg exactly as in the README's quick start, with the library's own Epps-Pulley defaults.

    (The README writes EppsPulley(num_points=17), which raises a TypeError: the argument is n_points.)
    The random directions are seeded from the module's `global_step` buffer, which is saved in the
    checkpoint, so a resumed run draws the same directions as an uninterrupted one."""
    return SlicingUnivariateTest(univariate_test=EppsPulley(t_max=args.t_max, n_points=args.n_points),
                                 num_slices=args.num_slices)


def lejepa_loss(proj: torch.Tensor, n_global_views: int, sigreg: SlicingUnivariateTest, lamb: float):
    """LeJEPA loss on projector outputs proj (V, N, K); the first n_global_views views are the global ones.

    agreement  every view's projection is pulled toward the mean of the global views' projections
               (paper Section 5.1: `centers = g_emb.view(-1, bs, K).mean(0)`; `(centers - a_emb).square().mean()`).
               For a CNN the paper says "set global_views = all_views", and then this is exactly
               MINIMAL.md L172: `inv_loss = (proj.mean(0) - proj).square().mean()`.
               No stop-gradient: the gradient also flows through the mean.
    sigreg     SIGReg of each view's N projections, averaged over views and random directions
               (MINIMAL.md L173 `sigreg(proj)` on the (V, N, K) tensor).
    total      (1 - lamb) * agreement + lamb * sigreg  (MINIMAL.md L174).

    Computed in float32 outside autocast (a change: MINIMAL computes it inside bf16 autocast). The
    cos/sin averages over the batch inside SIGReg lose precision in bfloat16; the cost is negligible."""
    with torch.autocast(proj.device.type, enabled=False):
        proj = proj.float()
        centers = proj[:n_global_views].mean(0)
        agreement = (centers - proj).square().mean()
        sig = sigreg(proj)
        total = sig * lamb + agreement * (1 - lamb)
    return total, agreement, sig


# ---- Collapse diagnostics (logged only, never used in the loss or to choose a checkpoint) -------------------
@torch.no_grad()
def collapse_diagnostics(emb: torch.Tensor, proj: torch.Tensor) -> dict:
    """emb (V, N, 1280) backbone features, proj (V, N, K) projector outputs of one batch.

    proj_std        std across the N images of each projector dimension, averaged over dimensions and
                    views. SIGReg's target is an isotropic standard Gaussian, so about 1 is healthy;
                    near 0 means the projections collapsed.
    proj_mean_abs   |mean across images| per dimension, averaged. Target 0.
    feature_cosine  mean cosine similarity between pooled backbone features of different images
                    (first view). 1.0 means all images look the same to the backbone. MobileNetV2
                    features are non-negative (ReLU6), so this starts well above 0; watch its trend.
                    Same definition as in the DINO runs.
    feature_std     std across images of the L2-normalized features, averaged over the 1280 dimensions.
                    0 means collapse. Same definition as in DINO."""
    p = proj.float()
    f = F.normalize(emb[0].float(), dim=-1)
    sim = f @ f.T
    n = len(f)
    return {
        "proj_std": p.std(1).mean().item(),
        "proj_mean_abs": p.mean(1).abs().mean().item(),
        "feature_cosine": ((sim.sum() - sim.diagonal().sum()) / max(n * (n - 1), 1)).item(),
        "feature_std": f.std(0).mean().item(),
    }


def lr_schedule(base_lr: float, total_steps: int, warmup_steps: int, start_factor: float, final_ratio: float):
    """Learning rate for every optimizer step, fixed in advance (so a resumed run uses the same values).

    README.md: "Linear warmup with cosine annealing decay. Final learning rate: initial_lr / 1000".
    MINIMAL.md L157-L160: LinearLR(start_factor=0.01) for one epoch, then CosineAnnealingLR. This is the
    same shape written out per step: linear from start_factor * lr to lr, then cosine to final_ratio * lr.
    The rate is NOT scaled with the batch size (the authors' README gives 5e-4 as an absolute value)."""
    out = []
    for s in range(total_steps):
        if s < warmup_steps:
            f = start_factor + (1 - start_factor) * s / max(warmup_steps, 1)
        else:
            t = (s - warmup_steps) / max(total_steps - warmup_steps, 1)
            f = final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * t))
        out.append(base_lr * f)
    return out
