"""LeJEPA for MobileNetV2 on single-channel SAR chips: network, loss, views and diagnostics.

Sources (untouched copies in reference/lejepa-main_original/, hashes in reference/SOURCE.md):
    SIGReg       lejepa_lib/ = the authors' library files, unchanged (EppsPulley + SlicingUnivariateTest)
    loss         MINIMAL.md L172-L174 and the paper's Section 5.1 listing (arXiv 2511.08544v3)
    projector    MINIMAL.md L88: torchvision MLP with BatchNorm, two hidden layers of 2048
    views        README.md "GOTO hyperparameters" (adapted to SAR; see README.md, "SSL recipe")
What is NOT here, on purpose: no teacher network, no EMA, no stop-gradient, no centering, no predictor.
LeJEPA prevents collapse with SIGReg alone (paper, Section 5.1 and 6).

Three different sizes appear here; do not confuse them:
    feature_dim  1280  MobileNetV2 backbone output per image (after global average pooling); exported
    proj_dim     config projector output; the loss works on these; thrown away after SSL
    40                 number of SOC-40 classes; only used later, in supervised fine-tuning
"""
import math
import random

import torch
import torch.nn.functional as F
from PIL import ImageFilter, ImageOps
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
from torchvision.ops import MLP

import paths
from lejepa_lib.multivariate import SlicingUnivariateTest
from lejepa_lib.univariate import EppsPulley


# ---- Backbone: same adapter as the DINO experiment (route A1), so A1 and A2 export the same thing ----
class MobileNetV2Backbone(nn.Module):
    """MobileNetV2 as an SSL backbone: one pooled feature vector per image.

    forward: (batch, 3, H, W) -> features (batch, 1280, H/32, W/32) -> global average pool
             -> flatten -> (batch, 1280). Works for any H, W.

    The torchvision classifier (Dropout 0.2 + Linear 1280->1000) is not kept. The submodule is named
    `features` so that `self.features.state_dict()` has exactly the keys of torchvision's
    `model.features`, which is the backbone checkpoint contract of the downstream code.
    """

    def __init__(self, imagenet_init: bool):
        super().__init__()
        net = mobilenet_v2(weights=None)  # architecture only, no download
        if imagenet_init:
            # Saved earlier from MobileNet_V2_Weights.IMAGENET1K_V2 (same file A0 and A1 start from).
            state = torch.load(paths.MOBILENETV2_IMAGENET_WEIGHTS, map_location="cpu", weights_only=True)
            net.load_state_dict(state, strict=True)  # fails loudly on any missing/unexpected key
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
        # MobileNetV2's 1280 pooled features instead of the ViT's 512-value output (see README.md).
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

    (The README writes EppsPulley(num_points=17), which raises a TypeError: the argument is n_points.)"""
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


# ---- Views ----------------------------------------------------------------------------------------
class RandomGaussianBlur:
    """Gaussian blur with probability p and a random sigma (PIL's radius is the standard deviation).
    Same operation and sigma range as the A1 DINO views (0.1 to 2 px); MINIMAL.md uses sigma 0.1 to 2."""

    def __init__(self, p: float, sigma: tuple[float, float]):
        self.p, self.sigma = p, sigma

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.sigma)))


class RandomSolarization:
    def __init__(self, p: float):
        self.p = p

    def __call__(self, img):
        return ImageOps.solarize(img) if random.random() < self.p else img


class DataAugmentationLeJEPA:
    """n_global_views views of global_crop_size, then n_local_views views of local_crop_size.

    README.md "GOTO hyperparameters": every view gets the same transforms, only the crop scale differs:
    RandomResizedCrop -> horizontal flip -> colour jitter (p 0.8) -> random grayscale -> blur -> solarize
    -> normalize. Adapted to SAR (README.md here, "Views"): saturation, hue and random grayscale do
    nothing to a gray image, so they are left out; solarization is off, as in A1.

    Each view is returned as a uint8 tensor (1, H, W), the gray image after all random transforms.
    The last, deterministic steps (gray copied into 3 channels, scaling to [0, 1], ImageNet mean/std;
    the same as A0 and A1) are done on the GPU by to_network_input(). Same arithmetic, but the data
    loader moves 12 times fewer bytes: 8 float32 RGB views per image made the pilot data-bound."""

    def __init__(self, args):
        def view(size, scale):
            return transforms.Compose([
                transforms.RandomResizedCrop(size, scale=tuple(scale), interpolation=transforms.InterpolationMode.BILINEAR,
                                             antialias=True),
                transforms.RandomHorizontalFlip(p=args.hflip_p),
                transforms.RandomApply([transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast)],
                                       p=args.color_jitter_p),
                RandomGaussianBlur(args.blur_p, args.blur_sigma),
                RandomSolarization(args.solarization_p),
                transforms.PILToTensor(),  # mode "L" image -> uint8 (1, H, W)
            ])
        self.n_global, self.n_local = args.n_global_views, args.n_local_views
        self.global_view = view(args.global_crop_size, args.global_crops_scale)
        self.local_view = view(args.local_crop_size, args.local_crops_scale)

    def __call__(self, image):
        return ([self.global_view(image) for _ in range(self.n_global)] +
                [self.local_view(image) for _ in range(self.n_local)])


def to_network_input(views: list[torch.Tensor], mean, std, device) -> list[torch.Tensor]:
    """uint8 gray views (N, 1, H, W) -> float32 (N, 3, H, W) on `device`: copy the gray channel into R, G
    and B, divide by 255 and normalize with the ImageNet mean/std. The same operations, in the same order,
    as Grayscale(3) -> ToTensor() -> Normalize() in A0 and A1 (smoke_test.py compares the two)."""
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return [(v.to(device, non_blocking=True).float().div(255).expand(-1, 3, -1, -1) - m) / s for v in views]


# ---- Collapse diagnostics (new; logged only, never used in the loss or to choose a checkpoint) ------
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
                    Same definition as in the A1 DINO runs.
    feature_std     std across images of the L2-normalized features, averaged over the 1280 dimensions.
                    0 means collapse. Same definition as in A1."""
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
    same shape written out per step: linear from start_factor * lr to lr, then cosine to final_ratio * lr."""
    out = []
    for s in range(total_steps):
        if s < warmup_steps:
            f = start_factor + (1 - start_factor) * s / max(warmup_steps, 1)
        else:
            t = (s - warmup_steps) / max(total_steps - warmup_steps, 1)
            f = final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * t))
        out.append(base_lr * f)
    return out
