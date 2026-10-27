"""SparK for MobileNetV2 on single-channel SAR chips: encoder adapter, model, view and diagnostics.

Sources (untouched copies in reference/SparK-main_original/, commit a63e386; hashes in reference/SOURCE.md):
    spark_lib/   the authors' sparse layers and dense-to-sparse conversion (encoder.py), masking, densifying
                 and loss (spark.py), UNet-style decoder (decoder.py), LAMB (lamb.py) and the lr/wd schedule
                 (lr_control.py). Only import lines differ from the originals.
    adapter      pretrain/models/custom.py: the three functions SparK needs from a custom CNN
    recipe       pretrain/utils/arg_util.py, main.py, utils/imagenet.py; paper Section 4.1 (arXiv 2301.03580v2)
Everything specific to MobileNetV2 or SAR is in this file and listed in SPARK_CHANGES.md.

Sizes, for a 224 x 224 input (do not confuse them):
    feature maps   24 x 56 x 56, 32 x 28 x 28, 96 x 14 x 14, 1280 x 7 x 7 (strides 4, 8, 16, 32)
    mask grid      7 x 7 patches of 32 x 32 px; the patch size equals the total downsampling (32), as in
                   the authors' code, so one mask cell is exactly one position of the 7 x 7 feature map
    decoder        width 768 (the authors' setting for every encoder); discarded after SSL together with
                   the densify layers and the mask tokens. Only the MobileNetV2 `features` are exported.
"""
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_
from torchvision import transforms
from torchvision.models import mobilenet_v2

import paths
from spark_lib import encoder as sparse  # holds the module-level mask (_cur_active) read by every sparse layer
from spark_lib.decoder import LightDecoder
from spark_lib.spark import SparK

DOWNSAMPLE_RATIO = 32              # MobileNetV2 halves the resolution 5 times
FEATURE_STAGES = (3, 6, 13, 18)    # features[i] whose outputs are decoded: the last layer at each resolution
DEPRECATED_BUFFERS = ("imn_m", "imn_s")  # spark.py L75-L77: torch.empty buffers, "deprecated and would never be used"


# ---- Encoder: MobileNetV2 with the interface SparK needs (pretrain/models/custom.py) ----------------
class MobileNetV2Hierarchical(nn.Module):
    """torchvision MobileNetV2 `features` plus the three functions of the authors' custom-CNN template:
    get_downsample_ratio(), get_feature_map_channels() and forward(x, hierarchical=True) -> list of maps.

    The submodule is named `features`, so its state_dict keys are exactly those of torchvision's
    `model.features` (the backbone contract of the downstream code). The ImageNet classifier is not kept.

    FEATURE_STAGES are the last layer of each resolution, like ResNet's layer1-layer4 in the authors'
    resnet.py: features[3] (24 channels, stride 4), [6] (32, stride 8), [13] (96, stride 16) and [18]
    (1280, stride 32). [18] is the final 1x1 conv, so every layer that is exported and fine-tuned later
    takes part in SSL (stopping at [17] would leave the 320 -> 1280 layer untrained)."""

    def __init__(self, imagenet_init: bool):
        super().__init__()
        net = mobilenet_v2(weights=None)  # architecture only, no download
        if imagenet_init:
            # Saved earlier from MobileNet_V2_Weights.IMAGENET1K_V2 (the same file A0, A1 and A2 start from).
            state = torch.load(paths.MOBILENETV2_IMAGENET_WEIGHTS, map_location="cpu", weights_only=True)
            net.load_state_dict(state, strict=True)  # fails loudly on any missing/unexpected key
        self.features = net.features
        self.feature_dim = net.classifier[1].in_features  # 1280

    def get_downsample_ratio(self) -> int:
        return DOWNSAMPLE_RATIO

    def get_feature_map_channels(self) -> List[int]:
        # Static attributes, no forward pass: SparseEncoder calls this after converting the layers to sparse ones.
        return [self.features[i].out_channels for i in FEATURE_STAGES]

    def forward(self, x, hierarchical: bool = False):
        if not hierarchical:  # pooled features, as the A1/A2 backbones return (SparK itself never uses this)
            return torch.flatten(F.adaptive_avg_pool2d(self.features(x), 1), 1)
        maps = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in FEATURE_STAGES:
                maps.append(x)
        return maps  # largest first, as the authors' ResNet forward returns layer1..layer4


# ---- SparK with the SAR reconstruction target ------------------------------------------------------
class SparKMobileNetV2(SparK):
    """The authors' SparK (spark_lib/spark.py) with one SAR change in forward(): the reconstruction target
    is the first `recon_channels` channels of the input. recon_channels = 3 is exactly the authors'
    target. recon_channels = 1 reconstructs the single SAR channel: the 3 input channels are copies of one
    gray channel shifted and scaled by the per-channel ImageNet mean/std, and per-patch normalization
    removes any shift and scale, so channel 0 normalized per patch IS the SAR image normalized per patch.
    With 3 channels, the constant channel offsets would become part of the target (measured on our chips:
    a median 41% of the target variance per patch, 72% on background patches; SPARK_CHANGES.md).

    forward() also returns the smallest encoder feature map and the mask, for logging only."""

    def __init__(self, *args, recon_channels: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.recon_channels = recon_channels
        for name in DEPRECATED_BUFFERS:  # uninitialized memory would make checkpoints differ for no reason
            getattr(self, name).zero_()

    def forward(self, inp_bchw: torch.Tensor, active_b1ff=None, vis=False):
        # A copy of SparK.forward (spark_lib/spark.py L87-L128). Lines marked "A3" differ; smoke_test.py
        # checks that with recon_channels = 3 the loss equals the authors' forward on the same mask.
        # step1. Mask
        if active_b1ff is None:     # rand mask
            active_b1ff: torch.BoolTensor = self.mask(inp_bchw.shape[0], inp_bchw.device)  # (B, 1, f, f)
        sparse._cur_active = active_b1ff    # (B, 1, f, f)
        active_b1hw = active_b1ff.repeat_interleave(self.downsample_raito, 2).repeat_interleave(self.downsample_raito, 3)  # (B, 1, H, W)
        masked_bchw = inp_bchw * active_b1hw

        # step2. Encode: get hierarchical encoded sparse features (a list containing 4 feature maps at 4 scales)
        fea_bcffs: List[torch.Tensor] = self.sparse_encoder(masked_bchw)
        fea_bcffs.reverse()  # after reversion: from the smallest feature map to the largest
        smallest = fea_bcffs[0]  # A3: (B, 1280, f, f), kept for the feature diagnostics (logging only)

        # step3. Densify: get hierarchical dense features for decoding
        cur_active = active_b1ff     # (B, 1, f, f)
        to_dec = []
        for i, bcff in enumerate(fea_bcffs):  # from the smallest feature map to the largest
            if bcff is not None:
                bcff = self.densify_norms[i](bcff)
                mask_tokens = self.mask_tokens[i].expand_as(bcff)
                bcff = torch.where(cur_active.expand_as(bcff), bcff, mask_tokens)   # fill in empty (non-active) positions with [mask] tokens
                bcff: torch.Tensor = self.densify_projs[i](bcff)
            to_dec.append(bcff)
            cur_active = cur_active.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # dilate the mask map, from (B, 1, f, f) to (B, 1, H, W)

        # step4. Decode and reconstruct
        rec_bchw = self.dense_decoder(to_dec)
        target_bchw = inp_bchw[:, :self.recon_channels]  # A3: the authors' target is all of inp_bchw (recon_channels = 3)
        inp, rec = self.patchify(target_bchw), self.patchify(rec_bchw)   # A3: target_bchw instead of inp_bchw
        mean = inp.mean(dim=-1, keepdim=True)
        var = (inp.var(dim=-1, keepdim=True) + 1e-6) ** .5
        inp = (inp - mean) / var
        l2_loss = ((rec - inp) ** 2).mean(dim=2, keepdim=False)    # (B, L, C) ==mean==> (B, L)

        non_active = active_b1ff.logical_not().int().view(active_b1ff.shape[0], -1)  # (B, 1, f, f) => (B, L)
        recon_loss = l2_loss.mul_(non_active).sum() / (non_active.sum() + 1e-8)  # loss only on masked (non-active) patches

        if vis:
            masked_bchw = target_bchw * active_b1hw  # A3: target_bchw instead of inp_bchw
            rec_bchw = self.unpatchify(rec * var + mean)
            rec_or_inp = torch.where(active_b1hw, target_bchw, rec_bchw)  # A3: target_bchw instead of inp_bchw
            return target_bchw, masked_bchw, rec_or_inp  # A3: target_bchw instead of inp_bchw
        return recon_loss, smallest, active_b1ff  # A3: the authors return recon_loss only


def build_spark(args, imagenet_init: bool = True) -> SparKMobileNetV2:
    """MobileNetV2 (ImageNet weights) -> sparse encoder -> SparK with the authors' LightDecoder.

    As in pretrain/main.py L57-L62 for a single GPU (sbn=False there: SyncBatchNorm needs several GPUs):
        enc = SparseEncoder(cnn, input_size, sbn=False); dec = LightDecoder(32, width, sbn=False);
        SparK(enc, dec, mask_ratio, densify_norm, sbn=False)
    The one change: for recon_channels = 1 the decoder's last 1x1 conv outputs 1 channel instead of 3
    (initialized like every Conv2d in LightDecoder.initialize(): truncated normal std 0.02, bias 0)."""
    cnn = MobileNetV2Hierarchical(imagenet_init)
    enc = sparse.SparseEncoder(cnn, input_size=args.input_size, sbn=False, verbose=False)
    dec = LightDecoder(enc.downsample_raito, width=args.decoder_width, sbn=False)
    if args.recon_channels != 3:
        dec.proj = nn.Conv2d(dec.proj.in_channels, args.recon_channels, kernel_size=1, stride=1, bias=True)
        trunc_normal_(dec.proj.weight, std=.02)
        nn.init.constant_(dec.proj.bias, 0)
    return SparKMobileNetV2(sparse_encoder=enc, dense_decoder=dec, mask_ratio=args.mask_ratio,
                            densify_norm=args.densify_norm, sbn=False, recon_channels=args.recon_channels)


def count_parameters(model: SparKMobileNetV2) -> dict:
    """Trainable parameters per part (the decoder side is discarded after SSL)."""
    part = {"encoder (MobileNetV2 features, exported)": model.sparse_encoder,
            "decoder (LightDecoder)": model.dense_decoder,
            "densify norms + projections": nn.ModuleList([model.densify_norms, model.densify_projs]),
            "mask tokens": model.mask_tokens}
    counts = {k: sum(p.numel() for p in m.parameters()) for k, m in part.items()}
    counts["total"] = sum(p.numel() for p in model.parameters())
    return counts


# ---- View ------------------------------------------------------------------------------------------
class DataAugmentationSparK:
    """ONE view per image, as the authors' pre-training transform (pretrain/utils/imagenet.py L66-L71):
    RandomResizedCrop(input_size, scale=(0.67, 1.0)) -> RandomHorizontalFlip() -> ToTensor -> Normalize.

    Changes (SPARK_CHANGES.md): bilinear interpolation with antialias instead of bicubic, as in A1 and A2
    (it matches the fine-tuning resize). The view leaves the data loader as a uint8 gray image (1, H, W);
    to_network_input() then does gray -> 3 channels -> [0, 1] -> ImageNet mean/std on the GPU, the same
    arithmetic as A0/A1/A2 (smoke_test.py compares the two)."""

    def __init__(self, args):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=tuple(args.crop_scale),
                                         interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.RandomHorizontalFlip(p=args.hflip_p),
            transforms.PILToTensor(),  # mode "L" image -> uint8 (1, H, W)
        ])

    def __call__(self, image):
        return self.transform(image)


def to_network_input(x: torch.Tensor, mean, std, device) -> torch.Tensor:
    """uint8 gray views (N, 1, H, W) -> float32 (N, 3, H, W) on `device`: copy the gray channel into R, G and
    B, divide by 255 and normalize with the ImageNet mean/std (as Grayscale(3) -> ToTensor -> Normalize)."""
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (x.to(device, non_blocking=True).float().div(255).expand(-1, 3, -1, -1) - m) / s


# ---- Diagnostics (new; logged only, never used in the loss or to choose a checkpoint) ---------------
@torch.no_grad()
def feature_diagnostics(smallest: torch.Tensor, active_b1ff: torch.Tensor) -> dict:
    """smallest (B, 1280, f, f): last encoder map of a masked batch, exactly 0 at masked positions;
    active_b1ff (B, 1, f, f): visible patches. Each image's feature is the mean over its VISIBLE positions
    (global average pooling restricted to what the encoder saw). Then the same definitions as A1/A2:
        feature_cosine  mean cosine similarity between different images' features (1.0 = all identical)
        feature_std     std across images of the L2-normalized features, averaged over the 1280 dims (0 = collapse)
    Watch the trend only: A1/A2 compute these on whole views, here 40% of each image is visible."""
    n_visible = active_b1ff.flatten(1).sum(1, keepdim=True).float()  # (B, 1)
    pooled = smallest.float().flatten(2).sum(2) / n_visible          # (B, 1280)
    f = F.normalize(pooled, dim=-1)
    sim = f @ f.T
    n = len(f)
    return {"feature_cosine": ((sim.sum() - sim.diagonal().sum()) / max(n * (n - 1), 1)).item(),
            "feature_std": f.std(0).mean().item()}
