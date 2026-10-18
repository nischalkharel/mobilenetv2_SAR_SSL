"""DINO components adapted to MobileNetV2 on single-channel SAR chips.

Most of this file is copied from the original DINO repository (facebookresearch/dino),
Copyright (c) Facebook, Inc. and its affiliates, licensed under the Apache License 2.0.
The license text is in reference/dino-main_original/LICENSE, and the untouched original files
are in reference/dino-main_original/.

Every copied block sits between two marker lines:
    # >>> DINO extract <ID>: <original file> L<first>-L<last> (<name>), <unchanged|modified>
    # <<< end <ID>
tools/make_change_record.py compares each block with the original lines and writes the exact
diffs to diffs/ and the line-by-line table in DINO_CHANGES.md. Code outside the markers is new.

Three different sizes appear here; do not confuse them:
    feature_dim  1280   MobileNetV2 backbone output per image (after global average pooling)
    out_dim      config DINO projection head output (number of "prototype" dimensions the
                        softmax runs over); unrelated to the number of classes
    40                  number of SOC-40 classes; only used later, in supervised fine-tuning
"""
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageFilter, ImageOps
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v2

import paths


# ---- New: MobileNetV2 feature adapter ------------------------------------------------------------
class MobileNetV2Backbone(nn.Module):
    """MobileNetV2 as a DINO backbone: one pooled feature vector per image.

    forward: (batch, 3, H, W) -> features (batch, 1280, H/32, W/32) -> global average pool
             -> flatten -> (batch, 1280). Works for any H, W (224 global and 96 local crops
             both give 1280 values per image).

    The torchvision classifier (Dropout 0.2 + Linear 1280->1000) is not kept at all. The original
    DINO code instead replaced `backbone.fc` with Identity, which works for ResNet but not for
    MobileNetV2 (it has no `fc`; its classifier would still run and output 1000 ImageNet logits).
    The submodule is named `features` so that `self.features.state_dict()` has exactly the keys of
    torchvision's `model.features`, which is the backbone checkpoint contract of the downstream code.
    """

    def __init__(self, imagenet_init: bool):
        super().__init__()
        net = mobilenet_v2(weights=None)  # architecture only, no download
        if imagenet_init:
            # Saved earlier from MobileNet_V2_Weights.IMAGENET1K_V2 (same file the A0 baseline uses).
            state = torch.load(paths.MOBILENETV2_IMAGENET_WEIGHTS, map_location="cpu", weights_only=True)
            net.load_state_dict(state, strict=True)  # fails loudly on any missing/unexpected key
        self.features = net.features
        self.feature_dim = net.classifier[1].in_features  # 1280 for the width-1.0 model (checked at run time)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)  # same pooling as torchvision's MobileNetV2.forward
        return torch.flatten(x, 1)


# ---- Copied from utils.py ------------------------------------------------------------------------
# >>> DINO extract E1: utils.py L36-L54 (GaussianBlur), unchanged
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
# <<< end E1


# >>> DINO extract E2: utils.py L57-L68 (Solarization), unchanged
class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
# <<< end E2


# >>> DINO extract E3: utils.py L132-L141 (clip_gradients), unchanged
def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms
# <<< end E3


# >>> DINO extract E4: utils.py L144-L149 (cancel_gradients_last_layer), unchanged
def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None
# <<< end E4


# >>> DINO extract E5: utils.py L187-L198 (cosine_scheduler), unchanged
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
# <<< end E5


# >>> DINO extract E6: utils.py L594-L629 (MultiCropWrapper), modified
class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # the backbone (MobileNetV2Backbone) already has no ImageNet classifier
        self.backbone = backbone
        self.head = head
        self.last_features = None

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # keep the pooled backbone features (no gradient) for the collapse diagnostics
        self.last_features = output.detach()
        # Run the head forward on the concatenated features.
        return self.head(output)
# <<< end E6


# >>> DINO extract E7: utils.py L632-L643 (get_params_groups), unchanged
def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]
# <<< end E7


# ---- Copied from vision_transformer.py -------------------------------------------------------------
# >>> DINO extract E8: vision_transformer.py L257-L291 (DINOHead), modified
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        if norm_last_layer:
            self.last_layer.parametrizations.weight.original0.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
# <<< end E8


# ---- Copied from main_dino.py ----------------------------------------------------------------------
# >>> DINO extract E9: main_dino.py L363-L416 (DINOLoss), modified
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # single process: the whole batch is on this GPU, so no all_reduce and world size 1
        batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
# <<< end E9


# >>> DINO extract E10: main_dino.py L419-L464 (DataAugmentationDINO), modified
class DataAugmentationDINO(object):
    def __init__(self, args):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=args.hflip_p),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast)],
                p=args.color_jitter_p
            ),
        ])
        normalize = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(args.normalize_mean, args.normalize_std),
        ])
        interpolation = transforms.InterpolationMode.BILINEAR

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(args.global_crop_size, scale=args.global_crops_scale, interpolation=interpolation, antialias=True),
            flip_and_color_jitter,
            GaussianBlur(args.blur_p[0]),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(args.global_crop_size, scale=args.global_crops_scale, interpolation=interpolation, antialias=True),
            flip_and_color_jitter,
            GaussianBlur(args.blur_p[1]),
            Solarization(args.solarization_p),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = args.local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(args.local_crop_size, scale=args.local_crops_scale, interpolation=interpolation, antialias=True),
            flip_and_color_jitter,
            GaussianBlur(p=args.blur_p[2]),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
# <<< end E10


# ---- New: collapse diagnostics ----------------------------------------------------------------------
@torch.no_grad()
def collapse_diagnostics(teacher_output, center, temp, teacher_features, batch_size):
    """A few numbers that show whether the representation is collapsing. They do not prove that
    the features are useful; only a downstream evaluation can show that.

    teacher_output   (2 * batch, out_dim) teacher logits for the two global views
    teacher_features (2 * batch, feature_dim) pooled MobileNetV2 features for the same views

    teacher_entropy           mean entropy of each image's teacher distribution (after centering and
                              sharpening). Near ln(out_dim): uniform, the "flat" collapse.
    teacher_marginal_entropy  entropy of the teacher distribution averaged over the batch. Near 0:
                              every image goes to the same dimension, the "one-hot" collapse.
                              Healthy training keeps this well above teacher_entropy.
    feature_cosine            mean cosine similarity between pooled backbone features of different
                              images (first global view). 1.0 means all images look the same to
                              the backbone. ImageNet MobileNetV2 features are non-negative (ReLU6),
                              so this starts well above 0 even when nothing is wrong; watch its trend.
    feature_std               standard deviation across images of the L2-normalized features,
                              averaged over the 1280 dimensions. 0 means collapse.
    """
    probs = F.softmax((teacher_output.float() - center) / temp, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1).mean()
    marginal = probs.mean(0)
    marginal_entropy = -(marginal * torch.log(marginal.clamp_min(1e-12))).sum()
    f = F.normalize(teacher_features[:batch_size].float(), dim=-1)
    sim = f @ f.T
    n = len(f)
    off_diagonal = (sim.sum() - sim.diagonal().sum()) / max(n * (n - 1), 1)
    return {
        "teacher_entropy": entropy.item(),
        "teacher_marginal_entropy": marginal_entropy.item(),
        "feature_cosine": off_diagonal.item(),
        "feature_std": f.std(0).mean().item(),
    }


def denormalize(x, mean, std):
    """Undo Normalize for display: (..., 3, H, W) -> values in [0, 1]."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)
