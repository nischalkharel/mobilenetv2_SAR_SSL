"""The ONE detector of the suite: MobileNetV2 features + FPN + torchvision Faster R-CNN.

build_detector() is the only way a detector is built, for every method: the backbone weights are the only
input that differs (ImageNet file for the baseline, an exported SSL backbone otherwise). Everything else
(FPN taps, channels, anchors, input size, scoring thresholds) comes from shared/finetune_config.json, and the
head initialisation is seeded right before the FPN and heads are created, so two detectors built with
different backbones have identical heads.
"""
import json

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from . import paths
from .backbone import load_imagenet_mobilenet_v2, fresh_features

NUM_CLASSES = len(paths.CLASS_NAMES) + 1   # + background (label 0)


def load_finetune_config() -> dict:
    with open(paths.SHARED_FINETUNE_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.pop("_comment", None)
    return cfg


def build_backbone_features(backbone_state=None) -> nn.Sequential:
    """MobileNetV2 features with ImageNet weights (backbone_state None) or an exported SSL backbone (strict)."""
    if backbone_state is None:
        return load_imagenet_mobilenet_v2().features
    features = fresh_features()
    features.load_state_dict(backbone_state, strict=True)
    return features


def build_detector(det_cfg: dict, backbone_state=None, head_seed: int = 0) -> FasterRCNN:
    features = build_backbone_features(backbone_state)
    layers = det_cfg["fpn_return_layers"]                     # {"3": 24, "6": 32, "13": 96, "18": 1280}
    return_layers = {name: str(i) for i, name in enumerate(layers)}
    in_channels = [layers[name] for name in layers]
    torch.manual_seed(head_seed)                              # identical FPN / RPN / box-head init for every backbone
    fpn = BackboneWithFPN(features, return_layers, in_channels, det_cfg["fpn_channels"], extra_blocks=LastLevelMaxPool())
    n_levels = len(layers) + 1                                # + P6 from LastLevelMaxPool
    anchors = AnchorGenerator(sizes=tuple(tuple(s) for s in det_cfg["anchor_sizes"]),
                              aspect_ratios=(tuple(det_cfg["aspect_ratios"]),) * n_levels)
    assert len(det_cfg["anchor_sizes"]) == n_levels, "one anchor-size tuple per pyramid level"
    model = FasterRCNN(
        fpn, num_classes=NUM_CLASSES, rpn_anchor_generator=anchors,
        min_size=det_cfg["input_size"], max_size=det_cfg["input_size"],
        image_mean=det_cfg["image_mean"], image_std=det_cfg["image_std"],
        rpn_pre_nms_top_n_train=det_cfg["rpn_pre_nms_top_n_train"], rpn_post_nms_top_n_train=det_cfg["rpn_post_nms_top_n_train"],
        rpn_pre_nms_top_n_test=det_cfg["rpn_pre_nms_top_n_test"], rpn_post_nms_top_n_test=det_cfg["rpn_post_nms_top_n_test"],
        box_score_thresh=det_cfg["box_score_thresh"], box_nms_thresh=det_cfg["box_nms_thresh"],
        box_detections_per_img=det_cfg["box_detections_per_img"])
    if det_cfg["backbone_bn"] != "train":
        raise ValueError("only backbone_bn='train' is implemented (frozen BN was rejected in the protocol review)")
    return model


def backbone_features_of(model: FasterRCNN) -> nn.Sequential:
    """The MobileNetV2 features module inside a built detector (for probes and response maps)."""
    return model.backbone.body


def head_state_fingerprint(model: FasterRCNN) -> str:
    """SHA-256 of every non-backbone tensor; two detectors built with different backbones must match."""
    from .run_record import sha256_of_state_dict
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.body.")}
    return sha256_of_state_dict(state)


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    backbone = sum(p.numel() for p in model.backbone.body.parameters())
    return {"total": total, "backbone": backbone, "fpn_and_heads": total - backbone}
