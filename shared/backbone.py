"""MobileNetV2 backbone: ImageNet start weights, export after SSL, import for fine-tuning.

Export format (same as mobilenetv2_SAR_SSL): a dict saved with torch.save and loaded with weights_only=True,
whose "backbone_state_dict" has exactly the key names of torchvision `mobilenet_v2().features` (parameters AND
BatchNorm buffers, 156 + 156 tensors), so the detector loads it with strict=True.
"""
import os
from pathlib import Path

import torch
from torch import nn
from torchvision.models import mobilenet_v2

from . import paths
from .run_record import sha256_of_file, sha256_of_state_dict

FEATURE_DIM = 1280


def load_imagenet_mobilenet_v2() -> nn.Module:
    """The full torchvision MobileNetV2 with the saved ImageNet-1k V2 weights (strict)."""
    net = mobilenet_v2(weights=None)
    state = torch.load(paths.MOBILENETV2_IMAGENET_WEIGHTS, map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=True)
    return net


def imagenet_weights_record() -> dict:
    return {"file": str(paths.MOBILENETV2_IMAGENET_WEIGHTS), "sha256": sha256_of_file(paths.MOBILENETV2_IMAGENET_WEIGHTS)}


def fresh_features() -> nn.Sequential:
    return mobilenet_v2(weights=None).features


def strip_prefix(state_dict: dict, prefix: str) -> dict:
    """Keys starting with prefix, with the prefix removed (e.g. 'backbone.features.' -> torchvision names)."""
    out = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if not out:
        raise KeyError(f"no keys start with {prefix!r}")
    return out


def export_backbone(features_state: dict, path, metadata: dict) -> dict:
    """Write an exported-backbone file; re-loads it strictly into a fresh features module as a check."""
    fresh = fresh_features()
    expected = len(list(fresh.parameters())) + len(list(fresh.buffers()))
    if len(features_state) != expected:
        raise ValueError(f"backbone state has {len(features_state)} tensors, expected {expected} (parameters + BN buffers)")
    fresh.load_state_dict(features_state, strict=True)
    state = {k: v.detach().cpu().clone() for k, v in features_state.items()}
    record = dict(metadata)
    record.update(backbone_state_dict=state, num_tensors=len(state), state_sha256=sha256_of_state_dict(state))
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(record, tmp)                                  # atomic: a kill during the write never leaves a partial file
    torch.load(tmp, map_location="cpu", weights_only=True)   # must be loadable without pickle code
    os.replace(tmp, path)
    record["file_sha256"] = sha256_of_file(path)
    return {k: v for k, v in record.items() if k != "backbone_state_dict"}


def load_backbone_file(path):
    """(features state dict, metadata without the weights) from an exported-backbone file."""
    record = torch.load(path, map_location="cpu", weights_only=True)
    state = record.pop("backbone_state_dict")
    record["file"] = str(path)
    record["file_sha256"] = sha256_of_file(path)
    return state, record


def pooled_features(features: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """(N, 1280) global-average-pooled output of the features module."""
    return torch.flatten(nn.functional.adaptive_avg_pool2d(features(x), 1), 1)
