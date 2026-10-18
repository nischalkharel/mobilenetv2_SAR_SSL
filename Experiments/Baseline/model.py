"""MobileNetV2 construction with an interchangeable backbone initialization.

MobileNetV2 in torchvision has two parts:
    model.features    the backbone (convolutions), output (batch, 1280, H/32, W/32)
    model.classifier  Dropout(0.2) -> Linear(1280 -> num_classes), applied after global pooling

The backbone initialization is chosen by config["init"]["type"]:
    "imagenet"             features from the saved IMAGENET1K_V2 weights   (A0 baseline)
    "backbone_checkpoint"  features from a file saved by an SSL run, e.g. DINO (A1, A2, ...)
    "none"                 random features; only used when a full fine-tuned checkpoint is
                           loaded on top afterwards (evaluation)

In every case the classifier is a fresh, randomly initialized 40-class layer, so routes
differ only in where the backbone starts.

Backbone checkpoint contract for SSL runs: torch.save({"backbone_state_dict":
model.features.state_dict(), ...}, path). Keys must match model.features exactly.
"""
import torch
from torch import nn
from torchvision.models import mobilenet_v2

import paths

FEATURE_DIM = 1280  # channels at the end of MobileNetV2's backbone


def build_model(init_cfg: dict, num_classes: int) -> nn.Module:
    model = mobilenet_v2(weights=None)  # architecture only
    init_type = init_cfg["type"]

    if init_type == "imagenet":
        # Saved earlier from MobileNet_V2_Weights.IMAGENET1K_V2 by MobileNetV2/download_mobileNetV2.py.
        state = torch.load(paths.MOBILENETV2_IMAGENET_WEIGHTS, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)  # fails loudly on any missing/unexpected key
    elif init_type == "backbone_checkpoint":
        ckpt = torch.load(init_cfg["backbone_checkpoint"], map_location="cpu", weights_only=True)
        model.features.load_state_dict(ckpt["backbone_state_dict"], strict=True)
    elif init_type != "none":
        raise ValueError(f"unknown init type: {init_type}")

    # Replace the 1000-class ImageNet layer with a new 40-class layer, initialized the same
    # way torchvision initializes MobileNetV2's classifier.
    head = nn.Linear(FEATURE_DIM, num_classes)
    nn.init.normal_(head.weight, 0, 0.01)
    nn.init.zeros_(head.bias)
    model.classifier[1] = head
    return model


def load_finetuned(checkpoint_path, num_classes: int) -> tuple[nn.Module, dict]:
    """Rebuild a model saved by train.py (backbone + 40-class head) for evaluation."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model({"type": "none"}, num_classes)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model, ckpt
