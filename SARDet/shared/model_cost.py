"""One-time cost of the shared detector (same for every method): parameters, multiply-accumulates at 512 px,
CPU latency per image. Written to compare/outputs/model_cost.json.

    python -m shared.model_cost [--repeats 10]
MACs are counted with forward hooks on Conv2d and Linear layers for one 512x512 input (backbone + FPN + RPN +
box head with the default proposal counts), so they are the inference cost of the whole detector, not FLOPs of
the backbone alone.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from . import paths
from .detector import build_detector, count_parameters, load_finetune_config


def count_macs(model, x):
    macs = [0]

    def conv_hook(m, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        macs[0] += out.numel() // out.shape[0] * k * out.shape[0] if out.dim() == 4 else 0

    def linear_hook(m, inp, out):
        macs[0] += inp[0].numel() // inp[0].shape[-1] * m.in_features * m.out_features

    hooks = [m.register_forward_hook(conv_hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    hooks += [m.register_forward_hook(linear_hook) for m in model.modules() if isinstance(m, nn.Linear)]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return macs[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args(argv)
    cfg = load_finetune_config()["detector"]
    model = build_detector(cfg).eval()
    x = [torch.rand(3, 512, 512)]
    torch.set_num_threads(max(1, torch.get_num_threads()))
    with torch.no_grad():
        model(x)                                            # warm-up
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            model(x)
        cpu_ms = (time.perf_counter() - t0) / args.repeats * 1000
    backbone_macs = count_macs(model.backbone.body, torch.rand(1, 3, 512, 512))
    total_macs = count_macs(model, x)
    out = {"parameters": count_parameters(model), "input": "1 x 3 x 512 x 512",
           "backbone_gmacs": round(backbone_macs / 1e9, 3), "detector_gmacs_inference": round(total_macs / 1e9, 3),
           "cpu_latency_ms_per_image": round(cpu_ms, 1), "cpu_threads": torch.get_num_threads(),
           "torch": torch.__version__, "note": "untrained head; proposal counts = test defaults; GMACs x2 = GFLOPs"}
    dest = paths.SUITE_ROOT / "compare" / "outputs"
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "model_cost.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
