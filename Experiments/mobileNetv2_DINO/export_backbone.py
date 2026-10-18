"""Export the MobileNetV2 backbone of a DINO run for supervised fine-tuning.

Run from the repository root (train_ssl.py already does this automatically after the last epoch):
    python Experiments/mobileNetv2_DINO/export_backbone.py --run Experiments/mobileNetv2_DINO/outputs/ssl/<run folder>

Default and fixed in advance: the TEACHER backbone from the run's final checkpoint.pth. The teacher is
the network DINO evaluates in the original paper. The student can be exported with --which student
for analysis, but teacher versus student must never be chosen by test accuracy.

Output file <run folder>/backbone_<which>_epoch<N>.pth:
    {"backbone_state_dict": <MobileNetV2 features.state_dict()>,   parameters AND BatchNorm buffers
     ...metadata: source run, checkpoint hash, config, seed, manifest hashes, SSL cost...}
Excluded: the DINO projection head and any classifier. The file is checked by loading it strictly
into a fresh torchvision MobileNetV2 `features`.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torchvision.models import mobilenet_v2

from utils import sha256_file

PREFIX = "backbone.features."  # key prefix of the backbone inside MultiCropWrapper(MobileNetV2Backbone, DINOHead)


def export(run_dir: Path, which: str = "teacher") -> Path:
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "checkpoint.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # our own file: also holds RNG states
    state = ckpt[which]
    backbone = {k[len(PREFIX):]: v for k, v in state.items() if k.startswith(PREFIX)}
    dropped = sorted({k.split(".")[0] + "." + k.split(".")[1] for k in state if not k.startswith(PREFIX)})

    # Strict load into a fresh downstream backbone: every parameter and buffer must match by name and shape.
    fresh = mobilenet_v2(weights=None).features
    fresh.load_state_dict(backbone, strict=True)
    n_params = sum(1 for _ in fresh.parameters())
    n_buffers = sum(1 for _ in fresh.buffers())
    if len(backbone) != n_params + n_buffers:
        raise ValueError("exported keys do not match parameters + buffers of MobileNetV2 features")

    run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    manifests = json.loads((run_dir / "manifests.json").read_text(encoding="utf-8"))
    out = run_dir / f"backbone_{which}_epoch{ckpt['epoch']}.pth"
    torch.save({
        "backbone_state_dict": backbone,
        "source": which,
        "ssl_method": "DINO (adapted to MobileNetV2, single GPU; see DINO_CHANGES.md)",
        "ssl_run_dir": run_dir.name,
        "ssl_checkpoint_sha256": sha256_file(ckpt_path),
        "ssl_epoch": ckpt["epoch"],
        "ssl_epochs_planned": ckpt["args"]["epochs"],
        "ssl_seed": ckpt["args"]["seed"],
        "ssl_config": ckpt["args"],
        "ssl_manifest": manifests,
        "ssl_cost": run_info.get("cost", {}),
        "initialization_before_ssl": run_info.get("initialization", {}),
        "num_parameter_tensors": n_params,
        "num_buffer_tensors": n_buffers,
        "excluded_from_export": dropped,
        "exported": datetime.now().isoformat(timespec="seconds"),
    }, out)
    # The downstream code loads with weights_only=True; make sure that works for this file.
    torch.load(out, map_location="cpu", weights_only=True)
    print(f"Exported {which} backbone (epoch {ckpt['epoch']}): {len(backbone)} tensors "
          f"({n_params} parameters, {n_buffers} BatchNorm buffers), loaded strictly into a fresh MobileNetV2.\n"
          f"  excluded: {', '.join(dropped)}\n  file: {out}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Export a DINO run's MobileNetV2 backbone")
    parser.add_argument("--run", type=Path, required=True, help="SSL run folder containing checkpoint.pth")
    parser.add_argument("--which", choices=["teacher", "student"], default="teacher")
    args = parser.parse_args()
    export(args.run, args.which)


if __name__ == "__main__":
    main()
