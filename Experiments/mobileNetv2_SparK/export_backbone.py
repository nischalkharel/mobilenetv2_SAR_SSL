"""Export the MobileNetV2 backbone (SparK's encoder) of a SparK run for supervised fine-tuning.

Run from the repository root (train_ssl.py already does this automatically after the last epoch,
and every setup.export_every epochs):
    python Experiments/mobileNetv2_SparK/export_backbone.py --run Experiments/mobileNetv2_SparK/outputs/ssl/<run folder>

SparK trains one encoder plus a decoder (no teacher). The backbone used for route A3 is the encoder of
the FINAL epoch, fixed in advance; files from earlier epochs are for the accuracy-versus-epoch analysis
only and must never be chosen by test accuracy.

Output file <run folder>/backbone_epoch<NNN>.pth (same contract as the A1 and A2 exports):
    {"backbone_state_dict": <MobileNetV2 features.state_dict()>,   parameters AND BatchNorm buffers
     ...metadata: source run, checkpoint hash, config, seed, manifest hashes, SSL cost...}
Excluded: the decoder, the densify norms and projections, and the mask tokens (the authors' export,
pretrain/main.py L106, also keeps only the encoder). During SSL the encoder's BatchNorm layers are the
authors' sparse version (BatchNorm1d over visible positions); they hold exactly the parameters and
buffers of an ordinary BatchNorm2d under the same names, so the file is checked by loading it strictly
into a fresh torchvision MobileNetV2 `features`.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torchvision.models import mobilenet_v2

from utils import sha256_file

PREFIX = "sparse_encoder.sp_cnn.features."  # key prefix of MobileNetV2's features inside the SparK model


def export(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "checkpoint.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # our own file: also holds RNG states
    state = ckpt["model"]
    backbone = {k[len(PREFIX):]: v for k, v in state.items() if k.startswith(PREFIX)}
    dropped = sorted({k.split(".")[0] for k in state if not k.startswith(PREFIX)})
    if any(k.startswith("sparse_encoder.") and not k.startswith(PREFIX) for k in state):
        raise ValueError("the encoder holds tensors outside MobileNetV2's features; refusing to export a partial encoder")

    # Strict load into a fresh downstream backbone: every parameter and buffer must match by name and shape.
    fresh = mobilenet_v2(weights=None).features
    fresh.load_state_dict(backbone, strict=True)
    n_params = sum(1 for _ in fresh.parameters())
    n_buffers = sum(1 for _ in fresh.buffers())
    if len(backbone) != n_params + n_buffers:
        raise ValueError("exported keys do not match parameters + buffers of MobileNetV2 features")

    run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    manifests = json.loads((run_dir / "manifests.json").read_text(encoding="utf-8"))
    out = run_dir / f"backbone_epoch{ckpt['epoch']:03d}.pth"
    torch.save({
        "backbone_state_dict": backbone,
        "source": "spark encoder (MobileNetV2 features; decoder side discarded)",
        "ssl_method": "SparK (MobileNetV2 integration, single GPU; see SPARK_CHANGES.md)",
        "ssl_run_dir": run_dir.name,
        "ssl_checkpoint_sha256": sha256_file(ckpt_path),
        "ssl_epoch": ckpt["epoch"],
        "ssl_epochs_planned": ckpt["args"]["epochs"],
        "is_final_epoch": ckpt["epoch"] == ckpt["args"]["epochs"],
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
    print(f"  exported backbone (epoch {ckpt['epoch']}): {len(backbone)} tensors ({n_params} parameters, "
          f"{n_buffers} BatchNorm buffers), loaded strictly into a fresh MobileNetV2; excluded: "
          f"{', '.join(dropped)} -> {out.name}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Export a SparK run's MobileNetV2 backbone (latest checkpoint)")
    parser.add_argument("--run", type=Path, required=True, help="SSL run folder containing checkpoint.pth")
    args = parser.parse_args()
    export(args.run)


if __name__ == "__main__":
    main()
