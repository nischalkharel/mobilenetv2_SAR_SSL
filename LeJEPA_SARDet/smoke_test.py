"""LeJEPA_SARDet smoke test: shared checks, a 2-epoch LeJEPA run on 48 images, export, fine-tune from it, probes, panels.

    python LeJEPA_SARDet/smoke_test.py [--cpu]
Pipeline check only (outputs go to LeJEPA_SARDet/outputs/smoke/); never a result.
"""
import argparse
import sys
from pathlib import Path

import torch
from torchvision.models import mobilenet_v2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared import smoke_common as sc  # noqa: E402
from shared import ssl_engine  # noqa: E402

sys.path.insert(0, str(HERE))
from train_ssl import LeJEPAMethod  # noqa: E402

SIGREG_ARGS = argparse.Namespace(t_max=3.0, n_points=17, num_slices=1024)


def check_lejepa_loss_sanity():
    """SIGReg is small for standard-normal data and large for collapsed data; agreement of identical views is 0;
    the total is the stated mix; the lr schedule has the stated shape."""
    sc.step("LeJEPA loss sanity")
    from lejepa_model import build_sigreg, lejepa_loss, lr_schedule
    torch.manual_seed(0)
    V, N, K = 8, 256, 64
    gauss = torch.randn(V, N, K)
    s_gauss = build_sigreg(SIGREG_ARGS)(gauss).item()
    s_const = build_sigreg(SIGREG_ARGS)(torch.ones(V, N, K)).item()          # every image the same point
    s_small = build_sigreg(SIGREG_ARGS)(0.01 * torch.randn(V, N, K)).item()  # squashed to a tiny cloud
    assert s_gauss < 2.0, f"SIGReg of standard-normal data should be about 1, got {s_gauss}"
    assert s_const > 5 * s_gauss and s_small > 5 * s_gauss, (s_gauss, s_const, s_small)
    same = gauss[:1].expand(V, N, K)                                           # identical views of each image
    total, agreement, sig = lejepa_loss(same, V, build_sigreg(SIGREG_ARGS), 0.05)
    assert agreement.item() == 0.0, agreement
    total, agreement, sig = lejepa_loss(gauss, V, build_sigreg(SIGREG_ARGS), 0.05)
    assert torch.allclose(agreement, (gauss.mean(0) - gauss).square().mean()), "agreement is not MINIMAL.md's inv_loss"
    assert torch.allclose(total, 0.05 * sig + 0.95 * agreement), "total is not lamb * SIGReg + (1 - lamb) * agreement"
    sr = build_sigreg(SIGREG_ARGS)
    sr(gauss)
    assert sr.global_step.item() == 1, "SIGReg did not advance its direction seed"
    lrs = lr_schedule(5e-4, total_steps=1000, warmup_steps=100, start_factor=0.01, final_ratio=0.001)
    assert abs(lrs[0] - 5e-6) < 1e-12 and abs(lrs[100] - 5e-4) < 1e-12 and lrs[-1] > 5e-7 and lrs[-1] < 5e-7 * 1.01, lrs[-1]
    print(f"  ok: SIGReg gaussian {s_gauss:.3f}, constant {s_const:.1f}, squashed {s_small:.1f}; agreement(identical) 0; "
          f"lr {lrs[0]:.1e} -> {lrs[100]:.1e} -> {lrs[-1]:.2e}")


def check_export_keys():
    """The exported encoder state has exactly torchvision mobilenet_v2().features key names (params + BN buffers)."""
    sc.step("export key names")
    from shared.backbone import strip_prefix
    from lejepa_model import LeJEPANet
    net = LeJEPANet(imagenet_init=False, proj_hidden_dim=2048, proj_dim=512)
    state = strip_prefix(net.state_dict(), "backbone.features.")
    ref = mobilenet_v2(weights=None).features.state_dict()
    assert list(state) == list(ref), "exported keys differ from torchvision features keys"
    assert all(state[k].shape == ref[k].shape for k in ref)
    print(f"  ok: {len(state)} tensors, same names and shapes as torchvision features")


def check_sigreg_step_in_checkpoint(run_dir):
    """After a resumed run the SIGReg direction seed equals the global optimizer step (no step lost or repeated)."""
    ckpt = torch.load(Path(run_dir) / "checkpoint.pth", map_location="cpu", weights_only=False)
    g = ckpt["method"]["sigreg"]["global_step"].item()
    assert g == ckpt["global_step"], (g, ckpt["global_step"])
    print(f"  ok: SIGReg global_step {g} == optimizer steps {ckpt['global_step']} after resume")


if __name__ == "__main__":
    device = sc.device_arg()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sc.check_paths()
    sc.check_detection_data()
    sc.check_detector(dev)
    check_lejepa_loss_sanity()
    check_export_keys()
    sc.check_ssl_views([("global", 8, 0.5)])
    sc.step("LeJEPA SSL smoke run (48 images, 2 epochs)")
    run_dir = ssl_engine.run(LeJEPAMethod, HERE, smoke=True, device=device, log_every_steps=1)
    backbone = next(run_dir.glob("backbone_epoch*.pth"))
    sc.step("resume check (stop after epoch 1, then resume)")
    run2 = ssl_engine.run(LeJEPAMethod, HERE, smoke=True, device=device, stop_after_epoch=1, log_every_steps=0)
    ssl_engine.run(LeJEPAMethod, HERE, resume_dir=run2, device=device, log_every_steps=0)
    assert next(run2.glob("backbone_epoch*.pth")).exists()
    check_sigreg_step_in_checkpoint(run2)
    ft = sc.run_finetune_smoke(HERE, "lejepa_mobilenetv2", backbone, device=device)
    sc.run_analysis_smoke(ft, backbone, device=device)
    print("\nSMOKE TEST PASSED (LeJEPA_SARDet)")
