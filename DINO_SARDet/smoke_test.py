"""DINO_SARDet smoke test: shared checks, a 2-epoch DINO run on 48 images, export, fine-tune from it, probes, panels.

    python DINO_SARDet/smoke_test.py [--cpu]
Pipeline check only (outputs go to DINO_SARDet/outputs/smoke/); never a result.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared import smoke_common as sc  # noqa: E402
from shared import ssl_engine  # noqa: E402

sys.path.insert(0, str(HERE))
from train_ssl import DINOMethod  # noqa: E402


def check_dino_loss_matches_reference():
    """The DINO loss with one teacher view pair must equal a hand-computed cross-entropy."""
    sc.step("DINO loss sanity")
    from dino import DINOLoss
    torch.manual_seed(0)
    loss_fn = DINOLoss(out_dim=16, ncrops=3, warmup_teacher_temp=0.04, teacher_temp=0.04, warmup_teacher_temp_epochs=0, nepochs=1)
    t = torch.randn(4, 16)            # 2 views x batch 2
    s = torch.randn(6, 16)            # 3 views x batch 2
    loss = loss_fn(s, t, 0)
    q = torch.softmax(t / 0.04, -1).chunk(2)
    p = torch.log_softmax(s / 0.1, -1).chunk(3)
    ref = sum((-(q[iq] * p[v]).sum(-1)).mean() for iq in range(2) for v in range(3) if v != iq) / 4
    assert torch.allclose(loss, ref, atol=1e-5), (loss, ref)
    print("  ok")


if __name__ == "__main__":
    device = sc.device_arg()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sc.check_paths()
    sc.check_detection_data()
    sc.check_detector(dev)
    check_dino_loss_matches_reference()
    sc.check_ssl_views([("global", 1, 1.0), ("global", 1, 0.1), ("local", 8, 0.5)])
    sc.step("DINO SSL smoke run (48 images, 2 epochs)")
    run_dir = ssl_engine.run(DINOMethod, HERE, smoke=True, device=device, log_every_steps=1)
    backbone = next(run_dir.glob("backbone_epoch*.pth"))
    sc.step("resume check (stop after epoch 1, then resume)")
    run2 = ssl_engine.run(DINOMethod, HERE, smoke=True, device=device, stop_after_epoch=1, log_every_steps=0)
    ssl_engine.run(DINOMethod, HERE, resume_dir=run2, device=device, log_every_steps=0)
    assert next(run2.glob("backbone_epoch*.pth")).exists()
    ft = sc.run_finetune_smoke(HERE, "dino_mobilenetv2", backbone, device=device)
    sc.run_analysis_smoke(ft, backbone, device=device)
    print("\nSMOKE TEST PASSED (DINO_SARDet)")
