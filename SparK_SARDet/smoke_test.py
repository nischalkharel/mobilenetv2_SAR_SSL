"""SparK_SARDet smoke test: shared checks, SparK sanity, a 2-epoch SparK run on 48 images, export, fine-tune, probes.

    python SparK_SARDet/smoke_test.py [--cpu]
Pipeline check only (outputs go to SparK_SARDet/outputs/smoke/); never a result.
"""
import copy
import hashlib
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision.models import mobilenet_v2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared import smoke_common as sc  # noqa: E402
from shared import ssl_engine  # noqa: E402
from shared.run_record import read_json  # noqa: E402

sys.path.insert(0, str(HERE))
from spark_lib import encoder as sparse  # noqa: E402
from spark_lib.spark import SparK  # noqa: E402
from spark_model import (DOWNSAMPLE_RATIO, FEATURE_STAGES, MobileNetV2Hierarchical, build_spark,  # noqa: E402
                         encoder_features_state)
from train_ssl import SparKMethod  # noqa: E402

OLD_SPARK_LIB = Path(r"C:\Users\nisch\Documents\SHREC\2026_2027\mobilenetv2_SAR_SSL\Experiments\mobileNetv2_SparK\spark_lib")
LIB_FILES = ["LICENSE", "decoder.py", "encoder.py", "lamb.py", "lr_control.py", "spark.py"]


def smoke_args():
    """The flat settings the engine would use (shared/ssl_common.json + config_ssl.json), smoke overrides applied."""
    cfg = ssl_engine.merged_config(read_json(HERE / "config_ssl.json"), smoke=True, smoke_overrides=SparKMethod.smoke_overrides)
    return ssl_engine.flat_args(cfg)


def upsample(active_b1ff, size):
    r = size // active_b1ff.shape[-1]
    return active_b1ff.repeat_interleave(r, 2).repeat_interleave(r, 3)


def check_spark_sanity():
    """Masking, sparse behaviour, loss and export contract of the SparK model (CPU, a few random inputs)."""
    sc.step("SparK sanity (mask, sparse encoder, loss, export keys)")
    args = smoke_args()
    SparKMethod(args)                                     # config validation
    S, f = args.global_size, args.global_size // DOWNSAMPLE_RATIO
    if OLD_SPARK_LIB.is_dir():                            # the authors' code must be the byte-identical ATRNet-STAR copy
        for name in LIB_FILES:
            a, b = (HERE / "spark_lib" / name).read_bytes(), (OLD_SPARK_LIB / name).read_bytes()
            assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest(), f"spark_lib/{name} differs from the ATRNet-STAR copy"
        print(f"  spark_lib: {len(LIB_FILES)} files byte-identical to the ATRNet-STAR A3 copies")
    torch.manual_seed(0)
    model = build_spark(args).eval()
    assert args.densify_norm == "bn" and all(type(m) is sparse.SparseBatchNorm2d for m in model.densify_norms), "densify norm is not sparse BN"
    cnn = model.sparse_encoder.sp_cnn
    convs = [m for m in cnn.modules() if isinstance(m, nn.Conv2d)]
    norms = [m for m in cnn.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    assert all(type(m) is sparse.SparseConv2d for m in convs) and all(type(m) is sparse.SparseBatchNorm2d for m in norms)
    assert sum(m.groups > 1 for m in convs) == 17 and all(m.groups == m.in_channels for m in convs if m.groups > 1), "depthwise convs lost their groups"
    # mask: exactly round(49 * 0.6) hidden patches per image, different per image
    mk = model.mask(16, torch.device("cpu"))
    hidden = (~mk).flatten(1).sum(1)
    assert tuple(mk.shape) == (16, 1, f, f) and mk.dtype == torch.bool
    assert (hidden == round(f * f * args.mask_ratio)).all(), hidden.tolist()
    assert model.len_keep == f * f - round(f * f * args.mask_ratio)
    assert len({tuple(m.flatten().tolist()) for m in mk}) == 16
    # dense vs sparse forward agree on a fully visible input (eval mode, the same ImageNet weights)
    dense = MobileNetV2Hierarchical(imagenet_init=True).eval()
    x = torch.randn(2, 3, S, S)
    with torch.no_grad():
        dense_maps = dense(x, hierarchical=True)
        sparse._cur_active = torch.ones(2, 1, f, f, dtype=torch.bool)
        sparse_maps = model.sparse_encoder(x)
    want = [(2, 24, 8 * f, 8 * f), (2, 32, 4 * f, 4 * f), (2, 96, 2 * f, 2 * f), (2, 1280, f, f)]
    assert [tuple(m.shape) for m in dense_maps] == want, [tuple(m.shape) for m in dense_maps]
    assert FEATURE_STAGES[-1] == len(dense.features) - 1
    d = max((a - b).abs().max().item() for a, b in zip(dense_maps, sparse_maps))
    assert d < 1e-4, f"dense and sparse encoder differ by {d}"
    # hidden positions stay exactly 0 after every encoder layer; the loss on hidden patches is finite
    probe = copy.deepcopy(model).train()
    xb = torch.randn(4, 3, S, S)
    active = probe.mask(4, torch.device("cpu"))
    outputs = []
    hooks = [m.register_forward_hook(lambda mod, inp, out: outputs.append(out)) for m in probe.sparse_encoder.sp_cnn.features.modules()]
    with torch.no_grad():
        loss, smallest, act = probe(xb, active_b1ff=active)
    for h in hooks:
        h.remove()
    maps4 = [o for o in outputs if torch.is_tensor(o) and o.ndim == 4]
    leak = max((o * ~upsample(active, o.shape[-1])).abs().max().item() for o in maps4)
    assert leak == 0.0, f"hidden positions are not 0 after every layer (max {leak})"
    assert torch.isfinite(loss) and loss.item() > 0 and tuple(smallest.shape) == (4, 1280, f, f) and torch.equal(act, active)
    # recon_channels = 3 gives the authors' loss exactly; recon_channels = 1 targets channel 0 (the SAR channel)
    args3 = copy.copy(args)
    args3.recon_channels = 3
    torch.manual_seed(2)
    m3 = build_spark(args3).train()
    x2 = torch.randn(2, 3, S, S)
    mk2 = m3.mask(2, torch.device("cpu"))
    with torch.no_grad():
        ours = m3(x2, active_b1ff=mk2)[0]
        authors = SparK.forward(m3, x2, active_b1ff=mk2)
    assert torch.allclose(ours, authors, rtol=1e-6, atol=0), (ours, authors)
    with torch.no_grad():
        target, _, _ = probe(x2, active_b1ff=mk2, vis=True)
    assert torch.equal(target, x2[:, :1])
    # export contract: torchvision features keys incl. BN buffers, strict load, values = the ImageNet file
    state = encoder_features_state(model)
    fresh = mobilenet_v2(weights=None).features
    assert set(state) == set(fresh.state_dict()), "exported keys differ from torchvision mobilenet_v2().features"
    assert len(state) == len(list(fresh.parameters())) + len(list(fresh.buffers()))
    fresh.load_state_dict(state, strict=True)
    imagenet = dense.features.state_dict()
    assert all(torch.equal(state[k], imagenet[k]) for k in state), "encoder does not start from the ImageNet file"
    print(f"  ok: {f}x{f} mask hides {hidden[0].item()}/{f * f} patches; dense-sparse max diff {d:.1e}; loss {loss.item():.4f}; "
          f"authors' loss {authors.item():.6f} == ours {ours.item():.6f}; export {len(state)} tensors")


if __name__ == "__main__":
    device = sc.device_arg()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sc.check_paths()
    sc.check_detection_data()
    sc.check_detector(dev)
    check_spark_sanity()
    sc.check_ssl_views([("global", 1, 0.0)])
    tf32_defaults = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    sc.step("SparK SSL smoke run (48 images, 2 epochs)")
    run_dir = ssl_engine.run(SparKMethod, HERE, smoke=True, device=device, log_every_steps=1)
    backbone = next(run_dir.glob("backbone_epoch*.pth"))
    for e in (1, 2):
        assert (run_dir / f"reconstruction_epoch{e:03d}.png").exists(), f"missing reconstruction_epoch{e:03d}.png"
    sc.step("resume check (stop after epoch 1, then resume)")
    run2 = ssl_engine.run(SparKMethod, HERE, smoke=True, device=device, stop_after_epoch=1, log_every_steps=0)
    ssl_engine.run(SparKMethod, HERE, resume_dir=run2, device=device, log_every_steps=0)
    assert next(run2.glob("backbone_epoch*.pth")).exists()
    # SparK switched TF32 off for its own fp32 training; the fine-tune protocol uses PyTorch's default flags
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32_defaults
    ft = sc.run_finetune_smoke(HERE, "spark_mobilenetv2", backbone, device=device)
    sc.run_analysis_smoke(ft, backbone, device=device)
    print("\nSMOKE TEST PASSED (SparK_SARDet)")
