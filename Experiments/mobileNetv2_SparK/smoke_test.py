"""Checks of the whole A3 (MobileNetV2 + SparK) pipeline before any long run. Not a result.

Run from the repository root:
    python Experiments/mobileNetv2_SparK/smoke_test.py
    python Experiments/mobileNetv2_SparK/smoke_test.py --device cpu

Everything it writes goes to outputs/smoke/ (folders ending in _SMOKE); delete them freely.
It never touches the official test set. Checks, in order:
  1. Data: split audit, counts and CSV hashes equal to those recorded by the A0 run; an SSL item is one
     uint8 view and no label; the GPU-side input equals the A0/A1/A2 preprocessing on the same view.
  2. Sources: the reference copies still have their recorded hashes; lamb.py, lr_control.py and LICENSE
     are byte-identical to the authors'; encoder.py, spark.py and decoder.py differ only in the
     documented import lines.
  3. Adapter, conversion, initialization: 4 maps (24@56, 32@28, 96@14, 1280@7), each the last layer of
     its resolution; every conv and BatchNorm of the encoder became the authors' sparse version, depthwise
     groups kept; encoder = ImageNet IMAGENET1K_V2 weights; mask tokens, densify layers and decoder sizes.
  4. Sparse behaviour: masked positions stay exactly 0 after every layer of the encoder (no "mask
     pattern vanishing"); sparse BatchNorm normalizes with visible positions only; with nothing masked
     the sparse encoder equals the ordinary MobileNetV2 (the paper's "dense = special case").
  5. Masking and loss: 20 of 49 patches visible per image; with recon_channels = 3 the loss equals the
     authors' SparK.forward exactly; with 1 the target is the SAR channel; only hidden patches count.
  6. One real training step on SSL images: the GPU computes in true float32 (TF32 off); finite loss and
     gradients; every weight receives a non-zero gradient; every encoder, densify, mask-token and decoder
     parameter updated; the authors' lr/weight-decay schedule and parameter groups; BatchNorm statistics
     updated once.
  7. Save/resume on the CPU (deterministic, so the check is exact): 2 epochs straight vs 1 epoch + resume
     give identical model, optimizer state and history, even after a simulated stop right after the
     checkpoint (history row, cost and export are rebuilt); a changed config is rejected. Then the same
     smoke run once on the GPU, starting from TF32 on, which run() must switch off.
  8. Export and pictures: backbone files after each epoch (smoke export_every = 1); 156 parameters + 156
     buffers, nothing from the decoder side, loads strictly, equals the checkpoint's encoder; the view
     and reconstruction previews exist.
  9. Downstream: a tiny fine-tuning run from that backbone; its best.pt reproduces the recorded
     validation accuracy, and evaluate.py --split validation reproduces it too.
 10. Imports: every module came from this folder, none from Baseline, DINO, LeJEPA or SparK-main.
"""
import subprocess
import sys

if __name__ == "__main__" and "--part" not in sys.argv:
    # Launcher: run the two parts as separate processes, one after the other (as in the A1/A2 smoke tests:
    # this process imports nothing heavy, so the SSL part's memory is returned before fine-tuning starts).
    for part in ["ssl", "downstream"]:
        code = subprocess.run([sys.executable, __file__, "--part", part, *sys.argv[1:]]).returncode
        if code:
            sys.exit(f"SMOKE TEST FAILED in the {part} part (exit code {code})")
    print("\nSMOKE TEST PASSED (both parts)")
    sys.exit(0)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import transforms as T  # noqa: E402
from torchvision.transforms import functional as TF  # noqa: E402

import paths  # noqa: E402
import train_ssl  # noqa: E402
import train_supervised  # noqa: E402
from data import SARDataset, build_transform, load_class_names, read_manifest  # noqa: E402
from engine import predict  # noqa: E402
from evaluate import evaluate  # noqa: E402
from model import build_model, load_finetuned  # noqa: E402
from spark_lib import encoder as sparse  # noqa: E402
from spark_lib.lamb import TheSameAsTimmLAMB  # noqa: E402
from spark_lib.lr_control import get_param_groups  # noqa: E402
from spark_lib.spark import SparK  # noqa: E402
from spark_model import (DOWNSAMPLE_RATIO, FEATURE_STAGES, DataAugmentationSparK, MobileNetV2Hierarchical,  # noqa: E402
                         build_spark, to_network_input)
from ssl_data import SSLDataset, audit_splits, flat_args, read_ssl_paths  # noqa: E402
from utils import resolve_device, save_json, sha256_file  # noqa: E402

REF = paths.EXPERIMENT_DIR / "reference" / "SparK-main_original"
LATEST = paths.SMOKE_OUTPUT_DIR / "latest_smoke_ssl.json"  # hands the SSL part's export to the downstream part
# Recorded by the A0 run (Baseline_Output/20261013-150506_A0_imagenet_mobilenetv2_seed0/manifests.json).
A0_RECORD = {"labeled_set.csv": ("3eefc78f9eb254d3e86d355c72e22c60097869ad032734b1b38194cc51c8c767", 1961),
             "validation.csv": ("c752adadce615c11ffaefb0f187f144042d83cb070d9c9c34775ab8b989a9980", 800),
             "class_mapping.csv": ("5768e0346b908a22e11e25f34fbdfbc167783c9ccb33f434353a10c268a4427d", 40)}
SAME_AS_AUTHORS = {"spark_lib/lamb.py": "pretrain/utils/lamb.py",
                   "spark_lib/lr_control.py": "pretrain/utils/lr_control.py",
                   "spark_lib/LICENSE": "LICENSE"}
# file -> the authors' lines that were replaced (only import lines); every other line must be identical
CHANGED_LINES = {"spark_lib/encoder.py": ("pretrain/encoder.py", {"from timm.models.layers import DropPath"}),
                 "spark_lib/spark.py": ("pretrain/spark.py", {"from timm.models.layers import trunc_normal_",
                                                              "import encoder", "from decoder import LightDecoder"}),
                 "spark_lib/decoder.py": ("pretrain/decoder.py", {"from timm.models.layers import trunc_normal_",
                                                                  "from utils.misc import is_pow2n"})}
FAILED = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILED.append(name)


def max_diff(a: dict, b: dict) -> float:
    return max(((a[k].float() - b[k].float()).abs().max().item() if a[k].numel() else 0.0) for k in a)


def upsample(active_b1ff: torch.Tensor, size: int) -> torch.Tensor:
    """(B, 1, f, f) mask -> (B, 1, size, size), as the sparse layers do it (encoder.py L14-L17)."""
    r = size // active_b1ff.shape[-1]
    return active_b1ff.repeat_interleave(r, 2).repeat_interleave(r, 3)


class FixedOutput(nn.Module):
    """Stands in for the decoder: returns a given reconstruction, so the loss can be checked by hand."""

    def __init__(self, rec):
        super().__init__()
        self.rec = rec

    def forward(self, to_dec):
        return self.rec


def imports_check():
    print("Imports")
    here = paths.EXPERIMENT_DIR.resolve()
    local = [p.stem for p in here.glob("*.py")]
    loaded = {n: Path(sys.modules[n].__file__).resolve().parent for n in local if n in sys.modules}
    check("every project module in use was loaded from this folder", all(d == here for d in loaded.values()),
          f"({', '.join(sorted(loaded))})")
    lib = Path(sys.modules["spark_lib"].__file__).resolve().parent if "spark_lib" in sys.modules else None
    check("SparK code loaded from this folder's spark_lib/", lib is None or lib == here / "spark_lib")
    bad = ("Baseline", "mobileNetv2_DINO", "mobileNetv2_LeJEPA", "SparK-main")
    check("no module from Baseline, the DINO or LeJEPA experiment, or SparK-main",
          not any(any(b in str(getattr(m, "__file__", "") or "") for b in bad) for m in list(sys.modules.values())))


def finish(part, folders):
    print(f"\n{'PASSED' if not FAILED else 'FAILED: ' + '; '.join(FAILED)} ({part} part)\n"
          f"Smoke folders (not results): {folders} in {paths.SMOKE_OUTPUT_DIR}", flush=True)
    if FAILED:
        sys.exit(1)


def part_ssl(cli):
    """Sections 1 to 8."""
    device = resolve_device(cli.device)
    base_cfg = json.loads(train_ssl.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    base_cfg["setup"]["device"] = cli.device
    args = flat_args(base_cfg)
    f = args.input_size // DOWNSAMPLE_RATIO
    print(f"Smoke test on {device}. Outputs go to {paths.SMOKE_OUTPUT_DIR}\n")

    print("1. Data")
    audit = audit_splits()
    check("split audit (leakage and consistency checks)", True, f"({len(audit['checks'])} checks, counts {audit['counts']})")
    count_key = {"labeled_set.csv": "labeled", "validation.csv": "validation", "class_mapping.csv": "classes"}
    for name, (sha, n) in A0_RECORD.items():
        same = audit["csv_sha256"][name] == sha
        count = audit["counts"][count_key[name]]
        check(f"{name} identical to the A0 run's file", same and count == n,
              f"({count} rows/classes, sha256 {'matches' if same else 'DIFFERS'})")
    ssl_paths = read_ssl_paths(paths.SPLITS_DIR / args.ssl_csv)
    data = SSLDataset(ssl_paths[:4], DataAugmentationSparK(args))
    views = [data[i] for i in range(4)]
    check("SSL item is one uint8 gray view and no label", all(torch.is_tensor(v) and v.dtype == torch.uint8 and
          tuple(v.shape) == (1, args.input_size, args.input_size) for v in views), f"({tuple(views[0].shape)})")
    net_in = to_network_input(torch.stack(views), args.normalize_mean, args.normalize_std, device).cpu()
    a0_way = T.Compose([T.Grayscale(num_output_channels=3), T.ToTensor(), T.Normalize(args.normalize_mean, args.normalize_std)])
    expected = torch.stack([a0_way(TF.to_pil_image(v)) for v in views])
    err = (net_in - expected).abs().max().item()
    check("GPU-side input equals the A0/A1/A2 pipeline (Grayscale(3) -> ToTensor -> Normalize) on the same views",
          tuple(net_in.shape) == tuple(expected.shape) and err < 1e-5, f"(max difference {err:.1e})")

    print("2. Sources")
    recorded = dict(re.findall(r"^\| (\S+) \| ([0-9a-f]{64}) \|\r?$", (paths.EXPERIMENT_DIR / "reference" / "SOURCE.md")
                               .read_text(encoding="utf-8"), re.M))
    check("reference copies have the hashes recorded in reference/SOURCE.md (commit a63e386)",
          len(recorded) == 18 and all(sha256_file(REF / k) == v for k, v in recorded.items()), f"({len(recorded)} files)")
    for ours, orig in SAME_AS_AUTHORS.items():
        check(f"{ours} byte-identical to the authors' {orig}", sha256_file(paths.EXPERIMENT_DIR / ours) == sha256_file(REF / orig))
    for ours, (orig, replaced) in CHANGED_LINES.items():
        a = (REF / orig).read_text(encoding="utf-8").split("\n")
        b = (paths.EXPERIMENT_DIR / ours).read_text(encoding="utf-8").split("\n")
        diff = [i for i in range(min(len(a), len(b))) if a[i] != b[i]]
        ok = (len(a) == len(b) and {a[i] for i in diff} == replaced and len(diff) == len(replaced)
              and all("# CHANGED (A3)" in b[i] for i in diff))
        check(f"{ours} differs from the authors' {orig} only in its {len(replaced)} import line(s)", ok,
              f"(lines {[i + 1 for i in diff]})")

    print("3. Adapter, conversion and initialization")
    imagenet = torch.load(paths.MOBILENETV2_IMAGENET_WEIGHTS, map_location="cpu", weights_only=True)
    print(f"  ImageNet weights sha256 {sha256_file(paths.MOBILENETV2_IMAGENET_WEIGHTS)[:16]}...")
    dense = MobileNetV2Hierarchical(imagenet_init=True).eval()
    x = torch.randn(2, 3, args.input_size, args.input_size)
    with torch.no_grad():
        maps = dense(x, hierarchical=True)
        sizes, h = [], x
        for layer in dense.features:
            h = layer(h)
            sizes.append(h.shape[-1])
    want = [(2, 24, 8 * f, 8 * f), (2, 32, 4 * f, 4 * f), (2, 96, 2 * f, 2 * f), (2, 1280, f, f)]
    check("4 feature maps 24@56, 32@28, 96@14, 1280@7 (strides 4, 8, 16, 32)", [tuple(m.shape) for m in maps] == want,
          f"({[tuple(m.shape)[1:] for m in maps]})")
    check("get_feature_map_channels() and get_downsample_ratio() agree with the maps",
          dense.get_feature_map_channels() == [m.shape[1] for m in maps]
          and dense.get_downsample_ratio() * maps[-1].shape[-1] == args.input_size)
    check("each stage is the last layer of its resolution; the last one is features[18]",
          all(sizes[i + 1] < sizes[i] for i in FEATURE_STAGES[:-1]) and FEATURE_STAGES[-1] == len(dense.features) - 1)
    torch.manual_seed(0)
    model = build_spark(args)
    cnn = model.sparse_encoder.sp_cnn
    convs = [m for m in cnn.modules() if isinstance(m, nn.Conv2d)]
    norms = [m for m in cnn.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    check("every encoder conv is the authors' SparseConv2d and every BatchNorm their SparseBatchNorm2d",
          len(convs) == 52 and all(type(m) is sparse.SparseConv2d for m in convs)
          and len(norms) == 52 and all(type(m) is sparse.SparseBatchNorm2d for m in norms),
          f"({len(convs)} convs, {len(norms)} BatchNorms)")
    dw = [m for m in convs if m.groups > 1]
    check("17 depthwise convs keep groups = channels", len(dw) == 17 and all(m.groups == m.in_channels == m.out_channels for m in dw))
    enc = cnn.state_dict()
    check("encoder equals the ImageNet IMAGENET1K_V2 weights (parameters and BatchNorm buffers, same keys)",
          {k for k in imagenet if k.startswith("features.")} == set(enc)
          and all(torch.equal(enc[k], imagenet[k]) for k in enc))
    state = model.state_dict()
    check("no ImageNet classifier in the SSL model", not any("classifier" in k for k in state))
    check("mask tokens for 1280, 96, 32, 24 channels (smallest map first)",
          [t.shape[1] for t in model.mask_tokens] == [1280, 96, 32, 24])
    check("densify: sparse BatchNorm + projections 1x1 1280->768, 3x3 96->384, 32->192, 24->96",
          all(type(m) is sparse.SparseBatchNorm2d for m in model.densify_norms)
          and [(p.in_channels, p.out_channels, p.kernel_size[0]) for p in model.densify_projs]
          == [(1280, 768, 1), (96, 384, 3), (32, 192, 3), (24, 96, 3)])
    dec = model.dense_decoder
    check(f"decoder: 5 upsampling blocks 768 -> 24 channels, output {args.recon_channels} channel(s) at {args.input_size} px",
          [b.conv[0].in_channels for b in dec.dec] == [768, 384, 192, 96, 48] and dec.proj.in_channels == 24
          and dec.proj.out_channels == args.recon_channels)
    check("every parameter trainable", all(p.requires_grad for p in model.parameters()))
    check("unused buffers imn_m, imn_s zeroed (were uninitialized memory)",
          not model.imn_m.any() and not model.imn_s.any() and not model.norm_black.any())

    print("4. Sparse behaviour")
    torch.manual_seed(1)
    probe = copy.deepcopy(model).train()
    xb = torch.randn(4, 3, args.input_size, args.input_size)
    active = probe.mask(4, torch.device("cpu"))
    outputs = []
    hooks = [m.register_forward_hook(lambda mod, inp, out: outputs.append(out))
             for m in probe.sparse_encoder.sp_cnn.features.modules()]
    with torch.no_grad():
        probe(xb, active_b1ff=active)
    for hk in hooks:
        hk.remove()
    maps4 = [o for o in outputs if torch.is_tensor(o) and o.ndim == 4]
    leak = max((o * ~upsample(active, o.shape[-1])).abs().max().item() for o in maps4)
    check("hidden positions stay exactly 0 after every encoder layer (mask pattern kept)", leak == 0.0,
          f"({len(maps4)} layer outputs checked, largest value at a hidden position {leak})")
    bn = copy.deepcopy(cnn.features[0][1]).train()
    xs = torch.randn(4, bn.num_features, 112, 112) * 3 + 1
    m112 = upsample(active, 112)
    sparse._cur_active = active
    before = bn.running_mean.clone()
    out = bn(xs * m112)
    vis = (xs * m112).permute(0, 2, 3, 1)[m112[:, 0]]  # (visible positions, C)
    mu, var = vis.mean(0), vis.var(0, unbiased=False)
    ref = ((xs.permute(0, 2, 3, 1) - mu) / torch.sqrt(var + bn.eps) * bn.weight + bn.bias).permute(0, 3, 1, 2) * m112
    check("sparse BatchNorm uses the statistics of visible positions only (output and running mean)",
          torch.allclose(out, ref, atol=1e-4) and torch.allclose(bn.running_mean, (1 - bn.momentum) * before + bn.momentum * mu, atol=1e-5)
          and (out * ~m112).abs().max().item() == 0.0)
    probe = copy.deepcopy(model).eval()  # a fresh copy: the probe above changed its BatchNorm running stats
    with torch.no_grad():
        sparse._cur_active = torch.ones(2, 1, f, f, dtype=torch.bool)
        sp_maps = probe.sparse_encoder(x)
    d = max((a - b).abs().max().item() for a, b in zip(sp_maps, maps))
    check("with nothing masked the sparse encoder equals the ordinary MobileNetV2 (eval mode)", d < 1e-4,
          f"(max difference {d:.1e})")
    del probe

    print("5. Masking and loss")
    mk = model.mask(16, torch.device("cpu"))
    check(f"mask: {f} x {f} patches, exactly {model.len_keep} visible per image, different per image",
          tuple(mk.shape) == (16, 1, f, f) and mk.dtype == torch.bool and model.len_keep == round(f * f * (1 - args.mask_ratio))
          and all(n == model.len_keep for n in mk.flatten(1).sum(1).tolist()) and len({tuple(m.flatten().tolist()) for m in mk}) == 16)
    args3 = copy.copy(args)
    args3.recon_channels = 3
    torch.manual_seed(2)
    m3 = build_spark(args3).train()
    x2 = torch.randn(2, 3, args.input_size, args.input_size)
    mk2 = m3.mask(2, torch.device("cpu"))
    with torch.no_grad():
        ours = m3(x2, active_b1ff=mk2)[0]
        authors = SparK.forward(m3, x2, active_b1ff=mk2)
    check("recon_channels = 3: loss equals the authors' SparK.forward (same input, same mask)",
          torch.allclose(ours, authors, rtol=1e-6, atol=0), f"({ours.item():.6f} vs {authors.item():.6f})")
    del m3
    probe = copy.deepcopy(model).train()
    with torch.no_grad():
        target, _, _ = probe(x2, active_b1ff=mk2, vis=True)
    check("recon_channels = 1: the target is channel 0 of the input (the SAR channel)", torch.equal(target, x2[:, :1]))
    gray = torch.stack([v.float() / 255 for v in views[:2]])  # (2, 1, H, W) the real SAR views
    net = to_network_input(torch.stack(views[:2]), args.normalize_mean, args.normalize_std, torch.device("cpu"))
    norm = lambda t: (lambda p: (p - p.mean(-1, keepdim=True)) / (p.var(-1, keepdim=True) + 1e-6) ** .5)(probe.patchify(t))  # noqa: E731
    textured = probe.patchify(gray).std(-1) >= 0.03
    dn = (norm(net[:, :1]) - norm(gray))[textured].abs().max().item()
    check("channel 0 normalized per patch = the SAR image normalized per patch (patches with gray std >= 0.03)",
          dn < 1e-2, f"(max difference {dn:.1e}; only the 1e-6 epsilon differs)")
    rec = torch.randn(2, 1, args.input_size, args.input_size)
    probe.dense_decoder = FixedOutput(rec)
    with torch.no_grad():
        l1 = probe(x2, active_b1ff=mk2)[0]
        probe.dense_decoder = FixedOutput(rec + 100 * upsample(mk2, args.input_size))  # change only VISIBLE patches
        l2 = probe(x2, active_b1ff=mk2)[0]
    per_patch = ((probe.patchify(rec) - norm(x2[:, :1])) ** 2).mean(-1)  # (B, L)
    hidden = ~mk2.flatten(1)
    manual = per_patch[hidden].mean()
    check("loss = mean squared error over hidden patches only (changing visible patches changes nothing)",
          torch.equal(l1, l2) and torch.allclose(l1, manual, rtol=1e-5), f"({l1.item():.5f} vs manual {manual.item():.5f})")
    del probe

    print("6. One real training step (8 SSL images, the configured view)")
    loader = DataLoader(SSLDataset(ssl_paths[:8], DataAugmentationSparK(args)), batch_size=8)
    if device.type == "cuda":
        train_ssl.use_true_fp32()
        batch = next(iter(loader))
        mk8 = model.mask(8, torch.device("cpu"))
        with torch.no_grad():
            l_cpu = copy.deepcopy(model).train()(to_network_input(batch, args.normalize_mean, args.normalize_std,
                                                                  torch.device("cpu")), active_b1ff=mk8)[0].item()
            l_gpu = copy.deepcopy(model).to(device).train()(to_network_input(batch, args.normalize_mean, args.normalize_std,
                                                                             device), active_b1ff=mk8.to(device))[0].item()
        check("GPU computes in true float32 (TF32 off): loss equals the CPU loss within 5e-6", abs(l_gpu - l_cpu) < 5e-6,
              f"({l_gpu:.8f} vs {l_cpu:.8f}; float32 on the PC GPU: 6e-7 apart; with TF32 on: about 2e-5)")
    model = model.to(device)
    step_args = flat_args(base_cfg)
    step_args.log_every_steps = 0
    with contextlib.redirect_stdout(io.StringIO()):
        groups = get_param_groups(model, nowd_keys=train_ssl.NOWD_KEYS)
    opt = TheSameAsTimmLAMB(params=groups, lr=0.0, weight_decay=0.0, betas=(0.9, args.beta2), max_grad_norm=args.max_grad_norm)
    peak = args.base_lr * args.batch_size / 256
    names = {id(p): n for n, p in model.named_parameters()}
    nowd = {names[id(p)] for g in opt.param_groups if g["weight_decay_scale"] == 0 for p in g["params"]}
    check("no weight decay exactly on 1-D tensors, biases and mask tokens (the authors' grouping)",
          nowd == {n for n, p in model.named_parameters() if p.ndim == 1 or n.endswith(".bias") or "mask_token" in n},
          f"({len(nowd)} of {len(names)} tensors)")
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stats = train_ssl.train_one_epoch(model, loader, opt, 0, step_args, device,
                                      {"peak_lr": peak, "warmup_iters": 0, "total_iters": 2})
    after = model.state_dict()
    params = [k for k, _ in model.named_parameters()]
    changed = {k for k in params if not torch.equal(after[k], before[k])}
    check("loss and gradient norm finite", math.isfinite(stats["loss"]) and math.isfinite(stats["grad_norm"]),
          f"(loss {stats['loss']:.4f}, gradient norm {stats['grad_norm']:.3f}, feature cosine {stats['feature_cosine']:.3f})")
    check("all gradients finite", all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
    # LAMB moves a weight-decay tensor even when its gradient is 0 (weight decay alone), so "updated" below cannot
    # catch a weight that the loss never reaches; check those gradients directly.
    decay = {names[id(p)] for g in opt.param_groups if g["weight_decay_scale"] != 0 for p in g["params"]}
    no_grad = [n for n, p in model.named_parameters() if n in decay and (p.grad is None or not p.grad.abs().sum() > 0)]
    check("every weight-decay tensor (conv/linear weights) received a non-zero gradient", not no_grad,
          f"({len(decay)} tensors{'; zero: ' + ', '.join(no_grad[:5]) if no_grad else ''})")
    for part, prefix in [("encoder (including features.18)", "sparse_encoder."), ("densify layers", "densify_"),
                         ("mask token", "mask_tokens."), ("decoder", "dense_decoder.")]:
        ks = [k for k in params if k.startswith(prefix)]
        check(f"every {part} parameter updated", len(ks) > 0 and all(k in changed for k in ks), f"({len(ks)} tensors)")
    lr_now = {g["lr"] for g in opt.param_groups}
    wd_now = sorted(g["weight_decay"] for g in opt.param_groups)
    check("schedule set lr = peak and weight decay = start value (0 for the no-decay group)",
          lr_now == {peak} and wd_now[0] == 0.0 and math.isclose(wd_now[1], args.weight_decay, rel_tol=1e-9),
          f"(lr {lr_now}, wd {wd_now}; the authors' formula gives 0.2 + (0.04 - 0.2) x 1 in floating point)")
    nbt = ["sparse_encoder.sp_cnn.features.0.1.num_batches_tracked", "dense_decoder.dec.0.conv.1.num_batches_tracked"]
    check("BatchNorm statistics updated once (encoder and decoder)", all((after[k] - before[k]).item() == 1 for k in nbt)
          and not torch.equal(after["sparse_encoder.sp_cnn.features.0.1.running_mean"],
                              before["sparse_encoder.sp_cnn.features.0.1.running_mean"]))
    del model, opt

    print("7. Save and resume (smoke runs, 48 images, 2 epochs; on the CPU, where the check can be exact)")
    # On a GPU, run-to-run differences of a few float32 roundings (non-deterministic kernels) are amplified
    # over the steps, so an uninterrupted and a resumed GPU run cannot be compared exactly. On the CPU
    # every operation is deterministic: any difference would be a resume bug.
    cpu_cfg = json.loads(json.dumps(base_cfg))
    cpu_cfg["setup"]["device"] = "cpu"
    straight = train_ssl.run(cpu_cfg, smoke=True)
    stopped = train_ssl.run(cpu_cfg, smoke=True, stop_after_epoch=1)
    info = json.loads((stopped / "run_info.json").read_text(encoding="utf-8"))
    check("stopped run is marked resumable after epoch 1", info["status"].startswith("stopped") and info["cost"]["epochs_done"] == 1)
    bad = json.loads(json.dumps(cpu_cfg))
    bad["spark"]["mask_ratio"] = 0.5
    try:
        train_ssl.run(bad, resume_dir=stopped)
        rejected = False
    except ValueError as e:
        rejected = "mask_ratio" in str(e)
    check("resume with a changed config (mask_ratio) is rejected", rejected)
    # Simulate a stop right after the epoch-1 checkpoint was written: its history row, run_info cost and
    # export never reached the disk. Resume must rebuild them from the checkpoint before continuing.
    lines = (stopped / "history.csv").read_text().splitlines()
    (stopped / "history.csv").write_text("\n".join(lines[:-1]) + "\n")
    (stopped / "backbone_epoch001.pth").unlink()
    info["cost"] = {}
    save_json(info, stopped / "run_info.json")
    train_ssl.run(None, resume_dir=stopped)
    repaired = torch.load(stopped / "backbone_epoch001.pth", map_location="cpu", weights_only=True)["backbone_state_dict"]
    original = torch.load(straight / "backbone_epoch001.pth", map_location="cpu", weights_only=True)["backbone_state_dict"]
    info_s = json.loads((straight / "run_info.json").read_text(encoding="utf-8"))
    info_b = json.loads((stopped / "run_info.json").read_text(encoding="utf-8"))
    check("a stop right after a checkpoint is repaired on resume (history row, cost and epoch-1 export rebuilt)",
          all(torch.equal(repaired[k], original[k]) for k in original)
          and {k: v for k, v in info_b["cost"].items() if k != "train_minutes"}
          == {k: v for k, v in info_s["cost"].items() if k != "train_minutes"})
    a = torch.load(straight / "checkpoint.pth", map_location="cpu", weights_only=False)
    b = torch.load(stopped / "checkpoint.pth", map_location="cpu", weights_only=False)
    check("same epoch, global step and schedule after resume", a["epoch"] == b["epoch"] == 2
          and a["global_step"] == b["global_step"] and a["schedule"] == b["schedule"])
    d = max(max_diff(a["model"], b["model"]),
            max_diff({f"{i}{k}": v for i, s in a["optimizer"]["state"].items() for k, v in s.items()},
                     {f"{i}{k}": v for i, s in b["optimizer"]["state"].items() for k, v in s.items()}))
    check("resumed run equals the uninterrupted run exactly (model, BatchNorm statistics, optimizer state)", d == 0.0,
          f"(max difference {d:.1e})")
    h_a = (straight / "history.csv").read_text().splitlines()
    h_b = (stopped / "history.csv").read_text().splitlines()
    strip = lambda rows: [r.rsplit(",", 1)[0] for r in rows]  # noqa: E731  (last column = seconds)
    check("history identical after resume (loss, diagnostics, gradient norm, lr, weight decay)", strip(h_a) == strip(h_b))
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = True  # PyTorch's default; run() itself must switch it off
        gpu_run = train_ssl.run(base_cfg, smoke=True, stop_after_epoch=1)
        g_info = json.loads((gpu_run / "run_info.json").read_text(encoding="utf-8"))
        g_hist = (gpu_run / "history.csv").read_text().splitlines()
        check("the same smoke run on the GPU completes epoch 1 with a finite loss, TF32 off",
              g_info["status"].startswith("stopped") and math.isfinite(float(g_hist[1].split(",")[1]))
              and g_info["precision"]["cudnn_allow_tf32"] is False, f"({gpu_run.name})")

    print("8. Export and pictures")
    files = sorted(straight.glob("backbone_epoch*.pth"))
    check("a backbone file after every epoch in the smoke run (export_every = 1)",
          [x.name for x in files] == ["backbone_epoch001.pth", "backbone_epoch002.pth"])
    exported = files[-1]
    e = torch.load(exported, map_location="cpu", weights_only=True)
    sd = e["backbone_state_dict"]
    fresh = build_model({"type": "none"}, 40).features
    fresh.load_state_dict(sd, strict=True)
    check("exported backbone loads strictly into a fresh MobileNetV2", len(sd) == 312
          and e["num_parameter_tensors"] == 156 and e["num_buffer_tensors"] == 156, "(156 parameters + 156 buffers)")
    check("export contains nothing from the decoder side",
          e["excluded_from_export"] == ["dense_decoder", "densify_norms", "densify_projs", "imn_m", "imn_s", "mask_tokens", "norm_black"])
    check("export equals the encoder of the final checkpoint and is marked final",
          all(torch.equal(sd[k], a["model"]["sparse_encoder.sp_cnn.features." + k]) for k in sd) and e["is_final_epoch"])
    pics = ["view_preview.png", "charts/reconstruction_epoch001.png", "charts/reconstruction_epoch002.png", "param_groups.txt"]
    check("view and reconstruction previews and the parameter-group listing were written",
          all((straight / p).exists() for p in pics), f"({', '.join(pics)})")

    save_json({"ssl_run": str(straight), "exported": str(exported)}, LATEST)
    imports_check()
    finish("SSL", f"{straight.name}, {stopped.name}")


def part_downstream(cli):
    """Section 9, using the backbone exported by the SSL part (same checks as the A1/A2 smoke tests)."""
    device = resolve_device(cli.device)
    latest = json.loads(LATEST.read_text(encoding="utf-8"))
    straight, exported = Path(latest["ssl_run"]), Path(latest["exported"])
    sd = torch.load(exported, map_location="cpu", weights_only=True)["backbone_state_dict"]
    print("9. Downstream fine-tuning from the exported backbone")
    ft_cfg = json.loads(train_supervised.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    ft_cfg["train"]["device"] = cli.device
    ft_cfg["init"] = {"type": "backbone_checkpoint", "backbone_checkpoint": str(exported)}
    class_names = load_class_names()
    model = build_model(ft_cfg["init"], len(class_names))
    check("fine-tuning model starts from the exported backbone",
          all(torch.equal(model.features.state_dict()[k], sd[k]) for k in sd))
    check("new head: 40 outputs, dropout 0.2 kept", model.classifier[1].out_features == 40 and model.classifier[0].p == 0.2)
    records = train_supervised.first_per_class(read_manifest("labeled_set.csv", class_names), 1)[:16]
    x, y = next(iter(DataLoader(SARDataset(records, build_transform(ft_cfg["data"], train=True)), batch_size=16)))
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    model.train()
    opt = train_supervised.make_optimizer(model, ft_cfg["train"])
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    changed = {k for k, p in model.named_parameters() if not torch.equal(p, before[k])}
    check("one step updates backbone and new classifier", "features.0.0.weight" in changed and "classifier.1.weight" in changed
          and len(changed) == len(before), f"({len(changed)}/{len(before)} tensors)")
    del model, opt, x, y, loss
    ft_dir = train_supervised.run(ft_cfg, smoke=True, output_root=paths.SMOKE_OUTPUT_DIR)
    ft_info = json.loads((ft_dir / "run_info.json").read_text(encoding="utf-8"))
    check("fine-tuning run completed and records the SSL provenance", ft_info["status"] == "completed"
          and ft_info["initialization"]["ssl_record"]["ssl_run_dir"] == straight.name
          and ft_info["initialization"]["sha256"] == sha256_file(exported))
    reloaded, ckpt = load_finetuned(ft_dir / "best.pt", len(class_names))
    val = train_supervised.first_per_class(read_manifest("validation.csv", class_names), train_supervised.SMOKE_PER_CLASS)
    loader = DataLoader(SARDataset(val, build_transform(ft_cfg["data"], train=False)), batch_size=64)
    acc = predict(reloaded.to(device), loader, device)["acc"]
    check("reloaded best.pt reproduces the recorded validation accuracy", abs(acc - ckpt["val_acc"]) <= 1 / len(val),
          f"({acc:.4f} vs {ckpt['val_acc']:.4f} recorded; tolerance one image of {len(val)})")
    del reloaded
    m = evaluate(ft_dir, "validation", cli.device)
    check("evaluate.py --split validation uses the run's manifest copies and reproduces the record",
          abs(m["overall_accuracy"] - ckpt["val_acc"]) < 1e-6, f"({m['overall_accuracy']:.4f})")

    imports_check()
    finish("downstream", ft_dir.name)


def main():
    parser = argparse.ArgumentParser(description="A3 pipeline checks (not a result)")
    parser.add_argument("--device", default="cuda", help='"cuda", "cpu" or "auto"')
    parser.add_argument("--part", choices=["ssl", "downstream"], help="run one part only (default: both)")
    cli = parser.parse_args()
    if cli.part == "ssl":
        part_ssl(cli)
    elif cli.part == "downstream":
        part_downstream(cli)


if __name__ == "__main__":
    main()
