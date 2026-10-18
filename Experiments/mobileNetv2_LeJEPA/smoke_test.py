"""Checks of the whole A2 (MobileNetV2 + LeJEPA) pipeline before any long run. Not a result.

Run from the repository root:
    python Experiments/mobileNetv2_LeJEPA/smoke_test.py
    python Experiments/mobileNetv2_LeJEPA/smoke_test.py --device cpu

Everything it writes goes to outputs/smoke/ (folders ending in _SMOKE); delete them freely.
It never touches the official test set. Checks, in order:
  1. Data: split audit, counts and CSV hashes equal to those recorded by the A0 run; SSL items are views only.
  2. Sources: the vendored SIGReg files are byte-identical to the authors' originals.
  3. Initialization: network backbone equals the ImageNet IMAGENET1K_V2 file; no ImageNet classifier;
     one network only (no teacher, nothing frozen); projector 1280 -> 2048 -> 2048 -> proj_dim.
  4. Loss against the authors' code: agreement = MINIMAL.md's inv_loss; SIGReg (library) = MINIMAL.md's
     SIGReg formula with the same random directions; with local views, agreement = the paper's listing
     (centers from global views only); SIGReg is about 1 for Gaussian and large for collapsed inputs.
  5. Shapes: 1280 features per view; mixed-resolution views give the same features as one view at a time.
  6. One real training step on SSL images: finite loss, agreement and SIGReg; finite gradients; every
     backbone and projector parameter updated; BatchNorm statistics updated; SIGReg directions advance.
  7. Save/resume: 2 epochs straight vs 1 epoch + resume give the same state; a changed config is rejected.
  8. Export: backbone files after each epoch (smoke export_every = 1); 156 parameters + 156 buffers,
     no projector, loads strictly, equals the checkpoint's backbone.
  9. Downstream: a tiny fine-tuning run from that backbone; its best.pt reproduces the recorded
     validation accuracy, and evaluate.py --split validation reproduces it too.
 10. Imports: every module came from this folder, none from Baseline, the DINO experiment or lejepa-main.
"""
import subprocess
import sys

if __name__ == "__main__" and "--part" not in sys.argv:
    # Launcher: run the two parts as separate processes, one after the other (as in the A1 smoke test:
    # this process imports nothing heavy, so the SSL part's memory is returned before fine-tuning starts).
    for part in ["ssl", "downstream"]:
        code = subprocess.run([sys.executable, __file__, "--part", part, *sys.argv[1:]]).returncode
        if code:
            sys.exit(f"SMOKE TEST FAILED in the {part} part (exit code {code})")
    print("\nSMOKE TEST PASSED (both parts)")
    sys.exit(0)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import transforms as T  # noqa: E402
from torchvision.transforms import functional as TF  # noqa: E402

import paths  # noqa: E402
import train_ssl  # noqa: E402
import train_supervised  # noqa: E402
from data import SARDataset, build_transform, load_class_names, read_manifest  # noqa: E402
from engine import predict  # noqa: E402
from evaluate import evaluate  # noqa: E402
from lejepa_model import DataAugmentationLeJEPA, LeJEPANet, build_sigreg, lejepa_loss, to_network_input  # noqa: E402
from model import build_model, load_finetuned  # noqa: E402
from ssl_data import SSLDataset, audit_splits, flat_args, read_ssl_paths  # noqa: E402
from utils import resolve_device, save_json, sha256_file  # noqa: E402

REF = paths.EXPERIMENT_DIR / "reference" / "lejepa-main_original"
LATEST = paths.SMOKE_OUTPUT_DIR / "latest_smoke_ssl.json"  # hands the SSL part's export to the downstream part
# Recorded by the A0 run (Baseline_Output/20261013-150506_A0_imagenet_mobilenetv2_seed0/manifests.json).
A0_RECORD = {"labeled_set.csv": ("3eefc78f9eb254d3e86d355c72e22c60097869ad032734b1b38194cc51c8c767", 1961),
             "validation.csv": ("c752adadce615c11ffaefb0f187f144042d83cb070d9c9c34775ab8b989a9980", 800),
             "class_mapping.csv": ("5768e0346b908a22e11e25f34fbdfbc167783c9ccb33f434353a10c268a4427d", 40)}
VENDORED = {"lejepa_lib/univariate/base.py": "lejepa/univariate/base.py",
            "lejepa_lib/univariate/epps_pulley.py": "lejepa/univariate/epps_pulley.py",
            "lejepa_lib/multivariate/slicing.py": "lejepa/multivariate/slicing.py"}
FAILED = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILED.append(name)


def minimal_sigreg_class():
    """The SIGReg class exactly as printed in the authors' MINIMAL.md, with ONE test change: the random
    directions come from self.A_fixed instead of a fresh torch.randn(..., 256, device="cuda"), so the
    library and this code can be compared on the same directions."""
    text = (REF / "MINIMAL.md").read_text(encoding="utf-8")
    src = re.search(r"(class SIGReg\(torch\.nn\.Module\):.*?return statistic\.mean\(\)\n)", text, re.S).group(1)
    original = 'A = torch.randn(proj.size(-1), 256, device="cuda")'
    assert original in src, "MINIMAL.md SIGReg changed"
    src = src.replace(original, "A = self.A_fixed.clone()")
    ns = {"torch": torch}
    exec(compile(src, "MINIMAL.md (SIGReg class)", "exec"), ns)
    return ns["SIGReg"]


def max_diff(a: dict, b: dict) -> float:
    return max(((a[k].float() - b[k].float()).abs().max().item() if a[k].numel() else 0.0) for k in a)


def imports_check():
    print("Imports")
    here = paths.EXPERIMENT_DIR.resolve()
    local = [p.stem for p in here.glob("*.py")]
    loaded = {n: Path(sys.modules[n].__file__).resolve().parent for n in local if n in sys.modules}
    check("every project module in use was loaded from this folder", all(d == here for d in loaded.values()),
          f"({', '.join(sorted(loaded))})")
    lib = Path(sys.modules["lejepa_lib"].__file__).resolve().parent if "lejepa_lib" in sys.modules else None
    check("SIGReg code loaded from this folder's lejepa_lib/", lib is None or lib == here / "lejepa_lib")
    bad = ("Baseline", "mobileNetv2_DINO", "lejepa-main")
    check("no module from Baseline, the DINO experiment or lejepa-main",
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
    print(f"Smoke test on {device}. Outputs go to {paths.SMOKE_OUTPUT_DIR}\n")

    print("1. Data")
    audit = audit_splits()
    check("split audit (leakage and consistency checks)", True, f"({len(audit['checks'])} checks, counts {audit['counts']})")
    count_key = {"labeled_set.csv": "labeled", "validation.csv": "validation", "class_mapping.csv": "classes"}
    for f, (sha, n) in A0_RECORD.items():
        same = audit["csv_sha256"][f] == sha
        count = audit["counts"][count_key[f]]
        check(f"{f} identical to the A0 run's file", same and count == n, f"({count} rows/classes, sha256 {'matches' if same else 'DIFFERS'})")
    ssl_paths = read_ssl_paths(paths.SPLITS_DIR / args.ssl_csv)
    views = SSLDataset(ssl_paths[:2], DataAugmentationLeJEPA(args))[0]
    check("SSL item is a list of views only (no label)", isinstance(views, list) and len(views) == args.n_global_views)
    check("views leave the data loader as uint8 gray images", {(tuple(v.shape), v.dtype) for v in views}
          == {((1, args.global_crop_size, args.global_crop_size), torch.uint8)}, f"({len(views)} x {tuple(views[0].shape)})")
    batch = torch.stack(views[:4])  # (4, 1, H, W), as the DataLoader delivers one view of 4 images
    net_in = to_network_input([batch], args.normalize_mean, args.normalize_std, device)[0].cpu()
    a0_way = T.Compose([T.Grayscale(num_output_channels=3), T.ToTensor(), T.Normalize(args.normalize_mean, args.normalize_std)])
    expected = torch.stack([a0_way(TF.to_pil_image(v)) for v in views[:4]])
    err = (net_in - expected).abs().max().item()
    check("GPU-side input equals the A0/A1 pipeline (Grayscale(3) -> ToTensor -> Normalize) on the same views",
          tuple(net_in.shape) == tuple(expected.shape) and err < 1e-5, f"(max difference {err:.1e})")
    check("3 channels are copies of one SAR channel after de-normalizing",
          bool(torch.allclose(net_in[:, 0] * args.normalize_std[0] + args.normalize_mean[0],
                              net_in[:, 1] * args.normalize_std[1] + args.normalize_mean[1], atol=1e-5)))

    print("2. Sources")
    for ours, orig in VENDORED.items():
        a, b = sha256_file(paths.EXPERIMENT_DIR / ours), sha256_file(REF / orig)
        check(f"{ours} byte-identical to the authors' {orig}", a == b, f"(sha256 {a[:12]}...)")

    print("3. Initialization")
    imagenet = torch.load(paths.MOBILENETV2_IMAGENET_WEIGHTS, map_location="cpu", weights_only=True)
    print(f"  ImageNet weights sha256 {sha256_file(paths.MOBILENETV2_IMAGENET_WEIGHTS)[:16]}...")
    torch.manual_seed(0)
    net = LeJEPANet(imagenet_init=True, proj_hidden_dim=args.proj_hidden_dim, proj_dim=args.proj_dim)
    state = net.state_dict()
    check("backbone equals the ImageNet IMAGENET1K_V2 weights (parameters and BatchNorm buffers)",
          all(torch.equal(state["backbone." + k], v) for k, v in imagenet.items() if k.startswith("features.")))
    check("no ImageNet classifier in the SSL network", not any("classifier" in k for k in state))
    check("one network: only backbone and projector, every parameter trainable",
          {k.split(".")[0] for k in state} == {"backbone", "proj"} and all(p.requires_grad for p in net.parameters()))
    linears = [m for m in net.proj.modules() if isinstance(m, torch.nn.Linear)]
    bns = [m for m in net.proj.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    check("projector 1280 -> 2048 -> 2048 -> proj_dim with BatchNorm after the hidden layers",
          [(l.in_features, l.out_features) for l in linears] == [(1280, args.proj_hidden_dim),
                                                                (args.proj_hidden_dim, args.proj_hidden_dim),
                                                                (args.proj_hidden_dim, args.proj_dim)] and len(bns) == 2,
          f"({' -> '.join(str(l.in_features) for l in linears)} -> {linears[-1].out_features})")

    print("4. Loss against the authors' code")
    torch.manual_seed(1)
    V, N, K = 8, 32, args.proj_dim
    proj = torch.randn(V, N, K) * 0.7 + 0.2
    sigreg = build_sigreg(args)
    g = torch.Generator().manual_seed(0)  # the library seeds its directions with its global_step (0 here)
    A = torch.randn((K, args.num_slices), generator=g)
    A /= A.norm(p=2, dim=0)
    total, agreement, sig = lejepa_loss(proj, V, sigreg, args.lamb)
    check("agreement equals MINIMAL.md inv_loss = (proj.mean(0) - proj).square().mean()",
          torch.allclose(agreement, (proj.mean(0) - proj).square().mean(), rtol=1e-6, atol=0))
    ref = minimal_sigreg_class()(knots=args.n_points)
    ref.A_fixed = A
    sig_ref = ref(proj)
    check("SIGReg (authors' library) equals MINIMAL.md's SIGReg formula on the same directions",
          torch.allclose(sig, sig_ref, rtol=1e-5), f"({sig.item():.6f} vs {sig_ref.item():.6f})")
    check("total = (1 - lamb) * agreement + lamb * SIGReg",
          torch.allclose(total, (1 - args.lamb) * agreement + args.lamb * sig, rtol=1e-6))
    check("SIGReg advanced its seed (new random directions next step)", sigreg.global_step.item() == 1)
    n_g = 2
    _, agr_local, _ = lejepa_loss(proj, n_g, build_sigreg(args), args.lamb)
    paper = (proj[:n_g].mean(0) - proj).square().mean()  # paper Section 5.1: centers from the global views only
    check("with local views: agreement equals the paper's listing (centers = mean of global views)",
          torch.allclose(agr_local, paper, rtol=1e-6))
    s_gauss = build_sigreg(args)(torch.randn(V, N, K)).item()
    s_coll = build_sigreg(args)(0.01 * torch.randn(V, N, K)).item()
    check("SIGReg is about 1 for Gaussian projections and large for collapsed ones", s_gauss < 2 and s_coll > 5 * s_gauss,
          f"(Gaussian {s_gauss:.2f}, collapsed {s_coll:.2f})")

    print("5. Shapes and mixed resolutions")
    net.eval()  # eval: BatchNorm uses stored statistics, so per-view and grouped outputs must agree
    xs = [torch.randn(3, 3, 224, 224), torch.randn(3, 3, 224, 224), torch.randn(3, 3, 96, 96), torch.randn(3, 3, 96, 96)]
    with torch.no_grad():
        emb, pj = net(xs)
        one_by_one = torch.stack([net.backbone(x) for x in xs])
    check("features (V, N, 1280) and projections (V, N, proj_dim)", tuple(emb.shape) == (4, 3, 1280)
          and tuple(pj.shape) == (4, 3, args.proj_dim))
    check("mixed 224/96 views give the same features as one view at a time (view-major order kept)",
          torch.allclose(emb, one_by_one, atol=1e-4))
    net.train()

    print("6. One real training step (8 SSL images, the configured views)")
    net = net.to(device)
    sigreg = build_sigreg(args).to(device)
    step_args = flat_args(base_cfg)
    step_args.log_every_steps = 0
    loader = DataLoader(SSLDataset(ssl_paths[:8], DataAugmentationLeJEPA(args)), batch_size=8)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    before = {k: v.detach().clone() for k, v in net.state_dict().items()}
    autocast = torch.bfloat16 if (args.precision == "bf16" and device.type == "cuda") else None
    stats = train_ssl.train_one_epoch(net, sigreg, loader, opt, [args.lr], 0, step_args, device, autocast)
    after = net.state_dict()
    params = [k for k, _ in net.named_parameters()]
    changed = {k for k in params if not torch.equal(after[k], before[k])}
    check("loss, agreement and SIGReg finite", all(math.isfinite(stats[k]) for k in ["loss", "agreement", "sigreg"]),
          f"(loss {stats['loss']:.4f}, agreement {stats['agreement']:.4f}, SIGReg {stats['sigreg']:.3f})")
    check("all gradients finite", all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
    check("every backbone parameter updated", all(k in changed for k in params if k.startswith("backbone.")),
          f"({sum(k.startswith('backbone.') for k in changed)} tensors)")
    check("every projector parameter updated", all(k in changed for k in params if k.startswith("proj.")))
    nbt = "backbone.features.0.1.num_batches_tracked"
    check("BatchNorm statistics updated once (one resolution group)", (after[nbt] - before[nbt]).item() == 1
          and not torch.equal(after["backbone.features.0.1.running_mean"], before["backbone.features.0.1.running_mean"]))
    check("SIGReg random directions advanced once", sigreg.global_step.item() == 1)
    del net, opt

    print("7. Save and resume (smoke runs, 48 images, 2 epochs)")
    straight = train_ssl.run(base_cfg, smoke=True)
    stopped = train_ssl.run(base_cfg, smoke=True, stop_after_epoch=1)
    info = json.loads((stopped / "run_info.json").read_text(encoding="utf-8"))
    check("stopped run is marked resumable after epoch 1", info["status"].startswith("stopped") and info["cost"]["epochs_done"] == 1)
    bad = json.loads(json.dumps(base_cfg))
    bad["lejepa"]["lamb"] = 0.02
    try:
        train_ssl.run(bad, resume_dir=stopped)
        rejected = False
    except ValueError as e:
        rejected = "lamb" in str(e)
    check("resume with a changed config (lamb) is rejected", rejected)
    train_ssl.run(None, resume_dir=stopped)
    a = torch.load(straight / "checkpoint.pth", map_location="cpu", weights_only=False)
    b = torch.load(stopped / "checkpoint.pth", map_location="cpu", weights_only=False)
    check("same epoch, global step, schedule and SIGReg step after resume", a["epoch"] == b["epoch"] == 2
          and a["global_step"] == b["global_step"] and a["schedule_info"] == b["schedule_info"]
          and a["sigreg"]["global_step"].item() == b["sigreg"]["global_step"].item() == a["global_step"])
    d = max_diff(a["model"], b["model"])
    tol = 0.0 if device.type == "cpu" else 2e-2
    check("resumed run matches the uninterrupted run", d <= tol,
          f"(max difference {d:.1e}; tolerance {tol} on {device.type}: cuDNN picks non-deterministic kernels on "
          f"the GPU, and bf16 rounding makes those differences larger than in fp32)")
    h_a = (straight / "history.csv").read_text().splitlines()
    h_b = (stopped / "history.csv").read_text().splitlines()
    check("learning rate per epoch identical after resume", [r.split(",")[8] for r in h_a] == [r.split(",")[8] for r in h_b])

    print("8. Export")
    files = sorted(straight.glob("backbone_epoch*.pth"))
    check("a backbone file after every epoch in the smoke run (export_every = 1)",
          [f.name for f in files] == ["backbone_epoch001.pth", "backbone_epoch002.pth"])
    exported = files[-1]
    e = torch.load(exported, map_location="cpu", weights_only=True)
    sd = e["backbone_state_dict"]
    fresh = build_model({"type": "none"}, 40).features
    fresh.load_state_dict(sd, strict=True)
    check("exported backbone loads strictly into a fresh MobileNetV2", len(sd) == 312
          and e["num_parameter_tensors"] == 156 and e["num_buffer_tensors"] == 156, "(156 parameters + 156 buffers)")
    check("export contains no projector or classifier", e["excluded_from_export"] == ["proj"]
          and not any(k.startswith(("proj", "classifier")) for k in sd))
    check("export equals the backbone of the final checkpoint and is marked final",
          all(torch.equal(sd[k], a["model"]["backbone.features." + k]) for k in sd) and e["is_final_epoch"])

    save_json({"ssl_run": str(straight), "exported": str(exported)}, LATEST)
    imports_check()
    finish("SSL", f"{straight.name}, {stopped.name}")


def part_downstream(cli):
    """Section 9, using the backbone exported by the SSL part (same checks as the A1 smoke test)."""
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
    parser = argparse.ArgumentParser(description="A2 pipeline checks (not a result)")
    parser.add_argument("--device", default="cuda", help='"cuda", "cpu" or "auto"')
    parser.add_argument("--part", choices=["ssl", "downstream"], help="run one part only (default: both)")
    cli = parser.parse_args()
    if cli.part == "ssl":
        part_ssl(cli)
    elif cli.part == "downstream":
        part_downstream(cli)


if __name__ == "__main__":
    main()
