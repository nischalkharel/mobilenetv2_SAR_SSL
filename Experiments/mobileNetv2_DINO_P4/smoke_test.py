"""Checks of the whole P4 pipeline before any long run. Not a result.

P4 COPY of Experiments/mobileNetv2_DINO/smoke_test.py (A1): the A1 checks are kept and the per-condition-cell
parts are added (1b cells, 4b the per-cell loss, the cell updates in the real step, the cell centers across
save/resume, the P4 provenance in the export and the fine-tuning record, no import from the A1 folder).

Run from the repository root:
    python Experiments/mobileNetv2_DINO_P4/smoke_test.py
    python Experiments/mobileNetv2_DINO_P4/smoke_test.py --device cpu

Everything it writes goes to outputs/smoke/ (folders ending in _SMOKE); delete them freely.
It never touches the official test set. Checks, in order:
  1. Data: split audit, counts and CSV hashes equal to those recorded by the latest A0 run.
  1b. Cells (P4): deterministic assignment from the metadata columns, names equal the fields parsed from the file
      names, every image in one cell, cells do not identify the class (label-leak check: classes per cell, largest
      class share, cell-only majority vote), the scrambled control keeps the histogram and is random, dataset and
      collation. --config <file> tests another SSL config (the scrambled or the 3-field variant).
  2. Initialization: ImageNet file hash; student backbone equals it; teacher equals student
     (every parameter and buffer); no ImageNet classifier anywhere in the SSL networks.
  3. Shapes: 1280 features at 224 and 96 px; multi-crop wrapper keeps view-major, sample order.
  4. Against the original code (loaded from the untouched copy in reference/): DINOHead
     initialization and output, DINO loss and center update on fixed synthetic inputs, and the
     explicit cross-view pairing (same view never compared with itself).
  4b. Per-cell loss (P4): equals DINOLoss exactly with one cell; after a warm-up step (non-zero, different centers)
      present cells move by the EMA rule and the absent cell does not, each teacher row uses its own cell's center
      with view-major ids (sample-major would give a different loss); wrong ids rejected; buffers in the state_dict.
  5. One real training step: finite loss, student backbone and head updated, last layer frozen
     in epoch 0 and updated in epoch 1, teacher has no gradients and follows the EMA formula,
     BatchNorm buffers behave as documented, global center and the per-cell centers updated.
  6. Save/resume: 2 epochs straight vs 1 epoch + resume give the same state (cell centers included); a changed
     config or a changed cell definition is rejected; cells.csv / cell_center_stats.csv / cells_seen written.
  7. Export: teacher backbone has all 156 parameters + 156 buffers, no head, loads strictly; P4 recorded.
  8. Downstream: a tiny fine-tuning run from that backbone updates backbone and new head; its
     best.pt reproduces the validation accuracy recorded during training; the P4 provenance is in its record.
  9. Imports: every module came from this folder, none from Baseline, dino-main or the A1 folder.
"""
import subprocess
import sys

if __name__ == "__main__" and "--part" not in sys.argv:
    # Launcher: run the two parts as separate processes, one after the other. This process imports
    # nothing heavy (no torch), so all memory used by the SSL part is returned before fine-tuning
    # starts. On this PC (no page file, so GPU allocations and other programs exhaust Windows commit memory),
    # one process doing both stages crashed with out-of-memory errors.
    for part in ["ssl", "downstream"]:
        code = subprocess.run([sys.executable, __file__, "--part", part, *sys.argv[1:]]).returncode
        if code:
            sys.exit(f"SMOKE TEST FAILED in the {part} part (exit code {code})")
    print("\nSMOKE TEST PASSED (both parts)")
    sys.exit(0)

import argparse  # noqa: E402
import ast  # noqa: E402
import json  # noqa: E402
import math
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import paths
import train_ssl
import train_supervised
from cond_center import CellSSLDataset, CondCenterDINOLoss, assign_cells, cell_key
from data import SARDataset, build_transform, load_class_names, read_manifest
from dino import DataAugmentationDINO, DINOHead, DINOLoss, MultiCropWrapper
from engine import predict
from evaluate import evaluate
from model import build_model, load_finetuned
from ssl_data import SSLDataset, audit_splits, flat_args, make_run_dir, read_rows, read_ssl_paths
from utils import resolve_device, save_json, sha256_file

REF = paths.EXPERIMENT_DIR / "reference" / "dino-main_original"
LATEST = paths.SMOKE_OUTPUT_DIR / "latest_smoke_ssl.json"  # hands the SSL part's export to the downstream part
# Recorded by the latest A0 run (Baseline_Output/20261013-150506_A0_imagenet_mobilenetv2_seed0/manifests.json).
# Copied here as constants so this check does not need the Baseline folder.
A0_RECORD = {"labeled_set.csv": ("3eefc78f9eb254d3e86d355c72e22c60097869ad032734b1b38194cc51c8c767", 1961),
             "validation.csv": ("c752adadce615c11ffaefb0f187f144042d83cb070d9c9c34775ab8b989a9980", 800),
             "class_mapping.csv": ("5768e0346b908a22e11e25f34fbdfbc167783c9ccb33f434353a10c268a4427d", 40)}
FAILED = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILED.append(name)


def load_original(file, names, namespace):
    """Compile only the named classes/functions from the untouched original file (no import of dino-main)."""
    tree = ast.parse((REF / file).read_text(encoding="utf-8"))
    body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(REF / file), "exec"), namespace)
    return namespace


def max_diff(a: dict, b: dict) -> float:
    return max(((a[k].float() - b[k].float()).abs().max().item() if a[k].numel() else 0.0) for k in a)


def imports_check():
    print("Imports")
    here = paths.EXPERIMENT_DIR.resolve()
    local = [p.stem for p in here.glob("*.py")]
    loaded = {n: Path(sys.modules[n].__file__).resolve().parent for n in local if n in sys.modules}
    check("every project module in use was loaded from this folder", all(d == here for d in loaded.values()),
          f"({', '.join(sorted(loaded))})")
    check("no module from Baseline or dino-main",
          not any(("Baseline" in str(getattr(m, "__file__", "") or "")) or ("dino-main" in str(getattr(m, "__file__", "") or ""))
                  for m in sys.modules.values()))
    check("no module from the A1 folder (Experiments/mobileNetv2_DINO)",
          not any(Path(str(getattr(m, "__file__", "") or "x")).resolve().parent.name == "mobileNetv2_DINO"
                  for m in list(sys.modules.values())))


def finish(part, folders):
    print(f"\n{'PASSED' if not FAILED else 'FAILED: ' + '; '.join(FAILED)} ({part} part)\n"
          f"Smoke folders (not results): {folders} in {paths.SMOKE_OUTPUT_DIR}", flush=True)
    if FAILED:
        sys.exit(1)


def part_ssl(cli):
    """Sections 1 to 7."""
    device = resolve_device(cli.device)
    config_path = cli.config or train_ssl.DEFAULT_CONFIG
    base_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    base_cfg["setup"]["device"] = cli.device
    args = flat_args(base_cfg)
    print(f"Smoke test on {device} with {config_path.name}. Outputs go to {paths.SMOKE_OUTPUT_DIR}\n")

    print("1. Data")
    audit = audit_splits()
    check("split audit (leakage and consistency checks)", True, f"({len(audit['checks'])} checks, counts {audit['counts']})")
    count_key = {"labeled_set.csv": "labeled", "validation.csv": "validation", "class_mapping.csv": "classes"}
    for f, (sha, n) in A0_RECORD.items():
        if paths.SPLIT != 1 and f != "class_mapping.csv":  # the hashes are split 1's files
            continue
        same = audit["csv_sha256"][f] == sha
        count = audit["counts"][count_key[f]]
        check(f"{f} identical to the A0 run's file", same and count == n, f"({count} rows/classes, sha256 {'matches' if same else 'DIFFERS'})")
    ssl_paths = read_ssl_paths(paths.SPLITS_DIR / args.ssl_csv)
    views = SSLDataset(ssl_paths[:2], DataAugmentationDINO(args))[0]
    check("SSL item is a list of views only (no label)", isinstance(views, list) and len(views) == 2 + args.local_crops_number)
    shapes = [tuple(v.shape) for v in views]
    check("view shapes", shapes[:2] == [(3, 224, 224)] * 2 and set(shapes[2:]) == {(3, 96, 96)},
          f"(global {shapes[0]}, local {shapes[2]})")
    raw = views[0] * torch.tensor(args.normalize_std).view(3, 1, 1) + torch.tensor(args.normalize_mean).view(3, 1, 1)
    check("3 channels are copies of one SAR channel", bool(torch.allclose(raw[0], raw[1], atol=1e-5) and torch.allclose(raw[0], raw[2], atol=1e-5)))

    print("1b. Condition cells (P4)")
    rows = read_rows(paths.SPLITS_DIR / args.ssl_csv)
    cc = args.cond_center
    check("config enables per-cell centering with at least one field", cc["enabled"] and len(cc["cell"]) > 0, f"({cc['cell']})")
    ids, names = assign_cells(rows, cc["cell"], cc["permute_cells"], cc["permute_seed"])  # what this config trains with
    ids2, names2 = assign_cells(rows, cc["cell"], cc["permute_cells"], cc["permute_seed"])
    phys, _ = assign_cells(rows, cc["cell"], False)  # the physical cells (= ids unless the config scrambles them)
    check("cell assignment is deterministic", np.array_equal(ids, ids2) and names == names2,
          f"({len(names)} cells from {' | '.join(cc['cell'])}, {len(rows)} images"
          f"{'; this config trains with the SCRAMBLED assignment' if cc['permute_cells'] else ''})")
    check("every image has exactly one cell id in range", len(ids) == len(rows) and ids.min() == 0 and ids.max() == len(names) - 1)
    def expected_name(r):  # independent of cond_center: band, polarization, depression, azimuth parsed from the FILE NAME
        band, pol, dep, az = Path(r["relative_path"]).name.split("_")[:4]  # <band>_<pol>_<dep>_<az>_<id>.tif
        src = {"scene": r["scene"], "band": band, "polarization": pol, "depression": dep, "azimuth": az,
               "az_sector30": str(int(az) // 30), "az_sector60": str(int(az) // 60)}
        return "|".join(src[f] for f in cc["cell"])
    check("physical cell names equal the fields parsed from each image's own file name (scene from the CSV)",
          all(expected_name(r) == names[i] for r, i in zip(rows[::97], phys[::97])))
    # label-leak check on the PHYSICAL cells: the class column is read HERE ONLY (never by training). A cell must not
    # identify the vehicle.
    from collections import Counter
    classes_per_cell = {}
    for r, i in zip(rows, phys):
        classes_per_cell.setdefault(int(i), Counter())[r["class_name"]] += 1
    n_cls = np.array([len(c) for c in classes_per_cell.values()])
    purity = np.array([max(c.values()) / sum(c.values()) for c in classes_per_cell.values()])
    majority = sum(max(c.values()) for c in classes_per_cell.values()) / len(rows)
    check("cells do not identify the class (median classes per cell >= 20, every cell >= 2, largest class in any cell "
          "<= 50 %, cell-only majority vote <= 10 %)",
          np.median(n_cls) >= 20 and n_cls.min() >= 2 and purity.max() <= 0.5 and majority <= 0.10,
          f"(classes per cell min/median/max {n_cls.min()}/{np.median(n_cls):.0f}/{n_cls.max()} of {audit['counts']['classes']}; "
          f"largest-class share max {100 * purity.max():.1f} %; majority vote from the cell alone {100 * majority:.1f} %, "
          f"chance {100 / audit['counts']['classes']:.1f} %)")
    pids, pnames = assign_cells(rows, cc["cell"], True, cc["permute_seed"])
    hist = np.bincount(phys, minlength=len(names))
    kept_true_cell = np.mean([cell_key(r, cc["cell"]) == names[p] for r, p in zip(rows, pids)])
    chance = float(((hist / len(rows)) ** 2).sum())  # a random bijection keeps an image in its own cell with this rate
    check("scrambled control keeps the cell histogram and maps images to cells at random",
          pnames == names and np.array_equal(np.bincount(pids, minlength=len(names)), hist)
          and (pids != phys).mean() > 0.5 and kept_true_cell < 0.5,
          f"(images landing in their own physical cell {100 * kept_true_cell:.2f} %, chance {100 * chance:.2f} %)")
    if cc["permute_cells"]:
        check("this config trains with exactly that scrambled assignment", np.array_equal(ids, pids))
    cell_ds = CellSSLDataset(ssl_paths[:3], DataAugmentationDINO(args), None, ids[:3])
    item = cell_ds[1]
    check("CellSSLDataset returns (views, cell id) with the image's own cell",
          isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], list) and len(item[0]) == 2 + args.local_crops_number
          and int(item[1]) == int(ids[1]))
    batch = next(iter(DataLoader(cell_ds, batch_size=3)))
    check("collated batch = list of view batches + (batch,) long tensor of cell ids",
          isinstance(batch[0], list) and tuple(batch[0][0].shape) == (3, 3, 224, 224) and batch[1].dtype == torch.long
          and batch[1].tolist() == ids[:3].tolist())

    print("2. Initialization")
    imagenet = torch.load(paths.MOBILENETV2_IMAGENET_WEIGHTS, map_location="cpu", weights_only=True)
    cached = Path(torch.hub.get_dir()) / "checkpoints" / "mobilenet_v2-7ebf99e0.pth"
    if cached.exists():  # torchvision's own IMAGENET1K_V2 file, compared only if already downloaded (no network)
        tv = torch.load(cached, map_location="cpu", weights_only=True)
        check("weights file equals torchvision IMAGENET1K_V2 (local cache)", all(torch.equal(imagenet[k], tv[k]) for k in imagenet))
    else:
        print("  [SKIP] torchvision's cached IMAGENET1K_V2 file not found; not downloading it")
    print(f"  ImageNet weights sha256 {sha256_file(paths.MOBILENETV2_IMAGENET_WEIGHTS)[:16]}...")
    torch.manual_seed(0)
    student, teacher, embed_dim, dims = train_ssl.build_networks(args, device)
    s_state, t_state = student.state_dict(), teacher.state_dict()
    check("student backbone equals ImageNet weights",
          all(torch.equal(s_state["backbone." + k].cpu(), v) for k, v in imagenet.items() if k.startswith("features.")))
    check("teacher equals student (all parameters and buffers)", s_state.keys() == t_state.keys()
          and all(torch.equal(s_state[k], t_state[k]) for k in s_state), f"({len(s_state)} entries)")
    check("no ImageNet classifier in the SSL networks", not any("classifier" in k for k in s_state))
    check("teacher parameters do not require gradients", not any(p.requires_grad for p in teacher.parameters()))

    print("3. Shapes and multi-crop order")
    check("feature dimension 1280 at both crop sizes", embed_dim == 1280 and all(d == (2, 1280) for d in dims.values()), str(dims))
    check("DINO head output = out_dim (not 40)", student.head.last_layer.out_features == args.out_dim, f"({args.out_dim})")

    class TagBackbone(nn.Module):  # returns each image's constant pixel value, and counts calls
        calls = 0

        def forward(self, x):
            TagBackbone.calls += 1
            return x.mean((1, 2, 3)).unsqueeze(1)
    B = 3
    crops = [torch.full((B, 3, 8 if v < 2 else 4, 8 if v < 2 else 4), 0.0) + 100 * v + torch.arange(B).view(B, 1, 1, 1)
             for v in range(5)]
    out = MultiCropWrapper(TagBackbone(), nn.Identity())(crops).squeeze(1)
    expected = torch.tensor([100.0 * v + b for v in range(5) for b in range(B)])
    check("multi-crop output is view-major then sample order", torch.equal(out, expected) and TagBackbone.calls == 2,
          "(one backbone call per resolution)")

    print("4. Against the original DINO code (reference/dino-main_original, untouched)")
    stub_dist = SimpleNamespace(all_reduce=lambda t: None, get_world_size=lambda: 1)  # one process
    orig = load_original("main_dino.py", {"DINOLoss"}, {"torch": torch, "nn": nn, "F": F, "np": np, "dist": stub_dist})
    tn = load_original("utils.py", {"_no_grad_trunc_normal_", "trunc_normal_"},
                       {"torch": torch, "math": math, "warnings": warnings})
    vt = load_original("vision_transformer.py", {"DINOHead"}, {"nn": nn, "trunc_normal_": tn["trunc_normal_"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # original uses the deprecated weight_norm
        torch.manual_seed(1)
        h_orig = vt["DINOHead"](1280, 512)
    torch.manual_seed(1)
    h_new = DINOHead(1280, 512)
    so, sn = h_orig.state_dict(), h_new.state_dict()
    rename = {"last_layer.weight_g": "last_layer.parametrizations.weight.original0",
              "last_layer.weight_v": "last_layer.parametrizations.weight.original1"}
    check("DINOHead initialization identical to the original (same seed)",
          all(torch.equal(so[k], sn[rename.get(k, k)]) for k in so) and len(so) == len(sn))
    x = torch.randn(16, 1280)
    check("DINOHead output identical to the original", torch.allclose(h_orig(x), h_new(x), atol=1e-6))
    check("weight-norm scale frozen and last layer named for freezing, as in the original",
          not h_new.last_layer.parametrizations.weight.original0.requires_grad
          and any("last_layer" in n for n, _ in h_new.named_parameters()))

    torch.manual_seed(2)
    nc, D = 6, 64
    l_orig = orig["DINOLoss"](D, nc, 0.04, 0.04, 0, 3)
    l_new = DINOLoss(D, nc, 0.04, 0.04, 0, 3)
    worst = 0.0
    for epoch in range(3):
        s_out, t_out = torch.randn(nc * 4, D), torch.randn(2 * 4, D)
        a, b = l_orig(s_out, t_out, epoch), l_new(s_out, t_out, epoch)
        worst = max(worst, abs(a.item() - b.item()), (l_orig.center - l_new.center).abs().max().item())
    check("DINO loss and center match the original on synthetic inputs (3 steps)", worst < 1e-6,
          f"(max difference {worst:.1e}; original run with world size 1)")
    s_out, t_out = torch.randn(nc * 4, D), torch.randn(2 * 4, D)
    center = l_new.center.clone()
    loss = l_new(s_out, t_out, 0)
    q = F.softmax((t_out - center) / 0.04, -1).chunk(2)
    s = (s_out / 0.1).chunk(nc)
    pairs = [(iq, v) for iq in range(2) for v in range(nc) if v != iq]
    manual = sum((-q[iq] * F.log_softmax(s[v], -1)).sum(-1).mean() for iq, v in pairs) / len(pairs)
    check("cross-view pairing: teacher views 0,1 x all other student views", torch.allclose(loss, manual, atol=1e-6)
          and len(pairs) == 2 * nc - 2, f"({len(pairs)} pairs, same-view pairs skipped)")

    print("4b. Per-cell centering loss (cond_center.py)")
    torch.manual_seed(3)
    l_ref = DINOLoss(D, nc, 0.04, 0.04, 0, 3)
    l_one = CondCenterDINOLoss(D, nc, 0.04, 0.04, 0, 3, n_cells=1)
    worst = 0.0
    for epoch in range(3):
        s_out, t_out = torch.randn(nc * 4, D), torch.randn(2 * 4, D)
        a, b = l_ref(s_out, t_out, epoch), l_one(s_out, t_out, epoch, torch.zeros(4, dtype=torch.long))
        worst = max(worst, abs(a.item() - b.item()), (l_ref.center - l_one.cell_center).abs().max().item(),
                    (l_ref.center - l_one.center).abs().max().item())
    check("with one cell the per-cell loss equals DINOLoss (loss, cell center and global center, 3 steps)", worst < 1e-6,
          f"(max difference {worst:.1e})")
    l3 = CondCenterDINOLoss(D, nc, 0.04, 0.04, 0, 3, n_cells=3)
    l3(torch.randn(nc * 4, D), torch.randn(2 * 4, D), 0, torch.tensor([0, 1, 2, 1]))  # warm-up: every center non-zero and different
    check("warm-up left three different non-zero cell centers", (l3.cell_center.norm(dim=1) > 0).all().item()
          and not torch.allclose(l3.cell_center[0], l3.cell_center[1]) and not torch.allclose(l3.cell_center[1], l3.cell_center[2]))
    cells3 = torch.tensor([0, 2, 2, 0])
    s_out, t_out = torch.randn(nc * 4, D), torch.randn(2 * 4, D)
    before3 = l3.cell_center.clone()
    loss3 = l3(s_out, t_out, 0, cells3)
    ids_rows = torch.cat([cells3, cells3])
    expected = before3.clone()
    for c in (0, 2):
        expected[c] = 0.9 * before3[c] + 0.1 * t_out[ids_rows == c].mean(0)
    check("present cells move to 0.9 * old + 0.1 * mean of their own teacher rows; the absent (non-zero) cell is untouched",
          torch.allclose(l3.cell_center, expected, atol=1e-6) and torch.equal(l3.cell_center[1], before3[1])
          and l3.cell_count.tolist() == [6, 4, 6])
    q3 = F.softmax((t_out - before3[ids_rows]) / 0.04, -1).chunk(2)
    s3 = (s_out / 0.1).chunk(nc)
    manual3 = sum((-q3[iq] * F.log_softmax(s3[v], -1)).sum(-1).mean() for iq, v in pairs) / len(pairs)
    wrong_order = torch.repeat_interleave(cells3, 2)  # sample-major ids would be wrong (MultiCropWrapper is view-major)
    q_wrong = F.softmax((t_out - before3[wrong_order]) / 0.04, -1).chunk(2)
    manual_wrong = sum((-q_wrong[iq] * F.log_softmax(s3[v], -1)).sum(-1).mean() for iq, v in pairs) / len(pairs)
    check("each teacher row is centered with its own cell's non-zero center (view-major ids; sample-major would differ)",
          torch.allclose(loss3, manual3, atol=1e-6) and not torch.allclose(loss3, manual_wrong, atol=1e-6))
    try:
        l3(s_out, t_out, 0, torch.tensor([0, 1, 2]))
        rejected = False
    except ValueError:
        rejected = True
    check("a wrong number of cell ids is rejected", rejected)
    sd3 = l3.state_dict()
    check("cell centers and counts are in the loss state_dict (saved in checkpoint.pth)",
          "cell_center" in sd3 and "cell_count" in sd3 and "center" in sd3)

    print("5. One training step (epoch 0) and one in epoch 1")
    # Full precision here: with mixed precision the gradient scaler may skip the very first optimizer step
    # (its initial scale can overflow), which is normal but would hide the update checks. The mixed
    # precision path is exercised by the training runs in section 6.
    step_args = SimpleNamespace(**{**vars(args), "warmup_epochs": 0, "epochs": 2, "batch_size_per_gpu": 8,
                                   "log_every_steps": 0, "use_fp16": False})
    step_cells = np.array([0, 1, 1, 2, 0, 2, 1, 0])  # P4: 3 cells among the 8 images of the step
    loader = DataLoader(CellSSLDataset(ssl_paths[:8], DataAugmentationDINO(step_args), None, step_cells), batch_size=8)
    dino_loss, optimizer, scaler, (lr_s, wd_s, mom_s), _ = train_ssl.build_training_parts(step_args, student, 1, device, n_cells=3)
    check("training parts build the per-cell loss with 3 cells", isinstance(dino_loss, CondCenterDINOLoss) and dino_loss.n_cells == 3)
    bn_t, bn_s = teacher.backbone.features[0][1], student.backbone.features[0][1]
    s_before = {k: v.detach().clone() for k, v in student.state_dict().items()}
    t_before = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
    captured, t_outputs = [], []
    h1 = bn_t.register_forward_pre_hook(lambda m, inp: captured.append(inp[0].detach().float()))
    h2 = teacher.register_forward_hook(lambda m, i, o: t_outputs.append(o.detach().float()))
    student.train()
    teacher.train()
    stats = train_ssl.train_one_epoch(student, teacher, dino_loss, loader, optimizer, lr_s, wd_s, mom_s, 0,
                                      scaler, step_args, device)
    h1.remove()
    h2.remove()
    s_after, t_after = student.state_dict(), teacher.state_dict()
    params = dict(student.named_parameters())
    changed = {k for k in params if not torch.equal(s_after[k], s_before[k])}
    check("loss finite", math.isfinite(stats["loss"]), f"(loss {stats['loss']:.3f}, ln(out_dim) = {math.log(args.out_dim):.2f})")
    check("every student backbone parameter updated",
          all(k in changed for k in params if k.startswith("backbone.")), f"({sum(k.startswith('backbone.') for k in changed)} tensors)")
    check("student head MLP updated", all(k in changed for k in params if k.startswith("head.mlp.")))
    lw = "head.last_layer.parametrizations.weight.original1"
    lg = "head.last_layer.parametrizations.weight.original0"
    check("last layer frozen in epoch 0 (freeze_last_layer = 1), weight-norm scale fixed", lw not in changed and lg not in changed)
    check("teacher parameters have no gradients", all(p.grad is None for p in teacher.parameters()))
    m = mom_s[0]
    ema_err = max((t_after[k].float() - (m * t_before[k].float() + (1 - m) * s_after[k].float())).abs().max().item()
                  for k, _ in teacher.named_parameters())
    check("teacher parameters = EMA of student parameters", ema_err < 1e-6, f"(momentum {m}, max error {ema_err:.1e})")
    nbt = "backbone.features.0.1.num_batches_tracked"
    check("BatchNorm batch counters: teacher +1 (global views), student +2 (global, local groups)",
          (t_after[nbt] - t_before[nbt]).item() == 1 and (s_after[nbt] - s_before[nbt]).item() == 2)
    xin = captured[0]
    exp_mean = 0.9 * t_before["backbone.features.0.1.running_mean"].float() + 0.1 * xin.mean((0, 2, 3))
    exp_var = 0.9 * t_before["backbone.features.0.1.running_var"].float() + 0.1 * xin.var((0, 2, 3), unbiased=True)
    bn_err = max((t_after["backbone.features.0.1.running_mean"].float() - exp_mean).abs().max().item(),
                 ((t_after["backbone.features.0.1.running_var"].float() - exp_var).abs() / exp_var.abs().clamp_min(1e-3)).max().item())
    check("teacher BN running stats updated from its own batch (not by EMA)", bn_err < 2e-3,
          f"(first BN layer, max error {bn_err:.1e})")
    check("teacher and student BN running stats now differ",
          not torch.equal(t_after["backbone.features.0.1.running_mean"], s_after["backbone.features.0.1.running_mean"]))
    exp_center = 0.1 * t_outputs[0].mean(0, keepdim=True)
    check("global center still tracked: 0.9 * old + 0.1 * batch mean of teacher outputs",
          (dino_loss.center.float() - exp_center).abs().max().item() < 1e-3 and dino_loss.center.abs().sum() > 0)
    row_cells = torch.tensor(np.concatenate([step_cells, step_cells]), device=t_outputs[0].device)
    exp_cells = torch.stack([0.1 * t_outputs[0][row_cells == c].mean(0) for c in range(3)])
    check("per-cell centers updated from each cell's own teacher rows (real step)",
          (dino_loss.cell_center.float() - exp_cells).abs().max().item() < 1e-3 and dino_loss.cell_count.tolist() == [6, 6, 4])
    before_e1 = s_after[lw].clone()
    cc_before = dino_loss.cell_center.clone()
    t_outputs_e1 = []
    h3 = teacher.register_forward_hook(lambda m, i, o: t_outputs_e1.append(o.detach().float()))
    train_ssl.train_one_epoch(student, teacher, dino_loss, loader, optimizer, lr_s, wd_s, mom_s, 1, scaler, step_args, device)
    h3.remove()
    exp_e1 = torch.stack([0.9 * cc_before[c] + 0.1 * t_outputs_e1[0][row_cells == c].mean(0) for c in range(3)])
    check("epoch-1 step: per-cell centers follow 0.9 * old (non-zero) + 0.1 * mean of the cell's own rows",
          (dino_loss.cell_center.float() - exp_e1).abs().max().item() < 1e-3 and dino_loss.cell_count.tolist() == [12, 12, 8])
    check("last layer updated in epoch 1", not torch.equal(student.state_dict()[lw], before_e1))
    check("weight-norm scale still fixed", torch.equal(student.state_dict()[lg], s_before[lg]))
    del student, teacher, optimizer

    print("6. Save and resume (smoke runs, 48 images, 2 epochs)")
    a = make_run_dir(paths.SMOKE_OUTPUT_DIR, "stamp_collision_check_SMOKE")  # two folders asked for in the same second
    b = make_run_dir(paths.SMOKE_OUTPUT_DIR, "stamp_collision_check_SMOKE")
    check("a run folder whose stamp is already taken is skipped, never reused (jobs started in the same second)",
          a != b and a.is_dir() and b.is_dir() and b.name > a.name, f"({a.name[:15]} then {b.name[:15]})")
    a.rmdir()
    b.rmdir()
    straight = train_ssl.run(base_cfg, smoke=True)
    stopped = train_ssl.run(base_cfg, smoke=True, stop_after_epoch=1)
    info = json.loads((stopped / "run_info.json").read_text(encoding="utf-8"))
    check("stopped run is marked resumable after epoch 1", info["status"].startswith("stopped") and info["cost"]["epochs_done"] == 1)
    bad = json.loads(json.dumps(base_cfg))
    bad["dino"]["momentum_teacher"] = 0.996
    try:
        train_ssl.run(bad, resume_dir=stopped)
        rejected = False
    except ValueError as e:
        rejected = "momentum_teacher" in str(e)
    check("resume with a changed config (momentum_teacher) is rejected", rejected)
    bad2 = json.loads(json.dumps(base_cfg))
    bad2["cond_center"]["permute_cells"] = not base_cfg["cond_center"]["permute_cells"]
    try:
        train_ssl.run(bad2, resume_dir=stopped)
        rejected2 = False
    except ValueError as e:
        rejected2 = "cond_center" in str(e)
    check("resume with a changed cell definition (permute_cells) is rejected", rejected2)
    train_ssl.run(None, resume_dir=stopped)
    a = torch.load(straight / "checkpoint.pth", map_location="cpu", weights_only=False)
    b = torch.load(stopped / "checkpoint.pth", map_location="cpu", weights_only=False)
    check("same epoch, global step and schedule after resume", a["epoch"] == b["epoch"] == 2
          and a["global_step"] == b["global_step"] and a["schedule_info"] == b["schedule_info"])
    d_s, d_t = max_diff(a["student"], b["student"]), max_diff(a["teacher"], b["teacher"])
    d_c = (a["dino_loss"]["center"] - b["dino_loss"]["center"]).abs().max().item()
    d_cc = (a["dino_loss"]["cell_center"] - b["dino_loss"]["cell_center"]).abs().max().item()
    tol = 0.0 if device.type == "cpu" else 1e-3
    check("resumed run matches the uninterrupted run (cell centers included)", max(d_s, d_t, d_c, d_cc) <= tol
          and torch.equal(a["dino_loss"]["cell_count"], b["dino_loss"]["cell_count"]),
          f"(max difference student {d_s:.1e}, teacher {d_t:.1e}, center {d_c:.1e}, cell centers {d_cc:.1e}; tolerance {tol} on "
          f"{device.type}: cuDNN picks non-deterministic kernels on the GPU)")
    col = train_ssl.HISTORY_COLUMNS.index("cells_seen")
    h_rows = (straight / "history.csv").read_text().splitlines()
    check("smoke run wrote cells.csv and cell_center_stats.csv; history records cells_seen > 0",
          (straight / "cells.csv").exists() and (straight / "cell_center_stats.csv").exists()
          and h_rows[0].split(",")[col] == "cells_seen" and int(float(h_rows[-1].split(",")[col])) > 0,
          f"(cells_seen {h_rows[-1].split(',')[col]} of {a['dino_loss']['cell_center'].shape[0]} cells in the 48-image smoke pool)")
    h_a = (straight / "history.csv").read_text().splitlines()
    h_b = (stopped / "history.csv").read_text().splitlines()
    check("learning rate and momentum per epoch identical after resume",
          [r.split(",")[6:9] for r in h_a] == [r.split(",")[6:9] for r in h_b])

    print("7. Export")
    exported = sorted(straight.glob("backbone_teacher_epoch*.pth"))[-1]
    e = torch.load(exported, map_location="cpu", weights_only=True)
    sd = e["backbone_state_dict"]
    fresh = build_model({"type": "none"}, 40).features
    fresh.load_state_dict(sd, strict=True)
    check("exported backbone loads strictly into a fresh MobileNetV2", len(sd) == 312
          and e["num_parameter_tensors"] == 156 and e["num_buffer_tensors"] == 156, "(156 parameters + 156 buffers)")
    check("export contains no DINO head or classifier", not any(k.startswith(("head", "classifier")) for k in sd))
    check("export equals the teacher backbone of the final checkpoint",
          all(torch.equal(sd[k], a["teacher"]["backbone.features." + k]) for k in sd) and e["source"] == "teacher")
    check("export records the P4 centering (ssl_config.cond_center and ssl_method)",
          e["ssl_config"]["cond_center"]["enabled"] is True and "per-condition" in e["ssl_method"]
          and e["ssl_config"]["cond_center"]["cell"] == base_cfg["cond_center"]["cell"])

    save_json({"ssl_run": str(straight), "exported": str(exported), "config": str(config_path)}, LATEST)
    imports_check()
    finish("SSL", f"{straight.name}, {stopped.name}")


def part_downstream(cli):
    """Section 8, using the backbone exported by the SSL part."""
    device = resolve_device(cli.device)
    latest = json.loads(LATEST.read_text(encoding="utf-8"))
    straight, exported = Path(latest["ssl_run"]), Path(latest["exported"])
    base_cfg = json.loads((cli.config or train_ssl.DEFAULT_CONFIG).read_text(encoding="utf-8"))
    sd = torch.load(exported, map_location="cpu", weights_only=True)["backbone_state_dict"]
    print("8. Downstream fine-tuning from the exported backbone")
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
    check("fine-tuning record carries the P4 cell definition",
          ft_info["initialization"]["ssl_record"]["ssl_config"]["cond_center"]["enabled"] is True)
    reloaded, ckpt = load_finetuned(ft_dir / "best.pt", len(class_names))
    val = train_supervised.first_per_class(read_manifest("validation.csv", class_names), train_supervised.SMOKE_PER_CLASS)
    loader = DataLoader(SARDataset(val, build_transform(ft_cfg["data"], train=False)), batch_size=64)
    acc = predict(reloaded.to(device), loader, device)["acc"]
    check("reloaded best.pt reproduces the recorded validation accuracy", abs(acc - ckpt["val_acc"]) <= 1 / len(val),
          f"({acc:.4f} vs {ckpt['val_acc']:.4f} recorded; tolerance one image of {len(val)}, because the GPU "
          f"can pick different kernels after a reload and flip a near-tie)")
    del reloaded
    m = evaluate(ft_dir, "validation", cli.device)
    check("evaluate.py --split validation uses the run's manifest copies and reproduces the record",
          abs(m["overall_accuracy"] - ckpt["val_acc"]) < 1e-6, f"({m['overall_accuracy']:.4f})")

    imports_check()
    finish("downstream", ft_dir.name)


def main():
    parser = argparse.ArgumentParser(description="P4 pipeline checks (not a result)")
    parser.add_argument("--device", default="cuda", help='"cuda", "cpu" or "auto"')
    parser.add_argument("--config", type=Path, help="SSL config to test (default: config_ssl.json; e.g. config_ssl_perm.json)")
    parser.add_argument("--part", choices=["ssl", "downstream"], help="run one part only (default: both)")
    cli = parser.parse_args()
    if cli.part == "ssl":
        part_ssl(cli)
    elif cli.part == "downstream":
        part_downstream(cli)


if __name__ == "__main__":
    main()
