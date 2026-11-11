"""Smoke-test steps shared by every experiment folder (each smoke_test.py calls these, then its own SSL step).

Every check prints "ok" or raises; a smoke test is a pipeline check on a few images, never a result.
"""
import sys
import time
from pathlib import Path

import torch

from . import manifests, paths
from .backbone import fresh_features
from .coco_data import DetectionDataset, build_loader
from .detector import build_detector, count_parameters, head_state_fingerprint, load_finetune_config
from .ssl_data import SSLPoolDataset, collate_views, load_pool_rows, to_network_input, view_box_coverage


def step(msg):
    print(f"\n== {msg}", flush=True)


def check_paths():
    step("paths")
    if not paths.check():
        raise SystemExit("missing paths; run data/make_metadata.py and data/make_splits.py, or set the environment variables")
    audit = manifests.quick_audit()
    print(f"  split {audit['split']} audit ok: {audit['counts']}")
    return audit


def check_detection_data(n=4):
    step("detection data")
    rows = manifests.read(manifests.labeled_csv("1pct"))[:n]
    ds = DetectionDataset("train", [r["file_name"] for r in rows], train=True)
    loader = build_loader(ds, 2, shuffle=False, num_workers=0)
    images, targets = next(iter(loader))
    for im, t in zip(images, targets):
        assert im.dtype == torch.float32 and im.shape[0] == 3 and 0 <= im.min() and im.max() <= 1
        assert t["boxes"].shape[1] == 4 and (t["boxes"][:, 2] > t["boxes"][:, 0]).all() and (t["boxes"][:, 3] > t["boxes"][:, 1]).all()
        assert t["labels"].min() >= 1 and t["labels"].max() <= len(paths.CLASS_NAMES)
    print(f"  ok: {len(ds)} images; first {tuple(images[0].shape)} with {len(targets[0]['boxes'])} boxes")


def check_detector(device):
    step("detector build + identical heads across backbones")
    det_cfg = load_finetune_config()["detector"]
    m1 = build_detector(det_cfg, None, head_seed=0)
    other = fresh_features().state_dict()            # a 'different backbone' (random weights)
    m2 = build_detector(det_cfg, other, head_seed=0)
    assert head_state_fingerprint(m1) == head_state_fingerprint(m2), "head init differs between backbones"
    m1 = m1.to(device).train()
    ds = DetectionDataset("train", [r["file_name"] for r in manifests.read(manifests.labeled_csv("1pct"))[:2]], train=True)
    images, targets = next(iter(build_loader(ds, 2, False, 0)))
    losses = m1([im.to(device) for im in images], [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets])
    assert all(torch.isfinite(v) for v in losses.values()), losses
    m1.eval()
    with torch.no_grad():
        out = m1([images[0].to(device)])[0]
    print(f"  ok: params {count_parameters(m1)}; losses {{{', '.join(f'{k} {float(v.detach()):.3f}' for k, v in losses.items())}}}; "
          f"{len(out['boxes'])} detections on an untrained head")


def check_ssl_views(view_plan, n=4):
    step("SSL views")
    rows = load_pool_rows(limit=200)[:n]
    ds = SSLPoolDataset(rows, view_plan, seed=0)
    views, idx = collate_views([ds[i] for i in range(n)])
    sizes = [tuple(v.shape) for v in views]
    for v in views:
        assert v.dtype == torch.uint8 and v.shape[0] == n and v.shape[1] == 1
    x = to_network_input(views[0])
    assert x.shape[1] == 3 and x.dtype == torch.float32
    cov = view_box_coverage(load_pool_rows(limit=2000), n_images=100)
    print(f"  ok: {len(views)} view slots {sizes}; views with a GT box: " +
          ", ".join(f"{k} {v['views_with_a_box']:.2f}" for k, v in cov.items()))


def run_finetune_smoke(experiment_dir, run_name, backbone="imagenet", n_images=8, device=None):
    step(f"fine-tune smoke ({run_name}, {n_images} images, 1 epoch, backbone {Path(str(backbone)).name})")
    from .finetune import main as finetune_main
    argv = ["--experiment-dir", str(experiment_dir), "--run-name", run_name, "--budget", "1pct", "--backbone", str(backbone),
            "--smoke", str(n_images), "--epochs", "1"]
    if device:
        argv += ["--device", device]
    t0 = time.time()
    run_dir = finetune_main(argv)
    for f in ["best.pt", "last.pt", "history.csv", "run_info.json", "eval_best_test/metrics.json", "eval_best_val/metrics.json",
              "charts/training_curves.png"]:
        assert (run_dir / f).exists(), f"missing {f}"
    print(f"  ok in {time.time() - t0:.0f} s: {run_dir}")
    return run_dir


def run_analysis_smoke(run_dir, backbone_spec, device=None):
    step("probes + visualisation smoke")
    from .probes import main as probes_main
    from .visualize import main as vis_main
    argv = ["--backbone", str(backbone_spec), "--out", str(Path(run_dir) / "probes"), "--smoke", "24"]
    if device:
        argv += ["--device", device]
    probes_main(argv)
    argv = ["--run", str(run_dir), "--smoke", "3", "--per-source", "1"]
    if device:
        argv += ["--device", device]
    vis_main(argv)
    print("  ok")


def device_arg(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--cpu" in argv:
        return "cpu"
    return None
