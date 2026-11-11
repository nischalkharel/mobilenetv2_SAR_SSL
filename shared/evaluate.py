"""COCO-style evaluation of a detector on an official split or a named subset of it.

Produces, per subset: metrics.json (mAP@[.5:.95], AP50, AP75, APs/m/l, per-class AP and AP50, per-source mAP
and AP50), per_class.csv, pr_curves.png (IoU 0.5, one curve per class), and detections.npz (image_id, label,
score, box) so later analyses (bootstrap CI, panels) never need the GPU again.
"""
import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import paths
from .coco_data import DetectionDataset, build_loader, coco_ground_truth
from .run_record import write_json


@torch.no_grad()
def run_detector(model, dataset, batch_size, num_workers, device, log_every=50):
    """Detections of every image: list of COCO result dicts (bbox xywh, category_id = label - 1)."""
    model.eval()
    loader = build_loader(dataset, batch_size, shuffle=False, num_workers=num_workers)
    results = []
    for b, (images, targets) in enumerate(loader):
        images = [im.to(device, non_blocking=True) for im in images]
        outputs = model(images)
        for t, o in zip(targets, outputs):
            boxes = o["boxes"].cpu().numpy()
            scores = o["scores"].cpu().numpy()
            labels = o["labels"].cpu().numpy()
            for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
                results.append({"image_id": int(t["image_id"]), "category_id": int(l) - 1,
                                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
        if log_every and (b + 1) % log_every == 0:
            print(f"    evaluated {min((b + 1) * batch_size, len(dataset))}/{len(dataset)} images", flush=True)
    return results


def _coco_eval(coco_gt, results, img_ids=None):
    from pycocotools.cocoeval import COCOeval
    if not results:
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        if img_ids is not None:
            ev.params.imgIds = sorted(img_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev


def _ap_of(ev, cat_index=None, iou_index=None, area=0, maxdet=-1):
    """Mean precision over recall thresholds (and IoUs unless iou_index given); -1 entries excluded."""
    p = ev.eval["precision"]                       # [T, R, K, A, M]
    p = p[:, :, :, area, maxdet]
    if iou_index is not None:
        p = p[iou_index:iou_index + 1]
    if cat_index is not None:
        p = p[:, :, cat_index:cat_index + 1]
    p = p[p > -1]
    return float(p.mean()) if p.size else float("nan")


def coco_metrics(coco_gt, results, img_ids=None) -> dict:
    ev = _coco_eval(coco_gt, results, img_ids)
    names = paths.CLASS_NAMES
    if ev is None:
        return {"map": 0.0, "map50": 0.0, "map75": 0.0, "map_small": 0.0, "map_medium": 0.0, "map_large": 0.0,
                "per_class": {n: {"ap": 0.0, "ap50": 0.0} for n in names}, "n_detections": 0}
    s = ev.stats
    metrics = {"map": float(s[0]), "map50": float(s[1]), "map75": float(s[2]), "map_small": float(s[3]),
               "map_medium": float(s[4]), "map_large": float(s[5]), "ar100": float(s[8]),
               "per_class": {n: {"ap": _ap_of(ev, k), "ap50": _ap_of(ev, k, iou_index=0)} for k, n in enumerate(names)},
               "n_detections": len(results), "n_images": len(ev.params.imgIds)}
    metrics["pr_curves_iou50"] = {n: ev.eval["precision"][0, :, k, 0, -1].tolist() for k, n in enumerate(names)}
    metrics["recall_thresholds"] = ev.params.recThrs.tolist()
    return metrics


def per_source_metrics(coco_gt, results, file_names, sources_of) -> dict:
    """mAP / AP50 restricted to the images of each source dataset (the 'condition' view of the results)."""
    id_of = {im["file_name"]: im["id"] for im in coco_gt.dataset["images"]}
    by_source = defaultdict(list)
    for f in file_names:
        by_source[sources_of[f]].append(id_of[f])
    out = {}
    for source, ids in sorted(by_source.items()):
        m = coco_metrics(coco_gt, [r for r in results if r["image_id"] in set(ids)], img_ids=ids)
        out[source] = {"n_images": len(ids), "map": m["map"], "map50": m["map50"],
                       "per_class_ap50": {n: v["ap50"] for n, v in m["per_class"].items()}}
    return out


def save_detections(path, results):
    n = len(results)
    image_id = np.fromiter((r["image_id"] for r in results), dtype=np.int64, count=n)
    label = np.fromiter((r["category_id"] for r in results), dtype=np.int8, count=n)
    score = np.fromiter((r["score"] for r in results), dtype=np.float32, count=n)
    box = np.array([r["bbox"] for r in results], dtype=np.float32).reshape(n, 4)
    np.savez_compressed(path, image_id=image_id, label=label, score=score, box=box)


def load_detections(path):
    z = np.load(path)
    return [{"image_id": int(i), "category_id": int(l), "bbox": b.tolist(), "score": float(s)}
            for i, l, s, b in zip(z["image_id"], z["label"], z["score"], z["box"])]


def plot_pr_curves(metrics, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    r = metrics["recall_thresholds"]
    for n, p in metrics["pr_curves_iou50"].items():
        p = np.array(p)
        ax.plot(r, np.where(p < 0, np.nan, p), label=f"{n} (AP50 {metrics['per_class'][n]['ap50']:.2f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title or "precision-recall at IoU 0.5")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def evaluate_subset(model, official_split, file_names, out_dir, det_cfg, batch_size, num_workers, device,
                    subset_name, sources_of=None, with_per_source=True, results=None):
    """Full evaluation of one subset; returns the metrics dict (also written to out_dir)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = DetectionDataset(official_split, file_names, train=False)
    if results is None:
        results = run_detector(model, dataset, batch_size, num_workers, device)
    coco_gt = coco_ground_truth(official_split, dataset.file_names)
    metrics = coco_metrics(coco_gt, results)
    metrics.update(subset=subset_name, official_split=official_split, n_images=len(dataset))
    if with_per_source and sources_of is not None:
        metrics["per_source"] = per_source_metrics(coco_gt, results, dataset.file_names, sources_of)
    save_detections(out_dir / "detections.npz", results)
    write_json(out_dir / "metrics.json", metrics)
    with open(out_dir / "per_class.csv", "w", encoding="utf-8") as f:
        f.write("class,ap,ap50\n")
        for n, v in metrics["per_class"].items():
            f.write(f"{n},{v['ap']:.4f},{v['ap50']:.4f}\n")
    plot_pr_curves(metrics, out_dir / "pr_curves.png", f"{subset_name}: PR at IoU 0.5 (mAP {metrics['map']:.3f})")
    return metrics


def summary_line(m: dict) -> str:
    return (f"mAP {m['map']:.4f}  AP50 {m['map50']:.4f}  AP75 {m['map75']:.4f}  "
            f"S/M/L {m['map_small']:.3f}/{m['map_medium']:.3f}/{m['map_large']:.3f}  n={m.get('n_images', '?')}")


def sources_from_metadata() -> dict:
    """file_name -> source for every image of the release (small; read once)."""
    from .metadata import read_metadata
    return {r["file_name"]: r["source"] for r in read_metadata()}
