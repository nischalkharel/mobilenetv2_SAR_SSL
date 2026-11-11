"""Visualisations of a fine-tuned detector: prediction-vs-ground-truth panels with response maps.

    python -m shared.visualize --run <finetune run folder> [--checkpoint best] [--per-source 2] [--split test] [--smoke N]

For a fixed sample of official test images (per source dataset, seed 0) draws a row per image:
  ground truth (green) | predictions with score >= 0.5 (red, with class and score) | backbone response map
  (channel L2 norm of the last MobileNetV2 feature map, stride 32) | FPN P3 response (stride 8)
MobileNetV2 has no attention layers; these response maps are the CNN substitute. Writes
<run>/charts/panels_<split>.png and response_stats.json (share of response mass inside GT boxes).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from . import paths
from .coco_data import boxes_and_labels, load_image_tensor
from .detector import build_detector, load_finetune_config
from .metadata import read_metadata


def response_maps(model, img, device):
    """(detections, body map (h, w) normalised 0..1, P3 map, scale factor used by the transform)."""
    model.eval()
    with torch.no_grad():
        images, _ = model.transform([img.to(device)])
        body = model.backbone.body(images.tensors)
        fpn = model.backbone.fpn(body)
        det = model([img.to(device)])[0]
    last = body["3"][0].norm(dim=0)                    # (H/32, W/32)
    p3 = fpn["1"][0].norm(dim=0)                       # stride 8
    scale = images.image_sizes[0][0] / img.shape[1]
    norm = lambda m: ((m - m.min()) / (m.max() - m.min() + 1e-6)).cpu().numpy()
    return det, norm(last), norm(p3), scale, images.image_sizes[0]


def mass_in_boxes(m, boxes, size_hw, scale):
    """Share of the response map's mass that lies inside the (scaled) GT boxes."""
    h, w = m.shape
    mask = np.zeros_like(m, dtype=bool)
    for x1, y1, x2, y2 in boxes.numpy() * scale:
        fx, fy = size_hw[1] / w, size_hw[0] / h
        mask[int(y1 / fy):int(np.ceil(y2 / fy)) + 1, int(x1 / fx):int(np.ceil(x2 / fx)) + 1] = True
    return float(m[mask].sum() / (m.sum() + 1e-9)), float(mask.mean())


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best", choices=["best", "last"])
    ap.add_argument("--per-source", type=int, default=2)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--smoke", type=int, default=0, help="limit to N images in total")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run = Path(args.run)
    ckpt = torch.load(run / f"{args.checkpoint}.pt", map_location="cpu", weights_only=False)
    model = build_detector(load_finetune_config()["detector"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    rows = [r for r in read_metadata() if r["official_split"] == args.split]
    rng = random.Random(0)
    sample = []
    for source in sorted({r["source"] for r in rows}):
        rs = sorted([r for r in rows if r["source"] == source], key=lambda r: r["file_name"])
        sample.extend(rng.sample(rs, min(args.per_source, len(rs))))
    if args.smoke:
        sample = sample[:args.smoke]
    stats = []
    fig, axes = plt.subplots(len(sample), 4, figsize=(16, 3.6 * len(sample)), squeeze=False)
    for row_i, r in enumerate(sample):
        img = load_image_tensor(args.split, r["file_name"])
        gt_boxes, gt_labels, _ = boxes_and_labels(args.split, r["file_name"])
        det, last, p3, scale, size_hw = response_maps(model, img, device)
        gray = img[0].numpy()
        for ax in axes[row_i]:
            ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
        for (x1, y1, x2, y2), l in zip(gt_boxes, gt_labels):
            axes[row_i][0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", lw=1))
        axes[row_i][0].set_title(f"{r['source']} {r['file_name']}: GT ({len(gt_boxes)})", fontsize=8)
        keep = det["scores"] >= args.score
        for (x1, y1, x2, y2), l, s in zip(det["boxes"][keep].cpu(), det["labels"][keep].cpu(), det["scores"][keep].cpu()):
            axes[row_i][1].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red", lw=1))
            axes[row_i][1].text(x1, y1, f"{paths.CLASS_NAMES[int(l) - 1]} {s:.2f}", color="red", fontsize=6)
        axes[row_i][1].set_title(f"predictions >= {args.score} ({int(keep.sum())})", fontsize=8)
        h, w = gray.shape
        for ax, m, name in ((axes[row_i][2], last, "backbone response (stride 32)"), (axes[row_i][3], p3, "FPN P3 response (stride 8)")):
            ax.imshow(m, cmap="jet", alpha=0.5, extent=(0, w, h, 0), interpolation="bilinear")
            ax.set_title(name, fontsize=8)
        share_last, box_frac = mass_in_boxes(last, gt_boxes, size_hw, scale)
        share_p3, _ = mass_in_boxes(p3, gt_boxes, size_hw, scale)
        stats.append({"file_name": r["file_name"], "source": r["source"], "box_area_fraction": box_frac,
                      "response_in_boxes_backbone": share_last, "response_in_boxes_p3": share_p3})
    (run / "charts").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(run / "charts" / f"panels_{args.split}_{args.checkpoint}.png", dpi=110)
    plt.close(fig)
    summary = {"n": len(stats), "mean_box_area_fraction": float(np.mean([s["box_area_fraction"] for s in stats])),
               "mean_response_in_boxes_backbone": float(np.mean([s["response_in_boxes_backbone"] for s in stats])),
               "mean_response_in_boxes_p3": float(np.mean([s["response_in_boxes_p3"] for s in stats])), "per_image": stats}
    with open(run / "charts" / f"response_stats_{args.split}_{args.checkpoint}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {run / 'charts'}: {len(stats)} panels; response mass in GT boxes: backbone "
          f"{summary['mean_response_in_boxes_backbone']:.3f}, P3 {summary['mean_response_in_boxes_p3']:.3f} "
          f"(boxes cover {summary['mean_box_area_fraction']:.3f} of the image)")


if __name__ == "__main__":
    main()
