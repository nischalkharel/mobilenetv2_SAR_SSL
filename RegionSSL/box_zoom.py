"""Zoom in on every ground-truth box and show what the region rule did inside it.

    python RegionSSL/box_zoom.py --method felz_grouped --sources SADD,SAR-AIRcraft --n 6

For each box: a crop of the image (with margin) | the same crop with regions coloured and the box drawn.
The title gives the two numbers the lab scores with:
    cover = share of the box's pixels that lie in ANY region        ("did we paint the object?")
    IoU   = overlap of the single best region with the box          ("did ONE region equal the object?")
A box counts as HIT when IoU >= 0.3 or cover >= 0.5; it counts as TOUCHED when cover > 0.
Output: RegionSSL/outputs/box_zoom/<method>_<sources>.png
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

import segment_lab as lab

HERE = Path(__file__).resolve().parent


def per_box(labels, box):
    x1, y1, x2, y2 = [int(v) for v in (box[0], box[1], np.ceil(box[2]), np.ceil(box[3]))]
    inside = labels[y1:y2, x1:x2]
    box_area = max(1, inside.size)
    cover = float((inside > 0).mean()) if inside.size else 0.0
    best_iou = 0.0
    for r in np.unique(inside):
        if r == 0:
            continue
        inter = int((inside == r).sum())
        best_iou = max(best_iou, inter / (int((labels == r).sum()) + box_area - inter))
    return cover, best_iou


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="felz_grouped")
    ap.add_argument("--sources", default="SADD,SAR-AIRcraft")
    ap.add_argument("--n", type=int, default=6, help="images (spread over the sources)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.6, help="crop margin as a fraction of the box size")
    ap.add_argument("--max-boxes", type=int, default=24)
    args = ap.parse_args()

    rows = [r for r in lab.read_metadata() if r["official_split"] == args.split and r["source"] in set(args.sources.split(","))]
    rng = random.Random(args.seed)
    by_source = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    sample = []
    while len(sample) < args.n and any(by_source.values()):
        for s in sorted(by_source):
            if by_source[s] and len(sample) < args.n:
                sample.append(by_source[s].pop(rng.randrange(len(by_source[s]))))

    fn = lab.METHODS[args.method]
    crops = []
    for r in sample:
        with Image.open(lab.paths.image_path(args.split, r["file_name"])) as im:
            img = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        boxes, _, _ = lab.boxes_and_labels(args.split, r["file_name"])
        labels = fn(img)
        coloured = lab.colour_regions(img, labels, np.random.default_rng(args.seed))
        for box in boxes.numpy():
            cover, iou = per_box(labels, box)
            crops.append((r, img, coloured, box, cover, iou))
    crops = crops[: args.max_boxes]

    ncol = 4
    nrow = int(np.ceil(len(crops) / 2))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow), squeeze=False)
    for k, (r, img, coloured, box, cover, iou) in enumerate(crops):
        x1, y1, x2, y2 = box
        mw, mh = args.margin * (x2 - x1), args.margin * (y2 - y1)
        cx1, cy1 = int(max(0, x1 - mw)), int(max(0, y1 - mh))
        cx2, cy2 = int(min(img.shape[1], np.ceil(x2 + mw))), int(min(img.shape[0], np.ceil(y2 + mh)))
        local = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
        ax_img, ax_reg = axes[k // 2][2 * (k % 2)], axes[k // 2][2 * (k % 2) + 1]
        ax_img.imshow(img[cy1:cy2, cx1:cx2], cmap="gray", vmin=0, vmax=1)
        lab.draw_boxes(ax_img, [local])
        ax_img.set_title(f"{r['source']} {r['file_name']}  box {int(x2 - x1)}x{int(y2 - y1)} px", fontsize=8)
        ax_reg.imshow(coloured[cy1:cy2, cx1:cx2])
        lab.draw_boxes(ax_reg, [local])
        hit = iou >= 0.3 or cover >= 0.5
        ax_reg.set_title(f"cover {100 * cover:.0f} %   best IoU {iou:.2f}   {'HIT' if hit else 'touched only' if cover > 0 else 'MISS'}", fontsize=8)
        print(f"{r['source']:13s} {r['file_name']}  box {int(x2 - x1):4d}x{int(y2 - y1):<4d}  cover {100 * cover:5.1f} %  best IoU {iou:.2f}  {'HIT' if hit else 'touched' if cover > 0 else 'MISS'}")
    for row in axes:
        for ax in row:
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = HERE / "outputs" / "box_zoom"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.method}_{args.sources.replace(',', '-')}.png"
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
