"""Segmentation lab: try label-free ways of splitting a SAR image into regions, and check them against the boxes.

    python RegionSSL/segment_lab.py --methods bright,cfar,felz,slic --n 12 [--split val] [--seed 0] [--sources HRSID,SSDD]

For each image: run every method, colour the regions, draw the ground-truth boxes (green) on top, and save one
panel per method (RegionSSL/outputs/segment_lab/<method>.png) plus summary.md with the scores below.
The boxes are used ONLY to score the methods; the methods themselves never see them.

Scores per method (over all GT boxes / regions of the sampled images):
    box hit rate    share of GT boxes that some region overlaps by at least --hit-iou (default 0.3) IoU,
                    or covers at least 50 % of the box area (a region that swallows the box also counts)
    boxes touched   share of GT boxes containing at least one region pixel (the softest "we noticed it")
    box coverage    mean share of each box's pixels that belong to some region
    region precision  share of region pixels that lie inside any GT box (how much of what we group is target)
    regions / image   how many regions the method proposes
    foreground %      share of the image covered by regions (background is "no region")

Methods (all label-free; a region = one connected blob of foreground pixels):
    bright   pixels above a high intensity quantile after a small blur (targets are the brightest scatterers)
    cfar     constant-false-alarm-rate style: a pixel is foreground when it is far above the mean of a ring of
             background around it (classic SAR target detection, no labels)
    felz     Felzenszwalb graph segmentation on the smoothed image; every superpixel brighter than the image
             median by a margin becomes a region (superpixels = "things that belong together" by edges)
    slic     SLIC superpixels, same brightness rule
    bright_shadow  'bright' regions grown to include the dark shadow next to them (SAR targets cast one)
    *_grouped / cfar_wide   the same, then nearby fragments are joined into one region (group_nearby): an
             aircraft or a row of cars is a cluster of scatterers, not one blob
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import warnings

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import segmentation, filters, morphology

warnings.filterwarnings("ignore", category=FutureWarning)   # scikit-image 0.26 renames a few morphology helpers

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "SARDet"))           # the detection suite's shared package (paths, data)
from shared import paths  # noqa: E402
from shared.coco_data import boxes_and_labels  # noqa: E402
from shared.metadata import read_metadata  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------
# Region proposers. Each takes a float image in [0, 1] and returns an int32 label map (0 = background, 1..K = regions).
# ---------------------------------------------------------------------------------------------------------------
def regions_bright(img, quantile=0.985, blur=1.0, min_area=6, grow=2):
    """Threshold the smoothed image at a high quantile; connected components are the regions."""
    sm = filters.gaussian(img, sigma=blur)
    fg = sm >= np.quantile(sm, quantile)
    fg = morphology.remove_small_objects(fg, min_area)
    fg = morphology.binary_dilation(fg, morphology.disk(grow))
    labels, _ = ndimage.label(fg)
    return labels.astype(np.int32)


def regions_cfar(img, guard=3, ring=8, k=3.0, min_area=6, grow=2):
    """Pixel is foreground when it exceeds the mean of its background ring by k standard deviations.

    Guard and ring are in pixels: the ring is the square annulus between radius `guard` and `guard + ring`.
    """
    outer = (2 * (guard + ring) + 1)
    inner = (2 * guard + 1)
    box_outer = ndimage.uniform_filter(img, outer)
    box_inner = ndimage.uniform_filter(img, inner)
    n_outer, n_inner = outer ** 2, inner ** 2
    ring_mean = (box_outer * n_outer - box_inner * n_inner) / (n_outer - n_inner)
    sq_outer = ndimage.uniform_filter(img ** 2, outer)
    sq_inner = ndimage.uniform_filter(img ** 2, inner)
    ring_var = (sq_outer * n_outer - sq_inner * n_inner) / (n_outer - n_inner) - ring_mean ** 2
    ring_std = np.sqrt(np.clip(ring_var, 1e-6, None))
    fg = img > ring_mean + k * ring_std
    fg = morphology.remove_small_objects(fg, min_area)
    fg = morphology.binary_dilation(fg, morphology.disk(grow))
    labels, _ = ndimage.label(fg)
    return labels.astype(np.int32)


def _superpixels_to_regions(img, sp, margin=0.12):
    """Keep superpixels whose mean brightness is above the image median by `margin`; merge touching ones."""
    med = np.median(img)
    means = ndimage.mean(img, labels=sp, index=np.arange(sp.max() + 1))
    keep = means > med + margin
    fg = keep[sp]
    labels, _ = ndimage.label(fg)
    return labels.astype(np.int32)


def regions_felz(img, scale=60, sigma=1.0, min_size=12, margin=0.12):
    sp = segmentation.felzenszwalb(img, scale=scale, sigma=sigma, min_size=min_size)
    return _superpixels_to_regions(img, sp, margin)


def regions_slic(img, n_segments=400, compactness=0.05, margin=0.12):
    sp = segmentation.slic(img, n_segments=n_segments, compactness=compactness, channel_axis=None, start_label=0)
    return _superpixels_to_regions(img, sp, margin)


def regions_bright_shadow(img, quantile=0.985, dark_quantile=0.08, blur=1.0, min_area=6, reach=6):
    """'bright' regions plus the dark shadow pixels within `reach` pixels of them."""
    bright = regions_bright(img, quantile, blur, min_area, grow=1) > 0
    sm = filters.gaussian(img, sigma=blur)
    dark = sm <= np.quantile(sm, dark_quantile)
    near = morphology.binary_dilation(bright, morphology.disk(reach))
    fg = bright | (dark & near)
    fg = morphology.binary_closing(fg, morphology.disk(2))
    labels, _ = ndimage.label(fg)
    return labels.astype(np.int32)


def group_nearby(labels, reach=5, min_area=12):
    """Merge regions whose pixels lie within `reach` pixels of each other (a target is a CLUSTER of scatterers:
    an aircraft or a row of cars breaks into many small bright fragments; this joins them), then drop tiny ones."""
    fg = labels > 0
    joined = morphology.binary_closing(fg, morphology.disk(reach))
    joined = morphology.remove_small_objects(joined, min_area)
    out, _ = ndimage.label(joined)
    return out.astype(np.int32)


def regions_bright_grouped(img):
    return group_nearby(regions_bright(img, grow=1), reach=5)


def regions_cfar_wide(img):
    """CFAR with a ring wide enough that a 40 px target does not sit inside its own background estimate."""
    return group_nearby(regions_cfar(img, guard=10, ring=14, k=2.5, grow=1), reach=5)


def regions_felz_grouped(img):
    return group_nearby(regions_felz(img), reach=5)


METHODS = {"bright": regions_bright, "cfar": regions_cfar, "felz": regions_felz, "slic": regions_slic,
           "bright_shadow": regions_bright_shadow, "bright_grouped": regions_bright_grouped,
           "cfar_wide": regions_cfar_wide, "felz_grouped": regions_felz_grouped}


# ---------------------------------------------------------------------------------------------------------------
# Scoring against the boxes
# ---------------------------------------------------------------------------------------------------------------
def score(labels, boxes, hit_iou=0.3):
    """box hits, region pixels inside boxes, region pixels total, number of regions."""
    h, w = labels.shape
    box_mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        box_mask[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] = True
    n_regions = int(labels.max())
    hits, touched, covered = 0, 0, 0.0
    for x1, y1, x2, y2 in boxes:
        y1i, y2i, x1i, x2i = int(y1), int(np.ceil(y2)), int(x1), int(np.ceil(x2))
        inside = labels[y1i:y2i, x1i:x2i]
        box_area = max(1, (y2i - y1i) * (x2i - x1i))
        touched += int((inside > 0).any())
        covered += float((inside > 0).mean()) if inside.size else 0.0
        best = 0.0
        for r in np.unique(inside):
            if r == 0:
                continue
            region_area = int((labels == r).sum())
            inter = int((inside == r).sum())
            iou = inter / (region_area + box_area - inter)
            cover = inter / box_area
            best = max(best, iou, cover if cover >= 0.5 else 0.0)
        hits += int(best >= min(hit_iou, 0.5))
    fg = labels > 0
    return hits, int((fg & box_mask).sum()), int(fg.sum()), n_regions, touched, covered


# ---------------------------------------------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------------------------------------------
def colour_regions(img, labels, rng):
    """Grey image with each region tinted in its own colour (alpha blend)."""
    rgb = np.stack([img] * 3, -1)
    out = rgb.copy()
    palette = rng.uniform(0.3, 1.0, size=(max(labels.max(), 1) + 1, 3))
    palette[0] = 0
    tint = palette[labels]
    mask = labels > 0
    out[mask] = 0.45 * rgb[mask] + 0.55 * tint[mask]
    return np.clip(out, 0, 1)


def draw_boxes(ax, boxes, colour="lime"):
    from matplotlib.patches import Rectangle
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=colour, linewidth=1.2))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", default="bright,bright_grouped,cfar_wide,felz,felz_grouped")
    ap.add_argument("--n", type=int, default=12, help="images to sample (spread over the source datasets)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--sources", default=None, help="comma list to restrict the sources, e.g. HRSID,SSDD")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hit-iou", type=float, default=0.3)
    ap.add_argument("--out", default=str(HERE / "outputs" / "segment_lab"))
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",")]
    for m in methods:
        if m not in METHODS:
            raise SystemExit(f"unknown method {m}; choose from {list(METHODS)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [r for r in read_metadata() if r["official_split"] == args.split]
    if args.sources:
        keep = set(args.sources.split(","))
        rows = [r for r in rows if r["source"] in keep]
    rng = random.Random(args.seed)
    by_source = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    sample = []
    sources = sorted(by_source)
    while len(sample) < args.n and any(by_source.values()):
        for s in sources:
            if by_source[s] and len(sample) < args.n:
                sample.append(by_source[s].pop(rng.randrange(len(by_source[s]))))
    print(f"{len(sample)} images from {args.split}: " + ", ".join(f"{r['source']}/{r['file_name']}" for r in sample))

    images = []
    for r in sample:
        with Image.open(paths.image_path(args.split, r["file_name"])) as im:
            img = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        boxes, _, _ = boxes_and_labels(args.split, r["file_name"])
        images.append((r, img, boxes.numpy()))

    totals = {}
    per_image = []
    for m in methods:
        fn = METHODS[m]
        tot = {"boxes": 0, "hits": 0, "fg_in_box": 0, "fg": 0, "regions": 0, "pixels": 0, "touched": 0, "covered": 0.0}
        ncol = 3
        nrow = len(images)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow), squeeze=False)
        for i, (r, img, boxes) in enumerate(images):
            labels = fn(img)
            hits, fg_in_box, fg, n_reg, touched, covered = score(labels, boxes, args.hit_iou)
            tot["boxes"] += len(boxes); tot["hits"] += hits; tot["fg_in_box"] += fg_in_box; tot["fg"] += fg
            tot["touched"] += touched; tot["covered"] += covered
            tot["regions"] += n_reg; tot["pixels"] += img.size
            per_image.append({"method": m, "file": r["file_name"], "source": r["source"], "boxes": len(boxes), "hits": hits, "touched": touched,
                              "regions": n_reg, "foreground_pct": round(100 * fg / img.size, 2),
                              "precision_pct": round(100 * fg_in_box / max(fg, 1), 2)})
            crng = np.random.default_rng(args.seed)
            axes[i][0].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[i][0].set_title(f"{r['source']} {r['file_name']} ({len(boxes)} boxes)", fontsize=8)
            axes[i][1].imshow(colour_regions(img, labels, crng))
            axes[i][1].set_title(f"{m}: {n_reg} regions, {100 * fg / img.size:.1f} % foreground", fontsize=8)
            axes[i][2].imshow(colour_regions(img, labels, crng))
            draw_boxes(axes[i][2], boxes)
            axes[i][2].set_title(f"boxes hit {hits}/{len(boxes)}, touched {touched}/{len(boxes)}", fontsize=8)
            for ax in axes[i]:
                ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out / f"{m}.png", dpi=100)
        plt.close(fig)
        totals[m] = tot
        print(f"{m:14s} box hit rate {100 * tot['hits'] / max(tot['boxes'], 1):5.1f} %   touched {100 * tot['touched'] / max(tot['boxes'], 1):5.1f} %   "
              f"box coverage {100 * tot['covered'] / max(tot['boxes'], 1):5.1f} %   region precision "
              f"{100 * tot['fg_in_box'] / max(tot['fg'], 1):5.1f} %   regions/image {tot['regions'] / len(images):5.1f}   "
              f"foreground {100 * tot['fg'] / tot['pixels']:5.1f} %   -> {out / (m + '.png')}")

    lines = [f"# Segmentation lab: {len(images)} {args.split} images, hit = IoU >= {args.hit_iou} or >= 50 % of the box covered", "",
             "| method | box hit rate | boxes touched | box coverage | region precision | regions / image | foreground % |", "|---|---|---|---|---|---|---|"]
    for m, t in totals.items():
        lines.append(f"| {m} | {100 * t['hits'] / max(t['boxes'], 1):.1f} % ({t['hits']}/{t['boxes']}) | {100 * t['touched'] / max(t['boxes'], 1):.1f} % | "
                     f"{100 * t['covered'] / max(t['boxes'], 1):.1f} % | {100 * t['fg_in_box'] / max(t['fg'], 1):.1f} % | "
                     f"{t['regions'] / len(images):.1f} | {100 * t['fg'] / t['pixels']:.1f} % |")
    lines += ["", "A good proposer has a high hit rate (it finds the targets) AND a high precision (it does not paint the whole",
              "image). Panels: one PNG per method, columns = image | regions | regions + GT boxes (green)."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)
    json.dump({"args": vars(args), "images": [r["file_name"] for r, _, _ in images]}, open(out / "sample.json", "w"), indent=1)
    print(f"\nwrote {out / 'summary.md'}")


if __name__ == "__main__":
    main()
