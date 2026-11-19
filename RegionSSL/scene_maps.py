"""Where do the frozen SARDet-trained backbones respond on SARDet-100K test SCENES, before any fine-tuning?

The scene counterpart of SARDet_to_ATR/response_maps.py (which measures the same backbones on ATRNet chips). Here the
images are the official SARDet-100K test scenes at the detector's working size (shorter side 512, like the SSL views
and the detection fine-tune), and the reference is the dataset's annotated object boxes, read only to measure the map:
  response map   = L2 norm over channels of the last feature map of the frozen backbone (stride 32: 16 x 16 cells for
                   a 512 x 512 scene), upsampled bilinearly to the scene
  mass in boxes  = share of the map's mass inside the union of the annotated boxes (a uniform map scores the boxes'
                   area share)
  concentration  = mass in boxes / boxes' area share (1 = no preference for the objects)
  peak in boxes  = share of scenes whose strongest pixel lies inside a box
over a fixed random sample of test scenes with at least one object (same scenes for every model and split; every
difference is paired per scene), overall and per source dataset. Backbones: the ImageNet start and, per split, the
suite's DINO / LeJEPA / SparK and the four RegionDINO runs (object-only, scene+object, their random-window controls),
located with SARDet_to_ATR/find_backbone.py (split s = SSL seed s - 1). Missing runs are skipped with a note.

Run from the repository root (GPU: a few minutes for 3 splits; CPU smoke: --images 8):
    python RegionSSL/scene_maps.py
    python RegionSSL/scene_maps.py --images 1500 --rows 6 --splits 1,2,3 --figure-split 1
    python RegionSSL/scene_maps.py --images 8 --rows 2 --splits 1 --device cpu --model "smoke=path/to/backbone.pth" --out outputs/scene_maps_smoke
Writes RegionSSL/outputs/scene_maps/scene_maps.md, scene_maps_per_model.csv, scene_maps_per_source.csv,
scene_maps_per_image.csv and scene_maps_split<k>.png / .pdf. Analysis only; nothing here is used by training. The
SARDet suite is imported read-only.
"""
import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "SARDet"))            # the suite's shared/ package (read-only)
sys.path.insert(0, str(REPO / "SARDet_to_ATR"))     # find_backbone (read-only)
from shared.backbone import fresh_features, load_backbone_file, load_imagenet_mobilenet_v2  # noqa: E402
from shared.coco_data import boxes_and_labels, load_annotations  # noqa: E402
from shared.metadata import read_metadata  # noqa: E402
from shared.ssl_data import load_resized_gray, to_network_input  # noqa: E402
from find_backbone import find  # noqa: E402

METHODS = [("DINO on SARDet", "dino"), ("LeJEPA on SARDet", "lejepa"), ("SparK on SARDet", "spark"),
           ("RegionDINO scene+object", "regiondino_so"), ("RegionDINO scene+object, random windows", "regiondino_so_random"),
           ("RegionDINO object-only", "regiondino_o"), ("RegionDINO object-only, random windows", "regiondino_o_random")]
IMAGENET = "ImageNet start (no SSL)"
REFERENCE = "DINO on SARDet"
SIZE = 512


def models_for_split(split: int, overrides: dict, only: set | None) -> dict:
    table = {IMAGENET: None}
    for label, key in METHODS:
        try:
            table[label] = find(key, split)
        except SystemExit as e:
            print(f"  split {split}: skipped {label}: {e}")
    table.update(overrides)
    if only:
        table = {k: v for k, v in table.items() if k in only}
    return table


def load_features(path):
    if path is None:
        return load_imagenet_mobilenet_v2().features
    state, _ = load_backbone_file(path)
    f = fresh_features()
    f.load_state_dict(state, strict=True)
    return f


def test_scenes(n: int, rng: np.random.Generator) -> list:
    """Metadata rows of n random official test scenes that have at least one annotated object."""
    anns = load_annotations("test")
    rows = [r for r in read_metadata() if r["official_split"] == "test" and anns["anns"].get(r["image_id"])]
    rows.sort(key=lambda r: r["file_name"])
    idx = np.sort(rng.choice(len(rows), min(n, len(rows)), replace=False))
    return [rows[i] for i in idx]


def box_mask(row: dict, h: int, w: int) -> np.ndarray:
    """0/1 mask of the union of the GT boxes, in the resized scene's pixel grid."""
    gt, _, _ = boxes_and_labels("test", row["file_name"])
    s = min(SIZE / min(row["width"], row["height"]), SIZE / max(row["width"], row["height"]))
    mask = np.zeros((h, w), dtype=np.float32)
    for x1, y1, x2, y2 in (gt.numpy() * s):
        x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        x2, y2 = min(w, max(x1 + 1, int(np.ceil(x2)))), min(h, max(y1 + 1, int(np.ceil(y2))))
        mask[y1:y2, x1:x2] = 1.0
    return mask


@torch.no_grad()
def response_map(features, u8: torch.Tensor, device) -> np.ndarray:
    """(H, W) upsampled channel-norm map of one scene given as a (1, H, W) uint8 tensor."""
    x = to_network_input(u8[None], device)
    m = features(x).norm(dim=1)[:, None]
    return F.interpolate(m, size=u8.shape[-2:], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def measures(m: np.ndarray, mask: np.ndarray):
    total = float(m.sum())
    mass = float((m * mask).sum()) / total if total > 0 else float("nan")
    peak = float(mask.flat[int(m.argmax())])
    return mass, peak


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=int, default=1500, help="random official test scenes with objects (seed-fixed, shared by all models)")
    ap.add_argument("--rows", type=int, default=6, help="scenes drawn in the figure (one per source first)")
    ap.add_argument("--splits", default="1,2,3")
    ap.add_argument("--figure-split", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", action="append", default=[], metavar="LABEL=PATH", help="add or override a backbone file (any split)")
    ap.add_argument("--only", default="", help="comma-separated labels to keep (after --model)")
    ap.add_argument("--out", default="outputs/scene_maps", help="output folder, relative to RegionSSL/")
    a = ap.parse_args()
    overrides = {n: Path(p) for n, p in (spec.split("=", 1) for spec in a.model)}
    only = {s.strip() for s in a.only.split(",") if s.strip()} or None
    splits = [int(s) for s in a.splits.split(",")]
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    sample = test_scenes(a.images, rng)
    print(f"{len(sample)} test scenes with objects, device {device}")
    # scenes, masks and (for the figure) pixels, loaded once
    pixels, masks, area = [], [], []
    for r in sample:
        im = load_resized_gray("test", r["file_name"], SIZE)
        u8 = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy())[None]
        pixels.append(u8)
        mk = box_mask(r, u8.shape[1], u8.shape[2])
        masks.append(mk)
        area.append(float(mk.mean()))
    area = np.array(area)
    sources = sorted({r["source"] for r in sample})
    fig_pos = []
    for round_ in range(3):
        for src in sources:
            cands = [i for i, r in enumerate(sample) if r["source"] == src]
            if len(cands) > round_ and len(fig_pos) < a.rows:
                fig_pos.append(cands[round_])
    fig_pos = sorted(set(fig_pos))
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)

    lines = ["# Frozen backbones on SARDet-100K test scenes: where the response sits, before fine-tuning", "",
             f"{len(sample)} official test scenes with at least one annotated object (seed {a.seed}, the same scenes for every model "
             f"and split), shorter side {SIZE} px as in the SSL views and the detector; boxes = the dataset's annotations, mean area share "
             f"{100 * area.mean():.1f} % of the scene (what a uniform map scores as mass in boxes). Response map = channel norm of the last "
             "feature map of the frozen backbone (stride 32), no labels, no fine-tuning. Concentration = mass in boxes / area share. "
             "Peak in boxes = share of scenes whose strongest pixel lies inside a box.", ""]
    per_model, per_image, per_source_acc = [], {}, defaultdict(lambda: defaultdict(list))
    for split in splits:
        print(f"split {split}")
        table = models_for_split(split, overrides, only)
        mass, peak, maps_fig = {}, {}, {}
        for label, path in table.items():
            feats = load_features(path).to(device).eval()
            ms, ps, keep = [], [], {}
            for i, (u8, mk) in enumerate(zip(pixels, masks)):
                m = response_map(feats, u8, device)
                a_, b_ = measures(m, mk)
                ms.append(a_)
                ps.append(b_)
                if i in fig_pos:
                    keep[i] = m
            mass[label], peak[label], maps_fig[label] = np.array(ms), np.array(ps), keep
            print(f"  {label}: mass in boxes {100 * np.nanmean(mass[label]):.1f} %, peak in boxes {100 * peak[label].mean():.0f} %")
            del feats
        lines += [f"## Split {split}", "",
                  "| backbone | mass in boxes (%) | concentration | peak in boxes (%) | vs ImageNet start: mean, % scenes higher | "
                  f"vs {REFERENCE}: mean, % scenes higher |", "|---|---|---|---|---|---|"]
        for label in table:
            s = mass[label]
            row = {"split": split, "model": label, "mass_in_boxes_pct": round(100 * float(np.nanmean(s)), 2),
                   "concentration": round(float(np.nanmean(s / area)), 3), "peak_in_boxes_pct": round(100 * float(peak[label].mean()), 1)}
            cells = []
            for ref, tag in ((IMAGENET, "imagenet"), (REFERENCE, "dino")):
                if ref in mass and ref != label:
                    d = s - mass[ref]
                    row[f"vs_{tag}_mean_pts"] = round(100 * float(np.nanmean(d)), 2)
                    row[f"vs_{tag}_higher_pct"] = round(100 * float(np.nanmean(d > 0)), 1)
                    cells.append(f"{100 * np.nanmean(d):+.1f}, {100 * np.nanmean(d > 0):.0f} %")
                else:
                    cells.append("-")
            lines.append(f"| {label} | {row['mass_in_boxes_pct']:.1f} | {row['concentration']:.2f} | {row['peak_in_boxes_pct']:.1f} | {cells[0]} | {cells[1]} |")
            per_model.append(row)
            for i, r in enumerate(sample):
                per_image.setdefault((split, r["file_name"]), {"split": split, "file_name": r["file_name"], "source": r["source"],
                                                               "boxes_area_share": round(area[i], 4)})[f"{label} | mass"] = round(float(s[i]), 4)
                per_source_acc[(label, r["source"])]["mass"].append(float(s[i]))
                per_source_acc[(label, r["source"])]["peak"].append(float(peak[label][i]))
        lines.append("")
        if split == a.figure_split and fig_pos:
            cols = ["scene + object boxes"] + list(table)
            fig, axes = plt.subplots(len(fig_pos), len(cols), figsize=(2.3 * len(cols), 2.4 * len(fig_pos)))
            axes = np.atleast_2d(axes)
            for ri, i in enumerate(fig_pos):
                r, img, mk = sample[i], pixels[i][0].numpy(), masks[i]
                gt, _, _ = boxes_and_labels("test", r["file_name"])
                sc = min(SIZE / min(r["width"], r["height"]), SIZE / max(r["width"], r["height"]))
                for ci, col in enumerate(cols):
                    ax = axes[ri, ci]
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                    if ci > 0:
                        up = maps_fig[col][i]
                        ax.imshow(up / (up.max() + 1e-12), cmap="jet", alpha=0.45, vmin=0, vmax=1)
                        ax.set_title(f"{col}\nboxes {100 * mass[col][i]:.0f} %", fontsize=6.5)
                    else:
                        ax.set_title(f"{r['source']}  {r['file_name']}\n{len(gt)} objects, {100 * mk.mean():.1f} % of the scene", fontsize=6.5)
                    for x1, y1, x2, y2 in (gt.numpy() * sc):
                        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=0.8))
                    ax.set_xticks([])
                    ax.set_yticks([])
            fig.suptitle(f"Frozen backbone response maps on SARDet-100K test scenes (split {split} backbones, no fine-tuning); "
                         "white boxes = object annotations", fontsize=9)
            fig.tight_layout()
            fig.savefig(out / f"scene_maps_split{split}.png", dpi=170)
            fig.savefig(out / f"scene_maps_split{split}.pdf")
            plt.close(fig)
    # per source, averaged over the splits run
    labels = [IMAGENET] + [l for l, _ in METHODS] + list(overrides)
    labels = [l for l in labels if any((l, s) in per_source_acc for s in sources)]
    lines += ["## Per source dataset (all splits pooled): mass in boxes % / peak in boxes %", "",
              "| source | n scenes | boxes area % | " + " | ".join(labels) + " |", "|---|---|---|" + "---|" * len(labels)]
    per_source = []
    for src in sources:
        n = sum(1 for r in sample if r["source"] == src)
        ar = 100 * np.mean([area[i] for i, r in enumerate(sample) if r["source"] == src])
        cells, rec = [], {"source": src, "n_scenes": n, "boxes_area_pct": round(float(ar), 2)}
        for l in labels:
            acc = per_source_acc.get((l, src))
            if acc:
                m, p = 100 * np.nanmean(acc["mass"]), 100 * np.mean(acc["peak"])
                cells.append(f"{m:.1f} / {p:.0f}")
                rec[f"{l} | mass"], rec[f"{l} | peak"] = round(float(m), 2), round(float(p), 1)
            else:
                cells.append("-")
        lines.append(f"| {src} | {n} | {ar:.1f} | " + " | ".join(cells) + " |")
        per_source.append(rec)
    lines += ["", f"Written {datetime.now().isoformat(timespec='seconds')}."]
    (out / "scene_maps.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, rows_, first in (("scene_maps_per_model.csv", per_model, ("split", "model")),
                               ("scene_maps_per_image.csv", list(per_image.values()), ("split", "file_name", "source", "boxes_area_share")),
                               ("scene_maps_per_source.csv", per_source, ("source", "n_scenes", "boxes_area_pct"))):
        keys = sorted({k for r in rows_ for k in r}, key=lambda k: (k not in first, k))
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows_)
    print("\n".join(lines))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
