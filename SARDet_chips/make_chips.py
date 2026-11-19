"""Cut one chip per annotated box of SARDet-100K into a single array file (DESIGN.md section 2 and 3).

    python SARDet_chips/make_chips.py                 # all splits, about 245,000 chips, 10 to 20 min on the PC
    python SARDet_chips/make_chips.py --limit 200     # first 200 images per split: a quick check (writes to a _smoke folder)
    python SARDet_chips/make_chips.py --dry-run       # only the crowding / fit table of the rule, nothing written

Rule: square window centred on the box, side = max(WINDOW_FLOOR, WINDOW_MARGIN x longer box side) in native pixels,
edge-padded where it leaves the image, resized to CHIP_SIZE x CHIP_SIZE, 8-bit grayscale.
Writes under SARDET_CHIPS_ROOT:
    chips_u8.npy      (N, CHIP_SIZE, CHIP_SIZE) uint8, one row per chip, in chips.csv order
    chips.csv         one row per chip: idx, official_split, file_name, image_id, ann_id, source, group, class_name, label,
                      box_x, box_y, box_w, box_h, window_side, clipped, n_others_inside, other_class_inside
    chips_meta.json   rule parameters, counts, SHA-256 of the csv and the array
    summary.md        counts per class x split and per source x class, flag rates
    previews/         <split>_<class>.png: an 8 x 8 grid of random chips (for looking, not for training)
A chip belongs to the official split of its image; train/val/test membership is never re-drawn here.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chip_paths as paths  # noqa: E402

COLUMNS = ["idx", "official_split", "file_name", "image_id", "ann_id", "source", "group", "class_name", "label",
           "box_x", "box_y", "box_w", "box_h", "window_side", "clipped", "n_others_inside", "other_class_inside"]


def load_split(split):
    """[(image dict, [annotations])] for one official split, sorted by file name (a stable order)."""
    d = json.loads(Path(paths.ANNOTATION_FILES[split]).read_text(encoding="utf-8"))
    anns = defaultdict(list)
    for a in d["annotations"]:
        anns[a["image_id"]].append(a)
    cats = {c["id"]: c["name"] for c in d["categories"]}
    if [cats[i] for i in sorted(cats)] != paths.CLASS_NAMES:
        raise ValueError(f"category names in {split}.json differ from CLASS_NAMES: {cats}")
    imgs = sorted((im for im in d["images"] if anns.get(im["id"])), key=lambda im: im["file_name"])
    return [(im, sorted(anns[im["id"]], key=lambda a: a["id"])) for im in imgs]


def plan_image(split, im, anns, floor, margin):
    """Chip rows for one image (everything but the pixels)."""
    boxes = np.array([a["bbox"] for a in anns], dtype=float)           # x, y, w, h
    cls = np.array([a["category_id"] for a in anns])
    cx, cy = boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2
    rows = []
    for k, a in enumerate(anns):
        side = max(floor, margin * max(boxes[k, 2], boxes[k, 3]))
        half = side / 2
        inside = (np.abs(cx - cx[k]) < half) & (np.abs(cy - cy[k]) < half)
        inside[k] = False
        x0, y0 = cx[k] - half, cy[k] - half
        clipped = x0 < 0 or y0 < 0 or x0 + side > im["width"] or y0 + side > im["height"]
        rows.append({"official_split": split, "file_name": im["file_name"], "image_id": im["id"], "ann_id": a["id"],
                     "class_name": paths.CLASS_NAMES[cls[k]], "label": int(cls[k]),
                     "box_x": boxes[k, 0], "box_y": boxes[k, 1], "box_w": boxes[k, 2], "box_h": boxes[k, 3],
                     "window_side": round(side, 1), "clipped": int(clipped),
                     "n_others_inside": int(inside.sum()), "other_class_inside": int((inside & (cls != cls[k])).any()),
                     "_cx": cx[k], "_cy": cy[k]})
    return rows


def cut_image(job):
    """Worker: read one image, return its chips as a (n, S, S) uint8 array in row order."""
    split, file_name, rows, size = job
    im = Image.open(paths.IMAGES_DIR / split / file_name)
    im.load()
    a = np.asarray(im.convert("L"))
    h, w = a.shape
    out = np.empty((len(rows), size, size), dtype=np.uint8)
    for i, r in enumerate(rows):
        side = int(round(r["window_side"]))
        x0, y0 = int(round(r["_cx"] - side / 2)), int(round(r["_cy"] - side / 2))
        # pad with the edge value only as much as this window needs, then crop
        px0, py0 = max(0, -x0), max(0, -y0)
        px1, py1 = max(0, x0 + side - w), max(0, y0 + side - h)
        src = a if not (px0 or py0 or px1 or py1) else np.pad(a, ((py0, py1), (px0, px1)), mode="edge")
        win = src[y0 + py0: y0 + py0 + side, x0 + px0: x0 + px0 + side]
        out[i] = np.asarray(Image.fromarray(win).resize((size, size), Image.BILINEAR))
    return out


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def write_previews(arr, rows, out_dir, seed=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    S = arr.shape[1]
    by = defaultdict(list)
    for r in rows:
        by[(r["official_split"], r["class_name"])].append(r["idx"])
    for (split, cls), idxs in sorted(by.items()):
        pick = rng.sample(idxs, min(64, len(idxs)))
        grid = np.zeros((8 * S, 8 * S), dtype=np.uint8)
        for j, i in enumerate(pick):
            grid[(j // 8) * S:(j // 8 + 1) * S, (j % 8) * S:(j % 8 + 1) * S] = arr[i]
        Image.fromarray(grid).save(out_dir / f"{split}_{cls}.png")


def summary_text(rows, meta):
    L = []
    L.append(f"# SARDet-100K chips: {meta['n_chips']} chips from {meta['n_images']} images, written {meta['written']}\n")
    L.append(f"Rule: window side = max({meta['window_floor']}, {meta['window_margin']} x longer box side) native px, "
             f"edge-padded, resized to {meta['chip_size']}. Source: {meta['dataset_root']}\n")
    splits = ["train", "val", "test"]
    c = Counter((r["official_split"], r["class_name"]) for r in rows)
    L.append("## Chips per class and official split\n\n| class | " + " | ".join(splits) + " |\n|---|" + "---|" * len(splits))
    for cls in paths.CLASS_NAMES:
        L.append(f"| {cls} | " + " | ".join(str(c[(s, cls)]) for s in splits) + " |")
    srcs = sorted({r["source"] for r in rows})
    cs = Counter((r["source"], r["class_name"]) for r in rows if r["official_split"] == "train")
    L.append("\n## Train chips per source and class\n\n| source | " + " | ".join(paths.CLASS_NAMES) + " |\n|---|" + "---|" * len(paths.CLASS_NAMES))
    for s in srcs:
        L.append(f"| {s} | " + " | ".join(str(cs[(s, cls)]) for cls in paths.CLASS_NAMES) + " |")
    L.append("\n## Flags on test chips (share of the class)\n\n| class | n | holds another object's centre | of another class | window clipped at the border | median window side (px) |\n|---|---|---|---|---|---|")
    for cls in paths.CLASS_NAMES:
        t = [r for r in rows if r["official_split"] == "test" and r["class_name"] == cls]
        if not t:
            continue
        n = len(t)
        L.append(f"| {cls} | {n} | {100 * sum(r['n_others_inside'] > 0 for r in t) / n:.1f} % | {100 * sum(r['other_class_inside'] for r in t) / n:.1f} % | "
                 f"{100 * sum(r['clipped'] for r in t) / n:.1f} % | {np.median([r['window_side'] for r in t]):.0f} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="images per split (smoke check; writes to <root>_smoke)")
    ap.add_argument("--dry-run", action="store_true", help="plan the chips and print the flag table; write nothing")
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--floor", type=float, default=paths.WINDOW_FLOOR)
    ap.add_argument("--margin", type=float, default=paths.WINDOW_MARGIN)
    ap.add_argument("--size", type=int, default=paths.CHIP_SIZE)
    a = ap.parse_args()
    root = paths.CHIPS_ROOT.with_name(paths.CHIPS_ROOT.name + "_smoke") if a.limit else paths.CHIPS_ROOT
    meta_rows = {r["file_name"]: r for r in csv.DictReader(open(paths.METADATA_CSV, encoding="utf-8"))}

    t0 = time.time()
    jobs, rows = [], []
    for split in ["train", "val", "test"]:
        images = load_split(split)
        if a.limit:
            images = images[:a.limit]
        for im, anns in images:
            r = plan_image(split, im, anns, a.floor, a.margin)
            m = meta_rows.get(im["file_name"], {})
            for x in r:
                x["source"], x["group"] = m.get("source", "?"), m.get("group", "")
                x["idx"] = len(rows)
                rows.append(x)
            jobs.append((split, im["file_name"], r, a.size))
    print(f"planned {len(rows)} chips from {len(jobs)} images in {time.time() - t0:.0f} s", flush=True)
    if a.dry_run:
        meta = {"n_chips": len(rows), "n_images": len(jobs), "written": "dry run", "window_floor": a.floor, "window_margin": a.margin,
                "chip_size": a.size, "dataset_root": str(paths.SARDET100K_ROOT)}
        print(summary_text(rows, meta))
        return

    root.mkdir(parents=True, exist_ok=True)
    npy = root / "chips_u8.npy"
    arr = np.lib.format.open_memmap(npy, mode="w+", dtype=np.uint8, shape=(len(rows), a.size, a.size))
    pos, done, t1 = 0, 0, time.time()
    with Pool(a.workers) as pool:
        for chips in pool.imap(cut_image, jobs, chunksize=16):
            arr[pos:pos + len(chips)] = chips
            pos += len(chips); done += 1
            if done % 5000 == 0 or done == len(jobs):
                rate = done / (time.time() - t1)
                print(f"  {done}/{len(jobs)} images, {pos} chips, {rate:.0f} images/s, about {(len(jobs) - done) / rate / 60:.1f} min left", flush=True)
    assert pos == len(rows)
    arr.flush(); del arr

    with open(root / "chips.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "box_x": f"{r['box_x']:.1f}", "box_y": f"{r['box_y']:.1f}", "box_w": f"{r['box_w']:.1f}", "box_h": f"{r['box_h']:.1f}"})
    meta = {"written": datetime.now().isoformat(timespec="seconds"), "n_chips": len(rows), "n_images": len(jobs), "chip_size": a.size,
            "window_floor": a.floor, "window_margin": a.margin, "resize": "PIL bilinear", "padding": "edge value",
            "dataset_root": str(paths.SARDET100K_ROOT), "limit": a.limit, "columns": COLUMNS,
            "chips_csv_sha256": sha256_file(root / "chips.csv"), "chips_npy_sha256": sha256_file(npy),
            "class_names": paths.CLASS_NAMES, "minutes": round((time.time() - t0) / 60, 1)}
    (root / "chips_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (root / "summary.md").write_text(summary_text(rows, meta), encoding="utf-8")
    write_previews(np.load(npy, mmap_mode="r"), rows, root / "previews")
    print(f"\nwrote {len(rows)} chips ({npy.stat().st_size / 1e9:.2f} GB) in {meta['minutes']} min to {root}\n"
          f"  chips_u8.npy, chips.csv, chips_meta.json, summary.md, previews/<split>_<class>.png")
    print(summary_text(rows, meta))


if __name__ == "__main__":
    main()
