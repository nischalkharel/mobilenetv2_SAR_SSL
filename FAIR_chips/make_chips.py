"""Cut one chip per annotated object of FAIR-CSAR SL into a single array file (DESIGN.md section 3).

    python FAIR_chips/make_chips.py                        # every slice of trainval and test (about 85,000 chips)
    python FAIR_chips/make_chips.py --available            # only slices whose PNG exists (PC sample): writes <root>_sample
    python FAIR_chips/make_chips.py --dry-run              # plan + flag table, nothing written

Rule: the SARDet_chips rule unchanged. Square window centred on the oriented box's centre, side =
max(WINDOW_FLOOR, WINDOW_MARGIN x longer box side) in native 1 m pixels, edge-padded where it leaves the slice,
resized to CHIP_SIZE x CHIP_SIZE, 8-bit grey. The pixel cutting (cut_image), previews and hashing are imported from
SARDet_chips/make_chips.py; only the planning (oriented boxes from text files) is new.
Writes under FAIR_CHIPS_ROOT: chips_u8.npy, chips.csv, chips_meta.json, summary.md, previews/<split>_<class>.png.
chips.csv columns: idx, official_split, slice, scene, obj_id, class_name, label, difficult, cx, cy, box_long, box_short,
box_x0, box_y0, box_x1, box_y1, window_side, clipped, n_others_inside, other_class_inside.
A chip belongs to the dataset split of its slice (trainval or test); the study's sets are drawn from slices.csv.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fair_paths as fp  # noqa: E402
import fair_annotations as fa  # noqa: E402
sys.path.insert(0, str(fp.SARDET_CHIPS_DIR))
from make_chips import sha256_file, write_previews  # noqa: E402  (SARDet_chips; cut_image is re-implemented below for PNG paths)

COLUMNS = ["idx", "official_split", "slice", "scene", "obj_id", "class_name", "label", "difficult", "cx", "cy", "box_long", "box_short",
           "box_x0", "box_y0", "box_x1", "box_y1", "window_side", "clipped", "n_others_inside", "other_class_inside"]


def plan_slice(split, name, floor, margin):
    obs = fa.objects(split, name)
    if not obs:
        return []
    cx = np.array([o["cx"] for o in obs]); cy = np.array([o["cy"] for o in obs]); cls = np.array([o["label"] for o in obs])
    rows = []
    for k, o in enumerate(obs):
        side = max(floor, margin * o["long"])
        half = side / 2
        inside = (np.abs(cx - cx[k]) < half) & (np.abs(cy - cy[k]) < half)
        inside[k] = False
        clipped = cx[k] - half < 0 or cy[k] - half < 0 or cx[k] + half > fp.SLICE_SIZE or cy[k] + half > fp.SLICE_SIZE
        rows.append({"official_split": split, "slice": name, "scene": fa.scene_of(name), "obj_id": o["obj_id"], "class_name": o["class_name"],
                     "label": o["label"], "difficult": o["difficult"], "cx": round(o["cx"], 1), "cy": round(o["cy"], 1),
                     "box_long": round(o["long"], 1), "box_short": round(o["short"], 1),
                     "box_x0": round(o["x0"], 1), "box_y0": round(o["y0"], 1), "box_x1": round(o["x1"], 1), "box_y1": round(o["y1"], 1),
                     "window_side": round(side, 1), "clipped": int(clipped),
                     "n_others_inside": int(inside.sum()), "other_class_inside": int((inside & (cls != cls[k])).any()), "_cx": cx[k], "_cy": cy[k]})
    return rows


def cut_slice(job):
    """Worker: one PNG -> its chips as a (n, S, S) uint8 array in row order (same arithmetic as SARDet_chips.cut_image)."""
    split, name, rows, size = job
    with Image.open(fp.png_path(split, name)) as im:
        a = np.asarray(im.convert("L"))
    h, w = a.shape
    out = np.empty((len(rows), size, size), dtype=np.uint8)
    for i, r in enumerate(rows):
        side = int(round(r["window_side"]))
        x0, y0 = int(round(r["_cx"] - side / 2)), int(round(r["_cy"] - side / 2))
        px0, py0 = max(0, -x0), max(0, -y0)
        px1, py1 = max(0, x0 + side - w), max(0, y0 + side - h)
        src = a if not (px0 or py0 or px1 or py1) else np.pad(a, ((py0, py1), (px0, px1)), mode="edge")
        win = src[y0 + py0: y0 + py0 + side, x0 + px0: x0 + px0 + side]
        out[i] = np.asarray(Image.fromarray(win).resize((size, size), Image.BILINEAR))
    return out


def summary_text(rows, meta):
    L = [f"# FAIR-CSAR SL chips: {meta['n_chips']} chips from {meta['n_images']} slices, written {meta['written']}\n",
         f"Rule: window side = max({meta['window_floor']}, {meta['window_margin']} x longer box side) native px, edge-padded, resized to {meta['chip_size']}. Source: {meta['dataset_root']}\n"]
    c = Counter((r["official_split"], r["class_name"]) for r in rows)
    L.append("## Chips per class and dataset split\n\n| class | trainval | test |\n|---|---|---|")
    for cls in fp.CLASS_NAMES:
        L.append(f"| {cls} | {c[('trainval', cls)]} | {c[('test', cls)]} |")
    L.append("\n## Flags on test chips (share of the class)\n\n| class | n | holds another object's centre | of another class | window clipped at the border | difficult | median window side (px) |\n|---|---|---|---|---|---|---|")
    for cls in fp.CLASS_NAMES:
        t = [r for r in rows if r["official_split"] == "test" and r["class_name"] == cls]
        if t:
            n = len(t)
            L.append(f"| {cls} | {n} | {100 * sum(r['n_others_inside'] > 0 for r in t) / n:.1f} % | {100 * sum(r['other_class_inside'] for r in t) / n:.1f} % | "
                     f"{100 * sum(r['clipped'] for r in t) / n:.1f} % | {100 * sum(r['difficult'] for r in t) / n:.1f} % | {np.median([r['window_side'] for r in t]):.0f} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--available", action="store_true", help="only slices whose PNG exists (PC sample); writes to <root>_sample")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--floor", type=float, default=fp.WINDOW_FLOOR)
    ap.add_argument("--margin", type=float, default=fp.WINDOW_MARGIN)
    ap.add_argument("--size", type=int, default=fp.CHIP_SIZE)
    a = ap.parse_args()
    root = fp.CHIPS_ROOT.with_name(fp.CHIPS_ROOT.name + "_sample") if a.available else fp.CHIPS_ROOT

    t0 = time.time()
    jobs, rows, skipped = [], [], 0
    for split in fp.SPLITS:
        for name in fa.slice_names(split):
            if a.available and not fp.png_path(split, name).exists():
                skipped += 1
                continue
            r = plan_slice(split, name, a.floor, a.margin)
            if not r:
                continue
            for x in r:
                x["idx"] = len(rows); rows.append(x)
            jobs.append((split, name, r, a.size))
    print(f"planned {len(rows)} chips from {len(jobs)} slices in {time.time() - t0:.0f} s" + (f" ({skipped} slices without a PNG skipped)" if a.available else ""), flush=True)
    meta = {"written": datetime.now().isoformat(timespec="seconds"), "n_chips": len(rows), "n_images": len(jobs), "chip_size": a.size,
            "window_floor": a.floor, "window_margin": a.margin, "resize": "PIL bilinear", "padding": "edge value",
            "dataset_root": str(fp.FAIR_CSAR_ROOT), "subset": "available PNGs only" if a.available else "all", "columns": COLUMNS,
            "class_names": fp.CLASS_NAMES}
    if a.dry_run:
        print(summary_text(rows, {**meta, "written": "dry run"})); return
    if not jobs:
        raise SystemExit("no slice with a PNG found")

    root.mkdir(parents=True, exist_ok=True)
    npy = root / "chips_u8.npy"
    arr = np.lib.format.open_memmap(npy, mode="w+", dtype=np.uint8, shape=(len(rows), a.size, a.size))
    pos, done, t1 = 0, 0, time.time()
    with Pool(a.workers) as pool:
        for chips in pool.imap(cut_slice, jobs, chunksize=4):
            arr[pos:pos + len(chips)] = chips
            pos += len(chips); done += 1
            if done % 1000 == 0 or done == len(jobs):
                rate = done / (time.time() - t1)
                print(f"  {done}/{len(jobs)} slices, {pos} chips, {rate:.1f} slices/s, about {(len(jobs) - done) / rate / 60:.1f} min left", flush=True)
    assert pos == len(rows)
    arr.flush(); del arr
    with open(root / "chips.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    meta.update(chips_csv_sha256=sha256_file(root / "chips.csv"), chips_npy_sha256=sha256_file(npy), minutes=round((time.time() - t0) / 60, 1))
    (root / "chips_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (root / "summary.md").write_text(summary_text(rows, meta), encoding="utf-8")
    write_previews(np.load(npy, mmap_mode="r"), rows, root / "previews")
    print(f"\nwrote {len(rows)} chips ({npy.stat().st_size / 1e9:.2f} GB) in {meta['minutes']} min to {root}")
    print(summary_text(rows, meta))


if __name__ == "__main__":
    main()
