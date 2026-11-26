"""How often do plain DINO's own random views land on bright content, per SSL pool?  (no labels needed)

    python RegionSSL/pool_hit_rate.py --pool sardet                 # 94,034 scenes, the brightness-rule region file
    python RegionSSL/pool_hit_rate.py --pool fair_tiles             # 19,036 tiles (needs regions_fair_tiles.npz, built on CRC)
    python RegionSSL/pool_hit_rate.py --pool sardet --limit 3000    # a quick look
    python RegionSSL/pool_hit_rate.py --pool sardet --gt 1000       # + the annotation check on 1,000 sampled images (labels read only to measure)

Why (2026-11-26, tracker c13): plain DINO trained on SARDet-100K scenes transfers to ATRNet chips WORSE than no SSL,
plain DINO trained on FAIR-CSAR tiles transfers BETTER. The suspected reason is what the random views see: DINO cuts
its global (224 px x zoom) and local (96 px x zoom) windows at random places on the 512-px frame, so on a pool of
mostly open water a view is mostly background. This script samples exactly those windows (same sampler, same sizes,
same zoom range as every SSL run: shared/ssl_data.random_window with ssl_common.json) and measures, against the
label-free bright regions the RegionDINO method itself uses (extent >= min_object_px, as in the method):

    per image   bright-area fraction (union of usable region boxes / frame area); share of images with no usable
                region; share of near-empty images (bright area < 1 % and < 0.2 %)
    per window  share of windows that contain at least one usable region centre; share whose area is < 1 % bright
                ("background-only view"); mean bright fraction inside the window
    per source  the same, by the pool's source dataset when the manifest has one (SARDet-100K: nine sources)

--gt N adds the labelled view of the same question on N sampled images: share of windows containing an annotated
object centre (the number quoted as "about 30 % of global views contain an object" for SARDet). Writes
RegionSSL/outputs/pool_hit_rate_<pool>[_limitN].json and prints the table.
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "SARDet"))
from rssl import pools  # noqa: E402
from rssl.regions import RegionStore  # noqa: E402
from shared.ssl_data import load_common_config, random_window  # noqa: E402  (read-only: the method's own sampler)

MIN_OBJECT_PX = 12          # RegionSSL/RegionDINO_object/config_ssl.json "min_object_px": the method's usable-region filter
NEAR_EMPTY = (0.01, 0.002)  # bright-area fractions below which an image counts as near-empty
BG_VIEW = 0.01              # a window with less than this bright fraction is a "background-only view"


def frame_size(pool, row):
    """(W, H) of the frame the views are cut from: the detector_512 rule for scenes, the pool's input size otherwise."""
    if "width" in row and "height" in row:
        w, h = int(row["width"]), int(row["height"])
        s = pool.input_size / max(w, h)
        return int(round(w * s)), int(round(h * s))
    return pool.input_size, pool.input_size


class Acc:
    """Running sums for one group of images (the whole pool, or one source)."""

    def __init__(self):
        self.images = 0
        self.no_region = 0
        self.near_empty = [0, 0]
        self.bright = []
        self.win = {k: {"n": 0, "centre_hit": 0, "bg_only": 0, "cover": 0.0, "gt_hit": 0, "gt_n": 0} for k in ("global", "local")}
        self.gt_images = 0
        self.gt_boxes = 0
        self.gt_none = 0

    def summary(self):
        b = np.array(self.bright) if self.bright else np.zeros(1)
        out = {"images": self.images, "images_without_usable_region": self.no_region / max(self.images, 1),
               "bright_area_fraction_mean": float(b.mean()), "bright_area_fraction_median": float(np.median(b)),
               f"near_empty_below_{NEAR_EMPTY[0]:g}": self.near_empty[0] / max(self.images, 1),
               f"near_empty_below_{NEAR_EMPTY[1]:g}": self.near_empty[1] / max(self.images, 1), "windows": {}}
        for k, w in self.win.items():
            d = {"n": w["n"], "with_a_region_centre": w["centre_hit"] / max(w["n"], 1), "background_only": w["bg_only"] / max(w["n"], 1),
                 "bright_fraction_mean": w["cover"] / max(w["n"], 1)}
            if w["gt_n"]:
                d["with_an_annotated_object_centre"] = w["gt_hit"] / w["gt_n"]
                d["n_gt_windows"] = w["gt_n"]
            out["windows"][k] = d
        if self.gt_images:
            out["gt"] = {"images": self.gt_images, "objects_per_image": self.gt_boxes / self.gt_images, "images_without_object": self.gt_none / self.gt_images}
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="sardet", choices=sorted(pools.POOLS))
    ap.add_argument("--limit", type=int, default=0, help="first N pool images only (quick look)")
    ap.add_argument("--views", type=int, default=4, help="windows per image and size (the SSL recipe uses 2 global + several local per image)")
    ap.add_argument("--gt", type=int, default=0, help="also read the annotations of N randomly sampled images (diagnostic)")
    ap.add_argument("--min-object-px", type=int, default=MIN_OBJECT_PX)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="use the pool's smoke region file and images")
    ap.add_argument("--out-dir", default=str(HERE / "outputs"))
    a = ap.parse_args()

    pool = pools.get(a.pool)
    rows = pool.rows()
    store = RegionStore(pool.regions_file(a.smoke), pool.region_scale)
    if a.smoke:
        rows = [r for r in rows if r["file_name"] in store]      # the smoke region file covers the pool's smoke images, whichever they are
    if a.limit:
        rows = rows[:a.limit]
    views = load_common_config()["views"]
    sizes = {"global": views["global_size"], "local": views["local_size"]}
    rng = random.Random(a.seed)
    gt_rows = set(id(r) for r in (rng.sample(rows, min(a.gt, len(rows))) if a.gt else []))

    total, by_source = Acc(), defaultdict(Acc)
    t0 = time.time()
    for i, r in enumerate(rows):
        W, H = frame_size(pool, r)
        b = store.get(r["file_name"]).astype(np.int32)
        if len(b):
            ext = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
            b = b[ext >= a.min_object_px]
        mask = np.zeros((H, W), np.uint8)
        for x1, y1, x2, y2 in b:
            mask[max(y1, 0):min(y2, H), max(x1, 0):min(x2, W)] = 1
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        bright = float(mask.mean())
        gt = pool.gt_boxes(r) if id(r) in gt_rows else None
        if gt is not None:
            gcx, gcy = (gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2

        accs = [total, by_source[r["source"]]] if "source" in r else [total]
        for acc in accs:
            acc.images += 1
            acc.no_region += int(len(b) == 0)
            acc.bright.append(bright)
            acc.near_empty[0] += int(bright < NEAR_EMPTY[0])
            acc.near_empty[1] += int(bright < NEAR_EMPTY[1])
            if gt is not None:
                acc.gt_images += 1
                acc.gt_boxes += len(gt)
                acc.gt_none += int(len(gt) == 0)
        for key, out in sizes.items():
            for _ in range(a.views):
                x0, y0, side = random_window(W, H, out, views["zoom_range"], rng)
                cover = float(mask[y0:y0 + side, x0:x0 + side].mean())
                hit = bool(((cx >= x0) & (cx < x0 + side) & (cy >= y0) & (cy < y0 + side)).any()) if len(b) else False
                ghit = bool(((gcx >= x0) & (gcx < x0 + side) & (gcy >= y0) & (gcy < y0 + side)).any()) if gt is not None and len(gt) else False
                for acc in accs:
                    w = acc.win[key]
                    w["n"] += 1
                    w["centre_hit"] += int(hit)
                    w["bg_only"] += int(cover < BG_VIEW)
                    w["cover"] += cover
                    if gt is not None:
                        w["gt_n"] += 1
                        w["gt_hit"] += int(ghit)
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(rows)} images, {time.time() - t0:.0f} s", flush=True)

    result = {"pool": a.pool, "pool_images": len(pool.rows()), "images_used": len(rows), "smoke": a.smoke, "limit": a.limit,
              "regions_file": store.coverage(rows), "min_object_px": a.min_object_px, "views_per_image_and_size": a.views,
              "view_recipe": {"global_size": sizes["global"], "local_size": sizes["local"], "zoom_range": views["zoom_range"],
                              "frame": "detector_512 (longer side 512)" if "width" in rows[0] else f"{pool.input_size} px"},
              "definitions": {"usable region": f"brightness-rule region with extent >= {a.min_object_px} px (the method's filter)",
                              "bright_area_fraction": "union of usable region boxes / frame area",
                              "background_only": f"window with < {BG_VIEW:g} of its area bright",
                              "near_empty": "image with bright_area_fraction below the threshold in the key"},
              "seed": a.seed, "seconds": round(time.time() - t0, 1), "all": total.summary(),
              "by_source": {k: v.summary() for k, v in sorted(by_source.items())}}
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pool_hit_rate_{a.pool}" + ("_smoke" if a.smoke else "") + (f"_limit{a.limit}" if a.limit else "")
    out = out_dir / f"{stem}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    def line(name, s):
        g, l = s["windows"]["global"], s["windows"]["local"]
        gt = f" | gt: global {g['with_an_annotated_object_centre']:.1%} local {l['with_an_annotated_object_centre']:.1%}" if "with_an_annotated_object_centre" in g else ""
        return (f"| {name} | {s['images']} | {s['bright_area_fraction_median']:.1%} / {s['bright_area_fraction_mean']:.1%} | "
                f"{s[f'near_empty_below_{NEAR_EMPTY[0]:g}']:.1%} | {g['with_a_region_centre']:.1%} / {g['background_only']:.1%} | "
                f"{l['with_a_region_centre']:.1%} / {l['background_only']:.1%} |{gt}")

    print(f"\n{a.pool}: {len(rows)} images, {a.views} windows per image and size, usable region = extent >= {a.min_object_px} px")
    print("| group | images | bright area median / mean | near-empty (< 1 %) | global view: has a region / background-only | local view: has a region / background-only |")
    print("|---|---|---|---|---|---|")
    print(line("all", result["all"]))
    for k, v in result["by_source"].items():
        print(line(k, v))
    print(f"\nwrote {out}  ({result['seconds']} s)")


if __name__ == "__main__":
    main()
