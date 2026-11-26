"""Fixed subsets of the SARDet-100K SSL pool, made once and committed (no labels read).

    python PoolComposition/make_pool_lists.py            # writes data/scene_bright_area.csv, data/pool_<subset>.csv, data/counts.json

Why (2026-11-26, tracker c13): plain DINO on the whole SARDet-100K pool transfers to ATRNet chips worse than no SSL,
plain DINO on FAIR-CSAR tiles transfers better, and RegionSSL/pool_hit_rate.py showed the difference in what the
random views see: on SARDet a quarter of the global views and 40 % of the local views hold no bright content at all,
almost all of it on the ship-at-sea sources. This script fixes four subsets of the SAME pool so plain DINO can be
trained on each with matched steps and tested on ATRNet:

    land     the sources whose scenes are mostly land: OGSOD, MSAR, SAR-AIRcraft, SIVED, SADD        (metadata: source name)
    sea      the ship datasets, mostly open water: Shipdataset, HRSID, SSDD, AIR_SARShip             (metadata: source name)
    dense    the half of the pool with the larger bright-area fraction (label-free, from the region file)
    sparse   the other half
    random   a random half of the pool (seed 0): the size control, same image count and repetition as dense / sparse
             but the whole pool's content mix (reviewer item, 2026-11-26)

bright-area fraction = share of the 512-px frame covered by the brightness-rule regions the RegionDINO method itself
uses (extent >= 12 px), from RegionSSL/regions/regions.npz. land / sea read only the pool manifest's source column;
dense / sparse read only the region file. Matched steps: the full pool runs 100 epochs x (94,034 // 64) = 146,900
steps; each subset gets round(146,900 / steps_per_epoch) epochs, and the warmup (10 epochs = 14,690 steps on the full
pool) is matched in steps the same way (both recorded in counts.json and checked by train_ssl.py).
"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "RegionSSL"))
sys.path.insert(0, str(REPO / "SARDet"))
from rssl import pools  # noqa: E402
from rssl.regions import RegionStore  # noqa: E402
from shared.ssl_data import load_common_config  # noqa: E402

DATA = HERE / "data"
LAND = ("OGSOD", "MSAR", "SAR-AIRcraft", "SIVED", "SADD")
SEA = ("Shipdataset", "HRSID", "SSDD", "AIR_SARShip")
MIN_OBJECT_PX = 12          # the method's usable-region filter (RegionSSL/RegionDINO_object/config_ssl.json)
FULL_EPOCHS = 100           # the full-pool DINO run (shared/ssl_common.json)
FULL_WARMUP_EPOCHS = 10     # optim.warmup_epochs of DINO_SARDet/config_ssl.json
RANDOM_SEED = 0             # the random half


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    pool = pools.get("sardet")
    rows = pool.rows()
    store = RegionStore(pool.regions_file(), pool.region_scale)
    batch = load_common_config()["batch_size"]
    target_steps = (len(rows) // batch) * FULL_EPOCHS
    target_warmup = (len(rows) // batch) * FULL_WARMUP_EPOCHS
    t0 = time.time()
    bright, usable = [], []
    for i, r in enumerate(rows):
        w, h = int(r["width"]), int(r["height"])
        s = pool.input_size / max(w, h)
        W, H = int(round(w * s)), int(round(h * s))
        b = store.get(r["file_name"]).astype(np.int32)
        if len(b):
            ext = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
            b = b[ext >= MIN_OBJECT_PX]
        mask = np.zeros((H, W), np.uint8)
        for x1, y1, x2, y2 in b:
            mask[max(y1, 0):min(y2, H), max(x1, 0):min(x2, W)] = 1
        bright.append(float(mask.mean()))
        usable.append(len(b))
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1}/{len(rows)}  {time.time() - t0:.0f} s", flush=True)
    bright = np.array(bright)
    median = float(np.median(bright))

    DATA.mkdir(exist_ok=True)
    with open(DATA / "scene_bright_area.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "source", "usable_regions", "bright_area_fraction"])
        for r, u, bf in zip(rows, usable, bright):
            w.writerow([r["file_name"], r["source"], u, f"{bf:.4f}"])

    subsets = {"land": [r for r in rows if r["source"] in LAND],
               "sea": [r for r in rows if r["source"] in SEA],
               "dense": [r for r, bf in zip(rows, bright) if bf > median],
               "sparse": [r for r, bf in zip(rows, bright) if bf <= median]}
    half = np.random.default_rng(RANDOM_SEED).permutation(len(rows))[: len(rows) // 2]
    subsets["random"] = [rows[i] for i in sorted(half)]
    assert len(subsets["land"]) + len(subsets["sea"]) == len(rows), "every source must be land or sea"
    counts = {"pool_images": len(rows), "batch_size": batch, "full_pool_steps_per_epoch": len(rows) // batch, "full_pool_epochs": FULL_EPOCHS,
              "target_steps": target_steps, "full_pool_warmup_epochs": FULL_WARMUP_EPOCHS, "target_warmup_steps": target_warmup,
              "random_seed": RANDOM_SEED, "min_object_px": MIN_OBJECT_PX, "bright_area_median": round(median, 4),
              "region_file": {"path": str(store.path.relative_to(REPO)).replace("\\", "/"), "sha256": store.sha256}, "land_sources": list(LAND), "sea_sources": list(SEA),
              "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "subsets": {}}
    for name, sub in subsets.items():
        names = sorted(r["file_name"] for r in sub)
        out = DATA / f"pool_{name}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file_name"])
            w.writerows([n] for n in names)
        spe = len(sub) // batch
        epochs = int(round(target_steps / spe))
        warmup = int(round(target_warmup / spe))
        chosen = set(names)
        idx = [i for i, r in enumerate(rows) if r["file_name"] in chosen]
        by_source = {}
        for r in sub:
            by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        counts["subsets"][name] = {"images": len(sub), "steps_per_epoch": spe, "matched_epochs": epochs, "total_steps": epochs * spe,
                                   "matched_warmup_epochs": warmup, "warmup_steps": warmup * spe,
                                   "bright_area_fraction_mean": round(float(bright[idx].mean()), 4), "bright_area_fraction_median": round(float(np.median(bright[idx])), 4),
                                   "images_without_usable_region": round(float((np.array(usable)[idx] == 0).mean()), 4),
                                   "by_source": dict(sorted(by_source.items(), key=lambda kv: -kv[1])), "list_csv": out.name, "list_sha256": sha256(out)}
    (DATA / "counts.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")

    print(f"\n{len(rows)} pool scenes; bright-area median {median:.1%}; target {target_steps:,} steps ({time.time() - t0:.0f} s)")
    print("| subset | images | steps / epoch | matched epochs (warmup) | bright area median / mean | without usable region | sources |")
    print("|---|---|---|---|---|---|---|")
    for name, c in counts["subsets"].items():
        src = ", ".join(f"{k} {v:,}" for k, v in c["by_source"].items())
        print(f"| {name} | {c['images']:,} | {c['steps_per_epoch']} | {c['matched_epochs']} ({c['total_steps']:,} steps; warmup {c['matched_warmup_epochs']}) | "
              f"{c['bright_area_fraction_median']:.1%} / {c['bright_area_fraction_mean']:.1%} | {c['images_without_usable_region']:.1%} | {src} |")
    print(f"wrote {DATA}")


if __name__ == "__main__":
    main()
