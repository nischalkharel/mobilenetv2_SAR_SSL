"""Regions from DINOSAR's ratio-gradient content map, for a whole SSL pool, once, on CPU (no labels read).

    python GradientRegions/make_regions.py --pool sardet --diagnose --preview 12          # 94,034 scenes -> outputs/regions_sardet/regions.npz
    python GradientRegions/make_regions.py --pool fair_tiles --diagnose --preview 12      # 19,036 tiles  -> outputs/regions_fair_tiles/regions_fair_tiles.npz
    python GradientRegions/make_regions.py --pool sardet --smoke                          # the 48 smoke images -> regions_smoke.npz
    python GradientRegions/make_regions.py --pool sardet --limit 300 --diagnose           # a quick look, never the full file

Map: content_map.py (DINOSAR's recipe). Map -> regions: the TeacherRegions rule (top q % of the map, connected
components, min area, at most N boxes by map mass), so the three proposers differ only in the map. The file name is
the pool's own (regions.npz for sardet, regions_fair_tiles.npz for fair_tiles), so a training run selects this
proposer by pointing REGIONSSL_REGIONS_DIR (or the FAIR config's "regions_dir") at the output folder.
--diagnose measures box agreement against the annotations on up to 300 images (annotations are read only to measure)
and writes <stem>_box_agreement.json; --preview N draws N images with the brightness-rule and gradient regions.
"""
import argparse
import json
import os
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "RegionSSL"))
sys.path.insert(0, str(REPO / "SARDet"))
sys.path.insert(0, str(REPO / "TeacherRegions"))
sys.path.insert(0, str(HERE))
from rssl import pools, regions as R  # noqa: E402
from rssl.region_data import region_box_diagnostic  # noqa: E402
from shared.ssl_engine import SMOKE_IMAGES  # noqa: E402
from content_map import PARAMETERS, content_map  # noqa: E402
from teacher_saliency import preview, regions_from_map  # noqa: E402  (the shared map -> regions rule and the preview sheet)


def one(pool_name, top_q, min_area, max_regions, row):
    pool = pools.get(pool_name)
    m = content_map(pool.load_for_proposal(row))
    boxes, areas = regions_from_map(m, top_q, min_area, max_regions)
    return row["file_name"], boxes, areas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="sardet", choices=sorted(pools.POOLS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--top-q", type=float, default=2.0, help="top q %% of the map becomes region candidates (the TeacherRegions default)")
    ap.add_argument("--min-area", type=int, default=12)
    ap.add_argument("--max-regions", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out-dir", type=Path, help="default GradientRegions/outputs/regions_<pool>")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--preview", type=int, default=0)
    a = ap.parse_args()
    pool = pools.get(a.pool)
    rows = pool.rows()
    if a.smoke:
        rows = rows[:: max(1, len(rows) // SMOKE_IMAGES)][:SMOKE_IMAGES]
    elif a.limit:
        rows = rows[: a.limit]
    elif hasattr(pool, "all_rows"):
        rows = pool.all_rows()
    out_dir = a.out_dir or (HERE / "outputs" / f"regions_{pool.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pool.regions_file(a.smoke).stem + (f"_limit{a.limit}" if a.limit else "")
    out = out_dir / f"{stem}.npz"
    print(f"{len(rows)} {pool.name} images, {a.workers} workers, DINOSAR content map {PARAMETERS['scales']} / ds {PARAMETERS['downsample']}"
          f" at {pool.propose_size} px, top {a.top_q} %, min area {a.min_area}, max {a.max_regions} -> {out}", flush=True)
    t0 = time.time()
    per_image = {}
    with Pool(a.workers) as p:
        for k, (name, boxes, areas) in enumerate(p.imap_unordered(partial(one, pool.name, a.top_q, a.min_area, a.max_regions), rows, chunksize=16), 1):
            per_image[name] = (boxes, areas)
            if k % 2000 == 0 or k == len(rows):
                el = time.time() - t0
                print(f"  {k}/{len(rows)}  {el / 60:.1f} min, about {el / k * (len(rows) - k) / 60:.1f} min left", flush=True)
    extra = {"proposer": "dinosar_ratio_gradient_content", "proposer_parameters": {**PARAMETERS, "top_q_pct": a.top_q, "min_area": a.min_area, "max_regions": a.max_regions},
             "pool": pool.name, "pool_images": len(rows), "subset": "smoke" if a.smoke else (f"first {a.limit}" if a.limit else "all"),
             "seconds": round(time.time() - t0, 1), "workers": a.workers}
    summary = R.save(out, per_image, extra, input_size=pool.propose_size)
    print(f"  wrote {out}: {summary['regions_total']} regions, per image mean {summary['regions_per_image']['mean']:.1f}, extent quantiles {summary['extent_px_quantiles']}")
    if a.diagnose or a.preview:
        stores = {}
        felz = REPO / "RegionSSL" / "regions" / pool.regions_file(a.smoke).name       # the brightness-rule file of this pool
        if felz.exists():
            stores["brightness rule"] = R.RegionStore(felz, scale=pool.region_scale)
        stores["gradient content"] = R.RegionStore(out, scale=pool.region_scale)
        if a.diagnose:
            res = {name: region_box_diagnostic(rows, st, 12, n_images=min(300, len(rows)), pool=pool) for name, st in stores.items()}
            print("\nbox agreement on the same images (annotations used only to measure):")
            print(f"{'proposer':18} {'GT boxes with a region':>24} {'region centres in a box':>24} {'regions/image':>14} {'images w/o':>11}")
            for name, v in res.items():
                print(f"{name:18} {100 * v['gt_boxes_with_a_region']:23.1f} % {100 * v['region_centre_in_a_box']:23.1f} % {v['usable_regions_per_image']:14.1f} {100 * v['images_without_regions']:10.1f} %")
            (out_dir / f"{stem}_box_agreement.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        if a.preview:
            preview(pool, rows[: a.preview], stores, out_dir / f"{stem}_preview.png")
            print(f"preview: {out_dir / (stem + '_preview.png')}")


if __name__ == "__main__":
    main()
