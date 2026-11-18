"""Propose label-free region boxes for every image of an SSL pool, once, offline (CPU only).

    python RegionSSL/make_regions.py                        # SARDet-100K: all 94,034 pool images -> regions/regions.npz (+ summary)
    python RegionSSL/make_regions.py --smoke                # the 48 smoke images only -> regions/regions_smoke.npz
    python RegionSSL/make_regions.py --limit 2000           # a subset (quick look) -> regions/regions_limit2000.npz, never the full file
    python RegionSSL/make_regions.py --pool atrnet          # ATRNet-STAR: the union of the three splits' SSL pools -> regions/regions_atrnet.npz
    python RegionSSL/make_regions.py --pool atrnet --smoke  # the 48 smoke chips of split 1 -> regions/regions_atrnet_smoke.npz

Rule: felz_grouped from segment_lab.py, run at the pool's proposal size (SARDet: shorter side 512, the detector scale;
ATRNet chips: native 128), see rssl/regions.py and rssl/pools.py. No labels are read. Deterministic (no randomness in
the rule), so the file can be recreated anywhere.
"""
import argparse
import os
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rssl import pools, regions as R  # noqa: E402
from shared.ssl_engine import SMOKE_IMAGES  # noqa: E402  (SARDet suite, read-only)


def one(pool_name, row):
    pool = pools.get(pool_name)
    im = pool.load_for_proposal(row)
    img = np.asarray(im, dtype=np.float32) / 255.0
    boxes, areas = R.propose_boxes(img)
    return row["file_name"], boxes, areas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="sardet", choices=sorted(pools.POOLS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=Path, help="output file (default: regions dir / regions[_<pool>][_smoke].npz)")
    args = ap.parse_args()
    pool = pools.get(args.pool)
    rows = pool.rows()
    if args.smoke:
        rows = rows[:: max(1, len(rows) // SMOKE_IMAGES)][:SMOKE_IMAGES]      # exactly the engine's smoke subset (of this split)
    elif args.limit:
        rows = rows[: args.limit]
    elif hasattr(pool, "all_rows"):
        rows = pool.all_rows()                                                 # one file serves every split
    full = pool.regions_file()
    out = args.out or (pool.regions_file(args.smoke) if not args.limit else full.with_name(f"{full.stem}_limit{args.limit}.npz"))
    print(f"{len(rows)} pool images ({pool.name}), {args.workers} workers, proposer {R.PROPOSER_NAME} at {pool.propose_size} px -> {out}", flush=True)
    t0 = time.time()
    per_image = {}
    with Pool(args.workers) as p:
        for k, (name, boxes, areas) in enumerate(p.imap_unordered(partial(one, pool.name), rows, chunksize=16), 1):
            per_image[name] = (boxes, areas)
            if k % 2000 == 0 or k == len(rows):
                el = time.time() - t0
                print(f"  {k}/{len(rows)}  {el / 60:.1f} min, about {el / k * (len(rows) - k) / 60:.1f} min left", flush=True)
    summary = R.save(out, per_image, {"pool": pool.name, "pool_images": len(rows),
                                      "subset": "smoke" if args.smoke else (f"first {args.limit}" if args.limit else "all"),
                                      "seconds": round(time.time() - t0, 1), "workers": args.workers}, input_size=pool.propose_size)
    print({k: v for k, v in summary.items() if k != "proposer_parameters"})


if __name__ == "__main__":
    main()
