"""Which PNGs a PC smoke check needs (the sample folder holds every annotation but few images).

    python FAIR_chips/smoke_list.py        -> FAIR_chips/data/pc_smoke_pngs.txt (relative paths under FAIR_CSAR_ROOT)

Then on CRC, from the extracted folder:  tar czf ../fair_smoke.tgz -T <repo>/FAIR_chips/data/pc_smoke_pngs.txt
and unpack the tar into the PC sample folder. The list holds: the 48 tiles-pool and 48 down-pool slices the engine's
smoke rule picks (rows[::len // 48][:48], see RegionSSL/rssl/region_engine.py), the split-1 1 % labeled slices and
the first 20 selection slices (fine-tune smoke), and the first 20 test slices (already in the sample).
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "RegionSSL")); sys.path.insert(0, str(HERE.parent / "SARDet"))
import fair_paths as fp  # noqa: E402
from rssl import pools  # noqa: E402
from rssl.region_engine import SMOKE_IMAGES  # noqa: E402


def smoke_rows(rows):
    return rows[:: max(1, len(rows) // SMOKE_IMAGES)][:SMOKE_IMAGES]


def main():
    need = set()
    for name in ("fair_tiles", "fair_down"):
        for r in smoke_rows(pools.get(name).rows()):
            need.add(("trainval", r["slice"]))
    for p, k in ((fp.labeled_csv(1, "1pct"), None), (fp.SELECT_CSV, 20)):
        with open(p, newline="", encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f)):
                if k is None or i < k:
                    need.add(("trainval", r["slice"]))
    with open(fp.SLICES_CSV, newline="", encoding="utf-8") as f:
        test = sorted(r["slice"] for r in csv.DictReader(f) if r["split"] == "test")[:20]
    need.update(("test", s) for s in test)
    out = fp.DATA_DIR / "pc_smoke_pngs.txt"
    with open(out, "w", newline="\n", encoding="utf-8") as f:          # LF endings: the list is fed to tar -T on Linux
        f.write("".join(f"{split}/PNGImages/{s}.png\n" for split, s in sorted(need)))
    print(f"{len(need)} PNGs listed in {out}")


if __name__ == "__main__":
    main()
