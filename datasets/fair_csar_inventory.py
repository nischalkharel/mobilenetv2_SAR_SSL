"""Inventory of FAIR-CSAR-V1.0 SL from its annotation files alone (no images needed).

    python datasets/fair_csar_inventory.py --root <folder with test/ and trainval/>
    python datasets/fair_csar_inventory.py            # root from FAIR_CSAR_ROOT

The root holds <split>/Annotations/<slice>.txt (DOTA style: x1 y1 x2 y2 x3 y3 x4 y4 class difficulty, one object
per line) and <split>/METAXmls/<slice>.xml (slice size, major category per object, sensor parameters). Slices are
named <scene>_<rangeStart>_<azimuthStart>; the scene id is everything before the last two numeric fields.

Writes datasets/outputs/fair_csar_inventory.md (tables) and .csv (one row per object) next to this script.
Nothing is hard-coded: the root comes from --root or the FAIR_CSAR_ROOT environment variable.
"""
import argparse
import csv
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

SLICE_RE = re.compile(r"^(?P<scene>.+)_(?P<r>\d{5})_(?P<a>\d{5})$")
SIZE_BINS = [0, 8, 16, 32, 64, 128, 256, 10**9]     # long-side bins in slice pixels


def side(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def read_split(root: Path, split: str):
    ann_dir, xml_dir = root / split / "Annotations", root / split / "METAXmls"
    objects, slices = [], []
    major_of = {}
    for txt in sorted(ann_dir.glob("*.txt")):
        stem = txt.stem
        m = SLICE_RE.match(stem)
        scene = m.group("scene") if m else stem
        n = 0
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 9:
                continue
            pts = [(float(parts[i]), float(parts[i + 1])) for i in range(0, 8, 2)]
            w, h = side(pts[0], pts[1]), side(pts[1], pts[2])
            cx = sum(p[0] for p in pts) / 4
            cy = sum(p[1] for p in pts) / 4
            objects.append(dict(split=split, scene=scene, slice=stem, cls=parts[8], difficult=parts[9] if len(parts) > 9 else "",
                                cx=round(cx, 1), cy=round(cy, 1), long=round(max(w, h), 1), short=round(min(w, h), 1)))
            n += 1
        slices.append(dict(split=split, scene=scene, slice=stem, n_obj=n))
        xmlp = xml_dir / f"{stem}.xml"
        if xmlp.exists() and n:
            try:
                tree = ET.parse(xmlp)
                for ob in tree.getroot().iter("object"):
                    maj = ob.findtext("category/majorCategory"); sub = ob.findtext("category/subCategory")
                    if maj and sub:
                        major_of.setdefault(sub, maj)
            except ET.ParseError:
                pass
    return objects, slices, major_of


def slice_size(root: Path, split: str):
    for xmlp in sorted((root / split / "METAXmls").glob("*.xml"))[:1]:
        r = ET.parse(xmlp).getroot()
        return r.findtext("size/width"), r.findtext("size/height"), r.findtext(".//nominalResolution"), r.findtext(".//imgingMode")
    return None, None, None, None


def pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals); k = (len(s) - 1) * q / 100
    f, c = math.floor(k), math.ceil(k)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=os.environ.get("FAIR_CSAR_ROOT"))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs")
    a = ap.parse_args()
    if not a.root:
        sys.exit("give --root or set FAIR_CSAR_ROOT")
    splits = [s for s in ("trainval", "test") if (a.root / s / "Annotations").is_dir()]
    if not splits:
        sys.exit(f"no <split>/Annotations under {a.root}")
    a.out.mkdir(parents=True, exist_ok=True)

    objs, slcs, major = [], [], {}
    for s in splits:
        o, sl, mj = read_split(a.root, s)
        objs += o; slcs += sl; major.update(mj)
    L = [f"# FAIR-CSAR-V1.0 SL inventory (from annotations under `{a.root}`)", ""]
    w, h, res, mode = slice_size(a.root, splits[0])
    L.append(f"Slice size {w} x {h} px (METAXmls), mode {mode}, nominal resolution {res} m. One DOTA-style txt per slice, "
             f"one line per object: 8 polygon coordinates, class, difficulty flag.")
    L.append("")

    # ---- per split ----
    L += ["## Per split", "", "| split | scenes | slices | slices with objects | objects | objects / slice (median, p95, max) | difficult flag = 1 |", "|---|---|---|---|---|---|---|"]
    for s in splits:
        sl = [x for x in slcs if x["split"] == s]; ob = [x for x in objs if x["split"] == s]
        n = [x["n_obj"] for x in sl]; nz = [v for v in n if v]
        L.append(f"| {s} | {len({x['scene'] for x in sl})} | {len(sl)} | {len(nz)} ({100*len(nz)/max(1,len(sl)):.0f} %) | {len(ob)} | "
                 f"{pct(nz,50):.0f}, {pct(nz,95):.0f}, {max(nz) if nz else 0} | {sum(1 for x in ob if x['difficult']=='1')} |")
    L.append("")

    # ---- per class ----
    cls = sorted({x["cls"] for x in objs}, key=lambda c: (major.get(c, "?"), c))
    L += ["## Per class (object counts, and the long side of the oriented box in slice pixels)", "",
          "| major | class | " + " | ".join(splits) + " | total | 1 % of trainval | long side median | p5 | p95 | short side median |",
          "|---|---|" + "---|" * (len(splits) + 6)]
    for c in cls:
        ob = [x for x in objs if x["cls"] == c]
        per = [sum(1 for x in ob if x["split"] == s) for s in splits]
        tv = sum(1 for x in ob if x["split"] == "trainval")
        lg = [x["long"] for x in ob]; sh = [x["short"] for x in ob]
        L.append(f"| {major.get(c,'?')} | {c} | " + " | ".join(str(v) for v in per) + f" | {len(ob)} | {tv*0.01:.1f} | "
                 f"{pct(lg,50):.0f} | {pct(lg,5):.0f} | {pct(lg,95):.0f} | {pct(sh,50):.0f} |")
    L.append("")

    # ---- size histogram ----
    L += ["## Object long side, all classes (slice pixels)", "", "| bin | objects | share |", "|---|---|---|"]
    lg = [x["long"] for x in objs]
    for lo, hi in zip(SIZE_BINS[:-1], SIZE_BINS[1:]):
        k = sum(1 for v in lg if lo <= v < hi)
        L.append(f"| {lo} to {hi if hi < 10**8 else 'inf'} | {k} | {100*k/max(1,len(lg)):.1f} % |")
    L.append("")
    L.append(f"All objects: long side median {pct(lg,50):.0f} px, p5 {pct(lg,5):.0f}, p95 {pct(lg,95):.0f}, max {max(lg):.0f}; "
             f"short side median {pct([x['short'] for x in objs],50):.0f} px.")
    L.append("")

    # ---- scenes ----
    L += ["## Scenes (slices and objects per scene, largest first)", "", "| split | scene | slices | objects | classes |", "|---|---|---|---|---|"]
    per_scene = defaultdict(lambda: dict(slices=0, objects=0, classes=Counter(), split=""))
    for x in slcs:
        per_scene[x["scene"]]["slices"] += 1; per_scene[x["scene"]]["split"] = x["split"]
    for x in objs:
        per_scene[x["scene"]]["objects"] += 1; per_scene[x["scene"]]["classes"][x["cls"]] += 1
    for sc, d in sorted(per_scene.items(), key=lambda kv: (-kv[1]["objects"])):
        top = ", ".join(f"{c} {n}" for c, n in d["classes"].most_common(3))
        L.append(f"| {d['split']} | {sc} | {d['slices']} | {d['objects']} | {top} |")
    L.append("")

    # ---- class overlap between splits ----
    tv = {x["cls"] for x in objs if x["split"] == "trainval"}; te = {x["cls"] for x in objs if x["split"] == "test"}
    L.append(f"Classes in trainval only: {sorted(tv - te) or 'none'}; in test only: {sorted(te - tv) or 'none'}.")

    md = a.out / "fair_csar_inventory.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(a.out / "fair_csar_inventory.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(objs[0].keys())); wr.writeheader(); wr.writerows(objs)
    print("\n".join(L[:40]))
    print(f"\nwrote {md} and the per-object csv ({len(objs)} objects)")


if __name__ == "__main__":
    main()
