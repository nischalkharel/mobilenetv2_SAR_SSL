"""Read FAIR-CSAR-V1.0 SL annotations: one DOTA-style text file per slice, one object per line.

    x1 y1 x2 y2 x3 y3 x4 y4 class difficulty        (oriented box as 4 corners in slice pixels)

objects(split, slice) -> list of dicts: obj_id (line index), class_name, label, difficult (0/1), cx, cy (box centre),
long, short (side lengths of the oriented box), x0, y0, x1, y1 (axis-aligned extent of the 4 corners).
slice_name -> scene id: the name without its last two numeric fields (range and azimuth offsets).
"""
import math
import re

import fair_paths as fp

SLICE_RE = re.compile(r"^(?P<scene>.+)_(?P<r>\d{5})_(?P<a>\d{5})$")
LABEL = {c: i for i, c in enumerate(fp.CLASS_NAMES)}


def scene_of(slice_name: str) -> str:
    m = SLICE_RE.match(slice_name)
    return m.group("scene") if m else slice_name


def objects(split: str, slice_name: str) -> list:
    out = []
    for k, line in enumerate(fp.annotation_path(split, slice_name).read_text(encoding="utf-8").splitlines()):
        parts = line.split()
        if len(parts) < 9:
            continue
        pts = [(float(parts[i]), float(parts[i + 1])) for i in range(0, 8, 2)]
        if parts[8] not in LABEL:
            raise ValueError(f"unknown class {parts[8]!r} in {split}/{slice_name} line {k}")
        w = math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1])
        h = math.hypot(pts[1][0] - pts[2][0], pts[1][1] - pts[2][1])
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        out.append({"obj_id": k, "class_name": parts[8], "label": LABEL[parts[8]], "difficult": int(parts[9]) if len(parts) > 9 and parts[9].isdigit() else 0,
                    "cx": sum(xs) / 4, "cy": sum(ys) / 4, "long": max(w, h), "short": min(w, h),
                    "x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)})
    return out


def frame(split: str, slice_name: str) -> dict:
    """Where the slice sits in its scene (METAXmls patch block): r0 = range start, a0 = azimuth start, and the SLC
    units per PNG pixel on each axis (range 1.0; azimuth about 3.7, the SLC is oversampled in azimuth). Used to map
    annotations of overlapping slices onto one scene frame: slices overlap 50 % in both directions."""
    import xml.etree.ElementTree as ET
    r = ET.parse(fp.FAIR_CSAR_ROOT / split / "METAXmls" / f"{slice_name}.xml").getroot()
    a0, a1 = float(r.findtext("processInfo/patch/azimuthStart")), float(r.findtext("processInfo/patch/azimuthEnd"))
    r0, r1 = float(r.findtext("processInfo/patch/rangeStart")), float(r.findtext("processInfo/patch/rangeEnd"))
    w, h = int(r.findtext("size/width")), int(r.findtext("size/height"))
    return {"r0": r0, "a0": a0, "range_per_px": (r1 - r0) / w, "az_per_px": (a1 - a0) / h, "width": w, "height": h}


def scene_xy(fr: dict, x: float, y: float):
    """Slice pixel (x, y) -> scene coordinates in PNG-pixel units (range px, azimuth px)."""
    return fr["r0"] / fr["range_per_px"] + x, fr["a0"] / fr["az_per_px"] + y


def slice_names(split: str) -> list:
    """Every slice of a split, sorted (from the annotation folder, so no image is needed)."""
    return sorted(p.stem for p in (fp.FAIR_CSAR_ROOT / split / "Annotations").glob("*.txt"))
