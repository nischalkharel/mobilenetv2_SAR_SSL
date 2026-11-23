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


def slice_names(split: str) -> list:
    """Every slice of a split, sorted (from the annotation folder, so no image is needed)."""
    return sorted(p.stem for p in (fp.FAIR_CSAR_ROOT / split / "Annotations").glob("*.txt"))
