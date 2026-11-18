"""Label-free region boxes for the SSL pool: how they are proposed, stored and looked up.

The proposer is `felz_grouped` from segment_lab.py (Felzenszwalb superpixels brighter than the image median,
nearby fragments grouped), run ONCE per pool image at the detector scale (shorter side 512, the scale at
which every SSL view is cut; see SARDet/shared/ssl_data.py). Only each region's bounding box is kept, because
the learning method cuts crops around regions and never needs the pixel mask.

Storage: one compressed file  <regions dir>/regions.npz  with
    names    (N,)   image file names, sorted
    offsets  (N+1,) regions of image i are rows offsets[i]:offsets[i+1]
    boxes    (M, 4) int16  x1, y1, x2, y2 at the 512 scale
    areas    (M,)   int32  region pixel count (at the 512 scale)
plus regions_summary.json (counts, size quantiles, proposer parameters, sha256, runtime) which IS committed.
The .npz is not committed (tens of MB); make_regions.py recreates it deterministically.

Where the files live: $REGIONSSL_REGIONS_DIR, default RegionSSL/regions/.
"""
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import REGIONSSL_ROOT  # noqa: F401  (sets sys.path)
import segment_lab  # noqa: E402  the lab's proposers, imported read-only

REGIONS_DIR = Path(os.environ.get("REGIONSSL_REGIONS_DIR", REGIONSSL_ROOT / "regions"))
PROPOSER_NAME = "felz_grouped"
INPUT_SIZE = 512


def proposer_parameters() -> dict:
    """The default arguments of the lab functions the proposer is built from (recorded in the summary)."""
    out = {}
    for fn in (segment_lab.regions_felz, segment_lab._superpixels_to_regions, segment_lab.group_nearby):
        out[fn.__name__] = {k: v.default for k, v in inspect.signature(fn).parameters.items() if v.default is not inspect.Parameter.empty}
    out["group_nearby"]["reach"] = 5   # regions_felz_grouped calls group_nearby(labels, reach=5)
    return out


def propose_boxes(img: np.ndarray, min_area: int = 12):
    """float image in [0, 1] -> (boxes int16 (K, 4) x1 y1 x2 y2, areas int32 (K,)), largest first."""
    labels = segment_lab.METHODS[PROPOSER_NAME](img)
    n = int(labels.max())
    if n == 0:
        return np.zeros((0, 4), np.int16), np.zeros((0,), np.int32)
    slices = ndimage.find_objects(labels)
    areas = ndimage.sum_labels(np.ones_like(labels, dtype=np.int32), labels, index=np.arange(1, n + 1)).astype(np.int32)
    boxes = np.array([[s[1].start, s[0].start, s[1].stop, s[0].stop] for s in slices], dtype=np.int32)
    keep = areas >= min_area
    boxes, areas = boxes[keep], areas[keep]
    order = np.argsort(-areas, kind="stable")
    return boxes[order].astype(np.int16), areas[order]


def regions_file(smoke: bool = False) -> Path:
    return REGIONS_DIR / ("regions_smoke.npz" if smoke else "regions.npz")


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path: Path, per_image: dict, extra: dict, input_size: int = INPUT_SIZE):
    """per_image: {file_name: (boxes, areas)} -> regions.npz + regions_summary.json next to it.
    input_size: the frame the boxes are in (the size the rule ran at; a pool's propose_size)."""
    names = sorted(per_image)
    offsets = np.zeros(len(names) + 1, dtype=np.int64)
    boxes, areas = [], []
    for i, n in enumerate(names):
        b, a = per_image[n]
        offsets[i + 1] = offsets[i] + len(b)
        boxes.append(b)
        areas.append(a)
    boxes = np.concatenate(boxes).astype(np.int16) if boxes else np.zeros((0, 4), np.int16)
    areas = np.concatenate(areas).astype(np.int32) if areas else np.zeros((0,), np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, names=np.array(names), offsets=offsets, boxes=boxes, areas=areas)
    os.replace(tmp, path)
    counts = np.diff(offsets)
    ext = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]).astype(np.float64) if len(boxes) else np.zeros(0)
    summary = {"file": path.name, "sha256": sha256_of_file(path), "proposer": PROPOSER_NAME, "input_size": input_size,
               "proposer_parameters": proposer_parameters(), "images": len(names),
               "images_with_regions": int((counts > 0).sum()), "regions_total": int(len(boxes)),
               "regions_per_image": {"mean": float(counts.mean()) if len(counts) else 0, "median": float(np.median(counts)) if len(counts) else 0,
                                     "max": int(counts.max()) if len(counts) else 0},
               "extent_px_quantiles": {q: float(np.quantile(ext, q / 100)) for q in (5, 25, 50, 75, 95)} if len(ext) else {},
               "written": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
    with open(path.with_name(path.stem + "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


class RegionStore:
    """Fast lookup of the region boxes of one image by file name (loaded once, kept in memory)."""

    def __init__(self, path: Path, scale: float = 1.0):
        """scale: multiply the stored boxes (proposal frame) into the frame the views are cut from (pool.region_scale)."""
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"region file missing: {self.path}  (run RegionSSL/make_regions.py first)")
        with np.load(self.path) as z:
            self.names = z["names"]
            self.offsets = z["offsets"]
            self.boxes = z["boxes"]
            self.areas = z["areas"]
        self.scale = float(scale)
        if self.scale != 1.0:
            self.boxes = np.round(self.boxes.astype(np.float64) * self.scale).astype(np.int16)
            self.areas = np.round(self.areas.astype(np.float64) * self.scale * self.scale).astype(np.int32)
        self.index = {n: i for i, n in enumerate(self.names.tolist())}
        self.sha256 = sha256_of_file(self.path)
        ext = np.maximum(self.boxes[:, 2] - self.boxes[:, 0], self.boxes[:, 3] - self.boxes[:, 1])
        self.extents = ext.astype(np.int32)            # pooled over all images: used by the random-regions control
        self.aspects = ((self.boxes[:, 3] - self.boxes[:, 1]).astype(np.float32) / np.maximum(1, self.boxes[:, 2] - self.boxes[:, 0]))

    def __contains__(self, file_name):
        return file_name in self.index

    def get(self, file_name):
        """(K, 4) int boxes of that image (empty if the image has none or is not in the file)."""
        i = self.index.get(file_name)
        if i is None:
            return np.zeros((0, 4), np.int16)
        return self.boxes[self.offsets[i]:self.offsets[i + 1]]

    def coverage(self, rows) -> dict:
        have = sum(r["file_name"] in self.index for r in rows)
        return {"file": str(self.path), "sha256": self.sha256, "images_in_file": int(len(self.names)),
                "pool_images_with_entry": have, "pool_images_missing": len(rows) - have}
