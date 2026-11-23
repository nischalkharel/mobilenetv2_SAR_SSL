"""The unlabeled image pools the RegionSSL engine can train on, behind one small interface.

    pool = pools.get("sardet")   # SARDet-100K scenes (the suite's pool; what every XA row used)
    pool = pools.get("atrnet")   # ATRNet-STAR SOC-40 train chips (the split's SSL pool; what A1 used)
    pool = pools.get("fair_tiles")  # FAIR-CSAR SL trainval slices cut 2 x 2 into native 512-px tiles (FAIR_chips study, main arm)
    pool = pools.get("fair_down")   # the same slices downsampled 1024 -> 512 (FAIR_chips study, resolution arm)

A pool gives: rows() (one dict per image with at least "file_name"), load(row) -> the gray PIL image at the working
size the views are cut from (input_size), load_for_proposal(row) -> the gray image the region rule runs on
(propose_size; boxes are scaled by region_scale when the store is read), regions_file(smoke), manifest(rows) for the
run record, gt_boxes(row) for the label diagnostic ONLY, and view_coverage(rows) (SARDet only). No labels are read
by anything that feeds training: the ATRNet rows keep only the image path.

Working sizes. SARDet scenes: shorter side 512 (the suite's detector scale), regions proposed at the same scale.
ATRNet chips: 128 x 128 native; the region rule runs at the native size (fewer, larger regions than at 224; see
RegionSSL/chip_lab.py) and the views are cut from the chip resized to 224, the input size of every ATRNet fine-tune
and of A1's global crops; boxes are scaled by 224 / 128.
"""
import csv
import hashlib
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
ATR_DINO_DIR = REPO / "Experiments" / "mobileNetv2_DINO"       # the A1 experiment: paths.py and data.py (read-only)
REGIONS_DIR = Path(os.environ.get("REGIONSSL_REGIONS_DIR", Path(__file__).resolve().parents[1] / "regions"))   # same variable as rssl/regions.py


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class SARDetPool:
    name = "sardet"
    dataset = "SARDet-100K (official train + val scenes, the suite's SSL pool)"
    input_size = 512
    propose_size = 512
    region_scale = 1.0

    def rows(self):
        from shared.ssl_data import load_pool_rows
        return load_pool_rows()

    def csv_path(self):
        from shared import manifests
        return manifests.ssl_pool_csv()

    def manifest(self, rows):
        from shared import manifests
        audit = manifests.quick_audit()
        return {"pool": self.name, "ssl_pool_csv": str(manifests.ssl_pool_csv()), "csv_sha256": audit["manifest_sha256"]["ssl_pool.csv"],
                "split_audit": audit, "columns_available_to_the_method": manifests.MANIFEST_COLUMNS}

    def load(self, row):
        from shared.ssl_data import load_resized_gray
        return load_resized_gray(row["official_split"], row["file_name"], self.input_size)

    def load_for_proposal(self, row):
        return self.load(row)

    def gt_boxes(self, row):
        """GT boxes (x1, y1, x2, y2) at the working scale; diagnostic only."""
        from shared.coco_data import boxes_and_labels
        gt, _, _ = boxes_and_labels(row["official_split"], row["file_name"])
        w, h = row["width"], row["height"]
        s = min(self.input_size / min(w, h), self.input_size / max(w, h))
        return gt.numpy() * s

    def view_coverage(self, rows):
        from shared.ssl_data import view_box_coverage
        return view_box_coverage(rows, n_images=min(300, len(rows)))

    def regions_file(self, smoke=False):
        return REGIONS_DIR / ("regions_smoke.npz" if smoke else "regions.npz")


class ATRNetPool:
    name = "atrnet"
    dataset = "ATRNet-STAR SOC-40 train chips (the split's ssl_training_data.csv, as used by A1)"
    native_size = 128
    input_size = 224
    propose_size = 128
    region_scale = input_size / propose_size
    _BOX_RE = {k: re.compile(rf"<{k}>(-?\d+)</{k}>") for k in ("xmin", "ymin", "xmax", "ymax")}

    def _mods(self):
        if str(ATR_DINO_DIR) not in sys.path:
            sys.path.append(str(ATR_DINO_DIR))           # appended, never inserted: nothing of the suite is shadowed
        import paths as atr_paths  # noqa: E402
        import data as atr_data  # noqa: E402
        return atr_paths, atr_data

    def split_csv(self, split=None):
        atr_paths, _ = self._mods()
        if split is None:
            return atr_paths.SPLITS_DIR / "ssl_training_data.csv"
        base = atr_paths.REPO_ROOT / "preprocessing_dataset"
        return (base if split == 1 else base / f"split{split}") / "ssl_training_data.csv"

    def rows(self, split=None):
        """Only the image path leaves the CSV: the class columns are never read."""
        with open(self.split_csv(split), encoding="utf-8") as f:
            return [{"file_name": r["relative_path"], "relative_path": r["relative_path"]} for r in csv.DictReader(f)]

    def all_rows(self):
        """The union of the three splits' pools (for one region file that serves every split)."""
        seen = {}
        for s in (1, 2, 3):
            for r in self.rows(s):
                seen.setdefault(r["file_name"], r)
        return [seen[k] for k in sorted(seen)]

    def csv_path(self):
        return self.split_csv()

    def manifest(self, rows):
        p = self.split_csv()
        return {"pool": self.name, "ssl_pool_csv": str(p), "csv_sha256": sha256_of_file(p),
                "columns_available_to_the_method": ["relative_path"],
                "note": "chips of the official train folder only; validation and labeled chips of the split are excluded by the CSV; "
                        "the test folder is never read by SSL"}

    def load(self, row):
        _, atr_data = self._mods()
        im = atr_data.load_image(row["relative_path"])
        return im.resize((self.input_size, self.input_size), Image.BILINEAR)

    def load_for_proposal(self, row):
        _, atr_data = self._mods()
        return atr_data.load_image(row["relative_path"])

    def gt_boxes(self, row):
        """The chip's vehicle box from its .xml sidecar, at the working scale; diagnostic only."""
        atr_paths, _ = self._mods()
        xml = (atr_paths.SOC40_DIR / row["relative_path"]).with_suffix(".xml").read_text(encoding="utf-8", errors="replace")
        b = {k: int(rx.search(xml).group(1)) for k, rx in self._BOX_RE.items()}
        s = self.region_scale
        return np.array([[max(b["xmin"], 0) * s, max(b["ymin"], 0) * s, min(b["xmax"] + 1, self.native_size) * s, min(b["ymax"] + 1, self.native_size) * s]])

    def view_coverage(self, rows):
        return None

    def regions_file(self, smoke=False):
        return REGIONS_DIR / ("regions_atrnet_smoke.npz" if smoke else "regions_atrnet.npz")


class FAIRTilesPool:
    """FAIR-CSAR-V1.0 SL: the study's SSL pool slices (FAIR_chips/data/ssl_pool.csv: trainval minus the selection slices)
    cut 2 x 2 into 512-px tiles at the native 1 m resolution. One row per tile; file_name = <slice>#t<k> (k = 0..3,
    row-major) is the key in the region file. Nothing here reads a label: the annotation files are opened only by
    gt_boxes (diagnostic)."""
    name = "fair_tiles"
    dataset = "FAIR-CSAR-V1.0 SL trainval slices (minus selection slices), 2 x 2 native 512-px tiles"
    input_size = 512
    propose_size = 512
    region_scale = 1.0
    TILE = 512
    TILES = 4

    def _fp(self):
        d = REPO / "FAIR_chips"
        if str(d) not in sys.path:
            sys.path.append(str(d))                      # appended: nothing of the suite is shadowed
        import fair_paths as fp  # noqa: E402
        import fair_annotations as fa  # noqa: E402
        return fp, fa

    def _slices(self):
        fp, _ = self._fp()
        with open(fp.SSL_POOL_CSV, newline="", encoding="utf-8") as f:
            return [r["slice"] for r in csv.DictReader(f)]

    @staticmethod
    def tile_origin(k):
        return (k % 2) * FAIRTilesPool.TILE, (k // 2) * FAIRTilesPool.TILE

    def rows(self):
        return [{"file_name": f"{s}#t{k}", "slice": s, "tile": k, "official_split": "trainval"} for s in self._slices() for k in range(self.TILES)]

    def csv_path(self):
        return self._fp()[0].SSL_POOL_CSV

    def manifest(self, rows):
        fp, _ = self._fp()
        return {"pool": self.name, "ssl_pool_csv": str(fp.SSL_POOL_CSV), "csv_sha256": sha256_of_file(fp.SSL_POOL_CSV),
                "tiles_per_slice": self.TILES, "tile_px": self.TILE, "columns_available_to_the_method": ["slice", "tile"],
                "note": "trainval slices only, minus the selection slices (FAIR_chips/data/select.csv); the test split is never read by SSL; no labels read"}

    def _open(self, row):
        fp, _ = self._fp()
        with Image.open(fp.png_path(row["official_split"], row["slice"])) as im:
            return im.convert("L")

    def load(self, row):
        x, y = self.tile_origin(row["tile"])
        return self._open(row).crop((x, y, x + self.TILE, y + self.TILE))

    def load_for_proposal(self, row):
        return self.load(row)

    def gt_boxes(self, row):
        """Axis-aligned extents of the oriented boxes that fall in this tile, in tile pixels; diagnostic only."""
        _, fa = self._fp()
        x, y = self.tile_origin(row["tile"])
        out = []
        for o in fa.objects(row["official_split"], row["slice"]):
            x0, y0, x1, y1 = max(o["x0"] - x, 0), max(o["y0"] - y, 0), min(o["x1"] - x, self.TILE), min(o["y1"] - y, self.TILE)
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                out.append([x0, y0, x1, y1])
        return np.array(out, dtype=np.float32).reshape(-1, 4)

    def view_coverage(self, rows):
        return None

    def regions_file(self, smoke=False):
        return REGIONS_DIR / ("regions_fair_tiles_smoke.npz" if smoke else "regions_fair_tiles.npz")


class FAIRDownPool(FAIRTilesPool):
    """The same slices downsampled 1024 -> 512 (2 m per pixel): one row per slice, file_name = <slice>."""
    name = "fair_down"
    dataset = "FAIR-CSAR-V1.0 SL trainval slices (minus selection slices), downsampled to 512 px"
    FACTOR = 2

    def rows(self):
        return [{"file_name": s, "slice": s, "official_split": "trainval"} for s in self._slices()]

    def manifest(self, rows):
        m = super().manifest(rows)
        m.update(tiles_per_slice=1, tile_px=None, downsample=self.FACTOR, columns_available_to_the_method=["slice"])
        return m

    def load(self, row):
        return self._open(row).resize((self.input_size, self.input_size), Image.BILINEAR)

    def gt_boxes(self, row):
        _, fa = self._fp()
        out = [[o["x0"], o["y0"], o["x1"], o["y1"]] for o in fa.objects(row["official_split"], row["slice"])]
        return np.array(out, dtype=np.float32).reshape(-1, 4) / self.FACTOR

    def regions_file(self, smoke=False):
        return REGIONS_DIR / ("regions_fair_down_smoke.npz" if smoke else "regions_fair_down.npz")


POOLS = {"sardet": SARDetPool, "atrnet": ATRNetPool, "fair_tiles": FAIRTilesPool, "fair_down": FAIRDownPool}


def get(name: str):
    if name not in POOLS:
        raise KeyError(f"unknown pool {name!r}; choose from {sorted(POOLS)}")
    return POOLS[name]()
