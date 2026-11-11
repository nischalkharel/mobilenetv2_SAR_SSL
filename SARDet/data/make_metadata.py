"""Build data/image_metadata.csv: one row per image of the whole release (train, val and test).

Columns: file_name, official_split, image_id, width, height, source, satellite, polarization, band,
resolution, parent, group, md5, original.

`group` is the no-leak unit used by make_splits.py: images with the same parent image OR byte-identical
content (same md5) share one group (union of the two relations), so a group never straddles subsets.
Run once per machine? No: once, then commit the CSV; it depends only on the release. ~3 min (hashes every file).

    python data/make_metadata.py [--overwrite]
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import metadata, paths  # noqa: E402


def md5_of_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def md5_of_pixels(path):
    """MD5 of the decoded single-channel pixels: equal for lossless re-encodings of the same image (PNG vs BMP)."""
    from PIL import Image
    with Image.open(path) as im:
        return hashlib.md5(im.convert("L").tobytes()).hexdigest()


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--regroup", action="store_true",
                    help="re-derive parent/scene/group from the existing CSV (no re-hashing), after a change of the grouping rule")
    args = ap.parse_args()
    if paths.METADATA_CSV.exists() and not (args.overwrite or args.regroup):
        sys.exit(f"{paths.METADATA_CSV} exists; use --overwrite or --regroup")
    anon_to_original = metadata.load_mapping()
    rows = []
    t0 = time.time()
    if args.regroup:
        for r in metadata.read_metadata():
            info = metadata.parse_original_name(r["original"])
            r.update(parent=info["parent"], scene=info["scene"])
            rows.append(r)
        print(f"re-read {len(rows)} rows; regrouping with the current rule")
    for split in [] if args.regroup else ["train", "val", "test"]:
        with open(paths.ANNOTATION_FILES[split], encoding="utf-8") as f:
            ann = json.load(f)
        for im in ann["images"]:
            orig = anon_to_original[im["file_name"]]
            info = metadata.parse_original_name(orig)
            rows.append({"file_name": im["file_name"], "official_split": split, "image_id": im["id"],
                         "width": im["width"], "height": im["height"], "source": info["source"],
                         "satellite": info["satellite"], "polarization": info["polarization"], "band": info["band"],
                         "resolution": info["resolution"], "parent": info["parent"], "scene": info["scene"], "group": "",
                         "md5": md5_of_file(paths.image_path(split, im["file_name"])),
                         "pixel_md5": md5_of_pixels(paths.image_path(split, im["file_name"])), "original": orig})
        print(f"{split}: {len(ann['images'])} images ({time.time() - t0:.0f} s)")
    # groups = connected components of (same parent) U (same file md5) U (same pixel md5); HRSID scenes are NOT parents
    uf = UnionFind()
    by_parent, by_md5, by_pixels = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        key = f"{r['official_split']}/{r['file_name']}"
        by_parent[r["parent"]].append(key)
        by_md5[r["md5"]].append(key)
        by_pixels[r["pixel_md5"]].append(key)
    for members in list(by_parent.values()) + list(by_md5.values()) + list(by_pixels.values()):
        for m in members[1:]:
            uf.union(members[0], m)
    root_name = {}
    for r in rows:
        key = f"{r['official_split']}/{r['file_name']}"
        root = uf.find(key)
        root_name.setdefault(root, f"g{len(root_name):06d}")
        r["group"] = root_name[root]
    rows.sort(key=lambda r: (r["official_split"], r["file_name"]))
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(paths.METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=metadata.METADATA_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    # summary
    n_groups = len(root_name)
    multi = sum(1 for g in by_md5.values() if len(g) > 1)
    multi_px = sum(1 for g in by_pixels.values() if len(g) > 1)
    print(f"wrote {paths.METADATA_CSV}: {len(rows)} images, {n_groups} groups, {multi} file-md5 duplicate groups, "
          f"{multi_px} pixel-md5 duplicate groups")


if __name__ == "__main__":
    main()
