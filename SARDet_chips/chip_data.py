"""Chip records and the dataset: which chips form a labeled set, the selection set and the two test sets.

Membership never moves: a chip is in the split of its image, and the image lists are the detection suite's
audited manifests (SARDet/data). DESIGN.md section 3.
    train set    chips of the images in SARDet/data/split<k>/labeled_<budget>.csv       (official train)
    select set   chips of the images in SARDet/data/val_select.csv                      (official val, 2,000 images)
    test         every chip of an official test image
    test_clean   test minus images that share content with train (eval_subsets.json) minus chips holding another class
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import chip_paths as cp

LABELED_CSV = {"1pct": "labeled_1pct.csv", "3pct": "labeled_3pct.csv", "10pct": "labeled_10pct.csv"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def load_chip_rows(root: Path = None):
    root = Path(root or cp.CHIPS_ROOT)
    with open(root / "chips.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["idx"], r["label"] = int(r["idx"]), int(r["label"])
        r["n_others_inside"], r["other_class_inside"] = int(r["n_others_inside"]), int(r["other_class_inside"])
    meta = json.loads((root / "chips_meta.json").read_text(encoding="utf-8"))
    if meta["class_names"] != cp.CLASS_NAMES:
        raise ValueError("chips_meta.json class names differ from the suite's CLASS_NAMES")
    return rows, meta


def image_names(csv_path: Path) -> set:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {r["file_name"] for r in csv.DictReader(f)}


def records(kind: str, rows, split: int = 1, budget: str = "3pct", root: Path = None) -> dict:
    """Chips for one set. Returns {"rows": [chip rows], "source": description, "csv_sha256": ...}."""
    root = Path(root or cp.CHIPS_ROOT)
    if kind == "train":
        p = cp.SUITE_DATA_DIR / f"split{split}" / LABELED_CSV[budget]
        names = image_names(p)
        sel = [r for r in rows if r["official_split"] == "train" and r["file_name"] in names]
        return {"rows": sel, "source": str(p.relative_to(cp.REPO_ROOT)), "csv_sha256": sha256_file(p), "images": len(names)}
    if kind == "select":
        p = cp.SUITE_DATA_DIR / "val_select.csv"
        names = image_names(p)
        sel = [r for r in rows if r["official_split"] == "val" and r["file_name"] in names]
        return {"rows": sel, "source": str(p.relative_to(cp.REPO_ROOT)), "csv_sha256": sha256_file(p), "images": len(names)}
    if kind == "test":
        sel = [r for r in rows if r["official_split"] == "test"]
        return {"rows": sel, "source": "every chip of an official test image", "csv_sha256": None, "images": len({r["file_name"] for r in sel})}
    if kind == "test_clean":
        clean = set(json.loads(Path(cp.EVAL_SUBSETS_JSON).read_text(encoding="utf-8"))["test_clean"])
        sel = [r for r in rows if r["official_split"] == "test" and r["file_name"] in clean and not r["other_class_inside"]]
        return {"rows": sel, "source": "test chips of test_clean images (eval_subsets.json) without another class inside the chip",
                "csv_sha256": sha256_file(cp.EVAL_SUBSETS_JSON), "images": len({r["file_name"] for r in sel})}
    raise ValueError(kind)


def class_counts(sel) -> dict:
    c = Counter(r["class_name"] for r in sel)
    return {k: c.get(k, 0) for k in cp.CLASS_NAMES}


def content_hash(sel) -> str:
    return hashlib.sha256("\n".join(f"{r['idx']},{r['label']}" for r in sorted(sel, key=lambda r: r["idx"])).encode()).hexdigest()


class ChipDataset(Dataset):
    """(image tensor, label) for a list of chip rows; the chips are read from the array once and kept in memory."""

    def __init__(self, sel, transform, arr=None, root: Path = None):
        arr = np.load(Path(root or cp.CHIPS_ROOT) / "chips_u8.npy", mmap_mode="r") if arr is None else arr
        idx = np.array([r["idx"] for r in sel], dtype=np.int64)
        order = np.argsort(idx)                       # sorted reads are sequential on the memory map
        self.chips = np.empty((len(idx),) + arr.shape[1:], dtype=np.uint8)
        self.chips[order] = arr[idx[order]]
        self.labels = torch.tensor([r["label"] for r in sel])
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.chips[i])), self.labels[i]
