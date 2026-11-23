"""Chip records of the FAIR-CSAR study: which chips form a labeled set, the selection set and the test sets.

Membership never moves: a chip is in the role of its slice (FAIR_chips/data). DESIGN.md section 2.
    train        chips of the slices in data/split<k>/labeled_<budget>.csv          (trainval, inside the SSL pool)
    select       chips of the slices in data/select.csv                            (trainval, outside the SSL pool)
    test         every chip of a test slice
    test_clean   test chips whose window holds no other class's box centre (the SARDet_chips rule, minus the
                 cross-split image sharing that FAIR-CSAR does not record)
The dataset class is SARDet_chips/chip_data.ChipDataset (one array file, chips kept in memory).
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import fair_paths as fp

if str(fp.SARDET_CHIPS_DIR) not in sys.path:
    sys.path.append(str(fp.SARDET_CHIPS_DIR))          # appended: SARDet_chips/chip_data.py is wanted, nothing of FAIR_chips is shadowed
from chip_data import ChipDataset, sha256_file  # noqa: E402, F401  (this file is named fair_* so it never shadows that one)


def load_chip_rows(root: Path = None):
    root = Path(root or fp.CHIPS_ROOT)
    with open(root / "chips.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("idx", "label", "difficult", "n_others_inside", "other_class_inside"):
            r[k] = int(r[k])
    meta = json.loads((root / "chips_meta.json").read_text(encoding="utf-8"))
    if meta["class_names"] != fp.CLASS_NAMES:
        raise ValueError("chips_meta.json class names differ from fair_paths.CLASS_NAMES")
    return rows, meta


def slice_names(csv_path: Path) -> set:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {r["slice"] for r in csv.DictReader(f)}


def records(kind: str, rows, split: int = 1, budget: str = "3pct") -> dict:
    """Chips for one set: {"rows": [...], "source": description, "csv_sha256": ..., "images": n slices}."""
    if kind == "train":
        p = fp.labeled_csv(split, budget)
        names = slice_names(p)
        sel = [r for r in rows if r["official_split"] == "trainval" and r["slice"] in names]
        return {"rows": sel, "source": str(p.relative_to(fp.REPO_ROOT)), "csv_sha256": sha256_file(p), "images": len(names)}
    if kind == "select":
        names = slice_names(fp.SELECT_CSV)
        sel = [r for r in rows if r["official_split"] == "trainval" and r["slice"] in names]
        return {"rows": sel, "source": str(fp.SELECT_CSV.relative_to(fp.REPO_ROOT)), "csv_sha256": sha256_file(fp.SELECT_CSV), "images": len(names)}
    if kind == "test":
        sel = [r for r in rows if r["official_split"] == "test"]
        return {"rows": sel, "source": "every chip of a test slice", "csv_sha256": None, "images": len({r["slice"] for r in sel})}
    if kind == "test_clean":
        sel = [r for r in rows if r["official_split"] == "test" and not r["other_class_inside"]]
        return {"rows": sel, "source": "test chips without another class's box centre inside the window", "csv_sha256": None,
                "images": len({r["slice"] for r in sel})}
    raise ValueError(kind)


def class_counts(sel) -> dict:
    c = Counter(r["class_name"] for r in sel)
    return {k: c.get(k, 0) for k in fp.CLASS_NAMES}


def content_hash(sel) -> str:
    return hashlib.sha256("\n".join(f"{r['idx']},{r['label']}" for r in sorted(sel, key=lambda r: r["idx"])).encode()).hexdigest()
