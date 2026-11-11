"""Subset manifests: reading, hashing and the quick leak check every run performs before training.

A manifest is a CSV with the columns MANIFEST_COLUMNS (no labels). The SHA-256 of the file bytes is recorded in
every run record; quick_audit() re-checks the protocol invariants in a second and returns them for the record.
"""
import csv
import json

from . import paths
from .run_record import sha256_of_file

MANIFEST_COLUMNS = ["file_name", "official_split", "image_id", "width", "height", "source", "satellite",
                    "polarization", "band", "group"]

LABELED_CSV = {"1pct": "labeled_1pct.csv", "3pct": "labeled_3pct.csv", "10pct": "labeled_10pct.csv"}


def read(csv_path) -> list:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["image_id"] = int(r["image_id"])
        r["width"] = int(r["width"])
        r["height"] = int(r["height"])
    return rows


def write(csv_path, rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def ssl_pool_csv():
    return paths.DATA_DIR / "ssl_pool.csv"


def labeled_csv(budget: str, split=None):
    split_dir = paths.SPLIT_DIR if split is None else paths.DATA_DIR / f"split{split}"
    return split_dir / LABELED_CSV[budget]


def eval_subsets() -> dict:
    with open(paths.EVAL_SUBSETS_JSON, encoding="utf-8") as f:
        return json.load(f)


def quick_audit(split=None) -> dict:
    """Protocol invariants for one split; raises on any violation. Returns hashes + counts for the run record."""
    split = split or paths.SPLIT
    pool = read(ssl_pool_csv())
    labeled = {b: read(labeled_csv(b, split)) for b in LABELED_CSV}
    val_select = read(paths.VAL_SELECT_CSV)
    subsets = eval_subsets()
    pool_names = {r["file_name"] for r in pool}
    if len(pool_names) != len(pool):
        raise AssertionError("duplicate file names in ssl_pool.csv")
    if any(r["official_split"] != "train" for r in pool):
        raise AssertionError("ssl_pool.csv contains non-train images")
    names = {b: {r["file_name"] for r in rows} for b, rows in labeled.items()}
    if not (names["1pct"] <= names["3pct"] <= names["10pct"] <= pool_names):
        raise AssertionError("labeled sets are not nested inside the SSL pool")
    train_groups = {r["group"] for r in pool}
    scored_groups = set(subsets["val_groups"]) | set(subsets["test_groups"])
    if train_groups & scored_groups:
        raise AssertionError(f"{len(train_groups & scored_groups)} SSL-pool groups also appear in official val/test")
    if any(r["official_split"] != "val" for r in val_select):
        raise AssertionError("val_select.csv contains non-val images")
    if {r["file_name"] for r in val_select} - set(subsets["val"]):
        raise AssertionError("val_select.csv is not a subset of official val")
    hashes = {"ssl_pool.csv": sha256_of_file(ssl_pool_csv()), "val_select.csv": sha256_of_file(paths.VAL_SELECT_CSV),
              "eval_subsets.json": sha256_of_file(paths.EVAL_SUBSETS_JSON),
              "image_metadata.csv": sha256_of_file(paths.METADATA_CSV)}
    for b in LABELED_CSV:
        hashes[f"split{split}/{LABELED_CSV[b]}"] = sha256_of_file(labeled_csv(b, split))
    return {"split": split, "manifest_sha256": hashes,
            "counts": {"ssl_pool": len(pool), **{f"labeled_{b}": len(rows) for b, rows in labeled.items()},
                       "val_select": len(val_select), "val": len(subsets["val"]), "test": len(subsets["test"]),
                       "test_clean": len(subsets["test_clean"]), "val_clean": len(subsets["val_clean"])}}
