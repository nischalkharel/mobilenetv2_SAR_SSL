"""Unlabeled SSL data and the split audit.

SSL training reads only the `relative_path` column of ssl_training_data.csv. The dataset returns
image views and nothing else: labels never reach the DINO loss, the sampler, the view generation,
or the choice of checkpoint. (The final epoch's checkpoint is used, fixed in advance.)

audit_splits() is the one place that reads labels and metadata: it checks that the saved subset
CSVs are consistent and leak-free before any training starts. It does not change any file.
"""
import csv
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import paths
from data import NATIVE_SIZE, load_image
from utils import sha256_file, sha256_text

SPLIT_FILES = {
    "ssl": "ssl_training_data.csv",
    "labeled": "labeled_set.csv",
    "validation": "validation.csv",
    "excluded": "excluded_near_duplicates.csv",
}


def flat_args(cfg: dict) -> SimpleNamespace:
    """Merge the config sections into one object with the original DINO argument names."""
    flat = {"run_name": cfg["run_name"]}
    for section in ["setup", "views", "dino", "optim"]:
        for k, v in cfg[section].items():
            if k.startswith("_"):
                continue
            if k in flat:
                raise ValueError(f"config key {k} appears twice")
            flat[k] = v
    return SimpleNamespace(**flat)


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_ssl_paths(csv_path: Path) -> list[str]:
    """Image paths of the SSL pool, in file order. Only the relative_path column is used."""
    paths_ = [r["relative_path"] for r in read_rows(csv_path)]
    if len(set(paths_)) != len(paths_):
        raise ValueError(f"duplicate paths in {csv_path}")
    if not all(p.startswith("train/") for p in paths_):
        raise ValueError(f"{csv_path} contains images outside the official train/ folder")
    return paths_


def paths_hash(relative_paths: list[str]) -> str:
    """Identity of a set of images (sorted paths). This is a manifest hash, not an image-content hash:
    it changes if membership changes, not if the pixels inside a file change."""
    return sha256_text("\n".join(sorted(relative_paths)))


def audit_splits(splits_dir: Path = paths.SPLITS_DIR, check_test_folder: bool = True) -> dict:
    """Leakage and consistency checks on the saved CSVs. Raises ValueError on the first failure.
    Returns counts and SHA-256 file hashes for the run records."""
    rows = {k: read_rows(splits_dir / f) for k, f in SPLIT_FILES.items()}
    mapping = read_rows(splits_dir / "class_mapping.csv")
    name_of = {int(r["label"]): r["class_name"] for r in mapping}
    if sorted(name_of) != list(range(len(name_of))):
        raise ValueError("class_mapping.csv labels are not 0..N-1")
    problems = []

    sets = {}
    for k, rs in rows.items():
        p = [r["relative_path"] for r in rs]
        sets[k] = set(p)
        if len(p) != len(sets[k]):
            problems.append(f"{k}: duplicate paths")
        if not all(x.startswith("train/") for x in p):
            problems.append(f"{k}: paths outside the official train/ folder")
        for r in rs:
            folder = r["relative_path"].split("/")[1]
            if name_of.get(int(r["label"])) != r["class_name"] or folder != r["class_name"]:
                problems.append(f"{k}: label/class mismatch {r['relative_path']}")
                break

    if not sets["labeled"] <= sets["ssl"]:
        problems.append("labeled set is not inside the SSL pool")
    for a, b in [("validation", "ssl"), ("validation", "labeled"), ("excluded", "ssl"),
                 ("excluded", "labeled"), ("excluded", "validation")]:
        if sets[a] & sets[b]:
            problems.append(f"{a} and {b} share {len(sets[a] & sets[b])} images")

    # Documented grouping rule (preprocessing_dataset/SUBSETS.md): no acquisition group
    # (class|band|depression|azimuth|scene) may appear in both validation and training data.
    val_groups = {r["acquisition_group"] for r in rows["validation"]}
    for k in ["ssl", "labeled"]:
        shared = val_groups & {r["acquisition_group"] for r in rows[k]}
        if shared:
            problems.append(f"validation and {k} share {len(shared)} acquisition groups")

    # Label-budget sets (preprocessing_dataset/make_label_budgets.py): same rules as labeled_set.csv.
    budget_files = sorted(p.name for p in splits_dir.glob("labeled_set_*.csv"))
    for f in budget_files:
        rs = read_rows(splits_dir / f)
        p = {r["relative_path"] for r in rs}
        if len(p) != len(rs) or not p <= sets["ssl"]:
            problems.append(f"{f}: duplicate paths or images outside the SSL pool")
        if any(name_of.get(int(r["label"])) != r["class_name"] for r in rs):
            problems.append(f"{f}: label/class mismatch")
        if val_groups & {r["acquisition_group"] for r in rs}:
            problems.append(f"validation and {f} share acquisition groups")

    n_test = None
    if check_test_folder:
        test_ids = {p.stem for p in paths.SOC40_TEST_DIR.glob("*/*.tif")}
        n_test = len(test_ids)
        used_ids = {Path(p).stem for s in sets.values() for p in s}
        if test_ids & used_ids:
            problems.append(f"{len(test_ids & used_ids)} official test image IDs appear in the CSVs")

    if problems:
        raise ValueError("split audit failed:\n  " + "\n  ".join(problems))
    return {
        "counts": {k: len(v) for k, v in sets.items()} | {"official_test": n_test, "classes": len(name_of)},
        "csv_sha256": {f: sha256_file(splits_dir / f) for f in [*SPLIT_FILES.values(), *budget_files, "class_mapping.csv"]},
        "checks": ["unique paths in every CSV", "labels agree with class_mapping.csv and class folders",
                   "labeled (and every labeled_set_*.csv) inside SSL pool", "validation/excluded disjoint from SSL and labeled",
                   "no shared acquisition group between validation and SSL/labeled",
                   "only official train/ images", "no official test image ID in any CSV"],
    }


def build_pixel_cache(relative_paths: list[str], cache_dir: Path) -> Path:
    """Decode every image once into one uint8 file of shape (N, 128, 128), about 1.1 GB for the
    65,387-image pool, named after the image list. Later runs (and every data-loading worker) open it
    memory-mapped, so the OS keeps one shared copy instead of one per process. The pixels are exactly
    what load_image() returns; a mismatch on a few spot-checked images raises an error."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"pixels_{paths_hash(relative_paths)[:16]}_{len(relative_paths)}.npy"
    if not path.exists():
        t0 = time.time()
        tmp = path.with_suffix(".tmp.npy")
        arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                        shape=(len(relative_paths), NATIVE_SIZE, NATIVE_SIZE))
        for i, p in enumerate(relative_paths):
            arr[i] = np.asarray(load_image(p))
            if (i + 1) % 10000 == 0:
                print(f"    decoded {i + 1}/{len(relative_paths)} images into the cache, {time.time() - t0:.0f}s", flush=True)
        arr.flush()
        del arr
        tmp.replace(path)  # the cache only gets its final name once it is complete
    arr = np.load(path, mmap_mode="r")
    for i in np.linspace(0, len(relative_paths) - 1, 20).astype(int):
        if arr.shape[0] != len(relative_paths) or not np.array_equal(arr[i], np.asarray(load_image(relative_paths[i]))):
            raise ValueError(f"pixel cache {path} does not match the image files; delete it and rerun")
    return path


class SSLDataset(Dataset):
    """Returns the list of DINO views of one image: [global1, global2, local1, ..., localN].
    No label is returned.

    cache_file: a file made by build_pixel_cache (fast); None reads each image file from disk."""

    def __init__(self, relative_paths: list[str], transform, cache_file: Path | None = None):
        self.paths = relative_paths
        self.transform = transform
        self.cache_file = cache_file
        self.cache = None  # opened lazily, so each worker process maps the file itself

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        if self.cache_file is not None and self.cache is None:
            self.cache = np.load(self.cache_file, mmap_mode="r")
        # A 2-D uint8 array becomes a mode "L" image again, exactly as in data.SARDataset.
        img = Image.fromarray(np.array(self.cache[i])) if self.cache is not None else load_image(self.paths[i])
        return self.transform(img)

    def __getstate__(self):  # do not send an open memory map to worker processes
        return {**self.__dict__, "cache": None}
