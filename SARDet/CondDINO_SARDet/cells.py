"""The four cell definitions of CondDINO on SARDet-100K (DESIGN.md section 5).

Every definition returns ONE integer cell id per pool row plus the cell-name table. The rows are the manifest
rows the engine passes to build() (columns file_name, official_split, image_id, width, height, source, satellite,
polarization, band, group; NO labels).

    source_pol    shared.metadata.CELL_DEFINITIONS["source_pol"] = source | satellite | polarization, from the
                  manifest columns. Labels never read. A condition in the sensor sense but NOT deployable (the
                  source is a curation artefact) and NOT class-independent (the cell report measures the leak).
    perm          the same cells with the image -> cell assignment scrambled (numpy default_rng(permute_seed));
                  the size-matched control. If it gains over DINO, the gain is not the physical grouping.
    oracle_class  cell = the sorted set of classes present in the image, read from the official train.json
                  (shared.coco_data.load_annotations). THIS USES LABELS: an explicit label-leak CEILING, never a
                  method. Marked "oracle_class control, labels used" in every record.
    radiometric   label-free and deployable: for every pool image a 32-bin intensity histogram (normalised) of
                  the detector-resized grey image plus log(width x height) of the native image; k-means (k = 16,
                  own numpy implementation, fixed seed) fitted on the SSL pool; cell = cluster id. The features
                  and the assignment are cached under outputs/cache/ keyed by the SHA-256 of the pool file list
                  (the engine's used_file_list_sha256), and a copy is written next to the run's cells.csv.

assign_variant(variant, rows, ...) -> (ids, names, info) is the single entry point used by train_ssl.py and
cell_report.py. The permutation and the deterministic id rule are in cond_center.py.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from shared import metadata, paths  # noqa: E402
from shared.run_record import sha256_of_text  # noqa: E402
from shared.ssl_data import load_resized_gray  # noqa: E402
from cond_center import assign_cells, ids_from_keys, permute_ids  # noqa: E402

# label_use strings are the ones every config / record / cells.csv must carry (train_ssl.py enforces equality)
VARIANTS = {
    "source_pol": {"label_use": "none (manifest columns source, satellite, polarization)",
                   "fields": metadata.CELL_DEFINITIONS["source_pol"], "deployable": False, "class_independent": False,
                   "what": "source dataset refined by per-image satellite x polarisation (Shipdataset) / sub-source (SIVED)"},
    "perm": {"label_use": "none (source_pol cells, image->cell scrambled)",
             "fields": metadata.CELL_DEFINITIONS["source_pol"], "deployable": False, "class_independent": True,
             "what": "size-matched control: the source_pol cell sizes with a random image -> cell assignment"},
    "oracle_class": {"label_use": "oracle_class control, labels used",
                     "fields": ["<classes present in the image, from train.json>"], "deployable": False, "class_independent": False,
                     "what": "LABEL-LEAK CEILING (never a method): cell = the set of classes present in the image"},
    "radiometric": {"label_use": "none (pixels and native image size only)",
                    "fields": ["<32-bin intensity histogram + log(width*height), k-means k=16>"], "deployable": True, "class_independent": False,
                    "what": "label-free deployable cells from the image itself (k-means on a radiometric descriptor)"},
}
N_HIST_BINS = 32
KMEANS_K = 16
KMEANS_SEED = 0
KMEANS_MAX_ITERS = 200
HIST_BATCH = 64


def pool_sha256(rows) -> str:
    """Same hash as the engine's manifest['used_file_list_sha256']."""
    return sha256_of_text("\n".join(r["file_name"] for r in rows))


def assignment_sha256(ids: np.ndarray) -> str:
    return sha256_of_text(",".join(map(str, ids.tolist())))


def output_dir(sub: str) -> Path:
    """outputs/<sub>/ of this experiment (or under SARDET_OUTPUT_ROOT, like every other output)."""
    d = paths.experiment_output_dir(HERE) / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---- source_pol / perm ------------------------------------------------------------------------------------------
def cells_source_pol(rows):
    return assign_cells(rows, VARIANTS["source_pol"]["fields"])


def cells_perm(rows, permute_seed: int):
    ids, names = cells_source_pol(rows)
    return permute_ids(ids, permute_seed), names


# ---- oracle_class (LABELS USED) ---------------------------------------------------------------------------------
def cells_oracle_class(rows):
    """Cell = sorted set of classes present in the image. READS train.json. Ceiling only."""
    from shared.coco_data import load_annotations
    ann = load_annotations("train")
    keys = []
    for r in rows:
        if r["official_split"] != "train":
            raise ValueError("oracle_class cells are defined on the official train annotations only")
        im = ann["images"][r["file_name"]]
        cats = sorted({a["category_id"] for a in ann["anns"].get(im["id"], [])})
        keys.append("+".join(paths.CLASS_NAMES[c] for c in cats) or "none")
    return ids_from_keys(keys)


# ---- radiometric (label-free) ------------------------------------------------------------------------------------
class _HistogramDataset(Dataset):
    """Normalised 32-bin histogram of the detector-resized grey image (uint8 values >> 3)."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        im = load_resized_gray(r["official_split"], r["file_name"])
        a = np.frombuffer(im.tobytes(), dtype=np.uint8)
        h = np.bincount(a >> 3, minlength=N_HIST_BINS).astype(np.float32) / a.size
        return torch.from_numpy(h), i


def radiometric_features(rows, num_workers=0, log_every=200) -> np.ndarray:
    """(N, 33) float32: 32 histogram bins (sum 1) + log(native width x height)."""
    feats = np.zeros((len(rows), N_HIST_BINS + 1), dtype=np.float32)
    loader = DataLoader(_HistogramDataset(rows), batch_size=HIST_BATCH, shuffle=False, num_workers=num_workers,
                        persistent_workers=False)
    t0 = time.time()
    for it, (h, idx) in enumerate(loader):
        feats[idx.numpy(), :N_HIST_BINS] = h.numpy()
        if log_every and (it + 1) % log_every == 0:
            print(f"    histograms {min((it + 1) * HIST_BATCH, len(rows))}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    feats[:, N_HIST_BINS] = [math.log(float(r["width"]) * float(r["height"])) for r in rows]
    return feats


def kmeans(x: np.ndarray, k: int, seed: int, max_iters: int = KMEANS_MAX_ITERS):
    """Plain k-means (k-means++ seeding, Lloyd iterations) in numpy float64 with a fixed seed.

    x is standardised per dimension first (the log-size column has a different scale than the histogram bins).
    Empty clusters keep their previous centroid. Returns (assignment, centroids in standardised space, mean, std,
    iterations, inertia)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < k:
        raise ValueError(f"k-means with k={k} needs at least k images, got {n}")
    mean, std = x.mean(0), x.std(0) + 1e-8
    z = (x - mean) / std
    rng = np.random.default_rng(seed)
    centers = np.empty((k, z.shape[1]))
    centers[0] = z[rng.integers(n)]
    d2 = ((z - centers[0]) ** 2).sum(1)
    for j in range(1, k):
        idx = rng.integers(n) if d2.sum() <= 0 else rng.choice(n, p=d2 / d2.sum())
        centers[j] = z[idx]
        d2 = np.minimum(d2, ((z - centers[j]) ** 2).sum(1))
    assign = None
    zz = (z ** 2).sum(1, keepdims=True)
    for it in range(1, max_iters + 1):
        dist = zz - 2.0 * z @ centers.T + (centers ** 2).sum(1)[None, :]
        new = dist.argmin(1)
        if assign is not None and np.array_equal(new, assign):
            break
        assign = new
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = z[m].mean(0)
    inertia = float(np.take_along_axis(dist, assign[:, None], 1).clip(min=0).sum())
    return assign.astype(np.int64), centers, mean, std, it, inertia


def radiometric_cache_file(rows) -> Path:
    return output_dir("cache") / f"radiometric_{pool_sha256(rows)[:16]}.npz"


def cells_radiometric(rows, num_workers=0, k=KMEANS_K, seed=KMEANS_SEED, verbose=True):
    """Returns (ids, names, info). Uses the cache when the pool file list matches; otherwise computes and caches."""
    pool_hash = pool_sha256(rows)
    cache = radiometric_cache_file(rows)
    if cache.exists():
        z = np.load(cache)
        if str(z["pool_sha256"]) == pool_hash and int(z["k"]) == k and int(z["seed"]) == seed and len(z["assignment"]) == len(rows):
            if verbose:
                print(f"  radiometric cells: loaded from cache {cache.name}", flush=True)
            ids = z["assignment"].astype(np.int64)
            info = json.loads(str(z["info"]))
            info["cache_file"] = str(cache)
            info["from_cache"] = True
            return ids, [f"kmeans{j:02d}" for j in range(k)], info
    if verbose:
        print(f"  radiometric cells: histograms of {len(rows)} images ({num_workers} workers) ...", flush=True)
    t0 = time.time()
    feats = radiometric_features(rows, num_workers=num_workers, log_every=200 if verbose else 0)
    t_hist = time.time() - t0
    ids, centers, mean, std, iters, inertia = kmeans(feats, k, seed)
    sizes = np.bincount(ids, minlength=k)
    info = {"variant": "radiometric", "pool_sha256": pool_hash, "num_images": len(rows), "k": k, "seed": seed,
            "feature": f"{N_HIST_BINS}-bin normalised intensity histogram of the 512-resized grey image + log(native width*height)",
            "standardised": True, "kmeans_iterations": iters, "inertia": inertia,
            "cluster_sizes": sizes.tolist(), "histogram_seconds": round(t_hist, 1),
            "kmeans_seconds": round(time.time() - t0 - t_hist, 1), "numpy": np.__version__, "from_cache": False}
    np.savez(cache, features=feats, assignment=ids, centroids=centers, mean=mean, std=std,
             pool_sha256=np.array(pool_hash), k=np.array(k), seed=np.array(seed), info=np.array(json.dumps(info)))
    info["cache_file"] = str(cache)
    if verbose:
        print(f"  radiometric cells: k-means k={k} done in {iters} iterations; sizes {sizes.tolist()}; "
              f"cached as {cache.name} ({time.time() - t0:.0f}s)", flush=True)
    return ids, [f"kmeans{j:02d}" for j in range(k)], info


# ---- entry point -------------------------------------------------------------------------------------------------
def assign_variant(variant: str, rows, permute_seed: int = 0, num_workers: int = 0, verbose: bool = True):
    """(ids int64 (N,), names list, info dict) for one of the four variants (or 'none': one cell, = DINO)."""
    if variant == "none":
        return np.zeros(len(rows), dtype=np.int64), ["all"], {"variant": "none", "label_use": "none"}
    if variant not in VARIANTS:
        raise ValueError(f"unknown cell variant {variant!r}; choose one of {sorted(VARIANTS)}")
    info = {"variant": variant, "label_use": VARIANTS[variant]["label_use"], "deployable": VARIANTS[variant]["deployable"],
            "class_independent_by_construction": VARIANTS[variant]["class_independent"], "fields": list(VARIANTS[variant]["fields"])}
    if variant == "source_pol":
        ids, names = cells_source_pol(rows)
    elif variant == "perm":
        ids, names = cells_perm(rows, permute_seed)
        info.update(permute_seed=int(permute_seed), numpy=np.__version__)
    elif variant == "oracle_class":
        ids, names = cells_oracle_class(rows)
        info["WARNING"] = "labels were read to build these cells; this is a leak ceiling, not a method"
    else:
        ids, names, extra = cells_radiometric(rows, num_workers=num_workers, verbose=verbose)
        info.update({k: v for k, v in extra.items() if k not in ("variant",)})
    info.update(n_cells=len(names), cell_assignment_sha256=assignment_sha256(ids), pool_sha256=pool_sha256(rows))
    return ids, names, info
