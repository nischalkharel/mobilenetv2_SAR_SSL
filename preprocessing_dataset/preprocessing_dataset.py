"""Build fixed SOC-40 subsets (validation, SSL pool, labeled set) as CSV manifests.

Run from the repository root:
    python preprocessing_dataset/preprocessing_dataset.py
For a re-drawn split (written to preprocessing_dataset/split<N>/, seed N-1):
    SAR_SPLIT=2 python preprocessing_dataset/preprocessing_dataset.py

What it does
1. Lists every image in SOC-40 train/ and test/, checks that each .tif has its .xml,
   that both splits have the same 40 classes, and that no image name appears twice.
   Reads the scene name of every train image from its .xml.
2. Validation (validation.csv): VAL_PER_CLASS images per class (about 800 in total), drawn from
   all of official train. Each one comes from a different viewing pose (angle), one image per pose.
3. Excluded near-copies (excluded_near_duplicates.csv): the other images of each validation image's
   acquisition group (see below). They go into no training set. Each row records which validation
   image it is a near-copy of and how similar the two are.
4. SSL pool (ssl_training_data.csv): every remaining train image.
5. Labeled set (labeled_set.csv): LABELED_FRACTION of the SSL pool, chosen per class. This simulates
   having only a small labeled budget.
6. Re-checks that nothing leaks, then writes the CSVs next to this script.
7. Writes SUBSETS.md and charts/*.pdf describing the subsets (subset_report.py).
The official test/ folder is only read for counting and leakage checks; it is never
written to any CSV.

Acquisition group: which images count as near-copies
The file name is <band>_<pol>_<depression>_<azimuth>_<id>.tif and the .xml gives the scene.
An acquisition group is one vehicle type at one band, depression, azimuth AND scene
(class|band|depression|azimuth|scene). Its images differ only by polarization (HH, HV, VH, VV)
or by a repeat pass over the same spot, so they are near-copies of each other. If one were in
validation and another in training, validation accuracy would be inflated. A measured check
(2026-10-13, vehicle region only, maximum normalized cross-correlation) gave a median similarity
of about 0.70 inside an acquisition group, 0.63 for the same vehicle 5 degrees apart in the same
scene, 0.47 for the same pose in a different scene, and 0.32 for two different vehicle types.
The same vehicle and angle in a different scene is therefore NOT treated as a near-copy and stays
usable for training. The labeled set lives entirely inside the SSL pool, so it is sampled image by
image; that cannot leak into validation.

The selection is deterministic for a given SEED and dataset copy, but the CSVs are the
record of truth: reuse them, do not regenerate them between experiments.
"""
import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so "import paths" works
import paths  # noqa: E402
from subset_report import make_report  # noqa: E402

SEED = paths.SPLIT - 1  # split 1 = seed 0 (the original subsets); set SAR_SPLIT=2 or 3 for the re-drawn splits
VAL_PER_CLASS = 20       # validation images per class, each from a different pose (40 classes -> 800)
LABELED_FRACTION = 0.03  # fraction of the SSL pool, per class (about 30 to 60 images per class)

CSV_COLUMNS = ["relative_path", "class_name", "label", "band", "polarization", "depression",
               "azimuth", "scene", "image_id", "pose_group", "acquisition_group"]
EXCLUDED_COLUMNS = CSV_COLUMNS + ["near_copy_of_validation_image", "validation_image_polarization",
                                  "similarity_vehicle_region", "similarity_full_image"]


def list_split(split_dir: Path, read_scene: bool) -> list[dict]:
    """Return one record per .tif in <split_dir>/<class>/, sorted, with parsed metadata."""
    if not split_dir.is_dir():
        sys.exit(f"ERROR: missing folder {split_dir}. Check paths.py.")
    records = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        names = {p.name for p in class_dir.iterdir()}
        tifs = sorted(n for n in names if n.endswith(".tif"))
        missing_xml = [n for n in tifs if n[:-4] + ".xml" not in names]
        if missing_xml:
            sys.exit(f"ERROR: {len(missing_xml)} .tif without .xml in {class_dir}, e.g. {missing_xml[0]}")
        for name in tifs:
            parts = name[:-4].split("_")  # band, pol, depression, azimuth, id
            if len(parts) != 5:
                sys.exit(f"ERROR: unexpected file name {class_dir / name}")
            band, pol, dep, az, image_id = parts
            r = {
                "relative_path": f"{split_dir.name}/{class_dir.name}/{name}",  # relative to SOC40_DIR
                "class_name": class_dir.name,
                "band": band, "polarization": pol, "depression": dep, "azimuth": az,
                "image_id": image_id,
                "pose_group": f"{class_dir.name}|{band}|{dep}|{az}",
            }
            if read_scene:
                xml = (class_dir / (name[:-4] + ".xml")).read_text(encoding="utf-8")
                m = re.search(r"<scene_name>(.*?)</scene_name>", xml)
                if not m:
                    sys.exit(f"ERROR: no <scene_name> in {class_dir / name[:-4]}.xml")
                r["scene"] = m.group(1)
                r["acquisition_group"] = f"{r['pose_group']}|{r['scene']}"
            records.append(r)
    return records


def pick_validation(train: list[dict], rng: random.Random) -> list[dict]:
    """Per class: VAL_PER_CLASS different poses at random, then one random image from each pose."""
    by_class_pose = defaultdict(lambda: defaultdict(list))
    for r in train:
        by_class_pose[r["class_name"]][r["pose_group"]].append(r)
    val = []
    for cls in sorted(by_class_pose):
        poses = sorted(by_class_pose[cls])
        for pose in rng.sample(poses, VAL_PER_CLASS):
            val.append(rng.choice(sorted(by_class_pose[cls][pose], key=lambda r: r["relative_path"])))
    return val


def pick_labeled(ssl: list[dict], rng: random.Random) -> set[str]:
    """Per class, take round(LABELED_FRACTION * n) images at random from the SSL pool."""
    by_class = defaultdict(list)
    for r in ssl:
        by_class[r["class_name"]].append(r["relative_path"])
    labeled = set()
    for cls in sorted(by_class):
        items = sorted(by_class[cls])
        labeled.update(rng.sample(items, round(LABELED_FRACTION * len(items))))
    return labeled


# ---- Similarity record for excluded near-copies ---------------------------------------

def _spectrum(relative_path: str, vehicle_only: bool) -> np.ndarray:
    """Standardized image (or vehicle bounding box from the .xml, plus 4 px) in the Fourier domain."""
    path = paths.SOC40_DIR / relative_path
    x = np.asarray(Image.open(path), dtype=np.float64)
    if vehicle_only:
        xml = path.with_suffix(".xml").read_text(encoding="utf-8")
        b = {k: int(float(re.search(rf"<{k}>\s*(-?[\d.]+)", xml).group(1))) for k in ["xmin", "xmax", "ymin", "ymax"]}
        x = x[max(b["ymin"] - 4, 0):b["ymax"] + 4, max(b["xmin"] - 4, 0):b["xmax"] + 4]
    x = (x - x.mean()) / (x.std() + 1e-9) / np.sqrt(x.size)  # unit length, so a perfect match scores 1
    return np.fft.rfft2(x, s=(256, 256))  # zero padding: shifts do not wrap around


def similarity(a: str, b: str, vehicle_only: bool) -> float:
    """Maximum normalized cross-correlation over all shifts (1 = identical up to a shift and brightness).
    Targets sit at random offsets in each chip, so the best-aligned shift is used."""
    cc = np.fft.irfft2(_spectrum(a, vehicle_only) * np.conj(_spectrum(b, vehicle_only)), s=(256, 256))
    return float(cc.max())


def write_csv(path: Path, rows: list[dict], columns=CSV_COLUMNS) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["relative_path"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing CSVs (only do this on purpose; experiments rely on them)")
    args = parser.parse_args()

    out_dir = paths.SPLITS_DIR
    outputs = {name: out_dir / name for name in
               ["validation.csv", "excluded_near_duplicates.csv", "ssl_training_data.csv",
                "labeled_set.csv", "class_mapping.csv"]}
    existing = [p.name for p in outputs.values() if p.exists()]
    if existing and not args.overwrite:
        sys.exit(f"ERROR: {existing} already exist in {out_dir}. They define the fixed subsets; "
                 "rerun with --overwrite only if you really want new ones.")

    # ---- 1. Inventory and basic checks ------------------------------------
    print(f"Reading {paths.SOC40_DIR} (including the scene name from every train .xml)")
    train = list_split(paths.SOC40_TRAIN_DIR, read_scene=True)
    test = list_split(paths.SOC40_TEST_DIR, read_scene=False)
    train_classes = sorted({r["class_name"] for r in train})
    test_classes = sorted({r["class_name"] for r in test})
    if train_classes != test_classes:
        sys.exit(f"ERROR: class folders differ between train and test: "
                 f"{set(train_classes) ^ set(test_classes)}")
    ids = Counter(r["image_id"] for r in train + test)
    repeated = [i for i, n in ids.items() if n > 1]
    if repeated:
        sys.exit(f"ERROR: {len(repeated)} image IDs appear more than once in SOC-40, e.g. {repeated[:5]}")

    # Labels are the index of the class name in sorted order (same as torchvision ImageFolder).
    class_to_label = {c: i for i, c in enumerate(train_classes)}
    for r in train + test:
        r["label"] = class_to_label[r["class_name"]]

    print(f"Unique images: train {len(train)}, test {len(test)}, total {len(train) + len(test)} "
          f"({len(train_classes)} classes)")
    test_groups = {r["pose_group"] for r in test}
    train_groups = {r["pose_group"] for r in train}
    print(f"Note (official benchmark, not fixable here): {len(train_groups & test_groups)} of "
          f"{len(train_groups | test_groups)} pose groups have images in both official train and test.")

    # ---- 2-5. Subsets -------------------------------------------------------
    rng = random.Random(SEED)
    validation = pick_validation(train, rng)
    val_paths = {r["relative_path"] for r in validation}
    val_by_acq = {r["acquisition_group"]: r for r in validation}
    excluded = [dict(r, near_copy_of_validation_image=val_by_acq[r["acquisition_group"]]["relative_path"],
                     validation_image_polarization=val_by_acq[r["acquisition_group"]]["polarization"])
                for r in train if r["acquisition_group"] in val_by_acq and r["relative_path"] not in val_paths]
    excl_paths = {r["relative_path"] for r in excluded}
    ssl = [r for r in train if r["relative_path"] not in val_paths | excl_paths]
    labeled_paths = pick_labeled(ssl, rng)
    labeled = [r for r in ssl if r["relative_path"] in labeled_paths]

    print(f"Measuring similarity of {len(excluded)} excluded near-copies to their validation image...")
    for r in excluded:
        r["similarity_vehicle_region"] = f"{similarity(r['relative_path'], r['near_copy_of_validation_image'], True):.3f}"
        r["similarity_full_image"] = f"{similarity(r['relative_path'], r['near_copy_of_validation_image'], False):.3f}"

    # ---- 6. Leakage checks (fail loudly) -----------------------------------
    test_ids = {r["image_id"] for r in test}
    val_ids = {r["image_id"] for r in validation}
    exc_ids = {r["image_id"] for r in excluded}
    ssl_ids = {r["image_id"] for r in ssl}
    lab_ids = {r["image_id"] for r in labeled}
    acq = lambda rows: {r["acquisition_group"] for r in rows}  # noqa: E731
    checks = {
        "validation and SSL pool share no image": not (val_ids & ssl_ids),
        "validation and labeled set share no image": not (val_ids & lab_ids),
        "excluded near-copies are in no other subset": not (exc_ids & (val_ids | ssl_ids)),
        "labeled set is inside the SSL pool": lab_ids <= ssl_ids,
        "SSL pool shares no acquisition group with validation": not (acq(ssl) & acq(validation)),
        "labeled set shares no acquisition group with validation": not (acq(labeled) & acq(validation)),
        "validation + excluded + SSL pool = all of train": len(val_ids) + len(exc_ids) + len(ssl_ids) == len(train),
        "no test image in any CSV": not (test_ids & (val_ids | exc_ids | ssl_ids | lab_ids)),
        "every CSV path comes from train/": all(r["relative_path"].startswith("train/")
                                               for r in validation + excluded + ssl + labeled),
        "validation images per class all from different poses": all(
            n == VAL_PER_CLASS for n in Counter(r["class_name"] for r in validation).values())
            and len({r["pose_group"] for r in validation}) == len(validation),
        "all 40 classes in validation and labeled set":
            len({r["class_name"] for r in validation}) == len({r["class_name"] for r in labeled}) == 40,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        sys.exit("ERROR: a leakage check failed; no CSVs were written.")

    # ---- Write ---------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(outputs["validation.csv"], validation)
    write_csv(outputs["excluded_near_duplicates.csv"], excluded, EXCLUDED_COLUMNS)
    write_csv(outputs["ssl_training_data.csv"], ssl)
    write_csv(outputs["labeled_set.csv"], labeled)
    with open(outputs["class_mapping.csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "class_name"])
        w.writerows((i, c) for c, i in class_to_label.items())

    # ---- Summary ------------------------------------------------------------
    def per_class(rows):
        return Counter(r["class_name"] for r in rows)
    v, s, l = per_class(validation), per_class(ssl), per_class(labeled)
    sims = np.array([float(r["similarity_vehicle_region"]) for r in excluded])
    print(f"\nvalidation.csv                {len(validation):6d} images ({VAL_PER_CLASS} per class)")
    print(f"excluded_near_duplicates.csv  {len(excluded):6d} images (in no training set; vehicle-region similarity "
          f"to their validation image: median {np.median(sims):.2f}, range {sims.min():.2f} to {sims.max():.2f})")
    print(f"ssl_training_data.csv         {len(ssl):6d} images")
    print(f"labeled_set.csv               {len(labeled):6d} images ({len(labeled) / len(ssl):.1%} of SSL pool)")
    print(f"test (not in any CSV)         {len(test):6d} images")
    print(f"Per class: validation {min(v.values())}-{max(v.values())}, "
          f"SSL pool {min(s.values())}-{max(s.values())}, labeled {min(l.values())}-{max(l.values())}")
    print(f"Wrote CSVs to {out_dir}  (paths in CSVs are relative to paths.SOC40_DIR)")
    make_report(VAL_PER_CLASS, LABELED_FRACTION, SEED)


if __name__ == "__main__":
    main()
