"""Build fixed SOC-40 subsets (validation, SSL pool, labeled set) as CSV manifests.

Run from the repository root:
    python preprocessing_dataset/preprocessing_dataset.py

What it does
1. Lists every image in SOC-40 train/ and test/, checks that each .tif has its .xml,
   that both splits have the same 40 classes, and that no image name appears twice.
2. Validation (validation.csv): about 10% of train, chosen per class.
3. SSL pool (ssl_training_data.csv): every train image not in validation.
4. Labeled set (labeled_set.csv): 10% of the SSL pool, chosen per class.
5. Re-checks that nothing leaks, then writes the CSVs next to this script.
The official test/ folder is only read for counting and leakage checks; it is never
written to any CSV.

Why validation is chosen by "pose group" and not image by image
The file name is <band>_<pol>_<depression>_<azimuth>_<id>.tif. The same vehicle seen at
the same band, depression and azimuth is usually imaged 4+ times (HH, HV, VH, VV, and
sometimes repeat passes). Those images are near-copies. If one went to validation and
its sibling stayed in training, validation accuracy would be inflated. So a whole pose
group (class, band, depression, azimuth) goes to validation or stays in training, never
both. The labeled set lives entirely inside the training side, so it is sampled image by
image (more pose variety for the same label count); that cannot leak into validation.

The selection is deterministic for a given SEED and dataset copy, but the CSVs are the
record of truth: reuse them, do not regenerate them between experiments.
"""
import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so "import paths" works
import paths  # noqa: E402

SEED = 0
VAL_FRACTION = 0.10
LABELED_FRACTION = 0.10  # fraction of the SSL pool (train minus validation)

CSV_COLUMNS = ["relative_path", "class_name", "label", "band", "polarization",
               "depression", "azimuth", "image_id", "pose_group"]


def list_split(split_dir: Path) -> list[dict]:
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
            records.append({
                "relative_path": f"{split_dir.name}/{class_dir.name}/{name}",  # relative to SOC40_DIR
                "class_name": class_dir.name,
                "band": band, "polarization": pol, "depression": dep, "azimuth": az,
                "image_id": image_id,
                "pose_group": f"{class_dir.name}|{band}|{dep}|{az}",
            })
    return records


def pick_validation(train: list[dict], rng: random.Random) -> set[str]:
    """Per class, take whole pose groups in random order until >= VAL_FRACTION of its images."""
    groups_by_class = defaultdict(lambda: defaultdict(list))
    for r in train:
        groups_by_class[r["class_name"]][r["pose_group"]].append(r["relative_path"])
    val = set()
    for cls in sorted(groups_by_class):
        groups = groups_by_class[cls]
        target = round(VAL_FRACTION * sum(len(v) for v in groups.values()))
        order = sorted(groups)
        rng.shuffle(order)
        taken = 0
        for g in order:
            if taken >= target:
                break
            val.update(groups[g])
            taken += len(groups[g])
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


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["relative_path"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing CSVs (only do this on purpose; experiments rely on them)")
    args = parser.parse_args()

    out_dir = paths.SPLITS_DIR
    outputs = {name: out_dir / name for name in
               ["validation.csv", "ssl_training_data.csv", "labeled_set.csv", "class_mapping.csv"]}
    existing = [p.name for p in outputs.values() if p.exists()]
    if existing and not args.overwrite:
        sys.exit(f"ERROR: {existing} already exist in {out_dir}. They define the fixed subsets; "
                 "rerun with --overwrite only if you really want new ones.")

    # ---- 1. Inventory and basic checks ------------------------------------
    print(f"Reading {paths.SOC40_DIR}")
    train = list_split(paths.SOC40_TRAIN_DIR)
    test = list_split(paths.SOC40_TEST_DIR)
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

    # ---- 2-4. Subsets -------------------------------------------------------
    rng = random.Random(SEED)
    val_paths = pick_validation(train, rng)
    validation = [r for r in train if r["relative_path"] in val_paths]
    ssl = [r for r in train if r["relative_path"] not in val_paths]
    labeled_paths = pick_labeled(ssl, rng)
    labeled = [r for r in ssl if r["relative_path"] in labeled_paths]

    # ---- 5. Leakage checks (fail loudly) -----------------------------------
    test_ids = {r["image_id"] for r in test}
    val_ids = {r["image_id"] for r in validation}
    ssl_ids = {r["image_id"] for r in ssl}
    lab_ids = {r["image_id"] for r in labeled}
    checks = {
        "validation and SSL pool share no image": not (val_ids & ssl_ids),
        "validation and labeled set share no image": not (val_ids & lab_ids),
        "labeled set is inside the SSL pool": lab_ids <= ssl_ids,
        "validation + SSL pool = all of train": len(val_ids) + len(ssl_ids) == len(train),
        "no test image in any CSV": not (test_ids & (val_ids | ssl_ids | lab_ids)),
        "every CSV path comes from train/": all(r["relative_path"].startswith("train/")
                                               for r in validation + ssl + labeled),
        "validation and SSL pool share no pose group":
            not ({r["pose_group"] for r in validation} & {r["pose_group"] for r in ssl}),
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
    print(f"\nvalidation.csv        {len(validation):6d} images ({len(validation) / len(train):.1%} of train)")
    print(f"ssl_training_data.csv {len(ssl):6d} images")
    print(f"labeled_set.csv       {len(labeled):6d} images ({len(labeled) / len(ssl):.1%} of SSL pool)")
    print(f"test (not in any CSV) {len(test):6d} images")
    print(f"Per class: validation {min(v.values())}-{max(v.values())}, "
          f"SSL pool {min(s.values())}-{max(s.values())}, labeled {min(l.values())}-{max(l.values())}")
    print(f"Wrote CSVs to {out_dir}  (paths in CSVs are relative to paths.SOC40_DIR)")


if __name__ == "__main__":
    main()
