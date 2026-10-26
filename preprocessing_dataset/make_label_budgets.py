"""Make the 1% and 10% labeled sets around the existing 3% labeled set (label-budget study).

Run from the repository root:
    python preprocessing_dataset/make_label_budgets.py

Budgets are defined like the existing labeled set: per class, a fraction of that class's images in
the SSL pool (ssl_training_data.csv). Nischal's decision, 2026-10-26 (NGC suggestion: 1%, 3%, 10%).
    labeled_set_1pct.csv   round(0.01 * n) images per class, drawn from inside labeled_set.csv
    labeled_set.csv        the existing 3% set (round(0.03 * n) per class); NOT changed by this script
    labeled_set_10pct.csv  labeled_set.csv plus round(0.10 * n) - (3% count) more images per class
                           from the rest of the SSL pool
So 1% is inside 3%, which is inside 10%: a larger budget only adds labels. Every set lies inside the
SSL pool, so it shares no image or acquisition group with validation (checked again below and by
audit_splits before every fine-tune).
"""
import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so "import paths" works
import paths  # noqa: E402

SEED = 1  # separate from the split's SEED 0, so the extra draws do not repeat its random sequence
BUDGETS = {"labeled_set_1pct.csv": 0.01, "labeled_set_10pct.csv": 0.10}
BASE_FRACTION = 0.03  # the fraction labeled_set.csv was made with (preprocessing_dataset.py)


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits-dir", type=Path, default=paths.SPLITS_DIR,
                        help="folder with ssl_training_data.csv, labeled_set.csv, validation.csv")
    parser.add_argument("--overwrite", action="store_true", help="replace existing budget CSVs (on purpose only)")
    args = parser.parse_args()
    d = args.splits_dir

    existing = [f for f in BUDGETS if (d / f).exists()]
    if existing and not args.overwrite:
        sys.exit(f"ERROR: {existing} already exist in {d}; rerun with --overwrite only on purpose.")

    columns, ssl = read_rows(d / "ssl_training_data.csv")
    _, base = read_rows(d / "labeled_set.csv")
    _, validation = read_rows(d / "validation.csv")

    pool_by_class, base_by_class = defaultdict(list), defaultdict(list)
    for r in ssl:
        pool_by_class[r["class_name"]].append(r)
    for r in base:
        base_by_class[r["class_name"]].append(r)
    for cls, rows in pool_by_class.items():  # labeled_set.csv must be what this rule says it is
        if len(base_by_class[cls]) != round(BASE_FRACTION * len(rows)):
            sys.exit(f"ERROR: labeled_set.csv has {len(base_by_class[cls])} images of {cls}, "
                     f"expected round({BASE_FRACTION} x {len(rows)})")

    rng = random.Random(SEED)
    out = {f: [] for f in BUDGETS}
    for cls in sorted(pool_by_class):
        n = len(pool_by_class[cls])
        in_base = sorted(base_by_class[cls], key=lambda r: r["relative_path"])
        base_paths = {r["relative_path"] for r in in_base}
        rest = sorted((r for r in pool_by_class[cls] if r["relative_path"] not in base_paths),
                      key=lambda r: r["relative_path"])
        out["labeled_set_1pct.csv"] += rng.sample(in_base, round(0.01 * n))
        out["labeled_set_10pct.csv"] += in_base + rng.sample(rest, round(0.10 * n) - len(in_base))

    # ---- Checks (fail loudly, write nothing) -----------------------------------------
    paths_of = lambda rows: {r["relative_path"] for r in rows}  # noqa: E731
    acq = lambda rows: {r["acquisition_group"] for r in rows}  # noqa: E731
    p1, p3, p10 = paths_of(out["labeled_set_1pct.csv"]), paths_of(base), paths_of(out["labeled_set_10pct.csv"])
    checks = {
        "1% inside 3% inside 10%": p1 <= p3 <= p10,
        "every set inside the SSL pool": p10 <= paths_of(ssl),
        "no image shared with validation": not (p10 & paths_of(validation)),
        "no acquisition group shared with validation": not (acq(out["labeled_set_10pct.csv"]) & acq(validation)),
        "all 40 classes in every set": all(len({r["class_name"] for r in rows}) == 40 for rows in out.values()),
        "no duplicate paths": all(len(paths_of(rows)) == len(rows) for rows in out.values()),
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        sys.exit("ERROR: a check failed; no CSVs were written.")

    for f, rows in out.items():
        with open(d / f, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=columns)
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: r["relative_path"]))
    for f, rows in [("labeled_set_1pct.csv", out["labeled_set_1pct.csv"]), ("labeled_set.csv", base),
                    ("labeled_set_10pct.csv", out["labeled_set_10pct.csv"])]:
        c = Counter(r["class_name"] for r in rows)
        print(f"{f:24s} {len(rows):5d} images ({len(rows) / len(ssl):.1%} of SSL pool), "
              f"{min(c.values())}-{max(c.values())} per class")
    print(f"Wrote {list(BUDGETS)} to {d}")


if __name__ == "__main__":
    main()
