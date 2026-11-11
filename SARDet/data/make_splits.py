"""Draw the subsets of DESIGN.md section 2 from data/image_metadata.csv.

Writes (all committed):
    data/ssl_pool.csv            official train minus every group that touches official val or test (same for all splits)
    data/excluded_groups.csv     the excluded train images and why (group shared with val / test)
    data/split{1,2,3}/labeled_{1,3,10}pct.csv   nested labeled sets drawn inside the SSL pool, seed = split - 1
    data/split{s}/counts.json    quota vs actual per source
    data/val_select.csv          2,000 official-val images (stratified by source, seed 0) for per-epoch model selection
    data/eval_subsets.json       file names of official val / test, val_clean / test_clean, and their group ids
    data/hashes.json             SHA-256 of every file above

Drawing unit = group (same parent image or identical content). Per source, groups are shuffled once; the
first groups fill the 1 % quota, the same order continues to 3 % and 10 %, so the sets nest. Quotas are
fractions of the OFFICIAL TRAIN count of that source (not of the pool), rounded; a source whose groups are
large (HRSID scenes) overshoots by at most one group.

    python data/make_splits.py [--overwrite]
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import manifests, metadata, paths  # noqa: E402
from shared.run_record import sha256_of_file  # noqa: E402

BUDGETS = {"1pct": 0.01, "3pct": 0.03, "10pct": 0.10}
SPLITS = [1, 2, 3]
VAL_SELECT_SIZE = 2000
VAL_SELECT_SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out = paths.DATA_DIR
    if (out / "ssl_pool.csv").exists() and not args.overwrite:
        sys.exit("split files exist; use --overwrite")
    rows = metadata.read_metadata()
    by_split = defaultdict(list)
    for r in rows:
        by_split[r["official_split"]].append(r)
    groups_of = defaultdict(set)
    for r in rows:
        groups_of[r["official_split"]].add(r["group"])
    scored_groups = groups_of["val"] | groups_of["test"]

    # ---- SSL pool and exclusions --------------------------------------------------------------------------
    pool, excluded = [], []
    for r in by_split["train"]:
        if r["group"] in scored_groups:
            why = []
            if r["group"] in groups_of["val"]:
                why.append("val")
            if r["group"] in groups_of["test"]:
                why.append("test")
            excluded.append({**r, "shared_with": "+".join(why)})
        else:
            pool.append(r)
    pool.sort(key=lambda r: r["file_name"])
    manifests.write(out / "ssl_pool.csv", pool)
    with open(out / "excluded_groups.csv", "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=manifests.MANIFEST_COLUMNS + ["parent", "md5", "shared_with"], extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(excluded, key=lambda r: r["file_name"]))
    print(f"official train {len(by_split['train'])}: SSL pool {len(pool)}, excluded {len(excluded)} "
          f"(groups shared with val/test)")

    # ---- Labeled sets per split -----------------------------------------------------------------------------
    train_per_source = Counter(r["source"] for r in by_split["train"])
    pool_groups = defaultdict(lambda: defaultdict(list))        # source -> group -> rows
    for r in pool:
        pool_groups[r["source"]][r["group"]].append(r)
    for split in SPLITS:
        rng = random.Random(split - 1)
        split_dir = out / f"split{split}"
        split_dir.mkdir(exist_ok=True)
        chosen = {b: [] for b in BUDGETS}
        counts = {}
        for source in sorted(pool_groups):
            group_ids = sorted(pool_groups[source])
            rng.shuffle(group_ids)
            taken, k = [], 0
            counts[source] = {}
            for budget, frac in BUDGETS.items():                    # 1pct, 3pct, 10pct in order -> nested
                quota = round(frac * train_per_source[source])
                while len(taken) < quota and k < len(group_ids):
                    taken.extend(pool_groups[source][group_ids[k]])
                    k += 1
                chosen[budget].extend(taken)
                counts[source][budget] = {"quota": quota, "actual": len(taken), "groups": k}
        for budget in BUDGETS:
            rows_b = sorted(chosen[budget], key=lambda r: r["file_name"])
            manifests.write(split_dir / manifests.LABELED_CSV[budget], rows_b)
        totals = {b: len(chosen[b]) for b in BUDGETS}
        with open(split_dir / "counts.json", "w", encoding="utf-8") as f:
            json.dump({"seed": split - 1, "totals": totals, "per_source": counts}, f, indent=2)
        print(f"split {split}: " + ", ".join(f"{b} {n}" for b, n in totals.items()))

    # ---- val_select and evaluation subsets ------------------------------------------------------------------
    rng = random.Random(VAL_SELECT_SEED)
    val_rows = sorted(by_split["val"], key=lambda r: r["file_name"])
    per_source = defaultdict(list)
    for r in val_rows:
        per_source[r["source"]].append(r)
    val_select = []
    for source in sorted(per_source):
        n = max(1, round(VAL_SELECT_SIZE * len(per_source[source]) / len(val_rows)))
        val_select.extend(rng.sample(per_source[source], min(n, len(per_source[source]))))
    val_select.sort(key=lambda r: r["file_name"])
    manifests.write(out / "val_select.csv", val_select)
    train_groups = groups_of["train"]
    subsets = {
        "val": [r["file_name"] for r in val_rows],
        "test": [r["file_name"] for r in sorted(by_split["test"], key=lambda r: r["file_name"])],
        "val_clean": [r["file_name"] for r in val_rows if r["group"] not in train_groups],
        "test_clean": [r["file_name"] for r in sorted(by_split["test"], key=lambda r: r["file_name"]) if r["group"] not in train_groups],
        "val_groups": sorted(groups_of["val"]), "test_groups": sorted(groups_of["test"]),
        "val_test_straddling_groups": len(groups_of["val"] & groups_of["test"]),
        "definitions": {"val_clean / test_clean": "official images whose group (parent image or identical content) has no member in official train"},
    }
    with open(out / "eval_subsets.json", "w", encoding="utf-8") as f:
        json.dump(subsets, f, indent=1)
    print(f"val_select {len(val_select)}; val_clean {len(subsets['val_clean'])}/{len(subsets['val'])}; "
          f"test_clean {len(subsets['test_clean'])}/{len(subsets['test'])}; val-test straddling groups "
          f"{subsets['val_test_straddling_groups']}")
    files = ["image_metadata.csv", "ssl_pool.csv", "excluded_groups.csv", "val_select.csv", "eval_subsets.json"] + \
            [f"split{s}/{manifests.LABELED_CSV[b]}" for s in SPLITS for b in BUDGETS]
    with open(out / "hashes.json", "w", encoding="utf-8") as f:
        json.dump({name: sha256_of_file(out / name) for name in files}, f, indent=2)
    print("wrote data/hashes.json")


if __name__ == "__main__":
    main()
