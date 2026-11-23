"""Draw the fixed subsets of the FAIR-CSAR chip study from the annotation files (DESIGN.md section 2).

    python FAIR_chips/make_manifests.py [--overwrite]

Writes (all committed, under FAIR_chips/data/):
    slices.csv                  every slice: slice, split, scene, n_obj, role (ssl_pool | select | test)
    select.csv                  10 % of the trainval slices (per scene, seed 0): per-epoch model selection only;
                                never in the SSL pool, never in a labeled set
    ssl_pool.csv                trainval minus select: the unlabeled SSL pool, the same for every split
    split{1,2,3}/labeled_{1,3,10}pct.csv   nested labeled slice sets drawn inside the SSL pool, seed = split - 1
    split{s}/counts.json        chips per class per budget, targets, floor top-ups
    hashes.json                 SHA-256 of every file above

Drawing unit = slice: every chip of a chosen slice is labeled (chips of one slice are never split between sets).
Budget b: slices in the shuffled order are taken until their chips reach b x (chips in the pool); then any class
with fewer than LABEL_FLOOR chips is topped up with the next slices in the same order that contain it. The order
continues for the next budget, so 1 % is inside 3 % is inside 10 %. The test split is the dataset's own.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fair_paths as fp  # noqa: E402
import fair_annotations as fa  # noqa: E402

SELECT_FRACTION = 0.10


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def draw_labeled(pool, seed):
    """pool: [slice dict with 'classes' Counter]. Returns {budget: [slices]} nested, and counts."""
    order = list(pool)
    random.Random(seed).shuffle(order)
    total = sum(s["n_obj"] for s in pool)
    chosen, counts, remaining, counts_out = [], Counter(), order, {}
    for b, frac in fp.BUDGETS.items():
        target = round(frac * total)
        while sum(counts.values()) < target and remaining:
            s = remaining.pop(0); chosen.append(s); counts.update(s["classes"])
        topped = Counter()
        for c in fp.CLASS_NAMES:
            while counts[c] < fp.LABEL_FLOOR:
                k = next((i for i, s in enumerate(remaining) if s["classes"].get(c)), None)
                if k is None:
                    break
                s = remaining.pop(k); chosen.append(s); counts.update(s["classes"]); topped[c] += s["classes"][c]
        counts_out[b] = {"target_chips": target, "chips": sum(counts.values()), "slices": len(chosen),
                         "per_class": {c: counts[c] for c in fp.CLASS_NAMES}, "floor_top_up_chips": dict(topped),
                         "classes_below_floor": [c for c in fp.CLASS_NAMES if counts[c] < fp.LABEL_FLOOR]}
        yield b, list(chosen), counts_out[b]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if fp.SLICES_CSV.exists() and not a.overwrite:
        raise SystemExit(f"{fp.SLICES_CSV} exists; pass --overwrite to redraw (the committed manifests are the study's fixed sets)")

    slices = []
    for split in fp.SPLITS:
        for name in fa.slice_names(split):
            obs = fa.objects(split, name)
            slices.append({"slice": name, "split": split, "scene": fa.scene_of(name), "n_obj": len(obs),
                           "classes": Counter(o["class_name"] for o in obs), "role": "test" if split == "test" else "ssl_pool"})
    trainval = [s for s in slices if s["split"] == "trainval"]
    print(f"{len(trainval)} trainval slices ({sum(s['n_obj'] for s in trainval)} objects), {len(slices) - len(trainval)} test slices")

    # selection slices: SELECT_FRACTION of each scene's slices, seed 0, at least one per scene with 5+ slices
    rng = random.Random(0)
    by_scene = defaultdict(list)
    for s in trainval:
        by_scene[s["scene"]].append(s)
    for scene, ss in sorted(by_scene.items()):
        rng.shuffle(ss)
        k = round(SELECT_FRACTION * len(ss)) if len(ss) >= 5 else 0
        for s in ss[:k]:
            s["role"] = "select"
    select = [s for s in trainval if s["role"] == "select"]
    pool = [s for s in trainval if s["role"] == "ssl_pool"]
    print(f"select {len(select)} slices ({sum(s['n_obj'] for s in select)} chips), SSL pool {len(pool)} slices ({sum(s['n_obj'] for s in pool)} chips)")

    cols = ["slice", "split", "scene", "n_obj", "role"]
    write_csv(fp.SLICES_CSV, sorted(slices, key=lambda s: (s["split"], s["slice"])), cols)
    write_csv(fp.SELECT_CSV, sorted(select, key=lambda s: s["slice"]), cols)
    write_csv(fp.SSL_POOL_CSV, sorted(pool, key=lambda s: s["slice"]), cols)
    files = [fp.SLICES_CSV, fp.SELECT_CSV, fp.SSL_POOL_CSV]
    for split in (1, 2, 3):
        counts = {}
        for b, chosen, c in draw_labeled(pool, seed=split - 1):
            p = fp.labeled_csv(split, b)
            write_csv(p, sorted(chosen, key=lambda s: s["slice"]), cols)
            files.append(p); counts[b] = c
            low = ", ".join(f"{k} {v['per_class'][k]}" for k, v in [(k, c) for k in c["classes_below_floor"]]) or "none"
            print(f"  split {split} {b:5s}: {c['slices']:4d} slices, {c['chips']:5d} chips (target {c['target_chips']}), "
                  f"floor top-ups {sum(c['floor_top_up_chips'].values())} chips, below floor: {low}")
        cp = fp.DATA_DIR / f"split{split}" / "counts.json"
        cp.write_text(json.dumps({"seed": split - 1, "unit": "slice", "label_floor": fp.LABEL_FLOOR, **counts}, indent=1), encoding="utf-8")
        files.append(cp)
    fp.HASHES_JSON.write_text(json.dumps({str(p.relative_to(fp.DATA_DIR)).replace("\\", "/"): sha256_file(p) for p in files}, indent=1), encoding="utf-8")
    print(f"wrote {len(files) + 1} files under {fp.DATA_DIR}")


if __name__ == "__main__":
    main()
