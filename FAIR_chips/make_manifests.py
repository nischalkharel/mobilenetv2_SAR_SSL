"""Draw the fixed subsets of the FAIR-CSAR chip study from the annotation and metadata files (DESIGN.md section 2).

    python FAIR_chips/make_manifests.py [--overwrite]

The slices are a 50 % overlapping grid in both directions, so one physical object is annotated in up to four slices.
Everything labeled is therefore counted in OBJECTS, not annotations: annotations of the same scene and class whose
scene coordinates (METAXmls patch offsets + slice position) agree within a few pixels are one object; its OWNER copy
is the annotation farthest from its slice's border (the most complete view), and only that copy becomes a chip.

Writes (all committed, under FAIR_chips/data/):
    objects.csv                 one row per physical object: gid, split, scene, class_name, label, n_copies, owner_slice,
                                owner_obj_id, difficult, box_long, border_margin
    slices.csv                  every slice: slice, split, scene, n_ann (annotations), n_obj (objects it owns), role
                                (ssl_pool | select | test), r0, a0, az_per_px (scene frame)
    select.csv                  10 % of the trainval slices (per scene, seed 0): per-epoch model selection only;
                                never in the SSL pool, never in a labeled set
    ssl_pool.csv                trainval minus select: the unlabeled SSL pool, the same for every split
    ssl_tiles.csv               the pool cut 2 x 2 into 512-px tiles, exact and near duplicates (grid-aligned
                                neighbours) removed: slice, tile, gx, gy (the fair_tiles SSL pool)
    split{1,2,3}/labeled_{1,3,10}pct.csv   nested labeled slice sets drawn inside the SSL pool, seed = split - 1
    split{s}/counts.json        owned chips per class per budget, targets, floor top-ups
    hashes.json                 SHA-256 of every file above

Drawing unit = slice: every object a chosen slice OWNS is labeled. Budget b: slices in the shuffled order are taken
until their owned objects reach b x (objects owned by pool slices); then any class with fewer than LABEL_FLOOR objects
is topped up with the next slices in the same order that own it. The order continues for the next budget, so 1 % is
inside 3 % is inside 10 %. The test split is the dataset's own.
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
SAME_OBJECT_PX = 4.0        # two annotations closer than this (scene frame, PNG px) with the same class are one object
TILE = 512
TILE_DUP_PX = 128           # a tile whose origin is within this of an already kept tile's origin is a duplicate (overlap > 75 %)


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


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]; i = self.p[i]
        return i

    def union(self, i, j):
        self.p[self.find(i)] = self.find(j)


def group_objects(anns):
    """anns: list of dicts with scene, class_name, gx, gy. Returns a group id per annotation (same scene and class,
    chained within SAME_OBJECT_PX)."""
    uf = UnionFind(len(anns))
    cell = defaultdict(list)
    q = SAME_OBJECT_PX
    for i, a in enumerate(anns):
        cell[(a["scene"], a["class_name"], int(a["gx"] // q), int(a["gy"] // q))].append(i)
    for i, a in enumerate(anns):
        cx, cy = int(a["gx"] // q), int(a["gy"] // q)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cell.get((a["scene"], a["class_name"], cx + dx, cy + dy), ()):
                    if j > i and abs(anns[j]["gx"] - a["gx"]) < q and abs(anns[j]["gy"] - a["gy"]) < q:
                        uf.union(i, j)
    return [uf.find(i) for i in range(len(anns))]


def draw_labeled(pool, seed):
    """pool: [slice dicts with 'owned' Counter]. Yields (budget, chosen slices, counts), nested."""
    order = list(pool)
    random.Random(seed).shuffle(order)
    total = sum(sum(s["owned"].values()) for s in pool)
    chosen, counts, remaining = [], Counter(), order
    for b, frac in fp.BUDGETS.items():
        target = round(frac * total)
        while sum(counts.values()) < target and remaining:
            s = remaining.pop(0); chosen.append(s); counts.update(s["owned"])
        topped = Counter()
        for c in fp.CLASS_NAMES:
            while counts[c] < fp.LABEL_FLOOR:
                k = next((i for i, s in enumerate(remaining) if s["owned"].get(c)), None)
                if k is None:
                    break
                s = remaining.pop(k); chosen.append(s); counts.update(s["owned"]); topped[c] += s["owned"][c]
        yield b, list(chosen), {"target_chips": target, "chips": sum(counts.values()), "slices": len(chosen),
                                "per_class": {c: counts[c] for c in fp.CLASS_NAMES}, "floor_top_up_chips": dict(topped),
                                "classes_below_floor": [c for c in fp.CLASS_NAMES if counts[c] < fp.LABEL_FLOOR]}


def unique_tiles(pool_slices):
    """2 x 2 tiles of every pool slice, keeping one tile per scene position (origins within TILE_DUP_PX are duplicates)."""
    tiles = []
    for s in pool_slices:
        fr = {"r0": s["r0"], "a0": s["a0"], "range_per_px": s["range_per_px"], "az_per_px": s["az_per_px"]}
        for k in range(4):
            gx, gy = fa.scene_xy(fr, (k % 2) * TILE, (k // 2) * TILE)
            tiles.append({"slice": s["slice"], "tile": k, "scene": s["scene"], "gx": round(gx, 1), "gy": round(gy, 1)})
    kept, seen = [], defaultdict(list)
    for t in sorted(tiles, key=lambda t: (t["scene"], t["slice"], t["tile"])):
        cx, cy = int(t["gx"] // TILE_DUP_PX), int(t["gy"] // TILE_DUP_PX)
        dup = any(abs(u["gx"] - t["gx"]) < TILE_DUP_PX and abs(u["gy"] - t["gy"]) < TILE_DUP_PX
                  for dx in (-1, 0, 1) for dy in (-1, 0, 1) for u in seen.get((t["scene"], cx + dx, cy + dy), ()))
        if not dup:
            kept.append(t); seen[(t["scene"], cx, cy)].append(t)
    return kept, len(tiles)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if fp.SLICES_CSV.exists() and not a.overwrite:
        raise SystemExit(f"{fp.SLICES_CSV} exists; pass --overwrite to redraw (the committed manifests are the study's fixed sets)")

    # ---- every annotation in scene coordinates ----
    slices, anns = [], []
    for split in fp.SPLITS:
        for name in fa.slice_names(split):
            fr = fa.frame(split, name)
            obs = fa.objects(split, name)
            slices.append({"slice": name, "split": split, "scene": fa.scene_of(name), "n_ann": len(obs), "role": "test" if split == "test" else "ssl_pool",
                           "r0": fr["r0"], "a0": fr["a0"], "range_per_px": round(fr["range_per_px"], 6), "az_per_px": round(fr["az_per_px"], 6)})
            for o in obs:
                gx, gy = fa.scene_xy(fr, o["cx"], o["cy"])
                anns.append({**o, "split": split, "slice": name, "scene": slices[-1]["scene"], "gx": gx, "gy": gy,
                             "border_margin": min(o["cx"], o["cy"], fr["width"] - o["cx"], fr["height"] - o["cy"])})
    print(f"{len(slices)} slices, {len(anns)} annotations", flush=True)

    # ---- physical objects: group the copies, choose the owner copy ----
    gid = group_objects(anns)
    groups = defaultdict(list)
    for i, g in enumerate(gid):
        groups[g].append(anns[i])
    objects = []
    for g, copies in groups.items():
        if len({c["split"] for c in copies}) > 1:
            raise ValueError(f"object copies in two splits: {[(c['split'], c['slice']) for c in copies]}")
        own = max(copies, key=lambda c: (c["border_margin"], -c["difficult"], c["slice"]))
        objects.append({"gid": len(objects), "split": own["split"], "scene": own["scene"], "class_name": own["class_name"], "label": own["label"],
                        "n_copies": len(copies), "owner_slice": own["slice"], "owner_obj_id": own["obj_id"], "difficult": own["difficult"],
                        "box_long": round(own["long"], 1), "border_margin": round(own["border_margin"], 1)})
    objects.sort(key=lambda o: (o["split"], o["owner_slice"], o["owner_obj_id"]))
    for i, o in enumerate(objects):
        o["gid"] = i
    owned = defaultdict(Counter)
    for o in objects:
        owned[o["owner_slice"]][o["class_name"]] += 1
    for s in slices:
        s["owned"] = owned.get(s["slice"], Counter()); s["n_obj"] = sum(s["owned"].values())
    cp = Counter(o["n_copies"] for o in objects)
    print(f"{len(objects)} physical objects (copies per object: {dict(sorted(cp.items()))}); "
          f"trainval {sum(o['split'] == 'trainval' for o in objects)}, test {sum(o['split'] == 'test' for o in objects)}", flush=True)

    # ---- roles ----
    trainval = [s for s in slices if s["split"] == "trainval"]
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
    tiles, n_all_tiles = unique_tiles(pool)
    print(f"select {len(select)} slices ({sum(s['n_obj'] for s in select)} owned objects), SSL pool {len(pool)} slices "
          f"({sum(s['n_obj'] for s in pool)} owned objects), tiles {len(tiles)} unique of {n_all_tiles}", flush=True)

    cols = ["slice", "split", "scene", "n_ann", "n_obj", "role", "r0", "a0", "range_per_px", "az_per_px"]
    write_csv(fp.SLICES_CSV, sorted(slices, key=lambda s: (s["split"], s["slice"])), cols)
    write_csv(fp.SELECT_CSV, sorted(select, key=lambda s: s["slice"]), cols)
    write_csv(fp.SSL_POOL_CSV, sorted(pool, key=lambda s: s["slice"]), cols)
    write_csv(fp.SSL_TILES_CSV, tiles, ["slice", "tile", "scene", "gx", "gy"])
    write_csv(fp.OBJECTS_CSV, objects, ["gid", "split", "scene", "class_name", "label", "n_copies", "owner_slice", "owner_obj_id", "difficult", "box_long", "border_margin"])
    files = [fp.SLICES_CSV, fp.SELECT_CSV, fp.SSL_POOL_CSV, fp.SSL_TILES_CSV, fp.OBJECTS_CSV]
    for split in (1, 2, 3):
        counts = {}
        for b, chosen, c in draw_labeled(pool, seed=split - 1):
            p = fp.labeled_csv(split, b)
            write_csv(p, sorted(chosen, key=lambda s: s["slice"]), cols)
            files.append(p); counts[b] = c
            low = ", ".join(f"{k} {c['per_class'][k]}" for k in c["classes_below_floor"]) or "none"
            print(f"  split {split} {b:5s}: {c['slices']:4d} slices, {c['chips']:5d} objects (target {c['target_chips']}), "
                  f"floor top-ups {sum(c['floor_top_up_chips'].values())}, below floor: {low}", flush=True)
        cpath = fp.DATA_DIR / f"split{split}" / "counts.json"
        cpath.write_text(json.dumps({"seed": split - 1, "unit": "slice (owned objects)", "label_floor": fp.LABEL_FLOOR, **counts}, indent=1), encoding="utf-8")
        files.append(cpath)
    fp.HASHES_JSON.write_text(json.dumps({str(p.relative_to(fp.DATA_DIR)).replace("\\", "/"): sha256_file(p) for p in files}, indent=1), encoding="utf-8")
    print(f"wrote {len(files) + 1} files under {fp.DATA_DIR}")


if __name__ == "__main__":
    main()
