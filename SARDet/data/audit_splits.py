"""Audit the subsets for leakage and print the numbers that go into the paper's data section.

Checks (fails loudly): pool + excluded = official train; pool and excluded disjoint; labeled sets nested and
inside the pool; no training-side group in official val/test; val_select inside official val; manifests
match data/hashes.json. Reports: duplicate and parent-sharing groups that straddle the OFFICIAL splits
(train-val, train-test, val-test), per source; sizes; box-size histogram after the 512 resize per source.

    python data/audit_splits.py            -> prints and writes data/audit_report.json
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import manifests, metadata, paths  # noqa: E402
from shared.run_record import sha256_of_file  # noqa: E402


def main():
    rows = metadata.read_metadata()
    by_split = defaultdict(list)
    for r in rows:
        by_split[r["official_split"]].append(r)
    report = {}

    # ---- official-release leakage (what the benchmark itself contains) ----------------------------------
    members = defaultdict(list)
    for r in rows:
        members[r["group"]].append(r)
    straddle = Counter()
    straddle_by_source = defaultdict(Counter)
    for g, ms in members.items():
        splits = {m["official_split"] for m in ms}
        if len(splits) > 1:
            key = "+".join(sorted(splits))
            straddle[key] += 1
            straddle_by_source[ms[0]["source"]][key] += 1
    md5_groups = defaultdict(set)
    for r in rows:
        md5_groups[r["md5"]].add(r["official_split"])
    px_groups = defaultdict(set)
    for r in rows:
        px_groups[r["pixel_md5"]].add(r["official_split"])
    scene_splits = defaultdict(set)
    for r in rows:
        if r.get("scene"):
            scene_splits[r["scene"]].add(r["official_split"])
    hrsid_scenes = [s for s in scene_splits if s.startswith("HRSID")]
    report["official_release"] = {
        "hrsid_scenes": len(hrsid_scenes),
        "hrsid_scenes_straddling_official_splits": sum(1 for s in hrsid_scenes if len(scene_splits[s]) > 1),
        "note_hrsid": "HRSID scene id is recorded, not a grouping unit (tiles overlap <= 25 %; grouping by scene would remove "
                      "3,638 of 3,642 train tiles). Scene straddling is a benchmark property shared by every method.",
        "images": {s: len(v) for s, v in by_split.items()},
        "groups_straddling_official_splits": dict(straddle),
        "straddling_by_source": {s: dict(c) for s, c in straddle_by_source.items()},
        "file_md5_groups_straddling": sum(1 for v in md5_groups.values() if len(v) > 1),
        "pixel_md5_groups_straddling": sum(1 for v in px_groups.values() if len(v) > 1),
        "sources_without_parent_id": ["MSAR", "OGSOD", "SSDD", "Shipdataset", "SADD", "SIVED", "AIR_SARShip-2"],
    }

    # ---- our subsets ----------------------------------------------------------------------------------------
    pool = manifests.read(manifests.ssl_pool_csv())
    excluded = manifests.read(paths.DATA_DIR / "excluded_groups.csv")
    train_names = {r["file_name"] for r in by_split["train"]}
    pool_names, excl_names = {r["file_name"] for r in pool}, {r["file_name"] for r in excluded}
    assert pool_names.isdisjoint(excl_names), "pool and excluded overlap"
    assert pool_names | excl_names == train_names, "pool + excluded != official train"
    per_split = {}
    for s in [1, 2, 3]:
        audit = manifests.quick_audit(s)
        with open(paths.DATA_DIR / f"split{s}" / "counts.json", encoding="utf-8") as f:
            counts = json.load(f)
        lab = {b: manifests.read(manifests.labeled_csv(b, s)) for b in manifests.LABELED_CSV}
        per_split[s] = {"counts": audit["counts"], "per_source": {b: dict(Counter(r["source"] for r in lab[b])) for b in lab},
                        "groups": {b: len({r["group"] for r in lab[b]}) for b in lab}, "quota_vs_actual": counts["per_source"]}
    overlap_12 = len({r["file_name"] for r in manifests.read(manifests.labeled_csv("10pct", 1))} &
                     {r["file_name"] for r in manifests.read(manifests.labeled_csv("10pct", 2))})
    with open(paths.DATA_DIR / "hashes.json", encoding="utf-8") as f:
        hashes = json.load(f)
    mismatched = [n for n, h in hashes.items() if sha256_of_file(paths.DATA_DIR / n) != h]
    assert not mismatched, f"manifests changed since hashes.json: {mismatched}"
    subsets = manifests.eval_subsets()
    report["subsets"] = {"ssl_pool": len(pool), "excluded_from_training": len(excluded),
                         "excluded_by_source": dict(Counter(r["source"] for r in excluded)),
                         "pool_by_source": dict(Counter(r["source"] for r in pool)), "splits": per_split,
                         "labeled_10pct_overlap_split1_split2": overlap_12,
                         "val_select": len(manifests.read(paths.VAL_SELECT_CSV)),
                         "val_clean": f"{len(subsets['val_clean'])}/{len(subsets['val'])}",
                         "test_clean": f"{len(subsets['test_clean'])}/{len(subsets['test'])}",
                         "val_test_straddling_groups": subsets["val_test_straddling_groups"], "hashes_ok": True}

    # ---- box sizes after the detector resize, per source -----------------------------------------------------
    from shared.coco_data import load_annotations
    size = 512
    hist = defaultdict(Counter)
    edges = [0, 8, 16, 32, 64, 128, 10 ** 9]
    src_of = {(r["official_split"], r["image_id"]): r["source"] for r in rows}
    ann = load_annotations("train")
    for im in ann["raw"]["images"]:
        s = min(size / min(im["width"], im["height"]), size / max(im["width"], im["height"]))
        for a in ann["anns"].get(im["id"], []):
            side = (a["bbox"][2] * a["bbox"][3]) ** 0.5 * s
            for lo, hi in zip(edges, edges[1:]):
                if lo <= side < hi:
                    hist[src_of[("train", im["id"])]][f"[{lo},{hi})" if hi < 10 ** 9 else f">={lo}"] += 1
                    break
    report["train_box_side_px_after_512_resize_by_source"] = {k: dict(v) for k, v in hist.items()}

    with open(paths.DATA_DIR / "audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "train_box_side_px_after_512_resize_by_source"}, indent=1)[:6000])
    print("ok: all invariants hold; wrote data/audit_report.json")


if __name__ == "__main__":
    main()
