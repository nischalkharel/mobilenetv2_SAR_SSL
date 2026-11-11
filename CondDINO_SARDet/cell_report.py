"""Cell report: how a CondDINO cell definition partitions the SSL pool, and how much class information a cell holds.

CPU only. THIS SCRIPT READS THE LABELS (official train.json) FOR THE REPORT ONLY; training never does (except the
oracle_class variant, which is the labelled ceiling by construction). DESIGN.md section 5: on SARDet-100K the
source IS the class for most images, so the cells are EXPECTED to leak class information; the report measures it
instead of hiding it. There are no pass/fail thresholds; a WARNING is printed when any cell is > 90 % one class.

For the pool and one cell definition it writes:
  - the class-vs-cell contingency table (per cell: images containing each class; the dominant class = the class
    with the most boxes in the image, ties -> lower category id; images without boxes count as "none"),
  - per-cell purity (share of the cell's most common dominant class) and the number of distinct class sets,
  - majority-vote accuracy of the dominant class from the cell alone, vs scrambled cells of the same sizes
    (finite-sample floor), vs chance (1 / number of classes) and vs the majority-class baseline,
  - mutual information I(cell; class-set) and I(cell; dominant class) in bits, vs scrambled cells,
  - cell sizes (min / median / max).

    python CondDINO_SARDet/cell_report.py --cells source_pol                      # default: split of SAR_SPLIT
    python CondDINO_SARDet/cell_report.py --cells oracle_class
    python CondDINO_SARDet/cell_report.py --cells radiometric --workers 8         # builds / reuses the k-means cache
    python CondDINO_SARDet/cell_report.py --cells perm --permute-seed 0

Outputs: outputs/cell_report/<variant>_split<s>.md / .csv / .json (the pool is the same for every split, so the
split only labels the file; the labeled sets are not involved).
"""
import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared import paths  # noqa: E402
from shared.ssl_data import load_pool_rows  # noqa: E402

sys.path.insert(0, str(HERE))
from cells import VARIANTS, assign_variant, output_dir  # noqa: E402
from cond_center import permute_ids  # noqa: E402

PURITY_WARNING = 0.90
NONE_CLASS = "none"


def entropy_bits(counts) -> float:
    counts = np.asarray(counts, dtype=float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def mutual_information_bits(a: np.ndarray, b: np.ndarray) -> float:
    """I(A; B) = H(A) + H(B) - H(A, B), plug-in estimate, for two integer-coded variables."""
    joint = Counter(zip(a.tolist(), b.tolist()))
    return entropy_bits(np.bincount(a)) + entropy_bits(np.bincount(b)) - entropy_bits(list(joint.values()))


def majority_vote_accuracy(groups: np.ndarray, classes: np.ndarray) -> float:
    correct = 0
    for g in np.unique(groups):
        correct += np.bincount(classes[groups == g]).max()
    return float(correct / len(classes))


def image_labels(rows):
    """LABELS READ HERE (report only). Returns (dominant class id per image, class-set key per image, class names)."""
    from shared.coco_data import load_annotations
    ann = load_annotations("train")
    names = list(paths.CLASS_NAMES) + [NONE_CLASS]
    dominant, class_sets, presence = [], [], []
    for r in rows:
        im = ann["images"][r["file_name"]]
        cats = Counter(a["category_id"] for a in ann["anns"].get(im["id"], []))
        if cats:
            dom = min(cats, key=lambda c: (-cats[c], c))
            dominant.append(dom)
            class_sets.append("+".join(paths.CLASS_NAMES[c] for c in sorted(cats)))
        else:
            dominant.append(len(paths.CLASS_NAMES))
            class_sets.append(NONE_CLASS)
        presence.append(sorted(cats))
    return np.array(dominant, dtype=np.int64), class_sets, names, presence


def report(variant: str, rows, permute_seed: int = 0, num_workers: int = 0, scramble_seed: int = 0, verbose=True):
    """(per-cell rows, summary dict). Labels are read inside image_labels() only."""
    ids, cell_names, info = assign_variant(variant, rows, permute_seed=permute_seed, num_workers=num_workers, verbose=verbose)
    y, class_sets, class_names, presence = image_labels(rows)
    set_names = sorted(set(class_sets))
    y_set = np.array([set_names.index(s) for s in class_sets], dtype=np.int64)
    scrambled = permute_ids(ids, scramble_seed)
    n_cells = len(cell_names)

    per_cell = []
    for cid in range(n_cells):
        m = ids == cid
        n_img = int(m.sum())
        if n_img == 0:
            per_cell.append({"cell_id": cid, "cell": cell_names[cid], "num_images": 0, "num_class_sets": 0, "purity": 0.0,
                             "largest_class": "", **{f"n_{c}": 0 for c in class_names}})
            continue
        dom_counts = np.bincount(y[m], minlength=len(class_names))
        largest = int(dom_counts.argmax())
        contains = Counter(c for i in np.flatnonzero(m) for c in presence[i])
        row = {"cell_id": cid, "cell": cell_names[cid], "num_images": n_img, "num_class_sets": len({class_sets[i] for i in np.flatnonzero(m)}),
               "purity": round(float(dom_counts[largest] / n_img), 4), "largest_class": class_names[largest]}
        row.update({f"n_{c}": int(contains.get(k, 0)) for k, c in enumerate(paths.CLASS_NAMES)})
        row[f"n_{NONE_CLASS}"] = int(dom_counts[len(paths.CLASS_NAMES)])
        per_cell.append(row)
    sizes = np.array([c["num_images"] for c in per_cell])
    nonempty = sizes > 0
    purity = np.array([c["purity"] for c in per_cell])[nonempty]
    impure_warning = [c for c in per_cell if c["num_images"] > 0 and c["purity"] > PURITY_WARNING]
    dom_share = np.bincount(y, minlength=len(class_names)) / len(y)
    summary = {
        "variant": variant, "what": VARIANTS[variant]["what"], "label_use_in_training": VARIANTS[variant]["label_use"],
        "labels_read_for_this_report": "yes (official train.json; dominant class and class set per image)",
        "split": paths.SPLIT, "num_images": len(rows), "num_cells": n_cells, "num_nonempty_cells": int(nonempty.sum()),
        "cell_sizes": {"min": int(sizes[nonempty].min()), "median": float(np.median(sizes[nonempty])), "max": int(sizes.max())},
        "classes": class_names, "num_class_sets_in_pool": len(set_names),
        "purity_dominant_class": {"median": float(np.median(purity)), "max": float(purity.max()),
                                  "image_weighted_mean": round(float((purity * sizes[nonempty]).sum() / sizes.sum()), 4)},
        "cells_above_90pct_one_class": [{"cell": c["cell"], "num_images": c["num_images"], "class": c["largest_class"], "purity": c["purity"]}
                                        for c in impure_warning],
        "majority_vote_accuracy_dominant_class": {
            "from_cell": round(majority_vote_accuracy(ids, y), 4),
            "from_scrambled_cells_same_sizes": round(majority_vote_accuracy(scrambled, y), 4),
            "majority_class_baseline": round(float(dom_share.max()), 4),
            "chance_uniform": round(1 / len(paths.CLASS_NAMES), 4)},
        "mutual_information_bits": {
            "cell_classset": round(mutual_information_bits(ids, y_set), 4),
            "scrambled_cell_classset": round(mutual_information_bits(scrambled, y_set), 4),
            "classset_entropy": round(entropy_bits(np.bincount(y_set)), 4),
            "cell_dominant": round(mutual_information_bits(ids, y), 4),
            "scrambled_cell_dominant": round(mutual_information_bits(scrambled, y), 4),
            "dominant_entropy": round(entropy_bits(np.bincount(y)), 4),
            "cell_entropy": round(entropy_bits(sizes), 4)},
        "dominant_class_share_in_pool": {c: round(float(s), 4) for c, s in zip(class_names, dom_share)},
        "cell_info": info, "scramble_seed": scramble_seed,
        "note": ("no pass/fail thresholds: DESIGN.md section 5 expects these cells to carry class information on SARDet-100K; "
                 "the report measures it. Scrambled cells = same sizes, random membership (finite-sample floor)."),
        "written": datetime.now().isoformat(timespec="seconds"),
    }
    return per_cell, summary


def summary_lines(s: dict, stem: str) -> list:
    mv, mi, cs, pu = (s["majority_vote_accuracy_dominant_class"], s["mutual_information_bits"], s["cell_sizes"], s["purity_dominant_class"])
    lines = [f"# CondDINO cell report: {s['variant']} (split {s['split']} pool, {s['num_images']} images)", "",
             f"- cells: {s['what']}",
             f"- label use in TRAINING: {s['label_use_in_training']}; labels read for this report: {s['labels_read_for_this_report']}",
             f"- cells {s['num_cells']} ({s['num_nonempty_cells']} non-empty); images per cell min / median / max: "
             f"{cs['min']} / {cs['median']:.0f} / {cs['max']}",
             f"- purity (share of the cell's most common dominant class): median {100 * pu['median']:.1f} %, max {100 * pu['max']:.1f} %, "
             f"image-weighted mean {100 * pu['image_weighted_mean']:.1f} %",
             f"- majority vote on the dominant class from the cell alone: {100 * mv['from_cell']:.2f} % "
             f"(scrambled cells of the same sizes {100 * mv['from_scrambled_cells_same_sizes']:.2f} %, majority-class baseline "
             f"{100 * mv['majority_class_baseline']:.2f} %, uniform chance {100 * mv['chance_uniform']:.2f} %)",
             f"- mutual information with the class SET, bits: cell {mi['cell_classset']:.3f} (scrambled {mi['scrambled_cell_classset']:.3f}; "
             f"class-set entropy {mi['classset_entropy']:.3f}, {s['num_class_sets_in_pool']} sets)",
             f"- mutual information with the DOMINANT class, bits: cell {mi['cell_dominant']:.3f} (scrambled {mi['scrambled_cell_dominant']:.3f}; "
             f"class entropy {mi['dominant_entropy']:.3f}; cell entropy {mi['cell_entropy']:.3f})",
             f"- dominant-class shares in the pool: {s['dominant_class_share_in_pool']}"]
    if s["cells_above_90pct_one_class"]:
        hot = ", ".join(f"{c['cell']} ({c['num_images']} images, {100 * c['purity']:.1f} % {c['class']})" for c in s["cells_above_90pct_one_class"])
        lines.append(f"- WARNING: {len(s['cells_above_90pct_one_class'])} cell(s) are > {100 * PURITY_WARNING:.0f} % one class: {hot}")
    else:
        lines.append(f"- no cell is > {100 * PURITY_WARNING:.0f} % one class")
    if s["variant"] == "oracle_class":
        lines.append("- oracle_class is the LABEL-LEAK CEILING: its cells are the labels by construction; never a method")
    lines += ["", s["note"], f"Per-cell table: {stem}.csv. Written {s['written']}."]
    return lines


def write_report(variant: str, per_cell, summary, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_cell[0]))
        w.writeheader()
        w.writerows(per_cell)
    lines = summary_lines(summary, stem)
    (out_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description="CondDINO cell report (CPU; reads labels for the report only)")
    ap.add_argument("--cells", default="source_pol", choices=sorted(VARIANTS))
    ap.add_argument("--permute-seed", type=int, default=0, help="perm variant: the permutation seed of the config")
    ap.add_argument("--scramble-seed", type=int, default=0, help="seed of the scrambled comparison cells")
    ap.add_argument("--workers", type=int, default=0, help="radiometric variant: loader workers for the histogram pass")
    ap.add_argument("--limit", type=int, default=0, help="first N pool rows only (checks)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default outputs/cell_report)")
    a = ap.parse_args(argv)
    rows = load_pool_rows(limit=a.limit or None)
    per_cell, s = report(a.cells, rows, permute_seed=a.permute_seed, num_workers=a.workers, scramble_seed=a.scramble_seed)
    out_dir = a.out or output_dir("cell_report")
    stem = f"{a.cells}_split{paths.SPLIT}" + (f"_first{a.limit}" if a.limit else "")
    lines = write_report(a.cells, per_cell, s, out_dir, stem)
    print("\n".join(lines))
    print(f"\nWritten to {out_dir}: {stem}.csv, {stem}.md, {stem}.json")
    return s


if __name__ == "__main__":
    main()
