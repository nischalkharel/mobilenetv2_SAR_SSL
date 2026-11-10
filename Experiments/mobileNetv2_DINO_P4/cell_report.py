"""Cell report: how P4's condition cells partition an SSL pool, and whether a cell could give the class away.

CPU only, seconds. This script reads the class column FOR THIS REPORT ONLY; training (train_ssl.py) never does.
A cell that held one or two vehicle types would be a label in disguise; the report shows, per cell, how many
classes it holds and the share of its largest class ("purity"), and for the whole pool two numbers that separate
"the cells carry class information" from "the scene carries class information":
  - majority vote: accuracy of predicting each image's class from its cell alone (the cell's most common class),
    next to the same number for scrambled cells of identical sizes (the finite-sample floor), for the scene alone,
    and chance (1 / number of classes);
  - mutual information I(cell; class) in bits, next to I(scene; class) and I(scrambled cell; class).
On ATRNet-STAR the scene carries some class information because bare_soil and woodland contain only 11 classes;
the question is whether the cells add anything beyond that.

Run from the repository root (SAR_SPLIT selects the split, as for every script in this repository):
    python Experiments/mobileNetv2_DINO_P4/cell_report.py                                  # split 1, cells of config_ssl.json
    SAR_SPLIT=2 python Experiments/mobileNetv2_DINO_P4/cell_report.py                      # split 2
    python Experiments/mobileNetv2_DINO_P4/cell_report.py --config Experiments/mobileNetv2_DINO_P4/config_ssl_3field.json
    python Experiments/mobileNetv2_DINO_P4/cell_report.py --fields scene band depression   # any field list

Writes outputs/cell_report/split<N>_<csv>_<config stem or fields>.csv (one row per cell), .md (the summary) and
.json, and prints the summary.
"""
import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

import paths
from cond_center import CELL_FIELDS, assign_cells, columns_read
from ssl_data import read_rows

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_ssl.json"


def entropy_bits(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def mutual_information_bits(a: np.ndarray, b: np.ndarray) -> float:
    """I(A; B) = H(A) + H(B) - H(A, B), plug-in estimate, for two integer-coded variables."""
    joint = Counter(zip(a.tolist(), b.tolist()))
    h_joint = entropy_bits(np.array(list(joint.values()), dtype=float))
    return entropy_bits(np.bincount(a).astype(float)) + entropy_bits(np.bincount(b).astype(float)) - h_joint


def majority_vote_accuracy(groups: np.ndarray, classes: np.ndarray) -> float:
    """Predict each image's class as the most common class of its group."""
    correct = 0
    for g in np.unique(groups):
        correct += np.bincount(classes[groups == g]).max()
    return float(correct / len(classes))


def report(rows: list[dict], fields: list[str], csv_name: str) -> tuple[list[dict], dict]:
    ids, names = assign_cells(rows, fields)
    class_names = sorted({r["class_name"] for r in rows})
    class_index = {c: i for i, c in enumerate(class_names)}
    y = np.array([class_index[r["class_name"]] for r in rows])
    scene_names = sorted({r["scene"] for r in rows})
    scene_ids = np.array([scene_names.index(r["scene"]) for r in rows])
    scrambled, _ = assign_cells(rows, fields, permute=True, seed=0)

    per_cell = []
    for cid, name in enumerate(names):
        members = [r for r, i in zip(rows, ids) if i == cid]
        classes = Counter(r["class_name"] for r in members)
        scenes = sorted({r["scene"] for r in members})
        per_cell.append({"cell_id": cid, "cell": name, "num_images": len(members), "num_classes": len(classes),
                         "largest_class_share": round(max(classes.values()) / len(members), 4),
                         "largest_class": classes.most_common(1)[0][0], "scenes": "+".join(scenes)})
    n_img = np.array([c["num_images"] for c in per_cell])
    n_cls = np.array([c["num_classes"] for c in per_cell])
    purity = np.array([c["largest_class_share"] for c in per_cell])
    summary = {
        "csv": csv_name, "fields": fields, "columns_read": ["relative_path"] + columns_read(fields),
        "num_images": len(rows), "num_classes_in_pool": len(class_names), "num_cells": len(names),
        "images_per_cell": {"min": int(n_img.min()), "median": float(np.median(n_img)), "mean": round(float(n_img.mean()), 1),
                            "max": int(n_img.max())},
        "classes_per_cell": {"min": int(n_cls.min()), "median": float(np.median(n_cls)), "max": int(n_cls.max()),
                             "image_weighted_mean": round(float((n_cls * n_img).sum() / n_img.sum()), 1)},
        "cells_with_fewer_than_5_classes": int((n_cls < 5).sum()),
        "share_of_images_in_cells_with_at_least_20_classes": round(float(n_img[n_cls >= 20].sum() / n_img.sum()), 4),
        "largest_class_share_per_cell": {"median": float(np.median(purity)), "max": float(purity.max())},
        "majority_vote_accuracy": {"from_cell": round(majority_vote_accuracy(ids, y), 4),
                                   "from_scrambled_cells_same_sizes": round(majority_vote_accuracy(scrambled, y), 4),
                                   "from_scene_alone": round(majority_vote_accuracy(scene_ids, y), 4),
                                   "chance": round(1 / len(class_names), 4)},
        "mutual_information_bits": {"cell_class": round(mutual_information_bits(ids, y), 4),
                                    "scrambled_cell_class": round(mutual_information_bits(scrambled, y), 4),
                                    "scene_class": round(mutual_information_bits(scene_ids, y), 4),
                                    "class_entropy": round(entropy_bits(np.bincount(y).astype(float)), 4)},
        "cells_per_scene": dict(sorted(Counter(c["scenes"] for c in per_cell).items())),
        "note": ("a cell is one combination of the fields; labels were read for this report only, never by training. "
                 "Scrambled cells = the same cell sizes with random membership (seed 0), the finite-sample floor of "
                 "the majority vote and of the mutual information."),
        "written": datetime.now().isoformat(timespec="seconds"),
    }
    return per_cell, summary


def main():
    ap = argparse.ArgumentParser(description="P4 condition-cell report (CPU, reads labels for the report only)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="SSL config whose cond_center.cell fields to report")
    ap.add_argument("--fields", nargs="+", choices=sorted(CELL_FIELDS), help="override the fields of the config")
    ap.add_argument("--csv", default="ssl_training_data.csv", help="CSV in the split folder to report on")
    a = ap.parse_args()
    fields = a.fields or json.loads(a.config.read_text(encoding="utf-8"))["cond_center"]["cell"]
    rows = read_rows(paths.SPLITS_DIR / a.csv)
    per_cell, s = report(rows, fields, a.csv)

    out_dir = paths.OUTPUT_DIR / "cell_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    # named after the config (so the scrambled control's report does not overwrite P4's) or after the fields given
    stem = f"split{paths.SPLIT}_{Path(a.csv).stem}_{'-'.join(fields) if a.fields else a.config.stem}"
    with open(out_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_cell[0]))
        w.writeheader()
        w.writerows(per_cell)
    mv, mi = s["majority_vote_accuracy"], s["mutual_information_bits"]
    lines = [f"# P4 cell report: split {paths.SPLIT}, {a.csv}, cells = {' | '.join(fields)}", "",
             f"- images {s['num_images']}, classes in the pool {s['num_classes_in_pool']}, cells {s['num_cells']}",
             f"- images per cell min / median / mean / max: {s['images_per_cell']['min']} / {s['images_per_cell']['median']:.0f} / "
             f"{s['images_per_cell']['mean']} / {s['images_per_cell']['max']}",
             f"- classes per cell min / median / max: {s['classes_per_cell']['min']} / {s['classes_per_cell']['median']:.0f} / "
             f"{s['classes_per_cell']['max']} (image-weighted mean {s['classes_per_cell']['image_weighted_mean']})",
             f"- cells with fewer than 5 classes: {s['cells_with_fewer_than_5_classes']}",
             f"- share of images in cells holding at least 20 classes: {100 * s['share_of_images_in_cells_with_at_least_20_classes']:.1f} % "
             f"(= the share of the pool outside bare_soil and woodland, which hold 11 classes)",
             f"- largest-class share within a cell, median / max: {100 * s['largest_class_share_per_cell']['median']:.1f} % / "
             f"{100 * s['largest_class_share_per_cell']['max']:.1f} %",
             f"- majority vote on the class from the cell alone: {100 * mv['from_cell']:.2f} % (scrambled cells of the same sizes "
             f"{100 * mv['from_scrambled_cells_same_sizes']:.2f} %, scene alone {100 * mv['from_scene_alone']:.2f} %, chance {100 * mv['chance']:.2f} %)",
             f"- mutual information with the class, bits: cell {mi['cell_class']:.3f} (scrambled cells {mi['scrambled_cell_class']:.3f}, "
             f"scene {mi['scene_class']:.3f}; class entropy {mi['class_entropy']:.2f})",
             f"- cells per scene: {s['cells_per_scene']}",
             f"- CSV columns the cell definition reads: {s['columns_read']} (labels read for this report only)",
             "", f"Per-cell table: {stem}.csv. Written {s['written']}."]
    (out_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten to {out_dir}: {stem}.csv, {stem}.md, {stem}.json")


if __name__ == "__main__":
    main()
