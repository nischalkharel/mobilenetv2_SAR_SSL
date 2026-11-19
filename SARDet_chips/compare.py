"""Chip study tables: every completed fine-tune run -> accuracy per method x budget x split, paired differences,
per-class and per-source accuracy.

    python SARDet_chips/compare.py            -> prints and writes SARDet_chips/outputs/compare/chips.{md,csv}, chips_per_class.csv,
                                                 chips_per_source.csv (newest completed run per cell; duplicates reported)
Headline = mean per-class accuracy on test_clean (DESIGN.md section 3); overall accuracy and the plain test set are next to it.
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chip_paths as cp  # noqa: E402

LABELS = {"imagenet": "ImageNet start (no SSL)", "dino": "DINO on scenes", "lejepa": "LeJEPA on scenes", "spark": "SparK on scenes",
          "regiondino_so": "RegionDINO scene+object", "regiondino_so_random": "RegionDINO scene+object, random windows (control)",
          "regiondino_o": "RegionDINO object-only", "regiondino_o_random": "RegionDINO object-only, random windows (control)"}
REF = {"dino": "imagenet", "lejepa": "imagenet", "spark": "imagenet", "regiondino_so": "dino", "regiondino_so_random": "dino",
       "regiondino_o": "dino", "regiondino_o_random": "dino"}
BUDGETS = ["1pct", "3pct", "10pct"]
NAME_RE = re.compile(r"^\d{8}-\d{6}_S1_(?P<method>.+?)_mobilenetv2_(?P<budget>\d+pct)_split(?P<split>[123])_seed0$")


def collect():
    cells, dups = {}, []
    for run in sorted(cp.FINETUNE_OUTPUT_DIR.glob("*")):
        m = NAME_RE.match(run.name)
        if not m or not (run / "eval_test_clean" / "metrics.json").exists():
            continue
        info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
        if info.get("status") != "completed":
            continue
        key = (m["method"], m["budget"], int(m["split"]))
        rec = {"run": run.name, "method": m["method"], "budget": m["budget"], "split": int(m["split"]),
               "best_epoch": info.get("best_epoch"), "train_minutes": info.get("train_minutes"),
               "backbone_sha256": (info.get("initialization") or {}).get("sha256")}
        for k in ("test", "test_clean"):
            mt = json.loads((run / f"eval_{k}" / "metrics.json").read_text(encoding="utf-8"))
            rec[f"{k}_overall"] = 100 * mt["overall_accuracy"]; rec[f"{k}_mean_class"] = 100 * mt["mean_per_class_accuracy"]
            rec[f"{k}_per_class"] = {c: 100 * v for c, v in mt["per_class_accuracy"].items()}
            rec[f"{k}_per_source"] = {s: 100 * v for s, v in mt["per_source_accuracy"].items()}
            rec[f"{k}_n"] = mt["num_chips"]
        if key in cells:
            dups.append((key, cells[key]["run"], run.name))
        cells[key] = rec
    return cells, dups


def fmt(v):
    return "-" if v is None else f"{v:.1f}"


def main():
    cells, dups = collect()
    out = cp.COMPARE_OUTPUT_DIR; out.mkdir(parents=True, exist_ok=True)
    methods = [m for m in LABELS if any(k[0] == m for k in cells)]
    L = ["# SARDet-100K chip classification (study S1: SSL on the scenes, fine-tuned and tested on chips)\n",
         "Mean per-class accuracy on test_clean, % (headline); per split and mean (sd). Then overall accuracy on test_clean, then both on the full test set.\n"]
    for metric, title in (("test_clean_mean_class", "Mean per-class accuracy, test_clean"), ("test_clean_overall", "Overall accuracy, test_clean"),
                          ("test_mean_class", "Mean per-class accuracy, full test"), ("test_overall", "Overall accuracy, full test")):
        L.append(f"\n## {title}\n\n| method | " + " | ".join(f"{b} (s1 / s2 / s3) | mean (sd)" for b in BUDGETS) + " |\n|---|" + "---|---|" * len(BUDGETS))
        for m in methods:
            row = f"| {LABELS[m]} |"
            for b in BUDGETS:
                v = [cells.get((m, b, s), {}).get(metric) for s in (1, 2, 3)]; p = [x for x in v if x is not None]
                row += f" {' / '.join(fmt(x) for x in v)} | {fmt(np.mean(p)) if p else '-'}{f' ({np.std(p, ddof=1):.1f})' if len(p) > 1 else ''} |"
            L.append(row)
    L.append("\n## Paired differences, mean per-class accuracy on test_clean (points): method minus its reference, same split and budget\n")
    L.append("Reference: no SSL for DINO / LeJEPA / SparK; plain DINO for the RegionDINO rows and controls.\n\n| method | budget | vs reference (s1 / s2 / s3) | mean | vs no SSL | mean |\n|---|---|---|---|---|---|")
    for m in methods:
        if m == "imagenet":
            continue
        for b in BUDGETS:
            d_ref = [cells[(m, b, s)]["test_clean_mean_class"] - cells[(REF[m], b, s)]["test_clean_mean_class"] if (m, b, s) in cells and (REF[m], b, s) in cells else None for s in (1, 2, 3)]
            d_base = [cells[(m, b, s)]["test_clean_mean_class"] - cells[("imagenet", b, s)]["test_clean_mean_class"] if (m, b, s) in cells and ("imagenet", b, s) in cells else None for s in (1, 2, 3)]
            pr, pb = [x for x in d_ref if x is not None], [x for x in d_base if x is not None]
            sg = lambda x: "-" if x is None else f"{x:+.1f}"
            L.append(f"| {LABELS[m]} | {b} | {' / '.join(sg(x) for x in d_ref)} | {sg(np.mean(pr)) if pr else '-'} | {' / '.join(sg(x) for x in d_base)} | {sg(np.mean(pb)) if pb else '-'} |")
    # per class and per source (test_clean, mean over splits) at each budget
    L.append("\n## Per-class accuracy on test_clean, mean of splits (%)\n")
    for b in BUDGETS:
        L.append(f"\n{b}:\n\n| method | " + " | ".join(cp.CLASS_NAMES) + " |\n|---|" + "---|" * len(cp.CLASS_NAMES))
        for m in methods:
            rows = [cells[(m, b, s)]["test_clean_per_class"] for s in (1, 2, 3) if (m, b, s) in cells]
            if rows:
                L.append(f"| {LABELS[m]} | " + " | ".join(fmt(np.mean([r[c] for r in rows])) for c in cp.CLASS_NAMES) + " |")
    sources = sorted({s for r in cells.values() for s in r["test_clean_per_source"]})
    L.append("\n## Per-source accuracy on test_clean, mean of splits (%)\n")
    for b in BUDGETS:
        L.append(f"\n{b}:\n\n| method | " + " | ".join(sources) + " |\n|---|" + "---|" * len(sources))
        for m in methods:
            rows = [cells[(m, b, s)]["test_clean_per_source"] for s in (1, 2, 3) if (m, b, s) in cells]
            if rows:
                L.append(f"| {LABELS[m]} | " + " | ".join(fmt(np.mean([r.get(src, np.nan) for r in rows])) for src in sources) + " |")
    if dups:
        L.append("\nDuplicate cells (newest used): " + "; ".join(f"{k}: {a} vs {b}" for k, a, b in dups))
    text = "\n".join(L) + "\n"
    print(text)
    (out / "chips.md").write_text(text, encoding="utf-8")
    with open(out / "chips.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["method", "budget", "split", "test_clean_mean_class", "test_clean_overall", "test_mean_class", "test_overall", "test_clean_n", "test_n", "best_epoch", "train_minutes", "run"])
        for (m, b, s), r in sorted(cells.items()):
            w.writerow([m, b, s, f"{r['test_clean_mean_class']:.2f}", f"{r['test_clean_overall']:.2f}", f"{r['test_mean_class']:.2f}", f"{r['test_overall']:.2f}", r["test_clean_n"], r["test_n"], r["best_epoch"], r["train_minutes"], r["run"]])
    with open(out / "chips_per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["method", "budget", "split", "set"] + cp.CLASS_NAMES)
        for (m, b, s), r in sorted(cells.items()):
            for k in ("test_clean", "test"):
                w.writerow([m, b, s, k] + [f"{r[f'{k}_per_class'][c]:.2f}" for c in cp.CLASS_NAMES])
    with open(out / "chips_per_source.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["method", "budget", "split", "set"] + sources)
        for (m, b, s), r in sorted(cells.items()):
            for k in ("test_clean", "test"):
                w.writerow([m, b, s, k] + [f"{r[f'{k}_per_source'].get(src, float('nan')):.2f}" for src in sources])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
