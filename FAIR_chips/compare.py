"""Collect every completed FAIR-CSAR chip fine-tune into one table (DESIGN.md section 5).

    python FAIR_chips/compare.py            -> outputs/compare/chips.md, chips.csv, per_class.csv

Headline: mean per-class accuracy on test_clean (22 classes), per split and mean (sd); then overall accuracy, then the
full test set, then paired differences against each method's reference (same split and budget), then per major
category and per class at 1 % labels.
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fair_paths as fp  # noqa: E402
METHODS = fp.METHODS

# Row labels follow Nischal's convention: SSL-<dataset> / FINETUNE-<dataset> / TEST-<dataset> / <technique>
_F = "FAIR-CSAR"
LABELS = {"imagenet": f"SSL-none / FINETUNE-{_F} / TEST-{_F} / ImageNet start",
          "fair_dino_tiles": f"SSL-{_F} tiles / FINETUNE-{_F} / TEST-{_F} / DINO",
          "fair_ro_tiles": f"SSL-{_F} tiles / FINETUNE-{_F} / TEST-{_F} / RegionDINO object-only",
          "fair_ror_tiles": f"SSL-{_F} tiles / FINETUNE-{_F} / TEST-{_F} / random windows (control)",
          "fair_dino_down": f"SSL-{_F} downsampled / FINETUNE-{_F} / TEST-{_F} / DINO",
          "fair_ro_down": f"SSL-{_F} downsampled / FINETUNE-{_F} / TEST-{_F} / RegionDINO object-only",
          "fair_ror_down": f"SSL-{_F} downsampled / FINETUNE-{_F} / TEST-{_F} / random windows (control)",
          "sardet_dino": f"SSL-SARDet-100K / FINETUNE-{_F} / TEST-{_F} / DINO",
          "sardet_regiondino_o": f"SSL-SARDet-100K / FINETUNE-{_F} / TEST-{_F} / RegionDINO object-only",
          "sardet_regiondino_o_random": f"SSL-SARDet-100K / FINETUNE-{_F} / TEST-{_F} / random windows (control)"}
REF = {"fair_dino_tiles": "imagenet", "fair_ro_tiles": "fair_dino_tiles", "fair_ror_tiles": "fair_dino_tiles",
       "fair_dino_down": "imagenet", "fair_ro_down": "fair_dino_down", "fair_ror_down": "fair_dino_down",
       "sardet_dino": "imagenet", "sardet_regiondino_o": "sardet_dino", "sardet_regiondino_o_random": "sardet_dino"}
BUDGETS = list(fp.BUDGETS)
RUN_RE = re.compile(r"^(?P<stamp>\d{8}-\d{6})_F1_(?P<method>.+)_mobilenetv2_(?P<budget>\d+pct)_split(?P<split>[123])_seed0$")


def fmt(x, nd=1):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def load_runs():
    cells = {}
    for run in sorted(fp.FINETUNE_OUTPUT_DIR.glob("*")):
        m = RUN_RE.match(run.name)
        if not m or m["method"] not in METHODS:
            continue
        info_p = run / "run_info.json"
        if not info_p.exists():
            continue
        info = json.loads(info_p.read_text(encoding="utf-8"))
        if info.get("status") != "completed":
            continue
        ev = {}
        for k in ("test", "test_clean"):
            mp = run / f"eval_{k}" / "metrics.json"
            if mp.exists():
                ev[k] = json.loads(mp.read_text(encoding="utf-8"))
        if len(ev) < 2:
            continue
        key = (m["method"], m["budget"], int(m["split"]))
        if key in cells and cells[key]["run"] > run.name:
            continue                                                   # newest completed run wins
        cells[key] = {"run": run.name, "best_epoch": info.get("best_epoch"), "minutes": info.get("train_minutes"),
                      **{f"{k}_{mm}": 100 * ev[k][mm] for k in ev for mm in ("mean_per_class_accuracy", "overall_accuracy")},
                      **{f"{k}_per_class": ev[k]["per_class_accuracy"] for k in ev},
                      **{f"{k}_per_major": ev[k]["per_source_accuracy"] for k in ev}}
    return cells


def table(cells, metric, title):
    L = [f"\n## {title}\n", "| method | " + " | ".join(f"{b} (s1 / s2 / s3) | mean (sd)" for b in BUDGETS) + " |", "|---|" + "---|---|" * len(BUDGETS)]
    for m in METHODS:
        if not any((m, b, s) in cells for b in BUDGETS for s in (1, 2, 3)):
            continue
        row = [LABELS[m]]
        for b in BUDGETS:
            v = [cells[(m, b, s)][metric] if (m, b, s) in cells else None for s in (1, 2, 3)]
            have = [x for x in v if x is not None]
            row.append(" / ".join(fmt(x) for x in v))
            row.append(f"{np.mean(have):.1f} ({np.std(have, ddof=1) if len(have) > 1 else 0:.1f})" if have else "-")
        L.append("| " + " | ".join(row) + " |")
    return L


def main():
    cells = load_runs()
    if not cells:
        raise SystemExit(f"no completed runs under {fp.FINETUNE_OUTPUT_DIR}")
    L = ["# FAIR-CSAR SL chip classification (22 classes): SSL on the trainval slices, fine-tuned and tested on chips", "",
         "Mean per-class accuracy on test_clean, % (headline); per split and mean (sd). Then overall accuracy, then both on the full test set."]
    L += table(cells, "test_clean_mean_per_class_accuracy", "Mean per-class accuracy, test_clean")
    L += table(cells, "test_clean_overall_accuracy", "Overall accuracy, test_clean")
    L += table(cells, "test_mean_per_class_accuracy", "Mean per-class accuracy, full test")
    L += table(cells, "test_overall_accuracy", "Overall accuracy, full test")

    L += ["\n## Paired differences, mean per-class accuracy on test_clean (points): method minus its reference, same split and budget\n",
          "References: no SSL for the DINO rows; the arm's DINO row for the object-only rows and their controls.\n",
          "| method | reference | " + " | ".join(f"{b} (s1 / s2 / s3) | mean" for b in BUDGETS) + " |", "|---|---|" + "---|---|" * len(BUDGETS)]
    for m, ref in REF.items():
        row = [LABELS[m], LABELS[ref]]; any_ = False
        for b in BUDGETS:
            d = [cells[(m, b, s)]["test_clean_mean_per_class_accuracy"] - cells[(ref, b, s)]["test_clean_mean_per_class_accuracy"]
                 if (m, b, s) in cells and (ref, b, s) in cells else None for s in (1, 2, 3)]
            have = [x for x in d if x is not None]; any_ |= bool(have)
            row.append(" / ".join(("+" if x >= 0 else "") + fmt(x) if x is not None else "-" for x in d))
            row.append(("+" if np.mean(have) >= 0 else "") + f"{np.mean(have):.1f}" if have else "-")
        if any_:
            L.append("| " + " | ".join(row) + " |")

    majors = ["Aircraft", "Ship", "Bridge", "Tank", "Tower_Crane"]
    L += ["\n## Accuracy per major category, test_clean, 1 % labels (mean over splits)\n", "| method | " + " | ".join(majors) + " |", "|---|" + "---|" * len(majors)]
    for m in METHODS:
        rows = [cells[(m, "1pct", s)]["test_clean_per_major"] for s in (1, 2, 3) if (m, "1pct", s) in cells]
        if rows:
            L.append(f"| {LABELS[m]} | " + " | ".join(fmt(100 * np.mean([r.get(c, np.nan) for r in rows])) for c in majors) + " |")
    L += ["\n## Per-class accuracy, test_clean, 1 % labels (mean over splits)\n", "| method | " + " | ".join(fp.CLASS_NAMES) + " |", "|---|" + "---|" * len(fp.CLASS_NAMES)]
    for m in METHODS:
        rows = [cells[(m, "1pct", s)]["test_clean_per_class"] for s in (1, 2, 3) if (m, "1pct", s) in cells]
        if rows:
            L.append(f"| {LABELS[m]} | " + " | ".join(fmt(100 * np.mean([r[c] for r in rows]), 0) for c in fp.CLASS_NAMES) + " |")

    fp.COMPARE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (fp.COMPARE_OUTPUT_DIR / "chips.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(fp.COMPARE_OUTPUT_DIR / "chips.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "budget", "split", "run", "best_epoch", "minutes", "test_clean_mean_class", "test_clean_overall", "test_mean_class", "test_overall"])
        for (m, b, s), c in sorted(cells.items()):
            w.writerow([m, b, s, c["run"], c["best_epoch"], c["minutes"]] + [f"{c[k]:.2f}" for k in ("test_clean_mean_per_class_accuracy", "test_clean_overall_accuracy", "test_mean_per_class_accuracy", "test_overall_accuracy")])
    with open(fp.COMPARE_OUTPUT_DIR / "per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["method", "budget", "split", "set"] + fp.CLASS_NAMES)
        for (m, b, s), c in sorted(cells.items()):
            for k in ("test", "test_clean"):
                w.writerow([m, b, s, k] + [f"{100 * c[f'{k}_per_class'][cl]:.2f}" for cl in fp.CLASS_NAMES])
    print("\n".join(L[:30])); print(f"\nwrote {fp.COMPARE_OUTPUT_DIR} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
