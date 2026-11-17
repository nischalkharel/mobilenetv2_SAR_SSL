"""Cross-dataset study: test accuracy on ATRNet-STAR of the SARDet-pretrained backbones next to the existing rows.

    python SARDet_to_ATR/compare.py            -> prints and writes SARDet_to_ATR/outputs/compare/cross_dataset.{md,csv}

Rows: A0 (ImageNet, no SSL), A1/A2/A3 (SSL on ATRNet chips), P4 (DINO + per-cell centering on ATRNet),
XA_<method> (SSL on SARDet-100K scenes, arm A), later XC_<method> (mixed pool, arm C). Columns: 1 / 3 / 10 %
labels; per split and mean (sd). Paired differences: SARDet-SSL minus ATRNet-SSL of the same method and split,
and minus A0. Only completed runs with eval_test/metrics.json count; the newest run per cell is used and
duplicates are reported.
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOTS = {
    "A0": REPO / "Experiments" / "Baseline" / "Baseline_Output",
    "A1_dino": REPO / "Experiments" / "mobileNetv2_DINO" / "outputs" / "finetune",
    "A2_lejepa": REPO / "Experiments" / "mobileNetv2_LeJEPA" / "outputs" / "finetune",
    "A3_spark": REPO / "Experiments" / "mobileNetv2_SparK" / "outputs" / "finetune",
    "P4": REPO / "Experiments" / "mobileNetv2_DINO_P4" / "outputs" / "finetune",
    "X": REPO / "SARDet_to_ATR" / "outputs" / "finetune",
}
NAME_RE = re.compile(r"^\d{8}-\d{6}_(?P<name>.+?)(?:_(?P<budget>labeled_set_1pct|labeled_set_10pct))?(?:_split(?P<split>[23]))?_seed(?P<seed>\d+)$")
BUDGETS = ["1pct", "3pct", "10pct"]


def row_key(folder: Path):
    m = NAME_RE.match(folder.name)
    if not m or "SMOKE" in folder.name or "parity" in folder.name or "lrcheck" in folder.name:
        return None
    name = m["name"]
    method = {"A0_imagenet_mobilenetv2": "A0 ImageNet (no SSL)", "A1_dino_mobilenetv2": "A1 DINO on ATRNet",
              "A2_lejepa_mobilenetv2": "A2 LeJEPA on ATRNet", "A3_spark_mobilenetv2": "A3 SparK on ATRNet",
              "P4_dino_mobilenetv2": "P4 CondDINO on ATRNet", "P4perm_dino_mobilenetv2": "P4 scrambled (control)",
              "P4d_dino_mobilenetv2": "P4 3-field (control)",
              "XA_dino_sardetssl_mobilenetv2": "XA DINO on SARDet", "XA_lejepa_sardetssl_mobilenetv2": "XA LeJEPA on SARDet",
              "XA_spark_sardetssl_mobilenetv2": "XA SparK on SARDet",
              "XC_dino_mixed_mobilenetv2": "XC DINO on mixed pool", "XC_lejepa_mixed_mobilenetv2": "XC LeJEPA on mixed pool",
              "XC_spark_mixed_mobilenetv2": "XC SparK on mixed pool"}.get(name)
    if method is None:
        return None
    budget = {"labeled_set_1pct": "1pct", "labeled_set_10pct": "10pct", None: "3pct"}[m["budget"]]
    return method, budget, int(m["split"] or 1), int(m["seed"])


def collect():
    cells = {}
    dups = []
    for root in ROOTS.values():
        if not root.exists():
            continue
        for folder in sorted(root.iterdir()):
            key = row_key(folder)
            metrics = folder / "eval_test" / "metrics.json"
            if key is None or not metrics.exists() or key[3] != 0:
                continue
            acc = 100 * json.loads(metrics.read_text(encoding="utf-8"))["overall_accuracy"]
            if key[:3] in cells:
                dups.append((key[:3], cells[key[:3]][1], folder.name))
            cells[key[:3]] = (acc, folder.name)
    return cells, dups


def main():
    cells, dups = collect()
    methods = []
    for m, _, _ in cells:
        if m not in methods:
            methods.append(m)
    order = ["A0", "A1", "A2", "A3", "P4 C", "P4 s", "P4 3", "XA D", "XA L", "XA S", "XC"]
    methods.sort(key=lambda m: next((i for i, o in enumerate(order) if m.startswith(o)), 99))
    lines = ["# SSL on SARDet-100K scenes, tested on ATRNet-STAR SOC-40 (official test accuracy, %)", "",
             "| method | 1 % (s1 / s2 / s3) | mean (sd) | 3 % | mean (sd) | 10 % | mean (sd) |", "|---|---|---|---|---|---|---|"]
    rows_csv = []
    for m in methods:
        line = f"| {m} |"
        for b in BUDGETS:
            vals = [cells.get((m, b, s), (None,))[0] for s in (1, 2, 3)]
            have = [v for v in vals if v is not None]
            line += " " + " / ".join("-" if v is None else f"{v:.1f}" for v in vals) + " |"
            line += (f" {np.mean(have):.1f} ({np.std(have, ddof=1) if len(have) > 1 else 0:.1f}) |" if have else " - |")
            for s, v in zip((1, 2, 3), vals):
                if v is not None:
                    rows_csv.append({"method": m, "budget": b, "split": s, "test_accuracy": round(v, 2)})
        lines.append(line)
    lines += ["", "## Paired differences per split (test accuracy points): SARDet-SSL minus the same method's ATRNet-SSL, and minus A0", "",
              "| method | budget | vs ATRNet-SSL (s1 / s2 / s3) | mean | vs A0 (s1 / s2 / s3) | mean |", "|---|---|---|---|---|---|"]
    pairs = {"XA DINO on SARDet": "A1 DINO on ATRNet", "XA LeJEPA on SARDet": "A2 LeJEPA on ATRNet", "XA SparK on SARDet": "A3 SparK on ATRNet",
             "XC DINO on mixed pool": "A1 DINO on ATRNet", "XC LeJEPA on mixed pool": "A2 LeJEPA on ATRNet", "XC SparK on mixed pool": "A3 SparK on ATRNet"}
    for m, ref in pairs.items():
        if m not in methods:
            continue
        for b in BUDGETS:
            d1, d0 = [], []
            for s in (1, 2, 3):
                v = cells.get((m, b, s))
                r = cells.get((ref, b, s))
                a0 = cells.get(("A0 ImageNet (no SSL)", b, s))
                d1.append(None if not (v and r) else v[0] - r[0])
                d0.append(None if not (v and a0) else v[0] - a0[0])
            f = lambda d: " / ".join("-" if x is None else f"{x:+.1f}" for x in d)
            mean = lambda d: f"{np.mean([x for x in d if x is not None]):+.1f}" if any(x is not None for x in d) else "-"
            lines.append(f"| {m} | {b} | {f(d1)} | {mean(d1)} | {f(d0)} | {mean(d0)} |")
    if dups:
        lines += ["", "Duplicate cells (newest used): " + "; ".join(f"{k}: {a} vs {b}" for k, a, b in dups)]
    out = REPO / "SARDet_to_ATR" / "outputs" / "compare"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cross_dataset.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "cross_dataset.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "budget", "split", "test_accuracy"])
        w.writeheader()
        w.writerows(rows_csv)
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
