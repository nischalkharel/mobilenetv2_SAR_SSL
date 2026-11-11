"""P4 versus A1 (and A0): test and validation accuracy per split and label budget, mean over splits, SSL cost.

Reads the finished CRC runs only (A100 in run_info.json), exactly as Biweekly_nischal_ngc/11_2_2026/make_split_pdfs.py
does for the biweekly tables: one run per (method, budget, split), eval_test/metrics.json and eval_validation/metrics.json.
Methods: A0 = ImageNet only, A1 = DINO (Experiments/mobileNetv2_DINO), P4 = DINO with per-condition-cell centering,
P4perm = scrambled-cell control, P4d = 3-field cells (this folder). Test scores are recorded, never used for decisions;
the P4 - A1 difference is the result of the experiment. Seed noise on this benchmark is about 1 point (one SSL seed).

Run from the repository root (CPU, seconds):
    python Experiments/mobileNetv2_DINO_P4/compare_p4.py
Writes outputs/compare/p4_vs_a1.md (tables) and p4_vs_a1.csv (one row per run) and prints the tables.
"""
import csv
import json
import re
import statistics as st
from datetime import datetime
from pathlib import Path

import paths

REPO = paths.EXPERIMENT_DIR.parents[1]
METHODS = [("A0", "ImageNet only (A0)"), ("A1", "DINO (A1)"), ("P4", "DINO + per-cell centering (P4)"),
           ("P4perm", "scrambled cells (P4perm)"), ("P4d", "3-field cells (P4d)")]
BUDGETS = [("1pct", "1 %"), ("3pct", "3 %"), ("10pct", "10 %")]
SPLITS = (1, 2, 3)
RUN = re.compile(r"\d{8}-\d{6}_(?P<method>A\d|P4perm|P4d|P4)_\w+?_mobilenetv2"
                 r"(?:_labeled_set_(?P<budget>1pct|10pct))?(?:_split(?P<split>\d))?_seed0")
SSL_RUN = re.compile(r"\d{8}-\d{6}_(?P<method>A1|P4perm|P4d|P4)_dino_ssl_mobilenetv2(?:_split(?P<split>\d))?_seed0")


def is_a100(run_dir: Path) -> bool:
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    return "A100" in info["environment"].get("gpu", "")


def collect_finetunes() -> dict:
    """(method, budget, split) -> record, from every finished CRC fine-tune with a test evaluation."""
    runs = {}
    for f in sorted((REPO / "Experiments").rglob("eval_test/metrics.json")):
        run_dir = f.parent.parent
        if run_dir.name.endswith("_SMOKE") or "smoke" in run_dir.parts:
            continue
        m = RUN.fullmatch(run_dir.name)
        if not m or not is_a100(run_dir):
            continue
        key = (m["method"], m["budget"] or "3pct", int(m["split"] or 1))
        if key in runs:
            raise SystemExit(f"two CRC runs for {key}: {runs[key]['run']} and {run_dir.name}")
        test = json.loads(f.read_text(encoding="utf-8"))
        assert test["split"] == "test" and test["num_images"] == 29284, f
        val_file = run_dir / "eval_validation" / "metrics.json"
        val = json.loads(val_file.read_text(encoding="utf-8")) if val_file.exists() else {}
        info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
        runs[key] = {"method": key[0], "budget": key[1], "split": key[2], "run": run_dir.name,
                     "test_acc": 100 * test["overall_accuracy"], "test_mean_per_class": 100 * test["mean_per_class_accuracy"],
                     "val_acc": 100 * val["overall_accuracy"] if val else None,
                     "best_epoch": info.get("best_epoch"), "best_val_acc_at_training": 100 * info["best_val_acc"]}
    return runs


def collect_ssl() -> dict:
    """(method, split) -> SSL cost of the finished CRC run that produced the backbone."""
    out = {}
    for f in sorted((REPO / "Experiments").glob("*/outputs/ssl/*/run_info.json")):
        m = SSL_RUN.fullmatch(f.parent.name)
        if not m:
            continue
        info = json.loads(f.read_text(encoding="utf-8"))
        if info.get("status") != "completed" or "A100" not in info["environment"].get("gpu", ""):
            continue
        key = (m["method"], int(m["split"] or 1))
        if key in out:
            raise SystemExit(f"two completed CRC SSL runs for {key}: {out[key]['run']} and {f.parent.name}")
        cost = info["cost"]
        out[key] = {"run": f.parent.name, "hours": cost["train_minutes"] / 60, "gpu_gb": cost["peak_gpu_memory_gb"],
                    "cells": (info.get("cond_center") or {}).get("n_cells")}
    return out


def fmt(x, digits=1):
    return "" if x is None else f"{x:.{digits}f}"


def table(rows: list[list[str]], header: list[str]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def main():
    runs, ssl = collect_finetunes(), collect_ssl()
    present = [(m, name) for m, name in METHODS if any(k[0] == m for k in runs)]
    lines = [f"# P4 versus A1: CRC A100 runs, official SOC-40 test set (29,284 images), seed 0", "",
             "Test accuracy (%), checkpoint chosen on validation. Differences in parentheses are against A1 on the same",
             "split and budget. Test scores are recorded, never used for decisions. Seed noise is about 1 point.", ""]

    def acc(metric, m, b, s):
        r = runs.get((m, b, s))
        return None if r is None else r[metric]

    def cell(metric, m, b, s):
        v = acc(metric, m, b, s)
        if v is None:
            return "-"
        a1 = acc(metric, "A1", b, s)
        return fmt(v) if m == "A1" or a1 is None else f"{fmt(v)} ({v - a1:+.1f})"

    for metric, title in [("test_acc", "Test accuracy"), ("val_acc", "Validation accuracy (800 images, the model-selection set)")]:
        lines += [f"## {title}", ""]
        for s in SPLITS:
            rows = [[name] + [cell(metric, m, b, s) for b, _ in BUDGETS] for m, name in present
                    if any((m, b, s) in runs for b, _ in BUDGETS)]
            lines += [f"**Split {s}**", ""] + table(rows, ["Method"] + [f"{lab} labels" for _, lab in BUDGETS]) + [""]
        # mean over the splits for methods that have all three
        rows = []
        for m, name in present:
            if all((m, b, s) in runs for b, _ in BUDGETS for s in SPLITS):
                vals = {b: [acc(metric, m, b, s) for s in SPLITS] for b, _ in BUDGETS}
                if any(v is None for vs in vals.values() for v in vs):
                    continue
                row = [name]
                for b, _ in BUDGETS:
                    mean, sd = st.mean(vals[b]), st.stdev(vals[b])
                    a1 = [acc(metric, "A1", b, s) for s in SPLITS]
                    delta = "" if m == "A1" or any(v is None for v in a1) else f" ({mean - st.mean(a1):+.1f})"
                    row.append(f"{mean:.1f} +/- {sd:.1f}{delta}")
                rows.append(row)
        if rows:
            lines += ["**Mean +/- standard deviation over splits 1-3** (methods run on all three splits)", ""] + \
                     table(rows, ["Method"] + [f"{lab} labels" for _, lab in BUDGETS]) + [""]

    # per-split deltas P4 - A1 in one line each, the number the paper would quote
    for m, name in present:
        if m in ("P4", "P4perm", "P4d"):
            deltas = {b: [acc("test_acc", m, b, s) - acc("test_acc", "A1", b, s) for s in SPLITS
                          if (m, b, s) in runs and ("A1", b, s) in runs] for b, _ in BUDGETS}
            parts = [f"{lab}: " + ", ".join(f"{d:+.1f}" for d in ds) for (b, lab), ds in zip(BUDGETS, deltas.values()) if ds]
            lines.append(f"- {name} minus A1, test accuracy per split: " + "; ".join(parts))
    lines.append("")

    lines += ["## SSL cost on one A100 (100 epochs)", ""]
    rows = []
    for m, name in METHODS:
        for s in SPLITS:
            if (m, s) in ssl:
                c = ssl[(m, s)]
                rows.append([name, str(s), fmt(c["hours"]), fmt(c["gpu_gb"]), str(c["cells"] or "-"), c["run"]])
    lines += table(rows, ["Method", "Split", "Hours", "Peak GPU GB", "Cells", "SSL run"]) + [""]
    lines += [f"Runs: {len(runs)} fine-tunes with test evaluation. Written {datetime.now().isoformat(timespec='seconds')}."]

    out_dir = paths.OUTPUT_DIR / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "p4_vs_a1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out_dir / "p4_vs_a1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(runs.values()))))
        w.writeheader()
        w.writerows(sorted(runs.values(), key=lambda r: (r["method"], r["split"], r["budget"])))
    print("\n".join(lines))
    print(f"\nWritten to {out_dir}: p4_vs_a1.md, p4_vs_a1.csv")


if __name__ == "__main__":
    main()
