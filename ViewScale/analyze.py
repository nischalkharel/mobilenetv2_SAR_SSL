"""ViewScale analysis: gain against SSL window extent, from the two compare tables that already exist.

    python ViewScale/analyze.py      -> ViewScale/outputs/analysis/{curve.png, curve.csv, summary.md}

Reads
  SARDet_to_ATR/outputs/compare/cross_dataset.csv   (ATRNet transfer: method, budget, split, test_accuracy)
  SARDet/compare/outputs/<newest>/runs.csv           (SARDet detection: method, budget, split, best_test_map)
and plots, per label budget, the paired gain of each window extent over its reference (XA DINO for transfer,
no-SSL baseline for detection), with the two existing RegionDINO object-only rows (proposed regions; random
windows of the region sizes) as marked reference points. Transfer = mean over the 3 splits; detection = split 1 only.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EXTENTS = [32, 64, 128, 256, 512]
BUDGETS = ["1pct", "3pct", "10pct"]
OUT = HERE / "outputs" / "analysis"

# transfer rows (SARDet_to_ATR/compare.py names)
XFER_REF = "XA DINO on SARDet"
XFER_ROWS = {e: f"XA ViewScale e{e} (fixed {e}-px windows)" for e in EXTENTS}
XFER_MARKS = {"object-only (proposed regions)": "XA RegionDINO object-only on SARDet",
              "object-only, random windows": "XA RegionDINO object-only, random windows (control)"}
# detection rows (SARDet/compare/compare.py method names)
DET_REF = "baseline_SARDet"
DET_ROWS = {e: f"ViewScale:viewscale_e{e}_ssl_mobilenetv2" for e in EXTENTS}
DET_MARKS = {"object-only (proposed regions)": "RegionDINO_object:regiondino_o_ssl_mobilenetv2",
             "object-only, random windows": "RegionDINO_object:regiondino_o_random_ssl_mobilenetv2"}
# object sizes for the axis caption (SARDet_chips/DESIGN.md: annotation box longer side, native px, medians)
OBJECT_MEDIANS = {"tank": 21, "bridge": 17, "ship": 37, "aircraft": 79, "harbor": 80}


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def paired(rows, key, ref, value, scale):
    """{(budget, split): gain x scale} of row `key` over row `ref`, same budget and split."""
    val = {(r["method"], r["budget"], int(r["split"])): float(r[value]) for r in rows if r.get(value) not in (None, "")}
    out = {}
    for (m, b, s), v in val.items():
        if m == key and (ref, b, s) in val:
            out[(b, s)] = (v - val[(ref, b, s)]) * scale
    return out


def mean_by_budget(g):
    by = defaultdict(list)
    for (b, s), v in g.items():
        by[b].append(v)
    return {b: (sum(v) / len(v), min(v), max(v), len(v)) for b, v in by.items()}


def main():
    xfer = read(REPO / "SARDet_to_ATR" / "outputs" / "compare" / "cross_dataset.csv")
    det_dirs = sorted((REPO / "SARDet" / "compare" / "outputs").glob("20*"))
    det = read(det_dirs[-1] / "runs.csv")
    det = [r for r in det if r["seed"] == "0"]
    panels = [("ATRNet-STAR chips, transfer (gain over plain DINO, accuracy points, mean of 3 splits)",
               xfer, "test_accuracy", 1.0, XFER_REF, XFER_ROWS, XFER_MARKS),
              ("SARDet-100K detection (gain over no SSL, mAP x 100, split 1)",
               det, "best_test_map", 100.0, DET_REF, DET_ROWS, DET_MARKS)]
    OUT.mkdir(parents=True, exist_ok=True)
    table = []          # csv rows
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    colors = {"1pct": "#1f5fa8", "3pct": "#d2691e", "10pct": "#2e8b57"}
    for ax, (title, rows, value, scale, ref, keys, marks) in zip(axes, panels):
        for b in BUDGETS:
            xs, ys, lo, hi = [], [], [], []
            for e in EXTENTS:
                g = mean_by_budget(paired(rows, keys[e], ref, value, scale)).get(b)
                if g is None:
                    continue
                xs.append(e); ys.append(g[0]); lo.append(g[1]); hi.append(g[2])
                table.append({"panel": "transfer" if value == "test_accuracy" else "detection", "row": f"e{e}", "budget": b,
                              "gain_mean": round(g[0], 2), "gain_min": round(g[1], 2), "gain_max": round(g[2], 2), "n_splits": g[3]})
            ax.plot(xs, ys, "-o", color=colors[b], lw=2, ms=6, label=f"{b.replace('pct', ' %')} labels")
            if any(h - l > 0 for l, h in zip(lo, hi)):
                ax.fill_between(xs, lo, hi, color=colors[b], alpha=0.12, lw=0)
            for name, key in marks.items():
                g = mean_by_budget(paired(rows, key, ref, value, scale)).get(b)
                if g is None:
                    continue
                table.append({"panel": "transfer" if value == "test_accuracy" else "detection", "row": name, "budget": b,
                              "gain_mean": round(g[0], 2), "gain_min": round(g[1], 2), "gain_max": round(g[2], 2), "n_splits": g[3]})
                mk = "*" if "proposed" in name else "s"
                x = 1000 if "proposed" in name else 1500
                ax.plot([x], [g[0]], mk, color=colors[b], ms=11 if mk == "*" else 7, mfc="white" if mk == "s" else colors[b], mew=1.5)
        ax.axhline(0, color="#999", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(EXTENTS + [1000, 1500]); ax.set_xlim(26, 1900)
        ax.set_xticklabels([str(e) for e in EXTENTS] + ["regions", "random\nctrl"], fontsize=9)
        ax.set_xlabel("SSL window extent, px in the 512-px frame (right: the two existing object-only rows)", fontsize=9.5)
        ax.set_ylabel("gain, points")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", color="#e5e5e5")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    obj = ", ".join(f"{k} {v}" for k, v in OBJECT_MEDIANS.items())
    fig.suptitle("ViewScale: how the SSL gain changes with the window size (object-only recipe, random square windows of one extent)", fontsize=11)
    fig.text(0.5, 0.012, f"Object sizes for scale (median longer box side, native px): {obj}.\n"
             "Star = RegionDINO object-only (windows on proposed regions, median region 10 px); open square = its random-window control (same sizes, random places).",
             ha="center", fontsize=8.5, color="#444")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(OUT / "curve.png", dpi=150)
    with open(OUT / "curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)
    # markdown summary
    lines = ["# ViewScale: gain against window extent", "",
             f"Sources: SARDet_to_ATR/outputs/compare/cross_dataset.csv and {det_dirs[-1].name}/runs.csv.",
             "Transfer gains are over XA DINO on SARDet (mean of 3 splits, min..max); detection gains over the no-SSL baseline (split 1).", ""]
    for panel in ("transfer", "detection"):
        lines += [f"## {panel}", "", "| row | 1 % | 3 % | 10 % |", "|---|---|---|---|"]
        rows_here = sorted({t["row"] for t in table if t["panel"] == panel}, key=lambda r: (not r.startswith("e"), int(r[1:]) if r.startswith("e") and r[1:].isdigit() else 0, r))
        for r in rows_here:
            cells = []
            for b in BUDGETS:
                t = next((t for t in table if t["panel"] == panel and t["row"] == r and t["budget"] == b), None)
                cells.append("-" if t is None else (f"{t['gain_mean']:+.1f} ({t['gain_min']:+.1f}..{t['gain_max']:+.1f})" if t["n_splits"] > 1 else f"{t['gain_mean']:+.1f}"))
            lines.append(f"| {r} | " + " | ".join(cells) + " |")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
