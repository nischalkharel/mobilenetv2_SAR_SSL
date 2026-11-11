"""Gather every completed fine-tune run of every experiment into tables and charts: one command.

    python compare/compare.py [--out compare/outputs/<timestamp>] [--include-deviations] [--from-smoke]

Reads <experiment>/outputs/finetune/*/run_info.json (status completed, no protocol deviation, seed 0) and the
eval_*/metrics.json next to them. Method = experiment folder name (plus the SSL config variant for CondDINO).
Writes: runs.csv (one row per run), summary.md (method x budget: mean, std and per-split test mAP / AP50;
paired differences vs the baseline per split with mean, range and sign consistency; last-vs-best selection
sensitivity; seed-repeat noise when a seed-1 run exists; SSL GPU-hours), per_class.csv, per_source.csv,
charts/map_vs_budget.png, charts/paired_diff.png, charts/per_class_ap50_10pct.png, charts/per_source_map.png.
No t-tests: with 3 splits the honest statistics are the paired per-split differences and their sign.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import paths  # noqa: E402

EXPERIMENTS = ["baseline_SARDet", "DINO_SARDet", "LeJEPA_SARDet", "SparK_SARDet", "CondDINO_SARDet"]
BUDGETS = ["1pct", "3pct", "10pct"]
BASELINE = "baseline_SARDet"


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def collect(include_deviations=False, subfolder="finetune"):
    runs = []
    for exp in EXPERIMENTS:
        out = paths.experiment_output_dir(paths.SUITE_ROOT / exp) / subfolder
        for info_path in sorted(out.glob("*/run_info.json")):
            info = read_json(info_path)
            if info.get("status") != "completed" or info.get("kind") != "finetune":
                continue
            if info.get("protocol_deviation") and not include_deviations:
                continue
            cfg = info["config"]
            method = exp
            bb = info.get("backbone", {})
            variant = ""
            if exp == "CondDINO_SARDet":
                variant = (bb.get("ssl_config") or {}).get("run_name", "") if isinstance(bb.get("ssl_config"), dict) else ""
                variant = variant or Path(str(bb.get("ssl_run_dir", ""))).name.split("_split")[0].split("_", 1)[-1]
                method = f"{exp}:{variant}" if variant else exp
            run_dir = info_path.parent
            row = {"method": method, "experiment": exp, "budget": cfg["budget"], "split": info["split"], "seed": cfg["seed"],
                   "run_dir": str(run_dir), "best_epoch": info.get("best_epoch"), "best_val_select_map": info.get("best_val_select_map"),
                   "train_minutes": info.get("train_minutes"), "backbone_kind": bb.get("kind"),
                   "backbone_file_sha256": bb.get("file_sha256") or bb.get("sha256"), "ssl_run_dir": bb.get("ssl_run_dir"),
                   "ssl_gpu_hours": (bb.get("ssl_cost") or {}).get("gpu_hours") if isinstance(bb.get("ssl_cost"), dict) else None}
            for name in ["best_val", "best_test", "best_test_clean", "last_test"]:
                m = info.get("final_metrics", {}).get(name)
                if m:
                    for k in ["map", "map50", "map75", "map_small", "map_medium", "map_large"]:
                        row[f"{name}_{k}"] = m.get(k)
            mp = run_dir / "eval_best_test" / "metrics.json"
            if mp.exists():
                m = read_json(mp)
                row["per_class"] = {c: v for c, v in m["per_class"].items()}
                row["per_source"] = {s: {"map": v["map"], "map50": v["map50"]} for s, v in m.get("per_source", {}).items()}
            runs.append(row)
    return runs


def fmt(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--include-deviations", action="store_true")
    ap.add_argument("--from-smoke", action="store_true", help="read outputs/smoke instead of outputs/finetune (code check only)")
    args = ap.parse_args()
    out = Path(args.out or paths.SUITE_ROOT / "compare" / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    (out / "charts").mkdir(parents=True, exist_ok=True)
    runs = collect(args.include_deviations or args.from_smoke, "smoke" if args.from_smoke else "finetune")
    if not runs:
        sys.exit("no completed fine-tune runs found")
    main_runs = [r for r in runs if r["seed"] == 0]
    with open(out / "runs.csv", "w", newline="", encoding="utf-8") as f:
        cols = [k for k in runs[0] if k not in ("per_class", "per_source")]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(runs)

    # ---- method x budget table -----------------------------------------------------------------------------
    by = defaultdict(dict)                      # (method, budget) -> {split: run}
    for r in main_runs:
        prev = by[(r["method"], r["budget"])].get(r["split"])
        if prev is not None:
            print(f"WARNING: two completed seed-0 runs for {r['method']} {r['budget']} split {r['split']}: "
                  f"{prev['run_dir']} and {r['run_dir']}; the NEWER folder is used. Delete or move one of them.")
        by[(r["method"], r["budget"])][r["split"]] = r
    # one backbone per (method, split) across budgets, or the comparison is not what it claims to be
    for (m, b), cell in list(by.items()):
        for s in cell:
            shas = {by[(m, b2)][s].get("backbone_file_sha256") for b2 in BUDGETS if (m, b2) in by and s in by[(m, b2)]}
            if len(shas) > 1:
                print(f"WARNING: {m} split {s} uses different backbones across budgets: {shas}")
    methods = sorted({r["method"] for r in main_runs}, key=lambda m: (m != BASELINE, m))
    lines = ["# SARDet-100K detection: method comparison", f"generated {datetime.now():%Y-%m-%d %H:%M}; {len(main_runs)} seed-0 runs", "",
             "Official test mAP@[.5:.95] (AP50 in brackets); per split; mean +/- std over the splits present.", "",
             "| method | budget | split 1 | split 2 | split 3 | mean +/- std | n |", "|---|---|---|---|---|---|---|"]
    for m in methods:
        for b in BUDGETS:
            cell = by.get((m, b), {})
            vals = [cell[s]["best_test_map"] for s in sorted(cell) if cell[s].get("best_test_map") is not None]
            cols = []
            for s in (1, 2, 3):
                r = cell.get(s)
                cols.append(f"{fmt(r['best_test_map'])} ({fmt(r.get('best_test_map50'))})" if r and r.get("best_test_map") is not None else "-")
            ms = f"{np.mean(vals):.3f} +/- {np.std(vals, ddof=1) if len(vals) > 1 else 0:.3f}" if vals else "-"
            lines += [f"| {m} | {b} | {' | '.join(cols)} | {ms} | {len(vals)} |"]
    # ---- paired differences vs baseline -------------------------------------------------------------------
    lines += ["", "## Paired differences vs. baseline (same split, same labeled set): test mAP points x100", "",
              "| method | budget | per split | mean | min..max | sign |", "|---|---|---|---|---|---|"]
    paired = {}
    for m in methods:
        if m == BASELINE:
            continue
        for b in BUDGETS:
            diffs = {}
            for s, r in by.get((m, b), {}).items():
                base = by.get((BASELINE, b), {}).get(s)
                if base and r.get("best_test_map") is not None and base.get("best_test_map") is not None:
                    diffs[s] = 100 * (r["best_test_map"] - base["best_test_map"])
            if diffs:
                d = list(diffs.values())
                sign = "all +" if all(x > 0 for x in d) else ("all -" if all(x < 0 for x in d) else "mixed")
                paired[(m, b)] = diffs
                lines += [f"| {m} | {b} | {', '.join(f's{s}: {v:+.1f}' for s, v in sorted(diffs.items()))} | {np.mean(d):+.1f} | "
                          f"{min(d):+.1f}..{max(d):+.1f} | {sign} |"]
    # ---- selection sensitivity, clean test, seed noise, cost ----------------------------------------------
    lines += ["", "## Selection sensitivity and secondary numbers", "",
              "| method | budget | split | best-ckpt test | last-ckpt test | test_clean | full val | best epoch |", "|---|---|---|---|---|---|---|---|"]
    for m in methods:
        for b in BUDGETS:
            for s, r in sorted(by.get((m, b), {}).items()):
                lines += [f"| {m} | {b} | {s} | {fmt(r.get('best_test_map'))} | {fmt(r.get('last_test_map'))} | "
                          f"{fmt(r.get('best_test_clean_map'))} | {fmt(r.get('best_val_map'))} | {r.get('best_epoch')} |"]
    repeats = [r for r in runs if r["seed"] != 0]
    if repeats:
        lines += ["", "## Fine-tune seed noise (same data, different seed)", ""]
        for r in repeats:
            base = by.get((r["method"], r["budget"]), {}).get(r["split"])
            if base and base.get("best_test_map") is not None and r.get("best_test_map") is not None:
                lines += [f"- {r['method']} {r['budget']} split {r['split']}: seed {r['seed']} vs seed 0: "
                          f"{100 * (r['best_test_map'] - base['best_test_map']):+.1f} mAP points"]
    lines += ["", "## Cost", "", "| method | SSL GPU-hours (per split run) | fine-tune minutes (10 %) |", "|---|---|---|"]
    for m in methods:
        g = [r["ssl_gpu_hours"] for r in main_runs if r["method"] == m and r.get("ssl_gpu_hours")]
        t = [r["train_minutes"] for r in main_runs if r["method"] == m and r["budget"] == "10pct" and r.get("train_minutes")]
        lines += [f"| {m} | {fmt(np.mean(g), 1) if g else '-'} | {fmt(np.mean(t), 0) if t else '-'} |"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- per-class and per-source CSVs ---------------------------------------------------------------------
    with open(out / "per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "budget", "split"] + [f"{c}_ap50" for c in paths.CLASS_NAMES] + [f"{c}_ap" for c in paths.CLASS_NAMES])
        for r in main_runs:
            if "per_class" in r:
                w.writerow([r["method"], r["budget"], r["split"]] + [f"{r['per_class'][c]['ap50']:.4f}" for c in paths.CLASS_NAMES]
                           + [f"{r['per_class'][c]['ap']:.4f}" for c in paths.CLASS_NAMES])
    sources = sorted({s for r in main_runs for s in r.get("per_source", {})})
    with open(out / "per_source.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "budget", "split"] + [f"{s}_map" for s in sources])
        for r in main_runs:
            if "per_source" in r:
                w.writerow([r["method"], r["budget"], r["split"]] + [f"{r['per_source'].get(s, {}).get('map', float('nan')):.4f}" for s in sources])

    # ---- charts ------------------------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    x = {b: i for i, b in enumerate(BUDGETS)}
    for m in methods:
        means, xs = [], []
        for b in BUDGETS:
            vals = [r["best_test_map"] for r in by.get((m, b), {}).values() if r.get("best_test_map") is not None]
            if vals:
                means.append(np.mean(vals)); xs.append(x[b])
                ax.scatter([x[b]] * len(vals), vals, s=12, alpha=0.6)
        if means:
            ax.plot(xs, means, marker="o", label=m)
    ax.set_xticks(list(x.values())); ax.set_xticklabels(BUDGETS); ax.set_ylabel("test mAP@[.5:.95]"); ax.set_xlabel("labeled budget")
    ax.set_title("test mAP vs budget (dots = splits, line = mean)"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(out / "charts" / "map_vs_budget.png", dpi=130); plt.close(fig)
    if paired:
        fig, ax = plt.subplots(figsize=(7, 4))
        keys = sorted(paired)
        for i, k in enumerate(keys):
            d = list(paired[k].values())
            ax.bar(i, np.mean(d), color="tab:blue" if np.mean(d) > 0 else "tab:red", alpha=0.7)
            ax.scatter([i] * len(d), d, color="k", s=12, zorder=3)
        ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(len(keys))); ax.set_xticklabels([f"{m.replace('_SARDet', '')}\n{b}" for m, b in keys], fontsize=6)
        ax.set_ylabel("test mAP minus baseline (x100)"); ax.set_title("paired differences per split (dots) and mean (bar)")
        fig.tight_layout(); fig.savefig(out / "charts" / "paired_diff.png", dpi=130); plt.close(fig)
    cls = {m: by[(m, "10pct")][1]["per_class"] for m in methods if (m, "10pct") in by and 1 in by[(m, "10pct")] and "per_class" in by[(m, "10pct")][1]}
    if cls:
        from shared.plots import per_class_bars
        per_class_bars({m: {"per_class": v} for m, v in cls.items()}, out / "charts" / "per_class_ap50_10pct.png", title="per-class AP50, 10 %, split 1, test")
    if sources:
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(methods) + 1.5))
        grid = np.full((len(methods), len(sources)), np.nan)
        for i, m in enumerate(methods):
            for j, s in enumerate(sources):
                vals = [r["per_source"][s]["map"] for r in by.get((m, "10pct"), {}).values() if s in r.get("per_source", {})]
                if vals:
                    grid[i, j] = np.mean(vals)
        im = ax.imshow(grid, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(sources))); ax.set_xticklabels(sources, rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=7)
        for i in range(len(methods)):
            for j in range(len(sources)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w")
        ax.set_title("test mAP per source dataset, 10 % (mean over splits)"); fig.colorbar(im, ax=ax); fig.tight_layout()
        fig.savefig(out / "charts" / "per_source_map.png", dpi=130); plt.close(fig)
    print("\n".join(lines[:40]))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
