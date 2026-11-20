"""Gather every result in the repository into small JSON files for the tracker pages (tracker/data/).

    python tracker/collect.py                 # read the run folders and the compare outputs, write tracker/data/*.json
    python tracker/collect.py --run-compare   # first re-run SARDet_to_ATR/compare.py and SARDet/compare/compare.py

Reads only: run folders (run_info.json, config.json, history.csv, eval_*/metrics.json), the two compare outputs,
the response-map CSVs. Writes:
    classification.json   ATRNet-STAR test accuracy per method x budget x split, with the run behind each cell
    detection.json        SARDet-100K test mAP per method x budget x split, per-source and per-class tables
    maps.json             frozen-backbone response maps on ATRNet chips and on SARDet scenes
    runs.json             every SSL and fine-tune run folder: status, time, cost, loss curve
    meta.json             when this was collected and at which commit
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tracker" / "data"
sys.path.insert(0, str(REPO / "SARDet_to_ATR"))
sys.path.insert(0, str(REPO / "tracker"))
import compare as atr_compare  # noqa: E402  (SARDet_to_ATR/compare.py: ROOTS and row_key)
import registry  # noqa: E402

SKIP = re.compile(r"smoke|SMOKE|_nockpt|OLD|parity|lrcheck", re.I)


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_csv(p):
    try:
        with open(p, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def rel(p):
    return Path(p).resolve().relative_to(REPO).as_posix()


def fnum(x, nd=4):
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- classification (ATRNet-STAR) ----------------
def classification():
    rows = {}
    for root in atr_compare.ROOTS.values():
        if not root.exists():
            continue
        for folder in sorted(root.iterdir()):
            key = atr_compare.row_key(folder)
            if key is None or key[3] != 0:
                continue
            method, budget, split, _ = key
            info = read_json(folder / "run_info.json") or {}
            metrics = read_json(folder / "eval_test" / "metrics.json")
            reg = registry.CLASSIFICATION.get(method)
            if reg is None:
                continue
            hist = read_csv(folder / "history.csv")
            cell = {
                "run": rel(folder), "status": info.get("status", "unknown"),
                "acc": round(100 * metrics["overall_accuracy"], 2) if metrics else None,
                "mean_class_acc": round(100 * metrics["mean_per_class_accuracy"], 2) if metrics else None,
                "best_epoch": info.get("best_epoch"), "epochs_run": len(hist),
                "train_minutes": fnum(info.get("train_minutes"), 1), "started": info.get("started"), "finished": info.get("finished"),
                "commit": (info.get("git") or {}).get("commit", "")[:7],
                "backbone": (info.get("initialization") or {}).get("file", "").split("/")[-1],
                "ssl_run": ((info.get("initialization") or {}).get("ssl_record") or {}).get("ssl_run_dir"),
                "val_curve": [fnum(h.get("val_acc"), 4) for h in hist],
            }
            r = rows.setdefault(reg["id"], {"compare_label": method, **reg, "cells": {}})
            prev = r["cells"].get(f"{budget}/{split}")
            if cell["acc"] is not None and (prev is None or (cell["started"] or "") > (prev["started"] or "")):
                r["cells"][f"{budget}/{split}"] = cell
    order = [v["id"] for v in registry.CLASSIFICATION.values()]
    return {"task": "ATRNet-STAR SOC-40 classification, official test set, accuracy %", "budgets": ["1pct", "3pct", "10pct"],
            "splits": [1, 2, 3], "methods": [rows[i] for i in order if i in rows], "pools": registry.POOLS}


# ---------------------------------------------------------------- detection (SARDet-100K) ----------------------
def det_registry_key(method):
    """runs.csv method -> registry key: 'CondDINO_SARDet:condcenter_perm' -> 'CondDINO_SARDet (permuted)'."""
    exp, _, variant = method.partition(":")
    if exp == "CondDINO_SARDet":
        return "CondDINO_SARDet (permuted)" if "perm" in variant else "CondDINO_SARDet"
    if exp.startswith("RegionDINO"):
        return f"{exp} (random)" if "random" in variant else exp
    return exp


def detection():
    outs = sorted((REPO / "SARDet" / "compare" / "outputs").glob("20*"))
    if not outs:
        return {"methods": [], "note": "no SARDet/compare/outputs/<timestamp> folder yet"}
    latest = outs[-1]
    rows = {}
    for r in read_csv(latest / "runs.csv"):
        reg = registry.DETECTION.get(det_registry_key(r["method"]))
        if reg is None or r["seed"] != "0":
            continue
        run_dir = Path(r["run_dir"])
        info = read_json(run_dir / "run_info.json") or {}
        hist = read_csv(run_dir / "history.csv")
        per_class = read_json(run_dir / "eval_best_test" / "metrics.json") or {}
        cell = {
            "run": rel(run_dir) if run_dir.exists() else r["run_dir"], "status": info.get("status", "from runs.csv"),
            "map": fnum(r["best_test_map"]), "map50": fnum(r["best_test_map50"]), "map75": fnum(r["best_test_map75"]),
            "map_small": fnum(r["best_test_map_small"]), "map_medium": fnum(r["best_test_map_medium"]), "map_large": fnum(r["best_test_map_large"]),
            "val_select_map": fnum(r["best_val_select_map"]), "best_epoch": int(r["best_epoch"]) if r["best_epoch"] else None,
            "epochs_run": len(hist), "train_minutes": fnum(r["train_minutes"], 1), "started": info.get("started"), "finished": info.get("finished"),
            "commit": (info.get("git") or {}).get("commit", "")[:7], "ssl_run": r.get("ssl_run_dir") or None,
            "ssl_gpu_hours": fnum(r.get("ssl_gpu_hours"), 1),
            "per_class_ap": {k: fnum(v.get("ap")) for k, v in (per_class.get("per_class") or {}).items()},
            "val_curve": [fnum(h.get("val_select_map"), 4) for h in hist if h.get("val_select_map")],
        }
        row = rows.setdefault(reg["id"], {"compare_label": r["method"], **reg, "cells": {}})
        row["cells"][f"{r['budget']}/{r['split']}"] = cell
    per_source = {}
    for r in read_csv(latest / "per_source.csv"):
        reg = registry.DETECTION.get(det_registry_key(r["method"]))
        if reg is None:
            continue
        per_source.setdefault(reg["id"], {})[f"{r['budget']}/{r['split']}"] = {k[:-4]: fnum(v) for k, v in r.items() if k.endswith("_map")}
    order = [v["id"] for v in registry.DETECTION.values()]
    return {"task": "SARDet-100K 6-class detection, official test set, mAP@[.5:.95]", "budgets": ["1pct", "3pct", "10pct"], "splits": [1, 2, 3],
            "classes": ["ship", "aircraft", "car", "tank", "bridge", "harbor"], "compare_folder": rel(latest),
            "methods": [rows[i] for i in order if i in rows], "per_source": per_source, "pools": registry.POOLS}


# ---------------------------------------------------------------- chip classification (SARDet-100K chips) ---------
CHIP_RE = re.compile(r"^\d{8}-\d{6}_S1_(?P<method>.+?)_mobilenetv2_(?P<budget>\d+pct)_split(?P<split>[123])_seed0$")


def chips():
    rows = {}
    for run in sorted((REPO / "SARDet_chips" / "outputs" / "finetune").glob("*")):
        m = CHIP_RE.match(run.name)
        reg = registry.CHIPS.get(m["method"]) if m else None
        if reg is None:
            continue
        info = read_json(run / "run_info.json") or {}
        mc, mt = read_json(run / "eval_test_clean" / "metrics.json"), read_json(run / "eval_test" / "metrics.json")
        if info.get("status") != "completed" or not mc or not mt:
            continue
        hist = read_csv(run / "history.csv")
        cell = {
            "run": rel(run), "status": "completed",
            "acc": round(100 * mc["mean_per_class_accuracy"], 2), "overall": round(100 * mc["overall_accuracy"], 2),
            "test_acc": round(100 * mt["mean_per_class_accuracy"], 2), "test_overall": round(100 * mt["overall_accuracy"], 2),
            "per_class": {c: round(100 * v, 1) for c, v in mc["per_class_accuracy"].items()},
            "per_source": {c: round(100 * v, 1) for c, v in mc["per_source_accuracy"].items()},
            "n_clean": mc["num_chips"], "n_test": mt["num_chips"],
            "best_epoch": info.get("best_epoch"), "epochs_run": len(hist), "train_minutes": fnum(info.get("train_minutes"), 1),
            "started": info.get("started"), "finished": info.get("finished"), "commit": (info.get("git") or {}).get("commit", "")[:7],
            "backbone": (info.get("initialization") or {}).get("file", "").split("/")[-1], "ssl_run": (info.get("initialization") or {}).get("ssl_run_dir"),
            "val_curve": [fnum(h.get("val_mean_class_acc"), 4) for h in hist],
        }
        r = rows.setdefault(reg["id"], {"compare_label": m["method"], **reg, "cells": {}})
        prev = r["cells"].get(f"{m['budget']}/{m['split']}")
        if prev is None or (cell["started"] or "") > (prev["started"] or ""):
            r["cells"][f"{m['budget']}/{m['split']}"] = cell
    order = [v["id"] for v in registry.CHIPS.values()]
    return {"task": "SARDet-100K chip classification (6 classes), mean per-class accuracy % on the clean test chips; overall accuracy and the full test set stored too",
            "budgets": ["1pct", "3pct", "10pct"], "splits": [1, 2, 3], "classes": ["ship", "aircraft", "car", "tank", "bridge", "harbor"],
            "methods": [rows[i] for i in order if i in rows], "pools": registry.POOLS,
            "chip_rule": "square window = max(64 px, 1.5 x longer box side), resized to 128; test_clean = test chips minus images shared with train minus chips holding another class"}


# ---------------------------------------------------------------- response maps ----------------------------------
def maps():
    chips = read_csv(REPO / "SARDet_to_ATR" / "outputs" / "compare" / "response_maps_per_model.csv")
    scenes = read_csv(REPO / "RegionSSL" / "outputs" / "scene_maps" / "scene_maps_per_model.csv")
    src = read_csv(REPO / "RegionSSL" / "outputs" / "scene_maps" / "scene_maps_per_source.csv")
    per_source = []
    for r in src:
        per_source.append({"source": r["source"], "n": int(r["n_scenes"]), "box_area": fnum(r["boxes_area_pct"], 1),
                           "models": {k.split(" | ")[0]: None for k in r if " | " in k}})
        for k, v in r.items():
            if " | " in k:
                model, what = k.split(" | ")
                per_source[-1]["models"].setdefault(model, {})
                if per_source[-1]["models"][model] is None:
                    per_source[-1]["models"][model] = {}
                per_source[-1]["models"][model][what] = fnum(v, 1)
    figs = {
        "chips": rel(REPO / "SARDet_to_ATR" / "outputs" / "compare" / "response_maps_split1.png"),
        "scenes": rel(REPO / "RegionSSL" / "outputs" / "scene_maps" / "scene_maps_split1.png"),
    }
    figs = {k: v for k, v in figs.items() if (REPO / v).exists()}
    return {
        "chips": {"what": "2,000 official ATRNet-STAR test chips, vehicle box = 17.2 % of the chip", "box_area": 17.2,
                  "rows": [{"model": r["model"], "split": int(r["split"]), "mass": fnum(r["mass_in_box_pct"], 1), "peak": fnum(r["peak_in_box_pct"], 1),
                            "concentration": fnum(r["concentration"], 2)} for r in chips]},
        "scenes": {"what": "1,500 official SARDet-100K test scenes with annotations, boxes = 3.4 % of the scene", "box_area": 3.4,
                   "rows": [{"model": r["model"], "split": int(r["split"]), "mass": fnum(r["mass_in_boxes_pct"], 1), "peak": fnum(r["peak_in_boxes_pct"], 1),
                             "concentration": fnum(r["concentration"], 2)} for r in scenes],
                   "per_source": per_source},
        "figures": figs,
    }


# ---------------------------------------------------------------- every run folder ---------------------------------
SSL_GLOBS = ["Experiments/*/outputs/ssl/*", "SARDet/*/outputs/ssl/*", "RegionSSL/*/outputs/ssl/*"]
FT_GLOBS = ["Experiments/*/outputs/finetune/*", "Experiments/Baseline/Baseline_Output/*", "SARDet_to_ATR/outputs/finetune/*",
            "SARDet/*/outputs/finetune/*", "RegionSSL/*/outputs/finetune/*", "SARDet_chips/outputs/finetune/*"]


def curve(hist, col):
    vals = [fnum(h.get(col), 4) for h in hist]
    vals = [v for v in vals if v is not None]
    step = max(1, len(vals) // 100)
    return vals[::step]


def runs():
    out = []
    for kind, globs in (("ssl", SSL_GLOBS), ("finetune", FT_GLOBS)):
        for g in globs:
            for folder in sorted(REPO.glob(g)):
                if not folder.is_dir() or SKIP.search(folder.name) or not (folder / "run_info.json").exists():
                    continue
                info = read_json(folder / "run_info.json") or {}
                cfg = read_json(folder / "config.json") or {}
                hist = read_csv(folder / "history.csv")
                cost = info.get("cost") or {}
                minutes = info.get("train_minutes") or cost.get("train_minutes") or info.get("wall_minutes")
                env = info.get("environment") or {}
                r = {
                    "kind": kind, "name": folder.name, "path": rel(folder), "group": folder.relative_to(REPO).parts[0] + "/" + folder.relative_to(REPO).parts[1],
                    "status": info.get("status", "unknown"), "started": info.get("started"), "finished": info.get("finished"),
                    "minutes": fnum(minutes, 1), "epochs_run": len(hist),
                    "epochs_planned": (cfg.get("common") or {}).get("epochs") or cfg.get("epochs") or (cfg.get("training") or {}).get("epochs"),
                    "gpu": env.get("gpu") or cost.get("device"), "commit": (info.get("git") or {}).get("commit", "")[:7],
                    "deviation": info.get("protocol_deviation"), "split": info.get("split") or cfg.get("split"),
                    "method": info.get("method") or cfg.get("method") or cfg.get("run_name"),
                }
                if kind == "ssl":
                    r["curve"] = curve(hist, "loss"); r["curve_name"] = "SSL loss per epoch"
                    r["exported"] = (info.get("exported_backbone") or {}).get("file")
                    r["gpu_hours"] = fnum(cost.get("gpu_hours") or (float(minutes) / 60 if minutes else None), 1)
                else:
                    col = "val_mean_class_acc" if hist and "val_mean_class_acc" in hist[0] else "val_acc" if hist and "val_acc" in hist[0] else "val_select_map"
                    r["curve"] = curve(hist, col); r["curve_name"] = {"val_acc": "validation accuracy per epoch", "val_select_map": "val-select mAP per 2 epochs", "val_mean_class_acc": "selection mean per-class accuracy per epoch"}[col]
                    r["best_epoch"] = info.get("best_epoch")
                    init = info.get("initialization") or info.get("backbone") or {}
                    r["backbone"] = (init.get("file") or init.get("path") or "").split("/")[-1] if isinstance(init, dict) else str(init)
                out.append(r)
    return {"runs": out, "count": {"ssl": sum(r["kind"] == "ssl" for r in out), "finetune": sum(r["kind"] == "finetune" for r in out)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-compare", action="store_true", help="re-run both compare scripts first (slow: reads every eval file)")
    a = ap.parse_args()
    if a.run_compare:
        subprocess.run([sys.executable, str(REPO / "SARDet_to_ATR" / "compare.py")], check=True, cwd=REPO)
        subprocess.run([sys.executable, str(REPO / "SARDet" / "compare" / "compare.py")], check=True, cwd=REPO / "SARDet")
    OUT.mkdir(parents=True, exist_ok=True)
    data = {"classification": classification(), "detection": detection(), "chips": chips(), "maps": maps(), "runs": runs()}
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    except OSError:
        commit = ""
    data["meta"] = {"collected": datetime.now().isoformat(timespec="seconds"), "commit": commit, "repo": str(REPO),
                    "settings": registry.SETTINGS, "families": registry.FAMILIES, "pools": registry.POOLS}
    for name, d in data.items():
        (OUT / f"{name}.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    c, d, r = data["classification"], data["detection"], data["runs"]
    print(f"classification: {len(c['methods'])} methods, {sum(len(m['cells']) for m in c['methods'])} cells")
    print(f"detection: {len(d['methods'])} methods, {sum(len(m['cells']) for m in d['methods'])} cells ({d.get('compare_folder')})")
    print(f"chips: {len(data['chips']['methods'])} methods, {sum(len(m['cells']) for m in data['chips']['methods'])} cells")
    print(f"maps: {len(data['maps']['chips']['rows'])} chip rows, {len(data['maps']['scenes']['rows'])} scene rows")
    print(f"runs: {r['count']}  -> {OUT}")


if __name__ == "__main__":
    main()
