"""Where do the frozen SARDet-trained backbones respond on ATRNet-STAR test chips, before any fine-tuning?

For every SSL backbone of the cross-dataset study (SSL on SARDet-100K scenes: DINO, LeJEPA, SparK, RegionDINO
scene+object, RegionDINO object-only, and the two random-window controls), plus the ImageNet start and, when its
file is on this machine, the in-domain A1 DINO backbone (SSL on ATRNet chips):
  response map  = L2 norm over channels of the last feature map (7 x 7 for a 224 input; one cell = 18 x 18 pixels of
                  the 128 x 128 chip), upsampled bilinearly. Where the frozen backbone's activity sits. No labels and
                  no fine-tuning are involved in the map; the vehicle box (the dataset's .xml annotation) is read only
                  to measure it.
Measures over a fixed random sample of official test chips (same chips for every model and split, so every
comparison is paired per chip):
  mass in box     share of the map's mass inside the vehicle box (a uniform map scores the box's area share)
  concentration   mass in box / box area share (1 = no preference for the vehicle)
  peak in box     share of chips whose strongest pixel of the upsampled map lies inside the box
and paired differences per chip against the ImageNet start and against XA DINO of the same split.
Pairing rule as in the fine-tunes: ATRNet split s uses the SARDet SSL run of split s (find_backbone.py).

Run from the repository root (GPU, a few minutes for 3 splits):
    python SARDet_to_ATR/response_maps.py
    python SARDet_to_ATR/response_maps.py --chips 2000 --rows 6 --splits 1,2,3 --figure-split 1
    python SARDet_to_ATR/response_maps.py --chips 64 --rows 3 --splits 1 --model "name=path/to/backbone.pth" --out outputs/smoke
Writes SARDet_to_ATR/outputs/compare/response_maps.md, response_maps_per_model.csv, response_maps_per_chip.csv and
response_maps_split<k>.png / .pdf. Analysis only: nothing here is used by training.
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "Experiments" / "mobileNetv2_DINO_P4"))   # analysis_data + its paths/data/model (read-only reuse)
sys.path.insert(0, str(HERE))
import analysis_data as ad  # noqa: E402
from data import NATIVE_SIZE, load_class_names, load_image  # noqa: E402
from find_backbone import find  # noqa: E402
from utils import resolve_device, set_seed  # noqa: E402

# table order; label -> find_backbone method key
SARDET_METHODS = [("XA DINO on SARDet", "dino"), ("XA LeJEPA on SARDet", "lejepa"), ("XA SparK on SARDet", "spark"),
                  ("RegionDINO scene+object", "regiondino_so"), ("RegionDINO scene+object, random windows", "regiondino_so_random"),
                  ("RegionDINO object-only", "regiondino_o"), ("RegionDINO object-only, random windows", "regiondino_o_random")]
REFERENCE = "XA DINO on SARDet"
IMAGENET = "ImageNet start (no SSL)"
A1 = "A1 DINO on ATRNet (in-domain, split 1)"


def models_for_split(split: int, overrides: dict, only: set | None) -> dict:
    """label -> backbone path (None = ImageNet). Missing SARDet runs are skipped with a note, never substituted."""
    table = {IMAGENET: None}
    if split == 1 and Path(ad.BACKBONES["A1"]).exists():
        table[A1] = ad.BACKBONES["A1"]
    for label, key in SARDET_METHODS:
        try:
            table[label] = find(key, split)
        except SystemExit as e:
            print(f"  split {split}: skipped {label}: {e}")
    table.update(overrides)
    if only:
        table = {k: v for k, v in table.items() if k in only}
    return table


@torch.no_grad()
def response_map(model, x: torch.Tensor) -> torch.Tensor:
    return model.features(x).norm(dim=1)                                      # (B, 7, 7)


def upsample(maps: torch.Tensor) -> torch.Tensor:
    return F.interpolate(maps[:, None].float(), size=(NATIVE_SIZE, NATIVE_SIZE), mode="bilinear", align_corners=False)[:, 0]


def box_measures(maps: torch.Tensor, boxes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per chip: share of map mass inside the box (nan for an all-zero map), and whether the map's peak is inside."""
    up = upsample(maps).cpu()
    share, peak = [], []
    for m, b in zip(up, boxes):
        total = float(m.sum())
        inside = float(m[b["ymin"]:b["ymax"] + 1, b["xmin"]:b["xmax"] + 1].sum())
        share.append(inside / total if total > 0 else float("nan"))
        py, px = np.unravel_index(int(m.argmax()), m.shape)
        peak.append(float(b["ymin"] <= py <= b["ymax"] and b["xmin"] <= px <= b["xmax"]))
    return np.array(share), np.array(peak)


def box_area_share(b: dict) -> float:
    return (b["xmax"] - b["xmin"] + 1) * (b["ymax"] - b["ymin"] + 1) / (NATIVE_SIZE * NATIVE_SIZE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chips", type=int, default=2000, help="random official test chips for the measures (seed-fixed, shared by all models)")
    ap.add_argument("--rows", type=int, default=6, help="chips drawn in the figure (one per scene first)")
    ap.add_argument("--splits", default="1,2,3")
    ap.add_argument("--figure-split", type=int, default=1, help="which split's backbones to draw (one figure)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", action="append", default=[], metavar="LABEL=PATH", help="add or override a backbone file (any split)")
    ap.add_argument("--only", default="", help="comma-separated labels to keep (after --model)")
    ap.add_argument("--out", default="outputs/compare", help="output folder, relative to SARDet_to_ATR/")
    a = ap.parse_args()
    overrides = {}
    for spec in a.model:
        n, p = spec.split("=", 1)
        overrides[n] = Path(p)
    only = {s.strip() for s in a.only.split(",") if s.strip()} or None
    splits = [int(s) for s in a.splits.split(",")]
    device = resolve_device(a.device)
    set_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    class_names = load_class_names()
    test = ad.test_rows(class_names)
    idx = np.sort(rng.choice(len(test), min(a.chips, len(test)), replace=False))
    sample = [test[i] for i in idx]
    boxes = [{k: r[k] for k in ("xmin", "xmax", "ymin", "ymax")} for r in sample]
    area = np.array([box_area_share(b) for b in boxes])
    scenes = sorted({r["scene"] for r in sample})
    fig_pos = []
    for round_ in range(3):
        for sc in scenes:
            cands = [i for i, r in enumerate(sample) if r["scene"] == sc]
            if len(cands) > round_ and len(fig_pos) < a.rows:
                fig_pos.append(cands[round_])
    fig_pos = sorted(set(fig_pos))
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)

    lines = ["# Frozen backbones on ATRNet-STAR test chips: where the response sits, before fine-tuning", "",
             f"{len(sample)} official test chips (seed {a.seed}, the same chips for every model and split); box = the dataset's "
             f"vehicle annotation, mean box area {100 * area.mean():.1f} % of the chip (what a uniform map scores as mass in box). "
             "Response map = channel norm of the last feature map of the frozen backbone, no labels, no fine-tuning. "
             "Concentration = mass in box / box area share. Peak in box = share of chips whose strongest pixel lies in the box.", ""]
    per_model, per_chip = [], {}
    for split in splits:
        print(f"split {split}")
        table = models_for_split(split, overrides, only)
        shares, peaks, maps_fig = {}, {}, {}
        for label, path in table.items():
            model = ad.load_backbone_model(label, path, len(class_names)).to(device).eval()
            s_all, p_all, keep = [], [], {}
            pos = 0
            for x, _ in ad.chip_loader(sample, batch_size=128):
                m = response_map(model, x.to(device, non_blocking=True))
                s, p = box_measures(m, boxes[pos:pos + len(m)])
                s_all += s.tolist()
                p_all += p.tolist()
                for i in range(len(m)):
                    if pos + i in fig_pos:
                        keep[pos + i] = m[i].cpu().numpy()
                pos += len(m)
            shares[label], peaks[label], maps_fig[label] = np.array(s_all), np.array(p_all), keep
            print(f"  {label}: mass in box {100 * np.nanmean(shares[label]):.1f} %")
            del model
        lines += [f"## Split {split}", "",
                  "| backbone | mass in box (%) | concentration | peak in box (%) | vs ImageNet start: mean, % chips higher | "
                  f"vs {REFERENCE}: mean, % chips higher |", "|---|---|---|---|---|---|"]
        for label in table:
            s = shares[label]
            row = {"split": split, "model": label, "mass_in_box_pct": round(100 * float(np.nanmean(s)), 2),
                   "concentration": round(float(np.nanmean(s / area)), 3), "peak_in_box_pct": round(100 * float(peaks[label].mean()), 1)}
            cells = []
            for ref in (IMAGENET, REFERENCE):
                if ref in shares and ref != label:
                    d = s - shares[ref]
                    row[f"vs_{'imagenet' if ref == IMAGENET else 'xa_dino'}_mean_pts"] = round(100 * float(np.nanmean(d)), 2)
                    row[f"vs_{'imagenet' if ref == IMAGENET else 'xa_dino'}_higher_pct"] = round(100 * float(np.nanmean(d > 0)), 1)
                    cells.append(f"{100 * np.nanmean(d):+.1f}, {100 * np.nanmean(d > 0):.0f} %")
                else:
                    cells.append("-")
            lines.append(f"| {label} | {row['mass_in_box_pct']:.1f} | {row['concentration']:.2f} | {row['peak_in_box_pct']:.1f} | {cells[0]} | {cells[1]} |")
            per_model.append(row)
            for i, r in enumerate(sample):
                per_chip.setdefault((split, r["relative_path"]), {"split": split, "relative_path": r["relative_path"], "class": r["class_name"],
                                                                  "scene": r["scene"], "box_area_share": round(area[i], 4)})[f"{label} | mass"] = round(float(s[i]), 4)
        lines.append("")
        if split == a.figure_split and fig_pos:
            cols = ["chip + vehicle box"] + list(table)
            fig, axes = plt.subplots(len(fig_pos), len(cols), figsize=(2.1 * len(cols), 2.3 * len(fig_pos)))
            axes = np.atleast_2d(axes)
            for ri, i in enumerate(fig_pos):
                r = sample[i]
                img = np.asarray(load_image(r["relative_path"]), dtype=np.float32)
                b = boxes[i]
                for ci, col in enumerate(cols):
                    ax = axes[ri, ci]
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                    if ci > 0:
                        up = upsample(torch.tensor(maps_fig[col][i])[None])[0].numpy()
                        ax.imshow(up / (up.max() + 1e-12), cmap="jet", alpha=0.45, vmin=0, vmax=1)
                        ax.set_title(f"{col}\nbox {100 * shares[col][i]:.0f} %", fontsize=6.5)
                    else:
                        ax.set_title(f"{r['class_name'].replace('_', ' ')}\n{r['scene']}, {r['band']}, dep {r['depression']}, az {r['azimuth']}", fontsize=6.5)
                    ax.add_patch(Rectangle((b["xmin"] - 0.5, b["ymin"] - 0.5), b["xmax"] - b["xmin"] + 1, b["ymax"] - b["ymin"] + 1,
                                           fill=False, edgecolor="white", linewidth=1.0))
                    ax.set_xticks([])
                    ax.set_yticks([])
            fig.suptitle(f"Frozen backbone response maps on ATRNet-STAR test chips (split {split} backbones, no fine-tuning); "
                         "white box = vehicle annotation", fontsize=9)
            fig.tight_layout()
            fig.savefig(out / f"response_maps_split{split}.png", dpi=200)
            fig.savefig(out / f"response_maps_split{split}.pdf")
            plt.close(fig)
    lines.append(f"Written {datetime.now().isoformat(timespec='seconds')}.")
    (out / "response_maps.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    keys = sorted({k for r in per_model for k in r}, key=lambda k: (k not in ("split", "model"), k))
    with open(out / "response_maps_per_model.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(per_model)
    chip_rows = list(per_chip.values())
    keys = sorted({k for r in chip_rows for k in r}, key=lambda k: (k not in ("split", "relative_path", "class", "scene", "box_area_share"), k))
    with open(out / "response_maps_per_chip.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(chip_rows)
    print("\n".join(lines))
    print(f"\nWritten to {out}: response_maps.md, response_maps_per_model.csv, response_maps_per_chip.csv, response_maps_split{a.figure_split}.png / .pdf")


if __name__ == "__main__":
    main()
