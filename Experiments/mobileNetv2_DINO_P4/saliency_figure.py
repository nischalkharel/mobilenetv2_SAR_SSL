"""Where do the models respond: vehicle box versus the rest of the chip, A1 against P4, measured and drawn.

Two kinds of map, both 7 x 7 at the end of MobileNetV2's backbone (one cell = 32 x 32 input pixels, 18 x 18 pixels of
the 128 x 128 chip), upsampled bilinearly for display and for the box measure:
  Grad-CAM        of the 1 % fine-tuned model for the chip's TRUE class (gradient of that logit with respect to the last
                  feature map, channel weights = mean gradient, ReLU of the weighted sum). Where the classifier's
                  evidence sits.
  response map    of the frozen SSL backbone: L2 norm over channels of the last feature map. Where the backbone's
                  activity sits, no labels involved.
Measure, over a random sample of test chips: the share of a map's mass inside the dataset's vehicle box (the .xml
annotation), compared with the box's share of the chip area (what a uniform map would give); the ratio of the two is
the concentration on the vehicle. The box holds the vehicle's bright returns; the radar shadow lies outside it, so a
map on the shadow counts as "outside" here. Reported per model, with the mean over chips and the share of chips where
P4's map puts more mass on the vehicle than A1's.
Figure: a fixed set of test chips (one per scene, seed 0, not chosen by result) with the box drawn, and the four maps.

Run from the repository root (GPU, about a minute):
    python Experiments/mobileNetv2_DINO_P4/saliency_figure.py
    python Experiments/mobileNetv2_DINO_P4/saliency_figure.py --chips 2000 --rows 6
Writes outputs/compare/saliency_summary.md, saliency_per_chip.csv, saliency_figure.png / .pdf.
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import paths
from analysis_data import BACKBONES, FINETUNED_1PCT, chip_loader, load_backbone_model, load_finetuned_model, test_rows
from data import NATIVE_SIZE, load_class_names, load_image
from utils import resolve_device, set_seed


def grad_cam(model, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, 7, 7) Grad-CAM for the true classes y, and the predicted classes. model in eval mode."""
    x = x.clone().requires_grad_(True)                  # parameters are frozen; the gradient reaches A through the input
    A = model.features(x)                               # (B, 1280, 7, 7)
    A.retain_grad()
    logits = model.classifier(A.mean((2, 3)))           # torchvision's forward: global average pool -> classifier
    logits.gather(1, y[:, None]).sum().backward()
    w = A.grad.mean((2, 3), keepdim=True)               # (B, 1280, 1, 1)
    cam = F.relu((w * A).sum(1))                        # (B, 7, 7)
    return cam.detach(), logits.argmax(1).detach()


@torch.no_grad()
def response_map(model, x: torch.Tensor) -> torch.Tensor:
    return model.features(x).norm(dim=1)                # (B, 7, 7)


def box_share(maps: torch.Tensor, boxes: list[dict]) -> np.ndarray:
    """Share of each map's mass inside its chip's vehicle box (maps upsampled to the chip frame; zero map -> nan)."""
    up = F.interpolate(maps[:, None].float(), size=(NATIVE_SIZE, NATIVE_SIZE), mode="bilinear", align_corners=False)[:, 0]
    out = []
    for m, b in zip(up, boxes):
        total = float(m.sum())
        inside = float(m[b["ymin"]:b["ymax"] + 1, b["xmin"]:b["xmax"] + 1].sum())
        out.append(inside / total if total > 0 else float("nan"))
    return np.array(out)


def box_area_share(b: dict) -> float:
    return (b["xmax"] - b["xmin"] + 1) * (b["ymax"] - b["ymin"] + 1) / (NATIVE_SIZE * NATIVE_SIZE)


def main():
    ap = argparse.ArgumentParser(description="Grad-CAM and response maps, A1 vs P4, with the vehicle-box measure")
    ap.add_argument("--chips", type=int, default=2000, help="random test chips for the measure")
    ap.add_argument("--rows", type=int, default=6, help="chips in the figure (one per scene first, seed 0)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone", action="append", default=[], metavar="A1|P4=PATH", help="override a backbone file")
    ap.add_argument("--finetuned", action="append", default=[], metavar="A1|P4=PATH", help="override a 1 %% fine-tuned best.pt")
    a = ap.parse_args()
    backbones, finetuned = dict(BACKBONES), dict(FINETUNED_1PCT)
    for spec in a.backbone:
        n, path = spec.split("=", 1)
        backbones[n] = Path(path)
    for spec in a.finetuned:
        n, path = spec.split("=", 1)
        finetuned[n] = Path(path)
    device = resolve_device(a.device)
    set_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    class_names = load_class_names()
    test = test_rows(class_names)
    idx = np.sort(rng.choice(len(test), a.chips, replace=False))
    sample = [test[i] for i in idx]

    models = {}
    for name in ("A1", "P4"):
        models[f"{name} fine-tuned 1 %"] = load_finetuned_model(finetuned[name], len(class_names)).to(device).eval()
        models[f"{name} backbone"] = load_backbone_model(name, backbones[name], len(class_names)).to(device).eval()
    for p in (q for m in models.values() for q in m.parameters()):
        p.requires_grad_(False)

    shares = {k: [] for k in models}
    correct = {k: [] for k in models if "fine-tuned" in k}
    maps_fig = {k: {} for k in models}
    boxes = [{k: r[k] for k in ("xmin", "xmax", "ymin", "ymax")} for r in sample]
    # figure chips: the first sampled chip of each scene, then more scenes' second chips until `rows` are chosen
    scenes = sorted({r["scene"] for r in sample})
    fig_pos = []
    for round_ in range(3):
        for sc in scenes:
            cands = [i for i, r in enumerate(sample) if r["scene"] == sc]
            if len(cands) > round_ and len(fig_pos) < a.rows:
                fig_pos.append(cands[round_])
    fig_pos = sorted(set(fig_pos))

    pos = 0
    for x, y in chip_loader(sample, batch_size=64):
        x, y = x.to(device), y.to(device)
        b = boxes[pos:pos + len(y)]
        for name, model in models.items():
            if "fine-tuned" in name:
                m, pred = grad_cam(model, x, y)
                correct[name] += (pred == y).cpu().tolist()
            else:
                m = response_map(model, x)
            shares[name] += box_share(m, b).tolist()
            for i in range(len(y)):
                if pos + i in fig_pos:
                    maps_fig[name][pos + i] = m[i].cpu().numpy()
        pos += len(y)

    area = np.array([box_area_share(b) for b in boxes])
    lines = ["# Map mass on the vehicle box: A1 versus P4 (split 1), random test chips", "",
             f"{a.chips} official test chips (seed {a.seed}); box = the dataset's vehicle annotation, mean box area "
             f"{100 * area.mean():.1f} % of the chip (what a uniform map would score). Grad-CAM: 1 % fine-tuned model, true class.",
             "Response map: frozen SSL backbone, channel norm of the last feature map. Concentration = mass share / box area share.", "",
             "| Map | Mass in box (%) | Concentration on vehicle | Accuracy on these chips (%) |", "|---|---|---|---|"]
    rows_csv = []
    for name in models:
        s = np.array(shares[name])
        acc = f"{100 * np.mean(correct[name]):.1f}" if name in correct else "-"
        lines.append(f"| {name} | {100 * np.nanmean(s):.1f} | {np.nanmean(s / area):.2f} | {acc} |")
    for kind in ("fine-tuned 1 %", "backbone"):
        d = np.array(shares[f"P4 {kind}"]) - np.array(shares[f"A1 {kind}"])
        lines.append(f"\n{kind}: P4 minus A1 mass in box, mean {100 * np.nanmean(d):+.1f} points; P4 higher on "
                     f"{100 * np.nanmean(d > 0):.0f} % of chips.")
    for i, r in enumerate(sample):
        rows_csv.append({"relative_path": r["relative_path"], "class": r["class_name"], "scene": r["scene"], "azimuth": r["azimuth"],
                         "box_area_share": round(area[i], 4), **{f"{k}_box_share": round(shares[k][i], 4) for k in models},
                         **{f"{k}_correct": int(correct[k][i]) for k in correct}})
    lines.append(f"\nWritten {datetime.now().isoformat(timespec='seconds')}.")

    # ---- figure ----
    cols = ["chip + vehicle box"] + list(models)
    fig, axes = plt.subplots(len(fig_pos), len(cols), figsize=(2.3 * len(cols), 2.35 * len(fig_pos)))
    axes = np.atleast_2d(axes)
    for ri, i in enumerate(fig_pos):
        r = sample[i]
        img = np.asarray(load_image(r["relative_path"]), dtype=np.float32)
        b = boxes[i]
        for ci, col in enumerate(cols):
            ax = axes[ri, ci]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            if ci > 0:
                m = maps_fig[col][i]
                up = F.interpolate(torch.tensor(m)[None, None], size=(NATIVE_SIZE, NATIVE_SIZE), mode="bilinear", align_corners=False)[0, 0].numpy()
                ax.imshow(up / (up.max() + 1e-12), cmap="jet", alpha=0.45, vmin=0, vmax=1)
                ax.set_title(f"{col}\nbox {100 * shares[col][i]:.0f} %", fontsize=7.5)
            else:
                ax.set_title(f"{r['class_name'].replace('_', ' ')}\n{r['scene']}, {r['band']}, dep {r['depression']}, az {r['azimuth']}", fontsize=7.5)
            ax.add_patch(Rectangle((b["xmin"] - 0.5, b["ymin"] - 0.5), b["xmax"] - b["xmin"] + 1, b["ymax"] - b["ymin"] + 1,
                                   fill=False, edgecolor="white", linewidth=1.0))
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Where the models respond: Grad-CAM (fine-tuned, true class) and backbone response maps, A1 vs P4; white box = vehicle annotation",
                 fontsize=9)
    fig.tight_layout()
    out = paths.OUTPUT_DIR / "compare"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "saliency_figure.png", dpi=200)
    fig.savefig(out / "saliency_figure.pdf")
    (out / "saliency_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "saliency_per_chip.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)
    print("\n".join(lines))
    print(f"\nWritten to {out}: saliency_summary.md, saliency_per_chip.csv, saliency_figure.png / .pdf")


if __name__ == "__main__":
    main()
