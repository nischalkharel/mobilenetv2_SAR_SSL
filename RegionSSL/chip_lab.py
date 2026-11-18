"""Does the region rule (felz_grouped) find the vehicle on ATRNet-STAR chips? A lab check before SSL on that pool.

    python RegionSSL/chip_lab.py --n 300 --sizes 128,224 --rows 8

Chips come from the ATRNet SSL pool CSV (train chips); the vehicle box of each chip (its .xml sidecar) is read ONLY to
score the rule, exactly as box_zoom.py does on SARDet: cover = share of the box painted by any region, IoU = best single
region vs the box, HIT = IoU >= 0.3 or cover >= 0.5, TOUCHED = cover > 0. The rule is run at each working size in
--sizes (the chip is resized, the box scaled). Output: RegionSSL/outputs/chip_lab/summary.md + chips_<size>.png.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "Experiments" / "mobileNetv2_DINO_P4"))
import segment_lab as lab  # noqa: E402
from box_zoom import per_box  # noqa: E402
import analysis_data as ad  # noqa: E402  (parse_xml only)
import paths as atr_paths  # noqa: E402  (the P4 copy of the ATRNet paths module)
from data import load_image  # noqa: E402


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--sizes", default="128,224")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default="felz_grouped")
    a = ap.parse_args()
    rows = ad.pool_rows()
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.n]
    out = HERE / "outputs" / "chip_lab"
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# Region rule `{a.method}` on {len(rows)} ATRNet-STAR pool chips (seed {a.seed})", ""]
    for size in [int(s) for s in a.sizes.split(",")]:
        stats = []
        panel = []
        for i, r in enumerate(rows):
            im = load_image(r["relative_path"])
            s = size / im.size[0]
            if size != im.size[0]:
                im = im.resize((size, size), Image.BILINEAR)
            img = np.asarray(im, dtype=np.float32) / 255.0
            labels = lab.METHODS[a.method](img)
            b = ad.parse_xml((atr_paths.SOC40_DIR / r["relative_path"]).with_suffix(".xml"))
            box = (b["xmin"] * s, b["ymin"] * s, (b["xmax"] + 1) * s, (b["ymax"] + 1) * s)
            cover, iou = per_box(labels, box)
            n_reg = int(labels.max())
            ext = []
            for k in range(1, n_reg + 1):
                ys, xs = np.nonzero(labels == k)
                ext.append(max(np.ptp(ys) + 1, np.ptp(xs) + 1))
            stats.append((cover, iou, n_reg, float(np.median(ext)) if ext else 0.0, (box[2] - box[0]) * (box[3] - box[1]) / size / size))
            if i < a.rows:
                panel.append((img, labels, box, r["class_name"], cover, iou))
        st = np.array(stats)
        hit = np.mean((st[:, 1] >= 0.3) | (st[:, 0] >= 0.5))
        touched = np.mean(st[:, 0] > 0)
        lines += [f"## working size {size} px", "",
                  f"- hit (IoU >= 0.3 or cover >= 0.5): {100 * hit:.0f} %; touched (cover > 0): {100 * touched:.0f} %",
                  f"- mean cover {100 * st[:, 0].mean():.0f} %, median best IoU {np.median(st[:, 1]):.2f}",
                  f"- regions per chip: median {np.median(st[:, 2]):.0f} (5-95 %: {np.percentile(st[:, 2], 5):.0f}-{np.percentile(st[:, 2], 95):.0f}); "
                  f"median region extent {np.median(st[:, 3]):.0f} px; vehicle box = {100 * st[:, 4].mean():.0f} % of the chip", ""]
        fig, axes = plt.subplots(len(panel), 2, figsize=(5.2, 2.5 * len(panel)))
        axes = np.atleast_2d(axes)
        for k, (img, labels, box, cls, cover, iou) in enumerate(panel):
            axes[k, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[k, 0].set_title(f"{cls.replace('_', ' ')}", fontsize=7)
            axes[k, 1].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[k, 1].imshow(np.ma.masked_equal(labels, 0), cmap="tab20", alpha=0.6, interpolation="nearest")
            axes[k, 1].set_title(f"{int(labels.max())} regions, cover {100 * cover:.0f} %, IoU {iou:.2f}", fontsize=7)
            for ax in axes[k]:
                ax.add_patch(Rectangle((box[0] - 0.5, box[1] - 0.5), box[2] - box[0], box[3] - box[1], fill=False, edgecolor="white", linewidth=1))
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(f"{a.method} on ATRNet chips at {size} px; white box = vehicle annotation", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"chips_{size}.png", dpi=170)
        plt.close(fig)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
