"""Look at the DINO views before trusting them.

1. save_view_preview(): one row per image: the original chip, then the 2 global and the local views
   produced by the exact transform used in training (called automatically by train_ssl.py).
2. crop_coverage(): how much of the vehicle each crop keeps. It reads the vehicle bounding box
   from each image's XML (an annotation, used here for this check only, never for training) and
   samples crop boxes from the same RandomResizedCrop distribution that training uses. It measures
   the distribution of crops; it does not track the particular crops of a particular training step.

Run from the repository root:
    python Experiments/mobileNetv2_DINO/preview_views.py                       # uses config_ssl.json
    python Experiments/mobileNetv2_DINO/preview_views.py --images 3000 --out <folder>
Writes view_preview.png and crop_coverage.json (default folder: outputs/view_check/).
"""
import argparse
import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision import transforms as T  # noqa: E402

import paths  # noqa: E402
from data import NATIVE_SIZE, load_image  # noqa: E402
from dino import DataAugmentationDINO, denormalize  # noqa: E402
from ssl_data import flat_args, read_ssl_paths  # noqa: E402
from utils import save_json  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_ssl.json"


def save_view_preview(relative_paths: list[str], args, out_path: Path, rows: int = 6, seed: int = 0) -> None:
    rng = random.Random(seed)
    chosen = rng.sample(relative_paths, min(rows, len(relative_paths)))
    aug = DataAugmentationDINO(args)
    n_cols = 1 + 2 + args.local_crops_number
    fig, axes = plt.subplots(len(chosen), n_cols, figsize=(1.6 * n_cols, 1.7 * len(chosen)), squeeze=False)
    for r, p in enumerate(chosen):
        img = load_image(p)
        views = aug(img)
        panels = [np.asarray(img)] + [denormalize(v, args.normalize_mean, args.normalize_std)[0].numpy() for v in views]
        titles = [f"original {NATIVE_SIZE}"] + [f"global {args.global_crop_size}"] * 2 + \
                 [f"local {args.local_crop_size}"] * args.local_crops_number
        for c, (panel, title) in enumerate(zip(panels, titles)):
            ax = axes[r, c]
            ax.imshow(panel, cmap="gray", vmin=0, vmax=255 if c == 0 else 1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=7)
        axes[r, 0].set_ylabel(Path(p).parent.name, fontsize=6)
    fig.suptitle("DINO views used in training (each panel shown at the same display size; "
                 "pixel sizes in the titles)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def vehicle_box(relative_path: str) -> tuple[float, float, float, float]:
    xml = (paths.SOC40_DIR / relative_path).with_suffix(".xml").read_text(encoding="utf-8")
    b = {k: float(re.search(rf"<{k}>\s*(-?[\d.]+)", xml).group(1)) for k in ["xmin", "ymin", "xmax", "ymax"]}
    return b["xmin"], b["ymin"], b["xmax"], b["ymax"]


def crop_coverage(relative_paths: list[str], scale, n_images: int, crops_per_image: int, seed: int = 0) -> dict:
    """Fraction of the vehicle box area that falls inside a crop, over sampled crops."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    chosen = rng.sample(relative_paths, min(n_images, len(relative_paths)))
    dummy = torch.empty(1, NATIVE_SIZE, NATIVE_SIZE)  # get_params only needs the image size
    cover = []
    for p in chosen:
        x0, y0, x1, y1 = vehicle_box(p)
        area = max(x1 - x0, 1e-6) * max(y1 - y0, 1e-6)
        for _ in range(crops_per_image):
            top, left, h, w = T.RandomResizedCrop.get_params(dummy, scale=list(scale), ratio=[3 / 4, 4 / 3])
            ix = max(0.0, min(x1, left + w) - max(x0, left))
            iy = max(0.0, min(y1, top + h) - max(y0, top))
            cover.append(ix * iy / area)
    c = np.array(cover)
    return {
        "crop_area_scale": list(scale),
        "crops_sampled": len(c),
        "vehicle_kept_median": round(float(np.median(c)), 3),
        "share_keeping_at_least_90pct": round(float((c >= 0.9).mean()), 3),
        "share_keeping_at_least_50pct": round(float((c >= 0.5).mean()), 3),
        "share_keeping_less_than_25pct": round(float((c < 0.25).mean()), 3),
        "share_with_no_vehicle_at_all": round(float((c == 0).mean()), 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Preview DINO views and measure vehicle coverage of crops")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--images", type=int, default=2000, help="images sampled for the coverage check")
    parser.add_argument("--out", type=Path, default=paths.OUTPUT_DIR / "view_check")
    args_cli = parser.parse_args()
    args = flat_args(json.loads(args_cli.config.read_text(encoding="utf-8")))
    ssl_paths = read_ssl_paths(paths.SPLITS_DIR / args.ssl_csv)
    args_cli.out.mkdir(parents=True, exist_ok=True)
    save_view_preview(ssl_paths, args, args_cli.out / "view_preview.png")
    box = np.array([vehicle_box(p) for p in random.Random(1).sample(ssl_paths, 500)])
    sizes = {"vehicle_box_width_px_median": float(np.median(box[:, 2] - box[:, 0])),
             "vehicle_box_height_px_median": float(np.median(box[:, 3] - box[:, 1])),
             "vehicle_box_area_share_median": round(float(np.median(
                 (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]) / NATIVE_SIZE ** 2)), 3)}
    result = {"config": str(args_cli.config), "vehicle_box_sizes_500_images": sizes,
              "global": crop_coverage(ssl_paths, args.global_crops_scale, args_cli.images, 2),
              "local": crop_coverage(ssl_paths, args.local_crops_scale, args_cli.images, args.local_crops_number),
              "note": "vehicle boxes from the XML annotations, used only for this check"}
    save_json(result, args_cli.out / "crop_coverage.json")
    print(json.dumps(result, indent=2))
    print(f"Wrote {args_cli.out / 'view_preview.png'} and {args_cli.out / 'crop_coverage.json'}")


if __name__ == "__main__":
    main()
