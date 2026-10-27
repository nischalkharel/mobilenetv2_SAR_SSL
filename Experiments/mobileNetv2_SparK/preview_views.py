"""Look at SparK's views, masks and reconstructions before trusting them (adapted from the LeJEPA experiment).

1. save_view_preview(): one row per image: the original chip, the training view (crop + flip), and the
   masked view the encoder sees (called automatically by train_ssl.py at the start of a run).
2. save_reconstruction_preview(): original, masked input and reconstruction for fixed chips and fixed
   masks (called by train_ssl.py at every backbone export, so the pictures are comparable across epochs).
3. main(): measurements on many chips, written to crop_and_mask_coverage.json:
   - vehicle coverage: how much of the vehicle each crop keeps, and how much of it stays VISIBLE after
     masking. It reads the vehicle bounding box from each image's XML (an annotation, used here for this
     check only, never for training) and samples crops and masks from the same distributions as training.
   - target check: the share of the per-patch normalized target that would come from the per-channel
     ImageNet constants if all 3 input channels were reconstructed (the reason for recon_channels = 1).

Run from the repository root:
    python Experiments/mobileNetv2_SparK/preview_views.py                       # uses config_ssl.json
    python Experiments/mobileNetv2_SparK/preview_views.py --images 3000 --out <folder>
Writes view_preview.png and crop_and_mask_coverage.json (default folder: outputs/view_check/).
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
from torchvision.transforms import functional as TF  # noqa: E402

import paths  # noqa: E402
from data import NATIVE_SIZE, load_image  # noqa: E402
from spark_model import DOWNSAMPLE_RATIO, DataAugmentationSparK, to_network_input  # noqa: E402
from ssl_data import flat_args, read_ssl_paths  # noqa: E402
from utils import save_json  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_ssl.json"


def random_mask(n_patches: int, n_visible: int, generator: torch.Generator) -> torch.Tensor:
    """Same rule as SparK.mask (spark_lib/spark.py L81-L85): a uniformly random set of n_visible patches."""
    keep = torch.rand(n_patches, generator=generator).argsort()[:n_visible]
    active = torch.zeros(n_patches, dtype=torch.bool)
    active[keep] = True
    return active


def visible_count(args) -> int:
    """Visible patches per image, computed exactly as SparK.__init__ does (spark.py L29)."""
    f = args.input_size // DOWNSAMPLE_RATIO
    return round(f * f * (1 - args.mask_ratio))


def save_view_preview(relative_paths: list[str], args, out_path: Path, rows: int = 6, seed: int = 0) -> None:
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    chosen = rng.sample(relative_paths, min(rows, len(relative_paths)))
    aug = DataAugmentationSparK(args)
    f = args.input_size // DOWNSAMPLE_RATIO
    fig, axes = plt.subplots(len(chosen), 3, figsize=(5.4, 1.8 * len(chosen)), squeeze=False)
    for r, p in enumerate(chosen):
        img = load_image(p)
        view = aug(img)[0].numpy()  # uint8 (H, W)
        active = random_mask(f * f, visible_count(args), g).view(f, f)
        pixel_mask = active.repeat_interleave(DOWNSAMPLE_RATIO, 0).repeat_interleave(DOWNSAMPLE_RATIO, 1).numpy()
        masked = np.where(pixel_mask, view, 0)
        for c, (panel, title) in enumerate([(np.asarray(img), f"original {NATIVE_SIZE}"),
                                            (view, f"view {args.input_size}"),
                                            (masked, f"encoder input ({active.sum().item()}/{f * f} visible)")]):
            ax = axes[r, c]
            ax.imshow(panel, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=7)
        axes[r, 0].set_ylabel(Path(p).parent.name, fontsize=6)
    fig.suptitle("SparK view and mask used in training (hidden patches shown black)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def save_reconstruction_preview(model, relative_paths: list[str], args, device, out_path: Path,
                                rows: int = 6, seed: int = 0) -> None:
    """Whole chips (resized, no random crop) with fixed masks, through the model in eval mode
    (BatchNorm uses its running statistics; nothing is updated). Uses its own random generators."""
    rng = random.Random(seed)
    chosen = rng.sample(relative_paths, min(rows, len(relative_paths)))
    views = torch.stack([TF.pil_to_tensor(TF.resize(load_image(p), [args.input_size, args.input_size],
                                                    interpolation=T.InterpolationMode.BILINEAR, antialias=True))
                         for p in chosen])  # uint8 (R, 1, H, W)
    x = to_network_input(views, args.normalize_mean, args.normalize_std, device)
    active = model.mask(len(chosen), device, generator=torch.Generator().manual_seed(seed))
    was_training = model.training
    model.eval()
    try:
        target, masked, rec_or_inp = model(x, active_b1ff=active, vis=True)
    finally:
        model.train(was_training)
    # Channel 0 of the network input is (gray - mean[0]) / std[0]; undo that to show gray values.
    to_gray = lambda t: (t[:, 0] * args.normalize_std[0] + args.normalize_mean[0]).clamp(0, 1).cpu().numpy()  # noqa: E731
    shown = active.repeat_interleave(DOWNSAMPLE_RATIO, 2).repeat_interleave(DOWNSAMPLE_RATIO, 3)[:, 0].cpu().numpy()
    panels = [to_gray(target), np.where(shown, to_gray(masked), 0.0), to_gray(rec_or_inp)]  # hidden shown black
    titles = ["view (whole chip)", "encoder input\n(hidden = black)", "visible + reconstructed\nhidden patches"]
    fig, axes = plt.subplots(len(chosen), 3, figsize=(5.4, 1.8 * len(chosen)), squeeze=False)
    for r, p in enumerate(chosen):
        for c in range(3):
            ax = axes[r, c]
            ax.imshow(panels[c][r], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=7)
        axes[r, 0].set_ylabel(Path(p).parent.name, fontsize=6)
    fig.suptitle(f"{out_path.stem}\nEach hidden patch shows the predicted pattern; its mean and contrast\n"
                 f"are taken from the original (the per-patch normalized loss ignores both)", fontsize=6)
    fig.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def vehicle_box(relative_path: str) -> tuple[float, float, float, float]:
    xml = (paths.SOC40_DIR / relative_path).with_suffix(".xml").read_text(encoding="utf-8")
    b = {k: float(re.search(rf"<{k}>\s*(-?[\d.]+)", xml).group(1)) for k in ["xmin", "ymin", "xmax", "ymax"]}
    return b["xmin"], b["ymin"], b["xmax"], b["ymax"]


def vehicle_coverage(relative_paths: list[str], args, n_images: int, samples_per_image: int, seed: int = 0) -> dict:
    """For sampled (crop, mask) pairs: the share of the vehicle box inside the crop, and the share of the
    vehicle box the encoder actually sees (inside the crop AND in a visible patch). Box areas in chip pixels."""
    rng = random.Random(seed)
    torch.manual_seed(seed)  # RandomResizedCrop.get_params uses the global torch generator
    g = torch.Generator().manual_seed(seed)
    f = args.input_size // DOWNSAMPLE_RATIO
    chosen = rng.sample(relative_paths, min(n_images, len(relative_paths)))
    dummy = torch.empty(1, NATIVE_SIZE, NATIVE_SIZE)  # get_params only needs the image size
    in_crop, visible, size_patches = [], [], []
    for p in chosen:
        x0, y0, x1, y1 = vehicle_box(p)
        area = max(x1 - x0, 1e-6) * max(y1 - y0, 1e-6)
        for _ in range(samples_per_image):
            top, left, h, w = T.RandomResizedCrop.get_params(dummy, scale=list(args.crop_scale), ratio=[3 / 4, 4 / 3])
            ix = max(0.0, min(x1, left + w) - max(x0, left))
            iy = max(0.0, min(y1, top + h) - max(y0, top))
            in_crop.append(ix * iy / area)
            size_patches.append(((x1 - x0) / (w / f), (y1 - y0) / (h / f)))  # vehicle size in mask patches
            active = random_mask(f * f, visible_count(args), g).view(f, f).numpy()
            seen = 0.0
            for r, c in zip(*np.nonzero(active)):  # visible cell (r, c) covers this part of the chip
                cx0, cx1 = left + c * w / f, left + (c + 1) * w / f
                cy0, cy1 = top + r * h / f, top + (r + 1) * h / f
                seen += max(0.0, min(x1, cx1) - max(x0, cx0)) * max(0.0, min(y1, cy1) - max(y0, cy0))
            visible.append(seen / area)
    a, v, s = np.array(in_crop), np.array(visible), np.array(size_patches)
    return {
        "samples": len(a),
        "vehicle_size_in_mask_patches_median": [round(float(np.median(s[:, 0])), 2), round(float(np.median(s[:, 1])), 2)],
        "vehicle_inside_crop_median": round(float(np.median(a)), 3),
        "share_crops_keeping_at_least_90pct": round(float((a >= 0.9).mean()), 3),
        "vehicle_visible_after_masking_median": round(float(np.median(v)), 3),
        "share_vehicle_visible_at_least_50pct": round(float((v >= 0.5).mean()), 3),
        "share_vehicle_visible_less_than_10pct": round(float((v < 0.1).mean()), 3),
        "share_vehicle_completely_hidden": round(float((v == 0).mean()), 3),
    }


def target_channel_check(relative_paths: list[str], args, n_images: int, seed: int = 0) -> dict:
    """If all 3 input channels were the target: per patch, the share of the target's variance that is the
    constant offset between the channels (from the ImageNet mean/std), not image structure."""
    p = DOWNSAMPLE_RATIO
    f = args.input_size // p
    m = torch.tensor(args.normalize_mean).view(3, 1, 1)
    s = torch.tensor(args.normalize_std).view(3, 1, 1)
    share, gray_std = [], []
    for rp in random.Random(seed).sample(relative_paths, min(n_images, len(relative_paths))):
        gray = TF.pil_to_tensor(TF.resize(load_image(rp), [args.input_size, args.input_size],
                                          interpolation=T.InterpolationMode.BILINEAR, antialias=True)).float() / 255
        x = (gray.expand(3, -1, -1) - m) / s                                                   # (3, H, W)
        patches = x.reshape(3, f, p, f, p).permute(1, 3, 0, 2, 4).reshape(f * f, 3, p * p)  # (L, 3, p*p)
        total = patches.reshape(f * f, -1).var(1)
        within = patches.var(2).mean(1)  # image structure; equal in every channel up to the channel's scale
        share.append((1 - within / total).numpy())
        gray_std.append(gray.reshape(f, p, f, p).permute(0, 2, 1, 3).reshape(f * f, -1).std(1).numpy())
    sh, gs = np.concatenate(share), np.concatenate(gray_std)
    flat = gs < np.median(gs)  # the flatter half of the patches (mostly background clutter)
    return {"patches": len(sh), "offset_share_of_target_variance_median": round(float(np.median(sh)), 3),
            "offset_share_90th_percentile": round(float(np.percentile(sh, 90)), 3),
            "offset_share_median_flatter_half_of_patches": round(float(np.median(sh[flat])), 3),
            "note": "with recon_channels = 1 this share is 0 by construction (one channel, normalized per patch)"}


def main():
    parser = argparse.ArgumentParser(description="Preview SparK views and masks; measure vehicle visibility")
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
              "crop_scale": args.crop_scale, "mask_ratio": args.mask_ratio,
              "visible_patches_per_image": visible_count(args),
              "vehicle_coverage": vehicle_coverage(ssl_paths, args, args_cli.images, 4),
              "target_if_3_channels": target_channel_check(ssl_paths, args, min(400, args_cli.images)),
              "note": "vehicle boxes from the XML annotations, used only for this check"}
    save_json(result, args_cli.out / "crop_and_mask_coverage.json")
    print(json.dumps(result, indent=2))
    print(f"Wrote {args_cli.out / 'view_preview.png'} and {args_cli.out / 'crop_and_mask_coverage.json'}")


if __name__ == "__main__":
    main()
