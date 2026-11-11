"""SSL data: the unlabeled pool and the ONE view recipe shared by every SSL method (DESIGN.md section 4).

Every image is first resized exactly like the detector does (shorter side 512, longer side <= 512), so the
object scale during SSL equals the object scale during detection. Views are square crops at that scale with a
zoom jitter (window = view size x zoom in [0.75, 1.33], then resized to the view size), horizontal flip,
brightness/contrast jitter and Gaussian blur. Views leave the loader as uint8 (1, S, S) tensors; the method's
training step calls to_network_input() on the GPU (replicate to RGB, /255, ImageNet mean/std), the same
arithmetic the detector's transform uses.

Methods differ only in HOW MANY views of each size they ask for (DINO 2 global + 8 local, LeJEPA 8 global,
SparK 1 global) and in method-specific blur probabilities, passed as `view_plan`.
"""
import json
import random
from functools import lru_cache

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

from . import paths
from .manifests import read as read_manifest

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@lru_cache(maxsize=None)
def load_common_config() -> dict:
    with open(paths.SHARED_SSL_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.pop("_comment", None)
    return cfg


def detector_resize(im: Image.Image, size: int = 512) -> Image.Image:
    """Same rule as torchvision GeneralizedRCNNTransform(min_size=size, max_size=size)."""
    w, h = im.size
    s = min(size / min(w, h), size / max(w, h))
    new = (max(1, round(w * s)), max(1, round(h * s)))
    return im if new == (w, h) else im.resize(new, Image.BILINEAR)


def load_resized_gray(official_split: str, file_name: str, size: int = 512) -> Image.Image:
    with Image.open(paths.image_path(official_split, file_name)) as im:
        return detector_resize(im.convert("L"), size)


def random_window(w: int, h: int, out: int, zoom_range, rng: random.Random):
    """(x0, y0, side) of a square window of side out*zoom (clipped to the image) at a random position."""
    side = int(round(out * rng.uniform(*zoom_range)))
    side = max(8, min(side, w, h))
    return rng.randint(0, w - side), rng.randint(0, h - side), side


def make_view(im: Image.Image, out: int, cfg: dict, blur_p: float, rng: random.Random) -> torch.Tensor:
    """One augmented uint8 (1, out, out) view."""
    x0, y0, side = random_window(im.size[0], im.size[1], out, cfg["zoom_range"], rng)
    v = im.crop((x0, y0, x0 + side, y0 + side))
    if side != out:
        v = v.resize((out, out), Image.BILINEAR)
    if rng.random() < cfg["hflip_p"]:
        v = v.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < cfg["color_jitter_p"]:
        v = ImageEnhance.Brightness(v).enhance(rng.uniform(1 - cfg["brightness"], 1 + cfg["brightness"]))
        v = ImageEnhance.Contrast(v).enhance(rng.uniform(1 - cfg["contrast"], 1 + cfg["contrast"]))
    if rng.random() < blur_p:
        v = v.filter(ImageFilter.GaussianBlur(radius=rng.uniform(*cfg["blur_sigma"])))
    t = torch.frombuffer(bytearray(v.tobytes()), dtype=torch.uint8).view(1, out, out)
    return t


class SSLPoolDataset(Dataset):
    """Unlabeled images -> list of uint8 views per the view_plan, plus the row index (for cell ids etc.).

    view_plan: list of (size_key, count, blur_p) e.g. [("global", 2, 1.0), ("local", 8, 0.5)]; views are
    returned in plan order (all of the first entry, then the next), which is what DINO's multi-crop expects.
    """

    def __init__(self, rows, view_plan, seed: int = 0, input_size: int = 512):
        self.rows = rows
        self.view_plan = view_plan
        self.cfg = load_common_config()["views"]
        self.seed = seed
        self.epoch = 0
        self.input_size = input_size

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, i):
        r = self.rows[i]
        rng = random.Random(f"{self.seed}-{self.epoch}-{i}")
        im = load_resized_gray(r["official_split"], r["file_name"], self.input_size)
        views = []
        for size_key, count, blur_p in self.view_plan:
            out = self.cfg[f"{size_key}_size"]
            for _ in range(count):
                views.append(make_view(im, out, self.cfg, blur_p, rng))
        return views, i


def collate_views(batch):
    """[(views, idx), ...] -> (list of (B,1,S,S) uint8 tensors, one per view slot; LongTensor of indices)."""
    n_views = len(batch[0][0])
    stacked = [torch.stack([b[0][v] for b in batch]) for v in range(n_views)]
    idx = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return stacked, idx


def to_network_input(u8: torch.Tensor, device=None) -> torch.Tensor:
    """(B,1,S,S) uint8 -> (B,3,S,S) float, ImageNet-normalised; identical to the detector's preprocessing."""
    x = u8.to(device, non_blocking=True) if device is not None else u8
    x = x.float().div_(255.0).expand(-1, 3, -1, -1)
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def build_ssl_loader(dataset, batch_size, num_workers, seed, epoch, drop_last=True):
    """A new loader each epoch with a seeded shuffle, so data order never depends on resume."""
    g = torch.Generator()
    g.manual_seed(seed * 100003 + epoch)
    dataset.set_epoch(epoch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_views,
                      generator=g, pin_memory=torch.cuda.is_available(), drop_last=drop_last, persistent_workers=False)


def load_pool_rows(csv_path=None, limit=None):
    rows = read_manifest(csv_path or (paths.DATA_DIR / "ssl_pool.csv"))
    return rows[:limit] if limit else rows


def view_box_coverage(rows, n_images=500, seed=0, input_size=512) -> dict:
    """Diagnostic only (reads boxes): fraction of global / local windows that contain >= 1 GT box centre."""
    from .coco_data import boxes_and_labels
    cfg = load_common_config()["views"]
    rng = random.Random(seed)
    stats = {}
    sample = rows if len(rows) <= n_images else rng.sample(rows, n_images)
    for size_key in ("global", "local"):
        out = cfg[f"{size_key}_size"]
        hit = total = 0
        for r in sample:
            boxes, _, _ = boxes_and_labels(r["official_split"], r["file_name"])
            w, h = r["width"], r["height"]
            s = min(input_size / min(w, h), input_size / max(w, h))
            cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2 * s, (boxes[:, 1] + boxes[:, 3]) / 2 * s
            for _ in range(4):
                x0, y0, side = random_window(round(w * s), round(h * s), out, cfg["zoom_range"], rng)
                inside = ((cx >= x0) & (cx < x0 + side) & (cy >= y0) & (cy < y0 + side)).any().item()
                hit += bool(inside)
                total += 1
        stats[size_key] = {"views_with_a_box": hit / max(total, 1), "n_views": total}
    return stats
