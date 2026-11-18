"""Views for RegionDINO: the scene views of plain DINO plus "object views" cut around proposed regions.

One pool image gives
    scene views   exactly DINO's: 2 global (224) + n local (96) random crops of the whole image, from the shared
                  view recipe (SARDet/shared/ssl_data.py). Absent in the object-only method.
    object views  for K regions of the image (K = objects_per_image), each region gives
                    2 object-global views: window = region extent x zoom in obj_global_zoom (the thing plus a
                                           margin of its surroundings), centre jittered, resized to obj_global_size
                    m object-local views:  window = region extent x zoom in obj_local_zoom (a part of the thing),
                                           centre anywhere inside the region, resized to obj_local_size
                  with the shared flip / brightness / contrast / blur augmentation.
Nothing refers to a fixed pixel size: every object window is sized from the region's own extent. The only
guard is max_magnification (a window is never smaller than view size / max_magnification, so a tiny blob is
not blown up into a blur). Regions smaller than min_object_px are ignored.

If an image has fewer than K usable regions the chosen ones repeat; if it has none, K random windows stand
in (counted as `fallback` so the run record shows how often that happened).
region_source "random" is the control: K random windows per image whose extents and aspects are drawn from
the pooled, sampling-weighted size distribution of the real regions (see __init__), so window sizes match
what the real runs train on while positions carry no information.

Batch layout (collate): scene slots -> list of (B,1,S,S); object slots -> list of (B*K,1,S,S), image-major
(image 0 object 0, image 0 object 1, ..., image 1 object 0, ...). DINO's loss then treats each object as a
sample of its own, with the 2 object-global views as the teacher views, exactly as it treats images.
"""
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

from shared.ssl_data import load_common_config, load_resized_gray, make_view  # read-only

from .regions import RegionStore


def augment(v: Image.Image, out: int, cfg: dict, blur_p: float, rng: random.Random) -> torch.Tensor:
    """The shared recipe's augmentation (same code path as make_view, after the window) -> uint8 (1,out,out)."""
    if v.size != (out, out):
        v = v.resize((out, out), Image.BILINEAR)
    if rng.random() < cfg["hflip_p"]:
        v = v.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < cfg["color_jitter_p"]:
        v = ImageEnhance.Brightness(v).enhance(rng.uniform(1 - cfg["brightness"], 1 + cfg["brightness"]))
        v = ImageEnhance.Contrast(v).enhance(rng.uniform(1 - cfg["contrast"], 1 + cfg["contrast"]))
    if rng.random() < blur_p:
        v = v.filter(ImageFilter.GaussianBlur(radius=rng.uniform(*cfg["blur_sigma"])))
    return torch.frombuffer(bytearray(v.tobytes()), dtype=torch.uint8).view(1, out, out)


def object_window(box, zoom_range, out, max_magnification, W, H, rng, centre_mode):
    """Square window (x0, y0, side) around a region box.  centre_mode 'jitter' keeps the whole region (with a
    margin) in view; 'inside' puts the centre anywhere in the region (a partial view)."""
    x1, y1, x2, y2 = [float(v) for v in box]
    e = max(x2 - x1, y2 - y1, 1.0)
    side = e * rng.uniform(*zoom_range)
    side = max(side, out / max_magnification, 8.0)
    side = int(round(min(side, W, H)))
    if centre_mode == "inside":
        cx, cy = rng.uniform(x1, x2), rng.uniform(y1, y2)
    else:
        cx = (x1 + x2) / 2 + rng.uniform(-0.15, 0.15) * e
        cy = (y1 + y2) / 2 + rng.uniform(-0.15, 0.15) * e
    x0 = int(round(min(max(cx - side / 2, 0), W - side)))
    y0 = int(round(min(max(cy - side / 2, 0), H - side)))
    return x0, y0, side


class RegionViewDataset(Dataset):
    def __init__(self, rows, scene_plan, store: RegionStore, opts: dict, seed: int = 0, input_size: int = 512):
        self.rows = rows
        self.scene_plan = scene_plan                 # [] for the object-only method
        self.store = store
        self.o = opts
        self.cfg = load_common_config()["views"]
        self.seed, self.epoch, self.input_size = seed, 0, input_size
        self.K = int(opts["objects_per_image"])
        self.obj_slots = [("obj_global", 1, opts["obj_blur_p"][0]), ("obj_global", 1, opts["obj_blur_p"][1]),
                          ("obj_local", int(opts["obj_local_number"]), opts["obj_blur_p"][2])]
        self.views_per_image = sum(c for _, c, _ in scene_plan) + self.K * sum(c for _, c, _ in self.obj_slots)
        # pooled size distribution for the control / fallback windows, resampled ONCE with the same weights the
        # real draw uses (sqrt of box area when region_sampling is sqrt_area), so the control's window sizes
        # follow the sizes the felz runs actually train on (weighted with replacement; exact per-image
        # without-replacement matching is impossible across the pool)
        keep = store.extents >= opts["min_object_px"]
        ext, asp = store.extents[keep], store.aspects[keep]
        if len(ext) == 0:
            ext, asp = np.array([32], np.int32), np.array([1.0], np.float32)
        b = store.boxes[keep] if keep.any() else None
        w = np.sqrt((b[:, 2] - b[:, 0]).astype(np.float64) * (b[:, 3] - b[:, 1])) if (b is not None and opts["region_sampling"] == "sqrt_area") else np.ones(len(ext))
        pick = np.random.default_rng(12345).choice(len(ext), size=min(len(ext), 200000), replace=True, p=w / w.sum())
        self.pooled_extents, self.pooled_aspects = ext[pick], asp[pick]

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def plan_description(self):
        return {"scene": self.scene_plan, "object": [(s, c * self.K, b) for s, c, b in self.obj_slots],
                "objects_per_image": self.K, "obj_global_size": self.o["obj_global_size"], "obj_local_size": self.o["obj_local_size"]}

    # ---- region choice -------------------------------------------------------------------------------------
    def usable_boxes(self, file_name):
        """(boxes, sampling weights) of the regions at least min_object_px wide or high."""
        b = self.store.get(file_name)
        if len(b) == 0:
            return b, np.zeros(0)
        ext = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
        keep = ext >= self.o["min_object_px"]
        b, ext = b[keep], ext[keep]
        # bigger things are picked more often than specks (weight = sqrt of the box area), unless "uniform"
        w = np.sqrt((b[:, 2] - b[:, 0]).astype(np.float64) * (b[:, 3] - b[:, 1])) if self.o["region_sampling"] == "sqrt_area" else np.ones(len(b))
        return b, w

    def random_box(self, W, H, rng):
        """A box with a real region's extent and aspect at a random place (control / fallback)."""
        j = rng.randrange(len(self.pooled_extents))
        e = int(min(self.pooled_extents[j], W, H))
        a = float(self.pooled_aspects[j])
        w, h = (e, max(4, int(round(e * a)))) if a <= 1 else (max(4, int(round(e / a))), e)
        w, h = min(w, W), min(h, H)
        x1, y1 = rng.randint(0, W - w), rng.randint(0, H - h)
        return np.array([x1, y1, x1 + w, y1 + h], dtype=np.int32)

    def choose_boxes(self, file_name, W, H, rng):
        """K boxes for this image; returns (boxes, n_real_usable, fallback flag)."""
        if self.o["region_source"] == "random":
            return [self.random_box(W, H, rng) for _ in range(self.K)], 0, 0
        real, weights = self.usable_boxes(file_name)
        if len(real) == 0:
            return [self.random_box(W, H, rng) for _ in range(self.K)], 0, 1
        idx = []
        pool, wts = list(range(len(real))), list(weights)
        for _ in range(self.K):                       # weighted draw without replacement; repeats once exhausted
            if not pool:
                pool, wts = list(range(len(real))), list(weights)
            j = rng.choices(range(len(pool)), weights=wts, k=1)[0]
            idx.append(pool.pop(j))
            wts.pop(j)
        return [real[i] for i in idx], int(len(real)), 0

    # ---- views ---------------------------------------------------------------------------------------------
    def object_views(self, im, box, rng):
        W, H = im.size
        out = []
        for slot, count, blur_p in self.obj_slots:
            size = self.o[f"{slot}_size"]
            zoom = self.o[f"{slot}_zoom"]
            mode = "inside" if slot == "obj_local" else "jitter"
            for _ in range(count):
                x0, y0, side = object_window(box, zoom, size, self.o["max_magnification"], W, H, rng, mode)
                out.append(augment(im.crop((x0, y0, x0 + side, y0 + side)), size, self.cfg, blur_p, rng))
        return out

    def __getitem__(self, i):
        r = self.rows[i]
        rng = random.Random(f"{self.seed}-{self.epoch}-{i}")
        im = load_resized_gray(r["official_split"], r["file_name"], self.input_size)
        scene = []
        for size_key, count, blur_p in self.scene_plan:
            for _ in range(count):
                scene.append(make_view(im, self.cfg[f"{size_key}_size"], self.cfg, blur_p, rng))
        boxes, n_real, fallback = self.choose_boxes(r["file_name"], im.size[0], im.size[1], rng)
        per_object = [self.object_views(im, b, rng) for b in boxes]         # K lists of (2 + m) tensors
        n_slots = len(per_object[0])
        objects = [torch.stack([per_object[k][s] for k in range(self.K)]) for s in range(n_slots)]   # (K,1,S,S) per slot
        meta = torch.tensor([n_real, fallback], dtype=torch.long)
        return (scene, objects, meta), i


def collate_region_views(batch):
    scene_n = len(batch[0][0][0])
    scene = [torch.stack([b[0][0][v] for b in batch]) for v in range(scene_n)]
    obj_n = len(batch[0][0][1])
    objects = [torch.cat([b[0][1][v] for b in batch]) for v in range(obj_n)]          # (B*K,1,S,S), image-major
    meta = torch.stack([b[0][2] for b in batch])
    idx = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return (scene, objects, meta), idx


def build_region_loader(dataset, batch_size, num_workers, seed, epoch, drop_last=True):
    g = torch.Generator()
    g.manual_seed(seed * 100003 + epoch)
    dataset.set_epoch(epoch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_region_views,
                      generator=g, pin_memory=torch.cuda.is_available(), drop_last=drop_last, persistent_workers=False)


# ---- diagnostics (labels read here ONLY to describe the regions; never seen by training) --------------------
def region_box_diagnostic(rows, store: RegionStore, min_object_px: int, n_images=300, seed=0, input_size=512) -> dict:
    """How the proposed regions relate to the GT boxes of up to n_images pool images (at the 512 scale):
    region_centre_in_a_box     share of usable regions whose centre lies inside some GT box
    gt_boxes_with_a_region     share of GT boxes that contain at least one usable region centre
    usable_regions_per_image   mean count after the min_object_px filter; images_without_regions share."""
    from shared.coco_data import boxes_and_labels
    rng = random.Random(seed)
    sample = rows if len(rows) <= n_images else rng.sample(rows, n_images)
    reg_in, reg_total, box_hit, box_total, counts, empty = 0, 0, 0, 0, [], 0
    for r in sample:
        gt, _, _ = boxes_and_labels(r["official_split"], r["file_name"])
        w, h = r["width"], r["height"]
        s = min(input_size / min(w, h), input_size / max(w, h))
        gt = gt.numpy() * s
        b = store.get(r["file_name"])
        if len(b):
            ext = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
            b = b[ext >= min_object_px]
        counts.append(len(b))
        empty += int(len(b) == 0)
        if len(b):
            cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
            inside = np.zeros(len(b), bool)
            for x1, y1, x2, y2 in gt:
                hit = (cx >= x1) & (cx < x2) & (cy >= y1) & (cy < y2)
                inside |= hit
                box_hit += int(hit.any())
            reg_in += int(inside.sum())
        reg_total += len(b)
        box_total += len(gt)
    return {"n_images": len(sample), "region_centre_in_a_box": reg_in / max(reg_total, 1),
            "gt_boxes_with_a_region": box_hit / max(box_total, 1), "usable_regions_per_image": float(np.mean(counts)) if counts else 0.0,
            "images_without_regions": empty / max(len(sample), 1)}


def preview_panel(dataset: RegionViewDataset, out_png, n=4):
    """A picture of what the network sees for n images: scene views (if any) and the object views."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = min(n, len(dataset))
    has_scene = bool(dataset.scene_plan)
    rows_per = 2 if has_scene else 1
    cols = max(sum(c for _, c, _ in dataset.scene_plan) if has_scene else 1, dataset.K * sum(c for _, c, _ in dataset.obj_slots))
    fig, axes = plt.subplots(rows_per * n, cols, figsize=(1.6 * cols, 1.9 * rows_per * n), squeeze=False)
    for i in range(n):
        (scene, objects, meta), _ = dataset[i]
        r0 = rows_per * i
        for ax in axes[r0: r0 + rows_per].ravel():
            ax.axis("off")
        for j, v in enumerate(scene):
            axes[r0][j].imshow(v[0].numpy(), cmap="gray", vmin=0, vmax=255)
            axes[r0][j].set_title(f"scene {v.shape[-1]}px", fontsize=6)
        j = 0
        for k in range(dataset.K):
            for s, v in enumerate(objects):
                axes[r0 + rows_per - 1][j].imshow(v[k, 0].numpy(), cmap="gray", vmin=0, vmax=255)
                axes[r0 + rows_per - 1][j].set_title(f"obj{k} {'G' if s < 2 else 'L'} {v.shape[-1]}px", fontsize=6)
                j += 1
        axes[r0][0].set_title(f"{dataset.rows[i]['file_name']}  usable regions {int(meta[0])}  fallback {int(meta[1])}", fontsize=6, loc="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
