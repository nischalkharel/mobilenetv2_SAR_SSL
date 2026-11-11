"""Detection data: the official COCO annotations, subsets by manifest, and the (tiny) augmentation.

An image is opened as 8-bit grey ("L"; the release stores most files as RGB with three identical channels, so
the conversion is lossless), replicated to 3 channels as a float tensor in [0, 1]; the detector's
own transform does the 512 resize and ImageNet normalisation. Boxes are converted from COCO xywh to xyxy;
labels are category_id + 1 (0 = background). Training augmentation: horizontal flip only (DESIGN.md section 3).
"""
import csv
import json
import random
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import paths


@lru_cache(maxsize=None)
def load_annotations(official_split: str) -> dict:
    """{'images': {file_name: image dict}, 'anns': {image_id: [ann, ...]}, 'raw': the json}"""
    with open(paths.ANNOTATION_FILES[official_split], encoding="utf-8") as f:
        raw = json.load(f)
    images = {im["file_name"]: im for im in raw["images"]}
    anns = {}
    for a in raw["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    return {"images": images, "anns": anns, "raw": raw}


def read_manifest(csv_path) -> list:
    """Rows of a subset manifest CSV (file_name, official_split, image_id, ...)."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["image_id"] = int(r["image_id"])
    return rows


def manifest_file_names(csv_path) -> list:
    return [r["file_name"] for r in read_manifest(csv_path)]


def load_image_tensor(official_split: str, file_name: str) -> torch.Tensor:
    """(3, H, W) float in [0, 1]; the single SAR channel replicated three times."""
    with Image.open(paths.image_path(official_split, file_name)) as im:
        gray = im.convert("L")
        t = torch.frombuffer(bytearray(gray.tobytes()), dtype=torch.uint8).view(gray.size[1], gray.size[0])
    return t.unsqueeze(0).expand(3, -1, -1).float().div_(255.0)


def boxes_and_labels(official_split: str, file_name: str):
    """GT boxes (N, 4) xyxy float and labels (N,) int64 (category_id + 1); degenerate boxes are widened to 1 px."""
    ann = load_annotations(official_split)
    im = ann["images"][file_name]
    boxes, labels = [], []
    for a in ann["anns"].get(im["id"], []):
        x, y, w, h = a["bbox"]
        x2, y2 = max(x + w, x + 1.0), max(y + h, y + 1.0)
        x, y = min(x, im["width"] - 1.0), min(y, im["height"] - 1.0)
        boxes.append([x, y, min(x2, im["width"]), min(y2, im["height"])])
        labels.append(a["category_id"] + 1)
    boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    return boxes, torch.tensor(labels, dtype=torch.int64), im["id"]


class DetectionDataset(Dataset):
    """Images of one official split, restricted to file_names (a manifest) or all of them."""

    def __init__(self, official_split: str, file_names=None, train: bool = False, hflip_p: float = 0.5, seed: int = 0):
        self.official_split = official_split
        ann = load_annotations(official_split)
        self.file_names = list(file_names) if file_names is not None else sorted(ann["images"])
        missing = [f for f in self.file_names if f not in ann["images"]]
        if missing:
            raise KeyError(f"{len(missing)} manifest images are not in {official_split}.json, e.g. {missing[:3]}")
        self.train = train
        self.hflip_p = hflip_p
        self.seed = seed

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, i):
        file_name = self.file_names[i]
        img = load_image_tensor(self.official_split, file_name)
        boxes, labels, image_id = boxes_and_labels(self.official_split, file_name)
        if self.train and self.hflip_p > 0:
            # per-sample RNG so the augmentation depends only on (seed, epoch, index), not on worker scheduling
            rng = random.Random(hash((self.seed, i)))
            if rng.random() < self.hflip_p:
                w = img.shape[-1]
                img = img.flip(-1)
                boxes = boxes.clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        target = {"boxes": boxes, "labels": labels, "image_id": image_id}
        return img, target

    def set_epoch(self, epoch: int):
        """Changes the per-sample augmentation stream each epoch (called by the training loop)."""
        self.seed = (self.seed * 1000003 + epoch) & 0xFFFFFFFF


def collate(batch):
    return tuple(zip(*batch))


def build_loader(dataset, batch_size, shuffle, num_workers, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate,
                      generator=g, pin_memory=torch.cuda.is_available(), persistent_workers=False, drop_last=False)


def coco_ground_truth(official_split: str, file_names=None):
    """pycocotools COCO object for the official split, restricted to file_names when given."""
    from pycocotools.coco import COCO
    raw = load_annotations(official_split)["raw"]
    if file_names is not None:
        keep = set(file_names)
        images = [im for im in raw["images"] if im["file_name"] in keep]
        ids = {im["id"] for im in images}
        anns = [a for a in raw["annotations"] if a["image_id"] in ids]
    else:
        images, anns = raw["images"], raw["annotations"]
    import contextlib
    import io
    coco = COCO()
    coco.dataset = {"images": images, "annotations": anns, "categories": raw["categories"]}
    with contextlib.redirect_stdout(io.StringIO()):     # pycocotools prints "creating index..." otherwise
        coco.createIndex()
    return coco
