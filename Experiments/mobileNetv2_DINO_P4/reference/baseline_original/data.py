"""Reading SOC-40 manifests and images, and the image preprocessing shared by all routes.

On disk every image is a 128x128, single-channel (PIL mode "L"), uint8 GeoTIFF of
non-linearly quantized ground-range amplitude. The pipeline is:

    train:  random translate (pad + crop back to 128) -> random horizontal flip
            -> resize to input_size -> replicate gray to 3 channels -> [0, 1] -> normalize
    eval:   resize to input_size -> replicate gray to 3 channels -> [0, 1] -> normalize

Evaluation preprocessing is deterministic. See README.md for why these choices.
"""
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

import paths
from utils import sha256_text

NATIVE_SIZE = 128  # every ATRNet-STAR ground-range chip is 128x128


# ---- Manifests -----------------------------------------------------------------
def load_class_names() -> list[str]:
    """Class names indexed by label, from the saved class_mapping.csv."""
    with open(paths.SPLITS_DIR / "class_mapping.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = [r["class_name"] for r in sorted(rows, key=lambda r: int(r["label"]))]
    if [int(r["label"]) for r in sorted(rows, key=lambda r: int(r["label"]))] != list(range(len(names))):
        raise ValueError("class_mapping.csv labels are not 0..N-1")
    return names


def read_manifest(csv_name: str, class_names: list[str]) -> list[tuple[str, int]]:
    """(relative_path, label) pairs from one of the saved subset CSVs, with consistency checks."""
    with open(paths.SPLITS_DIR / csv_name, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    records = []
    for r in rows:
        label = int(r["label"])
        folder_class = r["relative_path"].split("/")[1]  # train/<class>/<file>.tif
        if class_names[label] != r["class_name"] or folder_class != r["class_name"]:
            raise ValueError(f"label/class mismatch in {csv_name}: {r}")
        records.append((r["relative_path"], label))
    return records


def test_manifest(class_names: list[str]) -> list[tuple[str, int]]:
    """(relative_path, label) for the official SOC-40 test folder, sorted for a stable order."""
    folders = sorted(p.name for p in paths.SOC40_TEST_DIR.iterdir() if p.is_dir())
    if sorted(folders) != sorted(class_names):
        raise ValueError("test class folders do not match class_mapping.csv")
    label_of = {c: i for i, c in enumerate(class_names)}
    return [(f"test/{c}/{p.name}", label_of[c])
            for c in sorted(folders)
            for p in sorted((paths.SOC40_TEST_DIR / c).glob("*.tif"))]


def manifest_hash(records: list[tuple[str, int]]) -> str:
    """Identity of a manifest's content (paths + labels), independent of how it was stored."""
    return sha256_text("\n".join(f"{p},{l}" for p, l in sorted(records)))


# ---- Images ----------------------------------------------------------------------
def load_image(relative_path: str) -> Image.Image:
    img = Image.open(paths.SOC40_DIR / relative_path)
    img.load()
    if img.mode != "L" or img.size != (NATIVE_SIZE, NATIVE_SIZE):
        raise ValueError(f"unexpected image format {img.mode} {img.size}: {relative_path}")
    return img


def build_transform(data_cfg: dict, train: bool) -> T.Compose:
    steps = []
    if train:
        pad = data_cfg["augment_translate_pad"]
        if pad > 0:
            # Shifts the chip by up to +/-pad pixels. Targets are already off-center in the
            # data, so this mimics natural variation. Reflection avoids black borders.
            steps.append(T.RandomCrop(NATIVE_SIZE, padding=pad, padding_mode="reflect"))
        if data_cfg["augment_hflip_p"] > 0:
            # Left-right flip along the cross-range axis (image width). Range runs along the
            # image height, so shadows stay on the correct side; it is like a mirrored azimuth.
            # Vertical flips and rotations would move the shadow and are not used.
            steps.append(T.RandomHorizontalFlip(data_cfg["augment_hflip_p"]))
    if data_cfg["input_size"] != NATIVE_SIZE:
        steps.append(T.Resize(data_cfg["input_size"], interpolation=T.InterpolationMode.BILINEAR,
                              antialias=True))
    steps += [
        T.Grayscale(num_output_channels=3),  # copies the one SAR channel into R, G and B
        T.ToTensor(),                        # uint8 [0, 255] -> float [0, 1], shape (3, H, W)
        T.Normalize(data_cfg["normalize_mean"], data_cfg["normalize_std"]),
    ]
    return T.Compose(steps)


class SARDataset(Dataset):
    """Returns (image tensor (3, input_size, input_size), label int)."""

    def __init__(self, records: list[tuple[str, int]], transform, cache_in_memory: bool = True):
        self.records = records
        self.transform = transform
        # Each chip is only 16 KB, so caching the decoded pixels avoids re-reading thousands
        # of small files from disk every epoch (slow on this PC).
        self.cache = ([np.array(load_image(p)) for p, _ in records] if cache_in_memory else None)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        path, label = self.records[i]
        # A 2-D uint8 array becomes a mode "L" image again.
        img = Image.fromarray(self.cache[i]) if self.cache is not None else load_image(path)
        return self.transform(img), torch.tensor(label)
