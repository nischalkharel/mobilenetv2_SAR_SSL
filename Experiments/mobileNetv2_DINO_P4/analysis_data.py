"""Chip lists with metadata, backbone loading and frozen-feature extraction for the P4 analyses
(feature_probe.py, saliency_figure.py). Analysis only: labels and metadata are read here for measuring what a
backbone encodes, never by training.

Test chips: the official SOC-40 test folder (test_manifest), band / polarization / depression / azimuth parsed from
the file name (<band>_<pol>_<depression>_<azimuth>_<id>.tif) and the scene name plus the vehicle bounding box read
from the .xml next to each image (the box is the dataset's own annotation of the vehicle's pixels, 128 x 128 frame).
Pool chips: the split's ssl_training_data.csv rows (same columns as preprocessing_dataset.py wrote).
Preprocessing for features: exactly the fine-tuning / evaluation preprocessing of config_finetune.json
(resize 224, gray -> 3 channels, [0, 1], ImageNet normalization), global average pool of model.features, 1280-d.
"""
import csv
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import paths
from data import SARDataset, build_transform, load_class_names, read_manifest, test_manifest
from model import build_model, load_finetuned

NAME_RE = re.compile(r"(?P<band>[A-Z]+)_(?P<polarization>[HV]{2})_(?P<depression>\d+)_(?P<azimuth>\d+)_(?P<image_id>\d+)\.tif")
DATA_CFG = json.loads((Path(__file__).resolve().parent / "config_finetune.json").read_text(encoding="utf-8"))["data"]

# Canonical CRC run folders (split 1) whose weights are copied to the PC with scp when an analysis needs them.
A1_DIR = paths.EXPERIMENT_DIR.parent / "mobileNetv2_DINO" / "outputs"
P4_DIR = paths.OUTPUT_DIR
BACKBONES = {  # name -> exported SSL backbone (teacher, epoch 100)
    "ImageNet": None,
    "A1": A1_DIR / "ssl" / "20261018-192425_A1_dino_ssl_mobilenetv2_seed0" / "backbone_teacher_epoch100.pth",
    "P4": P4_DIR / "ssl" / "20261110-194252_P4_dino_ssl_mobilenetv2_seed0" / "backbone_teacher_epoch100.pth",
    "P4perm": P4_DIR / "ssl" / "20261110-204010_P4perm_dino_ssl_mobilenetv2_seed0" / "backbone_teacher_epoch100.pth",
    "P4d": P4_DIR / "ssl" / "20261110-150743_P4d_dino_ssl_mobilenetv2_seed0" / "backbone_teacher_epoch100.pth",
}
FINETUNED_1PCT = {  # name -> best.pt of the 1 % fine-tune (split 1)
    "A1": A1_DIR / "finetune" / "20261027-155146_A1_dino_mobilenetv2_labeled_set_1pct_seed0" / "best.pt",
    "P4": P4_DIR / "finetune" / "20261111-034248_P4_dino_mobilenetv2_labeled_set_1pct_seed0" / "best.pt",
}


def parse_xml(xml_path: Path) -> dict:
    xml = xml_path.read_text(encoding="utf-8", errors="replace")
    scene = re.search(r"<scene_name>(.*?)</scene_name>", xml)
    box = {k: re.search(rf"<{k}>(-?\d+)</{k}>", xml) for k in ("xmin", "xmax", "ymin", "ymax")}
    if scene is None or any(v is None for v in box.values()):
        raise ValueError(f"scene or box missing in {xml_path}")
    # long vehicles can extend past the chip edge (e.g. ymin -4, ymax 131): clip the box to the 128 x 128 frame
    return {"scene": scene.group(1), **{k: min(max(int(v.group(1)), 0), 127) for k, v in box.items()}}


TEST_CACHE = paths.OUTPUT_DIR / "compare" / "test_chips_cache.csv"  # parsed once per machine (29k .xml reads); git-ignored
TEST_INT = ("label", "xmin", "xmax", "ymin", "ymax")


def test_rows(class_names: list[str] | None = None) -> list[dict]:
    """One dict per official test chip: relative_path, class_name, label, band, polarization, depression, azimuth,
    scene, xmin/xmax/ymin/ymax, acquisition_group (class|band|depression|azimuth|scene, as in the split CSVs)."""
    class_names = class_names or load_class_names()
    manifest = test_manifest(class_names)
    if TEST_CACHE.exists():
        with open(TEST_CACHE, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if [r["relative_path"] for r in rows] == [rel for rel, _ in manifest]:
            for r in rows:
                for k in TEST_INT:
                    r[k] = int(r[k])
            return rows
    rows = []
    for rel, label in manifest:
        name = rel.split("/")[-1]
        m = NAME_RE.fullmatch(name)
        if not m:
            raise ValueError(f"unexpected test file name {rel}")
        r = {"relative_path": rel, "class_name": class_names[label], "label": label, **m.groupdict()}
        r.update(parse_xml((paths.SOC40_DIR / rel).with_suffix(".xml")))
        r["acquisition_group"] = f"{r['class_name']}|{r['band']}|{r['depression']}|{r['azimuth']}|{r['scene']}"
        rows.append(r)
    TEST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_CACHE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def pool_rows(csv_name: str = "ssl_training_data.csv") -> list[dict]:
    with open(paths.SPLITS_DIR / csv_name, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
    return rows


def labeled_rows(csv_name: str, class_names: list[str]) -> list[dict]:
    return [{"relative_path": rel, "label": label} for rel, label in read_manifest(csv_name, class_names)]


def load_backbone_model(name: str, path: Path | None, num_classes: int = 40) -> torch.nn.Module:
    """Full MobileNetV2 (features + an unused fresh head) with the named SSL backbone, or the ImageNet start."""
    if path is None:
        return build_model({"type": "imagenet"}, num_classes)
    if not Path(path).exists():
        raise FileNotFoundError(f"{name}: {path} (copy it from CRC with scp; checkpoints are git-ignored)")
    return build_model({"type": "backbone_checkpoint", "backbone_checkpoint": str(path)}, num_classes)


def load_finetuned_model(path: Path, num_classes: int = 40) -> torch.nn.Module:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} (copy it from CRC with scp; checkpoints are git-ignored)")
    return load_finetuned(path, num_classes)[0]


_DATASETS: dict[tuple, SARDataset] = {}  # decoded chips per row set, read from disk once per process (slow part)


def chip_loader(rows: list[dict], batch_size: int = 128, num_workers: int = 0) -> DataLoader:
    records = [(r["relative_path"], int(r["label"])) for r in rows]
    key = tuple(p for p, _ in records)
    if key not in _DATASETS:
        _DATASETS[key] = SARDataset(records, build_transform(DATA_CFG, train=False))  # caches the decoded pixels
    return DataLoader(_DATASETS[key], batch_size=batch_size, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def extract_features(model: torch.nn.Module, rows: list[dict], device: torch.device, batch_size: int = 128) -> torch.Tensor:
    """(N, 1280) global-average-pooled features of model.features, fp32, in the order of rows."""
    model = model.to(device).eval()
    out = []
    for x, _ in chip_loader(rows, batch_size):
        f = model.features(x.to(device, non_blocking=True))
        out.append(f.mean((2, 3)).float().cpu())
    return torch.cat(out)
