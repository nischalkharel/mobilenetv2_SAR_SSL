"""Quick pipeline check before a real training run. Not a result.

Run from the repository root:
    python Experiments/Baseline/smoke_test.py
    python Experiments/Baseline/smoke_test.py --device cpu     # same checks on the CPU

Checks, in order:
 1. Manifests: labels agree with class_mapping.csv and with each image's class folder;
    train and validation do not overlap.
 2. Images: tensors have the expected shape and range; the 3 channels are identical copies.
 3. Initialization: the saved ImageNet file matches torchvision's IMAGENET1K_V2 weights
    (skipped if torchvision cannot download/cache them); backbone loaded, head is new.
 4. One training step: loss is finite, backbone and head parameters both change.
 5. A tiny 2-epoch training run (8 images per class) writes a complete run folder.
 6. Reloading best.pt and evaluating on that same validation subset reproduces the
    validation accuracy recorded during training.
It also saves sample_inputs.png (augmented training inputs, top; validation inputs, bottom)
into the smoke run folder so you can look at what the network actually sees.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V2_Weights
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402
import train  # noqa: E402
from data import SARDataset, build_transform, load_class_names, read_manifest  # noqa: E402
from engine import predict  # noqa: E402
from model import build_model, load_finetuned  # noqa: E402
from utils import resolve_device  # noqa: E402

CONFIG = json.loads(train.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        sys.exit("Smoke test failed.")


def main():
    parser = argparse.ArgumentParser(description="Pipeline check before a real training run")
    parser.add_argument("--device", help='override train.device in the config: "cuda", "cpu" or "auto"')
    args = parser.parse_args()
    if args.device:
        CONFIG["train"]["device"] = args.device
    torch.manual_seed(0)
    data_cfg = CONFIG["data"]
    size = data_cfg["input_size"]
    class_names = load_class_names()

    print("1. Manifests")
    train_records = read_manifest(data_cfg["train_csv"], class_names)  # raises on any label mismatch
    val_records = read_manifest(data_cfg["val_csv"], class_names)
    check("labels match class_mapping.csv and class folders", True, f"({len(class_names)} classes)")
    check("train and validation are disjoint", not ({p for p, _ in train_records} & {p for p, _ in val_records}),
          f"(train {len(train_records)}, val {len(val_records)})")

    print("2. Images")
    sample = train.first_per_class(train_records, 1)[:16]
    train_ds = SARDataset(sample, build_transform(data_cfg, train=True))
    val_ds = SARDataset(sample, build_transform(data_cfg, train=False))
    x, y = next(iter(DataLoader(train_ds, batch_size=16)))
    check("batch shape", tuple(x.shape) == (16, 3, size, size), str(tuple(x.shape)))
    check("values finite", bool(torch.isfinite(x).all()), f"(min {x.min():.2f}, max {x.max():.2f})")
    mean = torch.tensor(data_cfg["normalize_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(data_cfg["normalize_std"]).view(1, 3, 1, 1)
    raw = x * std + mean  # undo normalization -> [0, 1]
    check("3 channels are copies of one SAR channel",
          bool(torch.allclose(raw[:, 0], raw[:, 1], atol=1e-5) and torch.allclose(raw[:, 0], raw[:, 2], atol=1e-5)))
    check("labels in batch match records", y.tolist() == [l for _, l in sample])
    v = torch.stack([val_ds[i][0] for i in range(len(val_ds))])
    check("validation transform is deterministic", bool(torch.equal(v, torch.stack([val_ds[i][0] for i in range(len(val_ds))]))))
    preview = torch.cat([raw, v * std + mean]).clamp(0, 1)

    print("3. Initialization")
    saved = torch.load(paths.MOBILENETV2_IMAGENET_WEIGHTS, map_location="cpu", weights_only=True)
    try:
        official = MobileNet_V2_Weights.IMAGENET1K_V2.get_state_dict(progress=False)
        same = saved.keys() == official.keys() and all(torch.equal(saved[k], official[k]) for k in saved)
        check("saved file equals torchvision IMAGENET1K_V2", same)
    except Exception as e:  # no internet and no cache
        print(f"  [SKIP] could not fetch torchvision weights to compare: {e}")
    model = build_model(CONFIG["init"], len(class_names))
    feats = model.features.state_dict()
    check("backbone equals ImageNet weights", all(torch.equal(feats[k], saved["features." + k]) for k in feats))
    check("new head has 40 outputs", model.classifier[1].out_features == len(class_names))

    print("4. One training step")
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    model.train()
    opt = train.make_optimizer(model, CONFIG["train"])
    loss = torch.nn.functional.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    changed = {k for k, p in model.named_parameters() if not torch.equal(p, before[k])}
    check("loss finite", bool(torch.isfinite(loss)), f"(loss {loss.item():.3f}, ln 40 = 3.689)")
    check("first backbone conv updated", "features.0.0.weight" in changed)
    check("classifier updated", "classifier.1.weight" in changed)
    check("all trainable parameters updated", len(changed) == len(before), f"({len(changed)}/{len(before)})")

    print("5. Tiny training run")
    run_dir = train.run(CONFIG, smoke=True)
    expected = ["config.json", "class_mapping.csv", "manifests.json", "run_info.json", "history.csv", "best.pt"]
    check("run folder complete", all((run_dir / f).exists() for f in expected))
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    check("run status completed", info["status"] == "completed")
    save_image(preview, run_dir / "sample_inputs.png", nrow=16)

    print("6. Reload best.pt")
    reloaded, ckpt = load_finetuned(run_dir / "best.pt", len(class_names))
    smoke_val = train.first_per_class(val_records, train.SMOKE_PER_CLASS)
    loader = DataLoader(SARDataset(smoke_val, build_transform(data_cfg, train=False)), batch_size=64)
    device = resolve_device(CONFIG["train"]["device"])
    acc = predict(reloaded.to(device), loader, device)["acc"]
    check("reloaded validation accuracy matches training record", abs(acc - ckpt["val_acc"]) < 1e-6,
          f"({acc:.4f} vs {ckpt['val_acc']:.4f})")
    print(f"\nSmoke test passed. Folder: {run_dir}\nPer-epoch time for {len(smoke_val)} val + "
          f"{train.SMOKE_PER_CLASS * len(class_names)} train images is in history.csv.")


if __name__ == "__main__":
    main()
