"""Frozen-feature probes of a backbone on official val: what do the pooled features encode?

    python -m shared.probes --backbone <exported .pth | imagenet | finetuned:<best.pt>> --out <dir> [--n 4000] [--smoke N]

Features: 1280-d global-average-pooled output of MobileNetV2 `features` on the whole image after the
detector's 512 resize (same preprocessing as detection). Probes, all on a fixed 50/50 split of the sampled
val images (seed 0), features standardised on the probe-train half:
  - class presence (6 binary linear probes, mean average precision)  -> content
  - source dataset (9-way linear probe accuracy vs majority)          -> acquisition source
  - satellite and polarisation within Shipdataset (linear probes)    -> condition where it is per-image
  - 10-NN agreement of source and of class set (cosine)               -> what nearest neighbours share
  - feature std and effective rank (collapse diagnostics)
Writes probe_results.json and probe_results.md into --out.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import manifests, paths
from .backbone import fresh_features, load_backbone_file, load_imagenet_mobilenet_v2
from .coco_data import load_annotations
from .ssl_data import load_resized_gray, to_network_input


class WholeImageDataset(Dataset):
    def __init__(self, rows, pad_to=512):
        self.rows, self.pad_to = rows, pad_to

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        im = load_resized_gray(r["official_split"], r["file_name"], self.pad_to)
        t = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).view(1, im.size[1], im.size[0])
        canvas = torch.zeros(1, self.pad_to, self.pad_to, dtype=torch.uint8)      # zero padding (= black, as the detector pads)
        canvas[:, :t.shape[1], :t.shape[2]] = t
        return canvas, i


def load_features_module(spec: str):
    """'imagenet' | exported backbone file | 'finetuned:<best.pt>' -> (features module, description)"""
    if spec == "imagenet":
        return load_imagenet_mobilenet_v2().features, {"kind": "imagenet"}
    if spec.startswith("finetuned:"):
        ckpt = torch.load(spec.split(":", 1)[1], map_location="cpu", weights_only=False)
        state = {k[len("backbone.body."):]: v for k, v in ckpt["model_state_dict"].items() if k.startswith("backbone.body.")}
        f = fresh_features()
        f.load_state_dict(state, strict=True)
        return f, {"kind": "finetuned", "file": spec.split(":", 1)[1], "epoch": ckpt.get("epoch")}
    state, rec = load_backbone_file(spec)
    f = fresh_features()
    f.load_state_dict(state, strict=True)
    return f, {"kind": "ssl", **{k: v for k, v in rec.items() if k in ("file", "file_sha256", "ssl_method", "ssl_epoch")}}


@torch.no_grad()
def extract(features, rows, device, batch_size=32, num_workers=0):
    features = features.to(device).eval()
    loader = DataLoader(WholeImageDataset(rows), batch_size=batch_size, num_workers=num_workers)
    out = []
    for u8, _ in loader:
        x = to_network_input(u8, device)
        out.append(torch.flatten(torch.nn.functional.adaptive_avg_pool2d(features(x), 1), 1).cpu())
    return torch.cat(out)


def linear_probe(x_tr, y_tr, x_te, y_te, n_classes, epochs=300, lr=0.1, wd=1e-4):
    """Multinomial logistic regression (full-batch, LBFGS-free Adam) on standardised features; returns accuracy."""
    torch.manual_seed(0)
    w = torch.zeros(x_tr.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(x_tr @ w + b, y_tr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(((x_te @ w + b).argmax(1) == y_te).float().mean())


def binary_probe_ap(x_tr, y_tr, x_te, y_te, epochs=300, lr=0.1, wd=1e-4):
    torch.manual_seed(0)
    w = torch.zeros(x_tr.shape[1], 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits((x_tr @ w + b).squeeze(1), y_tr.float())
        loss.backward()
        opt.step()
    with torch.no_grad():
        s = (x_te @ w + b).squeeze(1)
    return average_precision(s.numpy(), y_te.numpy())


def average_precision(scores, labels):
    order = np.argsort(-scores)
    labels = labels[order]
    if labels.sum() == 0:
        return float("nan")
    tp = np.cumsum(labels)
    precision = tp / np.arange(1, len(labels) + 1)
    return float((precision * labels).sum() / labels.sum())


def knn_agreement(x, keys, k=10):
    """Mean fraction of an item's k nearest cosine neighbours that share its key (string)."""
    xn = torch.nn.functional.normalize(x, dim=1)
    sims = xn @ xn.T
    sims.fill_diagonal_(-2)
    nn_idx = sims.topk(k, dim=1).indices.numpy()
    keys = np.array(keys)
    return float((keys[nn_idx] == keys[:, None]).mean())


def effective_rank(x):
    xc = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(xc)
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000, help="val images sampled (seed 0)")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from .metadata import read_metadata
    rows = [r for r in read_metadata() if r["official_split"] == "val"]
    rng = random.Random(0)
    rows = sorted(rows, key=lambda r: r["file_name"])
    n = args.smoke or args.n
    rows = rng.sample(rows, min(n, len(rows)))
    features, desc = load_features_module(args.backbone)
    x = extract(features, rows, device, num_workers=0 if args.smoke else 4)
    ann = load_annotations("val")
    class_sets = [frozenset(a["category_id"] for a in ann["anns"].get(r["image_id"], [])) for r in rows]
    half = len(rows) // 2
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    tr, te = idx[:half], idx[half:]
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    z = (x - mu) / sd
    res = {"backbone": desc, "n_images": len(rows), "feature_std_mean": float(x.std(0).mean()),
           "effective_rank": effective_rank(x), "feature_cosine_mean": float(
               (torch.nn.functional.normalize(x, dim=1) @ torch.nn.functional.normalize(x, dim=1).T).mean())}
    # class presence
    aps = {}
    for c, name in enumerate(paths.CLASS_NAMES):
        y = torch.tensor([c in s for s in class_sets], dtype=torch.long)
        if y[tr].sum() > 0 and y[te].sum() > 0:
            aps[name] = binary_probe_ap(z[tr], y[tr], z[te], y[te])
    res["class_presence_ap"] = aps
    res["class_presence_mean_ap"] = float(np.nanmean(list(aps.values()))) if aps else None
    # source
    sources = sorted({r["source"] for r in rows})
    ys = torch.tensor([sources.index(r["source"]) for r in rows])
    res["source_probe_acc"] = linear_probe(z[tr], ys[tr], z[te], ys[te], len(sources))
    res["source_majority_acc"] = max(Counter(ys[te].tolist()).values()) / len(te)
    # satellite / polarisation within Shipdataset
    ship = [i for i, r in enumerate(rows) if r["source"] == "Shipdataset" and r["polarization"] != "unknown"]
    for field in ("satellite", "polarization"):
        vals = sorted({rows[i][field] for i in ship})
        if len(vals) >= 2 and len(ship) >= 40:
            yv = torch.tensor([vals.index(rows[i][field]) for i in ship])
            s_tr = [j for j, i in enumerate(ship) if i in set(tr)]
            s_te = [j for j, i in enumerate(ship) if i in set(te)]
            zz = z[ship]
            res[f"shipdataset_{field}_probe_acc"] = linear_probe(zz[s_tr], yv[s_tr], zz[s_te], yv[s_te], len(vals))
            res[f"shipdataset_{field}_majority_acc"] = max(Counter(yv[s_te].tolist()).values()) / max(len(s_te), 1)
    # kNN agreement
    res["knn10_source_agreement"] = knn_agreement(x, [r["source"] for r in rows])
    res["knn10_classset_agreement"] = knn_agreement(x, ["+".join(map(str, sorted(s))) for s in class_sets])
    res["knn10_source_chance"] = float(sum((c / len(rows)) ** 2 for c in Counter(r["source"] for r in rows).values()))
    res["knn10_classset_chance"] = float(sum((c / len(rows)) ** 2 for c in Counter(class_sets).values()))
    with open(out / "probe_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    lines = [f"# Frozen-feature probes ({desc['kind']}), {len(rows)} val images", "",
             "| probe | value | reference |", "|---|---|---|",
             f"| class presence mean AP | {res['class_presence_mean_ap']} | - |",
             f"| source dataset accuracy | {res['source_probe_acc']:.3f} | majority {res['source_majority_acc']:.3f} |",
             f"| 10-NN same source | {res['knn10_source_agreement']:.3f} | chance {res['knn10_source_chance']:.3f} |",
             f"| 10-NN same class set | {res['knn10_classset_agreement']:.3f} | chance {res['knn10_classset_chance']:.3f} |",
             f"| effective rank | {res['effective_rank']:.1f} | max 1280 |",
             f"| mean pairwise cosine | {res['feature_cosine_mean']:.3f} | 1 = collapsed |"]
    for field in ("satellite", "polarization"):
        k = f"shipdataset_{field}_probe_acc"
        if k in res:
            lines.append(f"| Shipdataset {field} accuracy | {res[k]:.3f} | majority {res[f'shipdataset_{field}_majority_acc']:.3f} |")
    (out / "probe_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return res


if __name__ == "__main__":
    main()
