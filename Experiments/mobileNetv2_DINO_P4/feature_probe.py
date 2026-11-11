"""What does each frozen backbone encode? Class versus recording condition, measured the same way for every backbone.

For each backbone (ImageNet start, A1, P4, and the controls whose weights are on this machine) the features of
analysis_data.extract_features are probed without any training of the backbone:
  (1) class, few labels: linear probe (multinomial logistic regression, L-BFGS, L2 strength chosen on validation.csv)
      trained on labeled_set_1pct.csv (652 chips) and nearest-class-mean with the same chips; scored on the full
      official test set (29,284). This is the frozen-feature version of the fine-tuning result.
  (2) class, many labels: the same probe trained on a pool sample of 150 chips per class (6,000; labels read from the
      SSL CSV for this analysis), scored on a test sample of 100 chips per class (4,000).
  (3) condition decodability: probes trained on the pool sample to predict scene, 30-degree azimuth sector, depression
      and band, scored on the test sample (majority-class baseline printed next to each).
  (4) neighbourhood: on the test sample, cosine nearest neighbours excluding the chip's own acquisition group
      (same class, band, depression, azimuth and scene); 1-NN agreement and 10-NN purity for class, scene, azimuth
      sector, depression, band (chance = sum of squared label frequencies).
Reading: P4 is expected to raise the class numbers; whether the condition numbers fall says whether the condition
left the feature geometry or only stopped dominating it (the pilot saw the latter: RESULTS.md, 1-NN scene agreement).

Run from the repository root (GPU, a few minutes per backbone; feature extraction dominates):
    python Experiments/mobileNetv2_DINO_P4/feature_probe.py                      # every backbone whose weights exist
    python Experiments/mobileNetv2_DINO_P4/feature_probe.py --backbones ImageNet A1 P4
    python Experiments/mobileNetv2_DINO_P4/feature_probe.py --backbone-file myrun=path/to/backbone_teacher_epoch100.pth
Writes outputs/compare/feature_probe.md and feature_probe.csv (one row per backbone and measure).
"""
import argparse
import csv
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import paths
from analysis_data import BACKBONES, extract_features, labeled_rows, load_backbone_model, pool_rows, test_rows
from data import load_class_names
from utils import resolve_device, set_seed

CONDITIONS = {"scene": lambda r: r["scene"], "az_sector": lambda r: str(int(r["azimuth"]) // 30),
              "depression": lambda r: str(r["depression"]), "band": lambda r: r["band"]}


def standardize(train: torch.Tensor, *others: torch.Tensor) -> list[torch.Tensor]:
    mu, sd = train.mean(0, keepdim=True), train.std(0, keepdim=True).clamp_min(1e-6)
    return [(t - mu) / sd for t in (train, *others)]


def fit_logreg(X: torch.Tensor, y: torch.Tensor, k: int, lam: float, max_iter: int = 300) -> tuple[torch.Tensor, torch.Tensor]:
    W = torch.zeros(X.shape[1], k, device=X.device, requires_grad=True)
    b = torch.zeros(k, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(X @ W + b, y) + 0.5 * lam * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return 100.0 * float((pred == y).float().mean())


def logreg_probe(Xtr, ytr, k, Xte, yte, Xva=None, yva=None, lams=(1e-3, 1e-2, 1e-1)) -> tuple[float, float]:
    """Test accuracy with the L2 strength chosen on validation (or the single lam given)."""
    Xtr, Xte, *rest = standardize(Xtr, Xte, *( [Xva] if Xva is not None else []))
    best = None
    for lam in lams:
        W, b = fit_logreg(Xtr, ytr, k, lam)
        va = accuracy((rest[0] @ W + b).argmax(1), yva) if Xva is not None else 0.0
        if best is None or va > best[0]:
            best = (va, lam, accuracy((Xte @ W + b).argmax(1), yte))
    return best[2], best[1]


def encode(rows: list[dict], getter) -> tuple[torch.Tensor, int]:
    vals = sorted({getter(r) for r in rows})
    code = {v: i for i, v in enumerate(vals)}
    return torch.tensor([code[getter(r)] for r in rows]), len(vals)


def main():
    ap = argparse.ArgumentParser(description="frozen-feature probes: class versus condition, per backbone")
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES), help="names from analysis_data.BACKBONES")
    ap.add_argument("--backbone-file", action="append", default=[], metavar="NAME=PATH", help="extra backbone file(s)")
    ap.add_argument("--per-class-pool", type=int, default=150)
    ap.add_argument("--per-class-test", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = resolve_device(a.device)
    set_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    class_names = load_class_names()
    K = len(class_names)
    t0 = time.time()
    test = test_rows(class_names)
    pool = pool_rows()
    lab1 = labeled_rows("labeled_set_1pct.csv", class_names)
    val = labeled_rows("validation.csv", class_names)
    by_class = lambda rows: {c: [i for i, r in enumerate(rows) if r["label"] == c] for c in range(K)}
    pool_idx = np.sort(np.concatenate([rng.choice(ix, min(a.per_class_pool, len(ix)), replace=False) for ix in by_class(pool).values()]))
    test_idx = np.sort(np.concatenate([rng.choice(ix, min(a.per_class_test, len(ix)), replace=False) for ix in by_class(test).values()]))
    pool_s = [pool[i] for i in pool_idx]
    test_s = [test[i] for i in test_idx]
    print(f"chips: test {len(test)}, test sample {len(test_s)}, pool sample {len(pool_s)}, labeled 1 % {len(lab1)}, "
          f"validation {len(val)} (metadata read in {time.time() - t0:.0f} s)")

    backbones = {n: BACKBONES[n] for n in a.backbones}
    for spec in a.backbone_file:
        n, p = spec.split("=", 1)
        backbones[n] = Path(p)
    records, lines = [], []
    header = ["Backbone", "class LP 1 %", "class NCM 1 %", "class LP 150/class", "1-NN class", "1-NN scene", "1-NN az sector",
              "1-NN depression", "dec. scene", "dec. az sector", "dec. depression", "dec. band"]
    table = []
    for name, path in backbones.items():
        if path is not None and not Path(path).exists():
            print(f"\n{name}: weights not on this machine ({path}); skipped")
            continue
        t1 = time.time()
        model = load_backbone_model(name, path, K)
        feats = {k: extract_features(model, rows, device).to(device)
                 for k, rows in (("test", test), ("pool", pool_s), ("lab1", lab1), ("val", val))}
        feats["test_s"] = feats["test"][torch.tensor(test_idx, device=device)]
        del model
        y = {k: torch.tensor([r["label"] for r in rows], device=device)
             for k, rows in (("test", test), ("pool", pool_s), ("lab1", lab1), ("val", val), ("test_s", test_s))}
        res = {}

        def rec(measure, value, note=""):
            res[measure] = value
            records.append({"backbone": name, "measure": measure, "value": round(value, 2), "note": note})

        # (1) class with few labels
        acc1, lam = logreg_probe(feats["lab1"], y["lab1"], K, feats["test"], y["test"], feats["val"], y["val"])
        rec("class_lp_1pct", acc1, f"test 29,284; lam {lam:g} chosen on validation")
        Z = F.normalize(feats["lab1"], dim=1)
        protos = F.normalize(torch.stack([Z[y["lab1"] == c].mean(0) for c in range(K)]), dim=1)
        rec("class_ncm_1pct", accuracy((F.normalize(feats["test"], dim=1) @ protos.T).argmax(1), y["test"]), "test 29,284")
        # (2) class with many labels
        acc150, _ = logreg_probe(feats["pool"], y["pool"], K, feats["test_s"], y["test_s"], lams=(1e-2,))
        rec("class_lp_150", acc150, f"pool sample {len(pool_s)} -> test sample {len(test_s)}")
        # (3) condition decodability
        for cond, g in CONDITIONS.items():
            ytr, k = encode(pool_s + test_s, g)
            ytr, yte = ytr[:len(pool_s)].to(device), ytr[len(pool_s):].to(device)
            maj = 100.0 * Counter(g(r) for r in test_s).most_common(1)[0][1] / len(test_s)
            acc_c, _ = logreg_probe(feats["pool"], ytr, k, feats["test_s"], yte, lams=(1e-2,))
            rec(f"dec_{cond}", acc_c, f"{k} values, majority {maj:.1f}")
        # (4) neighbourhood on the test sample
        Zs = F.normalize(feats["test_s"], dim=1)
        S = Zs @ Zs.T
        acq = np.array([r["acquisition_group"] for r in test_s])
        S[torch.tensor(acq[:, None] == acq[None, :], device=device)] = -float("inf")
        top = S.topk(10, dim=1).indices.cpu().numpy()
        factors = {"class": np.array([r["label"] for r in test_s]), **{c: np.array([g(r) for r in test_s]) for c, g in CONDITIONS.items()}}
        for fct, vals in factors.items():
            _, cnt = np.unique(vals, return_counts=True)
            chance = float(((cnt / len(vals)) ** 2).sum())
            rec(f"nn1_{fct}", 100 * float((vals[top[:, 0]] == vals).mean()), f"chance {100 * chance:.1f}")
            rec(f"purity10_{fct}", 100 * float((vals[top] == vals[:, None]).mean()), f"chance {100 * chance:.1f}")
        table.append([name] + [f"{res[k]:.1f}" for k in ("class_lp_1pct", "class_ncm_1pct", "class_lp_150", "nn1_class", "nn1_scene",
                                                            "nn1_az_sector", "nn1_depression", "dec_scene", "dec_az_sector", "dec_depression", "dec_band")])
        print(f"\n{name} ({time.time() - t1:.0f} s): " + ", ".join(f"{k} {v:.1f}" for k, v in res.items() if not k.startswith("purity")))
        del feats, S

    maj = {c: 100.0 * Counter(g(r) for r in test_s).most_common(1)[0][1] / len(test_s) for c, g in CONDITIONS.items()}
    chance = {}
    for fct, vals in factors.items():
        _, cnt = np.unique(vals, return_counts=True)
        chance[fct] = 100 * float(((cnt / len(vals)) ** 2).sum())
    lines += ["# Frozen-feature probes: class versus recording condition, per backbone", "",
              "Percent. Class LP 1 %: linear probe on the 652 labeled chips, scored on the full official test set (29,284);",
              "NCM 1 %: nearest class mean of the same chips. Class LP 150/class: probe trained on a pool sample (150 per class),",
              "scored on a test sample (100 per class). 1-NN: nearest-neighbour agreement on the test sample, own acquisition",
              "group excluded. dec.: condition decoded by a probe trained on the pool sample, scored on the test sample.", "",
              f"Chance for 1-NN (sum of squared frequencies): " + ", ".join(f"{k} {v:.1f}" for k, v in chance.items()) + ".",
              f"Majority baseline for dec.: " + ", ".join(f"{k} {v:.1f}" for k, v in maj.items()) + ".", ""]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in table]
    lines += ["", f"Backbones: " + "; ".join(f"{n} = {p.name if p else 'IMAGENET1K_V2'}" + (f" ({p.parent.name})" if p else "")
                                            for n, p in backbones.items() if p is None or Path(p).exists()),
              f"Written {datetime.now().isoformat(timespec='seconds')}."]
    out = paths.OUTPUT_DIR / "compare"
    out.mkdir(parents=True, exist_ok=True)
    (out / "feature_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "feature_probe.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["backbone", "measure", "value", "note"])
        w.writeheader()
        w.writerows(records)
    print("\n" + "\n".join(lines[-len(table) - 3:]))
    print(f"\nWritten to {out}: feature_probe.md, feature_probe.csv")


if __name__ == "__main__":
    main()
