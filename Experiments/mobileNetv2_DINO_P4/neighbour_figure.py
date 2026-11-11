"""Nearest neighbours in feature space, A1 versus P4: the picture of the neighbourhood numbers in feature_probe.py.

For a fixed set of query chips (the first sampled chip of each scene, seed 0; not chosen by result) the k nearest
test chips by cosine similarity of the frozen backbone features are shown for each backbone, own acquisition group
(same class, band, depression, azimuth and scene) excluded, each labelled with class, scene and aspect angle; a green
frame = same vehicle class as the query, red = a different one. Over the whole test sample (100 chips per class) the
share of the k nearest that share the query's class / scene / aspect sector is written next to the figure, so the
picture and the number come from the same computation.

Run from the repository root (GPU, about two minutes; the backbone files must be on this machine):
    python Experiments/mobileNetv2_DINO_P4/neighbour_figure.py
    python Experiments/mobileNetv2_DINO_P4/neighbour_figure.py --models A1_ft1pct P4_ft1pct --out-name neighbour_figure_ft1pct
    python Experiments/mobileNetv2_DINO_P4/neighbour_figure.py --model A1=path/to/backbone_teacher_epoch100.pth
    python Experiments/mobileNetv2_DINO_P4/neighbour_figure.py --model myft=finetuned:path/to/best.pt
Writes outputs/compare/<out-name>.png / .pdf and <out-name>_summary.md (default out-name: neighbour_figure).
"""
import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paths
from analysis_data import LABELS, MODELS, extract_features, load_model, model_spec, test_rows
from data import load_class_names, load_image
from utils import resolve_device, set_seed


def short(name: str, n: int = 16) -> str:
    s = name.replace("_", " ")
    return s if len(s) <= n else s[:n - 1] + "."


def main():
    ap = argparse.ArgumentParser(description="nearest-neighbour panel, A1 vs P4 (and others), frozen features")
    ap.add_argument("--models", nargs="+", default=["A1", "P4"], help="names from analysis_data.MODELS")
    ap.add_argument("--model", action="append", default=[], metavar="NAME=[finetuned:]PATH", help="override or add a model file")
    ap.add_argument("--out-name", default="neighbour_figure")
    ap.add_argument("--per-class-test", type=int, default=100)
    ap.add_argument("--queries", type=int, default=6, help="query chips in the figure (one per scene first)")
    ap.add_argument("--k", type=int, default=5)
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
    test = test_rows(class_names)
    by_class = {c: [i for i, r in enumerate(test) if r["label"] == c] for c in range(K)}
    idx = np.sort(np.concatenate([rng.choice(ix, min(a.per_class_test, len(ix)), replace=False) for ix in by_class.values()]))
    sample = [test[i] for i in idx]
    acq = np.array([r["acquisition_group"] for r in sample])
    cls = np.array([r["label"] for r in sample])
    scene = np.array([r["scene"] for r in sample])
    az = np.array([int(r["azimuth"]) // 30 for r in sample])

    files = {n: MODELS[n] for n in a.models}
    for spec in a.model:
        n, kind, p = model_spec(spec)
        files[n] = (kind, p)

    # query chips: the first sampled chip of each scene, then second chips, until `queries` are chosen
    scenes = sorted(set(scene))
    queries = []
    for round_ in range(5):
        for sc in scenes:
            cands = np.where(scene == sc)[0]
            if len(cands) > round_ and len(queries) < a.queries:
                queries.append(int(cands[round_]))
    queries = sorted(set(queries))

    neighbours, shares = {}, {}
    same_acq = torch.tensor(acq[:, None] == acq[None, :], device=device)
    for name, (kind, path) in files.items():
        model = load_model(name, kind, path, K)
        Z = F.normalize(extract_features(model, sample, device).to(device), dim=1)
        del model
        S = Z @ Z.T
        S[same_acq] = -float("inf")   # excludes the chip itself as well (same acquisition group)
        top = S.topk(a.k, dim=1).indices.cpu().numpy()
        neighbours[name] = top
        shares[name] = {"class": float((cls[top] == cls[:, None]).mean()), "scene": float((scene[top] == scene[:, None]).mean()),
                        "aspect sector": float((az[top] == az[:, None]).mean())}
        print(f"{name}: of the {a.k} nearest, same class {100 * shares[name]['class']:.1f} %, same scene "
              f"{100 * shares[name]['scene']:.1f} %, same aspect sector {100 * shares[name]['aspect sector']:.1f} %")
        del S, Z

    chance = {"class": float(((np.bincount(cls) / len(cls)) ** 2).sum()),
              "scene": float(((np.unique(scene, return_counts=True)[1] / len(scene)) ** 2).sum()),
              "aspect sector": float(((np.unique(az, return_counts=True)[1] / len(az)) ** 2).sum())}
    lines = [f"# Nearest neighbours in feature space ({a.k} nearest, cosine, own acquisition group excluded)", "",
             f"Test sample: {len(sample)} official test chips ({a.per_class_test} per class, seed {a.seed}). Share of the {a.k} nearest",
             "chips that share the query's class / scene / 30-degree aspect sector, percent (chance = sum of squared frequencies).", "",
             "| Model | same class | same scene | same aspect sector |", "|---|---|---|---|"]
    lines += [f"| {LABELS.get(n, n)} | {100 * s['class']:.1f} | {100 * s['scene']:.1f} | {100 * s['aspect sector']:.1f} |" for n, s in shares.items()]
    lines += [f"| chance | {100 * chance['class']:.1f} | {100 * chance['scene']:.1f} | {100 * chance['aspect sector']:.1f} |", "",
              f"Figure queries: {', '.join(sample[q]['relative_path'] for q in queries)}.",
              f"Written {datetime.now().isoformat(timespec='seconds')}."]

    # ---- figure: per query, one row per backbone: [query | k neighbours] ----
    names = list(files)
    n_rows = len(queries) * len(names)
    fig, axes = plt.subplots(n_rows, a.k + 1, figsize=(1.75 * (a.k + 1), 2.0 * n_rows))
    axes = np.atleast_2d(axes)
    for qi, q in enumerate(queries):
        for bi, name in enumerate(names):
            row = qi * len(names) + bi
            ax = axes[row, 0]
            ax.imshow(np.asarray(load_image(sample[q]["relative_path"]), dtype=np.float32), cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"query, {LABELS.get(name, name)} neighbours\n{short(sample[q]['class_name'])}\n{sample[q]['scene']}, az {sample[q]['azimuth']}", fontsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("white")
                sp.set_linewidth(2)
            for j, nb in enumerate(neighbours[name][q]):
                ax = axes[row, j + 1]
                r = sample[nb]
                ax.imshow(np.asarray(load_image(r["relative_path"]), dtype=np.float32), cmap="gray", vmin=0, vmax=255)
                same = r["label"] == sample[q]["label"]
                ax.set_title(f"{short(r['class_name'])}\n{r['scene']}, az {r['azimuth']}", fontsize=7,
                             color="#1a7f37" if same else "#b3261e")
                for sp in ax.spines.values():
                    sp.set_edgecolor("#1a7f37" if same else "#b3261e")
                    sp.set_linewidth(2.5)
            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
    fig.suptitle(f"{a.k} nearest test chips in feature space ({' vs '.join(LABELS.get(n, n) for n in names)}); green = same vehicle "
                 f"class as the query, red = different. Same {a.k}-NN class share: "
                 + ", ".join(f"{LABELS.get(n, n)} {100 * shares[n]['class']:.0f} %" for n in names),
                 fontsize=8.5)
    fig.tight_layout()
    out = paths.OUTPUT_DIR / "compare"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{a.out_name}.png", dpi=200)
    fig.savefig(out / f"{a.out_name}.pdf")
    (out / f"{a.out_name}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten to {out}: {a.out_name}.png / .pdf, {a.out_name}_summary.md")


if __name__ == "__main__":
    main()
