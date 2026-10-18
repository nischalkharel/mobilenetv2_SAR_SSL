"""PDF charts for a run folder, one chart per file. Reads only the CSV/JSON files the run already saved.

Called automatically:
    train.py     -> <run folder>/charts/                (redrawn after every epoch)
                      loss.pdf, accuracy.pdf, overfitting_gap.pdf, learning_rate.pdf
    evaluate.py  -> <run folder>/eval_<split>/charts/
                      per_class_accuracy.pdf, confusion_matrix.pdf, top_mistakes.pdf,
                      confidence.pdf, calibration.pdf

    train_ssl.py -> <SSL run folder>/charts/            (redrawn after every SSL epoch)
                      ssl_loss.pdf, ssl_teacher_entropy.pdf, ssl_feature_similarity.pdf, ssl_schedules.pdf

Redraw for an existing run (for example one trained before this file existed):
    python Experiments/mobileNetv2_DINO/plots.py --run Experiments/mobileNetv2_DINO/outputs/finetune/<run folder>
    python Experiments/mobileNetv2_DINO/plots.py --run Experiments/mobileNetv2_DINO/outputs/ssl/<run folder>
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, path: Path, header: str) -> None:
    """Put the run/metric summary line on top of the chart and write it as its own PDF."""
    fig.suptitle(header, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---- Training curves ----------------------------------------------------------------

def plot_history(run_dir: Path) -> Path:
    """<run>/charts/: loss, accuracy, train-minus-val accuracy gap, learning rate per epoch."""
    run_dir = Path(run_dir)
    out = run_dir / "charts"
    out.mkdir(exist_ok=True)
    rows = read_csv(run_dir / "history.csv")
    col = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    epoch = col("epoch").astype(int)
    tr_acc, va_acc = col("train_acc"), col("val_acc")
    best = int(np.argmax(va_acc))  # same rule as train.py: highest val acc, earliest on ties
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    header = (f"{run_dir.name}\nbest val acc {va_acc[best]:.4f} at epoch {epoch[best]} "
              f"(init: {cfg['init']['type']}, train: {cfg['data']['train_csv']})")

    def new_axes(title, ylabel):
        fig, a = plt.subplots(figsize=(8, 5))
        a.set(title=title, xlabel="epoch", ylabel=ylabel)
        a.grid(alpha=0.3)
        a.axvline(epoch[best], color="r", lw=0.8, ls="--", label=f"best epoch ({epoch[best]})")
        return fig, a

    fig, a = new_axes("Cross-entropy loss", "loss")
    a.plot(epoch, col("train_loss"), "o-", ms=3, label="train")
    a.plot(epoch, col("val_loss"), "o-", ms=3, label="validation")
    a.legend()
    save(fig, out / "loss.pdf", header)

    fig, a = new_axes("Accuracy", "accuracy")
    a.plot(epoch, tr_acc, "o-", ms=3, label="train")
    a.plot(epoch, va_acc, "o-", ms=3, label="validation")
    a.plot(epoch[best], va_acc[best], "r*", ms=14, label="best (saved as best.pt)")
    a.set_ylim(0, 1.02)
    a.legend(loc="lower right")
    save(fig, out / "accuracy.pdf", header)

    fig, a = new_axes("Train minus validation accuracy (larger = more overfitting)", "gap")
    a.bar(epoch, tr_acc - va_acc, color="tab:gray")
    save(fig, out / "overfitting_gap.pdf", header)

    fig, a = new_axes(f"Learning rate at the start of each epoch ({cfg['train']['scheduler']} schedule)", "lr")
    a.plot(epoch, col("lr"), "o-", ms=3, color="tab:green")
    save(fig, out / "learning_rate.pdf", header)
    return out


# ---- DINO SSL curves (new in this experiment; not in the baseline) ---------------------------------

def plot_ssl_history(run_dir: Path) -> Path:
    """<SSL run>/charts/: DINO loss, teacher entropies, feature similarity, schedules per epoch."""
    run_dir = Path(run_dir)
    out = run_dir / "charts"
    out.mkdir(exist_ok=True)
    rows = read_csv(run_dir / "history.csv")
    col = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    epoch = col("epoch").astype(int)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    out_dim = cfg["dino"]["out_dim"]
    header = (f"{run_dir.name}\nDINO SSL, {cfg['optim']['epochs']} epochs planned, out_dim {out_dim}. "
              f"A falling loss alone does not show useful features; check the diagnostics and fine-tuning.")

    def new_axes(title, ylabel):
        fig, a = plt.subplots(figsize=(8, 5))
        a.set(title=title, xlabel="epoch", ylabel=ylabel)
        a.grid(alpha=0.3)
        return fig, a

    fig, a = new_axes("DINO loss (cross-entropy between teacher and student views)", "loss")
    a.plot(epoch, col("loss"), "o-", ms=3)
    save(fig, out / "ssl_loss.pdf", header)

    fig, a = new_axes("Teacher output entropy (collapse check)", "entropy (nats)")
    a.plot(epoch, col("teacher_entropy"), "o-", ms=3, label="per image (low = sharp)")
    a.plot(epoch, col("teacher_marginal_entropy"), "o-", ms=3, label="of the batch average (near 0 = one-hot collapse)")
    a.axhline(np.log(out_dim), color="k", ls="--", lw=1, label=f"ln(out_dim) = {np.log(out_dim):.2f} (uniform collapse)")
    a.legend()
    save(fig, out / "ssl_teacher_entropy.pdf", header)

    fig, a = new_axes("Teacher backbone features across different images (collapse check)", "value")
    a.plot(epoch, col("feature_cosine"), "o-", ms=3, label="mean cosine similarity between images (1 = all identical)")
    a.plot(epoch, col("feature_std"), "o-", ms=3, label="std across images of normalized features (0 = collapse)")
    a.legend()
    save(fig, out / "ssl_feature_similarity.pdf", header)

    fig, a = new_axes("Schedules (value at the end of each epoch)", "value")
    a.plot(epoch, col("lr") / col("lr").max(), "o-", ms=3, label=f"learning rate / {col('lr').max():.3g}")
    a.plot(epoch, col("momentum"), "o-", ms=3, label="teacher EMA momentum")
    a.plot(epoch, col("wd") / max(col("wd").max(), 1e-12), "o-", ms=3, label=f"weight decay / {col('wd').max():.3g}")
    a.legend()
    save(fig, out / "ssl_schedules.pdf", header)
    return out


# ---- Evaluation charts ------------------------------------------------------------

def plot_eval(eval_dir: Path) -> Path:
    """<eval dir>/charts/: per-class accuracy, confusion matrix, top mistakes, confidence, calibration."""
    eval_dir = Path(eval_dir)
    out = eval_dir / "charts"
    out.mkdir(exist_ok=True)
    metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    per_class = read_csv(eval_dir / "per_class_accuracy.csv")
    with open(eval_dir / "confusion_matrix.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    names = rows[0][1:]
    cm = np.array([[int(v) for v in r[1:]] for r in rows[1:]])  # rows = true, columns = predicted
    preds = read_csv(eval_dir / "predictions.csv")
    conf = np.array([float(p["confidence"]) for p in preds])
    correct = np.array([p["correct"] == "1" for p in preds])

    split = metrics["split"]
    header = (f"{eval_dir.parent.name} | {split}: {metrics['num_images']} images | "
              f"overall acc {metrics['overall_accuracy']:.4f} | mean per-class acc "
              f"{metrics['mean_per_class_accuracy']:.4f} | checkpoint epoch {metrics['checkpoint_epoch']}")

    # Per-class accuracy, worst at the top
    order = sorted(per_class, key=lambda r: float(r["accuracy"]))
    acc = [float(r["accuracy"]) for r in order]
    fig, a = plt.subplots(figsize=(10, 11))
    a.barh(range(len(order)), acc, color=plt.cm.RdYlGn(np.array(acc)))
    a.set_yticks(range(len(order)), [f"{r['class_name']} (n={r['num_images']})" for r in order], fontsize=7)
    a.invert_yaxis()
    a.axvline(metrics["overall_accuracy"], color="k", ls="--", lw=1, label="overall accuracy")
    a.axvline(metrics["mean_per_class_accuracy"], color="b", ls=":", lw=1, label="mean per-class accuracy")
    for i, v in enumerate(acc):
        a.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=6)
    a.set(xlim=(0, 1.08), xlabel="accuracy", title=f"Per-class accuracy ({split}), lowest first")
    a.legend(loc="upper right")
    save(fig, out / "per_class_accuracy.pdf", header)

    # Confusion matrix, each row divided by its class size (so rows sum to 1)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, a = plt.subplots(figsize=(12, 11))
    im = a.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    a.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
    a.set_yticks(range(len(names)), names, fontsize=6)
    a.set(xlabel="predicted class", ylabel="true class",
          title=f"Confusion matrix ({split}), row-normalized: cell = fraction of the true class predicted as that column")
    fig.colorbar(im, fraction=0.04)
    save(fig, out / "confusion_matrix.pdf", header)

    # The 25 largest off-diagonal confusions
    off = cm.copy()
    np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1][:25], off.shape))[0]
    top = [(t, p) for t, p in top if off[t, p] > 0]
    fig, a = plt.subplots(figsize=(10, 9))
    a.barh(range(len(top)), [off[t, p] for t, p in top], color="tab:red")
    a.set_yticks(range(len(top)), [f"{names[t]}  ->  {names[p]}" for t, p in top], fontsize=7)
    a.invert_yaxis()
    for i, (t, p) in enumerate(top):
        a.text(off[t, p], i, f" {off[t, p]} ({off[t, p] / cm[t].sum():.0%} of true class)", va="center", fontsize=6)
    a.set(xlabel="number of images", title=f"Most frequent mistakes ({split}): true class -> predicted class")
    save(fig, out / "top_mistakes.pdf", header)

    # Confidence (highest softmax probability) of correct vs wrong predictions
    bins = np.linspace(0, 1, 21)
    fig, a = plt.subplots(figsize=(8, 5))
    a.hist(conf[correct], bins=bins, alpha=0.7, label=f"correct ({correct.sum()})")
    a.hist(conf[~correct], bins=bins, alpha=0.7, label=f"wrong ({(~correct).sum()})")
    a.set(xlabel="confidence of the predicted class", ylabel="images",
          title=f"Confidence of correct vs wrong predictions ({split})")
    a.legend()
    a.grid(alpha=0.3)
    save(fig, out / "confidence.pdf", header)

    # Calibration: accuracy within each confidence bin
    idx = np.clip(np.digitize(conf, bins) - 1, 0, len(bins) - 2)
    centers = (bins[:-1] + bins[1:]) / 2
    filled = [i for i in range(len(centers)) if (idx == i).any()]
    fig, a = plt.subplots(figsize=(6.5, 6))
    a.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    a.plot(centers[filled], [correct[idx == i].mean() for i in filled], "o-", label="model")
    a.set(xlabel="confidence", ylabel="actual accuracy in that bin", xlim=(0, 1), ylim=(0, 1),
          title=f"Calibration ({split}): is the model as accurate as it is confident?")
    a.legend()
    a.grid(alpha=0.3)
    save(fig, out / "calibration.pdf", header)
    return out


def main():
    parser = argparse.ArgumentParser(description="Redraw the PDF charts for a run folder")
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if (args.run / "checkpoint.pth").exists():  # an SSL run folder
        print(f"Wrote charts to {plot_ssl_history(args.run)}")
        return
    print(f"Wrote charts to {plot_history(args.run)}")
    for eval_dir in sorted(args.run.glob("eval_*")):
        print(f"Wrote charts to {plot_eval(eval_dir)}")


if __name__ == "__main__":
    main()
