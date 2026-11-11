"""Charts written into run folders: training curves (fine-tune and SSL) and per-class bars."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_history(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0] if rows else []:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        out[k] = vals
    return out


def _plot(ax, h, keys, title, logy=False):
    for k in keys:
        if k in h:
            ax.plot(h["epoch"], h[k], marker=".", label=k)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=7)


def finetune_curves(history_csv, out_png):
    h = read_history(history_csv)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    _plot(axes[0], h, ["loss", "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"], "training losses")
    _plot(axes[1], h, ["val_select_map", "val_select_map50"], "model-selection mAP (val_select)")
    _plot(axes[2], h, ["lr"], "learning rate", logy=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def ssl_curves(history_csv, out_png, loss_keys, diag_keys):
    h = read_history(history_csv)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    _plot(axes[0], h, loss_keys, "SSL loss")
    _plot(axes[1], h, diag_keys, "collapse diagnostics")
    _plot(axes[2], h, ["lr", "wd", "momentum", "teacher_temp"], "schedule", logy=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def per_class_bars(metrics_by_name: dict, out_png, key="ap50", title="per-class AP50"):
    """metrics_by_name: {label: metrics dict (with per_class)}; grouped bars per class."""
    names = list(next(iter(metrics_by_name.values()))["per_class"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    n = len(metrics_by_name)
    width = 0.8 / max(n, 1)
    for j, (label, m) in enumerate(metrics_by_name.items()):
        vals = [m["per_class"][c][key] for c in names]
        ax.bar([i + j * width for i in range(len(names))], vals, width, label=label)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(names))])
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
