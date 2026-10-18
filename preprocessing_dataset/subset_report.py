"""Document the saved subsets: SUBSETS.md (tables) and charts/*.pdf (one chart per file).

Reads only the CSVs in this folder and the image names in the official test folder, so it
always describes the subsets that experiments actually use. preprocessing_dataset.py calls it
after writing the CSVs; it can also be run on its own:
    python preprocessing_dataset/subset_report.py

Writes, next to this script:
    SUBSETS.md                          files, leakage rules, per-class tables
    charts/subset_sizes.pdf             table: images in each subset
    charts/subset_sizes_bar.pdf         bar chart of the same numbers
    charts/per_class_table.pdf          table: images per class in every subset
    charts/per_class_bars.pdf           bar chart: images per class in validation, excluded, SSL pool, labeled
    charts/excluded_similarity.pdf      how similar each excluded near-copy is to its validation image
"""
import csv
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so "import paths" works
import paths  # noqa: E402

# Reference medians measured on 2026-10-13 (vehicle region, maximum normalized cross-correlation,
# about 400 random pose groups). Used to put the excluded images' similarity in context.
REFERENCE_SIMILARITY = [
    ("same pose, same scene, different polarization", 0.70),
    ("same vehicle, 5 deg apart, same scene", 0.63),
    ("same pose, different scene", 0.47),
    ("different vehicle type", 0.32),
]


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def table_figure(cell_text, col_labels, title, path: Path, col_widths=None, font=9, row_height=0.3, bold_last=True):
    """Draw a plain table as its own PDF page. Text columns left-aligned, numbers right-aligned."""
    fig, a = plt.subplots(figsize=(11, 0.9 + row_height * (len(cell_text) + 1)))
    a.axis("off")
    t = a.table(cellText=cell_text, colLabels=col_labels, loc="upper center", colWidths=col_widths, cellLoc="right")
    t.auto_set_font_size(False)
    t.set_fontsize(font)
    t.scale(1, 1.25)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dde6f0")
        elif bold_last and r == len(cell_text):  # total row
            cell.set_text_props(weight="bold")
        if r > 0 and not cell_text[r - 1][c].replace(",", "").replace(".", "").replace("%", "").isdigit():
            cell.set_text_props(ha="left")
            cell.PAD = 0.03
    a.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_report(val_per_class: int, labeled_fraction: float, seed: int) -> None:
    out_dir = paths.SPLITS_DIR
    charts = out_dir / "charts"
    charts.mkdir(exist_ok=True)
    classes = [r["class_name"] for r in read_rows(out_dir / "class_mapping.csv")]
    excluded_rows = read_rows(out_dir / "excluded_near_duplicates.csv")
    val = Counter(r["class_name"] for r in read_rows(out_dir / "validation.csv"))
    exc = Counter(r["class_name"] for r in excluded_rows)
    ssl = Counter(r["class_name"] for r in read_rows(out_dir / "ssl_training_data.csv"))
    lab = Counter(r["class_name"] for r in read_rows(out_dir / "labeled_set.csv"))
    test = Counter({c: sum(1 for p in (paths.SOC40_TEST_DIR / c).iterdir() if p.suffix == ".tif") for c in classes})
    train = Counter({c: val[c] + exc[c] + ssl[c] for c in classes})  # the three together = official train
    n = {k: sum(v.values()) for k, v in dict(val=val, exc=exc, ssl=ssl, lab=lab, test=test, train=train).items()}
    rng = lambda cnt: f"{min(cnt[c] for c in classes):,} to {max(cnt[c] for c in classes):,}"  # noqa: E731
    sim_v = np.array([float(r["similarity_vehicle_region"]) for r in excluded_rows])
    sim_f = np.array([float(r["similarity_full_image"]) for r in excluded_rows])

    # ---- Chart: subset sizes (table and bar) ------------------------------------------
    sizes = [
        ("Official SOC-40 train", n["train"], "100%", rng(train), "validation + excluded + SSL pool"),
        ("validation.csv", n["val"], f"{n['val'] / n['train']:.1%}", rng(val),
         f"picks the best epoch; {val_per_class} per class, all different poses"),
        ("excluded_near_duplicates.csv", n["exc"], f"{n['exc'] / n['train']:.1%}", rng(exc),
         "near-copies of validation images; used nowhere"),
        ("ssl_training_data.csv (SSL pool)", n["ssl"], f"{n['ssl'] / n['train']:.1%}", rng(ssl),
         "unlabeled pretraining"),
        ("labeled_set.csv", n["lab"], f"{n['lab'] / n['train']:.1%}", rng(lab),
         f"{n['lab'] / n['ssl']:.1%} of SSL pool; labeled training"),
        ("Official SOC-40 test", n["test"], "-", rng(test), "held out; in no CSV"),
    ]
    table_figure([[s[0], f"{s[1]:,}", s[2], s[3], s[4]] for s in sizes],
                 ["Subset", "Images", "% of official train", "Images per class", "Use"],
                 f"SOC-40 subsets (seed {seed}, 40 classes)", charts / "subset_sizes.pdf",
                 col_widths=[0.27, 0.09, 0.13, 0.13, 0.38], font=9, row_height=0.4, bold_last=False)
    fig, a = plt.subplots(figsize=(10, 4.5))
    names = ["validation", "excluded\nnear-copies", "SSL pool", "labeled set\n(inside SSL pool)", "official test"]
    vals = [n["val"], n["exc"], n["ssl"], n["lab"], n["test"]]
    bars = a.bar(names, vals, color=["tab:orange", "tab:red", "tab:blue", "tab:green", "tab:gray"])
    a.bar_label(bars, labels=[f"{v:,}" for v in vals])
    a.set(ylabel="images", title=f"Images in each SOC-40 subset (official train = {n['train']:,})")
    a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(charts / "subset_sizes_bar.pdf")
    plt.close(fig)

    # ---- Chart: per-class table and bars ------------------------------------------------
    rows = [[str(i), c, f"{val[c]:,}", f"{exc[c]:,}", f"{ssl[c]:,}", f"{lab[c]:,}", f"{train[c]:,}", f"{test[c]:,}"]
            for i, c in enumerate(classes)]
    rows.append(["", "Total", f"{n['val']:,}", f"{n['exc']:,}", f"{n['ssl']:,}", f"{n['lab']:,}",
                 f"{n['train']:,}", f"{n['test']:,}"])
    table_figure(rows, ["Label", "Class", "Validation", "Excluded", "SSL pool", "Labeled", "Official train",
                        "Official test"],
                 "Images per class in each subset (labeled set is inside the SSL pool)",
                 charts / "per_class_table.pdf", col_widths=[0.06, 0.3, 0.1, 0.1, 0.1, 0.09, 0.12, 0.12], font=7,
                 row_height=0.2)
    fig, ax = plt.subplots(1, 4, figsize=(16, 11), sharey=True)
    for a, cnt, name, color in zip(ax, [val, exc, ssl, lab], ["validation", "excluded near-copies", "SSL pool",
                                                             "labeled set"],
                                   ["tab:orange", "tab:red", "tab:blue", "tab:green"]):
        v = [cnt[c] for c in classes]
        a.barh(range(len(classes)), v, color=color)
        for i, x in enumerate(v):
            a.text(x, i, f" {x}", va="center", fontsize=6)
        a.set(title=f"{name} ({sum(v):,})", xlabel="images", xlim=(0, max(v) * 1.18))
        a.grid(axis="x", alpha=0.3)
    ax[0].set_yticks(range(len(classes)), classes, fontsize=7)
    ax[0].invert_yaxis()
    fig.suptitle("Images per class (each panel has its own scale)")
    fig.tight_layout()
    fig.savefig(charts / "per_class_bars.pdf")
    plt.close(fig)

    # ---- Chart: similarity of excluded near-copies ----------------------------------------
    fig, a = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-0.1, 1, 45)
    a.hist(sim_v, bins=bins, alpha=0.75, label=f"vehicle region (median {np.median(sim_v):.2f})")
    a.hist(sim_f, bins=bins, alpha=0.55, label=f"full image (median {np.median(sim_f):.2f})")
    for (name, value), ls in zip(REFERENCE_SIMILARITY, ["-", "--", "-.", ":"]):
        a.axvline(value, color="k", ls=ls, lw=1, label=f"reference median, vehicle region: {name} ({value:.2f})")
    a.set(xlabel="similarity to its validation image (1 = identical up to a shift and brightness)",
          ylabel="excluded images",
          title=f"How similar the {n['exc']:,} excluded images are to their validation image")
    a.legend(fontsize=7, loc="upper left")
    a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(charts / "excluded_similarity.pdf")
    plt.close(fig)

    # ---- SUBSETS.md ---------------------------------------------------------------------
    def md_table(header, body):
        lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" if h != "Class name" else "---"
                                                               for h in header) + "|"]
        return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in body])

    pct = lambda a, b: f"{a / b:.1%}"  # noqa: E731
    lab_table = md_table(["Label", "Class name", "Labeled images", "Class SSL-pool total", "% of class"],
                         [[str(i), c, f"{lab[c]:,}", f"{ssl[c]:,}", pct(lab[c], ssl[c])] for i, c in enumerate(classes)]
                         + [["", "**Total**", f"**{n['lab']:,}**", f"**{n['ssl']:,}**", f"**{pct(n['lab'], n['ssl'])}**"]])
    all_table = md_table(["Label", "Class name", "Validation", "Excluded near-copies", "SSL pool",
                          "Labeled (inside SSL pool)", "Official train", "Official test"],
                         rows[:-1] + [["", "**Total**"] + [f"**{x}**" for x in rows[-1][2:]]])
    pol = Counter(f"{r['validation_image_polarization']} -> {r['polarization']}" for r in excluded_rows)
    same_pol = sum(1 for r in excluded_rows if r["validation_image_polarization"] == r["polarization"])
    ref = "\n".join(f"| {name} | {value:.2f} |" for name, value in REFERENCE_SIMILARITY)
    md = f"""# SOC-40 data subsets

Fixed subsets of ATRNet-STAR SOC-40 (Ground_Range, 8-bit amplitude) used by every experiment in this project.
Generated by `preprocessing_dataset.py` with `SEED = {seed}`, `VAL_PER_CLASS = {val_per_class}`, `LABELED_FRACTION = {labeled_fraction}`. Do not regenerate them between experiments; reuse the CSVs.
This file and `charts/` are written by `subset_report.py` from the CSVs (last written {date.today().isoformat()}).

## Files

| File | Images | Contents |
|---|---:|---|
| `validation.csv` | {n['val']:,} | {val_per_class} per class from official SOC-40 train, each from a different pose (one image per pose) |
| `excluded_near_duplicates.csv` | {n['exc']:,} | Near-copies of validation images (same acquisition group). Used in no training set. Each row names its validation image and their similarity |
| `ssl_training_data.csv` | {n['ssl']:,} | Every other official SOC-40 train image (the SSL pool) |
| `labeled_set.csv` | {n['lab']:,} | {labeled_fraction:.0%} of the SSL pool, sampled per class image by image; used for labeled training (simulates a small labeled budget) |
| `class_mapping.csv` | 40 rows | Label number (0 to 39) for each class name, in case-sensitive sorted order |
| Official `test/` folder | {n['test']:,} | Held out; not listed in any CSV |

Paths in the CSVs are relative to `paths.SOC40_DIR` (see `paths.py`). Validation + excluded + SSL pool = all {n['train']:,} official train images.

Charts (`charts/`, one per file): `subset_sizes.pdf`, `subset_sizes_bar.pdf` (images in each subset), `per_class_table.pdf`, `per_class_bars.pdf` (images per class in each subset), `excluded_similarity.pdf` (similarity of each excluded image to its validation image).

## Rules that prevent leakage

- An **acquisition group** is one vehicle type at one band, depression angle, azimuth angle and scene (for example `Buick_GL8|X|15|110|factory`; the scene comes from the image's .xml). Its images differ only by polarization (HH, HV, VH, VV) or by a repeat pass over the same spot, so they are near-copies.
- Validation takes one image from each chosen pose. The other images of that image's acquisition group go to `excluded_near_duplicates.csv` and are used nowhere, so no near-copy of a validation image is in the SSL pool or the labeled set.
- The same vehicle at the same angle in a **different scene** is not a near-copy (see the measurements below) and stays in the SSL pool.
- The labeled set is entirely inside the SSL pool. It shares no image or acquisition group with validation, and no image with test.
- No official test image appears in any CSV.
- All 97,375 SOC-40 images were hashed: no exact pixel duplicates exist within or across train and test.

### Why "same scene" defines a near-copy (measured 2026-10-13)

Similarity = maximum normalized cross-correlation over all shifts, on the vehicle bounding box from the .xml (plus 4 pixels), after standardizing brightness. 1 means identical up to a shift and brightness. Medians over about 400 random pose groups:

| Pair of images | Median similarity (vehicle region) |
|---|---:|
{ref}

The excluded images in this split: vehicle-region similarity to their validation image median {np.median(sim_v):.2f} (10th to 90th percentile {np.percentile(sim_v, 10):.2f} to {np.percentile(sim_v, 90):.2f}); full image median {np.median(sim_f):.2f}. {same_pol:,} of the {n['exc']:,} have the same polarization as their validation image (repeat passes); the rest differ only in polarization. Most common pairs (validation polarization -> excluded polarization): {", ".join(f"{k} {v}" for k, v in pol.most_common(6))}.

Remaining limitation: the same vehicle 5 degrees apart in the same scene is also quite similar (about 0.63) and is not excluded. That is normal for SAR benchmarks and is part of recognizing a vehicle across nearby angles, but it should be stated.

Known limitation of the official benchmark (not fixable without abandoning it): SOC-40's train/test split was random per image, so 12,779 of 15,321 pose groups have images in both official train and test. This affects all methods equally but likely inflates SOC-40 test accuracy and should be stated when reporting results.

## Labeled set per class

{labeled_fraction:.0%} (rounded to the nearest image) of each class's SSL-pool images.

{lab_table}

## All subsets per class

{all_table}
"""
    (out_dir / "SUBSETS.md").write_text(md, encoding="utf-8")
    print(f"Wrote {out_dir / 'SUBSETS.md'} and charts to {charts}")


if __name__ == "__main__":
    from preprocessing_dataset import LABELED_FRACTION, SEED, VAL_PER_CLASS
    make_report(VAL_PER_CLASS, LABELED_FRACTION, SEED)
