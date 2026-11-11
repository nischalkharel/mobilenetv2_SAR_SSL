# P4: DINO with per-condition-cell teacher centering (MobileNetV2 on unlabeled SAR)

Self-contained copy of the A1 route (`Experiments/mobileNetv2_DINO`) with ONE change inside the DINO loss. Nothing in
this folder imports from the A1 folder, and nothing in the A1 folder was touched. The route is: ImageNet MobileNetV2
-> DINO SSL on the unlabeled pool with per-cell centering (100 epochs) -> supervised fine-tuning on the 1 %, 3 % and
10 % labeled sets -> validation and test scores. Every setting other than the centering is the A1 value, so a P4 run
and the A1 run of the same split differ only in the centering.

## What changes, in plain words

DINO keeps the teacher's outputs from collapsing by subtracting a running average of the teacher output over all
images (the "center") before the sharpened softmax. "Different from the average image" is cheapest to achieve with the
recording condition (scene, radar band, depression angle), so a compact backbone learns to describe the condition
before the vehicle (measured in `../Independent_Study_For_agents/`, a sibling folder of this repository: frozen DINO
features decode the scene at 97 % and
the class at 62 %; a chip is closer to a different vehicle under the same condition than to the same vehicle in
another scene).

P4 keeps one running average PER CONDITION CELL and subtracts each image's own cell average:

    A1:  target = softmax( (teacher(x)        - center              ) / temperature )
    P4:  target = softmax( (teacher(x)        - center[cell(x)]     ) / temperature )

A cell is one combination of sensor-recorded fields of the chip. Inside a cell the condition is constant, so it can no
longer make an image "unusual"; what remains to spread the images by is the target and its pose. Each cell's center
is updated with the same momentum (0.9) from the teacher outputs of that cell's images in the batch; batches stay
mixed. Views, head, temperatures, EMA teacher, optimizer, schedules, checkpointing and export are unchanged.

Pilot evidence (`../Independent_Study_For_agents/pilot/RESULTS.md`, 25 epochs, azimuth-sector holdout, frozen probes):
P4 +4.5 (16-shot linear probe), +5.4 (10 %) over the unchanged recipe, on test chips at angles never seen in SSL; the
scrambled-cell control gained nothing. This folder is the real test: full pools, 100 epochs, three splits, fine-tuning.

## Metadata used (and the label-leak check)

| Field | Read from the SSL CSV column | Meaning | Values |
|---|---|---|---|
| `scene` | `scene` | where the vehicle was parked | urban, factory, sandstone, bare_soil, woodland |
| `band` | `band` | radar frequency | KU, X (urban is X only) |
| `depression` | `depression` | look-down angle in degrees | 15, 30, 45, 60 |
| `az_sector30` | `azimuth` // 30 | viewing-angle slice | 12 |

`config_ssl.json` uses all four (as the pilot did). `config_ssl_3field.json` drops the azimuth sector, which in this
dataset is measured relative to the target heading and may not be known for an unlabeled chip in the field; the
first three fields are always logged by the sensor. The class columns of the CSV are never read by training
(`run_info.json["cond_center"]["columns_read_from_ssl_csv"]` lists exactly what was read).

`cell_report.py` (CPU, reads labels for the report only) shows that cells carry the condition, not the target. On
the split-1 pool the 432 four-field cells hold 17 to 668 images (median 103, mean 151) and 7 to 40 classes (median
36); no cell has a class above 23.5 % of its images; guessing the class from the cell alone gives 4.9 % (chance 2.5 %),
and cells of the same sizes with scrambled membership give 5.6 %, so the cells add no class information beyond the
scene (mutual information of cell and class minus that of scene and class: 0.11 bits, below the 0.20 bits that
scrambled cells of the same sizes show from finite samples). The scene itself carries 0.16 bits (3 % of the class
entropy) because bare_soil and woodland hold only 11 classes, and A1's DINO features already decode the scene at 97 %
from the pixels, so the metadata gives the network nothing about the class that the pixels do not already give. The
smoke test fails if the median classes per cell drops below 20, any cell has fewer than 2, any cell has a class above
50 %, or the cell-only majority vote exceeds 10 %. See `outputs/cell_report/` for the per-split numbers.

`config_ssl_perm.json` is the scrambled control (P4perm): same cells and histogram, random image -> cell mapping. It
must not gain over A1; if it does, the gain is not coming from the physical grouping.

## Files

| File | Status vs A1 | What it does |
|---|---|---|
| `cond_center.py` | new | cell assignment from the CSV rows, `CellSSLDataset` (views + cell id), `CondCenterDINOLoss` |
| `train_ssl.py` | modified (CHANGES.md 2.3) | reads the cell fields, builds the per-cell loss, writes `cells.csv`, `cell_center_stats.csv`, `cells_seen` |
| `ssl_data.py` | modified (CHANGES.md 2.1) | `flat_args()` carries the `cond_center` section; per-process temp name for the pixel cache; `make_run_dir()` (stamped run folders safe for jobs started in the same second) |
| `export_backbone.py` | modified (CHANGES.md 2.2) | records "per-condition-cell" in the backbone file's `ssl_method`; command path |
| `smoke_test.py` | modified (CHANGES.md 2.4) | A1 checks kept, P4 checks added (cells, per-cell loss, real step, resume, export, imports, stamp collision) |
| `paths.py` | docstring only | identical settings; outputs land under this folder |
| `train_supervised.py` | modified (CHANGES.md 2.6) | run folder via `make_run_dir()`; training logic unchanged |
| `evaluate.py`, `plots.py`, `preview_views.py` | command paths in docstrings / printed hints only | unchanged logic |
| `dino.py`, `data.py`, `model.py`, `engine.py`, `utils.py`, `reference/` | identical | byte-for-byte A1 copies |
| `config_ssl.json` | A1 values + `cond_center` section | P4 |
| `config_ssl_perm.json`, `config_ssl_3field.json` | variants | scrambled control; deployable cells |
| `config_finetune.json` (+ `_perm`, `_3field`) | A1 values, run names | fine-tuning, identical recipe to A0 / A1 |
| `cell_report.py` | new | cell table and label-leak numbers for a split (CPU) |
| `compare_p4.py` | new | P4 / P4perm / P4d versus A1 and A0 from the finished CRC runs: tables and SSL cost (CPU) |
| `analysis_data.py`, `feature_probe.py`, `saliency_figure.py` | new | analyses of the finished backbones (GPU, minutes; `crc/p4_analysis.sbatch`): test-chip metadata from the .xml sidecars (scene, vehicle box); frozen-feature probes (class vs scene / azimuth sector / depression / band); Grad-CAM and backbone response maps with the share of map mass on the vehicle box, A1 vs P4 |
| `make_changes_record.py` | new | regenerates `diffs/` and `diffs/SUMMARY.md` (status and hashes of every file vs A1) |
| `CHANGES.md` | new | every hunk explained, and whether it can change what DINO computes |

## Commands on the PC (Git Bash, from the repository root)

```bash
cd <path to>/mobilenetv2_SAR_SSL
source .venv/Scripts/activate
python Experiments/mobileNetv2_DINO_P4/smoke_test.py          # about 3 minutes on the GPU, not a result; every line [PASS]
python Experiments/mobileNetv2_DINO_P4/cell_report.py         # seconds; split 1 cells (SAR_SPLIT=2 / 3 for the others)
python Experiments/mobileNetv2_DINO_P4/make_changes_record.py # after any edit: diffs/ and diffs/SUMMARY.md
```

A short real run on the PC (same schedule as the full run, stopped early; resumable) if ever wanted:
```bash
python Experiments/mobileNetv2_DINO_P4/train_ssl.py --stop-after-epoch 2
```

## Commands on CRC (from the repository root, after `git pull`)

```bash
mkdir -p Experiments/mobileNetv2_DINO_P4/outputs/cache
cp Experiments/mobileNetv2_DINO/outputs/cache/pixels_*.npy Experiments/mobileNetv2_DINO_P4/outputs/cache/   # optional: reuse A1's decoded pixels (saves about 15 min per job; the file is spot-checked against the images)
sbatch crc/p4.sbatch 1                                    # split 1: smoke test, cell report, SSL 100 epochs, fine-tune 1 / 3 / 10 %, validation + test
sbatch crc/p4.sbatch 2
sbatch crc/p4.sbatch 3
sbatch --job-name=P4perm_dino crc/p4.sbatch 1 perm        # scrambled-cell control on split 1
sbatch --job-name=P4d_dino crc/p4.sbatch 1 3field         # deployable cells on split 1
```
One job is 10 to 12 h wall time on an A100 (A1's split jobs took 9 h 40 min and 11 h 47 min: SSL training 7.8 h plus
checkpointing, the 20-minute smoke test, three fine-tunes and six evaluations). The job asks for 20 h. Watch with
`crc-squeue`; the log is `crc/logs/<job name>_<job id>.out`. Jobs on the same split may run at the same time (each
writes the shared pixel cache under a per-process temporary name). Afterwards commit the small result files and push
(the checkpoints and the pixel cache are git-ignored).

## Outputs

`outputs/ssl/<stamp>_P4_dino_ssl_mobilenetv2[_split<N>]_seed0/`: everything an A1 run writes (`config.json`,
`manifests.json` with the cell definition and the cell-assignment hash, `run_info.json` with the per-epoch cell
statistics, `history.csv` with the extra `cells_seen` column, charts, `checkpoint.pth` including the cell centers,
`backbone_teacher_epoch100.pth`) plus `cells.csv` (cell -> number of distinct images; a `mapping` column says
"physical" or "scrambled control") and `cell_center_stats.csv` (visits = image presentations summed over the epochs
so far, and center size per cell; rewritten every epoch). The `teacher_entropy` columns of `history.csv` are computed
with the per-cell centers (the actual training target), so they are comparable with A1's in meaning, not in value.

`outputs/finetune/<stamp>_P4_dino_mobilenetv2[_labeled_set_1pct | _labeled_set_10pct][_split<N>]_seed0/`: the same
layout as the A1 fine-tuning runs, with `eval_validation/` and `eval_test/` after `evaluate.py`.

## What to compare

| P4 run (this folder) | A1 counterpart (`Experiments/mobileNetv2_DINO/outputs/`) |
|---|---|
| `ssl/*_P4_dino_ssl_mobilenetv2_seed0` | `ssl/20261018-192425_A1_dino_ssl_mobilenetv2_seed0` |
| `ssl/*_P4_dino_ssl_mobilenetv2_split2_seed0` | `ssl/20261028-170414_A1_dino_ssl_mobilenetv2_split2_seed0` |
| `ssl/*_P4_dino_ssl_mobilenetv2_split3_seed0` | `ssl/20261028-190646_A1_dino_ssl_mobilenetv2_split3_seed0` |
| `finetune/*_P4_dino_mobilenetv2[_labeled_set_1pct or _labeled_set_10pct][_split<N>]_seed0/eval_test/metrics.json` | the A1 fine-tune folders with the same suffixes |

Same splits, same labeled sets, same fine-tuning recipe and seed, same test set; the only difference is the SSL
centering. Compare `overall_accuracy` (and `mean_per_class_accuracy`) per split and label budget, then the mean over
the three splits, as in `../Biweekly_nischal_ngc/11_2_2026/make_split_pdfs.py` (outside this repository). `compare_p4.py` does
this for the finished CRC runs (`python Experiments/mobileNetv2_DINO_P4/compare_p4.py`, CPU, seconds): test and
validation accuracy per split and budget with the P4 - A1 differences, the mean over the splits, and the SSL cost;
written to `outputs/compare/p4_vs_a1.md` and `.csv`. Expected SSL
cost: the same as A1 within noise (the per-cell centering adds one gather and one scatter-add per step).

## Known limitations

- The azimuth sector is relative to the target heading in ATRNet-STAR; `3field` is the deployable definition.
- One SSL seed per split (as for A1). Seed noise on this benchmark is about 1 point.
- `az_sector30` cells are small for the rare scenes (down to 17 images on split 1); their centers are updated only when
  the cell is present in a batch, from one to a few images at a time, so they move slowly and carry a trace of each
  image's own earlier outputs (weight 0.1 x 0.9^k after k later updates). This is the mechanism the pilot validated;
  the per-epoch `cells_seen` column and `cell_center_stats.csv` show whether every cell was reached.
- The scrambled control's permutation comes from numpy's generator with a fixed seed; the numpy version is recorded in
  `manifests.json` and the id vector's SHA-256 is checked on resume.
- The smoke test runs 48 images, so its cells are tiny; it checks the mechanics, not the effect.
- The pixel cache (`outputs/cache/`, 1.1 GB per pool) is rebuilt on each machine; it is git-ignored.
