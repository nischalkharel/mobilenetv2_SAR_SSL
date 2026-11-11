# SARDet: SSL pretraining comparison for a compact SAR object detector

The ATRNet-STAR study (`mobilenetv2_SAR_SSL`) compared a supervised baseline, DINO, LeJEPA, SparK and P4
(DINO with per-condition-cell teacher centering) for MobileNetV2 *classification* of 128 px vehicle chips.
This suite repeats that comparison as *object detection* on SARDet-100K (116,598 images, 6 classes, larger
scenes with context), with one shared detector, one shared fine-tuning protocol, audited splits and run
records. `DESIGN.md` is the full protocol (v2, after a red-team review); this file explains it in plain words
and says how to run things.

## Folder map

| Folder | What |
|---|---|
| `shared/` | The one package every experiment imports: `paths.py` (all locations, env-var overrides), `metadata.py` (per-image source / sensor / polarisation / parent from `mapping.json`), `manifests.py` (subset CSVs, hashes, quick leak audit), `coco_data.py` (detection data), `detector.py` (MobileNetV2 + FPN + Faster R-CNN), `finetune.py` (THE fine-tune + scoring protocol), `evaluate.py` (COCO metrics, per-class, per-source, PR curves), `ssl_data.py` (THE SSL view recipe), `ssl_engine.py` (THE SSL run loop: records, checkpoints, resume, export), `probes.py`, `visualize.py`, `plots.py`, `run_record.py`, `smoke_common.py`, `finetune_config.json`, `ssl_common.json` |
| `data/` | `make_metadata.py` -> `image_metadata.csv`; `make_splits.py` -> `ssl_pool.csv`, `split{1,2,3}/labeled_{1,3,10}pct.csv`, `val_select.csv`, `eval_subsets.json`, `hashes.json`; `audit_splits.py` -> `audit_report.json` (all committed) |
| `baseline_SARDet/` | ImageNet MobileNetV2 + detector, fine-tuned on the labeled subset only |
| `DINO_SARDet/` | DINO SSL on the unlabeled pool (ATRNet-STAR A1 code on the shared engine), then the same fine-tune |
| `LeJEPA_SARDet/` | LeJEPA SSL (A2 code), then the same fine-tune |
| `SparK_SARDet/` | SparK masked-reconstruction SSL (A3 code), then the same fine-tune |
| `CondDINO_SARDet/` | P4: DINO with the teacher center kept per condition cell; variants `source_pol` (default), `perm` (scrambled control), `oracle_class` (label-leak ceiling), `radiometric` (label-free, deployable cells); `cell_report.py` measures the class leak of every cell definition |
| `compare/compare.py` | Gathers every completed run into tables and charts (one command) |
| `crc/` | `env.sh` (CRC paths), `ssl.sbatch`, `finetune.sbatch`, `lr_check.sbatch`, `finetune_repeat.sbatch`, `README.md` |

Each experiment folder has `smoke_test.py` (end to end on a few images, PC), `train_ssl.py` + `config_ssl*.json`
(SSL stage; not in the baseline), `run_finetune.py` + `config_finetune.json` / `config.json` (fine-tune stage),
and writes to its own `outputs/` (git-ignored except the small result files the CRC workflow commits).

## The design in plain words

**Data.** SARDet-100K ships COCO boxes and anonymised file names. `mapping.json` (in the archive, extracted into
the dataset folder together with `test.json`) gives the original names, and those carry the source dataset
(Shipdataset, MSAR, SAR-AIRcraft, OGSOD, HRSID, SSDD, SIVED, SADD, AIR_SARShip), and for some sources the
satellite, polarisation and the parent image a patch was cut from. The paper's Table 2 adds sensor, band and
resolution per source. `data/image_metadata.csv` has all of it, plus two content hashes per image.

**Leakage.** The official release contains byte-identical copies across train/val/test (303 groups) and three
sources were sliced into overlapping patches. Every image belongs to a "group" (same parent patch image or
identical content). Groups touching official val or test are excluded from all training-side subsets, so
nothing we train on is a copy or a patch-neighbour of a scored image. HRSID's scene id is recorded but not
used for grouping (its 800 px tiles overlap at most 25 % and grouping by scene would remove the whole source;
`audit_report.json` counts the straddling scenes). Near-duplicates in sources without any parent information
(MSAR, OGSOD, SSDD, Shipdataset) cannot be detected; that is a property of the benchmark shared by all methods.

**Subsets.** SSL pool = official train minus the excluded groups (94,034 images; the same for all splits).
Labeled sets 1 % / 3 % / 10 % (962 / 2,841 / 9,460 on split 1) are drawn INSIDE the pool, nested, stratified
by source, in whole groups, with seed = split - 1 (as on ATRNet-STAR: the labeled set is inside the SSL pool;
SSL uses no labels). Model selection uses `val_select`, a fixed 2,000-image subset of official val; the chosen
checkpoint is scored once on the full official val, the official test (the headline number) and `test_clean`.
What the three splits vary: the labeled draw and the SSL seed. Comparisons between methods are therefore
PAIRED within a split, and `compare.py` reports per-split paired differences with their sign consistency,
not t-tests.

**Detector (same for every method).** MobileNetV2 `features` (ImageNet V2 weights for the baseline, an
exported SSL backbone otherwise) -> FPN on strides 4/8/16/32 (128 channels) + P6 -> torchvision Faster R-CNN,
anchors 8-256 px (after the resize only 2.7 % of boxes lack an anchor with IoU >= 0.5), input shorter side 512,
grayscale replicated to RGB with ImageNet normalisation. Fine-tune: AdamW lr 2e-4 (checked once with
`crc/lr_check.sbatch`), wd 0.05, 1-epoch warm-up + cosine, batch 8, epochs 60 / 40 / 24 for 1 / 3 / 10 %,
horizontal flip only, BatchNorm trained, fine-tune seed 0 with the head initialised and the loader seeded
identically for every backbone (the smoke test asserts identical heads). val_select mAP every 2 epochs;
`best.pt` and `last.pt` both scored on test so selection sensitivity is visible.

**SSL (equal epochs).** One view recipe for all methods (`shared/ssl_common.json`): the image is resized
exactly as the detector resizes it, views are crops at that scale with zoom jitter 0.75-1.33 (global 224 px,
local 96 px), flip, brightness/contrast jitter, blur. DINO: 2 global + 8 local; LeJEPA: 8 global; SparK: 1
global with a 60 % patch mask. `epochs` (100) is one number for all. Every run checkpoints each epoch, resumes
exactly, logs collapse diagnostics, GPU-hours and view-forwards, and exports the FINAL epoch (DINO/CondDINO:
teacher; LeJEPA, SparK: encoder) in one format (`backbone_epoch100.pth`, torchvision `features.*` keys).
Diagnostic recorded per run: the share of views containing at least one object (about 30 % of global and 7 %
of local views on this dataset; the SSL methods mostly see background).

**CondDINO (P4 here).** ATRNet-STAR had scene / band / depression / azimuth per chip. SARDet-100K has no
per-image acquisition field independent of the class: the source dataset is the condition bundle (sensor,
resolution, polarisation) and it is almost the class. So the method runs with the cells that exist
(`source_pol`: source refined by the per-image satellite x polarisation of Shipdataset and the sub-source of
SIVED) and the suite MEASURES the leak instead of hiding it: `cell_report.py` prints class-vs-cell purity and
mutual information; `perm` scrambles the assignment (size-matched control); `oracle_class` centers per
ground-truth class set (labels used; a ceiling, never a method); `radiometric` uses label-free, deployable
cells (k-means on an intensity histogram + size). Reading rule: if `source_pol` ~ `oracle_class` >> `perm`, the
gain is class information, not condition. Cells are used only inside the SSL loss; the detector never sees them.

**Metrics and analyses.** COCO mAP@[.5:.95], AP50, AP75, APs/m/l, per-class AP and AP50, per-source mAP (ship
across HRSID / SSDD / Shipdataset / AIR separates sensor from class), PR curves, saved detections; training
curves; SSL curves + collapse diagnostics; frozen-feature probes (class presence, source decodability,
Shipdataset satellite/polarisation decodability, kNN agreement, effective rank) of every exported backbone
and of the fine-tuned backbones; response-map + prediction panels per source. `compare.py` makes the tables.

## Running on the PC (setup, data preparation, smoke tests)

```
cd SARDet-work\SARDet
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install numpy pillow pycocotools matplotlib pandas psutil
.\.venv\Scripts\python.exe shared\paths.py                 # every path must say ok
.\.venv\Scripts\python.exe data\make_metadata.py           # ~7 min; already done and committed
.\.venv\Scripts\python.exe data\make_splits.py             # already done and committed
.\.venv\Scripts\python.exe data\audit_splits.py            # prints the leak numbers; must end with "ok"
.\.venv\Scripts\python.exe baseline_SARDet\smoke_test.py   # then DINO_SARDet, LeJEPA_SARDet, SparK_SARDet, CondDINO_SARDet
```
Smoke tests are pipeline checks on 8-48 images; nothing trained on the PC is a result. Full runs happen on CRC.

## Running on Pitt CRC

See `crc/README.md`: transfer, environment, the job scripts, and how results come back (small files committed
on CRC, pulled on the PC; `*.pt` / `*.pth` / `*.npz` never in git).

## Status (2026-11-11)

Built and smoke-tested on the PC (RTX 4070 SUPER): data preparation, audit, shared detector and fine-tune
protocol, DINO, LeJEPA, SparK, CondDINO, compare, CRC scripts. No full run has been started; the shared lr
check and the first SSL runs are the next CRC steps. See `DESIGN.md` for every decision and its reason.
