# SARDet-100K chip classification: SSL on scenes (or on blind tiles), tested on object chips of the same dataset

Status 2026-11-19: protocol proposed, nothing built yet. Numbers below come from a dry run over the SARDet-100K
annotations (scratch script, no files written).

## 1. The two questions

Both studies fine-tune and test on **object chips cut from SARDet-100K with its own detection labels** (six classes:
ship, aircraft, car, tank, bridge, harbor). Only the unlabeled SSL pool changes.

| study | unlabeled SSL pool | what it asks |
|---|---|---|
| S1 scenes | the existing SARDet-100K scene pool (94,034 full scenes, shorter side 512) | On its own dataset, does SSL on whole scenes help chip classification, and does RegionDINO's object view beat plain DINO when the test is a chip? |
| S2 tiles | the same scenes cut blindly into 2 x 2 tiles (no object knowledge; 4 x more images, a quarter of the field of view each) | Is "smaller images" alone enough? If tiles close the gap, RegionDINO's gain was about field of view; if not, it is about where the views are placed. |

Rows per study: ImageNet start (no SSL) as the baseline, then DINO, LeJEPA, SparK, RegionDINO scene+object,
RegionDINO object-only, and the two random-window controls (always run next to a RegionDINO row). S1 needs no new
SSL: the scene backbones from SARDet/<method>_SARDet/outputs/ssl and RegionSSL/*/outputs/ssl (3 splits each) are reused.
S2 needs new SSL runs on the tile pool.

Why this is the right test for the paper: the cross-dataset result (SSL on SARDet scenes, test on ATRNet chips) says
object-only RegionDINO transfers best; the in-domain ATRNet result says it does not help when the SSL images are
already chips. This study removes the dataset change: same sensor mix, same scenes, and the test is still "find the
object in a chip". A gain here cannot be blamed on ATRNet being a different dataset.

## 2. Chips: the rule and why

**Rule.** For every annotated box: a square window centred on the box, side = max(64 px, 1.5 x longer box side)
in native pixels, clipped at the image border (padding with the border value when the window leaves the image),
resized to 128 x 128, saved as an 8-bit grayscale PNG. The classifier then treats it exactly like an ATRNet chip
(128 native, resized to 224, same augmentation and hyperparameters).

**Why box-relative and not a fixed window.** A dry run over the 24,203 test boxes:

| rule | chip holds another object's centre | ... of another class | object does not fit | upscaled > 4x to 224 |
|---|---|---|---|---|
| relative, side = max(64, 1.5 x box) **(chosen)** | 21.7 % | 0.5 % | 0.0 % | 0.0 % |
| relative, side = max(32, 1.5 x box) | 15.6 % | 0.1 % | 0.0 % | 43.8 % |
| relative, side = max(64, 2.0 x box) | 27.2 % | 0.5 % | 0.0 % | 0.0 % |
| fixed 96 px | 24.8 % | 1.1 % | 10.7 % | 0.0 % |
| fixed 128 px | 30.3 % | 1.8 % | 3.5 % | 0.0 % |
| fixed 192 px | 43.0 % | 2.8 % | 0.6 % | 0.0 % |

- A fixed window cannot serve both ends of the size range: boxes run from 5 px (tanks, bridges) to 190 px (harbors).
  At 96 px 27 % of aircraft and 40 % of harbors do not fit; at 192 px tank chips contain another class 25 % of the time.
- A box-relative window is how chip datasets are framed anyway (ATRNet-STAR, MSTAR: target centred, fixed
  object-to-chip ratio), so the test matches the chip tests the rest of the paper uses, and the "find the object"
  task is the same for every class.
- The 64-px floor stops the tiny-object chips from being blown up (a 21-px tank would otherwise be upscaled 10x)
  and gives small objects real context; it caps the upscale at 3.5x.
- What is lost: absolute size as a class cue (a 17-px bridge and a 130-px harbor both fill the chip). The native
  window side is stored in the manifest, so a size-aware variant can be tested later if wanted.

**Neighbours in the chip.** Tanks sit in rows (OGSOD, MSAR tank parks): 90 % of tank chips contain another tank's
centre whatever the rule. That does not corrupt the label (same class). The risky case is another *class* inside
the chip: 0.5 % overall with the chosen rule (tank 3.3 %, harbor 6.5 %). Every chip carries the flags
`n_others_inside` and `other_class_inside`; the **clean test set** excludes other-class chips, and the test set
also excludes chips from the 50 test images whose content also appears in train (the suite's test_clean list).
Both test numbers are reported; the clean one is the headline.

## 3. Data flow and leakage

Everything inherits from the detection suite's audited manifests (SARDet/data/): a chip belongs to the split of
the image it is cut from, so there is nothing new to leak.

| set | images | chips (about) |
|---|---|---|
| SSL pool S1 | ssl_pool.csv: 94,034 train images (459 train images that share content with val/test already removed) | scenes, not chips |
| SSL pool S2 | the same 94,034 images, 4 tiles each | 376,000 tiles |
| labeled 1 / 3 / 10 % | the suite's split{1,2,3}/labeled_{1,3,10}pct.csv image lists | 2,500 / 7,500 / 25,000 chips per split |
| selection | the suite's val_select.csv (2,000 val images) | 5,000 chips |
| test | the 11,613 official test images | 24,200 chips |
| test_clean | test minus the 50 straddling images, minus other-class chips | about 24,000 chips |

Chips are cut once for every annotated image (train + val + test, about 245,000 chips, 4 GB of PNG) under
`SARDET_CHIPS_ROOT` (environment variable, default next to the dataset), layout `<official split>/<class>/<chip id>.png`,
with one `chips.csv` manifest (chip id, image file, official split, source, class, box, window side, flags, group)
whose SHA-256 is recorded in every run. On CRC the chip folder is staged to node disk from one tar, like the ATRNet chips.

Class balance is poor (test: ship 10,741, aircraft 6,779, bridge 3,281, tank 1,773, car 1,230, harbor 399) and
each class comes from only one to five source datasets (car only from SIVED, harbor only from OGSOD). Two
consequences, both reported rather than fixed: the headline metric is **mean per-class accuracy** with overall
accuracy next to it, and a per-source table shows where each method wins. The source-style shortcut is shared by
every method, so paired differences stay fair.

## 4. Fine-tuning and scoring

Same recipe as every ATRNet fine-tune (30 epochs, AdamW 5e-4, cosine, batch 64, input 224, horizontal flip and
8-px translate), head = linear layer on the 1280-dim pooled feature, all layers trained. Selection: epoch with the
best mean per-class accuracy on the selection chips. Test scored once with the selected epoch. Three splits
(labeled draw and SSL seed differ), fine-tune seed 0, as everywhere else. Paired differences: method minus the
no-SSL row, and RegionDINO minus plain DINO, on the same split and budget.

Budget per study: 8 rows x 3 budgets x 3 splits = 72 fine-tunes, each a few minutes on an A100 (2,500 to 25,000
training chips, 24,000 test chips).

## 5. The tile pool (S2)

A tile is one quarter of the scene as the SSL pipeline already sees it: the scene resized to shorter side 512
(what every scene SSL run used), cut into a 2 x 2 grid of 256 x 256 tiles (non-square images get an extra column
or row with the last tile shifted to fit). Stored as PNG; 94,034 scenes give about 376,000 tiles. The SSL loader
takes a tile as the whole image with no further resize, so views have the same pixel scale as in the scene run and
simply cannot extend past the tile. No object knowledge enters: every scene gives the same four tiles.

Compute rule: **equal optimizer steps**. The scene runs did 100 epochs x 1,469 steps = 146,900 steps (23 GPU-h for
DINO); the tile runs do the same number of steps, which is 25 epochs over the 4x larger pool. Same views, same
schedule, same batch; only the images are smaller. RegionDINO on tiles needs regions proposed per tile (one CPU job
with the existing proposer, as for the scene and ATRNet pools).

S2 cost: 8 SSL runs (DINO, LeJEPA, SparK, RegionDINO so, so-random, o, o-random; the baseline needs none) x 3 splits
at roughly 15 to 25 GPU-h each, about 450 GPU-h in total, plus 72 short fine-tunes.

## 6. What gets built (folder SARDet_chips/, self-contained like SARDet_to_ATR/)

| file | job |
|---|---|
| `paths.py` | SARDET_CHIPS_ROOT (env var, PC default), tile root, manifests; nothing hard-coded elsewhere |
| `make_chips.py` | cuts every chip, writes chips.csv and a per-class / per-source summary; `--dry-run` prints the table above |
| `make_tiles.py` | cuts the 2 x 2 tiles into the tile root, writes tiles.csv (the S2 SSL pool manifest) |
| `data.py` | chip manifest -> (chip, label) records for a labeled set / selection / test / test_clean |
| `finetune_chips.py` | the ATRNet fine-tune recipe on chip records (reuses the model and training loop code; own data loading) |
| `compare.py` | tables: accuracy per method x budget x split, paired differences, per-source, per-class |
| `crc/` | finetune.sbatch (3 budgets per job, chips staged from one tar), tile-pool SSL jobs (one per method and split) |
| SARDet suite / RegionSSL | a `tiles` pool (manifest + loader) so the existing SSL code runs on tiles unchanged |

Nothing under SARDet/ or RegionSSL/ changes except the added pool definition; nothing under Experiments/ changes.

## 7. Decisions taken here (say if any should change)

1. Chip rule: square, side = max(64, 1.5 x longer box side), native pixels, resized to 128.
2. Test = all test-image chips; clean test additionally drops other-class chips and the 50 straddling images.
   Headline metric = mean per-class accuracy; overall accuracy reported next to it.
3. Labeled sets = chips of the suite's existing 1 / 3 / 10 % image lists (no new draw); selection on the
   val_select chips.
4. Tiles = 2 x 2 quarters of the 512-resized scene (256 px), equal optimizer steps to the scene runs.
5. Both RegionDINO modes and both random-window controls run in both studies.
