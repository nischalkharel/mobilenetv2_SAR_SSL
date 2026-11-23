# FAIR-CSAR chip study: SSL on 1 m Gaofen-3 slices, fine-tuned and tested on 22-class object chips

Status 2026-11-22: built on the PC (manifests drawn and committed, code written, PC smoke pending the sample PNGs
from CRC), nothing run on CRC yet. Reviewer pass before the first cluster launch. Protocol decision in
`datasets/CANDIDATES.md` (2026-11-22), inventory in `datasets/outputs/fair_csar_inventory.md`.

## 1. Why this dataset and what it asks

The SARDet-100K chip test (SARDet_chips, S1) saturated: six classes, no SSL already at 92 % overall accuracy with 1 %
labels, every SSL method within the split noise (conclusion c09). The ATRNet transfer is the only test where the
RegionDINO recipe shows a large gain, and it is one dataset pair. FAIR-CSAR-V1.0 SL is a second scenes-in, chips-out
pair and is harder: 1 m spotlight slices, 22 fine-grained classes (11 aircraft types, 7 ship types, bridges, tanks,
tower cranes), objects of 18 to 158 px (5th to 95th percentile), strong class imbalance, and a published oriented
detector at only 39 mAP@0.5. SARMAE never trained on it (clean big-model reference); SARATR-X-v2 did (not used).

Questions, in the order the rows answer them:
1. On its own 1 m scenes, does SSL (plain DINO) help the compact model over the ImageNet start? (fair_dino vs imagenet)
2. Does placing the SSL views on label-free bright regions (object-only RegionDINO) beat plain DINO here, as it did
   on the ATRNet transfer (+9.4) but not on SARDet chips (null)? (fair_ro vs fair_dino, with fair_ror as the
   random-window control: always run next to a RegionDINO row)
3. Resolution: does SSL at the native 1 m (2 x 2 tiles) beat SSL on the slice downsampled to 2 m? The sponsors'
   question of 2026-11-13. (the tiles arm vs the down arm, same rows)
4. Cross-dataset: do the existing SARDet-pool backbones (another sensor mix, 0.5 to 3 m) transfer to FAIR chips, and
   does the object-only one transfer best, as on ATRNet? (sardet_* rows; no new SSL)

## 2. Data, subsets, leakage

Source: `FAIR_CSAR_ROOT/{trainval,test}/{PNGImages,Annotations,METAXmls,KeyPoints}` (the SL archives extracted
without the complex .mat files; 1024 x 1024 uint8 PNG at 1 m, one DOTA-style text file per slice). Every slice
holds at least one object (median 3, p95 21). trainval: 105 scenes, 11,959 slices, 69,298 objects; test: 26 scenes,
2,706 slices, 15,987 objects; all 22 classes in both.

Fixed subsets (`make_manifests.py`, committed under `data/`, SHA-256 in `hashes.json`):

| set | what | size |
|---|---|---|
| select | 10 % of each trainval scene's slices, seed 0; per-epoch model selection only | 1,190 slices, 6,728 chips |
| ssl_pool | trainval minus select; the unlabeled SSL pool, the same for every split | 10,769 slices, 62,570 chips |
| labeled 1 / 3 / 10 % (split 1, 2, 3) | slices drawn inside ssl_pool, seed = split - 1, nested; unit = slice (all chips of a chosen slice are labeled); classes under 5 chips topped up with the next slices containing them | about 115 / 320 / 1,060 slices, 660 to 770 / 1,880 / 6,260 chips |
| test | every test slice (the dataset's own split) | 2,706 slices, 15,987 chips |
| test_clean | test chips whose window holds no other class's box centre | subset of test |

Leakage notes. The SSL pool never contains a selection slice or a test slice (the fine-tune checks this and stops).
The dataset's split separates acquisitions, not places: 14 scenes are the same harbour (E139.7 N35.5) on different
dates, 12 in trainval and 2 in test, so a model can learn the place. That is the benchmark's own split; the paper
states it. The 1 % draw of split 2 overshoots its target by 150 chips because a many-object slice sat at the
boundary; the slice unit is kept and the counts are in `data/split2/counts.json`.

## 3. Chips

The SARDet_chips rule unchanged: square window centred on the oriented box's centre, side = max(64, 1.5 x longer
box side) in native pixels, edge-padded at the slice border, resized to 128 x 128, 8-bit grey. One chip per
annotated object, about 85,000 chips in one array (`make_chips.py`, 1.4 GB). Flags per chip: another object's
centre inside the window, another class inside, clipped at the border, the dataset's difficult flag. Bridges have a
long tail (95th percentile 850 px): their windows are up to 1,275 px, mostly padding; the flag table in
`summary.md` shows how many.

## 4. SSL and fine-tune protocol

SSL (RegionSSL engine, `train_ssl.py`, one entry for three method configs x two pools):

| row | method | views | pool |
|---|---|---|---|
| fair_dino_{tiles,down} | regiondino_scene (plain DINO, this engine's scene recipe: global 224 / local 96, zoom 0.75 to 1.33) | whole image | fair_tiles / fair_down |
| fair_ro_{tiles,down} | regiondino_object (object-only RegionDINO: 3 regions x (2 global 96 + 4 local 64), no whole-image view) | brightness-rule regions (`make_regions.py --pool ...`) | same |
| fair_ror_{tiles,down} | the same with region_source random | random windows, sizes matched to the regions | same |

Pools (RegionSSL/rssl/pools.py): `fair_tiles` = each ssl_pool slice cut 2 x 2 into 512-px tiles at native 1 m
(43,076 images per epoch); `fair_down` = the slice resized 1024 -> 512 (10,769 images per epoch, 2 m per pixel).
Everything shared (100 epochs, batch 64, seed = split - 1, view recipe) comes from SARDet/shared/ssl_common.json as
for every other SSL run, so the arms differ in images seen per epoch (4 x) as well as in resolution; the cost record
of each run states both. The three splits are three SSL seeds on the same pool, paired with the three labeled draws
(the suite's rule). Regions are proposed once per pool (CPU job) with the same brightness rule as SARDet.

Fine-tune (`finetune_chips.py`): the SARDet_chips / ATRNet recipe unchanged (128-px chips resized to 224, AdamW
5e-4, cosine, 30 epochs, batch 64, selection by mean per-class accuracy on the select chips), 22-class head. Methods:
imagenet, the six FAIR rows, and three SARDet-pool backbones (dino, regiondino_o, regiondino_o_random, found by
SARDet_to_ATR/find_backbone.py). Headline metric: mean per-class accuracy on test_clean; overall accuracy and the
full test set are stored; accuracy per major category (aircraft / ship / bridge / tank / tower crane) and per class
are written per run. 10 methods x 3 budgets x 3 splits = 90 fine-tunes, in 30 jobs of 3 budgets.

Expected effect sizes: the split-to-split spread on SARDet chips was 2 to 4 points of mean per-class accuracy at
1 %; with 22 classes and 5 to 150 chips per class at 1 % it will be larger here. Three splits are the minimum;
differences under about 3 points at 1 % are noise.

## 5. Order of work on CRC and what decides the next step

1. `crc/make_chips.sbatch` (CPU, once) and `crc/regions.sbatch fair_tiles` + `fair_down` (CPU, once); commit the
   region summary json files.
2. Tiles arm first: 9 SSL chains (`crc/ssl.sbatch config_ssl_{dino,ro,ror}_tiles.json {1,2,3}`), while the
   imagenet and sardet_* fine-tunes run (12 jobs; no SSL needed).
3. Fine-tunes of the tiles rows (9 jobs), then `compare.py`. Decide the down arm from the tiles result: run it in
   full if the tiles rows show any gain to compare resolution against; if everything is null, run only
   fair_dino_down to answer the resolution question for plain DINO.
4. Results go to the tracker (registry CHIPS-style rows, a chips page for FAIR) and to `datasets/CANDIDATES.md`.

## 6. Where things are

```
FAIR_chips/
  fair_paths.py        env FAIR_CSAR_ROOT / FAIR_CHIPS_ROOT, classes (22, fixed order), chip rule, manifest paths
  fair_annotations.py  DOTA text parser (oriented box -> centre, long/short side, extent), scene id from the slice name
  make_manifests.py    data/ (select, ssl_pool, labeled sets x 3 splits, counts, hashes)
  smoke_list.py        data/pc_smoke_pngs.txt: the PNGs a PC smoke check needs (fetched from CRC with tar -T)
  make_chips.py        chips array + csv + previews (FAIR_CHIPS_ROOT)
  chip_data.py         sets (train / select / test / test_clean) and the ChipDataset (from SARDet_chips)
  finetune_chips.py    one fine-tune; compare.py the table
  train_ssl.py         RegionSSL engine entry; config_ssl_{dino,ro,ror}_{tiles,down}.json; config_finetune.json
  crc/                 env.sh (staging), regions.sbatch, make_chips.sbatch, ssl.sbatch (chain), finetune.sbatch
  outputs/             ssl/, finetune/, compare/ (run records committed, weights not)
RegionSSL/rssl/pools.py        FAIRTilesPool, FAIRDownPool (additive)
SARDet_to_ATR/find_backbone.py fair_* entries (additive)
```
