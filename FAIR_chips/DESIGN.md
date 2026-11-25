# FAIR-CSAR chip study: SSL on 1 m Gaofen-3 slices, fine-tuned and tested on 22-class object chips

Status 2026-11-22 (night): built, reviewed by two agents (fixes applied), PC smoke passed on the sample PNGs (chips cut,
regions proposed for both pools, SSL smoke for dino / ro / ror on tiles and ro on down, resume, fine-tune from ImageNet and
from a smoke backbone). CRC 2026-11-22/23: chips + regions built; 9 tiles-arm SSL runs done (1.1 to 1.5 GPU-h each, 29,700 steps,
object-only fallback 1.8 %); 36 fine-tunes of imagenet + the three SARDet-pool backbones done (first results below);
the 9 tiles-arm fine-tunes done 2026-11-24; 500-epoch stage (15 SSL runs + teacher proposer + 45 fine-tunes) done
2026-11-25: 108 cells total. Down arm not launched.

First results (mean per-class accuracy on test_clean, 1 / 3 / 10 %, mean of 3 splits): no SSL 24.9 / 29.2 / 39.6
(overall 43.3 / 49.5 / 59.2): the test is hard, as wanted. SARDet-pool backbones transferred to FAIR: plain DINO
22.7 / 27.7 / 36.8 (hurts, -2.2 / -1.5 / -2.8 vs no SSL, all splits), object-only RegionDINO 26.6 / 31.9 / 40.9
(+3.9 / +4.2 / +4.1 vs that DINO, every split positive; +1.7 / +2.7 / +1.3 vs no SSL), its random control 25.8 /
29.2 / 40.7 (+3.1 / +1.5 / +3.9 vs DINO). So the cross-dataset pattern of ATRNet repeats: plain DINO on SARDet
scenes transfers worse than ImageNet, the object-only recipe repairs it and lands a little above ImageNet; the
placement itself adds about 1 point over random windows here (within noise except at 3 %). Protocol decision in
`datasets/CANDIDATES.md` (2026-11-22), inventory in `datasets/outputs/fair_csar_inventory.md`.

Results 2026-11-24, SSL on FAIR-CSAR's own tiles (SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR; mean
per-class accuracy on test_clean, 1 / 3 / 10 %, mean of 3 splits): DINO 26.2 / 30.5 / 43.3 (+1.3 / +1.4 / +3.7 vs no
SSL; at 10 % every split positive, +2.4 / +2.8 / +5.9; overall accuracy +2.9 / +3.1 / +3.6); RegionDINO object-only
27.9 / 33.4 / 42.4 (+1.7 / +2.9 / -0.9 vs that DINO); its random-window control 27.5 / 33.8 / 42.2 (+1.2 / +3.3 / -1.1
vs DINO). Object-only minus its control: +0.4 / -0.4 / +0.2: the placement adds nothing when SSL runs on FAIR itself.
Readings: (a) SSL on the dataset's own scenes helps, and the gain grows with the label budget (+3.7 at 10 %), the
first positive same-dataset SSL result on a hard chip test; (b) same-dataset beats cross-dataset SSL by a lot for
plain DINO (43.3 vs 36.8 at 10 %, 6.5 points), but RegionDINO object-only on SARDet closes most of that gap (40.9),
so the object-only recipe is what makes a cross-dataset SSL pool usable; (c) with SSL on FAIR, the small-window
recipe (object-only and control alike) beats standard DINO crops at 1 and 3 % and loses about 1 point at 10 %, same
crop-size story as SARDet detection; (d) the FAIR SSL runs are under-converged (29,700 steps, DINO loss 4.6 vs 0.74
on SARDet), so the FAIR-pool numbers are a floor. Aircraft types stay at 22 to 24 % for every method at 1 %.
Next: one matched-steps DINO run (about 500 epochs) on the tiles to see whether the +3.7 grows; the down arm is
worth running only for plain DINO (resolution question), since the region rows do not separate here.

Results 2026-11-25, the 500-epoch stage (148,500 steps; mean per-class accuracy on test_clean, 1 / 3 / 10 %, mean
of 3 splits; no SSL 24.9 / 29.2 / 39.6): DINO 27.3 / 32.9 / 43.6 (+2.4 / +3.8 / +4.0 vs no SSL; 3 and 10 % positive
on every split; overall accuracy +4.6 / +5.6 / +3.7); RegionDINO object-only 28.6 / 34.0 / 43.7 (+1.3 / +1.1 / +0.1
vs that DINO); its random-window control 28.8 / 32.9 / 41.5 (+1.5 / 0.0 / -2.1 vs DINO; object-only minus control
-0.2 / +1.1 / +2.2); teacher (SARMAE) regions 28.5 / 33.8 / 43.7 (-0.1 / -0.2 / 0.0 vs the brightness rule; best
overall accuracy of every row, 49.4 / 56.4 / 64.5, by about 1 point); teacher-size random control 25.0 / 31.4 / 40.8
(-3.5 / -2.4 / -2.9 vs the teacher row, 8 of 9 pairs negative). Convergence: DINO loss 4.6 -> 2.6 (teacher entropy
2.8 -> 0.74), object-only 2.55 -> 1.67 (0.96 -> 0.36). Teacher proposer on the 19,036 tiles: 20 regions per tile,
extent median 18 px (95 % under 48), and on 300 annotated tiles it reaches 41.5 % of the objects against 33.8 % for
the brightness rule (30 regions per tile, median 8 px).
Readings: (a) with matched steps, SSL on the dataset's own tiles is worth +2 to +4 points at every budget, not only
at 10 %; (b) placement now shows a little: both region rows beat their size-matched random controls at 10 % (+2.2
brightness rule, +2.9 teacher) and the teacher row beats its control at every budget, but object-only vs its control
is inside split noise at 1 and 3 %; (c) the teacher finds more objects than the brightness rule on 1 m imagery (41 vs
34 %) and still does not classify better: the fourth test where a better proposer buys nothing in mean per-class
accuracy (c11); (d) same-dataset still beats cross-dataset SSL by 6.8 points for plain DINO at 10 % (43.6 vs 36.8).
The 500-epoch rows are the FAIR headline from here; the 100-epoch rows stay as the short-schedule reference.

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
holds at least one object. trainval: 105 scenes, 11,959 slices, 69,298 annotations; test: 26 scenes, 2,706 slices,
15,987 annotations; all 22 classes in both.

**The slices overlap 50 % in both directions** (range offsets step 512 px, azimuth offsets step 1,897 SLC rows =
512 PNG px; a few slices sit off the grid). So one physical object is annotated in up to four slices (seen first in
the chip previews: the same ship four times). `make_manifests.py` maps every annotation into its scene's frame
(METAXmls patch offsets + slice position), merges annotations of the same scene and class within 4 px into one
object, and picks the OWNER copy (farthest from its slice border). Result: 85,285 annotations are 25,703 physical
objects (20,940 trainval, 4,763 test; 16,737 have 4 copies, 7,244 two, 1,237 one). Everything labeled counts
objects: one chip per object, cut from the owner slice (`data/objects.csv`).

Fixed subsets (`make_manifests.py`, committed under `data/`, SHA-256 in `hashes.json`):

| set | what | size |
|---|---|---|
| select | 10 % of each trainval scene's slices, seed 0; per-epoch model selection only | 1,190 slices, 1,950 owned objects |
| ssl_pool | trainval minus select; the unlabeled SSL pool, the same for every split | 10,769 slices, 18,990 owned objects |
| ssl_tiles | the pool cut 2 x 2 into 512-px tiles, one tile per scene position (origins within 128 px = duplicate) | 19,036 unique tiles of 43,076 |
| labeled 1 / 3 / 10 % (split 1, 2, 3) | slices drawn inside ssl_pool, seed = split - 1, nested; unit = slice (every object a chosen slice owns is labeled); classes under 5 objects topped up with the next slices owning them | 121 to 136 / 286 to 320 / 1,057 to 1,105 slices; 270 to 313 / 570 to 580 / 1,911 to 1,916 chips |
| test | every object owned by a test slice (the dataset's own split) | 2,706 slices, 4,763 chips |
| test_clean | test chips whose window holds no other class's box centre | subset of test |

The 1 % budget is small: 190 objects by the fraction, 270 to 313 after the per-class floor of 5, so the smallest
classes have exactly 5 training chips there (a few-shot regime; 3 % gives about 15, 10 % about 50 for the rarest
class). Keep 1 / 3 / 10 % for consistency with the other chip tests and read the 1 % row as such.

Leakage notes. The SSL pool never contains a selection slice or a test slice (the fine-tune checks this and stops),
and an object has one chip, so no chip is in two sets. Because the slices overlap, the pixels of a selection
object also appear, unlabeled, in neighbouring pool slices: SSL may have seen them, which is not label leakage but
makes the selection set less independent of SSL than SARDet's official val was; the test split's scenes are
separate acquisitions, so test pixels never reach SSL. The dataset's split separates acquisitions, not places: 14
scenes are the same harbour (E139.7 N35.5) on different dates, 12 in trainval and 2 in test, so a model can learn
the place. That is the benchmark's own split; the paper states it.

## 3. Chips

The SARDet_chips rule unchanged: square window centred on the oriented box's centre, side = max(64, 1.5 x longer
box side) in native pixels, edge-padded at the slice border, resized to 128 x 128, 8-bit grey. One chip per
physical object (the owner annotation, `data/objects.csv`), 25,703 chips in one array (`make_chips.py`, 0.4 GB). Flags per chip: another object's
centre inside the window, another class inside, clipped at the border, the dataset's difficulty level (0 for 89 %
of annotations, 1 for 7 %, 2 for 4 %; the paper's meaning of level 2 is still to be looked up; nothing in training
or scoring reads it). Bridges have a
long tail (95th percentile 850 px): their windows are up to 1,275 px, mostly padding; the flag table in
`summary.md` shows how many.

## 4. SSL and fine-tune protocol

SSL (RegionSSL engine, `train_ssl.py`, one entry for three method configs x two pools):

| row | method | views | pool |
|---|---|---|---|
| fair_dino_{tiles,down} | regiondino_scene (plain DINO, this engine's scene recipe: global 224 / local 96, zoom 0.75 to 1.33) | whole image | fair_tiles / fair_down |
| fair_ro_{tiles,down} | regiondino_object (object-only RegionDINO: 3 regions x (2 global 96 + 4 local 64), no whole-image view) | brightness-rule regions (`make_regions.py --pool ...`) | same |
| fair_ror_{tiles,down} | the same with region_source random | random windows, sizes matched to the regions | same |
| fair_{dino,ro,ror}_tiles500 | the three above with `"overrides": {"epochs": 500}` (148,500 steps = the SARDet-100K runs' count) | as above | fair_tiles |
| fair_rot_tiles500 | regiondino_object, 500 epochs, region file from frozen SARMAE's attention (`TeacherRegions/teacher_saliency.py --pool fair_tiles`, folder named by the config's `regions_dir`) | teacher regions | fair_tiles |
| fair_rotr_tiles500 | the same with region_source random | random windows, sizes matched to the teacher regions | fair_tiles |

Pools (RegionSSL/rssl/pools.py): `fair_tiles` = the unique 512-px tiles of the ssl_pool slices at native 1 m
(`data/ssl_tiles.csv`, 19,036 images per epoch; the 50 % slice overlap would otherwise make three of four tiles
exact repeats); `fair_down` = every ssl_pool slice resized 1024 -> 512 (10,769 images per epoch, 2 m per pixel;
overlapping slices kept, since no two are the same crop). Everything shared (100 epochs, batch 64, seed = split - 1,
view recipe) comes from SARDet/shared/ssl_common.json as for every other SSL run. Caveat (reviewer, 2026-11-22):
at equal epochs the two arms differ in more than resolution. Tiles: 19,036 de-duplicated images, 297 steps per
epoch, 29,700 steps in all, each ground point seen about once per epoch at 1 m. Down: 10,769 overlapping slices, 168
steps per epoch, 16,800 steps, each ground point seen about four times per epoch at 2 m (in four slice contexts), and
the 10-epoch warm-up is shorter in steps. So "tiles beats down" could be step count and "down beats tiles" could be
the 4 x redundancy. The arms are therefore read as "the practical recipe at each resolution"; the cost record of
every run lists images_seen and optimizer_steps. If the tiles result is worth the question, ONE matched-steps row is
added for plain DINO: a fixed subset of 10,769 of the tiles (seed 0, the same subset for every split), which gives the
down arm's images per epoch and steps at 1 m; it costs about 57 % of a tiles run and is not built until then.
Decision 2026-11-24 (Nischal): the 100-epoch tiles runs were far less converged than the SARDet-100K runs (29,700 vs
147k steps; DINO loss 4.6 vs 0.74), so all three tiles methods are rerun at 500 epochs (the `_tiles500` configs; the
engine records the deviation under `_overrides` in the run's config.json), plus the teacher-region method and its
teacher-size control at the same 500 epochs. Cost: DINO about 7.5 GPU-h per split, object-only about 6; 15 SSL runs
in all, about 95 GPU-h with the proposer job. The 100-epoch runs stay in the table as the short-schedule reference. The three splits are three SSL seeds on the same pool, paired with the three labeled draws
(the suite's rule). Regions are proposed once per pool (CPU job) with the same brightness rule as SARDet.

Fine-tune (`finetune_chips.py`): the SARDet_chips / ATRNet recipe unchanged (128-px chips resized to 224, AdamW
5e-4, cosine, 30 epochs, batch 64, selection by mean per-class accuracy on the select chips), 22-class head. Methods:
imagenet, the six FAIR rows, and three SARDet-pool backbones (dino, regiondino_o, regiondino_o_random, found by
SARDet_to_ATR/find_backbone.py). Headline metric: mean per-class accuracy on test_clean; overall accuracy and the
full test set are stored; accuracy per major category (aircraft / ship / bridge / tank / tower crane) and per class
are written per run. 10 methods x 3 budgets x 3 splits = 90 fine-tunes, in 30 jobs of 3 budgets.

Expected effect sizes: the split-to-split spread on SARDet chips was 2 to 4 points of mean per-class accuracy at
1 %; with 22 classes and 5 to about 60 chips per class at 1 % it will be larger here. Three splits are the minimum;
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
5. 500-epoch stage (2026-11-24): `crc/teacher_regions.sbatch` (GPU, once, about 1 h) and the 9 chains
   `crc/ssl.sbatch config_ssl_{dino,ro,ror}_tiles500.json {1,2,3}` at once; the 6 chains
   `config_ssl_{rot,rotr}_tiles500.json {1,2,3}` after the proposer job; then the 15 fine-tune jobs
   `crc/finetune.sbatch fair_{dino,ro,ror,rot,rotr}_tiles500 {1,2,3}` and `compare.py`.

## 6. Where things are

```
FAIR_chips/
  fair_paths.py        env FAIR_CSAR_ROOT / FAIR_CHIPS_ROOT, classes (22, fixed order), chip rule, manifest paths
  fair_annotations.py  DOTA text parser (oriented box -> centre, long/short side, extent), scene id from the slice name
  make_manifests.py    data/: objects.csv (de-duplicated annotations with their owner copy), slices.csv, select.csv,
                       ssl_pool.csv, ssl_tiles.csv (unique tiles, TILE_DUP_PX 128), split{1,2,3}/labeled_*.csv + counts.json, hashes.json
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
