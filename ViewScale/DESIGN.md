# ViewScale: the SSL window size against the object size

Status 2026-11-22: 15 SSL runs, 45 ATRNet transfer fine-tunes (3 splits) and 15 detection fine-tunes (split 1) done on CRC;
results in `outputs/analysis/` (curve.png, summary.md, from `python ViewScale/analyze.py`) and in the tracker (conclusion c10).
Headline: transfer gain over plain DINO at 1 % rises +0.6 / +2.2 / +3.7 / +4.9 / +4.3 for e32 ... e512; the region-placed
row is +9.4, its random control +3.2. Detection split 1 at 1 %: +0.5 / +1.6 / +1.9 / +2.1 / +1.6 mAP points (regions +1.6).
Open: detection splits 2 and 3; the optional constant-resolution point. The e32 anchor did NOT reproduce the random-window
control (+0.6 vs +3.2): the control's size mixture (tail of larger windows) carries part of its gain, see section 2.

## 1. The question

On SARDet detection, RegionDINO object-only gains +1.6 / +1.4 / +1.2 mAP over no SSL and its random-window control
gains +1.8 / +1.6 / +1.0 (tracker, conclusion c08). On the ATRNet chip transfer the placement of the windows carried
two thirds of a +9.4 gain (c04). So two things are mixed in the object-only recipe: the windows are small, and the
windows sit on objects. This study separates the first factor and measures it as a curve: **how does the gain
change with the size of the SSL training windows?**

## 2. One factor only

Every run is the object-only recipe exactly as in `RegionSSL/RegionDINO_object/config_ssl_random.json` (18 views
per image: 3 windows x (2 object-global views at 96 px + 4 object-local views at 64 px), DINO loss on those views
only, 100 epochs, batch 64, same optimiser, same augmentation), with one change: the 3 windows per image are random
squares of a **fixed extent** instead of windows with sizes drawn from the proposed regions.

| run | window extent e (px in the 512-px frame) | object-global window (e x 1.4 to 2.2, floor 48) | object-local window (e x 0.5 to 1.0, floor 32) |
|---|---|---|---|
| viewscale_e32 | 32 | 48 to 70 | 32 |
| viewscale_e64 | 64 | 90 to 141 | 32 to 64 |
| viewscale_e128 | 128 | 179 to 282 (plain DINO's global range is 168 to 298) | 64 to 128 (plain DINO's local range is 72 to 128) |
| viewscale_e256 | 256 | 358 to 512 | 128 to 256 |
| viewscale_e512 | 512 | 512 (the whole scene) | 256 to 512 |
| reference: object-only (exists) | region extents: median 10 px, 75 % below 21, 95 % below 70 | mostly the 48-px floor to about 150 | 32 to 70 |
| reference: random windows (exists) | same sizes as above, random places | same | same |

e128 is the plain-DINO window scale with the object-only view counts and sizes, so the two references plus the five
points span "object scale" to "whole scene" with everything else fixed. The floors come from `max_magnification`
(a window is never smaller than view size / 2) and are part of the recipe being measured, so they stay.

**What also changes with the extent, and must be said (review 2026-11-20).** The view sizes are fixed at 96 / 64 px,
so the resolution the network sees falls as the window grows: e32 is upsampled 1.4 to 2 x, e64 about 1 x, e128 is
shown at 0.3 to 0.5 x (plain DINO shows its 168 to 298-px windows at 0.75 to 1.33 x), e512 at 0.19 x (a 10-px ship
becomes 2 px). So the sweep varies "window extent at a fixed view size", which bundles extent and magnification; it
is the one-factor sweep of the object-only recipe, not a sweep of plain DINO's crop size. If the curve is interesting,
one constant-resolution point (e128 with 224 / 96 views, the plain-DINO view sizes) separates the two factors; it is
listed as optional below. Other differences from the two reference rows: square windows instead of region aspect
ratios; one extent instead of the region-size distribution; the object-local views are centred anywhere inside an
e x e box instead of inside a region; at e256 / e512 the two global views are near-identical crops, so the teacher
pair is augmentation only. Identical: 18 views per image, 96 / 64 px, batch 64, 100 epochs and steps, seed = split - 1,
optimiser and schedule (checked against region_engine.py and ssl_common.json).

End points are clamped by the recipe's floors and the frame (measured in the review): at e32 the object-local
window floors to 32 px, so every local view is the whole 32-px window at zoom 1; at e512 the global window is always
the full 512 frame. So e32 and e512 vary less inside a run than e64 to e256 do. Read the curve with that in mind.

Sanity anchor: e32 (global windows 48 to 70 px, local 32) nearly reproduces the window sizes of the existing
random-window control, so it should land near that row (+3.2 / +3.6 / +0.5 on transfer, +1.8 / +1.6 / +1.0 on
detection). If it does not, something other than the window extent differs and the sweep is not trustworthy.

Object sizes to put on the x-axis: the proposed-region extents above (SARDet pool, 512-px frame), and the
annotation boxes of SARDet (longer side, native px, medians: tank 21, bridge 17, ship 37, aircraft 79, harbor 80;
see SARDet_chips/DESIGN.md). A window of 64 px is about 1.7 ship lengths; 512 px is the scene.

## 3. What is measured

For each window size and split (3 splits, SSL seed = split - 1, as everywhere):
1. **Transfer to ATRNet chips** (the cross-dataset classification test, 1 / 3 / 10 % labels): through
   `SARDet_to_ATR/crc.sbatch viewscale_e<k> <split>`; run names `XA_viewscale_e<k>_sardetssl_mobilenetv2...`.
   Cheap (25 min per job). Paired against plain XA DINO and against the object-only row.
2. **SARDet detection** (1 / 3 / 10 % labels): through `SARDet/crc/finetune.sbatch ../ViewScale <split> auto viewscale_e<k>_ssl`
   (the folder has a `config_finetune.json` stub the job reads; the run name comes from the MATCH argument).
   Expensive (about 12 h per job). Paired against the no-SSL baseline and the object-only row.
3. **Frozen response maps** on ATRNet chips and SARDet scenes (the existing map tools take a `--model LABEL=PATH`),
   to see where the attention sits as the window grows.

Plot: gain (points) against window extent on a log axis, one line per budget, the two references as marked points.

## 4. Cost and order

| step | runs | time |
|---|---|---|
| SSL | 5 sizes x 3 splits = 15 | about 5 GPU-h each (same views and steps as object-only) = 75 GPU-h |
| ATRNet transfer fine-tunes | 15 jobs x 3 budgets | 25 min each |
| detection fine-tunes | 15 jobs x 3 budgets | about 12 h each = 180 GPU-h |
| response maps | 2 jobs | 1 h |
| optional: constant-resolution point (e128 with 224 / 96 views) | 3 SSL runs | 15 GPU-h |

Decision for Nischal: run detection for split 1 only at first (5 jobs, one night) and extend to splits 2 and 3 if
the curve is not flat. The transfer and map parts run for all three splits from the start.

Order: 1) smoke on the PC (done 2026-11-20, GPU: "regions: fixed", backbone exported), 2) 15 SSL jobs on CRC through the
suite's chain script, exactly as the RegionDINO runs were launched (from the suite root):
`sbatch --job-name=vs_e64_s1 crc/ssl.sbatch ../ViewScale 1 config_ssl_e64.json` (the regions file is loaded but not used
for "fixed"; it must exist, it does), 3) ATRNet transfer + maps as the SSL jobs finish, 4) detection split 1, 5) tables and the curve.

## 5. What changes elsewhere (all additive)

- `RegionSSL/rssl/region_data.py`: `random_box` returns a fixed square when `region_source == "fixed"`; `choose_boxes`
  treats "fixed" like "random". `RegionSSL/rssl/regiondino.py`: accepts "fixed", requires `fixed_extent_px`, passes
  it to the dataset. Existing configs do not set it, so existing runs and resumes are unaffected.
- `SARDet_to_ATR/find_backbone.py` and `finetune_atr.py`: five method names `viewscale_e32 ... e512` pointing at this folder.
- `SARDet/compare/compare.py`: `ViewScale` added to the folders with several SSL configs (one line).
- `tracker/registry.py`: rows for the five sizes once results exist.

## 6. What would make this defensible in a paper

- Same number of optimizer steps, views per image and view sizes at every point; only the window extent moves.
- Three splits for the cheap measurements; the expensive one extended only if the curve says so.
- The two existing rows (object-only, random windows) as reference points on the same axes, not re-run.
- Report the object-size distribution next to the curve, so "window over object size" is a real ratio.
