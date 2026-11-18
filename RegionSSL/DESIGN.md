# RegionDINO: teaching the backbone to learn objects, not scenes (design, 2026-11-17)

Status: proposed. Nothing here is implemented or run yet except the region proposer lab (`segment_lab.py`,
`box_zoom.py`).

## 1. The idea in plain words

DINO learns "two crops of the same picture should look the same to the network". On SARDet-100K scenes that
is mostly a lesson about backgrounds: only 29 % of DINO's global crops even contain an object, and the one
vector per crop mixes the ship with the water around it. The network becomes good at "which scene is this",
which is not what we need later.

What we want is the lesson a person gets from reading in several languages before learning a new one:
not the words themselves, but *where the words are* and *what a word is*. For SAR that means: this blob is a
thing, that is its surroundings, and a thing stays the same thing when the crop moves, flips or zooms. The
labeled stage on ATRNet-STAR (or on SARDet) then only has to say *which* thing it is.

RegionDINO keeps DINO as it is and adds three things, each using only the label-free region rule
(`felz_grouped`: Felzenszwalb superpixels brighter than the image median, nearby fragments grouped):

1. **Crops are placed on things.** Most global and local crops are centred on a proposed region instead of
   anywhere in the image, with the zoom chosen so the region fills a realistic share of the view. Every view
   now contains an object, and object-at-view-scale is exactly what an ATRNet-STAR chip looks like.
2. **Each thing gets its own vector.** Besides DINO's one vector per crop, the backbone's feature map is
   averaged inside each region (and once over the background), so a crop with a ship and a pier gives
   three vectors: ship, pier, rest. The regions are in image coordinates, so the same ship is known in both
   crops from the crop geometry alone.
3. **Student and teacher must agree thing by thing.** The DINO loss is applied to those region vectors too:
   the student's view of region k must match the teacher's view of region k. Background vectors are matched
   as well, with a lower weight, so the network still learns everything in the image, just separately.

Labels never enter. The boxes are used only by the lab scripts that *score* the region rule.

## 2. What is learned, frozen, saved

| part | in DINO | in RegionDINO |
|---|---|---|
| MobileNetV2 backbone (student) | learned | learned, same init (ImageNet) |
| teacher backbone | EMA of student | same |
| DINO head on the crop vector | learned | same |
| DINO head on region vectors | none | the same head, shared (no new parameters) |
| region masks | none | computed once, offline, by the pixel rule; a PNG label map per pool image |
| exported file | backbone only (`features.*`) | identical contract, so every downstream script is unchanged |

So the downstream code, the fine-tune configs, the ATRNet fine-tune (`SARDet_to_ATR`) and the SARDet detector
suite all consume the result without modification.

## 3. Why it could beat DINO (and how we would know it is not luck)

Mechanisms, each a testable claim:
- more signal per step: 100 % of views hold an object instead of 29 %;
- the per-region loss forces the backbone to represent each object on its own instead of the crop average;
- object-at-view-scale views look like ATRNet-STAR chips, so the SSL distribution is closer to the test one.

Controls (the P4 lesson: a gain without its control is not a finding):
- `DINO`: the existing SARDet DINO runs (same pool, same epochs, same views) are the baseline row.
- `RegionDINO-crops`: change 1 only. Isolates "where the crops land".
- `RegionDINO`: changes 1 to 3.
- `RegionDINO-random`: changes 1 to 3 but with random regions of matched size and count. If this matches
  RegionDINO, the pixel rule does not matter and the gain is just the extra loss terms.

Where it is measured, both already built:
- ATRNet-STAR 1 / 3 / 10 % via `SARDet_to_ATR` (the XA rows); the main target.
- SARDet-100K detection 1 / 3 / 10 % via `SARDet/compare` (the current null result is the bar to clear).
Three seeds each, as everywhere else. Later the same method on the mixed pool (arm C).

Honest expectations. Region-level SSL methods in the literature (DetCon, SlotCon, SoCo) gain one to two AP
points over their crop-level parents on ResNet-50 detection. Nothing is known for MobileNetV2 on SAR, and our
SARDet detection fine-tune has so far ignored every SSL backbone equally. The ATRNet path is where the view
change should show first, because that is where the object-scale mismatch was largest.

## 4. Where the region rule is weak, and what that means here

From `box_zoom.py` (24 aircraft boxes): bright ships and vehicles get one clean region each; aircraft parked
close together get painted but merged with their neighbours (fine: the view is still object-centred);
faint large aircraft on dark ground (SAR-AIRcraft) get a speck or nothing. Consequence: those aircraft are
simply treated as background by the method, which is no worse than DINO treats them. A local-contrast rule
can be added to the proposer later without touching the learning code, because masks are precomputed.

## 5. Build plan (small steps, each checkable)

1. `RegionSSL/make_regions.py`: run the rule over the 94,034-image SSL pool, write one label-map PNG per
   image plus a `regions.csv` (image, region id, bbox, area) and a summary (regions per image, foreground
   share, size histogram). CPU only; runs on the PC overnight or as a CRC CPU job.
2. `RegionSSL/ssl_data_regions.py`: dataset that returns the views *and* each view's region map, crop
   geometry recorded so region ids line up across views; object-centred window sampler; `random` mode that
   fabricates matched random regions for the control.
3. `RegionSSL/train_ssl.py`: a `Method` for the existing `SARDet/shared/ssl_engine.py` (same checkpoints,
   resume, export, sbatch). Config keys: `crop_mode` (random | object), `region_loss` (0 | weight),
   `background_weight`, `region_source` (felz | random). Everything else inherits `ssl_common.json`.
4. PC smoke (2 epochs, 64 images): losses finite, region vectors line up across views (check by feeding the
   same crop twice), export loads in both downstream paths.
5. CRC: 3 seeds x {crops-only, full, random-control} = 9 SSL runs of 100 epochs, then the existing
   fine-tune scripts. Compare with `SARDet_to_ATR/compare.py` and `SARDet/compare/compare.py`.

## 6. Revisions after review with Nischal (2026-11-17)

- **No dataset-specific scale.** Object-centred crops size the window relative to the region's own extent
  (the object fills a random 20 to 80 % of the view), never a fixed pixel size, and DINO's ordinary random
  crops are kept for a share of the views. Nothing in the method refers to 128 px chips, 224 px views or
  any property of ATRNet-STAR or SARDet-100K.
- **DINO's asymmetry is kept, not replaced.** The crop-level loss (teacher on 2 global views, student on
  global + local views, local-to-global matching) stays exactly as in DINO with its original weight. The
  region loss is an extra term with its own weight (0 recovers DINO plus crop placement), and it follows the
  same asymmetry: teacher region vectors come only from global views, student region vectors from global
  and local views, so a region seen in a small local crop must match the teacher's wider view of it.
- **Two ways to get "one vector per thing", to be chosen before coding.**
  (a) *Pooling*: one forward pass per view; the feature map (7x7 at stride 32 for a 224 view, 14x14 at
  stride 16) is averaged over the cells belonging to region k. Cheap; the region vector still carries some
  context from neighbouring cells.
  (b) *Separate crops*: each region is cut out as its own small image and passed through the backbone on
  its own. Cleaner "object alone" vectors; several more forward passes per image; the object loses its
  surroundings entirely.
  Hybrid: (a) for the loss, and (b) only as a view-sampling choice (local crops cut around regions), which
  is what change 1 already does.
- **Rows required for the ATRNet-STAR table** (SSL on SARDet-100K, fine-tune on ATRNet 1 / 3 / 10 %):
  A0 baseline (no SSL; exists, 3 splits), XA_dino, XA_lejepa, XA_spark (9 CRC jobs submitted), and the new
  XR_regiondino. First comparison: A0 vs XA_dino vs XR_regiondino; LeJEPA and SparK rows are reported
  when the full table is assembled.

## 7. Decision and implementation (2026-11-17, evening)

Decided: option (b), separate crops. Each region becomes its own DINO sample (2 object-global views for the
teacher, 4 object-local views for the student, same head, own centering buffer). Reason: the final feature grid
of a 224 px view is 7 x 7, so a small target would get a pooled vector made mostly of its surroundings; a crop
sized from the region's own extent does not have that problem. Two methods, each with a random-window control:

| folder | views per image | loss |
|---|---|---|
| `RegionDINO_scene_object` | DINO's 2 global + 8 local scene crops AND 2 regions x (2 + 4) object views | scene DINO + 1.0 x object DINO |
| `RegionDINO_object` | 3 regions x (2 + 4) object views only | object DINO only |

Implemented and smoke-tested on the PC (2 epochs, 48 images, exact resume, export, detection fine-tune,
probes, ATRNet fine-tune wrapper): `rssl/` (regions, region_data, regiondino, region_engine), `make_regions.py`,
both method folders, `crc/make_regions.sbatch`. The suite's `crc/ssl.sbatch` and `crc/finetune.sbatch` run
the new folders unchanged (first argument `../RegionSSL/<folder>`); `SARDet_to_ATR` got four additive method
names. Nothing under `SARDet/` changed except two additive lines in `compare/compare.py` (collect the new
folders). Status of every row: see README.md.

## 8. Arm I: the same methods on the ATRNet pool (added 2026-11-18)

Why: on SARDet scenes the object views turned a cross-dataset loss into a gain (object-only +9.4 / +7.1 / +1.4 over
plain DINO on ATRNet 1 / 3 / 10 %), but the in-domain A1 row (DINO on ATRNet chips, 35.1 / 55.9 / 77.6) is still far
ahead. Nischal's question: does the region focus also beat plain DINO when both train on ATRNet's own pool?

Rows (3 splits, 100 epochs, the split's `ssl_training_data.csv`, i.e. exactly A1's pool and seeds):
`XI DINO (suite recipe)` = this engine with scene views only (`DINO_reference/`), the paired reference;
`XI RegionDINO object-only`, `XI RegionDINO scene+object`, and their random-window controls. A1 stays in the table:
XI DINO minus A1 measures the crop-recipe difference (window-and-zoom at 224 vs A1's area-scale crops), the RegionDINO
rows minus XI DINO measure the object views with everything else equal.

Pool handling (`rssl/pools.py`): chips are 128 px; the views are cut from the chip resized to 224 (the fine-tune input
size and A1's global crop size), so a global view covers 56 to 100 % of the chip area (A1: 40 to 100 %) and a local view
10 to 33 % (A1: 15 to 40 %). Regions are proposed at the native 128 (fewer, larger regions than at 224: median 4 per
chip, extent 22 px; `chip_lab.py`) and scaled by 1.75 when read. The rule paints part of the vehicle on 99 % of chips
but ATRNet's boxes are generous (whole footprint), so the strict hit rate is 36 %: object views (window = 1.4 to 2.2 x
the region) still include most of the vehicle, and clutter regions (roads, vegetation edges) are sampled too, as on
SARDet. The control is the guard against "any small crop helps".
