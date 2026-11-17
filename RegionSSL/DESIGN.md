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
