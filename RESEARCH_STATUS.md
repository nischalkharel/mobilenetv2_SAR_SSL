# Research status, 2026-11-25: what we have, what it shows, what is missing

One document, updated in place when something changes. The numbers come from the tracker (`tracker/`, conclusions
c01 to c12 in `tracker/web/conclusions.js`); every experiment is named SSL-dataset / FINETUNE-dataset /
TEST-dataset / technique. Accuracies are mean per-class accuracy on the test set unless said otherwise, averaged
over three splits (each split = its own labeled draw and its own SSL seed), at 1 / 3 / 10 % of the labels.

## 1. The question and the setting

Can a compact network (MobileNetV2, 2.2 M backbone parameters, ImageNet start) learn useful SAR features from
unlabeled SAR imagery, and what kind of self-supervised training makes that work when the unlabeled images are big
scenes and the task is small object chips? Everything below is MobileNetV2 fine-tuned end to end on a small labeled
set after SSL; the comparison is always against the same network fine-tuned straight from ImageNet weights.

Datasets and protocols (all built, leakage-checked, manifests committed):

| dataset | used as | what we cut | notes |
|---|---|---|---|
| ATRNet-STAR SOC-40 | labeled chips (40 vehicle classes), plus an unlabeled chip pool | 128-px chips, official test set | validation = 800 chips; near-duplicates of validation removed from the SSL pool |
| SARDet-100K | unlabeled scene pool (94,034 scenes), detection labels, and chips cut from its boxes (6 classes) | full scenes at 512 px; 128-px chips | the chip test turned out too easy (c09) |
| FAIR-CSAR-V1.0 SL | unlabeled 1 m scene pool (19,036 unique tiles) and chips (22 fine-grained classes) | 512-px native tiles; one chip per physical object | slices overlap 50 %, de-duplicated; validation slices out of the pool; test never in the pool |

Methods implemented and run (all share one SSL recipe: 100 epochs on the SARDet / ATRNet pools, batch 64, same
views, same optimizer; the FAIR runs at 100 and at 500 epochs):

- Plain DINO, LeJEPA, SparK (three different SSL families, one recipe).
- CondDINO (DINO centred per acquisition-metadata group).
- RegionDINO: DINO's views cut around label-free bright regions proposed by a pixel rule (Felzenszwalb on
  brightness). Two modes (scene+object, object-only), each with a random-window control that keeps the window
  sizes and randomizes the places.
- ViewScale: the object-only recipe with fixed-size random windows (32 / 64 / 128 / 256 / 512 px).
- TeacherRegions: the object-only recipe with regions proposed by a frozen SAR foundation model (SARMAE ViT-B
  attention) instead of the pixel rule, with a size-matched random control.

Evaluations: ATRNet chip classification (SSL on ATRNet chips = in-domain; SSL on SARDet scenes = cross-dataset);
SARDet detection (Faster R-CNN, mAP@0.5:0.95); SARDet chip classification; FAIR-CSAR chip classification (SSL on
FAIR tiles = same dataset; SSL on SARDet scenes = cross-dataset). Plus frozen-backbone response maps on chips and
scenes (where the network looks).

Scale of what exists: 87 completed SSL runs, 531 fine-tune results (222 ATRNet classification, 111 detection, 90
SARDet chips, 108 FAIR chips), about 640 A100 GPU-hours on CRC. Tooling: one SSL engine shared by every method,
CRC job scripts with self-resuming chains, a local results tracker with per-experiment pop-ups, conclusions and a
glossary, run records with code commit, data hashes and cost per run.

## 2. What each experiment showed

| experiment (SSL / FINETUNE / TEST / technique) | 1 % | 3 % | 10 % | reading | conclusion |
|---|---|---|---|---|---|
| none / ATRNet / ATRNet / ImageNet start | 20.7 | 43.4 | 71.7 | the baseline | c01 |
| ATRNet chips / ATRNet / ATRNet / DINO | 35.1 | 55.9 | 77.6 | in-domain SSL helps a lot at 1 % | c01 |
| ATRNet chips / ATRNet / ATRNet / LeJEPA | 36.2 | 56.6 | 75.5 | ties DINO | c01 |
| ATRNet chips / ATRNet / ATRNet / SparK | 24.7 | 50.3 | 74.6 | weakest in-domain | c01 |
| ATRNet chips / ATRNet / ATRNet / CondDINO | 40.8 | 60.1 | 78.6 | +5.7 over DINO, but uses metadata; controls = DINO | c02 (parked) |
| SARDet scenes / ATRNet / ATRNet / DINO | 17.8 | 41.9 | 72.3 | scene SSL transfers WORSE than ImageNet | c03 |
| SARDet scenes / ATRNet / ATRNet / LeJEPA | 12.2 | 32.8 | 64.6 | much worse | c03 |
| SARDet scenes / ATRNet / ATRNet / SparK | 22.7 | 48.0 | 73.9 | the only plain method that carries over | c03 |
| SARDet scenes / ATRNet / ATRNet / RegionDINO object-only | 27.2 | 49.0 | 73.7 | +9.4 / +7.1 / +1.4 over DINO; best cross-dataset result | c04 |
| SARDet scenes / ATRNet / ATRNet / random windows (control) | 20.9 | 45.4 | 72.7 | a third of the gain: placement matters here | c04 |
| SARDet scenes / ATRNet / ATRNet / RegionDINO scene+object | 20.2 | 45.7 | 73.4 | keeping the scene views loses most of the gain | c05 |
| ATRNet chips / ATRNet / ATRNet / RegionDINO object-only | 32.0 | 54.5 | 75.7 | on a chip pool the object views hurt (-2) | c06 |
| SARDet scenes / ATRNet / ATRNet / ViewScale 32 to 512 px | 18.3 to 22.6 | 42.6 to 46.8 | 72.3 to 73.6 | window size matters up to 256 px; the region-placed row stays 5 points above | c10 |
| SARDet scenes / ATRNet / ATRNet / teacher (SARMAE) regions | 25.3 | 49.0 | 73.5 | = brightness rule; its size-matched control 18.4 | c11 |
| SARDet scenes / SARDet / SARDet / detection, no SSL (mAP) | 15.7 | 21.3 | 28.5 | the baseline | c08 |
| SARDet scenes / SARDet / SARDet / detection, RegionDINO object-only | 17.3 | 22.7 | 29.8 | +1.6 / +1.4 / +1.2 mAP, first method above a point | c08 |
| SARDet scenes / SARDet / SARDet / detection, random windows (control) | 17.5 | 22.9 | 29.6 | the same: for detection the crop size is the win, not placement | c08 |
| SARDet scenes / SARDet / SARDet / detection, DINO, LeJEPA, SparK, CondDINO | 15.9 to 16.8 | 21.5 to 21.9 | 27.6 to 29.2 | within a point of no SSL | c08, c02 |
| SARDet scenes / SARDet chips / SARDet chips / all methods | 80.4 to 83.6 | 90.2 to 92.1 | 96.5 to 97.3 | saturated, null | c09 |
| none / FAIR / FAIR / ImageNet start | 24.9 | 29.2 | 39.6 | a hard test (22 classes) | c12 |
| FAIR tiles / FAIR / FAIR / DINO, 500 epochs | 27.3 | 32.9 | 43.6 | same-dataset scene SSL helps: +2.4 / +3.8 / +4.0 | c12 |
| FAIR tiles / FAIR / FAIR / RegionDINO object-only, 500 epochs | 28.6 | 34.0 | 43.7 | within a point of DINO | c12 |
| FAIR tiles / FAIR / FAIR / random windows (control), 500 epochs | 28.8 | 32.9 | 41.5 | object-only beats it only at 10 % (+2.2) | c12 |
| FAIR tiles / FAIR / FAIR / teacher (SARMAE) regions, 500 epochs | 28.5 | 33.8 | 43.7 | = brightness rule again; its control 25.0 / 31.4 / 40.8 | c12 |
| SARDet scenes / FAIR / FAIR / DINO | 22.7 | 27.7 | 36.8 | cross-dataset scene SSL hurts again (-2 to -3) | c12 |
| SARDet scenes / FAIR / FAIR / RegionDINO object-only | 26.6 | 31.9 | 40.9 | repairs it: +3.9 / +4.2 / +4.1 over DINO, every split | c12 |
| SARDet scenes / FAIR / FAIR / random windows (control) | 25.8 | 29.2 | 40.7 | here the control gets most of it (placement +0.8 / +2.7 / +0.2 per-class, +3 to +5 overall) | c12 |

Response maps (c07): on full scenes the ImageNet weights are the most object-focused backbone (peak in a box 36 %),
plain DINO spreads over the scene (8 to 11 %), object-only brings it back (29 to 33 %); on chips the object-only
backbone peaks in the vehicle box 56 to 62 % of the time against 17 to 19 % for plain DINO.

## 3. The story as it stands, claim by claim

1. **Strong.** SSL on in-domain chips gives a compact model a large gain when labels are scarce (21 -> 35 % at 1 %
   labels on ATRNet). One dataset, three methods, three splits.
2. **Strong, two dataset pairs.** Plain SSL on big scene archives transfers to object chips worse than no SSL
   (ATRNet: -3 for DINO, -8 for LeJEPA; FAIR: -2 to -3 for DINO). The response maps explain why: scene SSL learns
   the background.
3. **Strong on ATRNet, medium on FAIR.** Cutting DINO's views around label-free bright regions (RegionDINO
   object-only) turns that loss into the best cross-dataset result (+9.4 at 1 % on ATRNet, +4 at every budget on
   FAIR, positive on every split of both). What the controls say about WHY differs: on ATRNet the random-window
   control gets a third of the gain, so placement carries two thirds; on FAIR the control gets most of it in mean
   per-class accuracy (placement +1 to +3) and object-only beats the control in overall accuracy on all nine pairs.
   Honest statement: the small object-sized views are the robust part; placement adds on top, clearly on ATRNet,
   modestly on FAIR.
4. **Medium.** Same-dataset scene SSL helps a hard chip test once trained long enough (+2.4 / +3.8 / +4.0 on FAIR
   at 500 epochs; the 100-epoch runs showed only the 10 % gain). There, placement is worth about 2 points at 10 %
   and nothing below.
5. **Medium.** Crop size and placement are two separate effects: small views help detection and transfer alike
   (control = object-only on detection; ViewScale curve rises to 256 px); placement matters only when the test is
   object chips cut from a different dataset.
6. **Medium, four tests.** A 1.3 GB SAR foundation model as the region proposer does not beat the brightness rule
   (ATRNet transfer, SARDet detection, SARDet chips, FAIR chips), even where it finds more objects (FAIR: 41 vs 34 %),
   while both proposers beat their own random controls. The method stays cheap and teacher-free.
7. **Bounds.** Object views do not help when the SSL images are already chips (c06). SARDet's own chips are too easy
   to separate methods (c09). CondDINO is a real but metadata-dependent effect (c02), a side section at most.

## 4. Is this enough for the master's thesis?

Yes, on content. A thesis needs a clear question, a controlled protocol, a method, ablations, more than one
dataset, negative results explained, and reproducibility. All of that exists:

| thesis chapter | what fills it | state |
|---|---|---|
| Background and related work | SSL families (DINO / LeJEPA / SparK), SAR foundation models (SARATR-X, SARMAE), object-centric SSL in optical vision | literature review NOT written; only one paper in `Papers/`. This is the biggest writing gap |
| Data and protocol | three datasets, leakage controls, splits, label budgets, metrics | done; `preprocessing_dataset/SUBSETS.md`, `SARDet/DESIGN.md`, `FAIR_chips/DESIGN.md`, `datasets/` |
| Baseline study | A0 vs DINO / LeJEPA / SparK in-domain, CondDINO | done (c01, c02) |
| The transfer problem | scene SSL vs chips, response maps | done (c03, c07) |
| RegionDINO | method, both modes, controls, ViewScale, TeacherRegions | done (c04, c05, c08, c10, c11) |
| Second dataset | FAIR-CSAR same-dataset and cross-dataset | done (c12) |
| Detection | SARDet detection suite | done for the main rows; ViewScale and teacher rows split 1 only |
| Cost | SSL cost per method is recorded (GPU-hours, peak memory, steps); detector MACs / CPU latency script exists | no per-method CPU inference table, no comparison against a larger reference model |
| Discussion and limits | c06, c09, the FAIR control nuance, equal-epoch caveats | material exists in DESIGN files and conclusions; not written as prose |

What the thesis does NOT have yet, in order of how much a committee would notice:

1. A written literature review and related-work positioning (object-centric SSL exists in optical vision, e.g.
   region- or object-level contrastive methods; the thesis must say what is new here: SAR, label-free brightness
   proposals, compact CNN, cross-dataset scene-to-chip transfer, the controls).
2. The second compact architecture (B0 / B1 in the original plan; sponsors asked for it on 2026-10-19). Nothing run.
3. A larger reference model (question 4 in CLAUDE.md: how close does the compact model get, and what does it save).
   Only MobileNetV2 has been trained; SARMAE was used as a proposer, never as a competitor.
4. Segmentation (in the original three-task plan). Not started; can be declared out of scope with the detection
   suite as the second task.
5. Operating-condition (EOC) generalization that Brittany asked for (depression angle, polarization, scene). The
   overlap check was done (TODO.md section 7), the study was not.

## 5. Is this enough for a conference paper?

The core result is paper-shaped: a label-free, teacher-free change to DINO's views that makes scene SSL transfer
to object chips on two dataset pairs, with controls that separate crop size from placement, a scale sweep, a
foundation-model proposer that does not help, and response maps as mechanism. The SHREC conference (January 2027,
Albuquerque) is the stated venue; a stronger venue would ask for the items below.

What a reviewer will ask for that we cannot answer today, ranked by how cheap the fix is:

| gap | why it matters | cost | status |
|---|---|---|---|
| Transfer of the FAIR-trained backbones to ATRNet chips (SSL-FAIR / FINETUNE-ATRNet / TEST-ATRNet) and to SARDet chips | a second SCENE ARCHIVE for claim 3 (today both cross-dataset results start from SARDet scenes); the 15 FAIR backbones already exist | fine-tunes only, about 5 GPU-h | not run; cheapest high-value item |
| Second compact architecture on the key setting (no SSL, DINO, RegionDINO object-only; SSL on SARDet, test on ATRNet) | "is it MobileNetV2-specific?"; sponsor request | about 30 GPU-h SSL + fine-tunes | not started; candidates MobileNetV3, EfficientNet-B0 |
| A larger reference (ResNet-50 with the same recipe, or SARMAE fine-tuned on the same labeled sets) | "how close does the compact model get" | 1 to 2 SSL runs x 3 splits, or fine-tunes only for SARMAE | not started |
| CPU cost table (parameters, MACs, latency per chip) for MobileNetV2 vs the reference | the deployment argument of the whole project | hours on the PC | script exists for the detector only |
| Placement evidence on FAIR is weaker than on ATRNet | claim 3's "why" | either accept the honest statement, or add the reverse direction (SSL on ATRNet chips, test on SARDet / FAIR chips) | not run |
| Detection on split 2 and 3 for ViewScale and teacher rows | three splits everywhere | about 20 GPU-h | split 1 only |
| Related-work positioning against object-centric SSL in optical imagery and SAR SSL papers | novelty claim | reading and writing | not done |
| One final re-run of the paper's rows under one frozen protocol, with the fine-tune seed also varied | reviewers ask for seeds; fine-tune seed is fixed today | depends on the final table | planned (tracker to-do) |
| Resolution question (downsampled FAIR pool, 2 m per pixel) | sponsor concern from 2026-11-13 | 3 x 8 GPU-h for plain DINO | configs exist, not launched |

## 6. Recommended order from here

1. Fine-tune the existing FAIR-trained backbones on ATRNet chips and on SARDet chips (cheap, closes the "second
   scene archive" hole for claim 3 and tests whether the FAIR placement result repeats cross-dataset).
2. Write the related-work section now, in parallel with any runs; it decides how the method is named and framed.
3. Second compact architecture on the key setting (three methods, three splits).
4. CPU cost table and one larger reference model.
5. Then freeze the protocol and re-run the paper's final table with fine-tune seeds varied.
6. Optional, if time: downsampled pool, detection splits 2 and 3, EOC study, detection on FAIR.

## 7. Where things are

- Tracker: `python tracker/serve.py` (port 8765); pages: overview, classification, detection, chips, FAIR-CSAR,
  maps, conclusions, glossary, ideas.
- Study folders, each with `WHAT_IS_THIS.txt` and `DESIGN.md`: `Experiments/` (A0 to A3, CondDINO), `SARDet/`
  (detection suite), `SARDet_to_ATR/`, `RegionSSL/`, `ViewScale/`, `TeacherRegions/`, `SARDet_chips/`, `FAIR_chips/`.
- Datasets: `datasets/CANDIDATES.md`, `datasets/outputs/fair_csar_inventory.md`.
- Meeting notes: `../Biweekly_nischal_ngc/`, `../P3_weekly/`. Open items from meetings: `TODO.md` sections 5 to 7.
- Running log of every session: `claude_log.md`.
