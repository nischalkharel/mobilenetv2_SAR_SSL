# TeacherRegions: regions proposed by a large pretrained SAR model, student stays compact

Status 2026-11-20: protocol written, teacher weights downloaded and verified (models/SARMAE/), proposer script
being built; no SSL run yet.

## 1. The idea and the question

RegionDINO object-only (RegionSSL/) is the best cross-dataset row because its DINO views are cut around label-free
regions. Its proposer is a brightness rule (bright compact superpixel groups), which is cheap and SAR-realistic
(targets are bright scatterers) but blind to faint targets and fooled by bright clutter. Here the proposer is the
**attention / saliency of a frozen large SAR foundation model**. The student is still MobileNetV2, the recipe is
still object-only RegionDINO, and the only thing that changes is the region file. The teacher never trains, is
never fine-tuned on our data, and is absent at test time: it is a filter on where the student looks while it learns.

Questions, in order:
1. Are teacher-proposed regions better proposals than the brightness rule? (measured against annotation boxes,
   before any SSL: a cheap, decisive check)
2. Does the student get better with them? (the same ATRNet transfer and SARDet detection protocols as c04 / c08)
3. Is that more than plain transfer of the teacher's knowledge? (feature distillation from the same teacher into
   the same student, same budget: if distillation gives the same gain, the regions were just a way to transfer;
   if regions beat distillation, "where to look" carries information that "what the teacher thinks" does not)

## 2. Which teacher (survey 2026-11-20, web-verified)

| model | why / why not |
|---|---|
| **SARMAE ViT-B** (CVPR 2026, arXiv 2512.16635) | chosen. Public ungated weights (HF), ViT-B/16 MAE trained on SAR-1M (1.31 M chips, 18 datasets, Sentinel-1 / Gaofen-3 / TerraSAR-X / RadarSat-2, 0.1 to 60 m), speckle-aware masking; the paper shows CLS attention sitting on objects; best public SARDet-100K detection number among downloadable models (57.9 B / 63.1 L). CC BY-NC 4.0 |
| SARATR-X v1 (TIP 2025) | fallback. HiViT-B, 186K ATR chips, click-through gate; hierarchical (no CLS token), weaker frozen features |
| SARATR-X v2 (2026) | strongest (68.7 mAP on SARDet) but weights only by email to the author; request in parallel |
| AFRL-DINOv2 (the "status of SAR foundation models" paper) | best frozen features reported, but not released |
| FG-MAE, DOFA, TerraMind | Sentinel-1 10 m dual-pol, two-channel inputs; wrong resolution and input for target chips |
| SARDet-100K MSFA | supervised on SARDet labels: would leak labels into a label-free method |

Caveat to state in the paper: SAR-1M contains SARDet-100K (and HRSID, SSDD, MSTAR) as unlabeled pretraining data.
Region proposals on SARDet scenes therefore come from a model that has seen those scenes without labels. ATRNet-STAR
is not in SAR-1M, so the cross-dataset transfer test (SSL on SARDet scenes, test on ATRNet chips) is the clean one;
the SARDet detection test is in-distribution for the teacher and is reported with that note.

## 3. From saliency to regions (teacher_saliency.py)

Input: each pool image at the working frame the student sees (SARDet: shorter side 512). The teacher takes 224 x 224,
so the scene is covered by a 3 x 3 grid of 224-px windows (stride 144) and the per-window maps are averaged where
they overlap; the result is a saliency map at stride 16 (32 x 32 for a 512 scene), then upsampled to the frame.
Preprocessing as in SARMAE's own pretraining data path: grayscale replicated to 3 channels, [0, 1], normalised
with mean 0.5 / std 0.5.

Two saliency signals are produced and compared (question 1 decides which one is used):
- **CLS attention**: the last block's attention from the CLS token to every patch, averaged over heads (the DINO
  "where does the model look" map).
- **token norm**: the L2 norm of the final patch tokens (what the response maps of the student use; no CLS needed,
  so it also works for the SARATR-X fallback).

Regions: threshold the map at its top q % of pixels (q = 5, matching the area share of the brightness rule's
regions), connected components (scipy.ndimage.label), one box per component, drop components under 12 px of area
(the brightness rule's floor), keep at most 20 per image by saliency mass, largest first. Written with
`RegionSSL.rssl.regions.save` so the file has exactly the format RegionDINO reads (names, offsets, boxes, areas,
summary json with sha256). No labels are read.

**Setting chosen on 300 pool scenes (2026-11-20, annotations read only to measure; 12-px floor as the brightness rule):**

| proposer | annotated boxes that get a region | region centres that lie on an object | regions per image |
|---|---|---|---|
| brightness rule (RegionSSL) | 61.7 % | 7.0 % | 27.5 |
| teacher CLS attention, zoom 1, top 5 % | 54.2 % | 9.0 % | 17.7 |
| teacher CLS attention, zoom 2, top 5 % | 62.5 % | 10.9 % | 19.8 |
| **teacher CLS attention, zoom 2, top 2 % (chosen)** | **61.7 %** | **13.2 %** | **17.5** |
| teacher token norm (any setting) | 9 to 13 % | 2 to 3 % | 12 to 18 |

Zoom 2 means the teacher looks at 112-px native windows upsampled to 224 (a 16-px patch covers 8 native px, so a
10-px ship is about a patch), 25 windows per scene. At equal recall the teacher's regions are nearly twice as often
on an object as the brightness rule's. Token-norm saliency is useless for this teacher and is dropped.

**Question 1 check**, before any SSL: on 300 annotated pool images (the same diagnostic RegionDINO already runs),
share of GT boxes that contain a region centre, share of region centres inside a GT box, and regions per image, for
brightness rule vs teacher-CLS vs teacher-norm. Plus preview grids with the three region sets drawn on the same
scenes. The teacher signal with the better box agreement goes forward; the numbers go into the tracker.

## 4. The three-way comparison (questions 2 and 3)

All with MobileNetV2, the SARDet scene pool, 100 epochs, three splits, the usual fine-tunes:

| row | what | status |
|---|---|---|
| RegionDINO object-only, brightness regions | exists (c04, c08) | done |
| RegionDINO object-only, random windows | exists | done |
| **RegionDINO object-only, teacher regions** | this study; `config_ssl.json`, region file from teacher_saliency.py | to run |
| **feature distillation from the teacher** | student pooled feature -> linear projector -> cosine to the teacher's CLS embedding of the same 224 view; same steps and batch; no DINO loss | phase 2, code to write (`distill/`) |
| teacher itself, fine-tuned | upper reference, not compact (86 M vs 2.2 M parameters); only at 1 % labels on ATRNet | optional |

Tests: ATRNet transfer (SARDet_to_ATR, 25 min per job) first; SARDet detection (12 h per job) for split 1, then the
rest if the transfer shows something. Frozen response maps on chips and scenes for every new backbone.

## 5. Cost

Proposer: SARMAE ViT-B forward on 25 windows x 94,034 scenes (zoom 2), about 3 to 4 h on one A100 (GPU job, once; 6.5 scenes/s on the PC GPU). SSL: 3 runs
(5 GPU-h each). Distillation: 3 runs (similar). Fine-tunes as in the other studies.

## 6. Decisions and what would make it defensible

- The teacher is used only to decide where the views are. The student never receives teacher features in the
  RegionDINO row; the distillation row is where it does, and that row is the control for "just transfer".
- Same proposer floor (12 px), same number of regions per image sampled (3), same view sizes and steps as the
  brightness-rule run, so the only difference between the rows is the region file.
- The region-quality check (section 3) is reported whether it favours the teacher or not.
- The SAR-1M overlap with SARDet is stated; ATRNet transfer is the headline test for this study.
